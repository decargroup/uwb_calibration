from typing import Tuple, Dict
import numpy as np
from .postprocess import PostProcess
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares
import pickle
import pandas as pd
from .utils import compute_range_meas, get_bias

class ApplyCalibration():

    def __init__():
        raise NotImplementedError("This class is not mean to be initialized.")

    @staticmethod
    def antenna_delays(
        df: pd.DataFrame, 
        delays: Dict[int, float],
        tx_rx_split: Dict[str, float] = {'tx':0.6, 'rx':0.4},
    ) -> pd.DataFrame:
        # Find the delays associated with the ranging tags for every measurement
        from_delay = np.array([delays[x] for x in np.array(df["from_id"])])
        to_delay = np.array([delays[x] for x in np.array(df["to_id"])])
        
        # Correct the intervals
        df["del_t1"] += from_delay
        df["del_t2"] -= to_delay
        
        # Compute the individual delays
        tx_from_delay = tx_rx_split['tx'] * from_delay
        rx_from_delay = tx_rx_split['rx'] * from_delay
        tx_to_delay = tx_rx_split['tx'] * to_delay
        rx_to_delay = tx_rx_split['rx'] * to_delay

        # Correct the individual timestamps
        df["tx1"] += -tx_from_delay
        df["rx2"] += rx_from_delay

        df["rx1"] += rx_to_delay
        df["tx2"] += -tx_to_delay

        if 'tx3' in df.columns:
            df["tx3"] += -tx_to_delay
            df["rx3"] += rx_to_delay

        # Correct the range measurements and bias
        df['range'] = compute_range_meas(df)
        df['bias'] = df.apply(
            get_bias, 
            axis=1
        )

        return df

    def power(
        df, 
        bias,
        bias_std, 
    ):
        pass


    
class UwbCalibrate(PostProcess):
    """A class to handle calibration of the UWB modules. 

    This class inherits attributes from a PostProcess object. The attributes to be inherited
    are specified in the attribute _inherited.    

    Attributes
    ----------
    lift: function
        The lifting function for power measurements.
    delays: dict
        Only exists after calib_antennas() is called.
        keys: int
            The ID of the tag.
        values: float
            The estimated antenna-delay for this tag.
    bias_spl: UnivariateSpline
        The learnt "bias vs. lifted power" spline.
        Only exists after fit_power_model() is called.
    std_spl: _type_
        The learnt "standard deviation vs. lifted power" spline.
        Only exists after fit_power_model() is called.
    # TODO: find a better way to deal with the "only exists" fields above

    Examples
    --------
    # Let the variable 'data' be a PostProces object.
    calib = UwbCalibrate(data, rm_static=True)

    # Calibrate antenna delays
    calib.calibrate_antennas()

    # Correct power-correlated bias
    calib.fit_power_model()
    """
    # Speed of light
    _c = 299702547 

    # Attributes to be inherited from the PostProcess object
    _inherited = [
        'df',
        'machine_ids',
        'tag_ids',
        'ds_twr',
        'fpp_exists',
    ]
    
    def __init__(
        self, 
        data, 
        rm_static = False, 
        f_lift = lambda x: 10**((x + 82)/10),
    ) -> None:
        """Constructor

        Parameters
        ----------
        data: PostProcess
            Processed data to be calibrated.
        rm_static: bool, optional
            Remove static regions at the beginning and end of experiments, 
            by default False
        f_lift: function, optional
            The lifting function for power measurements for better fitting, 
            by default lambda(x): 10 ** ( (x + 82) / 10 )
        """
        self._data = data
        
        if rm_static:
            self._remove_static_regions()

        self.lift = f_lift

    def __getattr__(self, attr) -> object:
        """Inherit missing self._inherited attributes from PostProcess object.

        Parameters
        ----------
        attr: object
            Name of the attribute.

        Returns
        -------
        object
            The retrieved attribute value from the PostProcess object.

        Raises
        ------
        Exception
            Could not find the attribute: *attr
        """
        if attr in self._inherited:
            return getattr(self._data, attr)
        else: 
            Exception(r"Could not find the attribute: " + attr)

    def _remove_static_regions(self) -> None:
        """Remove the static region in the extremes.
        """
        # Threshold for motion detection, in metres
        thresh = 0.1 # TODO: make this user-defined

        # Window size to compare measurements 
        window = 10 
        
        lower_idx = 0
        upper_idx = np.Inf

        # Iterate machine by machine
        for machine_i in self.machine_ids:
            # Get the position of the machine
            r_iw_a = self.get_machine_pos(machine_i, as_numpy = True)
            
            # Split the position into its dimensions
            x = r_iw_a[:,0]
            y = r_iw_a[:,1]
            z = r_iw_a[:,2]
            
            # Get the lower and upper indices associated with static regions
            # in the extremes
            x_lower, x_upper = self._find_static_extremes(x, thresh, window)
            y_lower, y_upper = self._find_static_extremes(y, thresh, window)
            z_lower, z_upper = self._find_static_extremes(z, thresh, window)
            
            # Find first index associated with the robot not moving in any direction
            lowest_low_idx = np.min([x_lower, y_lower, z_lower])
            highest_upper_idx = np.max([x_upper, y_upper, z_upper])
            
            # Check if this robot was static for longer than others
            if lowest_low_idx > lower_idx:
                lower_idx = lowest_low_idx
            if highest_upper_idx < upper_idx:
                upper_idx = highest_upper_idx
        
        # Drop measurements when one or more robots are static
        n = len(self.df)
        self.df.drop(np.linspace(0,lower_idx,lower_idx+1), inplace=True)
        self.df.drop(np.linspace(upper_idx, n-1, n-upper_idx), inplace=True)
        
    @staticmethod
    def _find_static_extremes(
        r,
        thresh,
        window
    ) -> Tuple[int, int]:
        """_summary_

        Parameters
        ----------
        r: np.ndarray
            The position measurements in one dimension.
        thresh: float
            Threshold for motion detection, in metres
        window: int
            # Window size to compare measurements 

        Returns
        -------
        Tuple[int, int]
            lower_idx: The lower index associated with the time the robot started 
                moving in this direction.
            upper_idx: The upper index associated with the time the robot stopped 
                moving in this direction.
        """
        # Get the rolling mean based on the window size
        rolling_mean = pd.DataFrame(r).rolling(window, center=True).mean()
        
        # Fill extremes
        rolling_mean = rolling_mean.fillna(method="bfill").fillna(method="ffill")
        rolling_mean = np.array(rolling_mean)
        
        # Find the deltas in the motion.
        diff_lower = np.abs(rolling_mean[1:] - rolling_mean[0])
        diff_upper = np.abs(np.flip(rolling_mean)[1:] - rolling_mean[-1])
        
        # Get the indices associated with changes greater than the threshold
        lower_idx = np.argmax(diff_lower>thresh)
        upper_idx = len(diff_lower) - np.argmax(np.flip(diff_upper)>thresh) - 1
        
        return lower_idx, upper_idx

    def calibrate_antennas(
        self, 
        loss='cauchy', 
        tx_rx_split={'tx':0.6, 'rx':0.4},
        inplace = False,
    ) -> None:
        """Estimate the antenna delays for the UWB tags, and correct the corresponding timestamps.

        This corrects the following fields in self.df:
            ['range', 
             'bias', 
             'tx1', 
             'rx1', 
             'tx2', 
             'rx2', 
             'tx3', 
             'rx3', 
             'del_t1', 
             'del_t2']

        TODO: The following fields remain uncorrected (to use tx_rx_split):
            ['tof1', 
             'tof2',
             'tof3',
             'sum_t1',
             'sum_t2']

        Parameters
        ----------
        loss: str, optional
            Loss function to be used in scipy.interpolate.least_squares, by default 'cauchy'
        tx_rx_split: dict, optional
            Splitting the calibrated delay between transmission and reception delay, 
            by default {'tx':0.6, 'rx':0.4} based on 
            "Decawave (2018), APS014: DW1000 Antenna Delay Calibration Version 1.2. 1.15."
        inplace: bool, optional
            Whether to apply the calibration directly to the object, by default False.
        """
        # Get IDs of all tags
        tags = list(np.concatenate(list(self.tag_ids.values())).flat)
        n = len(self.df)
        
        # Find the indices associated with the ranging tags for every measurement
        from_idx = [tags.index(x) for x in np.array(self.df["from_id"])]
        to_idx = [tags.index(x) for x in np.array(self.df["to_id"])]
        rows = np.linspace(0,n-1,n).astype(int)
        
        # Compute the clock-skew gain, if using DS-TWR
        if self.ds_twr:
            K = self.df["del_t3"] / self.df["del_t4"]
        else:
            K = 1
        
        # Compute the A matrix
        A = np.zeros((n,len(tags)))
        A[rows, from_idx] += 0.5
        A[rows, to_idx] += 0.5 * K
        
        # Compute the b column matrix
        b = 1 / self._c * self.df["gt_range"] * 1e9 \
            - 0.5 * self.df["del_t1"] \
            + 0.5 * K * self.df["del_t2"]
        b = np.array(b)

        # Solve for the delays
        x = self._solve_for_antenna_delays(A, b, loss)['x']
        x = x.flatten()

        # Separate the delays per tag
        self.delays = {id: x[i] for i,id in enumerate(tags)}
        
        if inplace:
            # Correct the stored range measurements and timestamps
            self.df = ApplyCalibration.antenna_delays(
                self.df, 
                self.delays,
                tx_rx_split
            )

        return self.delays

    def _solve_for_antenna_delays(
        self, 
        A, 
        b, 
        loss
    ) -> dict:
        """Solve the antenna-delay robust-least-squares problem.

        Parameters
        ----------
        A: np.ndarray
            The A matrix in the Ax=b linear system.
        b: np.ndarray
            The b column matrix in the Ax=b linear system.
        loss: str
            Loss function to be used in scipy.interpolate.least_squares.

        Returns
        -------
        np.ndarray
            The solution to x in the Ax=b linear system, using robust least squares.
        """
        n = A.shape[1]
        return least_squares(
            self._cost_func, 
            np.zeros(n), 
            loss=loss, 
            f_scale=0.1, 
            args=(A,b.T)
        )

    @staticmethod
    def _cost_func(x, A, b) -> np.ndarray:
        """The cost function used in the least squares problem.

        Parameters
        ----------
        x: np.ndarray 
            The unknowns to be solved for; in other words, the x column matrix 
            in the Ax=b linear system.
        A: np.ndarray
            The A matrix in the Ax=b linear system.
        b: np.ndarray
            The b column matrix in the Ax=b linear system.

        Returns
        -------
        np.ndarray
            The evaluated cost.
        """
        return (A@x - b).reshape(-1,)

    @staticmethod
    def get_avg_lifted_pr(pr1, pr2, f) -> float:
        """Get the average of the lifted powers. 

        Parameters
        ----------
        pr1: np.ndarray
            The first power value to be lifted.
        pr2: np.ndarray
            The second power value to be lifted.
        f: _type_
            The lifting function.

        Returns
        -------
        float
            Average lifted power.
        """
        return 0.5 * (f(pr1) + f(pr2))

    def fit_power_model(
        self, 
        std_window=25,
        thresh={'bias': 0.3, 'std': 3}
    ) -> None:
        """Fit the bias vs power and standard deviation vs power splines, and 
        correct the range measurements.

        This adds a new column with header 'std' to the main dataframe, 
        representing the standard deviation of the measurement based on the 
        received first-path power and the calibration results. 

        This corrects the following fields in self.df:
            ['range', 
             'bias']

        TODO: The following fields remain uncorrected:
            ['tx1', 
             'rx1', 
             'tx2', 
             'rx2', 
             'tx3', 
             'rx3', 
             'del_t1',
             'del_t2',
             'tof1',
             'tof2',
             'tof3',
             'sum_t1',
             'sum_t2']

        Parameters
        ----------
        std_window: int, optional
            The window size of measurements for computing the standard deviations, 
            by default 25
        thresh: dict, optional
            The bias thresholds for outlier rejection.
            keys: str
                The type of calibration.
            values: float
                The threshold used for this type of power-correlated calibration.
            by default {'bias': 0.3, 'std': 3}
        """
        # Get the average lifted power
        lifted_pr = self.get_avg_lifted_pr(
            np.array(self.df['fpp1']), 
            np.array(self.df['fpp2']), 
            self.lift
        )

        # Sort the bias and power data based on the power data
        sort_pr = np.argsort(lifted_pr)
        bias = np.array(self.df['bias'])[sort_pr]
        lifted_pr = lifted_pr[sort_pr]

        # Bias vs. FPP: Remove outliers and fit a spline
        lifted_pr_inl, bias_inl = \
                self.remove_outliers(lifted_pr, 
                                     bias, 
                                     thresh['bias'])
        self.bias_spl = self.fit_spline(lifted_pr_inl, 
                                        bias_inl)

        # Standard deviation vs. FPP: Remove outliers and fit a spline
        lifted_pr_inl, bias_inl = \
                self.remove_outliers(lifted_pr, 
                                     bias, 
                                     thresh['std'])
        std = pd.DataFrame(bias_inl).rolling(std_window).std()
        std = std.fillna(method="bfill").fillna(method="ffill").to_numpy()
        self.std_spl = self.fit_spline(lifted_pr_inl, 
                                       std,
                                       k=4)

        # Undo sort
        lifted_pr_unsorted = lifted_pr[np.argsort(sort_pr)]

        # Update the main dataframe with the calibration results
        self.df['std'] = self.std_spl(lifted_pr_unsorted)
        self.df['range'] -= self.bias_spl(lifted_pr_unsorted)
        self.df['bias'] = self.df.apply(
            get_bias, 
            axis=1
        )

    @staticmethod
    def remove_outliers(
        x, 
        y, 
        thresh
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outliers by fitting a spline through the data and 
        removing datapoints that are further from the spline than the
        threshold.

        Parameters
        ----------
        x : np.ndarray
            Independent input data.
        y : np.ndarray
            Dependent input data.
        thresh : float
            Threshold for rejecting outliers.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            x: np.ndarray
                Independent input data inliers.
            y: np.ndarray
                Dependent input data inliers.
        """
        # Fit spline
        spl = UnivariateSpline(x, y, k=3)
        y_fit = spl(x)

        # Find index of datapoints within a threshold away from the fit.
        inliers_idx = np.abs(y - y_fit) <= thresh

        return x[inliers_idx], y[inliers_idx]

    @staticmethod
    def fit_spline(
        x, 
        y, 
        k=3
    ) -> UnivariateSpline:
        """Fit a UnivariateSpline to the data.

        Parameters
        ----------
        x : np.ndarray
            Independent input data.
        y : np.ndarray
            Dependent input data.
        k: int, optional
            Degree of the smoothing spline, by default 3

        Returns
        -------
        UnivariateSpline
            The fitted spline.
        """
        return UnivariateSpline(x, 
                                y, 
                                k=k)

    def save_calib_results(
        self, 
        filename="calib_results.pickle"
    ) -> None:
        """Save the calibration results.

        Parameters
        ----------
        filename: str, optional
            The name of the pickle file, by default "calib_results.pickle"
        """
        calib_results = {
            'delays': self.delays,
            'bias_spl': self.bias_spl,
            'std_spl': self.std_spl,
        }

        with open(filename,"wb") as file:
            pickle.dump(calib_results, file)