from typing import Tuple
import numpy as np
from .postprocess import PostProcess
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares
import pickle
import pandas as pd

class UwbCalibrate(PostProcess):
    """_summary_

    Attributes
    ----------
    object : _type_
        _description_

    Examples
    --------
    """
    """_summary_

    Parameters
    ----------
    PostProcess : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    _c = 299702547  # speed of light
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
        """_summary_

        Parameters
        ----------
        data : _type_
            _description_
        rm_static : bool, optional
            _description_, by default False
        f_lift : _type_, optional
            _description_, by default lambdax:10**((x + 82)/10)
        """
        """
        Constructor
        Mention in the documentation somewhere that the range, bias, timestamps and intervals are updated
        with the antenna-delay calibration results (not tof and sums though), 
        but only the range and bias are updated with the 
        power-calibration results. So the timestamps might need further processing if used alone.
        """
        self._data = data
        
        if rm_static:
            self._remove_static_regions()

        self.lift = f_lift

    def __getattr__(self, attr) -> object:
        """_summary_

        Parameters
        ----------
        attr : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        if attr in self._inherited:
            return getattr(self._data, attr)

    def _remove_static_regions(self) -> None:
        """_summary_
        """
        '''
        Remove the static region in the extremes.
        TODO: extensively test this with a few datasets.
        '''        
        thresh = 0.1
        window = 10 # size of windows
        
        lower_idx = 0
        upper_idx = np.Inf

        for machine_i in self.machine_ids:
            r_iw_a = self.get_machine_pos(machine_i, as_numpy = True)
            
            # Lower bound
            x = r_iw_a[:,0]
            y = r_iw_a[:,1]
            z = r_iw_a[:,2]
            
            x_lower, x_upper = self._find_static_extremes(x, thresh, window)
            y_lower, y_upper = self._find_static_extremes(y, thresh, window)
            z_lower, z_upper = self._find_static_extremes(z, thresh, window)
            
            lowest_low_idx = np.min([x_lower, y_lower, z_lower])
            highest_upper_idx = np.max([x_upper, y_upper, z_upper])
            
            if lowest_low_idx > lower_idx:
                lower_idx = lowest_low_idx
            
            if highest_upper_idx < upper_idx:
                upper_idx = highest_upper_idx
            
        n = len(self.df)
        self.df.drop(np.linspace(0,lower_idx,lower_idx+1), inplace=True)
        self.df.drop(np.linspace(upper_idx, n-1, n-upper_idx), inplace=True)
        # self.df.reset_index(inplace=True, drop=True)
        
    @staticmethod
    def _find_static_extremes(
        r, 
        thresh, 
        window
    ) -> Tuple[int, int]:
        """_summary_

        Parameters
        ----------
        r : _type_
            _description_
        thresh : _type_
            _description_
        window : _type_
            _description_

        Returns
        -------
        Tuple[int, int]
            _description_
        """
        rolling_mean = pd.DataFrame(r).rolling(window, center=True).mean()
        rolling_mean = rolling_mean.fillna(method="bfill").fillna(method="ffill")
        rolling_mean = np.array(rolling_mean)
        
        diff_lower = np.abs(rolling_mean[1:] - rolling_mean[0])
        diff_upper = np.abs(np.flip(rolling_mean)[1:] - rolling_mean[-1])
        
        lower_idx = np.argmax(diff_lower>thresh)
        upper_idx = len(diff_lower) - np.argmax(np.flip(diff_upper)>thresh) - 1
        
        return lower_idx, upper_idx

    def calibrate_antennas(
        self, 
        loss='cauchy', 
        tx_rx_split={'tx':0.6, 'rx':0.4},
    ) -> None:
        """_summary_

        Parameters
        ----------
        loss : str, optional
            _description_, by default 'cauchy'
        tx_rx_split : dict, optional
            _description_, by default {'tx':0.6, 'rx':0.4}
        """
        """
        Calibrate the antenna delays by formulating and solving a linear least-squares problem.

        RETURNS:
        --------
        dict: Dictionary with 3 fields each for tag z \in {i,j,k}
            Module i: (float)
                Antenna delay for tag i
        """
        tags = list(np.concatenate(list(self.tag_ids.values())).flat)
        n = len(self.df)
        
        from_idx = [tags.index(x) for x in np.array(self.df["from_id"])]
        to_idx = [tags.index(x) for x in np.array(self.df["to_id"])]
        rows = np.linspace(0,n-1,n).astype(int)
        
        if self.ds_twr:
            K = self.df["del_t3"] / self.df["del_t4"]
        else:
            K = 1
        
        A = np.zeros((n,len(tags)))
        A[rows, from_idx] += 0.5
        A[rows, to_idx] += 0.5 * K
        
        b = 1 / self._c * self.df["gt_range"] * 1e9 \
            - 0.5 * self.df["del_t1"] \
            + 0.5 * K * self.df["del_t2"]
        b = np.array(b)

        x = self._solve_for_antenna_delays(A, b, loss)['x']
        x = x.flatten()

        self.delays = {id: x[i] for i,id in enumerate(tags)}
        
        self._correct_antenna_delays(tx_rx_split)

    def _correct_antenna_delays(self, tx_rx_split) -> None:
        """_summary_

        Parameters
        ----------
        tx_rx_split : _type_
            _description_
        """
        """
        Modifies the data of this object to correct for the antenna delay of a
        specific module.

        Mention that individual timestamps is based on Decawave's split of 60 40.

        PARAMETERS:
        -----------
        id: int
            Module ID whose antenna delay is to be corrected.
        delay: float
            The amount of antenna delay, in nanoseconds.

        TODO: What about D1 and D2? This seems to be a problem.
              We might have to calibrate for TX and RX delays separately
              if we are to proceed with Kalman filtering with this architecture.
        TODO: tof1, tof2, and tof3 as well, once individual delays are taken into consideration.
        TODO: access delays from self object
        """
        
        from_delay = np.array([self.delays[x] for x in np.array(self.df["from_id"])])
        to_delay = np.array([self.delays[x] for x in np.array(self.df["to_id"])])
        
        self.df["del_t1"] += from_delay
        self.df["del_t2"] -= to_delay
        
        tx_from_delay = tx_rx_split['tx'] * from_delay
        rx_from_delay = tx_rx_split['rx'] * from_delay
        tx_to_delay = tx_rx_split['tx'] * to_delay
        rx_to_delay = tx_rx_split['rx'] * to_delay

        self.df["tx1"] = -tx_from_delay
        self.df["rx2"] = rx_from_delay

        self.df["rx1"] = rx_to_delay
        self.df["tx2"] = -tx_to_delay

        if self.ds_twr:
            self.df["tx3"] = -tx_to_delay
            self.df["rx3"] = rx_to_delay

        self.df['range'] = self.compute_range_meas()
        self.df['bias'] = self.df.apply(
                                        self._get_bias, 
                                        axis=1
                                       )

    def _solve_for_antenna_delays(
        self, 
        A, 
        b, 
        loss
    ) -> dict:
        """_summary_

        Parameters
        ----------
        A : _type_
            _description_
        b : _type_
            _description_
        loss : _type_
            _description_

        Returns
        -------
        dict
            _description_
        """
        """
        Solves the linear least-squares problem.

        PARAMETERS:
        -----------
        A: 2D np.array
            The A matrix.
        b: np.array
            The b vector.

        RETURNS:
        --------
        np.array: The solution to the Ax=b problem.
        """
        n = A.shape[1]
        return least_squares(self._cost_func, np.zeros(n), loss=loss, f_scale=0.1, args=(A,b.T))

    @staticmethod
    def _cost_func(x, A, b) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        x : _type_
            _description_
        A : _type_
            _description_
        b : _type_
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        return (A@x - b).reshape(-1,)

    @staticmethod
    def _rolling_window(a, window) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        a : _type_
            _description_
        window : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        '''
        Copied from shorturl.at/adH38.
        '''
        pad = np.ones(len(a.shape), dtype=np.int32)
        pad[-1] = window-1
        pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
        a = np.pad(a, pad,mode='reflect')
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    @staticmethod
    def get_avg_lifted_pr(pr1, pr2, f) -> float:
        """_summary_

        Parameters
        ----------
        pr1 : _type_
            _description_
        pr2 : _type_
            _description_
        f : _type_
            _description_

        Returns
        -------
        float
            _description_
        """
        return 0.5 * (f(pr1) + f(pr2))

    def fit_power_model(
        self, 
        std_window=25,
        thresh={'bias': 0.3, 'std': 3}
    ) -> None:
        """_summary_

        Parameters
        ----------
        std_window : int, optional
            _description_, by default 25
        thresh : dict, optional
            _description_, by default {'bias': 0.3, 'std': 3}
        """
        bias = np.array(self.df['bias'])
        pr1 = np.array(self.df['fpp1'])
        pr2 = np.array(self.df['fpp2'])
        lifted_pr = self.get_avg_lifted_pr(pr1, pr2, self.lift)

        sort_pr = np.argsort(lifted_pr)
        bias = bias[sort_pr]
        lifted_pr = lifted_pr[sort_pr]

        lifted_pr_inl, bias_inl = \
                self.remove_outliers(lifted_pr, 
                                     bias, 
                                     thresh['bias'])
        self.bias_spl = self.fit_spline(lifted_pr_inl, 
                                        bias_inl)

        lifted_pr_inl, bias_inl = \
                self.remove_outliers(lifted_pr, 
                                     bias, 
                                     thresh['std'])
        std = self.get_rolling_std(bias_inl,
                                   std_window)
        self.std_spl = self.fit_spline(lifted_pr_inl, 
                                       std,
                                       k=4)

        lifted_pr_unsorted = lifted_pr[np.argsort(sort_pr)] # undo sort
        self.df['std'] = self.std_spl(lifted_pr_unsorted)
        self.df['range'] -= self.bias_spl(lifted_pr_unsorted)
        self.df['bias'] = self.df.apply(
                                        self._get_bias, 
                                        axis=1
                                       )

    @staticmethod
    def remove_outliers(
        lifted_pr, 
        bias, 
        thresh
    ) -> Tuple[np.ndarray, np.ndarray]:
        """_summary_

        Parameters
        ----------
        lifted_pr : _type_
            _description_
        np : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        spl = UnivariateSpline(lifted_pr, bias, k=3)
        bias_fit = spl(lifted_pr)
        inliers_idx = np.abs(bias - bias_fit) <= thresh
        return lifted_pr[inliers_idx], bias[inliers_idx]

    @staticmethod
    def fit_spline(
        lifted_pr, 
        bias, 
        k=3
    ) -> UnivariateSpline:
        """_summary_

        Parameters
        ----------
        lifted_pr : _type_
            _description_
        bias : _type_
            _description_
        k : int, optional
            _description_, by default 3

        Returns
        -------
        UnivariateSpline
            _description_
        """
        return UnivariateSpline(lifted_pr, 
                                bias, 
                                k=k)

    def get_rolling_std(
        self, 
        bias, 
        std_window
    ) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        bias : _type_
            _description_
        std_window : _type_
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        windows = self._rolling_window(bias.ravel(), std_window)
        return np.std(windows - np.mean(windows).reshape(-1,1), axis=-1)

    def compute_range_meas(self) -> np.ndarray:
        """_summary_

        Returns
        -------
        np.ndarray
            _description_
        """
        del_t1 = self.df['del_t1']
        del_t2 = self.df['del_t2']
        if self.ds_twr:
            del_t3 = self.df['del_t3']
            del_t4 = self.df['del_t4']
            range = 0.5 * self._c / 1e9 * \
                    (del_t1 - (del_t3 / del_t4) * del_t2)
        else:
            range = 0.5 * self._c / 1e9 * \
                    (del_t1 - del_t2)

        return range

    def save_calib_results(
        self, 
        filename="calib_results.pickle"
    ) -> None:
        """_summary_

        Parameters
        ----------
        filename : str, optional
            _description_, by default "calib_results.pickle"
        """
        calib_results = {
            'delays': self.delays,
            'bias_spl': self.bias_spl,
            'std_spl': self.std_spl,
        }

        with open(filename,"wb") as file:
            pickle.dump(calib_results, file)