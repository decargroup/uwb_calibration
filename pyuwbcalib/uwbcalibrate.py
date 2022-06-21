import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

class UwbCalibrate(object):
    """
    # TODO: Update this and subsequent documentation.
    Object to handle calibration for the DECAR/MRASL UWB modules.

    PARAMETERS:
    -----------
    filename_1: str
        Relative address of the file containing the timestamps of the TWR instances initiated
        by the first tag (hereafter referred to as "tag i").
    filename_2: str
        Relative address of the file containing the timestamps of the TWR instances initiated
        by the second tag (hereafter referred to as "tag j").
    tag_ids: list of ints
        List of IDs of the three tags involved in the calibration procedure.
        The order is as follows:
            1) TWR initializer in filename_1 (tag i).
            2) TWR initializer in filename_2 (tag j).
            3) The tag that never initialized a TWR instance (tag k).
    average: bool
        Flag to indicate whether measurements from static intervals should be averaged out.
    static: bool
        Flag to indicate whether the calibration experiment was done with static intervals.
    thresh: float
        Threshold to detect clock wraps and outliers, in nanoseconds.
    """

    _c = 299702547  # speed of light

    def __init__(self, processed_data, rm_static=True, training_ratio=0.8):
        """
        Constructor
        """
        # Retrieve attributes from processed_data
        self.tag_ids = processed_data.tag_ids
        self.mult_twr = processed_data.mult_twr
        self.num_meas = processed_data.num_meas
        self.tag_pairs = processed_data.tag_pairs
        self.num_of_tags = processed_data.num_of_tags

        if rm_static:
            self.ts_data = {}
            self.time_intervals = {}
            self.ts_data, self.time_intervals \
                = self._remove_static_regions(processed_data.ts_data, processed_data.time_intervals)
        else:
            self.ts_data = processed_data.ts_data
            self.time_intervals = processed_data.time_intervals

        self.range_idx = 1
        self.tx1_idx = 2
        self.rx1_idx = 3
        self.tx2_idx = 4
        self.rx2_idx = 5
        self.tx3_idx = 6
        self.rx3_idx = 7
        self.Pr1_idx = 8
        self.Pr2_idx = 9

        self.lift = processed_data.lift

        self._split_test_data(training_ratio) 

    def _split_test_data(self, training_ratio):
        # TODO:
        pass

    def _remove_static_regions(self, ts_data, time_intervals):
        '''
        Remove the static region in the extremes.
        '''
        lower_idxs, upper_idxs = self._find_static_extremes(ts_data, time_intervals)
        
        num_columns = 10
        num_rows = lambda pair: upper_idxs[pair] - lower_idxs[pair] 
        ts_data_trunc = {pair:np.zeros((num_rows(pair), num_columns)) for \
                                                                pair in ts_data}
        time_intervals_trunc = {pair:{} for pair in ts_data}

        for pair in ts_data:
            l_idx = lower_idxs[pair]
            u_idx = upper_idxs[pair]
            
            for column in range(0,num_columns):
                ts_data_trunc[pair][:,column] \
                    = ts_data[pair][:,column][l_idx:u_idx]

            for topic in time_intervals[pair]:
                time_intervals_trunc[pair][topic] \
                    = time_intervals[pair][topic][l_idx:u_idx]

        return ts_data_trunc, time_intervals_trunc

    @staticmethod
    def _find_static_extremes(ts_data, time_intervals):
        lower_idxs = {pair:[] for pair in ts_data}
        upper_idxs = {pair:[] for pair in ts_data}
        
        thresh = 0.2

        for pair in ts_data:
            gt = time_intervals[pair]['r_gt']
            
            # Lower bound
            p1 = gt[0]
            p2 = gt[100]
            p3 = gt[200]

            mean = (p1+p2+p3)/3
            cond1 = np.abs(p1 - mean) > thresh
            cond2 = np.abs(p2 - mean) > thresh
            cond3 = np.abs(p3 - mean) > thresh
            if cond1 or cond2 or cond3:
                lower_idxs[pair] = 0
            else:
                mean = np.mean(gt[:200])
                deviation = gt - mean
                deviation_bool = deviation > thresh

                # Find the first 2 consecutive true values
                found_idx = False
                for lv0 in range(401, len(gt)-5):
                    cond = np.all(deviation_bool[lv0:lv0+2])
                    if cond:
                        lower_idxs[pair] = lv0
                        found_idx = True
                        break

                if not found_idx:
                    lower_idxs[pair] = 0

            # Upper bound
            p1 = gt[-1]
            p2 = gt[-100]
            p3 = gt[-200]

            mean = (p1+p2+p3)/3
            cond1 = np.abs(p1 - mean) > thresh
            cond2 = np.abs(p2 - mean) > thresh
            cond3 = np.abs(p3 - mean) > thresh
            if cond1 or cond2 or cond3:
                upper_idxs[pair] = len(gt)
            else:
                mean = np.mean(gt[-200:])
                deviation = gt - mean
                deviation_bool = deviation > thresh

                # Find the first 2 consecutive true values
                found_idx = False
                for lv0 in range(-401, -len(gt)+5):
                    cond = np.all(deviation_bool[lv0:lv0-2])
                    if cond:
                        upper_idxs[pair] = lv0
                        found_idx = True
                        break

                if not found_idx:
                    upper_idxs[pair] = len(gt)

        return lower_idxs, upper_idxs
 

    def _calculate_skew_gain(self, pair):
        """
        Calculates the K parameter given by Ra2/Db2.
        Gain set to 1 if twr_type == 0.

        PARAMETERS:
        -----------
        initiating_idx: int
            The index of the initiating tag (initiator) in self.tag_ids.
        target_idx: int
            The index of the target tag in self.tag_ids.

        RETURNS:
        --------
        np.array: The K values for all the measurements.
        """
        Ra2 = self.time_intervals[pair]["Ra2"]
        Db2 = self.time_intervals[pair]["Db2"]

        if self.mult_twr:
            return Ra2 / Db2
        else:
            return Ra2 / Ra2

    def _setup_A_matrix(self, pair, tags, K):
        """
        Calculates the A matrix for the linear least-squares problem.

        PARAMETERS:
        -----------
        K: np.array
            The skew gain K.
        initiating_idx: int
            The index of the initiating tag (initiator) in self.tag_ids.
        target_idx: int
            The index of the target tag in self.tag_ids.

        RETURNS:
        --------
        2D np.array: The A matrix.
        """
        initiating_idx = tags.index(pair[0])
        target_idx = tags.index(pair[1]) 

        n = len(K)
        A = np.zeros((n, len(tags)))
        A[:, initiating_idx] += 0.5
        A[:, target_idx] = 0.5 * K

        return A

    def _setup_b_vector(self, pair, K):
        """
        Calculates the b vector for the linear least-squares problem.

        PARAMETERS:
        -----------
        K: np.array
            The skew gain K.
        initiating_idx: int
            The index of the initiating tag (initiator) in self.tag_ids.
        target_idx: int
            The index of the target tag in self.tag_ids.

        RETURNS:
        --------
        np.array: The b vector.
        """
        gt = self.time_intervals[pair]["r_gt"]
        Ra1 = self.time_intervals[pair]["Ra1"]
        Db1 = self.time_intervals[pair]["Db1"]

        b = 1 / self._c * gt * 1e9 - 0.5 * (Ra1) + 0.5 * K * (Db1)

        return np.reshape(b, (len(K), 1))

    def _solve_for_antenna_delays(self, A, b):
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
        return np.linalg.lstsq(A, b)

    def filter_data(self, Q, R, visualize=False):
        if visualize:
            num_of_pairs = len(self.time_intervals)
            fig, axs = plt.subplots(num_of_pairs)

        for lv0, pair in enumerate(self.tag_pairs):
            x_hist, P_hist = self._clock_filter(pair, Q, R)
            
            self._update_tof_intervals(pair, x_hist[0,:])

            if visualize: 
                Ra2 = self.time_intervals[pair]["Ra2"]
                Db2 = self.time_intervals[pair]["Db2"]
                S1 = self.time_intervals[pair]["S1"]
                S2 = self.time_intervals[pair]["S2"]
                Db1 = self.time_intervals[pair]["Db1"]
                y1 = 0.5*(S1 - S2)
                y2 = Db2 - Ra2
                y_tau = - y1 - 0.5*(Db1/(Db2-Db1))*y2

                self._plot_kf(x_hist, P_hist, axs[lv0], y_tau)

        if visualize:
            plt.show()

    def _update_tof_intervals(self, pair, tau):
        # TODO: Take uncertainty into consideration?
        self.time_intervals[pair]["tof1"] \
            = self.time_intervals[pair]["tof1"] - tau

        self.time_intervals[pair]["tof2"] \
            = self.time_intervals[pair]["tof2"] + tau

        if self.mult_twr:
            self.time_intervals[pair]["tof3"] \
                = self.time_intervals[pair]["tof3"] + tau

    @staticmethod
    def _plot_kf(x, P, axs, y_tau):
        axs.plot(x[0,:]-y_tau)

        axs.ticklabel_format(style='plain')
        
        P_iter = P[1,1,:]
        P_iter = P_iter.reshape(-1,)
        axs.plot(x[0,:] + 3*np.sqrt(P_iter))
        axs.plot(x[0,:] - 3*np.sqrt(P_iter))

    def _clock_filter(self, pair, Q, R):
        # Intervals
        dt = self.time_intervals[pair]["dt"]
        Ra2 = self.time_intervals[pair]["Ra2"]
        Db2 = self.time_intervals[pair]["Db2"]
        S1 = self.time_intervals[pair]["S1"]
        S2 = self.time_intervals[pair]["S2"]
        Db1 = self.time_intervals[pair]["Db1"]

        # Storage variables
        n = dt.size
        x_hist = np.zeros((2,n))
        P_hist = np.zeros((2,2,n))

        # Initial estimate and uncertainty
        tau = 0
        skew = 0
        x = np.array([tau, skew])
        x = x.reshape(2,1)
        P = np.array(([1e24,0],[0,1e24])) # TODO: better estimate of initial uncertainty

        for lv0, dt_iter in enumerate(dt):
            Ra2_iter = Ra2[lv0]
            Db2_iter = Db2[lv0]
            S1_iter = S1[lv0]
            S2_iter = S2[lv0]
            Db1_iter = Db1[lv0]

            if lv0>0:
                x, P = self._propagate_clocks(x, P, dt_iter, Q)
                P = 0.5*(P + P.T)

            y = self._compute_pseudomeasurement(Ra2_iter, Db2_iter, S1_iter, S2_iter, Db1_iter)
            x, P = self._correct_clocks(x, P, y, R, Db1_iter, Db2_iter)
            P = 0.5*(P + P.T)

            x_hist[:,lv0] = x.reshape(2,)
            P_hist[:,:,lv0] = P

        return x_hist, P_hist

    @staticmethod
    def _compute_pseudomeasurement(Ra2, Db2, S1, S2, Db1):
        y1 = 0.5*(S1 - S2)
        y2 = Db2 - Ra2
        # return np.array([y1, y2]).reshape(2,1)
        y = - y1 - 0.5*(Db1/(Db2-Db1))*y2
        return y

    @staticmethod
    def _correct_clocks(x, P, y, R, Db1, Db2):
        # C = np.array([[-1,0.5*Db1/1e9], [0, -Db2/1e9]])
        C = np.array([1, 0])
        C = C.reshape(1,2)

        y_check = C @ x

        # R_matrix = np.array([[1.25*R, 0.5*R], [0.5*R, 2*R]])
        R_matrix = R

        S = C @ P @ C.T + R_matrix
        K = P @ C.T @ inv(S)

        x_new = x + K @ (y - y_check)
        P_new = (np.eye(2) - K @ C) @ P

        return x_new, P_new

    @staticmethod
    def _propagate_clocks(x, P, dt, Q):
        dt = dt/1e9
        A = np.array(([1, dt], [0, 1]))
        L = np.array(([dt, 0.5*dt**2], [0, dt]))

        x_new = A @ x
        P_new = A @ P @ A.T + L @ Q @ L.T
        
        return x_new, P_new

    @staticmethod
    def _rolling_window(a, window):
        '''
        Copied from 
        https://stackoverflow.com/questions/27427618/how-can-i-simply-calculate-the-rolling-moving-variance-of-a-time-series-in-pytho
        '''
        pad = np.ones(len(a.shape), dtype=np.int32)
        pad[-1] = window-1
        pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
        a = np.pad(a, pad,mode='reflect')
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def _reject_outliers(self, bias, lifted_pr, std_window, chi_thresh, axs):
        '''
        keep fitting models and rejecting outliers until no more outliers
        '''
        outlier_bias = np.empty(0,)
        outlier_lifted_pr = np.empty(0,)
        while True:
            # Fit 
            bias_std = np.std(self._rolling_window(bias.ravel(), std_window), axis=-1)
            std_spl = UnivariateSpline(lifted_pr, bias_std, k=3)
            bias_std = std_spl(lifted_pr)

            # Fit spline
            spl = UnivariateSpline(lifted_pr, bias)
            bias_fit = spl(lifted_pr)

            # Remove outliers
            norm_e_squared = (bias - bias_fit)**2 / bias_std**2
            outliers = norm_e_squared > chi_thresh
            if np.sum(outliers):
                outlier_bias = np.hstack((outlier_bias, bias[outliers]))
                bias = bias[~outliers]

                outlier_lifted_pr = np.hstack((outlier_lifted_pr, lifted_pr[outliers]))
                lifted_pr = lifted_pr[~outliers] 
            else:
                break

        # PLOTTING
        axs.scatter(lifted_pr, bias)
        axs.scatter(outlier_lifted_pr, outlier_bias)
        axs.set_xlabel(r"$f(P_r)$")
        axs.set_ylabel(r"Bias [m]")

        return spl, std_spl, bias, lifted_pr

    def fit_model(self, std_window=50, chi_thresh=10.8):
        num_pairs = len(self.ts_data)
        fig, axs = plt.subplots(3,num_pairs,sharey='row')
        fig2, axs2 = plt.subplots(1) 
        fig3, axs3 = plt.subplots(num_pairs,sharey='row') 
        fig3.suptitle(r"Outlier rejection")
        axs[0,0].set_ylabel(r"Bias [m]")
        axs[1,0].set_ylabel(r"Bias std [m]")
        axs[2,0].set_ylabel(r"Bias std [m]")

        self.mean_spline = {pair:[] for pair in self.ts_data}

        for lv0, pair in enumerate(self.tag_pairs):
            range = self.compute_range_meas(pair)
            bias = range - self.time_intervals[pair]["r_gt"]
            lifted_pr = self.lift(0.5*self.ts_data[pair][:,self.Pr1_idx] \
                                  + 0.5*self.ts_data[pair][:,self.Pr2_idx])
            r_gt_unsorted = self.time_intervals[pair]["r_gt"]

            pr_thresh = 2
            bias = bias[lifted_pr < pr_thresh]
            r_gt_unsorted = r_gt_unsorted[lifted_pr < pr_thresh]
            lifted_pr = lifted_pr[lifted_pr < pr_thresh]   

            # rolling var along last axis
            sort_pr = np.argsort(lifted_pr)
            bias = bias[sort_pr]
            lifted_pr = lifted_pr[sort_pr]
            r_gt = r_gt_unsorted[sort_pr]

            spl, std_spl, bias_trunc, lifted_pr_trunc \
                        = self._reject_outliers(bias, lifted_pr, std_window, chi_thresh, axs3[lv0])
            self.mean_spline[pair] = spl
            bias_std = std_spl(lifted_pr)
            bias_fit = spl(lifted_pr)

            ### PLOT 1 ###
            axs[0,lv0].scatter(lifted_pr_trunc, bias_trunc, label=r"Raw data", linestyle="dotted", s=1)
            axs[0,lv0].plot(lifted_pr, bias_fit, label=r"Fit")
            axs[0,lv0].fill_between(
                lifted_pr.ravel(),
                bias_fit - 1.96 * bias_std,
                bias_fit + 1.96 * bias_std,
                alpha=0.5,
                label=r"95% confidence interval",
            )
            axs[0,lv0].set_xlabel(r"$f(P_r)$")
            
            ## Visualize std vs. Power
            axs[1,lv0].plot(lifted_pr, bias_std)
            axs[1,lv0].set_xlabel(r"$f(P_r)$")
            
            ## Visualize std vs. distance
            axs[2,lv0].scatter(r_gt, bias_std, s=1)
            axs[2,lv0].set_xlabel(r"Ground truth distance [m]")

            ### PLOT 2 ### Plot with all splines
            axs2.plot(lifted_pr, bias_fit, label=r"Pair "+str(pair))
            axs2.legend()
            axs2.set_xlabel(r"$f(P_r)$")
            axs2.set_ylabel(r"Bias [m]")
            fig2.suptitle(r"Bias-Power Fit")

        axs[0,-1].legend()

    def calibrate_antennas(self):
        """
        Calibrate the antenna delays by formulating and solving a linear least-squares problem.

        RETURNS:
        --------
        dict: Dictionary with 3 fields each for tag z \in {i,j,k}
            Module i: (float)
                Antenna delay for tag i
        """
        tags = sum(list(self.tag_ids.values()),[])

        A = np.zeros((0,len(tags)))
        b = np.zeros((0,1))
        for pair in self.tag_pairs:
            K = self._calculate_skew_gain(pair)
            A = np.vstack((A, self._setup_A_matrix(pair, tags, K)))
            b = np.vstack((b, self._setup_b_vector(pair, K)))

        nan_idx = ~np.isnan(b)
        nan_idx = nan_idx.flatten()
        A = A[nan_idx, :]
        b = b[nan_idx]

        x = self._solve_for_antenna_delays(A, b)[0]
        x = x.flatten()

        print(np.linalg.norm(b))
        print(np.linalg.norm(b.T - A @ x))

        return {id: x[i] for i,id in enumerate(tags)}

    def correct_antenna_delay(self, delays_dict):
        """
        Modifies the data of this object to correct for the antenna delay of a
        specific module.

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
        """
        for id in delays_dict.keys():
            delay = delays_dict[id]
            for pair in self.time_intervals:
                if pair[0] == id:
                    self.time_intervals[pair]["Ra1"] += delay
                elif pair[1] == id:
                    self.time_intervals[pair]["Db1"] -= delay

    def compute_range_meas(self, pair=(1,2), visualize=False, owr = False):
        # TODO: Inherit this function from PostProcess?
        interv = self.time_intervals[pair]
        if owr and self.mult_twr:
            range = 1/2 * self._c / 1e9 \
                    * (abs(interv["tof1"]) \
                       + abs(interv["tof2"]) \
                       + 0*abs(interv["tof3"])) # TODO: why *0? Antenna delay?
        elif owr:
            range = 1/2 * self._c / 1e9 \
                    * (abs(interv["tof1"]) \
                       + abs(interv["tof2"]))
        elif self.mult_twr:
            range = 0.5 * self._c / 1e9 * \
                (interv["Ra1"] - (interv["Ra2"] / interv["Db2"]) * interv["Db1"])
        else:
            range = 0.5 * self._c / 1e9 * \
                (interv["Ra1"] - interv["Db1"])

        if visualize:
            fig, axs = plt.subplots(1)

            axs.plot(interv["t"]/1e9, range, label='Range Measurements')

            axs.set_ylabel("Distance [m]")
            axs.set_xlabel("t [s]")
            axs.set_ylim([-1, 5])

        return range