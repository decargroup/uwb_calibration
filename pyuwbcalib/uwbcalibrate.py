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

    def __init__(self, processed_data, outliers=False):
        """
        Constructor
        """
        # Retrieve attributes from processed_data
        # TODO: there must be a better way to do this without confusing intelliSense
        self.num_of_recordings = processed_data.num_of_recordings
        self.tag_ids = processed_data.tag_ids
        self.mult_twr = processed_data.mult_twr
        self.num_meas = processed_data.num_meas

        self.num_of_tags = processed_data.num_of_tags

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
        
        if not outliers:
            self._remove_outliers()

    def _remove_outliers(self):
        # TODO: implement an outlier rejection algorithm.
        pass

    def _calculate_skew_gain(self, initiating_idx, target_idx):
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
        str_temp = (
            str(self.board_ids[initiating_idx]) + "->" + str(self.board_ids[target_idx])
        )
        data = self.data[str_temp]

        Ra2 = data["Ra2"]
        Db2 = data["Db2"]

        if self.twr_type == 0:
            return Ra2 / Ra2
        else:
            return Ra2 / Db2

    def _setup_A_matrix(self, K, initiating_idx, target_idx):
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
        n = len(K)
        A = np.zeros((n, 3))
        A[:, initiating_idx] += 0.5
        A[:, target_idx] = 0.5 * K

        return A

    def _setup_b_vector(self, K, initiating_idx, target_idx):
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
        str_temp = (
            str(self.board_ids[initiating_idx]) + "->" + str(self.board_ids[target_idx])
        )
        data = self.data[str_temp]

        gt = data["gt"]
        Ra1 = data["Ra1"]
        Db1 = data["Db1"]

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
            num_of_pairs = len(self.time_intervals[0])
            fig, axs = plt.subplots(self.num_of_recordings, num_of_pairs)
            axs = axs.reshape(self.num_of_recordings, num_of_pairs)

        for lv0, recording in enumerate(self.time_intervals):
            for lv1, pair in enumerate(self.time_intervals[recording]):
                x_hist, P_hist = self._clock_filter(recording, pair, Q, R)
                
                self._update_tof_intervals(recording, pair, x_hist[0,:])

                if visualize: 
                    Ra2 = self.time_intervals[recording][pair]["Ra2"]
                    Db2 = self.time_intervals[recording][pair]["Db2"]
                    S1 = self.time_intervals[recording][pair]["S1"]
                    S2 = self.time_intervals[recording][pair]["S2"]
                    Db1 = self.time_intervals[recording][pair]["Db1"]
                    y1 = 0.5*(S1 - S2)
                    y2 = Db2 - Ra2
                    y_tau = - y1 - 0.5*(Db1/(Db2-Db1))*y2
                    # axs[lv0,lv1].plot(-y_tau)

                    self._plot_kf(x_hist, P_hist, axs[lv0,lv1], y_tau)

        if visualize:
            plt.show()

    def _update_tof_intervals(self, recording, pair, tau):
        # TODO: Take uncertainty into consideration?
        self.time_intervals[recording][pair]["tof1"] \
            = self.time_intervals[recording][pair]["tof1"] - tau

        self.time_intervals[recording][pair]["tof2"] \
            = self.time_intervals[recording][pair]["tof2"] + tau

        if self.mult_twr:
            self.time_intervals[recording][pair]["tof3"] \
                = self.time_intervals[recording][pair]["tof3"] + tau

    @staticmethod
    def _plot_kf(x, P, axs, y_tau):
        axs.plot(x[0,:]-y_tau)

        axs.ticklabel_format(style='plain')
        
        # P_iter = P[1,1,:]
        # P_iter = P_iter.reshape(-1,)
        # axs.plot(x[0,:] + 3*np.sqrt(P_iter))
        # axs.plot(x[0,:] - 3*np.sqrt(P_iter))

    def _clock_filter(self, recording, pair, Q, R):
        # Intervals
        dt = self.time_intervals[recording][pair]["dt"]
        Ra2 = self.time_intervals[recording][pair]["Ra2"]
        Db2 = self.time_intervals[recording][pair]["Db2"]
        S1 = self.time_intervals[recording][pair]["S1"]
        S2 = self.time_intervals[recording][pair]["S2"]
        Db1 = self.time_intervals[recording][pair]["Db1"]

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

    def fit_model(self, pair, std_window=50):
        alpha = -82 # TODO: Copy this over from PostProcess
        lift = lambda x: 10**((x - alpha) /10)

        bias = self.ts_data[0][pair][:,self.range_idx] - self.time_intervals[0][pair]["r_gt"]
        lifted_pr = lift(self.ts_data[0][pair][:,self.Pr1_idx])
        r_gt_unsorted = self.time_intervals[0][pair]["r_gt"]

        ## TODO: REMOVE THIS ONCE PROPER OUTLIER DETECTION IS IMPLEMENTED
        thresh = 0.75
        keep_idx = np.logical_and(bias < thresh, lifted_pr < 1.5)
        bias = bias[keep_idx]
        lifted_pr = lifted_pr[keep_idx]
        r_gt_unsorted = r_gt_unsorted[keep_idx]
        bias = bias[200:-200]
        lifted_pr = lifted_pr[200:-200]
        r_gt_unsorted = r_gt_unsorted[200:-200]        

        # rolling var along last axis
        sort_pr = np.argsort(lifted_pr)
        bias = bias[sort_pr]
        lifted_pr = lifted_pr[sort_pr]
        r_gt = r_gt_unsorted[sort_pr]

        bias_std = np.std(self._rolling_window(bias.ravel(), std_window), axis=-1)
        std_spl = UnivariateSpline(lifted_pr, bias_std, k=5)
        bias_std = std_spl(lifted_pr)

        # Fit spline
        spl = UnivariateSpline(lifted_pr, bias)
        bias_fit = spl(lifted_pr)

        # PLOTTING
        fig, axs = plt.subplots(3,1)

        axs[0].scatter(lifted_pr, bias, label=r"Raw data", linestyle="dotted", s=1)
        # axs[0].scatter(X_train, y_train, label="Observations")
        axs[0].plot(lifted_pr, bias_fit, label="Fit")
        axs[0].fill_between(
            lifted_pr.ravel(),
            bias_fit - 1.96 * bias_std,
            bias_fit + 1.96 * bias_std,
            alpha=0.5,
            label=r"95% confidence interval",
        )
        axs[0].legend()
        axs[0].set_xlabel("$f(P_r)$")
        axs[0].set_ylabel("Bias [m]")
        # _ = axs[0].title("Gaussian process regression on noise-free dataset")

        ## Visualize std vs. Power
        axs[1].plot(lifted_pr, bias_std)
        axs[1].set_xlabel("$f(P_r)$")
        axs[1].set_ylabel("Bias std [m]")

        ## Visualize std vs. distance
        axs[2].scatter(r_gt, bias_std, s=1)
        axs[2].set_xlabel("Ground truth distance [m]")
        axs[2].set_ylabel("Bias std [m]")

        plt.show()

    def calibrate_antennas(self):
        """
        Calibrate the antenna delays by formulating and solving a linear least-squares problem.

        RETURNS:
        --------
        dict: Dictionary with 3 fields each for tag z \in {i,j,k}
            Module i: (float)
                Antenna delay for tag i
        """
        K1 = self._calculate_skew_gain(0, 1)
        A1 = self._setup_A_matrix(K1, 0, 1)
        b1 = self._setup_b_vector(K1, 0, 1)

        K2 = self._calculate_skew_gain(0, 2)
        A2 = self._setup_A_matrix(K2, 0, 2)
        b2 = self._setup_b_vector(K2, 0, 2)

        K3 = self._calculate_skew_gain(1, 2)
        A3 = self._setup_A_matrix(K3, 1, 2)
        b3 = self._setup_b_vector(K3, 1, 2)

        A = np.vstack((A1, A2, A3))
        b = np.vstack((b1, b2, b3))

        nan_idx = ~np.isnan(b)
        nan_idx = nan_idx.flatten()
        A = A[nan_idx, :]
        b = b[nan_idx]

        x = self._solve_for_antenna_delays(A, b)[0]
        x = x.flatten()

        print(np.linalg.norm(b))
        print(np.linalg.norm(b - A * np.array([x[0], x[1], x[2]])))

        return {
            "Module " + str(self.board_ids[0]): x[0],
            "Module " + str(self.board_ids[1]): x[1],
            "Module " + str(self.board_ids[2]): x[2],
        }

    def correct_antenna_delay(self, id, delay):
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
        """
        for key in self.data:
            if int(key.partition("-")[0]) == id:
                self.data[key]["Ra1"] = self.data[key]["Ra1"] + delay
            elif int(key.partition(">")[2]) == id:
                self.data[key]["Db1"] = self.data[key]["Db1"] - delay

    def compute_range_meas(self, pair=(1,2), visualize=False, owr = False):
        #TODO: Inherit this function from PostProcess?
        interv = self.time_intervals[0][pair]
        if owr and self.mult_twr:
            range = 1/2 * self._c / 1e9 \
                    * (abs(interv["tof1"]) \
                       + abs(interv["tof2"]) \
                       + 0*abs(interv["tof3"]))
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

            plt.show()

        return range