import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

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

        self.r = processed_data.r
        self.phi = processed_data.phi
        self.mean_gt_distance = processed_data.mean_gt_distance
        self.ts_data = processed_data.ts_data
        self.time_intervals = processed_data.time_intervals
        self.mean_range_meas = processed_data.mean_range_meas

        self.range_idx = 0
        self.tx1_idx = 1
        self.rx1_idx = 2
        self.tx2_idx = 3
        self.rx2_idx = 4
        self.tx3_idx = 5
        self.rx3_idx = 6
        self.Pr1_idx = 7
        self.Pr2_idx = 8

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
                x_hist, y_hist, P_hist = self._clock_filter(recording, pair, Q, R)
                
                # self._update_intervals() # TODO:

                if visualize: 
                    self._plot_kf(x_hist, y_hist, P_hist, axs[lv0,lv1])

        if visualize:
            plt.show()

    @staticmethod
    def _plot_kf(x, y, P, axs):
        axs.plot(x[0,:])

        axs.ticklabel_format(style='plain') #This is the line you need <-------

        axs.plot(y, color='red')
        
        # P_iter = P[1,1,:]
        # P_iter = P_iter.reshape(-1,)
        # axs.plot(x[0,:] + 3*np.sqrt(P_iter))
        # axs.plot(x[0,:] + -3*np.sqrt(P_iter))

    def _clock_filter(self, recording, pair, Q, R):
        # Intervals
        dt = self.time_intervals[recording][pair]["dt"]
        Ra2 = self.time_intervals[recording][pair]["Ra2"]
        Db2 = self.time_intervals[recording][pair]["Db2"]
        S1 = self.time_intervals[recording][pair]["S1"]
        S2 = self.time_intervals[recording][pair]["S2"]

        # Storage variables
        n = dt.size
        x_hist = np.zeros((2,n))
        y_hist = np.zeros((1,n))
        P_hist = np.zeros((2,2,n))

        # Initial estimate and uncertainty
        tau = 0
        skew = 0
        x = np.array([tau, skew])
        x = x.reshape(2,1)
        P = np.array(([1e18,0],[0,1e12])) # TODO: better estimate of initial uncertainty

        for lv0, dt_iter in enumerate(dt):
            Ra2_iter = Ra2[lv0]
            Db2_iter = Db2[lv0]
            S1_iter = S1[lv0]
            S2_iter = S2[lv0]
            
            if lv0>0:
                x, P = self._propagate_clocks(x, P, dt_iter, Q)

            y = self._compute_pseudomeasurement(Ra2_iter, Db2_iter, S1_iter, S2_iter)
            x, P = self._correct_clocks(x, P, y, R)

            x_hist[:,lv0] = x.reshape(2,)
            y_hist[0,lv0] = y
            P_hist[:,:,lv0] = P

        return x_hist, y_hist, P_hist

    @staticmethod
    def _compute_pseudomeasurement(Ra2, Db2, S1, S2):
        return 0.5*(S1 - Ra2/Db2 * S2)

    @staticmethod
    def _correct_clocks(x, P, y, R):
        C = np.array(([1,0]))
        C = C.reshape(1,2)

        y_check = C @ x

        S = C @ P @ C.T + R
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

    def compute_range_meas(self, id1, id2):
        """
        Only supports reverse double-sided TWR.
        TODO: support more TWR types, such as single-sided TWR.
        """
        for key in self.data:
            cond1 = (
                int(key.partition("-")[0]) == id1 and int(key.partition(">")[2]) == id2
            )
            cond2 = (
                int(key.partition("-")[0]) == id2 and int(key.partition(">")[2]) == id1
            )
            if cond1 or cond2:
                temp = self.data[key]
                if self.twr_type == 0:
                    temp = 0.5 * self._c * (temp["Ra1"] - temp["Db1"]) / 1e9
                else:
                    temp = (
                        0.5
                        * self._c
                        * (temp["Ra1"] - (temp["Ra2"] / temp["Db2"]) * temp["Db1"])
                        / 1e9
                    )
                return temp