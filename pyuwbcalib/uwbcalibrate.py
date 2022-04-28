import numpy as np

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
        [attr for attr in dir(processed_data) if not attr.startswith('__')]
        for attr in dir(processed_data):
            if not attr.startswith('_'):
                attr_value = getattr(processed_data, attr)
                setattr(self, attr, attr_value) # TODO: should only take required attributes

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

    def filter_data(self, R, Q):
        pass

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

    def plot_gt_vs_range(self, id, target):
        pass
