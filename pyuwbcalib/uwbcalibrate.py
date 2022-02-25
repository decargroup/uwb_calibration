import numpy as np

class UwbCalibrate(object):
    """
    Object to handle calibration for the DECAR/MRASL UWB modules.

    PARAMETERS:
    -----------
    filename_1: str
        Relative address of the file containing the timestamps of the TWR instances initiated
        by the first board (hereafter referred to as "board i").
    filename_2: str
        Relative address of the file containing the timestamps of the TWR instances initiated
        by the second board (hereafter referred to as "board j").
    board_ids: list of ints
        List of IDs of the three boards involved in the calibration procedure.
        The order is as follows:
            1) TWR initializer in filename_1 (board i).
            2) TWR initializer in filename_2 (board j).
            3) The board that never initialized a TWR instance (board k).
    average: bool
        Flag to indicate whether measurements from static intervals should be averaged out.
    static: bool
        Flag to indicate whether the calibration experiment was done with static intervals.
    """

    _c = 299702547 # speed of light

    def __init__(self, filename_1, filename_2, board_ids, average=True, static=True):
        """
        Constructor
        """
        self.files = [filename_1, filename_2]
        self.board_ids = board_ids
        self.average = average
        self.static = static

        self.data = {}

        # Boards i and j
        str_temp = str(board_ids[0]) + "->" + str(board_ids[1])
        self.data[str_temp] = self._extract_data(self.files[0], 0, 1)

        # Boards i and k
        str_temp = str(board_ids[0]) + "->" + str(board_ids[2])
        self.data[str_temp] = self._extract_data(self.files[0], 0, 2)

        # Boards j and k
        str_temp = str(board_ids[1]) + "->" + str(board_ids[2])
        self.data[str_temp] = self._extract_data(self.files[1], 1, 2)

    def _extract_data(self, filename, master_idx, slave_idx):
        """
        Reads the stored data and stores it in a dictionary for further processing.

        PARAMETERS:
        -----------
        filename: str
            Relative address of the file containing the timestamps of the TWR instances initiated
            by the board referred to here as the master.
        master_idx: int
            The index of the master board (initiator) in self.board_ids.
        slave_idx: int
            The index of the slave board in self.board_ids.

        RETURNS:
        --------
        dict: A dictionary with the following fields 
            master_id: int
                ID of the master board.
            slave_id: int
                ID of the slave board.
            gt: np.array
                Ground truth data.
            Ra1: np.array
                The delta rx2-tx1 in the master board's clock.
            Ra2: np.array
                The delta rx3-rx2 in the master board's clock.
            Db1: np.array
                The delta tx2-rx1 in the slave board's clock.
            Db2: np.array
                The delta tx3-tx2 in the slave board's clock.
        """
        dict = {"master_id": self.board_ids[master_idx], "slave_id": self.board_ids[slave_idx]}

        idx_diff = slave_idx - master_idx # this is used to determine how many columns to skip
        first_column = 2 + 11*(idx_diff-1)
        last_column = first_column + 9
        # Always read the first column to assert that the master board id is right
        columns_to_read = np.concatenate((np.array([0]), np.arange(first_column,last_column)))

        # Read the file
        my_data = np.genfromtxt(filename, delimiter=',', skip_header=1, 
                                usecols=tuple(columns_to_read))
        
        # Ensure right modules are communicating
        assert(my_data[0,0] == dict["master_id"])
        assert(my_data[0,1] == dict["slave_id"])

        gt = np.array([])
        Ra1 = np.array([])
        Ra2 = np.array([])
        Db1 = np.array([])
        Db2 = np.array([])

        # Average the static intervals if required
        if self.static is True and self.average is True:
            gap_idx = self._find_mocap_gaps(my_data[:,2])
            gap_idx = [0] + gap_idx + [np.size(my_data[:,2])] # pad the indices with the start and the end

            # Loop and average out sections of static formations
            for idx in range(len(gap_idx)-1):
                idx_beg = gap_idx[idx]
                idx_end = gap_idx[idx+1]
                
                # Ground truth
                gt = np.append(gt, np.mean(my_data[idx_beg:idx_end,3]))

                # Time stamps 
                tx1 = my_data[idx_beg:idx_end,4]
                rx1 = my_data[idx_beg:idx_end,5]
                tx2 = my_data[idx_beg:idx_end,6]
                rx2 = my_data[idx_beg:idx_end,7]
                tx3 = my_data[idx_beg:idx_end,8]
                rx3 = my_data[idx_beg:idx_end,9]

                Ra1 = np.append(Ra1, np.mean(rx2 - tx1))
                Ra2 = np.append(Ra2, np.mean(rx3 - rx2))
                Db1 = np.append(Db1, np.mean(tx2 - rx1))
                Db2 = np.append(Db2, np.mean(tx3 - tx2))

        elif self.static is True: # Otherwise, average out only the ground truth if static 
            gt = my_data[:,3]
            
            tx1 = my_data[:,4]
            rx1 = my_data[:,5]
            tx2 = my_data[:,6]
            rx2 = my_data[:,7]
            tx3 = my_data[:,8]
            rx3 = my_data[:,9]

            Ra1 = rx2 - tx1
            Ra2 = rx3 - rx2
            Db1 = tx2 - rx1
            Db2 = tx3 - tx2

        # Record ground truth and recorded time-stamps
        dict['gt'] = gt
        dict['Ra1'] = Ra1*(1e9*(1.0/499.2e6/128.0))
        dict['Ra2'] = Ra2*(1e9*(1.0/499.2e6/128.0))
        dict['Db1'] = Db1*(1e9*(1.0/499.2e6/128.0))
        dict['Db2'] = Db2*(1e9*(1.0/499.2e6/128.0))

        return dict

    def _find_mocap_gaps(self,mocap_ts):
        """
        Finds time gaps in the Mocap data to indicate a change in the static formation.

        PARAMETERS:
        -----------
        mocap_ts: np.array
            The timestamps recorded from the Mocap.

        RETURNS:
        --------
        list of ints: The indices of the measurements corresponding to the beginning of a new formation.
        """
        diff_ts = np.abs(mocap_ts[1:] - mocap_ts[:-1])
        gap = diff_ts > 10E7
        gap_idx = np.argwhere(gap)+1
        gap_idx = gap_idx.flatten()

        return gap_idx.tolist()

    def _calculate_skew_gain(self,master_idx,slave_idx):
        """
        Calculates the K parameter given by Ra2/Db2.

        PARAMETERS:
        -----------
        master_idx: int
            The index of the master board (initiator) in self.board_ids.
        slave_idx: int
            The index of the slave board in self.board_ids.

        RETURNS:
        --------
        np.array: The K values for all the measurements.
        """
        str_temp = str(self.board_ids[master_idx]) + "->" + str(self.board_ids[slave_idx])
        data = self.data[str_temp]

        Ra2 = data["Ra2"]
        Db2 = data["Db2"]

        return Ra2/Db2

    def _setup_A_matrix(self,K,master_idx,slave_idx):
        """
        Calculates the A matrix for the linear least-squares problem.

        PARAMETERS:
        -----------
        K: np.array
            The skew gain K.
        master_idx: int
            The index of the master board (initiator) in self.board_ids.
        slave_idx: int
            The index of the slave board in self.board_ids.

        RETURNS:
        --------
        2D np.array: The A matrix.
        """
        n = len(K)
        A = np.zeros((n,3))
        A[:,master_idx] += 0.5
        A[:,slave_idx] = 0.5*K
        
        return A

    def _setup_b_vector(self,K,master_idx,slave_idx):
        """
        Calculates the b vector for the linear least-squares problem.

        PARAMETERS:
        -----------
        K: np.array
            The skew gain K.
        master_idx: int
            The index of the master board (initiator) in self.board_ids.
        slave_idx: int
            The index of the slave board in self.board_ids.

        RETURNS:
        --------
        np.array: The b vector.
        """
        str_temp = str(self.board_ids[master_idx]) + "->" + str(self.board_ids[slave_idx])
        data = self.data[str_temp]

        gt = data["gt"]
        Ra1 = data["Ra1"]
        Db1 = data["Db1"]

        b = 1/self._c*gt*1e9 - 0.5*(Ra1) + 0.5*K*(Db1)

        return np.reshape(b, (len(K),1))

    def _solve_for_antenna_delays(self,A,b):
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
        return np.linalg.lstsq(A,b)

    def calibrate_antennas(self):
        """
        Calibrate the antenna delays by formulating and solving a linear least-squares problem.

        RETURNS:
        --------
        dict: Dictionary with 3 fields each for board z \in {i,j,k}
            Module i: (float)
                Antenna delay for Board i
        """
        K1 = self._calculate_skew_gain(0,1)
        A1 = self._setup_A_matrix(K1,0,1)
        b1 = self._setup_b_vector(K1,0,1)

        K2 = self._calculate_skew_gain(0,2)
        A2 = self._setup_A_matrix(K2,0,2)
        b2 = self._setup_b_vector(K2,0,2)

        K3 = self._calculate_skew_gain(1,2)
        A3 = self._setup_A_matrix(K3,1,2)
        b3 = self._setup_b_vector(K3,1,2)

        # Remove rows affected by the clock wrapping
        # TODO: Should probably do this at the beginning in case average=True
        idx_rows = np.abs(K1)>1.1
        K1 = np.delete(K1,idx_rows,0)
        A1 = np.delete(A1,idx_rows,0)
        b1 = np.delete(b1,idx_rows,0)
        idx_rows = np.abs(K1)<0.9
        A1 = np.delete(A1,idx_rows,0)
        b1 = np.delete(b1,idx_rows,0)

        idx_rows = np.abs(K2)>1.1
        K2 = np.delete(K2,idx_rows,0)
        A2 = np.delete(A2,idx_rows,0)
        b2 = np.delete(b2,idx_rows,0)
        idx_rows = np.abs(K2)<0.9
        A2 = np.delete(A2,idx_rows,0)
        b2 = np.delete(b2,idx_rows,0)

        idx_rows = np.abs(K3)>1.1
        K3 = np.delete(K3,idx_rows,0)
        A3 = np.delete(A3,idx_rows,0)
        b3 = np.delete(b3,idx_rows,0)
        idx_rows = np.abs(K3)<0.9
        A3 = np.delete(A3,idx_rows,0)
        b3 = np.delete(b3,idx_rows,0)

        idx_rows = np.abs(b1)>10000
        idx_rows = idx_rows.flatten()
        A1 = np.delete(A1,idx_rows,0)
        b1 = np.delete(b1,idx_rows,0)
        idx_rows = np.abs(b2)>10000
        idx_rows = idx_rows.flatten()
        A2 = np.delete(A2,idx_rows,0)
        b2 = np.delete(b2,idx_rows,0)
        idx_rows = np.abs(b3)>10000
        idx_rows = idx_rows.flatten()
        A3 = np.delete(A3,idx_rows,0)
        b3 = np.delete(b3,idx_rows,0)

        A = np.vstack((A1,A2,A3))
        b = np.vstack((b1,b2,b3))

        x = self._solve_for_antenna_delays(A,b)[0]
        x = x.flatten()

        print(np.linalg.norm(b))
        print(np.linalg.norm(b-A*np.array([x[0],x[1],x[2]])))

        return {"Module " + str(self.board_ids[0]): x[0],
                "Module " + str(self.board_ids[1]): x[1],
                "Module " + str(self.board_ids[2]): x[2]}

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
        """
        for key in self.data:
            if int(key.partition("-")[0]) == id:
                self.data[key]['Ra1'] = self.data[key]['Ra1'] + delay
            elif int(key.partition(">")[2]) == id:
                self.data[key]['Db1'] = self.data[key]['Db1'] - delay

    def compute_range_meas(self, id1, id2):
        """
        Only supports reverse double-sided TWR. 
        TODO: support more TWR types, such as single-sided TWR. 
        """
        for key in self.data:
            cond1 = int(key.partition("-")[0]) == id1 and int(key.partition(">")[2]) == id2
            cond2 = int(key.partition("-")[0]) == id2 and int(key.partition(">")[2]) == id1
            if cond1 or cond2:
                temp = self.data[key]
                temp = 0.5*self._c*(temp['Ra1'] - (temp['Ra2']/temp['Db2'])*temp['Db1'])/1e9
                return temp

    def plot_gt_vs_range(self, id, target):
        pass