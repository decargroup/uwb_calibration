import numpy as np
from sympy import rot_axis2


class UwbCalibrate(object):
    """
    board_ids must be in order of [filename_1 initializer, filename_2 initializer, last guy]
    these will be referred to as i j and k
    """

    _c = 299702547 # speed of light

    def __init__(self, filename_1, filename_2, board_ids, average=True, static=True):
        """
        
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
        tx1 = np.array([])
        rx1 = np.array([])
        tx2 = np.array([])
        rx2 = np.array([])
        tx3 = np.array([])
        rx3 = np.array([])

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

                # Time stamps TODO: FIX THIS, USE DELTAS
                tx1 = np.append(tx1, np.mean(my_data[idx_beg:idx_end,4]))
                rx1 = np.append(rx1, np.mean(my_data[idx_beg:idx_end,5]))
                tx2 = np.append(tx2, np.mean(my_data[idx_beg:idx_end,6]))
                rx2 = np.append(rx2, np.mean(my_data[idx_beg:idx_end,7]))
                tx3 = np.append(tx3, np.mean(my_data[idx_beg:idx_end,8]))
                rx3 = np.append(rx3, np.mean(my_data[idx_beg:idx_end,9]))

        elif self.static is True: # Otherwise, average out only the ground truth if static 
            gt = my_data[:,3]*0 + np.mean(my_data[:,3])
            
            tx1 = my_data[:,4]
            rx1 = my_data[:,5]
            tx2 = my_data[:,6]
            rx2 = my_data[:,7]
            tx3 = my_data[:,8]
            rx3 = my_data[:,9]

        # Record ground truth and recorded time-stamps
        dict['gt'] = gt
        dict['tx1'] = tx1
        dict['rx1'] = rx1
        dict['tx2'] = tx2
        dict['rx2'] = rx2
        dict['tx3'] = tx3
        dict['rx3'] = rx3

        return dict

    def _find_mocap_gaps(self,mocap_ts):
        """
        
        """
        diff_ts = np.abs(mocap_ts[1:] - mocap_ts[:-1])
        gap = diff_ts > 10E7
        gap_idx = np.argwhere(gap)+1
        gap_idx = gap_idx.flatten()

        return gap_idx.tolist()

    def _calculate_skew_gain(self,data):
        """
        
        """
        tx2 = data["tx2"]
        tx3 = data["tx3"]
        rx2 = data["rx2"]
        rx3 = data["rx3"]

        num = rx3 - rx2
        den = tx3 - tx2
        return num/den

    def _setup_A_matrix(self,K,idx_0,idx_1):
        """
        
        """
        n = len(K)
        A = np.zeros((n,3))
        A[:,idx_0] += 0.5
        A[:,idx_1] = 0.5*K
        
        return A

    def _setup_b_vector(self,K,data):
        """
        
        """
        gt = data["gt"]
        tx1 = data["tx1"]
        tx2 = data["tx2"]
        rx1 = data["rx1"]
        rx2 = data["rx2"]

        b = 1/self._c*gt*1e9 - 0.5*(rx2-tx1) - 0.5*K*(rx1-tx2)

        return np.reshape(b, (len(K),1))

    def _solve_for_antenna_delays(self,A,b):
        """
        
        """
        return np.linalg.lstsq(A,b)

    def calibrate_antennas(self):
        """
        
        """
        str_temp = str(self.board_ids[0]) + "->" + str(self.board_ids[1])
        data_temp = self.data[str_temp]
        K1 = self._calculate_skew_gain(data_temp)
        A1 = self._setup_A_matrix(K1, 0, 1)
        b1 = self._setup_b_vector(K1,data_temp)

        str_temp = str(self.board_ids[0]) + "->" + str(self.board_ids[2])
        data_temp = self.data[str_temp]
        K2 = self._calculate_skew_gain(data_temp)
        A2 = self._setup_A_matrix(K2, 0, 2)
        b2 = self._setup_b_vector(K2,data_temp)

        str_temp = str(self.board_ids[1]) + "->" + str(self.board_ids[2])
        data_temp = self.data[str_temp]
        K3 = self._calculate_skew_gain(data_temp)
        A3 = self._setup_A_matrix(K3, 1, 2)
        b3 = self._setup_b_vector(K3,data_temp)

        A = np.vstack((A1,A2,A3))
        b = np.vstack((b1,b2,b3))

        x = self._solve_for_antenna_delays(A,b)[0]

        return {"Module " + str(self.board_ids[0]): x[0],
                "Module " + str(self.board_ids[1]): x[1],
                "Module " + str(self.board_ids[2]): x[2]}