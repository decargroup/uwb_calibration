import numpy as np
import csv
import ast
from scipy import optimize

class PostProcess(object):
    """
    Object to generate csv files from raw range measurements to be used in UwbCalibrate.

    PARAMETERS:
    -----------
    
    """

    _c = 299702547 # speed of light
    _to_ns = 1e9*(1.0/499.2e6/128.0) # DW time unit to nanoseconds
    def __init__(self, file_prefix, num_of_formations, board_ids, twr_type=0):
        """
        Constructor
        """
        self.file_prefix = file_prefix
        self.num_of_formations = num_of_formations
        self.board_ids = board_ids
        self.twr_type = twr_type

        self.mean_gt_distance = np.zeros((num_of_formations,3))
        self.ts_data = [None] * num_of_formations
        self.mean_range_meas = np.zeros((num_of_formations,3))

        self._preprocess_data()

    def _preprocess_data(self):
        self._store_gt_means()
        self._store_ts_data()
        self._store_range_meas_mean()
        self._match_gt_with_range()

    def _extract_gt_data(self, formation_number):
        filename = self.file_prefix+"/Formation"+str(formation_number)+".csv"
        my_data = np.genfromtxt(filename, delimiter=',', skip_header=8)

        r_1 = my_data[:,2:5]
        r_2 = my_data[:,5:8]
        r_3 = my_data[:,8:11]

        return [r_1, r_2, r_3]

    def _extract_ts_data(self,formation_number,agent_number):
        prefix = self.file_prefix + "/Formation"
        filename = prefix+str(formation_number)+"_Agent"\
                + str(agent_number)+"_timestamps.txt"
        
        twr_type = -1

        ts_data = {}

        # Using readlines()
        file1 = open(filename, 'r')
        Lines = file1.readlines()

        for line in Lines:
            if line[0] == "A":
                target_agent = line[6]
                twr_type = int(line[-2])
                if target_agent not in ts_data:
                    ts_data[target_agent] = np.empty((0,7))
            elif "tx1" in line and twr_type == self.twr_type:
                row = ast.literal_eval(line)
                neighbour = str(row["neighbour"])
                if self.twr_type == 0:
                    temp = np.array([row["range"],
                                     row["tx1"]*self._to_ns,
                                     row["rx1"]*self._to_ns,
                                     row["tx2"]*self._to_ns,
                                     row["rx2"]*self._to_ns,
                                     0,0])
                else:
                    temp = np.array([row["range"],
                                     row["tx1"]*self._to_ns,
                                     row["rx1"]*self._to_ns,
                                     row["tx2"]*self._to_ns,
                                     row["rx2"]*self._to_ns,
                                     row["tx3"]*self._to_ns,
                                     row["rx3"]*self._to_ns])
                ts_data[neighbour]\
                    = np.vstack((ts_data[neighbour], temp))

        return ts_data

    def _calculate_mean_gt_distance(self,r):
        d_12 = np.linalg.norm(r[0] - r[1],axis=1)
        d_12 = d_12[~np.isnan(d_12)]
        d_12 = np.mean(d_12)

        d_13 = np.linalg.norm(r[0] - r[2],axis=1)
        d_13 = d_13[~np.isnan(d_13)]
        d_13 = np.mean(d_13)

        d_23 = np.linalg.norm(r[1] - r[2],axis=1)
        d_23 = d_23[~np.isnan(d_23)]
        d_23 = np.mean(d_23)

        return [d_12, d_13, d_23]

    def _calculate_mean_range(self,range1,range2):
        id1 = self.board_ids[1]
        id2 = self.board_ids[2]
        
        r01 = range1[str(id1)][:,0]
        r01 = r01[~np.isnan(r01)]
        r01 = np.mean(r01)

        r02 = range1[str(id2)][:,0]
        r02 = r02[~np.isnan(r02)]
        r02 = np.mean(r02)

        r12 = range2[str(id2)][:,0]
        r12 = r12[~np.isnan(r12)]
        r12 = np.mean(r12)

        return np.array([r01,r02,r12])

    def _store_gt_means(self):
        for lv1 in range(self.num_of_formations):
            r = self._extract_gt_data(lv1+1)
            self.mean_gt_distance[lv1,:] = self._calculate_mean_gt_distance(r)
        
    def _store_ts_data(self):
        id0 = self.board_ids[0]
        id1 = self.board_ids[1]
        for lv1 in range(self.num_of_formations):
            temp_dict = {}
            temp_dict[str(id0)] = self._extract_ts_data(lv1+1,id0)
            temp_dict[str(id1)] = self._extract_ts_data(lv1+1,id1)
            self.ts_data[lv1] = temp_dict

    def _store_range_meas_mean(self):
        id0 = self.board_ids[0]
        id1 = self.board_ids[1]
        for lv1 in range(self.num_of_formations):
            self.mean_range_meas[lv1,:]\
                = self._calculate_mean_range(self.ts_data[lv1][str(id0)],
                                             self.ts_data[lv1][str(id1)])

    def _match_gt_with_range(self):
        for lv1 in range(self.num_of_formations):
            # Assignment problem
            gt_iter = self.mean_gt_distance[lv1,:]
            meas_iter = self.mean_range_meas[lv1,:]
            
            dist = lambda x1, x2: abs(x1 - x2)
            
            n = gt_iter.shape[0]
            t = np.dtype(dist(gt_iter[0], meas_iter[0]))
            dist_matrix = np.fromiter((dist(x1, x2) for x1 in gt_iter for x2 in meas_iter),
                                      dtype=t, count=n*n).reshape(n, n)
            row_ind, col_ind = optimize.linear_sum_assignment(dist_matrix)
            self.mean_gt_distance[lv1,:] = gt_iter[col_ind]

    def _process_multitag_data(self,neighbours,id): 
        empty = np.nan
        col0a = np.array([id])
        col0b = np.array([empty])
        
        id_idx = np.where(np.array([self.board_ids]) == int(id))[0]

        data = np.hstack(((col0a,col0b)))
        data = np.reshape(data,(1,2))

        for lv0 in range(len(neighbours)):
            col1 = np.array([neighbours[lv0]])
            col2 = np.empty((0,1))
            col3 = np.empty((0,1))
            col4_to_9 = np.empty((0,6))

            for formation in range(self.num_of_formations):
                gt_formation = self.mean_gt_distance[formation,:]
                ts_formation = self.ts_data[formation][id][neighbours[lv0]]
                col4_to_9 = np.vstack((col4_to_9,ts_formation[:,1:]))
                n = np.size(ts_formation,0)
                
                col2_formation = np.reshape(np.array([10e10*formation]*n),(n,1))
                col2 = np.vstack((col2,col2_formation))

                col3_formation = np.reshape([gt_formation[lv0+id_idx]]*n,(n,1))
                col3 = np.vstack((col3,col3_formation))

            n = np.size(col3,0)
            
            none_vector = np.reshape(np.array([empty]*(n-1)),(n-1,1))
            col1 = np.vstack((col1,none_vector))

            col10_to_11 = np.array([empty,empty]*n)
            col10_to_11 = np.reshape(col10_to_11, (n,2))

            data_new = np.hstack((col1, col2, col3, col4_to_9, col10_to_11))

            m = np.size(data,0)
            if np.size(data)>np.size(data,0):
                col_num = np.size(data,1)
            else:
                col_num = 1
            if m<n:
                none_matrix = np.array([empty]*col_num*(n-m))
                none_matrix = np.reshape(none_matrix, (n-m,col_num))
                data = np.vstack((data,none_matrix))
            elif n<m:
                none_matrix = np.array([empty]*col_num*(m-n))
                none_matrix = np.reshape(none_matrix, (m-n,11))
                data_new = np.vstack((data_new,none_matrix))
            data = np.hstack((data,data_new))

        return data

    def manually_change_gt_order(self, row, col_idx0, col_idx1):
        gt0 = self.mean_gt_distance[row,col_idx0]
        gt1 = self.mean_gt_distance[row,col_idx1]
        
        self.mean_gt_distance[row,col_idx0] = gt1
        self.mean_gt_distance[row,col_idx1] = gt0

    def setup_formatted_files(self):
        # Get list of neighbours
        for id in self.ts_data[0]: 
            neighbours = []
            for neigh in self.ts_data[0][id]:
                neighbours = neighbours + [neigh]

            # Create csv file to store data
            prefix = self.file_prefix[:-4] # save outside the 'raw' folder
            with open(prefix+"/formatted_ID"+str(id)+"_twr"+str(self.twr_type)+".csv", 'w') as f:
                # create the csv writer
                writer = csv.writer(f)

                # write a row to the csv file -----------------------------------------------
                neighbour_headers = ["target_id", "mocap_ts", "gt",
                                    "tx1", "rx1", "tx2", "rx2", "tx3", "rx3", None, None]
                row = ["self_id", None] + neighbour_headers*len(neighbours)
                writer.writerow(row)
            
                data = self._process_multitag_data(neighbours,id)
                data = data.astype(str)
                data[data=='nan'] = ''
                np.savetxt(f, data, delimiter=",", fmt="%s")

    def plot_raw_data(self):
        pass