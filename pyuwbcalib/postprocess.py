from genericpath import isfile
import numpy as np
import ast
from bagpy import bagreader
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import seaborn as sns
from scipy import stats

class PostProcess(object):
    """
    Object to generate csv files from raw range measurements to be used in UwbCalibrate.

    PARAMETERS:
    -----------
    TODO 1: Check for missing rigid bodies when checking bag files.
    """

    _c = 299702547 # speed of light
    _to_ns = 1e9 * (1.0 / 499.2e6 / 128.0) # DW time unit to nanoseconds
    def __init__(self, folder_prefix='datasets', file_prefix='recording', num_of_recordings=1,
                 tag_ids=[1,2,3], mult_twr=1, num_meas=-1):
        """
        Constructor
        """
        self._folder_prefix = folder_prefix
        self._file_prefix = file_prefix
        self.num_of_recordings = num_of_recordings
        self.tag_ids = tag_ids
        self.mult_twr = mult_twr
        self.num_meas = num_meas

        self.num_of_tags = len(tag_ids)

        self.r = {i:[] for i in range(num_of_recordings)}
        self.phi = {i:{} for i in range(num_of_recordings)}
        self.mean_gt_distance = {i:[] for i in range(num_of_recordings)}
        self.ts_data = {i:{} for i in range(num_of_recordings)}
        self.time_intervals = {i:{} for i in range(num_of_recordings)}
        self.mean_range_meas = {i:{} for i in range(num_of_recordings)}

        self._preprocess_data()

    def _preprocess_data(self):
        self._store_gt_means()
        self._store_ts_data()
        self._store_time_intervals()
        self._store_range_meas_mean()

    def _extract_gt_data(self, recording_number):
        filename = self._folder_prefix+"ros_bags/"+self._file_prefix+str(recording_number)+".bag"
        bag_data = bagreader(filename)

        r = {lv1:np.empty(0) for lv1 in self.tag_ids}
        C = {lv1:[] for lv1 in self.tag_ids}
        for lv1 in self.tag_ids:
            topic_name = '/vrpn_client_node/tripod' + str(lv1) + '/pose'
            data = bag_data.message_by_topic(topic_name)
            data_pd = pd.read_csv(data)

            r[lv1] = np.array((data_pd['pose.position.x'],
                               data_pd['pose.position.y'],
                               data_pd['pose.position.z']))

            C[lv1] = R.from_quat(np.array([data_pd['pose.orientation.x'],
                                           data_pd['pose.orientation.y'], 
                                           data_pd['pose.orientation.z'], 
                                           data_pd['pose.orientation.w']]).T)

        return r, C

    def _extract_ts_data(self,recording_number,tag_number):
        filename = self._folder_prefix+"tag" + str(tag_number) \
                   + "/"+self._file_prefix+str(recording_number) + ".txt"

        ts_data = {}

        if isfile(filename):
            file1 = open(filename, 'r')
        else:
            return ts_data
        # Lines = file1.readlines()

        for i, line in enumerate(file1):
            if self.num_meas != -1 and i >= self.num_meas:
                break
            else:
                if "tx1" in line:
                    row = ast.literal_eval(line)
                    neighbour = row["neighbour"]
                    temp = np.array([row["range"],
                                    row["tx1"]*self._to_ns,
                                    row["rx1"]*self._to_ns,
                                    row["tx2"]*self._to_ns,
                                    row["rx2"]*self._to_ns,
                                    row["tx3"]*self._to_ns,
                                    row["rx3"]*self._to_ns,
                                    row["Pr1"],
                                    row["Pr2"]])

                    if (tag_number,neighbour) in ts_data:
                        ts_data[(tag_number,neighbour)] = \
                            np.vstack((ts_data[(tag_number,neighbour)], temp))
                    else:
                        ts_data[(tag_number,neighbour)] = np.empty((0,9))

        self.range_idx = 0
        self.tx1_idx = 1
        self.rx1_idx = 2
        self.tx2_idx = 3
        self.rx2_idx = 4
        self.tx3_idx = 5
        self.rx3_idx = 6
        self.Pr1_idx = 7
        self.Pr2_idx = 8

        return ts_data

    def _retrieve_time_intervals(self, recording, pair):
        ts = self.ts_data[recording][pair]

        intervals = {}

        intervals["dt"] = ts[1:,self.tx1_idx] - ts[:-1,self.tx1_idx] # TODO: Replace with ROS timestamp
        intervals["dt"] = np.hstack(([0], intervals["dt"]))
        intervals["Ra1"] = ts[:,self.rx2_idx] - ts[:,self.tx1_idx]
        intervals["Ra2"] = ts[:,self.rx3_idx] - ts[:,self.rx2_idx]
        intervals["Db1"] = ts[:,self.tx2_idx] - ts[:,self.rx1_idx]
        intervals["Db2"] = ts[:,self.tx3_idx] - ts[:,self.tx2_idx]
        intervals["tof1"] = ts[:,self.rx1_idx] - ts[:,self.tx1_idx]
        intervals["tof2"] = ts[:,self.rx2_idx] - ts[:,self.tx2_idx]
        intervals["tof3"] = ts[:,self.rx3_idx] - ts[:,self.tx3_idx]

        return intervals

    def _unwrap_clock(self, intervals):
        # --------------------- Unwrap dt ---------------------
        # Timestamps are represented as uint32
        max_time_ns = 2**32 * self._to_ns

        intervals["dt"][intervals["dt"] < 0] \
            = intervals["dt"][intervals["dt"] < 0] + max_time_ns

        # ------- Unwrap one-clock-dependent intervals --------
        wrap_Ra1_bool = intervals["Ra1"] < 0
        wrap_Ra2_bool = intervals["Ra2"] < 0
        wrap_Db1_bool = intervals["Db1"] < 0
        wrap_Db2_bool = intervals["Db2"] < 0

        intervals["Ra1"][wrap_Ra1_bool] = intervals["Ra1"][wrap_Ra1_bool] + max_time_ns
        intervals["Ra2"][wrap_Ra2_bool] = intervals["Ra2"][wrap_Ra2_bool] + max_time_ns
        intervals["Db1"][wrap_Db1_bool] = intervals["Db1"][wrap_Db1_bool] + max_time_ns
        intervals["Db2"][wrap_Db2_bool] = intervals["Db2"][wrap_Db2_bool] + max_time_ns

        # ------- Unwrap two-clock-dependent intervals --------
        intervals["tof1"] = self._wrap_tof(intervals["tof1"], max_time_ns)
        intervals["tof2"] = self._wrap_tof(intervals["tof2"], max_time_ns)
        intervals["tof3"] = self._wrap_tof(intervals["tof3"], max_time_ns)

        return intervals

    @staticmethod
    def _wrap_tof(tof, max_time_ns):
        tof_rounded = np.round(tof / 1e6, 0) * 1e6
        tof_mode = stats.mode(tof_rounded)
        tof_rounded = tof_rounded - tof_mode.mode
        idx_wrap = tof_rounded < -1e3
        tof[idx_wrap] = tof[idx_wrap] + max_time_ns
        idx_wrap = tof_rounded > 1e3
        tof[idx_wrap] = tof[idx_wrap] - max_time_ns

        return tof

    def _calculate_mean_gt_distance(self,r):
        tag_pairs = combinations(self.tag_ids,2)

        mean_gt = {(i,j):[] for (i,j) in tag_pairs}
        for pair in mean_gt:
            i = pair[0]
            j = pair[1]

            r_i_mean = r[i].mean(axis=1)
            r_j_mean = r[j].mean(axis=1)
            mean_gt[pair] = np.linalg.norm(r_i_mean - r_j_mean)

        return mean_gt

    def _calculate_mean_range(self,range_data):
        dict = {}
        for pair in range_data:
            dict[pair] = np.mean(range_data[pair][:,0])

        return dict

    def _store_gt_means(self):
        for recording in range(self.num_of_recordings):
            r, C = self._extract_gt_data(recording+1)
            self.r[recording] = r

            for tag in C:
                self.phi[recording].update({tag:C[tag].as_rotvec()})
            
            self.mean_gt_distance[recording] = self._calculate_mean_gt_distance(r)
        
    def _store_ts_data(self):
        for recording in range(self.num_of_recordings):
            for tag in self.tag_ids[:-1]: 
                temp_dict = self._extract_ts_data(recording+1,tag)
                self.ts_data[recording].update(temp_dict)

    def _store_time_intervals(self):
        for recording in range(self.num_of_recordings):
            for pair in self.ts_data[recording]: 
                temp_dict = self._retrieve_time_intervals(recording,pair)
                temp_dict = self._unwrap_clock(temp_dict)
                self.time_intervals[recording].update({pair:temp_dict})

    def _store_range_meas_mean(self):
        for lv1 in range(self.num_of_recordings):
            self.mean_range_meas[lv1] \
                = self._calculate_mean_range(self.ts_data[lv1])

    def _stitch_time_intervals(self, pair):
        all_interv = {}
        all_interv["Ra1"] = np.empty(0)
        all_interv["Ra2"] = np.empty(0)
        all_interv["Db1"] = np.empty(0)
        all_interv["Db2"] = np.empty(0)
        all_interv["tof1"] = np.empty(0)
        all_interv["tof2"] = np.empty(0)
        all_interv["tof3"] = np.empty(0)
        for recording in range(self.num_of_recordings):
            intervals_iter = self.time_intervals[recording][pair]
            all_interv["Ra1"] = np.hstack((all_interv["Ra1"], intervals_iter["Ra1"]))
            all_interv["Ra2"] = np.hstack((all_interv["Ra2"], intervals_iter["Ra2"]))
            all_interv["Db1"] = np.hstack((all_interv["Db1"], intervals_iter["Db1"]))
            all_interv["Db2"] = np.hstack((all_interv["Db2"], intervals_iter["Db2"]))
            all_interv["tof1"] = np.hstack((all_interv["tof1"], intervals_iter["tof1"]))
            all_interv["tof2"] = np.hstack((all_interv["tof2"], intervals_iter["tof2"]))
            all_interv["tof3"] = np.hstack((all_interv["tof3"], intervals_iter["tof3"]))

        return all_interv

    def _stitch_power_and_bias(self, pair):
        all_Pr = {}
        all_Pr["Pr1"] = np.empty(0)
        all_Pr["Pr2"] = np.empty(0)
        bias = np.empty(0)
        for recording in range(self.num_of_recordings):
            ts_iter = self.ts_data[recording][pair]
            all_Pr["Pr1"] = np.hstack((all_Pr["Pr1"], ts_iter[:,self.Pr1_idx]))
            all_Pr["Pr2"] = np.hstack((all_Pr["Pr2"], ts_iter[:,self.Pr2_idx]))
            try:
                bias = np.hstack((bias, ts_iter[:,self.range_idx] - self.mean_gt_distance[recording][pair]))
            except:
                bias = np.hstack((bias, ts_iter[:,self.range_idx] - self.mean_gt_distance[recording][pair[::-1]]))

        return all_Pr, bias

    def _ss_twr_plotting(self, all_interv):
        range = 0.5 * self._c / 1e9 * \
            (all_interv["Ra1"] - all_interv["Db1"])

        fig, axs = plt.subplots(1)

        axs.plot(range)
        axs.set_ylabel("Range Measurement [m]")
        axs.set_xlabel("Measurement Number")
        axs.set_ylim([-1, 10])

    def _ds_twr_plotting(self, all_interv):
        range = 0.5 * self._c / 1e9 * \
            (all_interv["Ra1"] - (all_interv["Ra2"] / all_interv["Db2"]) * all_interv["Db1"])

        fig, axs = plt.subplots(1)

        axs.plot(range)
        axs.set_ylabel("Range Measurement [m]")
        axs.set_xlabel("Measurement Number")
        axs.set_ylim([-1, 10])

    def visualize_raw_data(self, pair=(1,2)):
        all_interv = self._stitch_time_intervals(pair)
        all_Pr, bias = self._stitch_power_and_bias(pair)

        sns.set_theme()

        # Justin's lifting function
        alpha = -82 # TODO: make a function of the measured power data
        lift = lambda x: 10**((x - alpha) /10)

        # Axes limits
        bias_l = -0.5
        bias_u = 0.5
        Pr_l = -110
        Pr_h = -80

        ####################################### RANGE MEASUREMENTS #############################################
        if self.mult_twr:
            self._ds_twr_plotting(all_interv)
        else:
            self._ss_twr_plotting(all_interv)

        ######################################### TIME INTERVALS ###############################################
        fig, axs = plt.subplots(3,3)

        col_num = 0
        row_num = 0
        for interv_str in all_interv:
            interv = all_interv[interv_str]
            axs[row_num,col_num].plot(interv)
            axs[row_num,col_num].set_ylabel(interv_str + " [ns]")
            axs[row_num,col_num].set_xlabel("Measurement Number")

            if col_num == 2:
                row_num += 1
                col_num = 0
            else:
                col_num += 1

        ########################################## POWER VS BIAS ###############################################
        fig, axs = plt.subplots(len(all_Pr))

        for i, Pr_str in enumerate(all_Pr):
            Pr = all_Pr[Pr_str]
            axs[i].scatter(lift(Pr),bias,s=1)
            axs[i].set_ylabel("Bias [m]")
            axs[i].set_xlabel("$f("+ Pr_str + ")$ [dBm]")
            axs[i].set_xlim([lift(Pr_l), lift(Pr_h)])
            axs[i].set_ylim([bias_l, bias_u])

        ############################## BIAS AND POWER vs. MEASUREMENT NUMBER ###################################
        fig, axs = plt.subplots(3)

        axs[0].plot(bias)
        axs[0].set_ylabel("Bias [m]")
        axs[0].set_ylim([bias_l, bias_u])

        for i, Pr_str in enumerate(all_Pr):
            Pr = all_Pr[Pr_str]
            axs[i+1].plot(Pr)
            axs[i+1].set_ylabel(Pr_str + " [dBm]")
            axs[i+1].set_ylim([Pr_l, Pr_h])

        axs[i+1].set_xlabel("Measurement Number")

        plt.show()