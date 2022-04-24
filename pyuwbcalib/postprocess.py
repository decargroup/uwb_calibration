from genericpath import isfile
import numpy as np
import ast
from bagpy import bagreader
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import seaborn as sns

class PostProcess(object):
    """
    Object to generate csv files from raw range measurements to be used in UwbCalibrate.

    PARAMETERS:
    -----------
    TODO 1: Check for missing rigid bodies when checking bag files.
    """

    _c = 299702547 # speed of light
    _to_ns = 1e9*(1.0/499.2e6/128.0) # DW time unit to nanoseconds
    def __init__(self, folder_prefix='datasets', file_prefix='formation', num_of_formations=1,
                 tag_ids=[1,2,3], twr_type=0, num_meas=-1):
        """
        Constructor
        """
        self.folder_prefix = folder_prefix
        self.file_prefix = file_prefix
        self.num_of_formations = num_of_formations
        self.tag_ids = tag_ids
        self.twr_type = twr_type
        self.num_meas = num_meas

        self.num_of_tags = len(tag_ids)

        self.r = {i:[] for i in range(num_of_formations)}
        self.phi = {i:{} for i in range(num_of_formations)}
        self.mean_gt_distance = {i:[] for i in range(num_of_formations)}
        self.ts_data = {i:{} for i in range(num_of_formations)}
        self.mean_range_meas = {i:{} for i in range(num_of_formations)}

        self._preprocess_data()

    def _preprocess_data(self):
        self._store_gt_means()
        self._store_ts_data()
        self._store_range_meas_mean()

    def _extract_gt_data(self, formation_number):
        filename = self.folder_prefix+"ros_bags/"+self.file_prefix+str(formation_number)+".bag"
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

    def _extract_ts_data(self,formation_number,tag_number):
        filename = self.folder_prefix+"tag" + str(tag_number) \
                   + "/"+self.file_prefix+str(formation_number) + ".txt"

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
        for formation in range(self.num_of_formations):
            r, C = self._extract_gt_data(formation+1)
            self.r[formation] = r

            for tag in C:
                self.phi[formation].update({tag:C[tag].as_rotvec()})
            
            self.mean_gt_distance[formation] = self._calculate_mean_gt_distance(r)
        
    def _store_ts_data(self):
        for formation in range(self.num_of_formations):
            for tag in self.tag_ids[:-1]: 
                temp_dict = self._extract_ts_data(formation+1,tag)
                self.ts_data[formation].update(temp_dict)

    def _store_range_meas_mean(self):
        for lv1 in range(self.num_of_formations):
            self.mean_range_meas[lv1] \
                = self._calculate_mean_range(self.ts_data[lv1])

    def visualize_raw_data(self,pair=(1,2)):
        Pr1 = np.empty(0)
        Pr2 = np.empty(0)
        bias = np.empty(0)
        for formation in range(self.num_of_formations):
            data = self.ts_data[formation][pair]
            Pr1 = np.hstack((Pr1, data[:,self.Pr1_idx]))
            Pr2 = np.hstack((Pr2, data[:,self.Pr2_idx]))
            try:
                bias = np.hstack((bias, data[:,self.range_idx] - self.mean_gt_distance[formation][pair]))
            except:
                bias = np.hstack((bias, data[:,self.range_idx] - self.mean_gt_distance[formation][pair[::-1]]))

        sns.set_theme()

        # Justin's lifting function
        lift = lambda x: 10**((x + 82) /10)

        # Axes limits
        bias_l = -0.5
        bias_u = 0.5
        Pr_l = -110
        Pr_h = -80

        ########################################## POWER VS BIAS ###############################################
        fig, axs = plt.subplots(2)

        axs[0].scatter(lift(Pr1),bias,s=1)
        axs[0].set_ylabel("Bias [m]")
        axs[0].set_xlabel("$f(P_r)$ at Initiator [dBm]")
        axs[0].set_xlim([lift(Pr_l), lift(Pr_h)])
        axs[0].set_ylim([bias_l, bias_u])

        axs[1].scatter(lift(Pr2),bias,s=1)
        axs[1].set_ylabel("Bias [m]")
        axs[1].set_xlabel("$f(P_r)$ at Target [dBm]")
        axs[1].set_xlim([lift(Pr_l), lift(Pr_h)])
        axs[1].set_ylim([bias_l, bias_u])

        ############################## BIAS AND POWER vs. MEASUREMENT NUMBER ###################################
        fig, axs = plt.subplots(3)

        axs[0].plot(bias)
        axs[0].set_ylabel("Bias [m]")
        axs[0].set_ylim([bias_l, bias_u])

        axs[1].plot(Pr1)
        axs[1].set_ylabel("Reception Power at Initiator [dBm]")
        axs[1].set_ylim([Pr_l, Pr_h])

        axs[2].plot(Pr2)
        axs[2].set_ylabel("Reception Power at Target [dBm]")
        axs[2].set_ylim([Pr_l, Pr_h])
        axs[2].set_xlabel("Measurement Number")

        plt.show()