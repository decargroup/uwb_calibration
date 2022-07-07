import numpy as np
from bagpy import bagreader
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

class PostProcess(object):
    """
    Object to process ROS bags and extract the timestamps and data for UwbCalibrate.

    PARAMETERS:
    -----------
    file_path:
        Path of ROS bag relative to Python's working directory.
    tag_ids: list[int]
        List of tag IDs, in the order of tag attached to [tripod1, tripod2, tripod3].
        Default: [1,2,3].
    mult_twr: bool
        Multiplicative protocol using double-sided TWR. Default: True.
    num_meas: float
        Number of measurements to process. -1 means no cap. Default: -1.

    TODO 1: Check for missing rigid bodies when checking bag files.
         2: Add tests.
         3: Should I do all this in Pandas?
    """

    _c = 299702547 # speed of light
    _to_ns = 1e9 * (1.0 / 499.2e6 / 128.0) # DW time unit to nanoseconds

    def __init__(self, file_path, tag_ids, moment_arms, mult_twr=True, num_meas=-1, ranging_with_self=False):
        """
        Constructor.
        """
        self.file_path = file_path

        self.tag_ids = tag_ids
        self.moment_arms = moment_arms
        self.mult_twr = mult_twr
        self.num_meas = num_meas
        self.ranging_with_self = ranging_with_self

        self.num_of_tags = len(tag_ids)

        self.r = [] # stores ground truth positions, per robot
        self.phi = {} # stores ground truth rotation vectors, per robot
        self._gt_distance = [] # stores ground truth distance between pairs
        self.ts_data = {} # stores measured raw timestamps and power during ranging, per pair
        self.time_intervals = {} # stores time intervals and power data, per pair

        # Read and store the ROS bag using bagpy.
        self.bag_data = bagreader(file_path)

        self._preprocess_data()

    def _preprocess_data(self):
        """
        Preprocess data. This is the main function.
        """
        self._store_gt_distance()
        self._store_ts_data()
        self._store_time_intervals()
        self._interpolate_gt_to_uwb() 

    def _store_gt_distance(self):
        """
        Stores the history of ground truth distance per pair.
        """
        t_sec, t_nsec, r, rot = self._extract_gt_data()
        
        # Store ground-truth position of every tag
        self.r = r
        self.rot = rot

        for machine in t_sec:
            # Unwrap the clock
            t_nsec[machine] = self._unwrap_gt(t_sec[machine], t_nsec[machine], 1e9)
        
        # Store the ground-truth distance between pairs 
        self._gt_distance = self._calculate_gt_distance(t_nsec)

    def _extract_gt_data(self):
        """
        Extract the ground truth data from the ROS bag, as recorded by the Mocap system and 
        streamed using VRPN.

        RETURNS:
        --------
        t_sec: dict[int: np.array(n,)]
            dict with the tag IDs as the keys. Contains recorded timestamps in seconds.
        t_nsec: dict[int: np.array(n,)]
            dict with the tag IDs as the keys. Contains recorded timestamps in nanoseconds.
        r: dict[int: np.array(n,3)]
            dict with the tag IDs as the keys. Contains recorded ground-truth position.
        q: dict[int: np.array(n,3)]
            dict with the tag IDs as the keys. Contains recorded ground-truth quaternion.
        """

        # Setting up the storage variables as dicts
        t_sec = {lv0:np.empty(0) for lv0 in self.tag_ids.keys()}
        t_nsec = {lv0:np.empty(0) for lv0 in self.tag_ids.keys()}
        r = {lv0:np.empty(0) for lv0 in self.tag_ids.keys()}
        rot = {lv0:[] for lv0 in self.tag_ids.keys()}

        # Iterating tag by tag. 
        for machine in self.tag_ids: 
            # TODO: needs a big overhaul. We now have pose of drone not module, and a moment arm to module.
            # Extracting the data streamed through VRPN.
            topic = '/'+machine+'/vrpn_client_node/'+machine+'/pose'
            data = self.bag_data.message_by_topic(topic)
            data_pd = pd.read_csv(data)

            # Timestamps.
            t_sec[machine] = np.array(data_pd['header.stamp.secs'])
            t_nsec[machine] = np.array(data_pd['header.stamp.nsecs'])
            
            # Pose.
            r[machine] = np.array((data_pd['pose.position.x'],
                               data_pd['pose.position.y'],
                               data_pd['pose.position.z']))
            rot[machine] = R.from_quat(np.array([data_pd['pose.orientation.x'],
                                           data_pd['pose.orientation.y'], 
                                           data_pd['pose.orientation.z'], 
                                           data_pd['pose.orientation.w']]).T)

        return t_sec, t_nsec, r, rot

    def _calculate_gt_distance(self, t):
        """
        Compute the ground-truth distance between pairs of tags from their ground-truth position.

        PARAMETERS:
        -----------
        t: dict[int: np.array(n,)]
            dict with the tag IDs as the keys. Contains recorded timestamps in nanoseconds.
        r: dict[int: np.array(n,3)]
            dict with the tag IDs as the keys. Contains recorded ground-truth position.

        RETURNS:
        --------
        gt: dict[tuple: np.array(n,1)]
            dict with the tag ID pairs as the keys. Contains computed ground-truth distances.
        """
        # All possible combinations of tags. Order does not matter.
        tags = sum(list(self.tag_ids.values()),[])
            
        tag_pairs_all = combinations(tags,2)
        if self.ranging_with_self:
            self.tag_pairs_set = tag_pairs_all
        else:
            self.tag_pairs_set = [i for i in tag_pairs_all if [i[0],i[1]] not in self.tag_ids.values()]

        gt = {(i,j):{"t":[],"dist":[]} for (i,j) in self.tag_pairs_set}
        for pair in gt:
            i = pair[0]
            j = pair[1]
            machine_i = [x for x in self.tag_ids.keys() if i in self.tag_ids[x]][0]
            machine_j = [x for x in self.tag_ids.keys() if j in self.tag_ids[x]][0]

            # Interpolate the measurements of Tag j to the timestamps of Tag i.
            r_j_interp = self._interpolate(self.r[machine_j], t[machine_j], t[machine_i])
            q_j_interp = self._interpolate(self.rot[machine_j].as_quat().T, t[machine_j], t[machine_i]).T
            rot_j_interp = R.from_quat(np.array([q_j_interp[:,0],
                                                 q_j_interp[:,1], 
                                                 q_j_interp[:,2], 
                                                 q_j_interp[:,3]]).T)
            gt[pair]["t"] = t[machine_i]

            # Find moment arms
            idx_i = self.tag_ids[machine_i].index(i)
            idx_j = self.tag_ids[machine_j].index(j)
            arm_i = self.moment_arms[machine_i][idx_i]
            arm_j = self.moment_arms[machine_j][idx_j]

            # Compute the distance between the tags.
            num_of_meas = t[machine_i].size
            gt[pair]["dist"] = np.zeros(num_of_meas,)
            C_ai = self.rot[machine_i].as_dcm()
            C_aj = rot_j_interp.as_dcm()
            for k in range(num_of_meas):
                gt[pair]["dist"][k] = np.linalg.norm(C_ai[k] @ arm_i
                                                     + self.r[machine_i][:,k]
                                                     - r_j_interp[:,k]
                                                     - C_aj[k] @ arm_j)

        return gt

    def _store_ts_data(self):
        """
        Stores the history of raw timestamp and power measurements, per pair.
        """
        temp_dict = self._extract_ts_data()
        self.ts_data.update(temp_dict)

    def _extract_ts_data(self):
        """
        Extract the raw data from the ROS bag, as recorded by the devices connected to the
        UWB tags.

        RETURNS:
        --------
        ts_data: dict[tuple: np.array(n,1)]
            dict with the tag ID pairs as the keys. Contains measured raw timestamps and power.
        """

        # Setting up the storage variable as a dict
        ts_data = {}

        # Iterating device by device.
        for machine in self.tag_ids.keys():
            # Extracting the data recorded by the device.
            topic = '/' + machine + '/uwb/range'
            data = self.bag_data.message_by_topic(topic)
            data_pd = pd.read_csv(data)

            # Iterating through the data row by row. # TODO: replace this for loop
            for idx, row in data_pd.iterrows():
                if self.num_meas != -1 and idx >= self.num_meas:
                    break

                initiator_id = row["from_id"]
                target_id = row["to_id"]
                pair = (initiator_id, target_id)

                # Ignore unexpected measurements
                if (pair not in self.tag_pairs_set) and (pair[::-1] not in self.tag_pairs_set):
                    continue

                # Ignore measurement-at-target
                if (initiator_id not in self.tag_ids[machine]):
                    continue

                temp = np.array([row["header.stamp.nsecs"],
                                 row["header.stamp.secs"],
                                 row["range"],
                                 row["tx1"]*self._to_ns,
                                 row["rx1"]*self._to_ns,
                                 row["tx2"]*self._to_ns,
                                 row["rx2"]*self._to_ns,
                                 row["tx3"]*self._to_ns,
                                 row["rx3"]*self._to_ns,
                                 row["fpp1"],
                                 row["fpp2"]])

                # Initialize this pair if not already part of the dict
                if pair not in ts_data:
                    ts_data[pair] = np.empty((0,11))

                ts_data[pair] = np.vstack((ts_data[pair], temp))

        # TODO: should make ts_data of the same form as time_intervals (i.e., a nested dict)
        self.range_idx = 2
        self.tx1_idx = 3
        self.rx1_idx = 4
        self.tx2_idx = 5
        self.rx2_idx = 6
        self.tx3_idx = 7
        self.rx3_idx = 8
        self.Pr1_idx = 9
        self.Pr2_idx = 10

        self.tag_pairs = list(ts_data.keys())

        return ts_data

    def _store_time_intervals(self):
        """
        Stores the history of time intervals and power measurements, per pair.
        """
        self._unwrap_all_clocks()
        for pair in self.tag_pairs: 
            temp_dict = self._retrieve_time_intervals(pair)
            self.time_intervals.update({pair:temp_dict})

    def _retrieve_time_intervals(self, pair):
        """
        Compute the time intervals from the recorded timestamps.

        PARAMETERS:
        -----------
        pair: tuple
            A tuple with the (inititating tag, target_tag).

        RETURNS:
        --------
        intervals: dict[tuple: np.array(n,1)]
            dict with the tag ID pairs as the keys. Contains computed time intervals and power.
        """
        ts = self.ts_data[pair]

        intervals = {}

        intervals["t"] = ts[:,0]
        intervals["dt"] = ts[1:,0] - ts[:-1,0]
        intervals["dt"] = np.hstack(([0], intervals["dt"]))
        intervals["Ra1"] = ts[:,self.rx2_idx] - ts[:,self.tx1_idx]
        intervals["Ra2"] = ts[:,self.rx3_idx] - ts[:,self.rx2_idx]
        intervals["Db1"] = ts[:,self.tx2_idx] - ts[:,self.rx1_idx]
        intervals["Db2"] = ts[:,self.tx3_idx] - ts[:,self.tx2_idx]
        intervals["tof1"] = ts[:,self.rx1_idx] - ts[:,self.tx1_idx]
        intervals["tof2"] = ts[:,self.rx2_idx] - ts[:,self.tx2_idx]
        intervals["tof3"] = ts[:,self.rx3_idx] - ts[:,self.tx3_idx]
        intervals["S1"] = ts[:,self.rx2_idx] + ts[:,self.tx1_idx]
        intervals["S2"] = ts[:,self.tx2_idx] + ts[:,self.rx1_idx]

        return intervals

    ### ------------------------ INTERPOLATION METHODS ------------------------ ###
    def _interpolate_gt_to_uwb(self):
        """
        Interpolate the computed ground truth distance to the timestamps where measurements 
        were recorded.
        """
        for pair in self.tag_pairs:
            t_new = self.time_intervals[pair]["t"]
            try:
                r = self._gt_distance[pair]["dist"]
                t_old = self._gt_distance[pair]["t"]
            except:
                r = self._gt_distance[pair[::-1]]["dist"]
                t_old = self._gt_distance[pair[::-1]]["t"]
            
            f = interp1d(t_old, r, kind='linear', fill_value='extrapolate')

            self.time_intervals[pair].update({'r_gt': f(t_new)})
            
            # try:
            #     self._gt_distance[pair]['t'] = t_new
            #     self._gt_distance[pair]['dist'] = f(t_new)
            # except:
            #     self._gt_distance[pair[::-1]]['t'] = t_new
            #     self._gt_distance[pair[::-1]]['dist'] = f(t_new)
            
    @staticmethod
    def _interpolate(x, t_old, t_new):
        """
        Interpolate ground truth position.

        PARAMETERS:
        -----------
        r: np.array(n,3)
            Recorded ground-truth position at t_old.
        t_old: np.array(n,)
            The timestamps of the recorded ground-truth position.
        t_new: np.array(n,)
            The new keypoints for interpolation.

        RETURNS:
        --------
        f(t_new): np.array(n,3)
            Recorded ground-truth position interpolated at t_new.
        """
        
        f = interp1d(t_old, x, kind='linear', fill_value='extrapolate')

        return f(t_new)
    ### ------------------------------------------------------------------------ ###


    ### -------------------------- UNWRAPPING METHODS -------------------------- ###
    def _unwrap_all_clocks(self):
        """
        Unwrap the clock for all the time-stamp measurements.
        """
        ### --- Unwrap dt --- ###
        # Timestamps are represented as uint32
        max_time_ns = 2**32 * self._to_ns

        ### --- Unwrap time-stamps --- ###
        for pair in self.tag_pairs: 
            iter_rx2 = 0
            iter_tx2 = 0
            iter_tx3 = 0
            iter_rx3 = 0
            # Check if a clock wrap occured at the first measurement, and unwrap
            if self.ts_data[pair][:,self.rx2_idx][0] < self.ts_data[pair][:,self.tx1_idx][0]:
                self.ts_data[pair][:,self.rx2_idx][0] + max_time_ns
                iter_rx2 = 1
                iter_rx3 = 1

            if self.ts_data[pair][:,self.tx2_idx][0] < self.ts_data[pair][:,self.rx1_idx][0]:
                self.ts_data[pair][:,self.tx2_idx][0] + max_time_ns
                iter_tx2 = 1
                iter_tx3 = 1

            if self.ts_data[pair][:,self.tx3_idx][0] < self.ts_data[pair][:,self.tx2_idx][0]:
                self.ts_data[pair][:,self.tx3_idx][0] + max_time_ns
                iter_tx3 = 1

            if self.ts_data[pair][:,self.rx3_idx][0] < self.ts_data[pair][:,self.rx2_idx][0]:
                self.ts_data[pair][:,self.rx3_idx][0] + max_time_ns
                iter_rx3 = 1

            # Individual unwraps
            self.ts_data[pair][:,0] \
                = self._unwrap_gt(self.ts_data[pair][:,1], self.ts_data[pair][:,0], 1e9)
            
            self.ts_data[pair][:,self.tx1_idx] \
                = self._unwrap(self.ts_data[pair][:,self.tx1_idx], max_time_ns)

            self.ts_data[pair][:,self.rx1_idx] \
                = self._unwrap(self.ts_data[pair][:,self.rx1_idx], max_time_ns)

            self.ts_data[pair][:,self.tx2_idx] \
                = self._unwrap(self.ts_data[pair][:,self.tx2_idx], max_time_ns, iter=iter_tx2)

            self.ts_data[pair][:,self.rx2_idx] \
                = self._unwrap(self.ts_data[pair][:,self.rx2_idx], max_time_ns, iter=iter_rx2)

            self.ts_data[pair][:,self.tx3_idx] \
                = self._unwrap(self.ts_data[pair][:,self.tx3_idx], max_time_ns, iter=iter_tx3)

            self.ts_data[pair][:,self.rx3_idx] \
                = self._unwrap(self.ts_data[pair][:,self.rx3_idx], max_time_ns, iter=iter_rx3)

    @staticmethod
    def _unwrap(data, max, iter=0):
        """
        Unwraps data. 

        PARAMETERS:
        -----------
        data: np.array(n,)
            Data to be unwrapped.
        max: float
            Max value of the data where the wrapping occurs.

        RETURNS:
        --------
        data: np.array(n,)
            Unwrapped data.
        """
        temp = data[1:] - data[:-1]
        idx = np.concatenate([np.array([0]), temp < 0])

        # iter = 0
        for lv0, _ in enumerate(data):
            if idx[lv0]:
                iter += 1
            data[lv0] += iter*max    

        return data

    @staticmethod
    def _unwrap_gt(t_sec, t_nsec, max):
        """
        Unwraps ground truth timestamps.
        
        PARAMETERS:
        -----------
        t_sec: np.array(n,)
            Timestamps in seconds.
        t_nsec: np.array(n,)
            Timestamps in nanoseconds.
        max: float
            Max value of the data where the wrapping occurs.

        RETURNS:
        --------
        t_nsec: np.array(n,)
            Unwrapped timestamps in nanoseconds.
        """
        iter = 0
        for lv0 in range(len(t_nsec)-1):
            if t_sec[lv0+1] - t_sec[lv0] > 0:
                iter += t_sec[lv0+1] - t_sec[lv0]
            t_nsec[lv0+1] += iter*max    

        return t_nsec
    ### ------------------------------------------------------------------------ ###


    ### --------------------------- PLOTTING METHODS --------------------------- ###
    def _ss_twr_plotting(self, pair):
        """
        Plot the single-sided TWR range measurements.

        PARAMETERS:
        -----------
        pair: tuple
            A tuple with the (inititating tag, target_tag).
        """
        range = 0.5 * self._c / 1e9 * \
            (self.time_intervals[pair]["Ra1"] - self.time_intervals[pair]["Db1"])

        fig, axs = plt.subplots(1)

        axs.plot(self.time_intervals[pair]["t"]/1e9, range, label='Range Measurements')
        axs.scatter(self.time_intervals[pair]["t"]/1e9, 
                    self.time_intervals[pair]["r_gt"], 
                    s=1,
                    label='Ground Truth')
        
        axs.set_ylabel("Distance [m]")
        axs.set_xlabel("t [s]")
        axs.set_ylim([-1, 5])

        axs.legend()

    def _ds_twr_plotting(self, pair):
        """
        Plot the double-sided TWR range measurements.

        PARAMETERS:
        -----------
        pair: tuple
            A tuple with the (inititating tag, target_tag).
        """
        range = 0.5 * self._c / 1e9 * \
            (self.time_intervals[pair]["Ra1"] - (self.time_intervals[pair]["Ra2"] \
                / self.time_intervals[pair]["Db2"]) * self.time_intervals[pair]["Db1"])

        fig, axs = plt.subplots(1)

        axs.plot(self.time_intervals[pair]["t"]/1e9, range, label='Range Measurements')
        axs.scatter(self.time_intervals[pair]["t"]/1e9, \
                    self.time_intervals[pair]["r_gt"], s=1, \
                    label='Ground Truth', color='r')

        axs.set_ylabel("Distance [m]")
        axs.set_xlabel("t [s]")
        axs.set_ylim([-1, 5])

        axs.legend()

    @staticmethod
    def lift(x, alpha=-82):
        """
        Lifting function for better visualization and calibration. 
        Based on Cano, J., Pages, G., Chaumette, E., & Le Ny, J. (2022). Clock 
                 and Power-Induced Bias Correction for UWB Time-of-Flight Measurements.
                 IEEE Robotics and Automation Letters, 7(2), 2431-2438. 
                 https://doi.org/10.1109/LRA.2022.3143202

        PARAMETERS:
        -----------
        x: np.array(n,1)
            Input to lifting function. Received Power in dBm in this context.
        alpha: scalar
            Centering parameter. Default: -82 dBm.

        RETURNS:
        --------
        intervals: dict[tuple: np.array(n,1)]
            dict with the tag ID pairs as the keys. Contains computed time intervals and power.
        """
        return 10**((x - alpha) /10)

    def visualize_raw_data(self, pair=(1,2), alpha=-82):
        """
        Generates multiple plots to visualize the raw data. 
        
        PARAMETERS:
        -----------
        pair: tuple
            A tuple with the (inititating tag, target_tag).
        alpha: scalar
            Centering parameter. Default: -82 dBm.
        """
        interv = self.time_intervals[pair]
        Pr = {}
        Pr["Pr1"] = self.ts_data[pair][:,self.Pr1_idx]
        Pr["Pr2"] = self.ts_data[pair][:,self.Pr2_idx]
        bias = self.ts_data[pair][:,self.range_idx] - interv["r_gt"]

        # Axes limits
        bias_l = -0.5
        bias_u = 0.5
        Pr_l = -110
        Pr_h = -80

        ### --- RANGE MEASUREMENTS --- ###
        if self.mult_twr:
            self._ds_twr_plotting(pair)
        else:
            self._ss_twr_plotting(pair)

        ### --- TIME INTERVALS --- ###
        fig, axs = plt.subplots(3,3)

        col_num = 0
        row_num = 0
        for interv_str in interv:
            if interv_str == "dt" or interv_str == "t" or interv_str == "r_gt":
                continue
            interv_data = interv[interv_str]
            axs[row_num,col_num].plot(interv["t"]/1e9,interv_data)
            axs[row_num,col_num].set_ylabel(interv_str + " [ns]")
            axs[row_num,col_num].set_xlabel("t [s]")

            if col_num == 2:
                row_num += 1
                col_num = 0
            else:
                col_num += 1

        ### --- POWER VS BIAS --- ###
        fig, axs = plt.subplots(len(Pr))

        for i, Pr_str in enumerate(Pr):
            Pr_iter = Pr[Pr_str]
            axs[i].scatter(self.lift(Pr_iter),bias,s=1)
            axs[i].set_ylabel("Bias [m]")
            axs[i].set_xlabel("$f("+ Pr_str + ")$ [dBm]")
            # axs[i].set_xlim([self.lift(Pr_l), self.lift(Pr_h)])
            # axs[i].set_ylim([bias_l, bias_u])

        ### --- BIAS AND POWER vs. TIME --- ###
        fig, axs = plt.subplots(3)

        axs[0].plot(interv["t"]/1e9,bias)
        axs[0].set_ylabel("Bias [m]")
        # axs[0].set_ylim([bias_l, bias_u])

        for i, Pr_str in enumerate(Pr):
            Pr_iter = Pr[Pr_str]
            axs[i+1].plot(interv["t"]/1e9,Pr_iter)
            axs[i+1].set_ylabel(Pr_str + " [dBm]")
            axs[i+1].set_ylim([Pr_l, Pr_h])

        axs[i+1].set_xlabel("t [s]")
    ### ------------------------------------------------------------------------ ###