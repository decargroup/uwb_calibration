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
    """

    _c = 299702547 # speed of light
    _to_ns = 1e9 * (1.0 / 499.2e6 / 128.0) # DW time unit to nanoseconds

    def __init__(self, file_path, tag_ids=[1,2,3], mult_twr=True, num_meas=-1):
        """
        Constructor.
        """
        self.file_path = file_path

        self.tag_ids = tag_ids
        self.mult_twr = mult_twr
        self.num_meas = num_meas

        self.num_of_tags = len(tag_ids)

        self.r = [] # stores ground truth positions
        self.phi = {} # stores ground truth rotation vectors
        self._gt_distance = [] # stores ground truth distance between pairs
        self.ts_data = {} # stores measured raw timestamps and power during ranging, per pair
        self.time_intervals = {} # stores time intervals and power data, per pair

        # Read and store the ROS bag using bagpy.
        self.bag_data = bagreader(file_path)

        self._preprocess_data()

    def _preprocess_data(self):
        """
        Preprocess data. This is the main method.
        """
        self._detect_setup()

        self._store_gt_distance()
        self._store_ts_data()
        self._store_time_intervals()
        self._interpolate_gt_to_uwb() 

    def _detect_setup(self):
        """
        Detects the devices that recorded to the bag, and the tag IDs associated with each device.
        """

        # Get all topics in the ROS bag.
        topics = self.bag_data.topics
        
        ###------- Generate a list of ranging devices -------###
        all_devices = []
        for topic in topics:
            if topic[-5:] == "range":
                device = topic.split('uwb/range')[0]
                all_devices.append(device)

        ###------- Get the tag(s) attached to each device -------###
        self.device_tags = {device:[] for device in all_devices}
        for device in all_devices:
            # Get the ranging data associated with this device
            topic = device + 'uwb/range'
            data = self.bag_data.message_by_topic(topic)
            data_pd = pd.read_csv(data)

            # Find all the tags that initiated while connected to this device
            id = data_pd['from_id']
            self.device_tags[device] = list(set(id))

    def _store_gt_distance(self):
        """
        Stores the history of ground truth distance per pair.
        """
        t_sec, t_nsec, r, q = self._extract_gt_data()
        
        # Store ground-truth position of every tag
        self.r = r

        for tag in t_sec:
            # Unwrap the clock
            t_nsec[tag] = self._unwrap_gt(t_sec[tag], t_nsec[tag], 1e9)

            # Store ground truth rotation vector of every tag
            self.phi.update({tag:q[tag].as_rotvec()})
        
        # Store the ground-truth distance between pairs 
        self._gt_distance = self._calculate_gt_distance(t_nsec, r)

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
        t_sec = {lv0:np.empty(0) for lv0 in self.tag_ids}
        t_nsec = {lv0:np.empty(0) for lv0 in self.tag_ids}
        r = {lv0:np.empty(0) for lv0 in self.tag_ids}
        q = {lv0:[] for lv0 in self.tag_ids}

        # Iterating tag by tag. 
        for lv0 in range(len(self.tag_ids)): 
            # Extracting the data streamed through VRPN.
            topic = '/vrpn_client_node/tripod' + str(lv0+1) + '/pose'
            data = self.bag_data.message_by_topic(topic)
            data_pd = pd.read_csv(data)

            # Tag ID.
            id = self.tag_ids[lv0]

            # Timestamps.
            t_sec[id] = np.array(data_pd['header.stamp.secs'])
            t_nsec[id] = np.array(data_pd['header.stamp.nsecs'])
            
            # Pose.
            r[id] = np.array((data_pd['pose.position.x'],
                               data_pd['pose.position.y'],
                               data_pd['pose.position.z']))
            q[id] = R.from_quat(np.array([data_pd['pose.orientation.x'],
                                           data_pd['pose.orientation.y'], 
                                           data_pd['pose.orientation.z'], 
                                           data_pd['pose.orientation.w']]).T)

        return t_sec, t_nsec, r, q

    def _calculate_gt_distance(self, t, r):
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
        tag_pairs = combinations(self.tag_ids,2)

        gt = {(i,j):{"t":[],"dist":[]} for (i,j) in tag_pairs}
        for pair in gt:
            i = pair[0]
            j = pair[1]

            # Interpolate the measurements of Tag j to the timestamps of Tag i.
            r_j_interp = self._interpolate_position(r[j], t[j], t[i])

            # Compute the distance between the tags.
            gt[pair]["t"] = t[i]
            gt[pair]["dist"] = np.linalg.norm(r[i] - r_j_interp, axis=0)

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
        for device in self.device_tags:
            # Extracting the data recorded by the device.
            topic = device + 'uwb/range'
            data = self.bag_data.message_by_topic(topic)
            data_pd = pd.read_csv(data)

            # Iterating through the data row by row. # TODO: replace this for loop
            for idx, row in data_pd.iterrows():
                if self.num_meas != -1 and idx >= self.num_meas:
                    break

                initiator_id = row["from_id"]
                target_id = row["to_id"]
                temp = np.array([row["header.stamp.nsecs"],
                                 row["range"],
                                 row["tx1"]*self._to_ns,
                                 row["rx1"]*self._to_ns,
                                 row["tx2"]*self._to_ns,
                                 row["rx2"]*self._to_ns,
                                 row["tx3"]*self._to_ns,
                                 row["rx3"]*self._to_ns,
                                 row["power1"],
                                 row["power2"]])

                # Initialize this pair if not already part of the dict
                if (initiator_id, target_id) not in ts_data:
                    ts_data[(initiator_id,target_id)] = np.empty((0,10))

                ts_data[(initiator_id,target_id)] = \
                            np.vstack((ts_data[(initiator_id,target_id)], temp))

        # TODO: should make ts_data of the same form as time_intervals (i.e., a nested dict)
        self.range_idx = 1
        self.tx1_idx = 2
        self.rx1_idx = 3
        self.tx2_idx = 4
        self.rx2_idx = 5
        self.tx3_idx = 6
        self.rx3_idx = 7
        self.Pr1_idx = 8
        self.Pr2_idx = 9

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

    @staticmethod
    def _interpolate_position(r, t_old, t_new):
        
        f = interp1d(t_old, r, kind='linear', fill_value='extrapolate')

        return f(t_new)
    ### ------------------------------------------------------------------------ ###


    ### -------------------------- UNWRAPPING METHODS -------------------------- ###
    def _unwrap_all_clocks(self):
        # --------------------- Unwrap dt ---------------------
        # Timestamps are represented as uint32
        max_time_ns = 2**32 * self._to_ns

        # ------- Unwrap time-stamps --------
        for pair in self.tag_pairs: 
            # Check if a clock wrap occured at the first measurement, and unwrap
            if self.ts_data[pair][:,self.rx2_idx][0] < self.ts_data[pair][:,self.tx1_idx][0]:
                self.ts_data[pair][:,self.rx2_idx][0] + max_time_ns

            if self.ts_data[pair][:,self.tx2_idx][0] < self.ts_data[pair][:,self.rx1_idx][0]:
                self.ts_data[pair][:,self.tx2_idx][0] + max_time_ns

            if self.ts_data[pair][:,self.tx3_idx][0] < self.ts_data[pair][:,self.tx2_idx][0]:
                self.ts_data[pair][:,self.tx3_idx][0] + max_time_ns

            if self.ts_data[pair][:,self.rx3_idx][0] < self.ts_data[pair][:,self.rx2_idx][0]:
                self.ts_data[pair][:,self.rx3_idx][0] + max_time_ns

            # Individual unwraps
            self.ts_data[pair][:,0] \
                = self._unwrap(self.ts_data[pair][:,0], 1e9)
            
            self.ts_data[pair][:,self.tx1_idx] \
                = self._unwrap(self.ts_data[pair][:,self.tx1_idx], max_time_ns)

            self.ts_data[pair][:,self.rx1_idx] \
                = self._unwrap(self.ts_data[pair][:,self.rx1_idx], max_time_ns)

            self.ts_data[pair][:,self.tx2_idx] \
                = self._unwrap(self.ts_data[pair][:,self.tx2_idx], max_time_ns)

            self.ts_data[pair][:,self.rx2_idx] \
                = self._unwrap(self.ts_data[pair][:,self.rx2_idx], max_time_ns)

            self.ts_data[pair][:,self.tx3_idx] \
                = self._unwrap(self.ts_data[pair][:,self.tx3_idx], max_time_ns)

            self.ts_data[pair][:,self.rx3_idx] \
                = self._unwrap(self.ts_data[pair][:,self.rx3_idx], max_time_ns)

    @staticmethod
    def _unwrap(data, max):
        temp = data[1:] - data[:-1]
        idx = np.concatenate([np.array([0]), temp < 0])

        iter = 0
        for lv0, _ in enumerate(data):
            if idx[lv0]:
                iter += 1
            data[lv0] += iter*max    

        return data

    @staticmethod
    def _unwrap_gt(t_sec, t, max):
        iter = 0
        for lv0 in range(len(t)-1):
            if t_sec[lv0+1] - t_sec[lv0] > 0:
                iter += t_sec[lv0+1] - t_sec[lv0]
            t[lv0+1] += iter*max    

        return t
    ### ------------------------------------------------------------------------ ###


    ### --------------------------- STITCHING METHODS -------------------------- ###
    def _stitch_time_intervals(self, pair):
        intervals_iter = self.time_intervals[pair]
        all_interv = {}
        all_interv["t"] = intervals_iter["t"]
        all_interv["Ra1"] =  intervals_iter["Ra1"]
        all_interv["Ra2"] =  intervals_iter["Ra2"]
        all_interv["Db1"] =  intervals_iter["Db1"]
        all_interv["Db2"] =  intervals_iter["Db2"]
        all_interv["tof1"] = intervals_iter["tof1"]
        all_interv["tof2"] = intervals_iter["tof2"]
        all_interv["tof3"] = intervals_iter["tof3"]
        all_interv["S1"] = intervals_iter["S1"]
        all_interv["S2"] = intervals_iter["S2"]

        return all_interv

    def _stitch_power(self, pair):        
        ts_iter = self.ts_data[pair]
        all_Pr = {}
        all_Pr["Pr1"] = ts_iter[:,self.Pr1_idx]
        all_Pr["Pr2"] = ts_iter[:,self.Pr2_idx]

        return all_Pr

    def _stitch_bias(self, pair):
        ts_iter = self.ts_data[pair]
        interv_iter = self.time_intervals[pair]

        # try:
        bias = ts_iter[:,self.range_idx] - interv_iter["r_gt"]
        # except:
            # bias = np.hstack((bias, ts_iter[:,self.range_idx] - self.mean_gt_distance[pair[::-1]]))

        return bias
    ### ------------------------------------------------------------------------ ###


    ### --------------------------- PLOTTING METHODS --------------------------- ###
    def _ss_twr_plotting(self, all_interv, pair):
        range = 0.5 * self._c / 1e9 * \
            (all_interv["Ra1"] - all_interv["Db1"])

        fig, axs = plt.subplots(1)

        axs.plot(all_interv["t"]/1e9, range, label='Range Measurements')
        axs.scatter(self._gt_distance[pair]["t"]/1e9, 
                    self._gt_distance[pair]["dist"], 
                    s=1,
                    label='Ground Truth')
        
        axs.set_ylabel("Distance [m]")
        axs.set_xlabel("t [s]")
        axs.set_ylim([-1, 5])

        axs.legend()

    def _ds_twr_plotting(self, all_interv, pair):
        range = 0.5 * self._c / 1e9 * \
            (all_interv["Ra1"] - (all_interv["Ra2"] / all_interv["Db2"]) * all_interv["Db1"])

        fig, axs = plt.subplots(1)

        axs.plot(all_interv["t"]/1e9, range, label='Range Measurements')
        axs.scatter(self._gt_distance[pair]["t"]/1e9, \
                    self._gt_distance[pair]["dist"], s=1, \
                    label='Ground Truth', color='r')

        axs.set_ylabel("Distance [m]")
        axs.set_xlabel("t [s]")
        axs.set_ylim([-1, 5])

        axs.legend()

    @staticmethod
    def lift(x, alpha=-82):
        return 10**((x - alpha) /10)

    def visualize_raw_data(self, pair=(1,2), alpha=-82):
        all_interv = self._stitch_time_intervals(pair)
        all_Pr = self._stitch_power(pair)
        bias = self._stitch_bias(pair)

        # Axes limits
        bias_l = -0.5
        bias_u = 0.5
        Pr_l = -110
        Pr_h = -80

        # RANGE MEASUREMENTS #
        if self.mult_twr:
            self._ds_twr_plotting(all_interv, pair)
        else:
            self._ss_twr_plotting(all_interv, pair)

        ### TIME INTERVALS 
        fig, axs = plt.subplots(3,3)

        col_num = 0
        row_num = 0
        for interv_str in all_interv:
            if interv_str == "t" or interv_str == "r_gt":
                continue
            interv = all_interv[interv_str]
            axs[row_num,col_num].plot(all_interv["t"]/1e9,interv)
            axs[row_num,col_num].set_ylabel(interv_str + " [ns]")
            axs[row_num,col_num].set_xlabel("t [s]")

            if col_num == 2:
                row_num += 1
                col_num = 0
            else:
                col_num += 1

        #### POWER VS BIAS ###
        fig, axs = plt.subplots(len(all_Pr))

        for i, Pr_str in enumerate(all_Pr):
            Pr = all_Pr[Pr_str]
            axs[i].scatter(self.lift(Pr),bias,s=1)
            axs[i].set_ylabel("Bias [m]")
            axs[i].set_xlabel("$f("+ Pr_str + ")$ [dBm]")
            # axs[i].set_xlim([self.lift(Pr_l), self.lift(Pr_h)])
            # axs[i].set_ylim([bias_l, bias_u])

        # BIAS AND POWER vs. TIME ##
        fig, axs = plt.subplots(3)

        axs[0].plot(all_interv["t"]/1e9,bias)
        axs[0].set_ylabel("Bias [m]")
        # axs[0].set_ylim([bias_l, bias_u])

        for i, Pr_str in enumerate(all_Pr):
            Pr = all_Pr[Pr_str]
            axs[i+1].plot(all_interv["t"]/1e9,Pr)
            axs[i+1].set_ylabel(Pr_str + " [dBm]")
            axs[i+1].set_ylim([Pr_l, Pr_h])

        axs[i+1].set_xlabel("t [s]")
    ### ------------------------------------------------------------------------ ###