import numpy as np
import pandas as pd
from pylie import SO3
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
    
    def __init__(
                    self, 
                    machines,
                    merge_pairs = False,
                ):
        self.merge_pairs = merge_pairs
        self._save_params(machines)    
        self._process_data(machines)

    def _save_params(self, machines):
        self.machine_ids = []
        self.tag_ids = {}
        self.moment_arms = {}
        for machine in machines:
            self.machine_ids = self.machine_ids + [machine]
            self.tag_ids[machine] = machines[machine].tag_ids
            self.moment_arms.update(machines[machine].moment_arms)
            
        self.max_ts_value = machines[self.machine_ids[0]].max_ts_value
        self.ts_to_ns = machines[self.machine_ids[0]].ts_to_ns
            
        self.ds_twr = machines[self.machine_ids[0]].ds_twr
        self.passive_listening = machines[self.machine_ids[0]].passive_listening
        self.fpp_exists = machines[self.machine_ids[0]].fpp_exists
        self.rxp_exists = machines[self.machine_ids[0]].rxp_exists
        self.std_exists = machines[self.machine_ids[0]].std_exists
        
        for machine in machines:
            if machines[machine].max_ts_value != self.max_ts_value:
                raise Exception(r"Not all machines are using the same timestamping types.")
            
            if machines[machine].ts_to_ns != self.ts_to_ns:
                raise Exception(r"Not all machines are using the same clock rate.")
            
            if machines[machine].ds_twr != self.ds_twr:
                raise Exception(r"Not all machines are using the same ranging protocol.")
            
            if machines[machine].ds_twr != self.ds_twr:
                raise Exception(r"Not all machines are using the same ranging protocol.")
            
            if machines[machine].passive_listening != self.passive_listening:
                self.passive_listening = False
            
            if machines[machine].fpp_exists != self.fpp_exists:
                self.fpp_exists = False
            
            if machines[machine].rxp_exists != self.rxp_exists:
                self.rxp_exists = False
            
            if machines[machine].std_exists != self.std_exists:
                self.std_exists = False

    def _process_data(self, machines):
        self._store_data(machines)
        self._unwrap_all_clocks()
        self._compute_intervals()
        
        if self.merge_pairs:
            self.df['pair'] = self.df['pair'].apply(sorted)
            self.pair_list = list(self.df['pair'].unique())

    def _store_data(self, machines):
        self._get_uwb_data(machines)
        self._get_pose_data(machines)
        self._get_distance_data()
        self._find_pairs()

    def _get_uwb_data(self, machines):
        all_dfs = []
        all_dfs_passive = []
        for machine in machines:
            all_dfs = all_dfs + [machines[machine].df_uwb]
            all_dfs_passive = all_dfs_passive + [machines[machine].df_passive]
        
        self.df = pd.concat(all_dfs)
        self.df.reset_index(inplace=True, drop=True)
        
        if self.passive_listening:
            self.df_passive = pd.concat(all_dfs_passive)
            self.df_passive.reset_index(inplace=True, drop=True)
            
            self._match_uwb_data()
    
    def _match_uwb_data(self):
        self.df_passive['idx'] = \
                        self.df_passive.apply(
                                                self._match_tx_ts, 
                                                axis=1
                                             )
    
    def _match_tx_ts(self, row):
        # TODO: Speed up this timestamp matching process
        t1 = row['tx1_n']
        t2 = row['tx2_n']
        t3 = row['tx3_n']
        
        index = self.df[(self.df['tx1'] == t1) 
                        & (self.df['tx2'] == t2) 
                        & (self.df['tx3'] == t3)].index 
        if not len(index):
            index = [np.NaN]
        
        return index[0]
    
    def _get_pose_data(self, machines):
        t_new = self.df["time"]
        for machine in machines:
            t = np.array(machines[machine].df_pose['time'].to_list())
            r_iw_a = np.array(machines[machine].df_pose['r_iw_a'].to_list())
            q_ai = np.array(machines[machine].df_pose['q_ai'].to_list())
             
            self.df['r_iw_a_'+machine] = list(self._interpolate(r_iw_a, t, t_new))
            self.df['q_ai_'+machine] = list(self._interpolate(q_ai, t, t_new))
            
    def _get_distance_data(self):
        self.df['gt_range'] = self.df.apply(
                                            self._compute_distance, 
                                            args=(self.tag_ids, self.moment_arms), 
                                            axis=1
                                           )
        self.df['bias'] = self.df.apply(
                                        self._compute_bias, 
                                        axis=1
                                       )
        
        
    @staticmethod
    def _compute_distance(row, tag_ids, moment_arms):
        id0 = row['from_id']
        id1 = row['to_id']
        
        machine0 = [machine for machine in tag_ids \
                            if id0 in tag_ids[machine]][0]
        machine1 = [machine for machine in tag_ids \
                            if id1 in tag_ids[machine]][0]
        
        r_0w_a = row['r_iw_a_'+machine0]
        q_a0 = row['q_ai_'+machine0]
        r_t0_0 = moment_arms[id0]
        C_a0 = SO3.from_quat(q_a0, order='xyzw')
        
        r_1w_a = row['r_iw_a_'+machine1]
        q_a1 = row['q_ai_'+machine1]
        r_t1_1 = moment_arms[id1]
        C_a1 = SO3.from_quat(q_a1, order='xyzw')
        
        return np.linalg.norm(
                                C_a0 @ r_t0_0
                                + r_0w_a
                                - r_1w_a
                                - C_a1 @ r_t1_1
                             )
        
    @staticmethod 
    def _compute_bias(df):
        return df['range'] - df['gt_range']

    def _find_pairs(self):
        self.df['pair'] = tuple(zip(self.df.from_id, self.df.to_id))
        self.pair_list = list(self.df['pair'].unique())
    
    def _compute_intervals(self):
        self.df["Ra1"] = self.df['rx2'] - self.df['tx1']
        self.df["Db1"] = self.df['tx2'] - self.df['rx1']
        self.df["tof1"] = self.df['rx1'] - self.df['tx1']
        self.df["tof2"] = self.df['rx2'] - self.df['tx2']
        self.df["S1"] = self.df['rx2'] + self.df['tx1']
        self.df["S2"] = self.df['tx2'] + self.df['rx1']
        
        if self.ds_twr:
            self.df["Ra2"] = self.df['rx3'] - self.df['rx2']
            self.df["Db2"] = self.df['tx3'] - self.df['tx2']
            self.df["tof3"] = self.df['rx3'] - self.df['tx3']
            
        if self.passive_listening:
            self.df_passive["tof1"] = self.df_passive["tx1_n"] - self.df_passive["rx1"]
            self.df_passive["tof2"] = self.df_passive["tx2_n"] - self.df_passive["rx2"]
            
            if self.ds_twr:
                self.df_passive["tof3"] = self.df_passive["tx3_n"] - self.df_passive["rx3"]

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
        
        f = interp1d(t_old, x, kind='linear', fill_value='extrapolate', axis=0)

        return f(t_new)
    ### ------------------------------------------------------------------------ ###


    ### -------------------------- UNWRAPPING METHODS -------------------------- ###
    def _unwrap_all_clocks(self):
        """
        Unwrap the clock for all the time-stamp measurements.
        TODO: I would like to find a better way to deal with this. 
        """
        # Timestamps are represented as uint32
        max_ts_ns = self.max_ts_value * self.ts_to_ns

        ### --- Unwrap time-stamps --- ###
        for pair in self.pair_list: 
            df_iter = self.df[self.df['pair']==pair].copy()
            
            iter_rx2 = 0
            iter_tx2 = 0
            iter_tx3 = 0
            iter_rx3 = 0
            
            # Check if a clock wrap occured at the first measurement, and unwrap
            if df_iter['rx2'].iloc[0] < df_iter['tx1'].iloc[0]:
                df_iter.at[0,'rx2'] += max_ts_ns
                iter_rx2 = 1
                iter_rx3 = 1

            if df_iter['tx2'].iloc[0] < df_iter['rx1'].iloc[0]:
                df_iter.at[0,'tx2'] += max_ts_ns
                iter_tx2 = 1
                iter_tx3 = 1

            if self.ds_twr:
                if df_iter['tx3'].iloc[0] < df_iter['tx2'].iloc[0]:
                    df_iter.at[0,'tx3'] += max_ts_ns
                    iter_tx3 = 1

                if df_iter['rx3'].iloc[0] < df_iter['rx2'].iloc[0]:
                    df_iter.at[0,'rx3'] += max_ts_ns
                    iter_rx3 = 1

            # Individual unwraps
            df_iter['tx1'] = self._unwrap(df_iter['tx1'], max_ts_ns)
            df_iter['rx1'] = self._unwrap(df_iter['rx1'], max_ts_ns)
            df_iter['tx2'] = self._unwrap(df_iter['tx2'], max_ts_ns, iter=iter_tx2)
            df_iter['rx2'] = self._unwrap(df_iter['rx2'], max_ts_ns, iter=iter_rx2)

            if self.ds_twr:
                df_iter['tx3'] = self._unwrap(df_iter['tx3'], max_ts_ns, iter=iter_tx3)
                df_iter['rx3'] = self._unwrap(df_iter['rx3'], max_ts_ns, iter=iter_rx3)
            
            # 
            self.df[self.df['pair']==pair] = df_iter

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
        data = np.array(data)
        
        temp = data[1:] - data[:-1]
        idx = np.concatenate([np.array([0]), temp < 0])

        # iter = 0
        for lv0, _ in enumerate(data):
            if idx[lv0]:
                iter += 1
            data[lv0] += iter*max    

        return data
    ### ------------------------------------------------------------------------ ###


    ### --------------------------- PLOTTING METHODS --------------------------- ###

    ### ------------------------------------------------------------------------ ###