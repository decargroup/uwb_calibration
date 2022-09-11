import numpy as np
import pandas as pd
from pylie import SO3
from scipy.interpolate import interp1d
import pickle

def load(filename='data.pickle'):
    with open(filename, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
        
    return data

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

    TODO 1: Need to make this robust to the user dropping rows and reindexing df, because otherwise
            the indices in df_passive get all mixed up. 
         2: Add tests.
         3: Parallelize some things?
    """
    
    def __init__(
                    self, 
                    machines,
                    compute_intervals = False,
                    merge_pairs = False,
                ):
        self._save_params(machines)    
        self._store_data(machines)
        
        if compute_intervals:
            self._compute_intervals()
        
        if merge_pairs:
            self.df['pair'] = self.df['pair'].apply(sorted)
            self.pair_list = list(self.df['pair'].unique())


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
        self.df.sort_values(by=["time"], inplace=True)
        self.df.reset_index(inplace=True, drop=True)
        
        if self.passive_listening:
            self.df_passive = pd.concat(all_dfs_passive)
            self.df_passive.sort_values(by=["time"], inplace=True)
            self.df_passive.reset_index(inplace=True, drop=True)
            
            self._match_uwb_data()
    
    def _match_uwb_data(self):
        self.df_passive['idx'] = \
                        self.df_passive.apply(
                                                self._match_tx_ts, 
                                                axis=1
                                             )
        self.df_passive.drop(columns=['tx1_n',
                                      'tx2_n',
                                      'tx3_n',
                                      'rx1_n',
                                      'rx2_n',
                                      'rx3_n',],
                             inplace=True)
        self.df_passive.dropna(subset=["idx"], inplace=True)
    
    def _match_tx_ts(self, row):
        # TODO: Speed up this timestamp matching process
        t1 = row['tx1_n']
        t2 = row['tx2_n']
        t3 = row['tx3_n']
        r1 = row['rx1_n']
        r2 = row['rx2_n']
        r3 = row['rx3_n']
        
        index = self.df[(self.df['tx1'] == t1) 
                        & (self.df['tx2'] == t2) 
                        & (self.df['tx3'] == t3)
                        & (self.df['rx1'] == r1)
                        & (self.df['rx2'] == r2)
                        & (self.df['rx3'] == r3)].index 
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
                                            self.compute_distance, 
                                            axis=1
                                           )
        self.df['bias'] = self.df.apply(
                                        self._compute_bias, 
                                        axis=1
                                       )
        
        
    def compute_distance(self, row, id=[]):
        if not id:
            id = [row['from_id'], row['to_id']]
        
        machine0 = [machine for machine in self.tag_ids \
                            if id[0] in self.tag_ids[machine]][0]
        machine1 = [machine for machine in self.tag_ids \
                            if id[1] in self.tag_ids[machine]][0]
        
        r_0w_a = row['r_iw_a_'+machine0]
        q_a0 = row['q_ai_'+machine0]
        r_t0_0 = self.moment_arms[id[0]]
        C_a0 = SO3.from_quat(q_a0, order='xyzw')
        
        r_1w_a = row['r_iw_a_'+machine1]
        q_a1 = row['q_ai_'+machine1]
        r_t1_1 = self.moment_arms[id[1]]
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
    def _compute_intervals(self):
        # TODO: make it applicable to SS
        # TODO: tidy it up a bit?
        # TODO: deal with the warning error. Could use the hacky
        #           pd.options.mode.chained_assignment = None  
        # TODO: What to do with the custom K thing that I have?
        max_ts_ns = self.max_ts_value * self.ts_to_ns
        
        self.df['del_41'] = np.zeros((len(self.df)))
        self.df['del_32'] = np.zeros((len(self.df)))
        self.df['del_21'] = np.zeros((len(self.df)))
        self.df['del_43'] = np.zeros((len(self.df)))
        self.df['del_64'] = np.zeros((len(self.df)))
        self.df['del_53'] = np.zeros((len(self.df)))
        self.df_passive['del_71'] = np.zeros((len(self.df_passive)))
        self.df_passive['del_83'] = np.zeros((len(self.df_passive)))
        self.df_passive['del_95'] = np.zeros((len(self.df_passive)))
        
        for tag_i in self.moment_arms:
            for tag_j in self.moment_arms:
                cond0 = self.df['pair']==(tag_i, tag_j)
                df_iter = self.df[cond0].copy()
                if df_iter.empty:
                    continue
                t1 = df_iter['tx1']
                t2 = df_iter['rx1']
                t3 = df_iter['tx2']
                t4 = df_iter['rx2']
                t5 = df_iter['tx3']
                t6 = df_iter['rx3']
                
                df_iter['del_64'] = self._unwrap_intervals(np.array(t6 - t4), 
                                                           max_ts_ns)
                df_iter['del_53'] = self._unwrap_intervals(np.array(t5 - t3), 
                                                           max_ts_ns)
                df_iter['del_41'] = self._unwrap_intervals(np.array(t4 - t1), 
                                                           max_ts_ns)
                df_iter['del_32'] = self._unwrap_intervals(np.array(t3 - t2), 
                                                           max_ts_ns)
                df_iter['del_21'] = self._unwrap_intervals(np.array(t2 - t1), 
                                                           max_ts_ns)
                df_iter['del_43'] = self._unwrap_intervals(np.array(t4 - t3), 
                                                           max_ts_ns)
                
                self.df[cond0] = df_iter
                
                for tag_p in self.moment_arms:
                    cond1 = (self.df_passive['from_id'] == tag_i)
                    cond2 = (self.df_passive['to_id'] == tag_j)
                    cond3 = (self.df_passive['my_id'] == tag_p)
                    df_passive_iter = self.df_passive[cond1 & cond2 & cond3]
                    if df_passive_iter.empty:
                        continue
                    
                    t7 = df_passive_iter['rx1']
                    t8 = df_passive_iter['rx2']
                    t9 = df_passive_iter['rx3']
                    
                    del_71 = t7 - self._match_indices(t1, np.array(df_passive_iter['idx']))
                    df_passive_iter['del_71'] = self._unwrap_intervals(np.array(del_71), max_ts_ns)
                    del_83 = t8 - self._match_indices(t3, np.array(df_passive_iter['idx']))
                    df_passive_iter['del_83'] = self._unwrap_intervals(np.array(del_83), max_ts_ns)
                    del_95 = t9 - self._match_indices(t5, np.array(df_passive_iter['idx']))
                    df_passive_iter['del_95'] = self._unwrap_intervals(np.array(del_95), max_ts_ns)
                                        
                    self.df_passive[cond1 & cond2 & cond3] = df_passive_iter
    
    @staticmethod
    def _unwrap_intervals(x, max_val):
        avg = np.mean(x)
        sum_lower = np.sum(x < avg)
        sum_higher = np.sum(x > avg)
        if sum_lower > sum_higher:
            x[x > avg] -= max_val
        else:
            x[x < avg] += max_val
            
        return x
        
    @staticmethod
    def _match_indices(x0, idx):
        return np.array(x0.loc[idx])
    
    def save(self, filename="data.pickle"):
        
        with open(filename,"wb") as file:
            pickle.dump(self, file)
    ### ------------------------------------------------------------------------ ###


    ### --------------------------- PLOTTING METHODS --------------------------- ###

    ### ------------------------------------------------------------------------ ###