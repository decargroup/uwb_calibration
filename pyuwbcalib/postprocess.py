import numpy as np
import pandas as pd
from pylie import SO3
from .utils import interpolate
from .machine import Machine
from typing import List

class PostProcess(object):
    """A class to process pose and UWB data from multiple UWB tags.

    Attributes
    ----------
    merge_pairs: bool
        If pairs are merged together irrespective of whom of the pair initiates.
    machine_ids: list of str
        The IDs of all the machines.
    tag_ids: dict
        keys: str
            The ID of the machine.
        values: list of int
            The IDs of the tag installed on this machine.
    moment_arms: dict
        keys: int
            The ID of the tag
        values: list
            The 3D position of the tag relative to the machine's reference point,
            in the machine's body frame.
    max_ts_value: float
        The maximum timestamp value that can be recorded by the machine before wrapping.
        For example, if the timestamp is recorded as uint32, then max_ts_value = 2**32.
    ts_to_ns: float
        The conversion from timestamp value to nanoseconds.
    ds_twr: bool
        If double-sided TWR is used; False represents single-sided TWR.
    fpp_exists: bool
        If the first-path power is recorded.
    rxp_exists: bool
        If the average received power is recorded.
    std_exists: bool
        If the leading-edge-detection's uncertainty metric is recorded.
    pair_list: list of tuple
        A list of all the unique pairs of ranging tags.
    df: pd.DataFrame
        A dataframe of the processed data with the following columns.
        'time': float
            Timestamp of the measurement, in seconds.
        'range': float
            The recorded range measurement.
        'from_id': int
            The ID of the initiating tag. 
        'to_id': int
            The ID of the target tag.
        'tx1': float
            The transmission timestamp of the first signal.
        'rx1': float
            The reception timestamp of the first signal.
        'tx2': float
            The transmission timestamp of the second signal.
        'rx2': float
            The reception timestamp of the second signal.
        'tx3': float
            The transmission timestamp of the third signal. 
            Only exists if self.ds_twr == True.
        'rx3': float
            The reception timestamp of the third signal. 
            Only exists if self.ds_twr == True. 
        'fpp1': float
            The received first-path power of the first signal.
            Only exists if self.fpp_exists == True.
        'fpp2': float
            The received first-path power of the second signal.
            Only exists if self.fpp_exists == True.
        'rxp1': float
            The receieved average power of the first signal. 
            Only exists if self.rxp_exists == True.
        'rxp2': float
            The received average power of the second signal.
            Only exists if self.rxp_exists == True.
        'std1': int
            The leading-edge detection algorithm's uncertainty on the first signal.
            Only exists if self.std_exists == True.
        'std2': int
            The leading-edge detection algorithm's uncertainty on the second signal.
            Only exists if self.std_exists == True.
        'r_iw_a_*': list of float
            Position of machine (*), as recorded by the motion-capture system.
            There exists one column per machine, where * is replaced with machine_id.
        'q_ai_*': np.ndarray
            Quaternion parametrization of the orientation of machine (*) using the 
            [x,y,z,w] convention, as recorded by the motion-capture system.
            There exists one column per machine, where * is replaced with machine_id.
        'gt_range': float
            The ground-truth distance between the ranging tags.
        'bias': float
            The bias in the measurement, computed as 
            >>> df['bias'] = df['range'] - df['gt_range']
        'pair': tuple
            The pair of ranging tags, computed as
            >>> df['pair'] = (df['from_id'], df['to_id'])
            If self.merge_pairs is True, the tuple might be switched to match pairs.
        'del_t1': float
            The unwrapped first time interval computed as
            >>> df['rx2'] - ['tx1']
        'del_t2': float
            The unwrapped second time interval computed as
            >>> df['tx2'] - ['rx1']
        'del_t3': float
            The unwrapped third time interval computed as
            >>> df['rx3'] - ['rx2']
        'del_t4': float
            The unwrapped fourth time interval computed as
            >>> df['tx3'] - ['tx2']
        'tof1': float
            The unwrapped first time-of-flight measurement computed as
            >>> df['rx1'] - ['tx1']
        'tof2': float
            The unwrapped second time-of-flight measurement computed as
            >>> df['rx2'] - ['tx2']
        'tof3': float
            The unwrapped third time-of-flight measurement computed as
            >>> df['rx3'] - ['tx3']
        'sum_t1': float
            The unwrapped first summed interval computed as
            >>> df['rx2'] + ['tx1'] 
        'sum_t2': float
            The unwrapped second summed interval computed as
            >>> df['tx2'] + ['rx1'] 
    # TODO: add df_passive documentation.

    Examples
    --------
    # Read config file and corresponding data
    config_file = 'config/ifo_3_drones_rosbag.config'

    parser = ConfigParser(interpolation=ExtendedInterpolation())
    parser.read(config_file)

    machines = {}
    for i,machine in enumerate(parser['MACHINES']):
        machine_id = parser['MACHINES'][machine]
        machines[machine_id] = RosMachine(parser, i)
    
    # Instantiate PostProcess
    data = PostProcess(machines)

    # Plot the evolution of the bias over time for ranging pair (1,3)
    df_pair = data.df[data.df['pair'] == (1,3)]
    plt.plot(df_pair['time'], df_pair['bias'])
    """
    def __init__(
        self, 
        machines,
        merge_pairs = False,
    ) -> None:
        """Constructor

        Parameters
        ----------
        machines: list of Machine
            A list of all the Machine objects, one per machine.
        merge_pairs: bool, optional
            If pairs are merged together irrespective of whom of the pair initiates, 
            by default False
        """
        self.merge_pairs = merge_pairs
        self._save_params(machines)    
        self._process_data(machines)

    def _save_params(self, machines: List[Machine]) -> None:
        """Inherit some parameters from the Machine objects, and check for inconsistencies
        between different machines.

        Parameters
        ----------
        machines: list of Machine
            A list of all the Machine objects, one per machine.

        Raises
        ------
        Exception
            Not all machines are using the same timestamping data types.
        Exception
            Not all machines are using the same clock rate.
        Exception
            Not all machines are using the same ranging protocol.
        """
        # Retrieve information about the machines
        self.machine_ids = []
        self.tag_ids = {}
        self.moment_arms = {}
        for machine in machines:
            self.machine_ids = self.machine_ids + [machine]
            self.tag_ids[machine] = machines[machine].tag_ids
            self.moment_arms.update(machines[machine].moment_arms)
            
        # Retrieve the timestamping information
        self.max_ts_value = machines[self.machine_ids[0]].max_ts_value
        self.ts_to_ns = machines[self.machine_ids[0]].ts_to_ns
            
        # Retrieve the ranging protocol and what data is collected
        self.ds_twr = machines[self.machine_ids[0]].ds_twr
        self.passive_listening = machines[self.machine_ids[0]].passive_listening
        self.fpp_exists = machines[self.machine_ids[0]].fpp_exists
        self.rxp_exists = machines[self.machine_ids[0]].rxp_exists
        self.std_exists = machines[self.machine_ids[0]].std_exists
        
        for machine in machines:
            # Check if any machine is using a different data type for timestamping
            if machines[machine].max_ts_value != self.max_ts_value:
                raise Exception(
                    r"Not all machines are using the same timestamping data types."
                )
            
            # Check if any machine is using a different clock frequency
            if machines[machine].ts_to_ns != self.ts_to_ns:
                raise Exception(
                    r"Not all machines are using the same clock rate."
                )
            
            # Check if any machine is using a different ranging protocol
            if machines[machine].ds_twr != self.ds_twr:
                raise Exception(
                    r"Not all machines are using the same ranging protocol."
                )
            
            # Check if any machine does not record fpp, if so, 
            # neglect fpp information for all machines
            if machines[machine].fpp_exists != self.fpp_exists:
                self.fpp_exists = False
            
            # Check if any machine does not record rxp, if so, 
            # neglect rxp information for all machines
            if machines[machine].rxp_exists != self.rxp_exists:
                self.rxp_exists = False
            
            # Check if any machine does not record std, if so, 
            # neglect std information for all machines
            if machines[machine].std_exists != self.std_exists:
                self.std_exists = False

    def _process_data(self, machines) -> None:
        """Store the pose and UWB data locally and do all necessary processing.

        Parameters
        ----------
        machines: list of Machine
            A list of all the Machine objects, one per machine.
        """
        self._store_uwb_data(machines)
        
        self._store_pose_data(machines)
        self._store_distance_data()
        
        self._get_pairs()
        
        self._unwrap_all_clocks()
        self._compute_intervals()
        
        # If pairs are to be merged, combine flipped pair tuples by sorting.
        if self.merge_pairs:
            self.df['pair'] = self.df['pair'].apply(sorted)
            self.pair_list = list(self.df['pair'].unique())

    def _store_uwb_data(self, machines) -> None:
        """Get and store UWB data locally.

        Parameters
        ----------
        machines: list of Machine
            A list of all the Machine objects, one per machine.
        """
        # Get UWB data from all machines
        all_dfs = []
        all_dfs_passive = []
        for machine in machines:
            all_dfs = all_dfs + [machines[machine].df_uwb]
            if self.passive_listening:
                all_dfs_passive = all_dfs_passive + [machines[machine].df_passive]

        # Combine dataframes into one local dataframe
        self.df = pd.concat(all_dfs)
        self.df.sort_values(by=['time'] , inplace=True)
        self.df.reset_index(inplace=True, drop=True)

        if self.passive_listening:
            self.df_passive = pd.concat(all_dfs_passive)
            self.df_passive.sort_values(by=["time"], inplace=True)
            self.df_passive.reset_index(inplace=True, drop=True)
            
            # Match entries in df_passive with entries in df_uwb
            self._match_uwb_data()

    def _match_uwb_data(self):
        """Adds a new "idx" field to df_passive which indicates the row in
        df_uwb that represents the corresponding TWR instance.
        """
        # Get the corresponding row in df_uwb.
        self.df_passive['idx'] = \
                        self.df_passive.apply(
                                                self._match_tx_ts, 
                                                axis=1
                                             )

        # Drop overlapping fields between the two dataframes
        self.df_passive.drop(columns=['tx1_n',
                                      'tx2_n',
                                      'tx3_n',
                                      'rx1_n',
                                      'rx2_n',
                                      'rx3_n',],
                             inplace=True)

        # Drop any passive listening measurements corresponding to
        # missed TWR measurements.
        self.df_passive.dropna(subset=["idx"], inplace=True)
    
    def _match_tx_ts(self, row):
        """Find the row in df_uwb corresponding to a single entry in df_passive.
        # TODO: Speed up this timestamp matching process

        Parameters
        ----------
        row : df
            A row from df_passive.

        Returns
        -------
        int
            The corresponding row in df_uwb.
        """
        # Get all timestamps that must overlap between the two dataframes.
        t1 = row['tx1_n']
        t2 = row['tx2_n']
        t3 = row['tx3_n']
        r1 = row['rx1_n']
        r2 = row['rx2_n']
        r3 = row['rx3_n']
        
        # Extract the index of the corresponding row. 
        index = self.df[(self.df['tx1'] == t1) 
                        & (self.df['tx2'] == t2) 
                        & (self.df['tx3'] == t3)
                        & (self.df['rx1'] == r1)
                        & (self.df['rx2'] == r2)
                        & (self.df['rx3'] == r3)].index 

        # If there is no overlap, return None
        if not len(index):
            index = [np.NaN]
        
        return index[0]
    
    def _store_pose_data(self, machines) -> None:
        """Get and store pose data locally.

        Parameters
        ----------
        machines: list of Machine
            A list of all the Machine objects, one per machine.
        """
        # Get timestamps of UWB measurements
        t_new = self.df["time"]

        # Address one machine at a time
        for machine in machines:
            # Get pose data for this machine
            t = np.array(machines[machine].df_pose['time'].to_list())
            r_iw_a = np.array(machines[machine].df_pose['r_iw_a'].to_list())
            q_ai = np.array(machines[machine].df_pose['q_ai'].to_list())
             
            # Interpolate pose data to UWB timestamps and save into main dataframe
            self.df['r_iw_a_'+machine] = list(interpolate(r_iw_a, t, t_new))
            self.df['q_ai_'+machine] = list(interpolate(q_ai, t, t_new))
            
    def _store_distance_data(self) -> None:
        """Get and store distance between tags from the ground-truth poses.
        """
        # Compute and store the ground-truth range.
        self.df['gt_range'] = self.df.apply(
            self._compute_distance, 
            args=(self.tag_ids, self.moment_arms), 
            axis=1
        )

        # Compute and store the range bias.
        self.df['bias'] = self.df.apply(
            self._get_bias, 
            axis=1
        )
        
        
    @staticmethod
    def _compute_distance(
        row, 
        tag_ids, 
        moment_arms
    ) -> float:
        """Compute the distance between two tags at one instant.

        Parameters
        ----------
        row: pd.dataframe
            One row corresponding to one measurement. This must include 
            the following headers:
                ['from_id', 'to_id', 'r_iw_a_*', 'q_ai_*']
        tag_ids: dict
            keys: str
                The ID of the machine.
            values: list of int
                The IDs of the tag installed on this machine.
        moment_arms: dict
            keys: int
                The ID of the tag
            values: list
                The 3D position of the tag relative to the machine's reference point,
                in the machine's body frame.

        Returns
        -------
        float
            Computed ground-truth range for this measurement.
        """
        # Get the IDs of the ranging tags
        id0 = row['from_id']
        id1 = row['to_id']
        
        # Get the IDs of the machines that have those two tags
        machine0 = [machine for machine in tag_ids \
                            if id0 in tag_ids[machine]][0]
        machine1 = [machine for machine in tag_ids \
                            if id1 in tag_ids[machine]][0]
        
        # Get the pose of the first machine
        r_0w_a = row['r_iw_a_'+machine0]
        q_a0 = row['q_ai_'+machine0]
        C_a0 = SO3.from_quat(q_a0, order='xyzw')
        
        # Get the position of the tag of the first machine relative to its reference point
        r_t0_0 = moment_arms[id0]
        
        # Get the pose of the second machine
        r_1w_a = row['r_iw_a_'+machine1]
        q_a1 = row['q_ai_'+machine1]
        C_a1 = SO3.from_quat(q_a1, order='xyzw')

        # Get the position of the tag of the second machine relative to its reference point
        r_t1_1 = moment_arms[id1]
        
        # Return the distance between the two tags.
        return np.linalg.norm(
            C_a0 @ r_t0_0
            + r_0w_a
            - r_1w_a
            - C_a1 @ r_t1_1
        )
        
    @staticmethod 
    def _get_bias(df) -> float:
        """Get the ranging bias.

        Parameters
        ----------
        df: pd.dataframe
            Dataframe containing range and ground truth range (gt_range).

        Returns
        -------
        pd.dataframe
            The computed bias.
        """
        return df['range'] - df['gt_range']

    def _get_pairs(self) -> None:
        """Save the ranging pair for every measurement, and get a list of all unique
        ranging pairs.
        """
        self.df['pair'] = tuple(zip(self.df.from_id, self.df.to_id))
        self.pair_list = list(self.df['pair'].unique())
    
    def _compute_intervals(self) -> None:
        """Compute the timestamp intervals.
        # TODO: compute intervals associated with passive listening.
        """
        self.df["del_t1"] = self.df['rx2'] - self.df['tx1']
        self.df["del_t2"] = self.df['tx2'] - self.df['rx1']
        if self.ds_twr:
            self.df["del_t3"] = self.df['rx3'] - self.df['rx2']
            self.df["del_t4"] = self.df['tx3'] - self.df['tx2']
        
        self.df["tof1"] = self.df['rx1'] - self.df['tx1']
        self.df["tof2"] = self.df['rx2'] - self.df['tx2']
        if self.ds_twr:
            self.df["tof3"] = self.df['rx3'] - self.df['tx3']

        self.df["sum_t1"] = self.df['rx2'] + self.df['tx1']
        self.df["sum_t2"] = self.df['tx2'] + self.df['rx1']

    ### ------------------------------------------------------------------------ ###


    ### -------------------------- UNWRAPPING METHODS -------------------------- ###
    def _unwrap_all_clocks(self) -> None:
        """Unwrap all timestamps, one at a time. 
        # TODO: Unwrap using the new way where we use the laptop timestamps.
        # TODO: Need to unwrap passive listening measurements as well. We have corresponding indices.
        """
        # Find the maximum timestamp value before unwrapping, in nanoseconds 
        max_ts_ns = self.max_ts_value * self.ts_to_ns

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
    def _unwrap(
        data, 
        max, 
        iter=0
    ) -> np.ndarray:
        """Unwrap one clock.

        Parameters
        ----------
        data: pd.dataframe
            Data corresponding to one clock.
        max: float
            The maximum timestamp value before unwrapping, in nanoseconds.
        iter: int, optional
            Number of times unwrapping occurred at the first measurement, by default 0.

        Returns
        -------
        np.ndarray
            Unwrapped timestamps.
        """
        # Convert to np.ndarray
        data = np.array(data)
        
        # Find indices associated with negative deltas.
        temp = data[1:] - data[:-1]
        idx = np.concatenate([np.array([0]), temp < 0])

        # Unwrap by adding the max value whenever a negative delta occurs.
        for lv0, _ in enumerate(data):
            if idx[lv0]:
                iter += 1
            data[lv0] += iter*max    

        return data
    ### ------------------------------------------------------------------------ ###
    
    ### -------------------------- GET PARAMS METHODS -------------------------- ###
    def get_machine_pos(
        self, 
        machine_id, 
        as_numpy = False
    ) -> np.ndarray:
        """Get the position of a machine over time. 

        Parameters
        ----------
        machine_id: str
            The ID of the machine whose position we want.
        as_numpy: bool, optional
            Return data as np.ndarray rather than pd.dataframe, by default False

        Returns
        -------
        pd.dataframe or np.ndarray
            Returns a list of the position of the machine over time.
            If as_numpy = False (default), return pd.dataframe,
            else, return np.ndarray.
        """
        if as_numpy:
            return np.vstack(self.df['r_iw_a_' + machine_id])
        else:
            return self.df['r_iw_a_' + machine_id]

    ### ------------------------------------------------------------------------ ###

    ### --------------------------- PLOTTING METHODS --------------------------- ###

    ### ------------------------------------------------------------------------ ###