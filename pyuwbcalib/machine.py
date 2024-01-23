from typing import Tuple
from bagpy import bagreader
import numpy as np
import pandas as pd
from configparser import ConfigParser

class Machine(object):
    """A base class for UWB machines.

    This class reads configuration files and provides generic methods common to all 
    machines.

    Attributes
    ----------
    max_ts_value : float
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
    pose_path: str
        The path of the file containing the ground-truth data.
    uwb_path: str
        The path of the file containing the UWB data.
    machine_id: str
        The ID of the machine.
    tag_ids: list of int
        The ID of the tags installed on this machine.
    moment_arms: dict
        keys: int
            The ID of the tag
        values: list
            The 3D position of the tag relative to the machine's reference point,
            in the machine's body frame.
    pose_topic: str,
        ROS ONLY. The topic name for the ground truth pose. 
    uwb_topic: str
        ROS ONLY. The topic name for the UWB topic.
    uwb_fields: list of str
        Fieldnames representing the fields collected through UWB.
    """
    def __init__(
        self,
        configs: ConfigParser,
        id: int,
        is_ros: bool = True,
    ) -> None:
        """Constructor

        Parameters
        ----------
        configs : ConfigParser
            A ConfigParser object where the config file is parsed.
        id : int
            The ID of the machine, as referenced in the config file.
        is_ros : bool, optional
            If this class is instantiated by RosMachine, by default True
        """
        
        # Retrieve timestamping-specific params
        self.max_ts_value = eval(configs['PARAMS']['max_ts_value'])
        self.ts_to_ns = eval(configs['PARAMS']['ts_to_ns'])
        
        # Retrieve booleans specifying what kind of data to expect
        self.ds_twr = eval(configs['PARAMS']['ds_twr'])
        self.passive_listening = eval(configs['PARAMS']['passive_listening'])
        self.fpp_exists = eval(configs['PARAMS']['fpp_exists'])
        self.rxp_exists = eval(configs['PARAMS']['rxp_exists'])
        self.std_exists = eval(configs['PARAMS']['std_exists'])
        
        # Retrieve path information for ground-truth data 
        pose_dir = configs['POSE_PATH']['directory']
        self.pose_path = pose_dir + configs['POSE_PATH'][str(id)]

        # Retrieve path information for UWB data
        uwb_dir = configs['UWB_PATH']['directory']
        self.uwb_path = uwb_dir + configs['UWB_PATH'][str(id)]
        
        # Retrieve IDs and physical information about the machine
        self.machine_id = configs['MACHINES'][str(id)]
        self.tag_ids = eval(configs['TAGS'][str(id)])
        self.moment_arms = {tag : eval(configs['MOMENT_ARMS'][str(tag)]) for tag in self.tag_ids}

        # Retrieve ROS topic names for the data to be collected
        if is_ros:
            self.pose_topic = configs['POSE_TOPIC'][str(id)]
            self.uwb_topic = configs['UWB_TOPIC'][str(id)]

            if self.passive_listening:
                self.passive_topic = configs['LISTENING_TOPIC'][str(id)]
        
        # Retrieve the message fields for the UWB data
        self.uwb_fields = configs['UWB_MESSAGE']
        if self.passive_listening:
            self.passive_fields = configs['LISTENING_MESSAGE']
        
    def convert_uwb_timestamps(self) -> None:
        """Covert UWB timestamps to nanoseconds.
        """
        self.df_uwb['tx1'] *= self.ts_to_ns
        self.df_uwb['rx1'] *= self.ts_to_ns
        self.df_uwb['tx2'] *= self.ts_to_ns
        self.df_uwb['rx2'] *= self.ts_to_ns
        
        if self.ds_twr:
            self.df_uwb['tx3'] *= self.ts_to_ns
            self.df_uwb['rx3'] *= self.ts_to_ns

        if self.passive_listening:
            self.df_passive['rx1'] *= self.ts_to_ns
            self.df_passive['rx2'] *= self.ts_to_ns
            self.df_passive['tx1_n'] *= self.ts_to_ns
            self.df_passive['rx1_n'] *= self.ts_to_ns
            self.df_passive['tx2_n'] *= self.ts_to_ns
            self.df_passive['rx2_n'] *= self.ts_to_ns
            
            if self.ds_twr:
                self.df_passive['rx3'] *= self.ts_to_ns
                self.df_passive['tx3_n'] *= self.ts_to_ns
                self.df_passive['rx3_n'] *= self.ts_to_ns
            
    def drop_target_meas(self) -> None:
        """If range measurements are stored at both initiating and targetted tag, 
        this function removes measurements at the targetted tag to avoid duplicates.
        """
        # Check if the initiating tag is not on this machine
        bool = [id not in self.tag_ids for id in self.df_uwb['from_id']]
        
        # Drop all rows that do satisfy the above condition
        self.df_uwb.drop(self.df_uwb[bool].index, inplace=True)
        self.df_uwb.reset_index(inplace=True, drop=True)
        
    def merge_pose_data(self) -> None:
        """Merge pose data from individual columns representing each dimension to
        two columns of lists, one for position and one for quaternion attitude.
        """
        # Merge position data.
        self.df_pose['r_iw_a'] = list(np.array((
            self.df_pose['pose.position.x'],
            self.df_pose['pose.position.y'],
            self.df_pose['pose.position.z'],
        )).T)

        # Merge quaternion attitude data.
        self.df_pose['q_ai'] = list(np.array((
            self.df_pose['pose.orientation.x'],
            self.df_pose['pose.orientation.y'],
            self.df_pose['pose.orientation.z'],
            self.df_pose['pose.orientation.w'],
        )).T)
        
        # Drop the individual columns.
        self.df_pose.drop(
            columns=[
                'pose.position.x',
                'pose.position.y',
                'pose.position.z',
                'pose.orientation.x',
                'pose.orientation.y',
                'pose.orientation.z',
                'pose.orientation.w',
            ],
            inplace=True,
        )
        
    def rename_fields(self) -> None:
        """This replaces the UWB fields based on the mapping in the config files. 
        """
        for key in self.uwb_fields.keys():
            if key not in self.df_uwb.columns:
                value = self.uwb_fields[key]
                self.df_uwb[key] = self.df_uwb[value]
                self.df_uwb.drop(columns=[value], inplace=True)

        if self.passive_listening:        
            for key in self.passive_fields.keys():
                if key not in self.df_passive.columns:
                    value = self.passive_fields[key]
                    self.df_passive[key] = self.df_passive[value]
                    self.df_passive.drop(columns=[value], inplace=True)
        
class RosMachine(Machine):
    """A class to handle ROS machines recording pose and UWB data in rosbags.

    All attributes from the Machine class are inherited.

    Attributes
    ----------
    df_pose: pd.DataFrame
        A dataframe for the ground-truth measurements with the following columns.
        'time': float
            ROS timestamp of the measurement, in seconds.
        r_iw_a: list
            The position of the machine relative to the motion-capture system's reference point.
        q_ai: list
            Quaternion parametrization of the robot's attitude, using the [x,y,z,w] convention.
    df_uwb: pd.DataFrame
        A dataframe for the UWB measurements with the following columns.
        'time': float
            ROS timestamp of the measurement, in seconds.
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
    # TODO: add df_passive documentation.

    Examples
    --------
    # *** Example reading a single machine ***
    config_file = 'config/ifo_3_drones_rosbag.config'

    parser = ConfigParser(interpolation=ExtendedInterpolation())
    parser.read(config_file)

    machine_num = 0
    machine = RosMachine(parser, i)

    # *** Example reading multiple machines ***
    config_file = 'config/ifo_3_drones_rosbag.config'

    parser = ConfigParser(interpolation=ExtendedInterpolation())
    parser.read(config_file)

    machines = {}
    for i,machine in enumerate(parser['MACHINES']):
        machine_id = parser['MACHINES'][machine]
        machines[machine_id] = RosMachine(parser, i)
    """
    def __init__(
        self,
        configs,
        id,
        meas_at_target = False,
    ) -> None:
        """Constructor

        Parameters
        ----------
        configs : ConfigParser
            A ConfigParser object where the config file is parsed.
        id : int
            The ID of the machine, as referenced in the config file.
        meas_at_target : bool, optional
            Keep range measurements when initiator is not on this machine,
            by default False
        """
        super().__init__(configs, id)

        # Read the ground-truth and UWB data
        self.df_pose, self.df_uwb, self.df_passive = self._read_data()
        self.rename_fields()
        
        # pre-process data
        self.merge_pose_data()
        self.convert_uwb_timestamps()
        self._process_ros_timestamps()
        
        # Drop duplicate measurements. 
        if not meas_at_target:
            self.drop_target_meas()
            
    def _read_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Read rosbags and extract pose and UWB information.

        Returns
        -------
        df_pose: pd.DataFrame
            A dataframe for the ground-truth measurements.
        df_uwb: pd.DataFrame
            A dataframe for the UWB measurements.
        """
        # Pose data 
        bag = bagreader(self.pose_path)
        pose_data = bag.message_by_topic(self.pose_topic)
        df_pose = pd.read_csv(pose_data)
        
        # UWB data
        if self.uwb_path != self.pose_path:
            bag = bagreader(self.uwb_path)
        uwb_data = bag.message_by_topic(self.uwb_topic)
        df_uwb = pd.read_csv(uwb_data)

        # Passive data
        if self.passive_listening:
            passive_data = bag.message_by_topic(self.passive_topic)
            df_passive = pd.read_csv(passive_data)
        else:
            df_passive = []

        return df_pose, df_uwb, df_passive
        
    def _process_ros_timestamps(self) -> None:
        """Process ROS timestamps.
        """
        self._merge_timestamp_headers(self.df_pose)
        self._merge_timestamp_headers(self.df_uwb)

        if self.passive_listening:
            self._merge_timestamp_headers(self.df_passive)
        
    @staticmethod
    def _merge_timestamp_headers(df) -> None:
        """Merges the ROS timestamp headers into one column that is both unwrapped
        and with nanosecond accuracy.

        The new column has the header "time, which merges and replaces the headers 
        "header.stamp.secs" and "header.stamp.nsecs".

        Parameters
        ----------
        df : pd.dataframe
            Any Pandas dataframe with the following two columns.
            "header.stamp.secs": int
                The ROS timestamps, to the order of seconds.
            "header.stamp.secs": int
                The ROS timestamps, the nanoseconds portion.
        """
        df.drop(columns=["Time"],inplace=True)
        df['time'] = df["header.stamp.secs"] + df["header.stamp.nsecs"]/1e9
        df.drop(columns=["header.stamp.secs", "header.stamp.nsecs"],inplace=True)

class CsvMachine(Machine):
    """TODO: CsvMachine
    """
    def __init__(
        self,
        configs,
        id,
        ts_to_ns = 1,
        meas_at_target=False,
    ) -> None:
        super().__init__(configs,
                         id,
                         ts_to_ns,
                         meas_at_target,
                         is_ros = False)

        self.df_pose, self.df_uwb, self.df_passive = self._read_data()
        
        self.convert_uwb_timestamps(ts_to_ns)
        # self._process_ros_timestamps()
        
        if not meas_at_target:
            self.drop_target_meas()
        
    def _read_data(self) -> None:
        raise NotImplementedError()