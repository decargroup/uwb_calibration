# TODO: 1) Should have a super class for these two classes since they'll share things
#          like __init__, timestamp computation, etc
from bagpy import bagreader
import pandas as pd

class Machine(object):
    def __init__(
                    self,
                    configs,
                    id,
                ):
        
        pose_dir = configs['POSE_PATH']['directory']
        self.pose_path = pose_dir + configs['POSE_PATH'][str(id)]

        uwb_dir = configs['UWB_PATH']['directory']
        self.uwb_path = uwb_dir + configs['UWB_PATH'][str(id)]
        
        self.machine_id = configs['MACHINES'][str(id)]
        self.tag_ids = eval(configs['TAGS'][str(id)])
        self.moment_arms = [configs['MOMENT_ARMS'][str(tag)] for tag in self.tag_ids]

        self.pose_topic = configs['POSE_TOPIC'][str(id)]

        self.uwb_topic = configs['UWB_TOPIC'][str(id)]
        self.uwb_fields = [configs['UWB_MESSAGE'][key] for key in configs['UWB_MESSAGE'].keys()]

        self.ds_twr = 'tx3' in self.uwb_fields
        self.fpp_exists = 'fpp1' in self.uwb_fields
        self.rxp_exists = 'rxp1' in self.uwb_fields
        self.std_exists = 'std1' in self.uwb_fields
        
    def _convert_uwb_timestamps(self, ts_to_ns):
        self.df_uwb['tx1'] *= ts_to_ns
        self.df_uwb['rx1'] *= ts_to_ns
        self.df_uwb['tx2'] *= ts_to_ns
        self.df_uwb['rx2'] *= ts_to_ns
        
        if self.ds_twr:
            self.df_uwb['tx3'] *= ts_to_ns
            self.df_uwb['rx3'] *= ts_to_ns
            
    def _drop_target_meas(self):
        bool1 = (self.df_uwb['from_id'] != self.tag_ids[0])
        bool2 = (self.df_uwb['from_id'] != self.tag_ids[1])
        
        self.df_uwb.drop(self.df_uwb[bool1 & bool2].index, inplace=True)
        
class RosMachine(Machine):
    def __init__(
                    self,
                    configs,
                    id,
                    ts_to_ns = 1,
                    meas_at_target = False,
                ):
        super().__init__(configs, id)

        self.df_pose, self.df_uwb = self._read_data()
        
        self._convert_uwb_timestamps(ts_to_ns)
        self._process_ros_timestamps()
        
        if not meas_at_target:
            self._drop_target_meas()
            
    def _read_data(self):
        # Pose data 
        bag = bagreader(self.pose_path)
        pose_data = bag.message_by_topic(self.pose_topic)
        df_pose = pd.read_csv(pose_data)
        
        if self.uwb_path != self.pose_path:
            bag = bagreader(self.uwb_path)
        uwb_data = bag.message_by_topic(self.uwb_topic)
        df_uwb = pd.read_csv(uwb_data)
        
        return df_pose, df_uwb
        
    def _process_ros_timestamps(self):
        self._merge_timestamp_headers(self.df_pose)
        self._merge_timestamp_headers(self.df_uwb)
        
    @staticmethod
    def _merge_timestamp_headers(df):
        df.drop(columns=["Time"],inplace=True)
        df['time'] = df["header.stamp.secs"] + df["header.stamp.nsecs"]/1e9
        df.drop(columns=["header.stamp.secs", "header.stamp.nsecs"],inplace=True)

class CsvMachine(Machine):
    def __init__(
                    self,
                    configs,
                    id,
                    ts_to_ns = 1,
                    meas_at_target=False,
                ):
        super().__init__(configs,
                         id,
                         ts_to_ns,
                         meas_at_target)

        self.df_pose, self.df_uwb = self._read_data()
        
        self._convert_uwb_timestamps(ts_to_ns)
        # self._process_ros_timestamps()
        
        if not meas_at_target:
            self._drop_target_meas()
        
    def _read_data(self):
        # TODO: implement reading data from csv files. 
        raise Exception("To be implemented.")