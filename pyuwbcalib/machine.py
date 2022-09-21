from bagpy import bagreader
import numpy as np
import pandas as pd

class Machine(object):
    def __init__(
        self,
        configs,
        id,
    ) -> None:
        """_summary_

        Parameters
        ----------
        configs : _type_
            _description_
        id : _type_
            _description_
        """
        
        self.max_ts_value = eval(configs['PARAMS']['max_ts_value'])
        self.ts_to_ns = eval(configs['PARAMS']['ts_to_ns'])
        
        self.ds_twr = eval(configs['PARAMS']['ds_twr'])
        self.fpp_exists = eval(configs['PARAMS']['fpp_exists'])
        self.rxp_exists = eval(configs['PARAMS']['rxp_exists'])
        self.std_exists = eval(configs['PARAMS']['std_exists'])
        
        pose_dir = configs['POSE_PATH']['directory']
        self.pose_path = pose_dir + configs['POSE_PATH'][str(id)]

        uwb_dir = configs['UWB_PATH']['directory']
        self.uwb_path = uwb_dir + configs['UWB_PATH'][str(id)]
        
        self.machine_id = configs['MACHINES'][str(id)]
        self.tag_ids = eval(configs['TAGS'][str(id)])
        self.moment_arms = {tag:eval(configs['MOMENT_ARMS'][str(tag)]) for tag in self.tag_ids}

        self.pose_topic = configs['POSE_TOPIC'][str(id)]

        self.uwb_topic = configs['UWB_TOPIC'][str(id)]
        self.uwb_fields = [configs['UWB_MESSAGE'][key] for key in configs['UWB_MESSAGE'].keys()]
        
    def convert_uwb_timestamps(self, ts_to_ns) -> None:
        """_summary_

        Parameters
        ----------
        ts_to_ns : _type_
            _description_
        """
        self.df_uwb['tx1'] *= ts_to_ns
        self.df_uwb['rx1'] *= ts_to_ns
        self.df_uwb['tx2'] *= ts_to_ns
        self.df_uwb['rx2'] *= ts_to_ns
        
        if self.ds_twr:
            self.df_uwb['tx3'] *= ts_to_ns
            self.df_uwb['rx3'] *= ts_to_ns
            
    def drop_target_meas(self) -> None:
        """_summary_
        """
        bool1 = (self.df_uwb['from_id'] != self.tag_ids[0])
        bool2 = (self.df_uwb['from_id'] != self.tag_ids[1])
        
        self.df_uwb.drop(self.df_uwb[bool1 & bool2].index, inplace=True)
        self.df_uwb.reset_index(inplace=True, drop=True)
        
    def merge_pose_data(self) -> None:
        """_summary_
        """
        self.df_pose['r_iw_a'] = list(np.array((
                                                self.df_pose['pose.position.x'],
                                                self.df_pose['pose.position.y'],
                                                self.df_pose['pose.position.z'],
                                              )).T)
        self.df_pose['q_ai'] = list(np.array((
                                                self.df_pose['pose.orientation.x'],
                                                self.df_pose['pose.orientation.y'],
                                                self.df_pose['pose.orientation.z'],
                                                self.df_pose['pose.orientation.w'],
                                            )).T)
        
        self.df_pose.drop(columns=[
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
        
class RosMachine(Machine):
    def __init__(
        self,
        configs,
        id,
        ts_to_ns = 1,
        meas_at_target = False,
    ) -> None:
        """_summary_

        Parameters
        ----------
        configs : _type_
            _description_
        id : _type_
            _description_
        ts_to_ns : int, optional
            _description_, by default 1
        meas_at_target : bool, optional
            _description_, by default False
        """
        super().__init__(configs, id)

        self.df_pose, self.df_uwb = self._read_data()
        
        self.merge_pose_data()
        self.convert_uwb_timestamps(ts_to_ns)
        self._process_ros_timestamps()
        
        if not meas_at_target:
            self.drop_target_meas()
            
    def _read_data(self) -> tuple(pd.DataFrame, pd.DataFrame):
        """_summary_

        Returns
        -------
        _type_
            _description_
        """
        # Pose data 
        bag = bagreader(self.pose_path)
        pose_data = bag.message_by_topic(self.pose_topic)
        df_pose = pd.read_csv(pose_data)
        
        if self.uwb_path != self.pose_path:
            bag = bagreader(self.uwb_path)
        uwb_data = bag.message_by_topic(self.uwb_topic)
        df_uwb = pd.read_csv(uwb_data)
        
        return df_pose, df_uwb
        
    def _process_ros_timestamps(self) -> None:
        """_summary_
        """
        self._merge_timestamp_headers(self.df_pose)
        self._merge_timestamp_headers(self.df_uwb)
        
    @staticmethod
    def _merge_timestamp_headers(df) -> None:
        """_summary_

        Parameters
        ----------
        df : _type_
            _description_
        """
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
    ) -> None:
        """_summary_

        Parameters
        ----------
        configs : _type_
            _description_
        id : _type_
            _description_
        ts_to_ns : int, optional
            _description_, by default 1
        meas_at_target : bool, optional
            _description_, by default False
        """
        super().__init__(configs,
                         id,
                         ts_to_ns,
                         meas_at_target)

        self.df_pose, self.df_uwb = self._read_data()
        
        self.convert_uwb_timestamps(ts_to_ns)
        # self._process_ros_timestamps()
        
        if not meas_at_target:
            self.drop_target_meas()
        
    def _read_data(self) -> None:
        """_summary_

        Raises
        ------
        Exception
            _description_
        """
        # TODO: implement reading data from csv files. 
        raise Exception("To be implemented.")