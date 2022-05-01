'''
Synthetic data. Scenario where the current device is connected to Tag 1, and
a Raspberry Pi is connected to Tag 2, and there is a rogue Tag 3.
'''

import pandas as pd
import rospy
import rosbag
from uwb_ros.msg import RangeStamped
from geometry_msgs.msg import PoseStamped

df = pd.read_csv('datasets/synthetic/synthetic.csv')

to_ns = 1e9 * (1.0 / 499.2e6 / 128.0) # DW time unit to nanoseconds

with rosbag.Bag('datasets/synthetic/ros_bags/synthetic1.bag', 'w') as bag:
    for row in range(df.shape[0]):
        ############## UWB 1 ##############
        timestamp = rospy.Time.from_sec(df['timestamp_1'][row]/1e9)
        uwb_msg = RangeStamped()
        uwb_msg.header.stamp = timestamp

        # Populate the data elements for UWB
        uwb_msg.to_id = df['to_id_1'][row]
        uwb_msg.from_id = df['from_id_1'][row]
        uwb_msg.range = df['range_1'][row]
        uwb_msg.tx1 = df['tx1_1'][row]
        uwb_msg.rx1 = df['rx1_1'][row]
        uwb_msg.tx2 = df['tx2_1'][row]
        uwb_msg.rx2 = df['rx2_1'][row]
        uwb_msg.tx3 = df['tx3_1'][row]
        uwb_msg.rx3 = df['rx3_1'][row]
        uwb_msg.power1 = df['Pr1_1'][row]
        uwb_msg.power2 = df['Pr2_1'][row]

        bag.write("/uwb/range", uwb_msg, timestamp)

        ############## UWB 2 ##############
        timestamp = rospy.Time.from_sec(df['timestamp_2'][row]/1e9)
        uwb_msg = RangeStamped()
        uwb_msg.header.stamp = timestamp

        # Populate the data elements for UWB
        uwb_msg.to_id = df['to_id_2'][row]
        uwb_msg.from_id = df['from_id_2'][row]
        uwb_msg.range = df['range_2'][row]
        uwb_msg.tx1 = df['tx1_2'][row]
        uwb_msg.rx1 = df['rx1_2'][row]
        uwb_msg.tx2 = df['tx2_2'][row]
        uwb_msg.rx2 = df['rx2_2'][row]
        uwb_msg.tx3 = df['tx3_2'][row]
        uwb_msg.rx3 = df['rx3_2'][row]
        uwb_msg.power1 = df['Pr1_2'][row]
        uwb_msg.power2 = df['Pr2_2'][row]

        bag.write("/rpi/uwb/range", uwb_msg, timestamp)

        ############## Mocap Data 1 ##############
        timestamp = rospy.Time.from_sec(df['timestamp_mocap_1'][row]/1e9)
        mocap_msg = PoseStamped()
        mocap_msg.header.stamp = timestamp

        # Populate the data elements for Mocap
        mocap_msg.pose.position.x = df['position_x_1'][row]
        mocap_msg.pose.position.y = df['position_y_1'][row]
        mocap_msg.pose.position.z = df['position_z_1'][row]
        mocap_msg.pose.orientation.x = df['orientation_x_1'][row]
        mocap_msg.pose.orientation.y = df['orientation_y_1'][row]
        mocap_msg.pose.orientation.z = df['orientation_z_1'][row]
        mocap_msg.pose.orientation.w = df['orientation_w_1'][row]

        # Populate the data elements for Mocap
        bag.write("/vrpn_client_node/tripod1/pose", mocap_msg, timestamp)


        ############## Mocap Data 2 ##############
        timestamp = rospy.Time.from_sec(df['timestamp_mocap_2'][row]/1e9)
        mocap_msg = PoseStamped()
        mocap_msg.header.stamp = timestamp

        # Populate the data elements for Mocap
        mocap_msg.pose.position.x = df['position_x_2'][row]
        mocap_msg.pose.position.y = df['position_y_2'][row]
        mocap_msg.pose.position.z = df['position_z_2'][row]
        mocap_msg.pose.orientation.x = df['orientation_x_2'][row]
        mocap_msg.pose.orientation.y = df['orientation_y_2'][row]
        mocap_msg.pose.orientation.z = df['orientation_z_2'][row]
        mocap_msg.pose.orientation.w = df['orientation_w_2'][row]

        # Populate the data elements for Mocap
        bag.write("/vrpn_client_node/tripod2/pose", mocap_msg, timestamp)


        ############## Mocap Data 3 ##############
        timestamp = rospy.Time.from_sec(df['timestamp_mocap_3'][row]/1e9)
        mocap_msg = PoseStamped()
        mocap_msg.header.stamp = timestamp

        # Populate the data elements for Mocap
        mocap_msg.pose.position.x = df['position_x_3'][row]
        mocap_msg.pose.position.y = df['position_y_3'][row]
        mocap_msg.pose.position.z = df['position_z_3'][row]
        mocap_msg.pose.orientation.x = df['orientation_x_3'][row]
        mocap_msg.pose.orientation.y = df['orientation_y_3'][row]
        mocap_msg.pose.orientation.z = df['orientation_z_3'][row]
        mocap_msg.pose.orientation.w = df['orientation_w_3'][row]

        # Populate the data elements for Mocap
        bag.write("/vrpn_client_node/tripod3/pose", mocap_msg, timestamp)