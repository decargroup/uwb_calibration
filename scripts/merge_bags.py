import os
import rosbag

directory_path = 'datasets/2022_06_15/bias_calibration/'

new_bag = rosbag.Bag(directory_path+"merged.bag", 'w')

for file in os.listdir(directory_path):
    filename = os.fsdecode(file)
    if filename == "merged.bag":
        continue

    bag = rosbag.Bag(directory_path+filename)
    
    for topic, msg, t in bag.read_messages():
        new_bag.write(topic, msg, t)

new_bag.reindex=()