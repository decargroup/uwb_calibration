import os
import rosbag

directory_path = 'datasets/2022_08_03/big_merge/'

new_bag = rosbag.Bag(directory_path+"merged.bag", 'w')

# Look for all rosbags in directory
for file in os.listdir(directory_path):
    filename = os.fsdecode(file)
    if filename == "merged.bag" or filename[-3:] != "bag":
        continue

    bag = rosbag.Bag(directory_path+filename)
    
    for topic, msg, t in bag.read_messages():
        new_bag.write(topic, msg, t)

new_bag.reindex=()