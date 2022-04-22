from pyuwbcalib.postprocess import PostProcess

data_obj = PostProcess(folder_prefix="datasets/2022_04_20/",
                       file_prefix="changing_attitude_formation",
                       num_of_formations=1,
                       num_meas=-1)

data_obj.visualize_data(pair=(1,3))

# print(data_obj.mean_gt_distance)
# print(data_obj.mean_range_meas)

# data_obj.setup_formatted_files()