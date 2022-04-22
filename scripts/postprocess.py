from pyuwbcalib.postprocess import PostProcess

data_obj = PostProcess(file_prefix="datasets/2022_04_20/", num_of_formations=14, num_meas=500)

data_obj.visualize_data()

# print(data_obj.mean_gt_distance)
# print(data_obj.mean_range_meas)

# data_obj.setup_formatted_files()