from pyuwbcalib.postprocess import PostProcess

file_prefix = "datasets/2022_04_20/"
num_of_formations = 10
board_ids = [1,2,3]
twr_type = 0

data_obj = PostProcess(file_prefix, num_of_formations, board_ids, twr_type)

# data_obj.visualize_data()

print(data_obj.mean_gt_distance)
print(data_obj.mean_range_meas)

data_obj.setup_formatted_files()