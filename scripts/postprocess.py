from pyuwbcalib.postprocess import PostProcess

file_prefix = "datasets/2022_02_23/raw"
num_of_formations = 7
board_ids = [1,2,3]
twr_type = 2

data_obj = PostProcess(file_prefix, num_of_formations, board_ids, twr_type)

data_obj.manually_change_gt_order(4,0,1)
data_obj.manually_change_gt_order(4,0,2)

print(data_obj.mean_gt_distance)
print(data_obj.mean_range_meas)

data_obj.setup_formatted_files()