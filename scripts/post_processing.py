from pyuwbcalib.postprocess import PostProcess
from pyuwbcalib.uwbcalibrate import UwbCalibrate

data_obj = PostProcess(folder_prefix="datasets/2022_04_20/",
                       file_prefix="formation",
                       num_of_recordings=1,
                       num_meas=-1)

initiator_id = 2
target_id = 3

data_obj.visualize_raw_data(pair=(initiator_id,target_id))

# TODO: Surely there is a better way to do this??
# calib_obj = UwbCalibrate(data_obj)

# print(data_obj.mean_gt_distance)
# print(data_obj.mean_range_meas)
