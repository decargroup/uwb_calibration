from pyuwbcalib.postprocess import PostProcess

data_obj = PostProcess(folder_prefix="datasets/2022_04_20/",
                       file_prefix="formation",
                       num_of_recordings=14,
                       num_meas=-1)

initiator_id = 2
target_id = 3

data_obj.visualize_raw_data(pair=(initiator_id,target_id))