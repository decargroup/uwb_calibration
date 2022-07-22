# %%
from pyuwbcalib.postprocess import PostProcess

raw_obj = PostProcess(folder_prefix="datasets/2022_05_02/",
                      file_prefix="test",
                      num_of_recordings=1,
                      num_meas=-1,
                      tag_ids=[4,1,3])

# %%
initiator_id = 1
target_id = 3

raw_obj.visualize_raw_data(pair=(initiator_id,target_id))
# %%
