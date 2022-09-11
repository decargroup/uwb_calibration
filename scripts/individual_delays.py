# %%
from pyuwbcalib.postprocess import load
import numpy as np

c = 299702547
data = load()

tags = [tag for tag in data.moment_arms]

n = len(data.df)*0+1000
n_tags = 6

A = np.diag(np.ones(2+2*(n_tags-1)*n+2*n_tags))
b_A = np.zeros((2+2*(n_tags-1)*n+2*n_tags, 1))

x0 = np.array((0,0))

# %%
# A matrix
dt = np.diff(data.df["time"])[0:n]
dt_repeated = np.repeat(dt,5)
A_dt = np.diag(dt_repeated)
A[1:n*(n_tags-1)+1, n*(n_tags-1)+1:-(2*n_tags)-1] += -A_dt

# %%
# Find total number of measurements:
meas_num = 0
for i in range(n):
    df_iter_passive = data.df_passive.loc[data.df_passive["idx"]==i]
    m = len(df_iter_passive)
    meas_num += 2 + 3*m
C = np.zeros((meas_num, 2+2*(n_tags-1)*n+2*n_tags))
b_C = np.zeros((meas_num, 1))

# C matrix
idx = 0
for i in range(n):
    df_iter_passive = data.df_passive.loc[data.df_passive["idx"]==i]
    m = len(df_iter_passive)
    print(i)
    df_iter = data.df.iloc[i]
    t1 = df_iter["tx1"]
    t2 = df_iter["rx1"]
    t3 = df_iter["tx2"]
    t4 = df_iter["rx2"]
    t5 = df_iter["tx3"]
    t6 = df_iter["rx3"]
    
    from_id = int(df_iter["from_id"])
    from_idx = int(np.where(np.array(tags)==from_id)[0])
    
    to_id = int(df_iter["to_id"])
    to_idx = int(np.where(np.array(tags)==to_id)[0])
    
    tf_ij = df_iter["gt_range"]/c
    
    K = (t6 - t4) / (t5 - t3)
        
    C[idx,2*(n_tags-1)*n+2*from_idx:2*(n_tags-1)*n+2*from_idx+2] += 1
    C[idx,2*(n_tags-1)*n+2*to_idx:2*(n_tags-1)*n+2*to_idx+2] += K
    b_C[idx] = 2*tf_ij - (t4 - t1) + K*(t3-t2)
    idx += 1
    
    C[idx,2*(n_tags-1)*n+2*from_idx] += 1
    C[idx,2*(n_tags-1)*n+2*from_idx+1] += -1
    C[idx,2*(n_tags-1)*n+2*to_idx] += -K
    C[idx,2*(n_tags-1)*n+2*to_idx+1] += 1
    if from_idx==0:
        C[idx,i*n_tags+to_idx-1] += -2
    elif to_idx==0:
        C[idx,i*n_tags+from_idx-1] += 2
    else:        
        C[idx,i*n_tags+to_idx-1] += -2
        C[idx,i*n_tags+from_idx-1] += 2
    b_C[idx] = t1 + t4 - t2 - K*t3
    idx += 1
    
    for _,row in df_iter_passive.iterrows():
        t7 = row["rx1"]
        t8 = row["rx2"]
        t9 = row["rx3"]
        
        passive_id = int(row["my_id"])
        passive_idx = int(np.where(np.array(tags)==passive_id)[0])
        
        tf_pi = data.compute_distance(df_iter, [passive_id, from_id])/c
        tf_pj = data.compute_distance(df_iter, [passive_id, to_id])/c
        
        C[idx,2*(n_tags-1)*n+2*from_idx] += 1
        C[idx,2*(n_tags-1)*n+2*passive_idx+1] += 1
        if from_idx==0:
            C[idx,i*n_tags+passive_idx-1] += -1
        elif passive_idx==0:
            C[idx,i*n_tags+from_idx-1] += 1
        else:        
            C[idx,i*n_tags+passive_idx-1] += -1
            C[idx,i*n_tags+from_idx-1] += 1
        b_C[idx] = tf_pi + t1 - t7
        idx += 1
        
        C[idx,2*(n_tags-1)*n+2*to_idx] += 1
        C[idx,2*(n_tags-1)*n+2*passive_idx+1] += 1
        if to_idx==0:
            C[idx,i*n_tags+passive_idx-1] += -1
        elif passive_idx==0:
            C[idx,i*n_tags+to_idx-1] += 1
        else:        
            C[idx,i*n_tags+passive_idx-1] += -1
            C[idx,i*n_tags+to_idx-1] += 1
        b_C[idx] = tf_pj + t3 - t8
        idx += 1
        
        C[idx,2*(n_tags-1)*n+2*from_idx] += 1
        C[idx,2*(n_tags-1)*n+2*passive_idx+1] += 1
        if from_idx==0:
            C[idx,i*n_tags+passive_idx-1] += -1
        elif passive_idx==0:
            C[idx,i*n_tags+from_idx-1] += 1
        else:        
            C[idx,i*n_tags+passive_idx-1] += -1
            C[idx,i*n_tags+from_idx-1] += 1
        b_C[idx] = tf_pi + t5 - t9
        idx += 1

    
# %%
