# %%
from pyuwbcalib.postprocess import load
import numpy as np
from scipy.linalg import block_diag

c = 299702547
data = load()

tags = [tag for tag in data.moment_arms]

n = len(data.df)*0+2000
n_tags = 6
n_clock = 2*(n_tags-1)
n_delay = 2*n_tags 
n_states = n_clock + n_delay

x = np.zeros(n_states)
P = np.eye(n_states)
P[0:n_clock,0:n_clock] \
                        = np.eye(n_clock)*1e15

Q_tau = np.eye(n_tags-1)*1
Q_gamma = np.eye(n_tags-1)*1
Q_i = block_diag(Q_tau,Q_gamma)
Q = np.zeros((n_states,n_states))
Q[0:n_clock,0:n_clock] = Q_i
R_i = 1
dt = np.hstack((0,np.diff(data.df["time"])[0:n]))

for i in range(n):
    print(i)
    # Prediction
    if i:
        dt_iter = dt[i]
        A = np.eye(n_states)
        A[0:n_tags-1, n_tags-1:2*(n_tags-1)] = np.eye(n_tags-1)*dt_iter
        x = A@x
        P = A@P@A.T + dt_iter * Q.T
        P = 0.5*(P + P.T)
        
    # Correction
    df_iter = data.df.iloc[i]
    df_iter_passive = data.df_passive.loc[data.df_passive["idx"]==i]
    m = len(df_iter_passive)
    
    y = np.zeros(2+m*3)
    y_check = np.zeros(2+m*3)
    C = np.zeros((2+m*3, n_states))
    R = np.eye(2+m*3)*R_i
    idx = 0
    
    del_41 = df_iter['del_41']
    del_32 = df_iter['del_32']
    del_21 = df_iter['del_21']
    del_43 = df_iter['del_43']
    del_64 = df_iter['del_64']
    del_53 = df_iter['del_53']
    
    from_id = int(df_iter["from_id"])
    from_idx = int(np.where(np.array(tags)==from_id)[0])
    
    to_id = int(df_iter["to_id"])
    to_idx = int(np.where(np.array(tags)==to_id)[0])
    
    tf_ij = df_iter["gt_range"]/c*1e9
    
    K = del_64 / del_53
    
    # Measurement 1:
    y[idx] = 2*tf_ij - del_41 + K*del_32
    y_check[idx] = x[n_clock+2*from_idx] + x[n_clock+2*from_idx+1] \
                   + K * (x[n_clock+2*to_idx] + x[n_clock+2*to_idx+1])
    C[idx,n_clock+2*from_idx] = 1
    C[idx,n_clock+2*from_idx+1] = 1
    C[idx,n_clock+2*to_idx] = K
    C[idx,n_clock+2*to_idx+1] = K
    idx += 1
    
    # Measurement 2:
    # y[idx] = t1 + t4 - t2 - K*t3
    y[idx] = del_43 - del_21
    C[idx,n_clock+2*from_idx] = 1
    C[idx,n_clock+2*from_idx+1] = -1
    C[idx,n_clock+2*to_idx] = -K
    C[idx,n_clock+2*to_idx+1] = 1
    if from_idx==0:
        y_check[idx] = x[n_clock+2*from_idx] - x[n_clock+2*from_idx+1] \
                   - K * x[n_clock+2*to_idx] + x[n_clock+2*to_idx+1] \
                   - 2*x[to_idx-1]
        
        C[idx,to_idx-1] = -2
    elif to_idx==0:
        y_check[idx] = x[n_clock+2*from_idx] - x[n_clock+2*from_idx+1] \
                   - K * x[n_clock+2*to_idx] + x[n_clock+2*to_idx+1] \
                   + 2*x[from_idx-1]
        C[idx,from_idx-1] = 2
    else:        
        y_check[idx] = x[n_clock+2*from_idx] - x[n_clock+2*from_idx+1] \
                   - K * x[n_clock+2*to_idx] + x[n_clock+2*to_idx+1] \
                   - 2*x[to_idx-1] + 2*x[from_idx-1]
        C[idx,from_idx-1] = 2
        C[idx,to_idx-1] = -2
    idx += 1
        
    for _,row in df_iter_passive.iterrows():
        del_71 = row['del_71']
        del_83 = row['del_83']
        del_95 = row['del_95']
        
        passive_id = int(row["my_id"])
        passive_idx = int(np.where(np.array(tags)==passive_id)[0])
        
        tf_pi = data.compute_distance(df_iter, [passive_id, from_id])/c*1e9
        tf_pj = data.compute_distance(df_iter, [passive_id, to_id])/c*1e9
        
        # Measurement 3i:
        y[idx] = tf_pi - del_71
        C[idx,n_clock+2*from_idx] = 1
        C[idx,n_clock+2*passive_idx+1] = 1
        if from_idx==0:
            y_check[idx] = x[n_clock+2*passive_idx+1] \
                           + x[n_clock+2*from_idx] \
                           - x[passive_idx-1]
            C[idx,passive_idx-1] = -1
        elif passive_idx==0:
            y_check[idx] = x[n_clock+2*passive_idx+1] \
                           + x[n_clock+2*from_idx] \
                           + x[from_idx-1]
            C[idx,from_idx-1] = 1
        else:        
            y_check[idx] = x[n_clock+2*passive_idx+1] \
                           + x[n_clock+2*from_idx] \
                           - x[passive_idx-1] + x[from_idx-1]
            C[idx,passive_idx-1] = -1
            C[idx,from_idx-1] = 1
        idx += 1
        
        # Measurement 4i:
        y[idx] = tf_pj - del_83
        C[idx,n_clock+2*to_idx] = 1
        C[idx,n_clock+2*passive_idx+1] = 1
        if to_idx==0:
            y_check[idx] = x[n_clock+2*passive_idx+1] \
                           + x[n_clock+2*to_idx] \
                           - x[passive_idx-1]
            C[idx,passive_idx-1] = -1
        elif passive_idx==0:
            y_check[idx] = x[n_clock+2*passive_idx+1] \
                           + x[n_clock+2*to_idx] \
                           + x[to_idx-1]
            C[idx,to_idx-1] = 1
        else:        
            y_check[idx] = x[n_clock+2*passive_idx+1] \
                           + x[n_clock+2*to_idx] \
                           - x[passive_idx-1] + x[to_idx-1]
            C[idx,passive_idx-1] = -1
            C[idx,to_idx-1] = 1
        idx += 1
        
        # Measurement 5i:
        y[idx] = tf_pj - del_95
        C[idx,n_clock+2*to_idx] = 1
        C[idx,n_clock+2*passive_idx+1] = 1
        if to_idx==0:
            y_check[idx] = x[n_clock+2*passive_idx+1] \
                           + x[n_clock+2*to_idx] \
                           - x[passive_idx-1]
            C[idx,passive_idx-1] = -1
        elif passive_idx==0:
            y_check[idx] = x[n_clock+2*passive_idx+1] \
                           + x[n_clock+2*to_idx] \
                           + x[to_idx-1]
            C[idx,to_idx-1] = 1
        else:        
            y_check[idx] = x[n_clock+2*passive_idx+1] \
                           + x[n_clock+2*to_idx] \
                           - x[passive_idx-1] + x[to_idx-1]
            C[idx,passive_idx-1] = -1
            C[idx,to_idx-1] = 1
        idx += 1
    
    S = (C @ P @ C.T + R)
    K = P @ C.T @ np.linalg.inv(S)
    
    innov = y - y_check
    x = x + (K @ (innov)).flatten()
    P = (np.eye(n_states) - K@C) @ P 
    P = 0.5*(P + P.T)
    
    # NOW THAT I THINK ABOUT IT, SKEW AFFECTS PASSIVE LIStENING MEASUREMENTS TOO, t3 - t7 and
    # t5 - t9 will not be the same...
    
    # PLAN: FIX THE PAPER, HAVE TWO VERSIONS ONE WITH AND ONE WITHOUT OPTIMAL DELAY,
    #       THEN SEND BOTH AND LET THEM DECIDE
    
# %%
