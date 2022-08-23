# %%
# The purpose of this script is to check the observability of the dynamic antenna delay calibration batch algorithm

import numpy as np
import control 
import scipy

T = 1e-2
K = 0.99
n = 3 # number of tags
m = n-1
N = n*2 + m*2

A0 = np.zeros((2*n, N))
A1 = np.hstack((np.zeros((m, 2*n)),np.eye(m),np.eye(m)*T))
A2 = np.hstack((np.zeros((m, 2*n+m)),np.eye(m)))
A = np.vstack((A0,A1,A2))

n_twr = n*(n-1)//2
n_passive = n*(n-1)
n_meas = n_twr + n_passive
C = np.zeros((n_twr+n_passive, N))

row = 0
tag_i = 0
tag_j = 1
for _ in range(n_twr):
    C[row,2*tag_i:2*tag_i+2] = [0.5, 0.5]
    C[row,2*tag_j:2*tag_j+2] = [-0.5*K, -0.5*K]

    row += 1
    tag_j += 1

    if tag_j == n:
        tag_i += 1
        tag_j = tag_i + 1

tag_i = 0
tag_j = 1
for i in range(n_passive):
    C[row,2*tag_i:2*tag_i+2] = [1, 0]
    C[row,2*tag_j:2*tag_j+2] = [0, 1]

    if tag_i != 0 and tag_j != 0:
        C[row,2*n+tag_i-1] = -1
        C[row,2*n+tag_j-1] = 1
    elif tag_i == 0:
        C[row,2*n+tag_j-1] = 1
    elif tag_j == 0:
        C[row,2*n+tag_i-1] = -1

    row += 1
    tag_j += 1

    if tag_i == tag_j:
        tag_j += 1
    if tag_j == n:
        tag_i += 1
        tag_j = 0

# C = np.vstack((C,[0,0,0,0,0,0,0,0,0,0]))
O = control.obsv(A,C)
print(np.linalg.matrix_rank(O))
print(scipy.linalg.null_space(O))

# %%
