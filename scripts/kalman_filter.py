import numpy as np
from pyuwbcalib.uwbcalibrate import UwbCalibrate
import matplotlib.pyplot as plt
import scipy.linalg

## PARAMETERS -----------------------------------------------------
c = 299702547 # speed of light in m/s

R = 3 # variance in ns^2
P0_delta = 1e9
P0_gamma = 10000
Q_delta = (2*1e4)**2
Q_gamma = 25.3**2

## GET DATA -------------------------------------------------------
x = UwbCalibrate("datasets/2022_02_23/formatted_ID1_twr2.csv",
                 "datasets/2022_02_23/formatted_ID2_twr2.csv",
                 [1,2,3],average=False,thresh=1e6)

id1 = 2
id2 = 3

meas_old = x.compute_range_meas(id1,id2)

delays = x.calibrate_antennas()
print(delays)
x.correct_antenna_delay(1, delays['Module 1'])
x.correct_antenna_delay(2, delays['Module 2'])
x.correct_antenna_delay(3, delays['Module 3'])

meas_new = x.compute_range_meas(id1,id2)

## TIMESTAMP DATA -------------------------------------------------
data = x.data[str(id1)+'->'+str(id2)]
gt = data['gt']
Ra1 = data['Ra1']
Ra2 = data['Ra2']
Db1 = data['Db1']
Db2 = data['Db2']
D1 = data['D1']
D2 = data['D2']
dt = data['dt']

n = np.size(dt)

## FLATTEN --------------------------------------------------------
gt = gt.flatten()
Ra1 = Ra1.flatten()
Ra2 = Ra2.flatten()
Db1 = Db1.flatten()
Db2 = Db2.flatten()
D1 = D1.flatten()
D2 = D2.flatten()
dt = dt.flatten()

## MEASUREMENTS ---------------------------------------------------
y_skew = Db2 - Ra2
y_offset = 0.5*(D1 - D2)

## KALMAN FILTER --------------------------------------------------
# Initialize storage variables
delta_hist = np.zeros((n,1))
gamma_hist = np.zeros((n,1))
P_hist = np.zeros((n,2))

# Initial guess
delta_hist[0] = 0
gamma_hist[0] = 0
P_hist[0,:] = np.array([P0_delta, P0_gamma])

# Filtering
x = np.array([delta_hist[0],gamma_hist[0]])
P = np.array([[P0_delta,0],
              [0,P0_gamma]])
for lv0 in range(1,n):
    # Prediction
    T = dt[lv0] # note that this now has unmodelled noise!
    A = np.array([[1,T],[0,1]])
    x = A@x
    P = A@P@A.T + (T)**2*np.array([[Q_delta,0],
                                   [0,Q_gamma]])
                        
    # Correction
    y_iter = np.array([[y_offset[lv0]],
                       [y_skew[lv0]]])

    C1 = np.array([1,Ra1[lv0]/2])
    C2 = np.array([0,Ra2[lv0]])
    C = np.vstack((C1,C2))

    M1 = np.array([-0.5, float(1+x[1])/2, float(1+x[1])/2, -0.5, 0, 0])
    M2 = np.array([0, 0, -float(1+x[1]), 1, float(1+x[1]), -1])
    M = np.vstack((M1,M2))

    R_blk = scipy.linalg.block_diag(R,R,R,R,R,R)

    S = C@P@C.T + M@R_blk@M.T
    K = P@C.T@(np.linalg.inv(S))
    x = x + K@(y_iter-C@x)
    P = (np.eye(2)-K@C)@P

    # Store values
    delta_hist[lv0] = x[0]
    gamma_hist[lv0] = x[1]
    P_hist[lv0,:] = np.array([P[0,0], P[1,1]])


twr_hat_1 = 0.5*(Ra1 - Ra2/Db2*Db1)*c/1e9
twr_hat_2 = (D1-delta_hist.flatten())/(1+gamma_hist.flatten())*c/1e9

e_1 = twr_hat_1 - gt
e_2 = twr_hat_2 - gt
e_1 = e_1[-500:-1]
e_2 = e_2[-500:-1]

print(np.mean(e_1))
print(np.mean(e_2))

print(np.var(e_1))
print(np.var(e_2))

# plt.subplot(1, 2, 1)
# plt.hist(e_1,bins=40)
# plt.subplot(1, 2, 2)
# plt.hist(e_2,bins=40)
# plt.show()

plt.subplot(1,2,1)
plt.plot(np.cumsum(dt),delta_hist)
# plt.plot(np.cumsum(dt),delta_hist+3*np.sqrt(P_hist[:,0]))
# plt.plot(np.cumsum(dt),delta_hist-3*np.sqrt(P_hist[:,0]))
plt.subplot(1,2,2)
plt.plot(np.cumsum(dt),gamma_hist)
# plt.plot(np.cumsum(dt),gamma_hist+3*np.sqrt(P_hist[:,1]))
# plt.plot(np.cumsum(dt),gamma_hist-3*np.sqrt(P_hist[:,1]))
plt.show()

plt.subplot(1,2,1)
plt.plot(np.cumsum(dt),gt)
plt.plot(np.cumsum(dt),twr_hat_1)
plt.ylim(0,5)
plt.subplot(1,2,2)
plt.plot(np.cumsum(dt),gt)
plt.plot(np.cumsum(dt),twr_hat_2)
plt.ylim(0,5)
plt.show()