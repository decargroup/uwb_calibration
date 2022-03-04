'''
The purpose of this script is to test the Kalman filter on artificial data.
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg 

## PARAMETERS -----------------------------------------------------
d = 3 # distance in metres
n = 10000 # number of measurements
c = 299702547 # speed of light in m/s

R = 1 # variance in ns^2
P0_delta = 100
P0_gamma = 100
Q_delta = 1
Q_gamma = 1

delay = 5e4 # in nanoseconds
gamma_a = 40e-6*0
gamma_b = 20e-6*0

tf = d/c*1e9

## TRUE CLOCK STATES ----------------------------------------------

delta = np.zeros((n,1))
gamma = np.zeros((n,1))

delta[0] = np.random.randn()*np.math.sqrt(P0_delta)
gamma[0] = np.random.randn()*np.math.sqrt(P0_gamma)

tx1 = np.array([range(0,n)]).T*1e6

for lv0 in range(1,n):
    T = tx1[lv0] - tx1[lv0-1]
    delta[lv0] = delta[lv0-1] + T/1e9*gamma[lv0-1] + T/1e9*np.random.randn()*np.math.sqrt(Q_delta)
    gamma[lv0] = gamma[lv0-1] + T/1e9*np.random.randn()*np.math.sqrt(Q_gamma)
    

## TIMESTAMP DATA -------------------------------------------------
rx1 = tx1 + delta + (1+gamma)*(tf+np.random.randn(n,1)*np.sqrt(R))
tx2 = tx1 + delta + (1+gamma)*(tf+delay+np.random.randn(n,1)*np.sqrt(R))
rx2 = tx1 + 2*tf + delay + np.random.randn(n,1)*np.sqrt(R)
tx3 = tx1 + delta + (1+gamma)*(tf+2*delay+np.random.randn(n,1)*np.sqrt(R))
rx3 = tx1 + 2*tf + 2*delay + np.random.randn(n,1)*np.sqrt(R)

tx1 = tx1 + np.random.randn(n,1)*np.sqrt(R)

Ra1 = (1+gamma_a) * (rx2 - tx1)
Ra2 = (1+gamma_a) * (rx3 - rx2)
Db1 = (1+gamma_b) * (tx2 - rx1)
Db2 = (1+gamma_b) * (tx3 - tx2)

## FLATTEN --------------------------------------------------------

tx1 = tx1.flatten()
rx1 = rx1.flatten()
tx2 = tx2.flatten()
rx2 = rx2.flatten()
tx3 = tx3.flatten()
rx3 = rx3.flatten()
Ra1 = Ra1.flatten()
Ra2 = Ra2.flatten()
Db1 = Db1.flatten()
Db2 = Db2.flatten()

## MEASUREMENTS ---------------------------------------------------
y_skew = Db2 - Ra2
y_offset = 0.5*((rx1+tx2)-(tx1+rx2))


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
    T = tx1[lv0] - tx1[lv0-1] # note that this now has unmodelled noise!
    A = np.array([[1,T/1e9],[0,1]])
    x = A@x
    P = A@P@A.T + (T/1e9)**2*np.array([[Q_delta,0],
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


# ## PLOT GROUND TRUTH -------------------------------------------
# plt.subplot(1,2,1)
# plt.plot(tx1,delta)
# plt.subplot(1,2,2)
# plt.plot(tx1,gamma)
# plt.show()

# ## PLOT ERRORS -------------------------------------------------
# plt.subplot(1,2,1)
# plt.plot(tx1,delta_hist - delta)
# plt.plot(tx1,3*np.sqrt(P_hist[:,0]))
# plt.plot(tx1,-3*np.sqrt(P_hist[:,0]))
# plt.subplot(1,2,2)
# plt.plot(tx1,gamma_hist - gamma)
# plt.plot(tx1,3*np.sqrt(P_hist[:,1]))
# plt.plot(tx1,-3*np.sqrt(P_hist[:,1]))
# plt.show()

# ## RANGE DATA -----------------------------------------------------

twr_hat_1 = 0.5*(Ra1 - Ra2/Db2*Db1)*c/1e9
twr_hat_2 = (rx1-tx1-delta_hist.flatten())/(1+gamma_hist.flatten())*c/1e9

e_1 = twr_hat_1 - d
e_2 = twr_hat_2 - d
e_2 = e_2[1:]

print(np.mean(e_1))
print(np.mean(e_2))

print(np.var(e_1))
print(np.var(e_2))

plt.subplot(1, 2, 1)
plt.hist(e_1,bins=40)
plt.subplot(1, 2, 2)
plt.hist(e_2,bins=40)
plt.show()


