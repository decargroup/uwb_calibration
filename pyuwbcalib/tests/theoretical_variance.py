'''
The purpose of this script is to compare the variance of the different
TWR approaches using simulated data in the presence of artificial Gaussian
noise.
'''
import numpy as np
import matplotlib.pyplot as plt

d = 1 # distance in metres
n = 1000000 # number of measurements
R = 3 # variance in ns^2
c = 299702547 # speed of light in m/s

tf = d/c*1e9
delay = 5e5 # in nanoseconds

gamma_a = 40e-6*0
gamma_b = 20e-6*0

tx1 = np.random.rand(n,1)*1e7
rx1 = tx1 + tf
tx2 = rx1 + delay
rx2 = tx2 + tf
tx3 = tx2 + delay
rx3 = tx3 + tf
tx1 = tx1 + np.random.randn(n,1)*np.sqrt(R)
rx1 = rx1 + np.random.randn(n,1)*np.sqrt(R)
tx2 = tx2 + np.random.randn(n,1)*np.sqrt(R)
rx2 = rx2 + np.random.randn(n,1)*np.sqrt(R)
tx3 = tx3 + np.random.randn(n,1)*np.sqrt(R)
rx3 = rx3 + np.random.randn(n,1)*np.sqrt(R)

Ra1 = (1+gamma_a) * (rx2 - tx1)
Ra2 = (1+gamma_a) * (rx3 - rx2)
Db1 = (1+gamma_b) * (tx2 - rx1)
Db2 = (1+gamma_b) * (tx3 - tx2)

twr_hat_1 = 0.5*(Ra1 - Db1)*c/1e9
twr_hat_2 = 0.5*(Ra1 - Ra2/Db2*Db1)*c/1e9

e_1 = twr_hat_1 - d
e_2 = twr_hat_2 - d

print(np.mean(e_1))
print(np.mean(e_2))

print(np.var(e_1))
print(np.var(e_2))

plt.subplot(1, 2, 1)
plt.hist(e_1,bins=40)
plt.subplot(1, 2, 2)
plt.hist(e_2,bins=40)
plt.show()