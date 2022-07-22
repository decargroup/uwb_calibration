import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
matplotlib.use('Qt5Agg')

sns.set_theme()

# Experimentally-determined parameters
T = 0.006 # total length of TWR transaction, minus second-response delay [s]
del_t2 = 0.0003 # first-response delay [s]
R = 1 # Uncertainty of an individual measurement [ns^2]

# 100 linearly spaced numbers
x = np.linspace(0,0.005,1000)

# the function
y = (T+x)*R + del_t2*T*R/x + del_t2*R + del_t2**2*T*R/x**2 + del_t2**2*R/x

# Plotting
fig, ax = plt.subplots(1)

# plot the function
plt.plot(x,y, 'r')
ax.set_ylim(0,0.05)

ax.set_title(r'Optimal Delay')
ax.set_xlabel(r'Delay [s]')
ax.set_ylabel(r'$R_t$ [ns$^2$]')

# show the plot
plt.show()