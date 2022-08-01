# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from bagpy import bagreader
import pandas as pd
matplotlib.use('Qt5Agg')

sns.set_theme()

exp_window = 180# 600
def read_range_data(filepath):
    bag_data = bagreader('datasets/2022_07_26_optimal_delay/' + filepath)

    data = bag_data.message_by_topic('/uwb/range')
    data_pd = pd.read_csv(data)
    # return np.array(data_pd['range'])
    return data_pd

all_range_data = [read_range_data('ss_2022-07-26-12-49-30.bag'),
                  read_range_data('700_2022-07-26-12-53-45.bag'),
                  read_range_data('850_2022-07-26-12-57-31.bag'),
                  read_range_data('1000_2022-07-26-13-01-14.bag'),
                  read_range_data('1200_2022-07-26-13-05-01.bag'),
                  read_range_data('1400_2022-07-26-13-08-43.bag'),
                  read_range_data('1500_2022-07-26-13-12-04.bag'),
                  read_range_data('1600_2022-07-26-13-15-57.bag'),
                  read_range_data('1800_2022-07-26-13-19-27.bag'),
                  read_range_data('2000_2022-07-26-13-23-03.bag'),
                  read_range_data('2500_2022-07-26-13-30-04.bag'),
                  read_range_data('3000_2022-07-26-13-33-32.bag'),
                  read_range_data('4000_2022-07-26-13-38-23.bag'),
                  read_range_data('5000_2022-07-26-13-41-38.bag'),
                  read_range_data('6000_2022-07-26-13-45-00.bag'),]
# all_range_data = [read_range_data('ss_2022-07-22-17-07-09.bag'),
#                   read_range_data('600_2022-07-22-16-57-20.bag'),
#                   read_range_data('750_2022-07-22-16-55-33.bag'),
#                   read_range_data('1000_2022-07-22-16-54-39.bag'),
#                   read_range_data('default_2022-07-22-16-51-23.bag'),
#                   read_range_data('2000_2022-07-22-16-58-39.bag'),
#                   read_range_data('3000_2022-07-22-17-03-00.bag'),
#                   read_range_data('4000_2022-07-22-17-03-52.bag'),
#                   read_range_data('5000_2022-07-22-16-59-42.bag'),
#                   read_range_data('6000_2022-07-22-17-01-29.bag'),
#                   ]
# all_range_data = [read_range_data('ss_2022-07-28-03-34-05.bag'),
#                   read_range_data('700_2022-07-28-01-53-34.bag'),
#                   read_range_data('850_2022-07-28-02-04-20.bag'),
#                   read_range_data('1000_2022-07-28-02-16-40.bag'),
#                   read_range_data('1250_2022-07-28-02-28-18.bag'),
#                   read_range_data('1500_2022-07-28-02-38-55.bag'),
#                   read_range_data('1750_2022-07-28-02-49-54.bag'),
#                   read_range_data('2000_2022-07-28-03-00-29.bag'),
#                   read_range_data('3000_2022-07-28-03-11-04.bag'),
#                   read_range_data('5000_2022-07-28-03-21-59.bag'),
#                   ]

# %% 
# Experimentally-determined parameters
T = 0.0064 # total length of TWR transaction, minus second-response delay [s]
del_t2 = 0.000303 # first-response delay [s]
R = 1/400 # Uncertainty of an individual measurement [ns^2]

# 100 linearly spaced numbers
x = np.linspace(0,0.005,1000)

# the function
y = (T+x)*R + del_t2*T*R/x + del_t2*R + del_t2**2*T*R/x**2 + del_t2**2*R/x

# Plotting
# fig, ax = plt.subplots(1)

# show the plot
# plt.show(block=True)

# %%
_to_us = 1e6 * (1.0 / 499.2e6 / 128.0)

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def get_Rt(x,T,del_t2,R):
    return (T+x)*R + del_t2*T*R/x + del_t2*R + del_t2**2*T*R/x**2 + del_t2**2*R/x

_c = 299702547

window = 10
row_num = 3
col_num = 5
fig,axs = plt.subplots(row_num,col_num,sharex=True,sharey=True)
fig2,axs2 = plt.subplots(row_num,col_num,sharex=True,sharey=True)

std_data = []
Rt_data = []
freq_data = []
t4_data = []
# t4_data = []
T_data = []
for i,data in enumerate(all_range_data):
    xi = all_range_data[i]['tx2'] - all_range_data[i]['rx1']
    tau = all_range_data[i]['tx3'] - all_range_data[i]['tx2']
    tau = tau*_to_us
    xi = xi*_to_us

    freq = len(all_range_data[i])/exp_window
    freq_data = freq_data + [freq]

    print(r"------------------- Run " + str(i) + " -------------------")
    t2 = np.mean(xi[xi > 0])
    print(r"\Delta t_2: " + str(t2))
    t4 = np.mean(tau[tau > 0])
    t4_data = t4_data + [t4]
    print(r"\Delta t_4: " + str(t4))
    T = 1/freq*1e6 - t4
    print(r"T: " + str(T))
    T_data = T_data + [T]

    t = all_range_data[i]['Time']
    t = t - t[0]

    n = len(t)

    range = all_range_data[i]['range']
    if i>0:
        # range = range[np.abs(range)<10][int(np.floor(3*n//4)):-int(np.floor(n//400))]
        range = range[np.abs(range)<10]
        # range = range[np.abs(range)>1.5]
        # range = range[int(np.floor(3*n//4)):-int(np.floor(n//10))]
        # range = range[12000:-12000]

    # std = np.std(range)
        std = (np.std(rolling_window(np.array(range),window),axis=1))
        axs[int(np.floor(i/col_num)),int(np.remainder(i,col_num))].plot(std)
        axs2[int(np.floor(i/col_num)),int(np.remainder(i,col_num))].plot(range)

    print(r"Std: " + str(np.mean(std)))
    std_data = std_data + [np.mean(std)]

    Rt = (np.mean(std)/_c*1e9)**2*(T+t4)/1e6
    # Rt = get_Rt(t4_data[i]/1e6,T/1e6,t2/1e6,R)
    print(r"R: " + str(Rt))
    Rt_data = Rt_data + [Rt]

plt.show(block=True)
# %%
# Plotting
fig, axs = plt.subplots(1,1, sharex='all')

# plot the theoretical function 
axs.plot(x,y, 'r', label=r'Theoretical')

# plot the experimental function
axs.scatter(np.array(t4_data)[1:]/1e6, Rt_data[1:], label=r'Experimental')

axs.set_ylim(0,0.00025)

axs.legend()
axs.set_title(r'Optimal Delay')
axs.set_xlabel(r'Delay [s]')
axs.set_ylabel(r'$R_t$ [ns$^2$]')

fig,axs = plt.subplots(4,1)

axs[0].scatter(t4_data[1:],Rt_data[1:])
axs[1].scatter(t4_data[1:],std_data[1:])
axs[2].scatter(t4_data[1:],freq_data[1:])
axs[3].scatter(t4_data[1:],T_data[1:])

plt.show(block=True)
# %%
# The verdict here is that this needs to be done experimentally especially in 
# multi-processor systems since. 
# I need to check if variance actually increases at high high delays.
# Re-collect all data, 1 min for each, 100 us intervals
#           ^^^ have them not face each other directly, and have them far apart