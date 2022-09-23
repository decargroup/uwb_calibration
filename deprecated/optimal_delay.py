# %%
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from csaps import csaps
from scipy.interpolate import BSpline
import pandas as pd
from scipy import fft
matplotlib.use('Qt5Agg')

sns.set_theme()

# %% Import all data into a dataframe

# folder_path = 'datasets/2022_08_09_optimal_delay/D1_1500'
folder_path = 'datasets/2022_08_05_optimal_delay'

# Initialize empty dataframe
df = pd.DataFrame(columns=['delay','freq','data'])
n = 2500

for filename in os.listdir(folder_path):
    f = os.path.join(folder_path, filename)

    split_filename = filename.split('_')
    delay = filename.split('_')[1]
    T = filename.split('_')[2][:-4] # in nanoseconds
    data = np.genfromtxt(f)

    if int(delay) != -1:
        df = df.append({'delay': int(delay),
                        'freq': n/float(T)*1e9,
                        'data': data}, ignore_index=True)

df = df.sort_values('delay')

# %%
_c = 299702547

def find_std(x):
    upper_outliers = x>3
    x_new = x[~upper_outliers]
    lower_outliers = x_new<1.4
    return np.std(x_new[~lower_outliers])/_c*1e9

def get_Rt(exp_std,freq):
    return (exp_std)**2/freq   

# df['freq_roll_med'] = df['freq'].rolling(window=100).median()
df['exp_std'] = df.data.apply(find_std)
df['Rt'] = df.apply(lambda df: get_Rt(df.exp_std, df.freq), axis=1)

# %% 
# Experimentally-determined parameters
T = 0.0045 # total length of TWR transaction, minus second-response delay [s]
del_t2 = 0.0003 # first-response delay [s]
# T = 0.010 # total length of TWR transaction, minus second-response delay [s]
# del_t2 = 0.0015 # first-response delay [s]
R = 1/175 # Uncertainty of an individual measurement [ns^2]

# 100 linearly spaced numbers
x = np.linspace(0.00025,0.00675,1000)

# the function
y = (T+x)*R + del_t2*T*R/x + del_t2*R + del_t2**2*T*R/x**2 + del_t2**2*R/x

# # %% 
# ## Plot FFT of std error 

# # New dataframe 
# df_fft = df[['delay','exp_std']].copy()

# # Remove outliers
# df_fft.drop(df_fft[df_fft.delay > 6500].index, inplace=True)

# # Average out stds of measurements with same delay
# df_fft = df_fft.groupby('delay', as_index=False)['exp_std'].mean()

# # Interpolate 
# df_fft.set_index(df_fft['delay'], drop=True, inplace=True)
# df_fft = df_fft.reindex(np.linspace(410,6499,(6499-410)+1))
# df_fft.interpolate(method='linear', inplace=True)

# delay = np.array(df_fft['delay']/1e6, dtype=float).copy()
# exp_std = np.array(df_fft['exp_std'])

# diff = np.array(exp_std-np.sqrt(R + del_t2/delay*R + (del_t2/delay)**2*R))
# n = diff.shape[0]
# data_fft = np.abs(fft.rfft(diff) / n)
# f = fft.rfftfreq(n,d=1e-6)
# plt.semilogx(f, data_fft)
# plt.xlabel(r'Frequency [Hz]')
# plt.title(r'FFT on standard deviation error vs. Second-response delay')
# # plt.scatter(delay,diff)
# plt.show(block=True)
# %% 
## Fit spline to Rt 

# def moving_average(x, w):
#     return np.convolve(x, np.ones(w), 'valid') / w

# # Compute moving average of Rt
delay = np.array(df['delay']/1e6, dtype=float).copy()
# delay_new = delay[4:-5]
# Rt_new = moving_average(df['Rt'], 10)

# keep_idx = (df['Rt']-df['Rt_roll_med']) < 0.00003

# Add eps to same delays as this triggers csaps error
num_same = np.Inf
while num_same>0:
    same_idx = np.hstack(([False],(delay[1:] - delay[:-1]) == 0)).astype(int)
    delay = delay + same_idx*0.00000001

    num_same = np.sum(same_idx)
Rt_smoothed = csaps(delay, df['Rt'], delay, smooth=0.5)

# idx_keep = (df['Rt'] - Rt_smoothed)<0.00003
# delay_new = delay[idx_keep]
# Rt_smoothed = Rt_smoothed[idx_keep]

# delay_new = delay_new[4:-5]
# Rt_smoothed = moving_average(Rt_smoothed, 10)

# num_same = np.Inf
# while num_same>0:
#     same_idx = np.hstack(([False],(delay_new[1:] - delay_new[:-1]) == 0)).astype(int)
#     delay_new = delay_new + same_idx*0.00000001

#     num_same = np.sum(same_idx)

# Rt_smoothed = csaps(delay_new, Rt_smoothed, delay, smooth=0.5)

# %%
# %% Plotting figure 0
fig, axs = plt.subplots(1)

# Subplot 1: Rt
axs.plot(x*1e6,(np.sqrt(y)/1e9*_c*100)**2, 'r', linewidth=3)
axs.set_ylabel(r'$R_t$ [cm$^2$]', fontsize=50)
axs.set_xlabel(r'Second-Response Delay $\Delta t_4$ [$\mu$s]', fontsize=50)

axs.tick_params(axis='both', labelsize=50)

fig.set_size_inches(44, 19)
fig.savefig("figs/optimal_delay.pdf", dpi=1200)

# Plotting figure 1
fig, axs = plt.subplots(3,1, sharex='all')

# Subplot 1: Rt
axs[0].scatter(delay*1e6, (np.sqrt(df['Rt'])/1e9*_c*100)**2, label=r'Experimental', s=1)
# axs[0].plot(delay, Rt_smoothed, label=r'Experimental')
axs[0].plot(x*1e6,(np.sqrt(y)/1e9*_c*100)**2, 'r', label=r'Theoretical', linewidth=3)
# axs[0].set_ylim(0,2)
axs[0].set_ylim(0,0.2)
axs[0].set_ylabel(r'$R_t$ [cm$^2$]', fontsize=50)

# Subplot 2: std
axs[1].scatter(delay*1e6, df['exp_std']/1e9*_c*100, s=1)
axs[1].set_ylabel(r'Std. [cm]', fontsize=50)
axs[1].plot(x*1e6, np.sqrt(R + del_t2/x*R + (del_t2/x)**2*R)/1e9*_c*100, color='r', linewidth=3)
# axs[1].set_ylim(0,16)
axs[1].set_ylim(1,7)
axs[1].set_yticks([2,4,6])

# Subplot 3: freq
axs[2].scatter(delay*1e6, df['freq'], s=1)
axs[2].plot(x*1e6, 1/(x+T), color='r', linewidth=3)
axs[2].set_ylabel(r'Frequency [Hz]', fontsize=50)
axs[2].set_xlabel(r'Second-Response Delay $\Delta t_4$ [$\mu$s]', fontsize=50)
# axs[2].set_ylim(50,125)
axs[2].set_yticks([100,150,200])


axs[0].tick_params(axis='both', labelsize=50)
axs[1].tick_params(axis='both', labelsize=50)
axs[2].tick_params(axis='both', labelsize=50)

axs[2].set_xlim(0,7000)

lgnd = fig.legend(fontsize=50, ncol=2, loc='upper center')
lgnd.legendHandles[1]._sizes = [150]

fig.set_size_inches(44, 19)
fig.savefig("figs/optimal_delay_metrics.pdf", dpi=1200)

# Plotting figure 2
fig, axs = plt.subplots(1,1)
i = 0
print(df.iloc[i])
n = len(df.iloc[i]['data'])
axs.scatter(np.linspace(0,n,n),df.iloc[i]['data'])
# axs.set_ylim(1.5,2.5)
axs.set_ylabel(r'Range Measurement [m]')
axs.set_xlabel(r'Measurement Number')

plt.show(block=True)

# %%
# The verdict here is that this needs to be done experimentally especially in 
# multi-processor systems since. 
# I need to check if variance actually increases at high high delays.
# Re-collect all data, 1 min for each, 100 us intervals
#           ^^^ have them not face each other directly, and have them far apart
# %%
