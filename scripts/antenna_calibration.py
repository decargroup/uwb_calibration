# %%
from turtle import position
from pyuwbcalib.uwbcalibrate import UwbCalibrate
from pyuwbcalib.postprocess import PostProcess
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# matplotlib.use('Qt5Agg')
from matplotlib import rc
rc('text', usetex=True)

sns.set_palette("colorblind")
sns.set_style('whitegrid')

plt.rc('figure', figsize=(16, 9))
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=35)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rc('legend', facecolor=[1,1,1])
plt.rc('legend', fontsize=30)
plt.rcParams['figure.constrained_layout.use'] = True

# tag_ids={'ifo001': [1,2],
#          'ifo002': [3,4],
#          'ifo003': [5,6]}
tag_ids={'ifo001': [1,7],
         'ifo002': [3,4],
         'ifo003': [5,6]}
# moment_arms={'ifo001': [[0.15846,-0.16067,-0.07762], [-0.19711,0.14649,-0.082706]],
            #  'ifo002': [[0.18620,-0.13653,-0.05268], [-0.16133,0.17290,-0.047776]],
            #  'ifo003': [[0.18776,-0.16791,-0.08407], [-0.15605,0.14864,-0.079526]]}
moment_arms={'ifo001': [[0.13189,-0.17245,-0.05249], [-0.17542,0.15712,-0.05307]],
             'ifo002': [[0.16544,-0.15085,-0.03456], [-0.15467,0.16972,-0.01680]],
             'ifo003': [[0.16685,-0.18113,-0.05576], [-0.13485,0.15468,-0.05164]]}
raw_obj = PostProcess("datasets/2022_09_01_tag7/bias_calibration0/merged.bag",
                      tag_ids,
                      moment_arms,
                      num_meas=-1)

# plt.show(block=True)

# %%
kf = False
power_calib = True
antenna_delay = True
initiator_id = 3
target_id = 5
pair = (initiator_id, target_id)

# raw_obj.visualize_raw_data(pair=(initiator_id,target_id))
# plt.show(block=True)

# %%
# TODO: Surely there is a better way to do this?? I inherit some attributes + lift function
calib_obj = UwbCalibrate(raw_obj, rm_static=True)

## Pre-calibration
num_pairs = len(calib_obj.ts_data)
meas_old = {pair_i:[] for pair_i in calib_obj.ts_data}
for pair_i in calib_obj.ts_data:
    meas_old[pair_i] = calib_obj.compute_range_meas(pair_i,
                                                  visualize=False)

plt.show(block=True)

# %%
if kf:
    # Implement the Kalman filter and update the estimates
    R = (1)**2
    # Q = np.array(([4*1e8/1e8,0], [0,640]))
    Q = np.array(([0.000005,0], [0,.3]))**2
    calib_obj.filter_data(Q, R, visualize = False)

    meas_filtered = calib_obj.compute_range_meas(pair,
                                                visualize=False, owr=True)

    fig, ax = plt.subplots()
    # ax.set_xlabel("Measurement Number")
    # ax.set_ylabel("Range [m]")
    # ax.set_ylim(0, 4)
    t = calib_obj.time_intervals[pair]['t']
    ax.plot(t, meas_old[pair], linewidth=1, label="Raw")
    ax.plot(t, meas_filtered, linewidth=1, label="Calibrated")
    ax.plot(t, calib_obj.time_intervals[pair]['r_gt'], label="Ground Truth")
    ax.legend()

    plt.show(block=True)

# %% Antenna delay: # TODO: Should we do power calibration first to remove outliers? 
                    # TODO: Alternatively, could do robust LS
if antenna_delay: 
    # Calibrate the antenna delays
    delays = calib_obj.calibrate_antennas()
    print(delays)

    calib_obj.correct_antenna_delay(delays)

    meas_new = calib_obj.compute_range_meas(pair)

    #%%
    fig, axs = plt.subplots(2)

    t = calib_obj.time_intervals[pair]['t']

    axs[0].set_title("Measurements for pair " + str(pair))
    axs[0].set_xlabel("Measurement Number")
    axs[0].set_ylabel("Range [m]")
    axs[0].set_ylim(0, 4)
    axs[0].plot(t,calib_obj.time_intervals[pair]["r_gt"], linewidth=3, label="GT")
    # axs[0].plot(t,meas_old[pair], linewidth=1, label="Raw")
    axs[0].scatter(t,meas_new, linewidth=1, label="Antenna-Delay Calibrated", color='g')
    axs[0].legend()

    axs[1].set_title("Error for pair " + str(pair))
    axs[1].set_xlabel("Measurement Number")
    axs[1].set_ylabel("Range Error [m]")
    axs[1].set_ylim(-0.4, 0.8)
    axs[1].plot(meas_old[pair] - calib_obj.time_intervals[pair]["r_gt"], linewidth=1, label="Raw")
    axs[1].plot(meas_new - calib_obj.time_intervals[pair]["r_gt"], linewidth=1, label="Antenna-Delay Calibrated")
    axs[1].legend()

# %% Power calibration
if power_calib:
    # calib_obj.fit_model(std_window=50, chi_thresh=16.8, merge_pairs=True)
    calib_obj.fit_model(std_window=25, chi_thresh=22.8, merge_pairs=True)

    # bias_fit, std_fit = calib_obj.get_average_model()

# %% Final plotting
num_pairs = len(calib_obj.mean_spline)
fig, axs = plt.subplots(num_pairs)
for lv0, pair_i in enumerate(calib_obj.mean_spline):
    meas = calib_obj.compute_range_meas(pair_i)
    gt = calib_obj.time_intervals[pair_i]["r_gt"]

    # TODO: full bias calibration inside compute_range_meas
    spl = calib_obj.mean_spline[pair_i]
    fpp1_idx = calib_obj.fpp1_idx
    fpp2_idx = calib_obj.fpp2_idx
    lifted_fpp1 = calib_obj.lift(calib_obj.ts_data[pair_i][:,fpp1_idx])
    lifted_fpp2 = calib_obj.lift(calib_obj.ts_data[pair_i][:,fpp2_idx])
    pr_bias = spl(0.5 * (lifted_fpp1 + lifted_fpp2))
    meas_calibrated = meas - pr_bias

    axs[lv0].plot(meas-gt, label = 'w/ Antenna Delay Calibration')
    axs[lv0].plot(meas_calibrated-gt, label = 'Fully Calibrated')
    axs[lv0].plot(meas_old[pair_i]-gt, label = 'Raw')
    axs[lv0].set_ylabel("Range Error [m]")
    axs[lv0].set_xlabel("Measurement Number")
    axs[lv0].set_ylim([-0.35, 0.6])

    print("Pair: "+ str(pair))
    print("Raw Mean: "+ str(np.mean(meas_old[pair_i]-gt)))
    print("Antenna-Calibrated Mean: " + str(np.mean(meas-gt)))
    print("Fully-Calibrated Mean: " + str(np.mean(meas_calibrated-gt)))
    print("Raw Std: "+ str(np.std(meas_old[pair_i]-gt)))
    print("Antenna-Calibrated Std: " + str(np.std(meas-gt)))
    print("Fully-Calibrated Std: " + str(np.std(meas_calibrated-gt)))
    print("---------------------------------------------------------------")

axs[0].legend()
plt.show(block=True)
# %% TESTING
# Load datasets
raw_obj2 = PostProcess("datasets/2022_09_01_tag7/random2/merged.bag",
                       tag_ids,
                       moment_arms,
                       num_meas=-1)

calib_obj2 = UwbCalibrate(raw_obj2, rm_static=False)

# %%
# Processing

## Find earliest time recorded
t0 = np.Inf
for pair_i in calib_obj2.ts_data:
    t_i = calib_obj2.ts_data[pair_i][0,0]
    if t_i < t0:
        t0 = t_i

## Pre-calibration
num_pairs = len(calib_obj2.ts_data)
meas_old = {}
lifted_pr = {}
gt = {}
t = {}
for lv0, pair_i in enumerate(calib_obj2.ts_data):
    sorted_pair = tuple(np.sort(pair_i))
    if sorted_pair not in meas_old:
        meas_old[sorted_pair] = np.array([])
        lifted_pr[sorted_pair] = np.array([])
        t[sorted_pair] = np.array([])
        gt[sorted_pair] = np.array([])

    meas = calib_obj2.compute_range_meas(pair_i, visualize=False)
    meas_old[sorted_pair] = np.hstack((meas_old[sorted_pair],meas))

    lifted_pr_i = 0.5*calib_obj2.lift(calib_obj2.ts_data[pair_i][:,calib_obj2.fpp1_idx]) + \
                  + 0.5*calib_obj2.lift(calib_obj2.ts_data[pair_i][:,calib_obj2.fpp2_idx])
    lifted_pr[sorted_pair] = np.hstack((lifted_pr[sorted_pair],lifted_pr_i))

    t_i = calib_obj2.ts_data[pair_i][:,0] - t0
    t[sorted_pair] = np.hstack((t[sorted_pair],t_i))

    gt_i = calib_obj2.time_intervals[pair_i]["r_gt"]
    gt[sorted_pair] = np.hstack((gt[sorted_pair],gt_i))

# Correct antenna delays
if antenna_delay:
    calib_obj2.correct_antenna_delay(delays)
    
    meas_delay_calib = {}
    for lv0, pair_i in enumerate(calib_obj2.ts_data):
        sorted_pair = tuple(np.sort(pair_i))
        if sorted_pair not in meas_delay_calib:
            meas_delay_calib[sorted_pair] = np.array([])
        meas = calib_obj2.compute_range_meas(pair_i, visualize=False)
        meas_delay_calib[sorted_pair] = np.hstack((meas_delay_calib[sorted_pair],meas))

# Correct power-correlated bias
meas_calib = {}
std_calib = {}
for lv0, pair_i in enumerate(calib_obj2.ts_data):
    sorted_pair = tuple(np.sort(pair_i))
    if sorted_pair not in meas_calib:
        meas_calib[sorted_pair] = np.array([])    
        std_calib[sorted_pair] = np.array([])    
    meas_calib[sorted_pair] = meas_delay_calib[sorted_pair] - calib_obj.spl(lifted_pr[sorted_pair])
    std_calib[sorted_pair] = calib_obj.std_spl(lifted_pr[sorted_pair])
    
# Sort
for pair_i in t:
    idx = np.argsort(t[pair_i])
    t[pair_i] = t[pair_i][idx]
    meas_old[pair_i] = meas_old[pair_i][idx]
    lifted_pr[pair_i] = lifted_pr[pair_i][idx]
    gt[pair_i] = gt[pair_i][idx]
    meas_delay_calib[pair_i] = meas_delay_calib[pair_i][idx]
    meas_calib[pair_i] = meas_calib[pair_i][idx]
    std_calib[pair_i] = std_calib[pair_i][idx]

# %% 
# Plotting

# PLOT 1: Range measurements and ground truth for all pairs, each as a subplot
n = len(meas_calib)
num_rows = 6
num_columns = np.ceil(n/num_rows).astype(int)
fig = plt.figure()

for i,pair_i in enumerate(t):
    ax = plt.subplot(num_rows, num_columns, i+1) 
    ax.plot(t[pair_i],gt[pair_i])
    ax.scatter(t[pair_i],meas_old[pair_i],color='darkorange',s=1)
    ax.scatter(t[pair_i],meas_calib[pair_i],color='green',s=1)
    ax.set_ylim(0,5)
    ax.set_title(str(pair_i))

fig.legend(['Ground Truth', 'Raw', 'Calibrated'])

# PLOT 2: Bias and std, each as a subplot
fig = plt.figure()
fig.subplots_adjust(hspace = 0.5)

for i,pair_i in enumerate(t):
    ax = plt.subplot(num_rows, num_columns, i+1) 
    # ax.plot(t[pair_i],meas_old[pair_i] - gt[pair_i])
    bias = meas_calib[pair_i] - gt[pair_i]
    eps = bias**2 / std_calib[pair_i]**2
    inliers = (eps < 3.84) & (np.abs(bias)<3)
    if i==0:
        ax.scatter(t[pair_i][inliers],bias[inliers], s=1, label=r"Inliers")
        ax.scatter(t[pair_i][~inliers],bias[~inliers], s=1, label=r"Outliers")
        ax.fill_between(t[pair_i],
            -2*std_calib[pair_i],
            2*std_calib[pair_i],
            color='b',
            alpha=0.5,
            label=r"95\% confidence interval",
            )
    else:
        ax.scatter(t[pair_i][inliers],bias[inliers], s=1)
        ax.scatter(t[pair_i][~inliers],bias[~inliers], s=1)
        ax.fill_between(t[pair_i],
            -2*std_calib[pair_i],
            2*std_calib[pair_i],
            color='b',
            alpha=0.5,
            )
    ax.set_ylim(-0.5,1)
    
    if np.mod(i,2) == 0:
        ax.set_ylabel(r'Bias [m]', fontsize=40)
        ax.set_yticks([-0.5, 0, 0.5])
    else:
        ax.set_yticks([])
    
    if i >= num_rows*num_columns-2:
        ax.set_xlabel(r'Time [s]', fontsize=40)
    else:
        ax.set_xticks([])

    mean_inlier_error = np.mean(bias[inliers])
    std_inlier_error = np.std(bias[inliers])
    perc_inliers = np.sum(inliers)/len(eps)*100
    pair_print = str((int(pair_i[0]),int(pair_i[1])))
    ax.set_title(r'\textbf{Pair}: '+pair_print+r', \textbf{Inliers}: %2.2f\%%, \textbf{Mean}: %1.3f [cm], \textbf{Std.}: %1.3f [cm]' % (perc_inliers, mean_inlier_error*100, std_inlier_error*100),
                 fontsize=35)
    ax.tick_params(axis='both', labelsize=40)

# fig.suptitle(r"Test-Data Calibrated Measurements", fontsize=36)
lgnd = fig.legend(fontsize=50, ncol=3, loc='upper center')
lgnd.legendHandles[0]._sizes = [150]
lgnd.legendHandles[1]._sizes = [150]

# PLOT 3: Histogram of all biases in one plot
all_bias_old = np.array([])
all_bias_delay = np.array([])
all_bias_calib = np.array([])
all_std_calib = np.array([])
all_t = np.array([])
for pair_i in t:
    all_bias_old = np.hstack((all_bias_old, meas_old[pair_i]-gt[pair_i]))
    all_bias_delay = np.hstack((all_bias_delay, meas_delay_calib[pair_i]-gt[pair_i]))
    all_bias_calib = np.hstack((all_bias_calib, meas_calib[pair_i]-gt[pair_i]))
    all_std_calib = np.hstack((all_std_calib, std_calib[pair_i]))
    all_t = np.hstack((all_t, t[pair_i]))

# Remove large outliers
idx = np.abs(all_bias_old)<3
all_bias_old = all_bias_old[idx]
all_bias_delay = all_bias_delay[idx]
all_bias_calib = all_bias_calib[idx]
all_std_calib = all_std_calib[idx]
all_t = all_t[idx]

# fig,axs = plt.subplots(3,1,sharex=True, sharey=True)
# bins = np.linspace(-0.4,1,125)
# axs[0].hist(all_bias_old, bins=bins, density=True, alpha=0.5, color='b', label='Raw')
# axs[0].hist(all_bias_delay, bins=bins, density=True, alpha=0.5, color='r', label='Antenna-Delay Calibrated')
# axs[1].hist(all_bias_old, bins=bins, density=True, alpha=0.5, color='b')
# axs[1].hist(all_bias_calib, bins=bins, density=True, alpha=0.5, color='g', label='Fully Calibrated')
# axs[2].hist(all_bias_delay, bins=bins, density=True, alpha=0.5, color='r')
# axs[2].hist(all_bias_calib, bins=bins, density=True, alpha=0.5, color='g')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig,axs = plt.subplots(2,1,sharex=True, sharey=True)
bins = np.linspace(-0.3,0.7,100)
axs[0].hist(np.clip(all_bias_old, bins[0], bins[-1]), bins=bins, density=True, alpha=0.5, color=colors[0], label='Raw')
axs[0].hist(np.clip(all_bias_delay, bins[0], bins[-1]), bins=bins, density=True, alpha=0.5, color=colors[1], label='Antenna-Delay Calibrated')
axs[1].hist(np.clip(all_bias_delay, bins[0], bins[-1]), bins=bins, density=True, alpha=0.5, color=colors[1])
axs[1].hist(np.clip(all_bias_calib, bins[0], bins[-1]), bins=bins, density=True, alpha=0.5, color=colors[2], label='Fully Calibrated')

axs[0].set_yticks([0,2,4,6])
axs[1].set_yticks([0,2,4,6])

lgnd = fig.legend(ncol=3, loc='upper center')
# fig.suptitle(r"Calibration Results on Test Data", fontsize=36)
axs[1].set_xlabel(r"Range Bias [m]")

fig.subplots_adjust(bottom=0.15, hspace=0.2)
fig.savefig('figs/test_data_histograms.pdf', dpi=300)

plt.show(block=True)
# %%
# Extra plot for paper
# PLOT 2: Bias and std, each as a subplot
fig,axs = plt.subplots(2, 1, sharex='all', sharey='all')

pair_i = (1,5)
bias = meas_calib[pair_i] - gt[pair_i]
eps = bias**2 / std_calib[pair_i]**2
inliers = (eps < 3.84) & (np.abs(bias)<3)
axs[0].scatter(t[pair_i][inliers],bias[inliers], s=10, label=r"Inliers")
axs[0].scatter(t[pair_i][~inliers],bias[~inliers], s=10, label=r"Outliers")
axs[0].fill_between(t[pair_i][inliers],
    -2*std_calib[pair_i][inliers],
    2*std_calib[pair_i][inliers],
    color=colors[0],
    alpha=0.3,
    label=r"95\% confidence interval",
    )
axs[0].set_ylim(-0.5,1)

axs[0].set_ylabel(r'Bias [m]')
axs[0].set_yticks([-0.5, 0, 0.5, 1.00])

mean_inlier_error = np.mean(bias[inliers])
std_inlier_error = np.std(bias[inliers])
perc_inliers = np.sum(inliers)/len(eps)*100
pair_print = str((int(pair_i[0]),int(pair_i[1])))
axs[0].set_title(r'\textbf{Inliers}: %2.2f\%%, \textbf{Mean}: %1.3f [cm], \textbf{Std.}: %1.3f [cm]' \
                 % (perc_inliers, mean_inlier_error*100, std_inlier_error*100),
                 fontsize=30)

pair_i = (4,7)
bias = meas_calib[pair_i] - gt[pair_i]
eps = bias**2 / std_calib[pair_i]**2
inliers = (eps < 3.84) & (np.abs(bias)<3)
axs[1].scatter(t[pair_i][inliers],bias[inliers], s=10)
axs[1].scatter(t[pair_i][~inliers],bias[~inliers], s=10)
axs[1].fill_between(t[pair_i][inliers],
    -2*std_calib[pair_i][inliers],
    2*std_calib[pair_i][inliers],
    color=colors[0],
    alpha=0.3,
    )
axs[1].set_ylim(-0.5,1)

axs[1].set_ylabel(r'Bias [m]')
axs[1].set_yticks([-0.5, 0, 0.5, 1.00])
axs[1].set_xlabel(r'Time [s]')

mean_inlier_error = np.mean(bias[inliers])
std_inlier_error = np.std(bias[inliers])
perc_inliers = np.sum(inliers)/len(eps)*100
pair_print = str((int(pair_i[0]),int(pair_i[1])))
axs[1].set_title(r'\textbf{Inliers}: %2.2f\%%, \textbf{Mean}: %1.3f [cm], \textbf{Std.}: %1.3f [cm]' \
                 % (perc_inliers, mean_inlier_error*100, std_inlier_error*100),
                 fontsize=30)

# fig.suptitle(r"Test-Data Calibrated Measurements", fontsize=36)
lgnd = fig.legend(ncol=3, loc="upper center")
lgnd.legendHandles[0]._sizes = [150]
lgnd.legendHandles[1]._sizes = [150]

fig.subplots_adjust(top=0.8,bottom=0.15, hspace=0.3)
fig.savefig('figs/test_data_calibrated_measurements_2pairs.pdf', dpi=300)
plt.show(block=True)
# %%
