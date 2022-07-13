# %%
from pyuwbcalib.uwbcalibrate import UwbCalibrate
from pyuwbcalib.postprocess import PostProcess
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
matplotlib.use('Qt5Agg')

sns.set_theme()

tag_ids={'ifo001': [1,2],
         'ifo002': [3,4],
         'ifo003': [5,6]}
moment_arms={'ifo001': [[0.15846,-0.16067,-0.07762], [-0.19711,0.14649,-0.082706]],
             'ifo002': [[0.18620,-0.13653,-0.05268], [-0.16133,0.17290,-0.047776]],
             'ifo003': [[0.18776,-0.16791,-0.08407], [-0.15605,0.14864,-0.079526]]}
raw_obj = PostProcess("datasets/2022_07_07/08/merged.bag",
                      tag_ids,
                      moment_arms,
                      num_meas=-1)


# plt.show(block=True)

# %%
kf = False
power_calib = True
antenna_delay = True
initiator_id = 2
target_id = 3
pair = (initiator_id, target_id)
# raw_obj.visualize_raw_data(pair=(initiator_id,target_id))

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

    axs[0].set_title("Measurements for pair " + str(pair))
    axs[0].set_xlabel("Measurement Number")
    axs[0].set_ylabel("Range [m]")
    axs[0].set_ylim(0, 4)
    axs[0].plot(calib_obj.time_intervals[pair]["r_gt"], linewidth=3, label="GT")
    axs[0].plot(meas_old[pair], linewidth=1, label="Raw")
    axs[0].plot(meas_new, linewidth=1, label="Antenna-Delay Calibrated")
    axs[0].legend()

    axs[1].set_title("Error for pair " + str(pair))
    axs[1].set_xlabel("Measurement Number")
    axs[1].set_ylabel("Range Error [m]")
    axs[1].set_ylim(-0.4, 0.8)
    axs[1].plot(meas_old[pair] - calib_obj.time_intervals[pair]["r_gt"], linewidth=1, label="Raw")
    axs[1].plot(meas_new - calib_obj.time_intervals[pair]["r_gt"], linewidth=1, label="Antenna-Delay Calibrated")
    axs[1].legend()

    # %%

# %% Power calibration
if power_calib:
    calib_obj.fit_model(std_window=25, chi_thresh=16.8, merge_pairs=True)

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
raw_obj2 = PostProcess("datasets/2022_07_07/08/merged.bag",
                       tag_ids,
                       moment_arms,
                       num_meas=-1)

calib_obj2 = UwbCalibrate(raw_obj2, rm_static=False)

## Pre-calibration
num_pairs = len(calib_obj2.ts_data)
meas_old = {pair_i:[] for pair_i in calib_obj2.ts_data}
for lv0, pair_i in enumerate(calib_obj2.ts_data):
    meas_old[pair_i] = calib_obj2.compute_range_meas(pair_i,
                                                  visualize=False)

# plt.show(block=True)

calib_obj2.correct_antenna_delay(delays)

meas_new = calib_obj2.compute_range_meas(pair)
avg_fpp = 0.5*(calib_obj2.ts_data[pair][:,calib_obj2.fpp1_idx] + calib_obj2.ts_data[pair][:,calib_obj2.fpp2_idx])
lifted_pr = calib_obj2.lift(avg_fpp)
meas_new -= calib_obj.spl(lifted_pr)

t = calib_obj2.ts_data[pair][:,0] - calib_obj2.ts_data[pair][0,0]
std = calib_obj.std_spl(lifted_pr)

fig, axs = plt.subplots(4,1, sharex='all')
gt = calib_obj2.time_intervals[pair]["r_gt"]

axs[0].plot((t-t[0])/1e9, meas_new-gt, label = r'Fully Calibrated')
axs[0].plot((t-t[0])/1e9, meas_old[pair]-gt, label = r'Raw')
axs[0].set_ylabel(r"Range Error [m]")
axs[0].set_xlabel(r"Time [s]")
# axs[0].set_ylim([-0.35, 0.6])

axs[0].fill_between(
                    (t-t[0])/1e9,
                    - 3 * std,
                    3 * std,
                    alpha=0.5,
                    label=r"99.97% confidence interval",
                )
axs[0].legend()

avg_rxp = 0.5*(calib_obj2.ts_data[pair][:,calib_obj2.rxp1_idx] + calib_obj2.ts_data[pair][:,calib_obj2.rxp2_idx])
lifted_rxp = calib_obj2.lift(avg_rxp)

axs[1].plot((t-t[0])/1e9, lifted_pr, label = r'Lifted FPP Power')
axs[1].plot((t-t[0])/1e9, lifted_rxp, label = r'Lifted RXP Power')
axs[1].set_ylabel(r"$f(P_r)$")
axs[1].set_xlabel(r"Time [s]")

axs[1].legend()

axs[2].plot((t-t[0])/1e9, lifted_rxp - lifted_pr, label = r'RXP - FPP')
axs[2].set_ylabel(r"RXP - FPP [?]")
axs[2].set_xlabel(r"Time [s]")

# axs[2].legend()

avg_std = 0.5*(calib_obj2.ts_data[pair][:,calib_obj2.std1_idx] + calib_obj2.ts_data[pair][:,calib_obj2.std2_idx])
axs[3].plot((t-t[0])/1e9, avg_std, label = r'LDE std avg')
axs[3].set_ylabel(r"LDE std [?]")
axs[3].set_xlabel(r"Time [s]")

axs[3].legend()

print("Raw Mean: "+ str(np.mean(meas_old[pair]-gt)))
print("Fully-Calibrated Mean: " + str(np.mean(meas_new-gt)))
print("Raw Std: "+ str(np.std(meas_old[pair]-gt)))
print("Fully-Calibrated Std: " + str(np.std(meas_new-gt)))
print("---------------------------------------------------------------")


fig, axs = plt.subplots(1)
axs.scatter(lifted_pr-lifted_rxp,meas_old[pair]-gt)
axs.scatter(lifted_pr,meas_old[pair]-gt)
axs.scatter(lifted_rxp,meas_old[pair]-gt)
# axs.set_xlabel(r"$f(P_r)$")
# axs.set_ylabel(r"Bias [m]")

fig, axs = plt.subplots(1)
axs.plot((t-t[0])/1e9, meas_old[pair])
axs.plot((t-t[0])/1e9, gt)
# axs.set_xlabel(r"$f(P_r)$")
# axs.set_ylabel(r"Bias [m]")


plt.show(block=True)
# %%
