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
moment_arms={'ifo001': [[0.17,-0.17,-0.03], [-0.17,0.17,-0.03]],
             'ifo002': [[0.17,-0.17,-0.03], [-0.17,0.17,-0.03]],
             'ifo003': [[0.17,-0.17,-0.03], [-0.17,0.17,-0.03]]}
raw_obj = PostProcess("datasets/2022_06_15/bias_calibration/merged.bag",
                      tag_ids,
                      moment_arms,
                      num_meas=-1)


plt.show(block=True)

# %%
kf = False
power_calib = True
antenna_delay = True
initiator_id = 1
target_id = 3
pair = (initiator_id, target_id)
raw_obj.visualize_raw_data(pair=(initiator_id,target_id))

# %%
# TODO: Surely there is a better way to do this?? I inherit some attributes + lift function
calib_obj = UwbCalibrate(raw_obj, rm_static=True)

## Pre-calibration
num_pairs = len(calib_obj.ts_data)
meas_old = {pair:[] for pair in calib_obj.ts_data}
for lv0, pair in enumerate(calib_obj.ts_data):
    meas_old[pair] = calib_obj.compute_range_meas(pair,
                                                  visualize=True)

plt.show(block=True)

# %%
if kf:
    # Implement the Kalman filter and update the estimates
    R = 20
    Q = np.array(([0.4,0], [0,640]))
    calib_obj.filter_data(Q, R, visualize = True)

    meas_filtered = calib_obj.compute_range_meas(pair,
                                                visualize=False, owr=True)

    fig, ax = plt.subplots()
    # ax.set_xlabel("Measurement Number")
    # ax.set_ylabel("Range [m]")
    # ax.set_ylim(0, 4)
    t = calib_obj.time_intervals[pair]['t']
    plt.plot(t, meas_old, linewidth=1, label="Raw")
    plt.plot(t, meas_filtered, linewidth=1, label="Calibrated")
    plt.plot(t, calib_obj.time_intervals[pair]['r_gt'])
    ax.legend()

    plt.show(block=True)

# %% Antenna delay: # TODO: Should we do power calibration first to remove outliers? 
                    # TODO: Alternatively, could do robust LS
if antenna_delay: 
    # Calibrate the antenna delays
    delays = calib_obj.calibrate_antennas()
    print(delays)

    id0 = tag_ids[0]
    id1 = tag_ids[1]
    id2 = tag_ids[2]
    calib_obj.correct_antenna_delay(id0, delays[id0])
    calib_obj.correct_antenna_delay(id1, delays[id1])
    calib_obj.correct_antenna_delay(id2, delays[id2])

    meas_new = calib_obj.compute_range_meas(pair)

    #%%
    fig, axs = plt.subplots(2)

    axs[0].set_title("Measurements for pair " + str(pair))
    axs[0].set_xlabel("Measurement Number")
    axs[0].set_ylabel("Range [m]")
    axs[0].set_ylim(0, 4)
    axs[0].plot(calib_obj.time_intervals[pair]["r_gt"], linewidth=3, label="GT")
    axs[0].plot(meas_old[pair], linewidth=1, label="Raw")
    axs[0].plot(meas_new, linewidth=1, label="Calibrated")
    axs[0].legend()

    axs[1].set_title("Error for pair " + str(pair))
    axs[1].set_xlabel("Measurement Number")
    axs[1].set_ylabel("Range Error [m]")
    axs[1].set_ylim(-0.4, 0.8)
    axs[1].plot(meas_old[pair] - calib_obj.time_intervals[pair]["r_gt"], linewidth=1, label="Raw")
    axs[1].plot(meas_new - calib_obj.time_intervals[pair]["r_gt"], linewidth=1, label="Calibrated")
    axs[1].legend()

    # %%

# %% Power calibration
if power_calib:
    calib_obj.fit_model(std_window=250, chi_thresh=10.8*1.25)

# %% Final plotting
num_pairs = len(calib_obj.ts_data)
fig, axs = plt.subplots(num_pairs)
for lv0, pair in enumerate(calib_obj.tag_pairs):
    meas = calib_obj.compute_range_meas(pair)
    gt = calib_obj.time_intervals[pair]["r_gt"]

    # TODO: full bias calibration inside compute_range_meas
    spl = calib_obj.mean_spline[pair]
    Pr1_idx = calib_obj.Pr1_idx
    Pr2_idx = calib_obj.Pr2_idx
    lifted_Pr1 = calib_obj.lift(calib_obj.ts_data[pair][:,Pr1_idx])
    lifted_Pr2 = calib_obj.lift(calib_obj.ts_data[pair][:,Pr2_idx])
    pr_bias = spl(0.5 * (lifted_Pr1 + lifted_Pr2))
    meas_calibrated = meas - pr_bias

    axs[lv0].plot(meas-gt, label = 'w/ Antenna Delay Calibration')
    axs[lv0].plot(meas_calibrated-gt, label = 'Fully Calibrated')
    axs[lv0].plot(meas_old[pair]-gt, label = 'Raw')
    axs[lv0].set_ylabel("Range Error [m]")
    axs[lv0].set_xlabel("Measurement Number")
    axs[lv0].set_ylim([-0.35, 0.6])

    print(np.mean(meas_old[pair]-gt))
    print(np.mean(meas-gt))
    print(np.mean(meas_calibrated-gt))
    print(np.std(meas_old[pair]-gt))
    print(np.std(meas-gt))
    print(np.std(meas_calibrated-gt))

axs[0].legend()
plt.show(block=True)
# %%
