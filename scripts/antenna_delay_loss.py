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
# moment_arms={'ifo001': [[0.15846,-0.16067,-0.07762], [-0.19711,0.14649,-0.082706]],
            #  'ifo002': [[0.18620,-0.13653,-0.05268], [-0.16133,0.17290,-0.047776]],
            #  'ifo003': [[0.18776,-0.16791,-0.08407], [-0.15605,0.14864,-0.079526]]}
moment_arms={'ifo001': [[0.13189,-0.17245,-0.05249], [-0.17542,0.15712,-0.05307]],
             'ifo002': [[0.16544,-0.15085,-0.03456], [-0.15467,0.16972,-0.01680]],
             'ifo003': [[0.16685,-0.18113,-0.05576], [-0.13485,0.15468,-0.05164]]}
raw_obj = PostProcess("datasets/2022_08_03/bias_calibration_new/merged.bag",
                      tag_ids,
                      moment_arms,
                      num_meas=-1)

meas_old = np.array([])
meas_new = {}
gt = np.array([])
# %%
# Robust least squares
calib_obj = UwbCalibrate(raw_obj, rm_static=True)

## Pre-calibration
for pair_i in calib_obj.ts_data:
    meas_old = np.hstack((meas_old,
                          calib_obj.compute_range_meas(pair_i, visualize=False)))
    gt = np.hstack((gt, calib_obj.time_intervals[pair_i]["r_gt"]))

# Calibrate the antenna delays
delays = calib_obj.calibrate_antennas()
print(delays)

calib_obj.correct_antenna_delay(delays)

## Post-calibration
meas_new['robust'] = np.array([])
for pair_i in calib_obj.ts_data:
    meas_new['robust'] = np.hstack((meas_new['robust'], 
                                   calib_obj.compute_range_meas(pair_i, visualize=False)))
    

# %%
# Linear least squares
calib_obj = UwbCalibrate(raw_obj, rm_static=True)

# Calibrate the antenna delays
delays = calib_obj.calibrate_antennas(loss='linear')
print(delays)

calib_obj.correct_antenna_delay(delays)

## Post-calibration
meas_new['linear'] = np.array([])
for pair_i in calib_obj.ts_data:
    meas_new['linear'] = np.hstack((meas_new['linear'], 
                                    calib_obj.compute_range_meas(pair_i, visualize=False)))

# %% Final plotting
fig, axs = plt.subplots(2,1,sharex='all', sharey='all')
bins=np.linspace(-0.5,1,100)
axs[0].hist(meas_new['linear']-gt, bins=bins, density=True, alpha=0.5, label=r'Antenna-Delay Calibrated')
axs[0].hist(meas_old-gt, bins=bins, density=True, alpha=0.5, label=r'Raw')
axs[0].set_title(r'Normal-Loss Least Squares')

axs[1].hist(meas_new['robust']-gt, bins=bins, density=True, alpha=0.5, label=r'Antenna-Delay Calibrated')
axs[1].hist(meas_old-gt, bins=bins, density=True, alpha=0.5, label=r'Raw')
axs[1].set_title(r'Cauchy-Loss Least Squares')
axs[1].set_xlabel(r'Bias [m]')

axs[0].legend()

plt.show(block=True)

print("Raw Mean: "+ str(np.mean(meas_old-gt)))
print("Linear Antenna-Calibrated Mean: " + str(np.mean(meas_new['linear']-gt)))
print("Robust Antenna-Calibrated Mean: " + str(np.mean(meas_new['robust']-gt)))
print("---------------------------------------------------------------")

axs[0].legend()
plt.show(block=True)
# %%
