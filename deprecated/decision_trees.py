# %%
from pyuwbcalib.uwbcalibrate import UwbCalibrate
from pyuwbcalib.postprocess import PostProcess
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
matplotlib.use('Qt5Agg')

sns.set_theme()

bias_thresh = 0.25

tag_ids={'ifo001': [1,2],
         'ifo002': [3,4],
         'ifo003': [5,6]}
# moment_arms={'ifo001': [[0.15846,-0.16067,-0.07762], [-0.19711,0.14649,-0.082706]],
#              'ifo002': [[0.18620,-0.13653,-0.05268], [-0.16133,0.17290,-0.047776]],
#              'ifo003': [[0.18776,-0.16791,-0.08407], [-0.15605,0.14864,-0.079526]]}
moment_arms={'ifo001': [[0.13189,-0.17245,-0.05249], [-0.17542,0.15712,-0.05307]],
             'ifo002': [[0.16544,-0.15085,-0.03456], [-0.15467,0.16972,-0.01680]],
             'ifo003': [[0.16685,-0.18113,-0.05576], [-0.13485,0.15468,-0.05164]]}
raw_obj = PostProcess("datasets/2022_08_03/bias_calibration_new2/merged.bag",
                      tag_ids,
                      moment_arms,
                      num_meas=-1)

# plt.show(block=True)

# %% TRAINING BIAS AND POWER FITS
calib_obj = UwbCalibrate(raw_obj, rm_static=True)

## Pre-calibration
meas_old = {}
for pair_i in calib_obj.ts_data:
    meas_old[pair_i] = calib_obj.compute_range_meas(pair_i,
                                                  visualize=False)

# Calibrate the antenna delays
delays = calib_obj.calibrate_antennas()
calib_obj.correct_antenna_delay(delays)
calib_obj.fit_model(std_window=25, chi_thresh=16.8, merge_pairs=True)

## Post-calibration
meas_new = {}
for pair_i in calib_obj.ts_data:
    meas_new[pair_i] = calib_obj.compute_range_meas(pair_i,
                                                    visualize=False)
    avg_fpp = 0.5*(calib_obj.ts_data[pair_i][:,calib_obj.fpp1_idx] + calib_obj.ts_data[pair_i][:,calib_obj.fpp2_idx])
    lifted_pr = calib_obj.lift(avg_fpp)
    meas_new[pair_i] -= calib_obj.spl(lifted_pr)                                                    


# %% TRAINING DECISION TREE
inliers = {}
undetected_outliers = {}
inliers['gt'] = np.array([])
inliers['avg_fpp'] = np.array([])
inliers['lifted_pr'] = np.array([])
inliers['avg_rxp'] = np.array([])
inliers['lifted_rxp'] = np.array([])
inliers['avg_std'] = np.array([])
inliers['bias'] = np.array([])
undetected_outliers['gt'] = np.array([])
undetected_outliers['avg_fpp'] = np.array([])
undetected_outliers['lifted_pr'] = np.array([])
undetected_outliers['avg_rxp'] = np.array([])
undetected_outliers['lifted_rxp'] = np.array([])
undetected_outliers['avg_std'] = np.array([])
undetected_outliers['bias'] = np.array([])
for pair_i in calib_obj.tag_pairs:
    gt = calib_obj.time_intervals[pair_i]["r_gt"]
    avg_fpp = 0.5*(calib_obj.ts_data[pair_i][:,calib_obj.fpp1_idx] + calib_obj.ts_data[pair_i][:,calib_obj.fpp2_idx])
    lifted_pr = calib_obj.lift(avg_fpp)
    avg_rxp = 0.5*(calib_obj.ts_data[pair_i][:,calib_obj.rxp1_idx] + calib_obj.ts_data[pair_i][:,calib_obj.rxp2_idx])
    lifted_rxp = calib_obj.lift(avg_rxp)
    avg_std = 0.5*(calib_obj.ts_data[pair_i][:,calib_obj.std1_idx] + calib_obj.ts_data[pair_i][:,calib_obj.std2_idx])

    undetected_outliers_idx = (np.abs(meas_new[pair_i]-gt) > bias_thresh)
    inliers_idx = ~undetected_outliers_idx
    inliers['gt'] = np.concatenate([inliers['gt'], gt[inliers_idx]])
    inliers['avg_fpp'] = np.concatenate([inliers['avg_fpp'], avg_fpp[inliers_idx]])
    inliers['lifted_pr'] = np.concatenate([inliers['lifted_pr'], lifted_pr[inliers_idx]])
    inliers['avg_rxp'] = np.concatenate([inliers['avg_rxp'], avg_rxp[inliers_idx]])
    inliers['lifted_rxp'] = np.concatenate([inliers['lifted_rxp'], lifted_rxp[inliers_idx]])
    inliers['avg_std'] = np.concatenate([inliers['avg_std'], avg_std[inliers_idx]])
    inliers['bias'] = np.concatenate([inliers['bias'], (meas_new[pair_i]-gt)[inliers_idx]])
    undetected_outliers['gt'] = np.concatenate([undetected_outliers['gt'], gt[undetected_outliers_idx]])
    undetected_outliers['avg_fpp'] = np.concatenate([undetected_outliers['avg_fpp'], avg_fpp[undetected_outliers_idx]])
    undetected_outliers['lifted_pr'] = np.concatenate([undetected_outliers['lifted_pr'], lifted_pr[undetected_outliers_idx]])
    undetected_outliers['avg_rxp'] = np.concatenate([undetected_outliers['avg_rxp'], avg_rxp[undetected_outliers_idx]])
    undetected_outliers['lifted_rxp'] = np.concatenate([undetected_outliers['lifted_rxp'], lifted_rxp[undetected_outliers_idx]])
    undetected_outliers['avg_std'] = np.concatenate([undetected_outliers['avg_std'], avg_std[undetected_outliers_idx]])
    undetected_outliers['bias'] = np.concatenate([undetected_outliers['bias'], (meas_new[pair_i]-gt)[undetected_outliers_idx]])

fig, axs = plt.subplots(4,1)
axs[0].set_title(r'Probability Density Function for different Metrics', size=20)
axs[0].hist(inliers['lifted_rxp'],np.linspace(1.25,3,100),alpha=0.5,density=True, label=r'Inliers')
axs[0].hist(undetected_outliers['lifted_rxp'],np.linspace(1.25,3,100),alpha=0.5,density=True, label=r'Outliers')
axs[0].set_xlabel(r'$f(P_r)$')
axs[0].legend(loc='upper right')
axs[1].hist(inliers['lifted_pr'],np.linspace(0,2,100),alpha=0.5,density=True)
axs[1].hist(undetected_outliers['lifted_pr'],np.linspace(0,2,100),alpha=0.5,density=True)
axs[1].set_xlabel(r'$f(P_f)$')
axs[2].hist(inliers['avg_rxp'] - inliers['avg_fpp'],np.linspace(0,15,100),alpha=0.5,density=True)
axs[2].hist(undetected_outliers['avg_rxp'] - undetected_outliers['avg_fpp'],np.linspace(0,15,100),alpha=0.5,density=True)
axs[2].set_xlabel(r'$P_r - P_f [dB]$')
axs[3].hist(inliers['avg_std'],np.linspace(0,100,100),alpha=0.5,density=True)
axs[3].hist(undetected_outliers['avg_std'],np.linspace(0,100,100),alpha=0.5,density=True)
axs[3].set_xlabel(r'$P_r - P_f [dB]$')

print('Inliers: ' + str(np.size(inliers['gt'])))
print('Undetected outliers: ' + str(np.size(undetected_outliers['gt'])))

X1 = np.vstack((inliers['lifted_rxp'], 
                inliers['lifted_pr'], 
                inliers['avg_rxp'] - inliers['avg_fpp'],
                ))
X2 = np.vstack((undetected_outliers['lifted_rxp'], 
                undetected_outliers['lifted_pr'], 
                undetected_outliers['avg_rxp'] - undetected_outliers['avg_fpp'],
                ))
X = np.hstack((X1,X2))
X = X.T

y1 = np.zeros((X1.shape[1]))
y2 = np.ones((X2.shape[1]))
y = np.hstack((y1,y2)) 

clf = RandomForestClassifier(class_weight={0:1/18.6,1:1},min_samples_leaf=10)

clf.fit(X,y)

# %% TESTING 
raw_obj_test = PostProcess("datasets/2022_08_03/line_triangle_line/merged.bag",
                       tag_ids,
                       moment_arms,
                       num_meas=-1)

calib_obj_test = UwbCalibrate(raw_obj_test, rm_static=False)

## Pre-calibration
meas_old = {}
for lv0, pair_i in enumerate(calib_obj_test.ts_data):
    meas_old[pair_i] = calib_obj_test.compute_range_meas(pair_i,
                                                  visualize=False)

## Post-calibration
calib_obj_test.correct_antenna_delay(delays)
meas_new = {}
for lv0, pair_i in enumerate(calib_obj_test.ts_data):
    meas_new[pair_i] = calib_obj_test.compute_range_meas(pair_i,
                                                  visualize=False)
    avg_fpp = 0.5*(calib_obj_test.ts_data[pair_i][:,calib_obj_test.fpp1_idx] + calib_obj_test.ts_data[pair_i][:,calib_obj_test.fpp2_idx])
    lifted_pr = calib_obj_test.lift(avg_fpp)
    meas_new[pair_i] -= calib_obj.spl(lifted_pr)

# %% TESTING DECISION TREE
inliers = {}
undetected_outliers = {}
inliers['t'] = np.array([])
inliers['gt'] = np.array([])
inliers['avg_fpp'] = np.array([])
inliers['lifted_pr'] = np.array([])
inliers['avg_rxp'] = np.array([])
inliers['lifted_rxp'] = np.array([])
inliers['avg_std'] = np.array([])
inliers['bias'] = np.array([])
undetected_outliers['t'] = np.array([])
undetected_outliers['gt'] = np.array([])
undetected_outliers['avg_fpp'] = np.array([])
undetected_outliers['lifted_pr'] = np.array([])
undetected_outliers['avg_rxp'] = np.array([])
undetected_outliers['lifted_rxp'] = np.array([])
undetected_outliers['avg_std'] = np.array([])
undetected_outliers['bias'] = np.array([])
for pair_i in calib_obj_test.tag_pairs:
    t = calib_obj_test.time_intervals[pair_i]["t"]
    gt = calib_obj_test.time_intervals[pair_i]["r_gt"]
    avg_fpp = 0.5*(calib_obj_test.ts_data[pair_i][:,calib_obj_test.fpp1_idx] + calib_obj_test.ts_data[pair_i][:,calib_obj_test.fpp2_idx])
    lifted_pr = calib_obj_test.lift(avg_fpp)
    avg_rxp = 0.5*(calib_obj_test.ts_data[pair_i][:,calib_obj_test.rxp1_idx] + calib_obj_test.ts_data[pair_i][:,calib_obj_test.rxp2_idx])
    lifted_rxp = calib_obj_test.lift(avg_rxp)
    avg_std = 0.5*(calib_obj_test.ts_data[pair_i][:,calib_obj_test.std1_idx] + calib_obj_test.ts_data[pair_i][:,calib_obj_test.std2_idx])

    undetected_outliers_idx = (np.abs(meas_new[pair_i]-gt) > bias_thresh)
    inliers_idx = ~undetected_outliers_idx
    inliers['t'] = np.concatenate([inliers['t'], t[inliers_idx]])
    inliers['gt'] = np.concatenate([inliers['gt'], gt[inliers_idx]])
    inliers['avg_fpp'] = np.concatenate([inliers['avg_fpp'], avg_fpp[inliers_idx]])
    inliers['lifted_pr'] = np.concatenate([inliers['lifted_pr'], lifted_pr[inliers_idx]])
    inliers['avg_rxp'] = np.concatenate([inliers['avg_rxp'], avg_rxp[inliers_idx]])
    inliers['lifted_rxp'] = np.concatenate([inliers['lifted_rxp'], lifted_rxp[inliers_idx]])
    inliers['avg_std'] = np.concatenate([inliers['avg_std'], avg_std[inliers_idx]])
    inliers['bias'] = np.concatenate([inliers['bias'], (meas_new[pair_i]-gt)[inliers_idx]])
    undetected_outliers['t'] = np.concatenate([undetected_outliers['t'], t[undetected_outliers_idx]])
    undetected_outliers['gt'] = np.concatenate([undetected_outliers['gt'], gt[undetected_outliers_idx]])
    undetected_outliers['avg_fpp'] = np.concatenate([undetected_outliers['avg_fpp'], avg_fpp[undetected_outliers_idx]])
    undetected_outliers['lifted_pr'] = np.concatenate([undetected_outliers['lifted_pr'], lifted_pr[undetected_outliers_idx]])
    undetected_outliers['avg_rxp'] = np.concatenate([undetected_outliers['avg_rxp'], avg_rxp[undetected_outliers_idx]])
    undetected_outliers['lifted_rxp'] = np.concatenate([undetected_outliers['lifted_rxp'], lifted_rxp[undetected_outliers_idx]])
    undetected_outliers['avg_std'] = np.concatenate([undetected_outliers['avg_std'], avg_std[undetected_outliers_idx]])
    undetected_outliers['bias'] = np.concatenate([undetected_outliers['bias'], (meas_new[pair_i]-gt)[undetected_outliers_idx]])

# fig, axs = plt.subplots(3,1)
# axs[0].set_title(r'Probability Density Function for different Metrics', size=20)
# axs[0].hist(inliers['lifted_rxp'],np.linspace(1.25,3,100),alpha=0.5,density=True, label=r'Inliers')
# axs[0].hist(undetected_outliers['lifted_rxp'],np.linspace(1.25,3,100),alpha=0.5,density=True, label=r'Outliers')
# axs[0].set_xlabel(r'$f(P_r)$')
# axs[0].legend(loc='upper right')
# axs[1].hist(inliers['lifted_pr'],np.linspace(0,2,100),alpha=0.5,density=True)
# axs[1].hist(undetected_outliers['lifted_pr'],np.linspace(0,2,100),alpha=0.5,density=True)
# axs[1].set_xlabel(r'$f(P_f)$')
# axs[2].hist(inliers['avg_rxp'] - inliers['avg_fpp'],np.linspace(0,15,100),alpha=0.5,density=True)
# axs[2].hist(undetected_outliers['avg_rxp'] - undetected_outliers['avg_fpp'],np.linspace(0,15,100),alpha=0.5,density=True)
# axs[2].set_xlabel(r'$P_r - P_f [dB]$')

print('Inliers: ' + str(np.size(inliers['gt'])))
print('Undetected outliers: ' + str(np.size(undetected_outliers['gt'])))

t = np.hstack((inliers['t'], undetected_outliers['t']))
bias = np.hstack((inliers['bias'], undetected_outliers['bias']))

X1 = np.vstack((inliers['lifted_rxp'], 
                inliers['lifted_pr'], 
                inliers['avg_rxp'] - inliers['avg_fpp'],
                ))
X2 = np.vstack((undetected_outliers['lifted_rxp'], 
                undetected_outliers['lifted_pr'], 
                undetected_outliers['avg_rxp'] - undetected_outliers['avg_fpp'],
                ))
X = np.hstack((X1,X2))
X = X.T

y1 = np.zeros((X1.shape[1]))
y2 = np.ones((X2.shape[1]))
y = np.hstack((y1,y2)) 

RandomForestClassifier.score(clf,X,y)

# %%
outliers_idx = clf.predict(X).astype(bool)
# dens_in = True
# dens_out = True
# fig, axs = plt.subplots(3,1)
# axs[0].set_title(r'Probability Density Function for different Metrics', size=20)
# axs[0].hist(X[:,0][~outliers_idx],np.linspace(1.25,3,100),alpha=0.5,density=dens_in, label=r'Inliers')
# axs[0].hist(X[:,0][outliers_idx],np.linspace(1.25,3,100),alpha=0.5,density=dens_out, label=r'Outliers')
# axs[0].set_xlabel(r'$f(P_r)$')
# axs[0].legend(loc='upper right')
# axs[1].hist(X[:,1][~outliers_idx],np.linspace(0,2,100),alpha=0.5,density=dens_in)
# axs[1].hist(X[:,1][outliers_idx],np.linspace(0,2,100),alpha=0.5,density=dens_out)
# axs[1].set_xlabel(r'$f(P_f)$')
# axs[2].hist(X[:,2][~outliers_idx],np.linspace(0,15,100),alpha=0.5,density=dens_in)
# axs[2].hist(X[:,2][outliers_idx],np.linspace(0,15,100),alpha=0.5,density=dens_out)
# axs[2].set_xlabel(r'$P_r - P_f [dB]$')

print('Inliers: ' + str(np.sum(~outliers_idx)))
print('Undetected outliers: ' + str(np.sum(outliers_idx)))


# %%
fig,axs = plt.subplots(1)

axs.scatter(t[~outliers_idx], bias[~outliers_idx])
axs.scatter(t[outliers_idx], bias[outliers_idx])

plt.show(block=True)

# %%
