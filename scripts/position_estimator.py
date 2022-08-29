# %%
from pyuwbcalib.postprocess import PostProcess
from pyuwbcalib.uwbcalibrate import UwbCalibrate
from scipy.interpolate import BSpline
from scipy.signal import butter,filtfilt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# matplotlib.use('Qt5Agg')

sns.set_theme()

# %%
class PositionEstimator(object):
    def __init__(self, ids, moment_arms, t, gt, uwb, 
                 tag=1, std_spl=lambda x:1, filter_inputs=False, visualize=False):
        self.tag = tag
        self.ids = ids
        self.t = t
        self.moment_arms = moment_arms
        self.gt = gt
        self.uwb = uwb
        self.std_spl = std_spl
        
        self.visualize = visualize

        if filter_inputs:
            self._filter_velocity_inputs()
        
        self.n = np.size(t) # Number of UWB measurements the tag is associated with
    
    def _filter_velocity_inputs(self):
        for tag in self.gt['v']:
            for dimension,v_1d in enumerate(self.gt['v'][tag]):
                self.gt['v'][tag][dimension] = self._butter_lowpass_filter(v_1d)
                
    def _butter_lowpass_filter(self,data):
        # Some parameters
        T = 5.0         # Sample Period
        fs = 100.0       # sample rate, Hz
        cutoff = 0.1      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
        nyq = 0.5 * fs  # Nyquist Frequency
        order = 3       # sin wave can be approx represented as quadrat
        normal_cutoff = cutoff / nyq
        
        # Get the filter coefficients 
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        
        filtered_data = filtfilt(b, a, data)
        # filtered_data += np.random.randn(np.size(filtered_data))*0.3
        
        if self.visualize:
            #TODO!
            pass
        
        return filtered_data
    
    def run_kf(self, Q=np.eye(3), R=0.1, visualize=False):
        # Storage variables
        r_hist = np.zeros((3,self.n))
        P_hist = np.zeros((3,3,self.n))

        # Initial condition
        r_hat = self.gt['r'][self.tag][:,0] + np.random.randn(3)*2
        P_hat = np.eye(3)*4

        # Recursive filter
        for i,_ in enumerate(self.t):
            if i>0:
                r_hat, P_hat = self._propagate(r_hat, P_hat, Q, i)

            lifted_fpp = self.uwb['lifted_fpp'][i]
            std_fpp = self.std_spl(lifted_fpp)
            R_fpp = R*(std_fpp**2)
            r_hat, P_hat = self._correct(r_hat, P_hat, R_fpp, i)

            r_hist[:,i] = r_hat
            P_hist[:,:,i] = P_hat

        return r_hist, P_hist

    def _propagate(self, r, P, Q, i):
        dt = self.t[i]-self.t[i-1]
        v = self.gt['v'][self.tag][:,i-1]
        r = r + dt*v
        P = P + dt**2/dt*Q
        return r,P

    def _correct(self, r, P, R, i):
        y = self.uwb['range'][i]
        neighbour = self.uwb['neighbour'][i]
        r_neighbour = self.gt['r'][neighbour][:,i]

        # r_true = self.gt['r'][self.tag][:,i]
        # y = np.linalg.norm(r_true - r_neighbour,axis=0)

        y_r = r - r_neighbour
        y_check = np.linalg.norm(y_r,axis=0)

        C = np.reshape(0.5/y_check * (y_r),(1,3))

        S = (C @ P @ C.T + R)
        K = P @ C.T / S 

        innov = y - y_check
        if self._nis_test_scalar(innov, S) or i<10:
            r = r + (K * (innov)).flatten()
            P = (np.eye(3) - K@C) @ P

        return r,P

    @staticmethod
    def _nis_test_scalar(innov, S):
        eps = innov**2 / S
        return eps < 10.8

    
# %%
# TODO: move a bunch of these, like get_velocity, to PostProcess
# if __name__ == "__main__":
tag_ids={'ifo001': [1,2],
         'ifo002': [3,4],
         'ifo003': [5,6]}
moment_arms={'ifo001': [[0.13189,-0.17245,-0.05249], [-0.17542,0.15712,-0.05307]],
             'ifo002': [[0.16544,-0.15085,-0.03456], [-0.15467,0.16972,-0.01680]],
             'ifo003': [[0.16685,-0.18113,-0.05576], [-0.13485,0.15468,-0.05164]]}
filename = "datasets/2022_08_03/bias_calibration_new2/merged.bag"
raw_obj = PostProcess(filename,
                      tag_ids,
                      moment_arms,
                      num_meas=-1)

main_tag = 5

# %%
### --- Get RAW FIRMWARE-COMPUTED UWB measurements between main tag and all other tags --- ###
pad_idx = 100 # Remove some of the extreme measurements as sometimes they correspond to instances
             # where no mocap was recorded.
t_uwb = np.empty(0)
range = np.empty(0)
neighbour = np.empty(0,dtype=int)
lifted_fpp = np.empty(0)
for pair in raw_obj.ts_data:
    if main_tag in pair:
        neighbour_id = int(np.array(pair)[np.array(pair) != main_tag])
        t_new = raw_obj.ts_data[pair][pad_idx:-pad_idx,0]
        t_uwb = np.concatenate((t_uwb, t_new))

        avg_fpp = 0.5*(raw_obj.ts_data[pair][pad_idx:-pad_idx,raw_obj.fpp1_idx] \
                       + raw_obj.ts_data[pair][pad_idx:-pad_idx,raw_obj.fpp2_idx])
        lifted_fpp = np.concatenate((lifted_fpp,
                                    raw_obj.lift(avg_fpp)))
        
        range = np.concatenate((range, 
                                raw_obj.ts_data[pair][pad_idx:-pad_idx,raw_obj.range_idx]))
        neighbour = np.concatenate((neighbour, np.ones(np.size(t_new))*neighbour_id))

idx_sorted = np.argsort(t_uwb)
t = t_uwb[idx_sorted]/1e9
uwb = {'range': range[idx_sorted],
       'neighbour': neighbour[idx_sorted], 
       'lifted_fpp': lifted_fpp[idx_sorted]}

# %%
### --- Get CALIBRATED UWB measurements between main tag and all other tags --- ###
if filename != "datasets/2022_08_03/bias_calibration_new/merged.bag":
    raw_obj_calib = PostProcess("datasets/2022_08_03/bias_calibration_new/merged.bag",
                                tag_ids,
                                moment_arms,
                                num_meas=-1)
else:
    raw_obj_calib = raw_obj

calib_obj_og = UwbCalibrate(raw_obj_calib, rm_static=True)
delays = calib_obj_og.calibrate_antennas()
calib_obj_og.correct_antenna_delay(delays)
calib_obj_og.fit_model(std_window=25, chi_thresh=22.8, merge_pairs=True)

calib_obj = UwbCalibrate(raw_obj, rm_static=False)
calib_obj.correct_antenna_delay(delays)

range = np.empty(0)
for pair in raw_obj.ts_data:
    if main_tag in pair:
        meas_new = calib_obj.compute_range_meas(pair)
        range = np.concatenate((range, meas_new[pad_idx:-pad_idx]))

range -= calib_obj_og.spl(lifted_fpp[idx_sorted])

uwb_calibrated = {'range': range[idx_sorted], 
                  'neighbour': neighbour[idx_sorted], 
                  'lifted_fpp': lifted_fpp[idx_sorted]}

# %%
### --- Get absolute position and velocity of every tag --- ###
r = {} # position
v = {} # velocity
# Iterate through machines
for machine in tag_ids:
    # Iterate through tags for every machine
    for i,tag in enumerate(raw_obj.tag_ids[machine]):
        t_iter = raw_obj.t_r[machine]/1e9
        r_iter = raw_obj.r[machine] \
                    + (raw_obj.rot[machine].as_matrix() @ moment_arms[machine][i]).T
        
        # Position spline
        r_spl = [BSpline(t_iter, r_iter[0,:], k=3),
                 BSpline(t_iter, r_iter[1,:], k=3),
                 BSpline(t_iter, r_iter[2,:], k=3)]
        # Velocity spline
        v_spl = [r_spl[0].derivative(),
                 r_spl[1].derivative(),
                 r_spl[2].derivative()]

        # Interpolated position
        r[tag] = np.vstack((r_spl[0](t),
                            r_spl[1](t),
                            r_spl[2](t)))
        # Absolute velocity
        v[tag] = np.vstack((v_spl[0](t),
                            v_spl[1](t),
                            v_spl[2](t)))
        
mocap = {'r':r, 'v':v}

# %%
np.random.seed(10)

estimator = PositionEstimator(ids=tag_ids, 
                                moment_arms=moment_arms,
                                t=t, 
                                gt=mocap, 
                                uwb=uwb, 
                                tag = main_tag, 
                                filter_inputs=True,
                                visualize = True)
r_hist, P_hist = estimator.run_kf(Q=np.eye(3)*0.5/5,R=0.15*2)

# TODO: Use std_spl
estimator_calib = PositionEstimator(ids=tag_ids, 
                                    moment_arms=moment_arms,
                                    t=t, 
                                    gt=mocap, 
                                    uwb=uwb_calibrated, 
                                    tag = main_tag, 
                                    filter_inputs=True,
                                    visualize = True,
                                    std_spl = calib_obj_og.std_spl
                                    )
r_hist_calib, P_hist_calib = estimator_calib.run_kf(Q=np.eye(3)*0.5/5,R=1*2)

# %%
fig,axs = plt.subplots(3,1, sharex='all', sharey='all')

axs[0].set_title(r"Position estimator")
axs[0].plot(t,r_hist_calib[0] - mocap['r'][main_tag][0], 'blue', label="Calibrated")
axs[0].plot(t,r_hist[0] - mocap['r'][main_tag][0], 'red', label="Raw")

axs[0].fill_between(t,
            -3*np.sqrt(P_hist_calib[0,0,:]),
            3*np.sqrt(P_hist_calib[0,0,:]),
            'blue',
            alpha=0.8,
            label=r"99.7% confidence interval",
            )

axs[0].fill_between(t,
            -3*np.sqrt(P_hist[0,0,:]),
            3*np.sqrt(P_hist[0,0,:]),
            'red',
            alpha=0.5,
            label=r"99.7% confidence interval",
            )

axs[0].set_ylim(-2,2)
axs[0].set_ylabel(r'$e_x$ [m]')

axs[1].plot(t,r_hist_calib[1] - mocap['r'][main_tag][1], 'blue')
axs[1].plot(t,r_hist[1] - mocap['r'][main_tag][1], 'red')

axs[1].fill_between(t,
            -3*np.sqrt(P_hist_calib[1,1,:]),
            3*np.sqrt(P_hist_calib[1,1,:]),
            'blue',
            alpha=0.8,
            label=r"99.7% confidence interval",
            )

axs[1].fill_between(t,
            -3*np.sqrt(P_hist[1,1,:]),
            3*np.sqrt(P_hist[1,1,:]),
            'red',
            alpha=0.5,
            label=r"99.7% confidence interval",
            )

axs[1].set_ylim(-2,2)
axs[1].set_ylabel(r'$e_y$ [m]')

axs[2].plot(t,r_hist_calib[2] - mocap['r'][main_tag][2], 'blue')
axs[2].plot(t,r_hist[2] - mocap['r'][main_tag][2], 'red')

axs[2].fill_between(t,
            -3*np.sqrt(P_hist_calib[2,2,:]),
            3*np.sqrt(P_hist_calib[2,2,:]),
            'blue',
            alpha=0.8,
            label=r"99.7% confidence interval",
            )

axs[2].fill_between(t,
            -3*np.sqrt(P_hist[2,2,:]),
            3*np.sqrt(P_hist[2,2,:]),
            'red',
            alpha=0.5,
            label=r"99.7% confidence interval",
            )

axs[2].set_ylim(-2,2)
axs[2].set_xlabel(r'$t$ [s]')
axs[2].set_ylabel(r'$e_z$ [m]')

axs[0].legend(loc='upper right')

fig2,axs = plt.subplots(1)

axs.plot(np.linalg.norm(r_hist_calib - mocap['r'][main_tag], axis=0), 'blue')
axs.plot(np.linalg.norm(r_hist - mocap['r'][main_tag], axis=0), 'orange')
# axs.plot(np.linalg.norm(r_hist_calib - mocap['r'][main_tag], axis=0) - \
        #  np.linalg.norm(r_hist - mocap['r'][main_tag], axis=0), 'blue')

print(np.mean(np.linalg.norm(r_hist_calib - mocap['r'][main_tag], axis=0)))
print(np.mean(np.linalg.norm(r_hist - mocap['r'][main_tag], axis=0)))

fig.savefig('figs/pos_estimator_3sigma_bound.pdf')
fig2.savefig('figs/pos_estimator_error_norm.pdf')

plt.show(block=True)

# %%

# TODO:: 2) USE THE STD SPLINE THING
#        3) show plots using the Husky
# %%
