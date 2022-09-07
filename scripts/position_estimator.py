# %%
from pyuwbcalib.postprocess import PostProcess
from pyuwbcalib.uwbcalibrate import UwbCalibrate
from scipy.interpolate import BSpline
from scipy.signal import butter,filtfilt
import matplotlib
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
matplotlib.use('Qt5Agg')

sns.set_theme()

# %%
class PositionEstimator(object):
    def __init__(self, ids, moment_arms, t, gt, uwb, 
                 machine='ifo001', std_spl=lambda x:1, filter_inputs=False, visualize=False):
        self.machine = machine
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
        for machine in self.gt['v_machine']:
            for dimension,v_1d in enumerate(self.gt['v_machine'][machine]):
                self.gt['v_machine'][machine][dimension] = self._butter_lowpass_filter(v_1d)
                
        for tag in self.gt['v_tag']:
            for dimension,v_1d in enumerate(self.gt['v_tag'][tag]):
                self.gt['v_tag'][tag][dimension] = self._butter_lowpass_filter(v_1d)
                
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
        # filtered_data += np.random.randn(np.size(filtered_data))*0.1
        
        if self.visualize:
            #TODO!
            pass
        
        return filtered_data
    
    def run_kf(self, Q=np.eye(3), R=0.1, visualize=False):
        # Storage variables
        r_hist = np.zeros((3,self.n))
        P_hist = np.zeros((3,3,self.n))

        # Initial condition
        r_hat = self.gt['r_machine'][self.machine][:,0] + np.random.randn(3)*0.3
        P_hat = np.eye(3)*0.3**2

        # Recursive filter
        for i,_ in enumerate(self.t):
            if i>0:
                r_hat, P_hat = self._propagate(r_hat, P_hat, Q, i)

            lifted_fpp = self.uwb['lifted_fpp'][i]
            std_fpp = self.std_spl(lifted_fpp)
            if std_fpp < 0.05:
                std_fpp = 0.05
            R_fpp = R*(std_fpp**2)
            r_hat, P_hat = self._correct(r_hat, P_hat, R_fpp, i)

            r_hist[:,i] = r_hat
            P_hist[:,:,i] = P_hat

        return r_hist, P_hist

    def _propagate(self, r, P, Q, i):
        dt = self.t[i]-self.t[i-1]
        v = self.gt['v_machine'][self.machine][:,i-1]
        r = r + dt*v
        P = P + dt**2/dt*Q
        return r,P

    def _correct(self, r, P, R, i):
        y = self.uwb['range'][i]
        main_tag = self.uwb['main_tag'][i]
        neighbour = self.uwb['neighbour'][i]
        
        r_neighbour = self.gt['r_tag'][neighbour][:,i]
        
        rot = Rotation.from_quat(self.gt['q_machine'][self.machine][:,i])
        tag_idx = int(np.where(np.array(tag_ids[self.machine]) == main_tag)[0])
        arm = self.moment_arms[self.machine][tag_idx]
        r_main_tag = r + (rot.as_matrix() @ arm).T

        # r_true = self.gt['r'][self.tag][:,i]
        # y = np.linalg.norm(r_true - r_neighbour,axis=0)

        y_r = r_main_tag - r_neighbour
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
# tag_ids={'ifo001': [1,2],
#          'ifo002': [3,4],
#          'ifo003': [5,6]}
tag_ids={'ifo001': [1,7],
         'ifo002': [3,4],
         'ifo003': [5,6]}
moment_arms={'ifo001': [[0.13189,-0.17245,-0.05249], [-0.17542,0.15712,-0.05307]],
             'ifo002': [[0.16544,-0.15085,-0.03456], [-0.15467,0.16972,-0.01680]],
             'ifo003': [[0.16685,-0.18113,-0.05576], [-0.13485,0.15468,-0.05164]]}
filename = "datasets/2022_09_01_tag7/random0/merged.bag"
raw_obj = PostProcess(filename,
                      tag_ids,
                      moment_arms,
                      num_meas=-1)

main_machine = 'ifo002'

# %%
### --- Get RAW FIRMWARE-COMPUTED UWB measurements between main tag and all other tags --- ###
pad_idx = 100 # Remove some of the extreme measurements as sometimes they correspond to instances
             # where no mocap was recorded.
t_uwb = np.empty(0)
range = np.empty(0)
main_tag = np.empty(0,dtype=int)
neighbour = np.empty(0,dtype=int)
lifted_fpp = np.empty(0)
for pair in raw_obj.ts_data:
    bool_list = [tag_ids[main_machine][0] in pair, tag_ids[main_machine][1] in pair]
    if np.any(bool_list):
        idx = int(np.where(bool_list)[0])
        main_tag_id = tag_ids[main_machine][idx]
        neighbour_id = int(np.array(pair)[np.array(pair) != main_tag_id])
        t_new = raw_obj.ts_data[pair][pad_idx:-pad_idx,0]
        t_uwb = np.concatenate((t_uwb, t_new))

        avg_fpp = 0.5*(raw_obj.ts_data[pair][pad_idx:-pad_idx,raw_obj.fpp1_idx] \
                       + raw_obj.ts_data[pair][pad_idx:-pad_idx,raw_obj.fpp2_idx])
        lifted_fpp = np.concatenate((lifted_fpp,
                                     raw_obj.lift(avg_fpp)))
        
        range = np.concatenate((range, 
                                raw_obj.ts_data[pair][pad_idx:-pad_idx,raw_obj.range_idx]))
        neighbour = np.concatenate((neighbour, np.ones(np.size(t_new))*neighbour_id))
        main_tag = np.concatenate((main_tag, np.ones(np.size(t_new))*main_tag_id))

idx_sorted = np.argsort(t_uwb)
t = t_uwb[idx_sorted]/1e9
uwb = {'range': range[idx_sorted],
       'neighbour': neighbour[idx_sorted], 
       'main_tag': main_tag[idx_sorted],
       'lifted_fpp': lifted_fpp[idx_sorted]}

# %%
### --- Get CALIBRATED UWB measurements between main tag and all other tags --- ###
raw_obj_calib = PostProcess("datasets/2022_09_01_tag7/bias_calibration0/merged.bag",
                            tag_ids,
                            moment_arms,
                            num_meas=-1)

calib_obj_og = UwbCalibrate(raw_obj_calib, rm_static=True)
delays = calib_obj_og.calibrate_antennas()
calib_obj_og.correct_antenna_delay(delays)
calib_obj_og.fit_model(std_window=25, chi_thresh=22.8, merge_pairs=True)

calib_obj = UwbCalibrate(raw_obj, rm_static=False)
calib_obj.correct_antenna_delay(delays)

range = np.empty(0)
for pair in raw_obj.ts_data:
    bool_list = [tag_ids[main_machine][0] in pair, tag_ids[main_machine][1] in pair]
    if np.any(bool_list):
        meas_new = calib_obj.compute_range_meas(pair)
        range = np.concatenate((range, meas_new[pad_idx:-pad_idx]))

range -= calib_obj_og.spl(lifted_fpp[idx_sorted])

uwb_calibrated = {'range': range[idx_sorted], 
                  'neighbour': neighbour[idx_sorted],
                  'main_tag': main_tag[idx_sorted], 
                  'lifted_fpp': lifted_fpp[idx_sorted]}

# %%
### --- Get absolute position and velocity of every tag --- ###
r_tag = {} # position of tags
v_tag = {} # velocity of tags
r_machine = {} # position of drones
v_machine = {} # velocity of drones
q_machine = {} # quaternion-parametrized attitude of drones

# Iterate through machines
for machine in tag_ids:
    t_iter = raw_obj.t_r[machine]/1e9
    
    # Machines -------------------
    r_iter = raw_obj.r[machine]
    q_iter = raw_obj.rot[machine].as_quat().T
    
    # Quaternion spline
    q_spl = [BSpline(t_iter, q_iter[0,:], k=3),
             BSpline(t_iter, q_iter[1,:], k=3),
             BSpline(t_iter, q_iter[2,:], k=3),
             BSpline(t_iter, q_iter[3,:], k=3)]
    # Position spline
    r_spl = [BSpline(t_iter, r_iter[0,:], k=3),
             BSpline(t_iter, r_iter[1,:], k=3),
             BSpline(t_iter, r_iter[2,:], k=3)]
    # Velocity spline
    v_spl = [r_spl[0].derivative(),
             r_spl[1].derivative(),
             r_spl[2].derivative()]

    # Interpolated quaternion
    q_machine[machine] = np.vstack((q_spl[0](t),
                                    q_spl[1](t),
                                    q_spl[2](t),
                                    q_spl[3](t)))
    # Interpolated position
    r_machine[machine] = np.vstack((r_spl[0](t),
                                    r_spl[1](t),
                                    r_spl[2](t)))
    # Absolute velocity
    v_machine[machine] = np.vstack((v_spl[0](t),
                                    v_spl[1](t),
                                    v_spl[2](t)))
    
    # Iterate through tags for every machine
    for i,tag in enumerate(raw_obj.tag_ids[machine]):
        # Tags -------------------
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
        r_tag[tag] = np.vstack((r_spl[0](t),
                                r_spl[1](t),
                                r_spl[2](t)))
        # Absolute velocity
        v_tag[tag] = np.vstack((v_spl[0](t),
                                v_spl[1](t),
                                v_spl[2](t)))
        
mocap = {'q_machine':q_machine, 
         'r_machine':r_machine,
         'v_machine':v_machine,
         'r_tag':r_tag,
         'v_tag':v_tag}

# %%
# Run the filters
np.random.seed(35)
gt_pos = mocap['r_machine'][main_machine]

estimator = PositionEstimator(ids=tag_ids, 
                                moment_arms=moment_arms,
                                t=t, 
                                gt=mocap.copy(), 
                                uwb=uwb_calibrated.copy(), 
                                machine = main_machine, 
                                filter_inputs=True,
                                visualize = True)
r_hist_calib, P_hist_calib = estimator.run_kf(Q=np.eye(3)*0.5,R=0.11**2)
print(np.mean(np.linalg.norm(r_hist_calib - gt_pos, axis=0)))

estimator = PositionEstimator(ids=tag_ids, 
                                moment_arms=moment_arms,
                                t=t, 
                                gt=mocap.copy(), 
                                uwb=uwb.copy(), 
                                machine = main_machine, 
                                filter_inputs=True,
                                visualize = True)
r_hist, P_hist = estimator.run_kf(Q=np.eye(3)*0.5,R=0.11**2)

# TODO: Use std_spl
estimator_calib = PositionEstimator(ids=tag_ids, 
                                    moment_arms=moment_arms,
                                    t=t, 
                                    gt=mocap.copy(), 
                                    uwb=uwb_calibrated.copy(), 
                                    machine = main_machine, 
                                    filter_inputs=True,
                                    visualize = True,
                                    std_spl = calib_obj_og.std_spl
                                    )
r_hist_calib_wVariance, P_hist_calib_wVariance = estimator_calib.run_kf(Q=np.eye(3)*0.5,R=1)

print(np.mean(np.linalg.norm(r_hist_calib_wVariance - gt_pos, axis=0)))
print(np.mean(np.linalg.norm(r_hist - gt_pos, axis=0)))
# %%
t = t-t[0]
fig,axs = plt.subplots(3,1, sharex='all', sharey='all')

axs[0].set_title(r"Position estimator")
axs[0].plot(t,r_hist_calib[0] - gt_pos[0], label="Calibrated")
axs[0].plot(t,r_hist_calib_wVariance[0] - gt_pos[0], label="Calibrated w/ Variance")
axs[0].plot(t,r_hist[0] - gt_pos[0], label="Raw")

axs[0].fill_between(t,
            -3*np.sqrt(P_hist_calib[0,0,:]),
            3*np.sqrt(P_hist_calib[0,0,:]),
            # color='blue',
            alpha=0.5,
            label=r"99.7% confidence interval",
            )

axs[0].fill_between(t,
            -3*np.sqrt(P_hist_calib_wVariance[0,0,:]),
            3*np.sqrt(P_hist_calib_wVariance[0,0,:]),
            # color='red',
            alpha=0.5,
            label=r"99.7% confidence interval",
            )

axs[0].fill_between(t,
            -3*np.sqrt(P_hist[0,0,:]),
            3*np.sqrt(P_hist[0,0,:]),
            # color='red',
            alpha=0.5,
            label=r"99.7% confidence interval",
            )

axs[0].set_ylabel(r'$e_x$ [m]')

axs[1].plot(t,r_hist_calib[1] - gt_pos[1])
axs[1].plot(t,r_hist_calib_wVariance[1] - gt_pos[1])
axs[1].plot(t,r_hist[1] - gt_pos[1])

axs[1].fill_between(t,
            -3*np.sqrt(P_hist_calib[1,1,:]),
            3*np.sqrt(P_hist_calib[1,1,:]),
            # color='blue',
            alpha=0.5,
            label=r"99.7% confidence interval",
            )

axs[1].fill_between(t,
            -3*np.sqrt(P_hist_calib_wVariance[1,1,:]),
            3*np.sqrt(P_hist_calib_wVariance[1,1,:]),
            # color='blue',
            alpha=0.5,
            label=r"99.7% confidence interval",
            )

axs[1].fill_between(t,
            -3*np.sqrt(P_hist[1,1,:]),
            3*np.sqrt(P_hist[1,1,:]),
            # color='red',
            alpha=0.5,
            label=r"99.7% confidence interval",
            )

axs[1].set_ylabel(r'$e_y$ [m]')

axs[2].plot(t,r_hist_calib[2] - gt_pos[2])
axs[2].plot(t,r_hist_calib_wVariance[2] - gt_pos[2])
axs[2].plot(t,r_hist[2] - gt_pos[2])

axs[2].fill_between(t,
            -3*np.sqrt(P_hist_calib[2,2,:]),
            3*np.sqrt(P_hist_calib[2,2,:]),
            # color='blue',
            alpha=0.5,
            label=r"99.7% confidence interval",
            )

axs[2].fill_between(t,
            -3*np.sqrt(P_hist_calib_wVariance[2,2,:]),
            3*np.sqrt(P_hist_calib_wVariance[2,2,:]),
            # color='blue',
            alpha=0.5,
            label=r"99.7% confidence interval",
            )

# axs[2].fill_between(t,
#             -3*np.sqrt(P_hist[2,2,:]),
#             3*np.sqrt(P_hist[2,2,:]),
#             # color='red',
#             alpha=0.5,
#             label=r"99.7% confidence interval",
#             )

axs[2].set_ylim(-3,3)
axs[2].set_xlabel(r'$t$ [s]')
axs[2].set_ylabel(r'$e_z$ [m]')

axs[0].legend(fontsize=20,loc='upper right')

fig2,axs = plt.subplots(1)

axs.plot(t,np.linalg.norm(r_hist - gt_pos, axis=0), label="Raw")
axs.plot(t,np.linalg.norm(r_hist_calib - gt_pos, axis=0), label="Calibrated")
axs.plot(t,np.linalg.norm(r_hist_calib_wVariance - gt_pos, axis=0), label="Calibrated w/ Variance")
axs.set_ylim(0,3.5)
axs.legend(fontsize=50,loc='upper right')
axs.set_xlabel(r'$t$ [s]', fontsize=50)
axs.set_ylabel(r'RMSE [m]', fontsize=50)
axs.tick_params(axis='both', labelsize=50)
# axs.plot(np.linalg.norm(r_hist_calib - gt_pos, axis=0) - \
        #  np.linalg.norm(r_hist - gt_pos, axis=0), 'blue')

fig.savefig('figs/pos_estimator_3sigma_bound.pdf')
fig2.savefig('figs/pos_estimator_error_norm.pdf')

plt.show(block=True)
# %%

# TODO:: 2) USE THE STD SPLINE THING
#        3) show plots using the Husky
# %%
