# %%
from pyuwbcalib.postprocess import PostProcess
from scipy.interpolate import BSpline
from scipy.signal import butter,filtfilt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
matplotlib.use('Qt5Agg')

sns.set_theme()

# %%
class PositionEstimator(object):
    def __init__(self, ids, moment_arms, t, gt, uwb,
                 tag=1, filter_inputs=False, visualize=False):
        self.tag = tag
        self.ids = ids
        self.t = t
        self.moment_arms = moment_arms
        self.gt = gt
        self.uwb = uwb
        
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
        fs = 30.0       # sample rate, Hz
        cutoff = 0.5      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
        nyq = 0.5 * fs  # Nyquist Frequency
        order = 4       # sin wave can be approx represented as quadrat
        normal_cutoff = cutoff / nyq
        
        # Get the filter coefficients 
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        
        filtered_data = filtfilt(b, a, data)
        
        if self.visualize:
            #TODO!
            pass
        
        return filtered_data
    
    def run_kf(self, Q=np.eye(3), R=0.1, visualize=False):
        # Storage variables
        r_hist = np.zeros((3,self.n))
        P_hist = np.zeros((3,3,self.n))

        # Initial condition
        r_hat = self.gt['r'][self.tag][:,0]
        P_hat = np.eye(3)*0.01

        # Recursive filter
        for i,_ in enumerate(self.t):
            if i>0:
                r_hat, P_hat = self._propagate(r_hat, P_hat, Q, i)

            r_hat, P_hat = self._correct(r_hat, P_hat, R, i)

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

        if np.abs(y-y_check) < 300:
            r = r + (K * (y - y_check)).flatten()
            P = (np.eye(3) - K@C) @ P

        return r,P

    
# %%
# TODO: move a bunch of these, like get_velocity, to PostProcess
# if __name__ == "__main__":
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

main_tag = 3

# %%
### --- Get firmware-computed UWB measurements between main tag and all other tags --- ###
pad_idx = 50 # Remove some of the extreme measurements as sometimes they correspond to instances
             # where no mocap was recorded.
t_uwb = np.empty(0)
range = np.empty(0)
neighbour = np.empty(0,dtype=int)
for pair in raw_obj.ts_data:
    if main_tag in pair:
        neighbour_id = int(np.array(pair)[np.array(pair) != main_tag])
        t_new = raw_obj.ts_data[pair][pad_idx:-pad_idx,0]
        t_uwb = np.concatenate((t_uwb, t_new))
        range = np.concatenate((range, 
                                raw_obj.ts_data[pair][pad_idx:-pad_idx,raw_obj.range_idx]))
        neighbour = np.concatenate((neighbour, np.ones(np.size(t_new))*neighbour_id))

idx_sorted = np.argsort(t_uwb)
t = t_uwb[idx_sorted]/1e9
uwb = {'range': range[idx_sorted], 'neighbour': neighbour[idx_sorted]}
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
estimator = PositionEstimator(ids=tag_ids, 
                                moment_arms=moment_arms,
                                t=t, 
                                gt=mocap, 
                                uwb=uwb, 
                                tag = main_tag, 
                                filter_inputs=True,
                                visualize = True)
r_hist, P_hist = estimator.run_kf(Q=np.eye(3)*0.01,R=0.5)
    
    
    
# %%
