# %%
from pyuwbcalib.uwbcalibrate import UwbCalibrate
from pyuwbcalib.postprocess import PostProcess
from scipy.interpolate import UnivariateSpline,BSpline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
matplotlib.use('Qt5Agg')

sns.set_theme()

# %%
class PositionEstimator(object):
    def __init__(self):
        pass
    
    def func(self):
        pass
    
# %%
# TODO: move a bunch of these, like get_velocity, to PostProcess
if __name__ == "__main__":
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
    
    main_tag = 1
    
    ### --- Get absolute position and velocity of every tag --- ###
    r = {} # position
    v = {} # velocity
    mocap = {} # all data
    # Iterate through machines
    for machine in tag_ids:
        # Iterate through tags for every machine
        t = raw_obj.t_r[machine]/1e9
        for i,tag in enumerate(raw_obj.tag_ids[machine]):
            r[tag] = raw_obj.r[machine] + (raw_obj.rot[machine].as_matrix() @ moment_arms[machine][i]).T
            
            # Position spline
            r_spl = [BSpline(t, r[tag][0,:], k=3),
                     BSpline(t, r[tag][1,:], k=3),
                     BSpline(t, r[tag][2,:], k=3)]
            # Velocity spline
            v_spl = [r_spl[0].derivative(),
                     r_spl[1].derivative(),
                     r_spl[2].derivative()]
            # Absolute velocity
            v[tag] = np.vstack((v_spl[0](t),
                                v_spl[1](t),
                                v_spl[2](t)))
            
        mocap[machine] = {'t':t, 'r':r, 'v':v}
        
    ### --- Get firmware-computed UWB measurements between main tag and all other tags --- ###
    uwb = {}
    for pair in raw_obj.ts_data:
        if main_tag in pair:
            t = raw_obj.ts_data[pair][:,0]
            range = raw_obj.ts_data[pair][:,raw_obj.range_idx]
            uwb[pair] = {'t':t, 'range':range}
    
    # %%
    # estimator = PositionEstimator(main_tag, tag_ids, moment_arms, mocap, uwb)
    # estimator.lowpass_inputs()
    # estimator.run_kf(visualize = True)
    
    
    
# %%
