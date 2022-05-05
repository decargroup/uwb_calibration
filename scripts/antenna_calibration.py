# %%
from pyuwbcalib.uwbcalibrate import UwbCalibrate
from pyuwbcalib.postprocess import PostProcess
import matplotlib.pyplot as plt
import numpy as np

raw_obj = PostProcess(folder_prefix="datasets/2022_05_02/",
                      file_prefix="test",
                      num_of_recordings=1,
                      num_meas=-1,
                      tag_ids=[4,1,3])

# %%
kf = False
GP = True
antenna_delay = False
initiator_id = 4
target_id = 1
pair = (initiator_id, target_id)
# raw_obj.visualize_raw_data(pair=(initiator_id,target_id))

# %%
# TODO: Surely there is a better way to do this??
calib_obj = UwbCalibrate(raw_obj)

meas_old = calib_obj.compute_range_meas(pair,
                                        visualize=False)

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
    t = calib_obj.time_intervals[0][pair]['t']
    plt.plot(t, meas_old, linewidth=1, label="Raw")
    plt.plot(t, meas_filtered, linewidth=1, label="Calibrated")
    plt.plot(t, calib_obj.time_intervals[0][pair]['r_gt'])
    ax.legend()
    plt.show()

if GP:
    calib_obj.fit_gp(pair)

# %%
if antenna_delay:
    # Calibrate the antenna delays
    delays = calib_obj.calibrate_antennas()
    print(delays)
    calib_obj.correct_antenna_delay(1, delays["Module 1"])
    calib_obj.correct_antenna_delay(2, delays["Module 2"])
    calib_obj.correct_antenna_delay(3, delays["Module 3"])

    meas_new = calib_obj.compute_range_meas(initiator_id, target_id)

    #%%
    fig, ax = plt.subplots()
    ax.set_title("Measurements " + str(initiator_id) + "->" + str(target_id))
    ax.set_xlabel("Measurement Number")
    ax.set_ylabel("Range [m]")
    ax.set_ylim(0, 4)
    plt.plot(meas_old, linewidth=1, label="Raw")
    plt.plot(meas_new, linewidth=1, label="Calibrated")
    plt.plot(x.data[str(initiator_id) + "->" + str(target_id)]["gt"], linewidth=3, label="GT")
    ax.legend()
    plt.show()

    # %%
