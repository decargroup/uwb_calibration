# %%
import numpy as np
import matplotlib.pyplot as plt
from pyuwbcalib.uwbcalibrate import ApplyCalibration
from pyuwbcalib.postprocess import PostProcess
from pyuwbcalib.machine import RosMachine
from pyuwbcalib.postprocess import PostProcess
from pyuwbcalib.utils import load, read_anchor_positions, merge_calib_results
from configparser import ConfigParser, ExtendedInterpolation
np.random.seed(12345)

# TODO: replace with examples available in the repo
# The calibration file
calib_root = "/home/shalaby/Desktop/datasets/miluv_dataset/2024_01_12/"
calib_results = merge_calib_results([
    # load(calib_root + "bias_calibration_anchors0/uwb_calib_results.pickle"),
    load(calib_root + "bias_calibration_anchors1/uwb_calib_results.pickle"),
    # load(calib_root + "bias_calibration_tags0/uwb_calib_results.pickle"),
    load(calib_root + "bias_calibration_tags1/uwb_calib_results.pickle"),
])

# The configuration file
config_file = './config/miluv_dataset_17_01_2024_opt_cov_trial1.config'

# Parse through the configuration file
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read(config_file)

# Create a RosMachine object for every machine
machines = {}
for i,machine in enumerate(parser['MACHINES']):
    machine_id = parser['MACHINES'][machine]
    machines[machine_id] = RosMachine(parser, i)
    
# Read anchor positions
anchor_positions = read_anchor_positions(parser)

# # %%
# Process and merge the data from all the machines
data = PostProcess(machines, anchor_positions)

# # Compute the raw bias measurements
bias_raw = np.array(data.df['bias'])

data.df.drop(data.df[np.abs(data.df["range"])>10].index, inplace=True)

data.df = ApplyCalibration.antenna_delays(
    data.df, 
    calib_results['delays'],
    max_value = 1e9 * (1.0 / 499.2e6 / 128.0) * 2.0**32
)

# Compute the antenna-delay-corrected measurements
bias_antenna_delay = np.array(data.df['bias'])

data.df = ApplyCalibration.power(
    data.df, 
    calib_results['bias_spl'], 
    calib_results['std_spl'], 
    max_value = 1e9 * (1.0 / 499.2e6 / 128.0) * 2.0**32,
)

# Compute the fully-calibrated measurements
bias_fully_calib = np.array(data.df['bias'])

print(data.df[np.abs(data.df["range"])>10])

# Save the fully-calibrated measurements
data.df.to_csv(r'./uwb_range_merged.csv')
# %%
# Plot the measurements pre- and post-correction.
bins = np.linspace(-0.5,1,100)
plt.hist(bias_raw,bins=bins, alpha=0.5, density=True)
plt.hist(bias_antenna_delay, bins=bins, alpha=0.5, density=True)
plt.hist(bias_fully_calib, bins=bins, alpha=0.5, density=True)
plt.grid()

plt.show()
# %%
