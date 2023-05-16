# %%
from pyuwbcalib.utils import load
from pyuwbcalib.uwbcalibrate import ApplyCalibration
from pyuwbcalib.postprocess import PostProcess
import numpy as np
import matplotlib.pyplot as plt
# TODO: the passive listening calibration is untested. Could be more verbose in this example.

data: PostProcess = load("data.pickle")
calib_results = load("calib_results.pickle")

# Compute the raw bias measurements
bias_raw = np.array(data.df['bias'])

data.df = ApplyCalibration.antenna_delays(
    data.df, 
    calib_results['delays']
)
data.df_passive = ApplyCalibration.antenna_delays_passive(
    data.df_passive, 
    calib_results['delays']
)

# Compute the antenna-delay-corrected measurements
bias_antenna_delay = np.array(data.df['bias'])

data.df = ApplyCalibration.power(
    data.df, 
    calib_results['bias_spl'], 
    calib_results['std_spl']
)
data.df_passive = ApplyCalibration.power_passive(
    data.df_passive, 
    calib_results['bias_spl'], 
    calib_results['std_spl']
)

# Compute the fully-calibrated measurements
bias_fully_calib = np.array(data.df['bias'])

# %%
# Plot the measurements pre- and post-correction.
bins = np.linspace(-0.5,1,100)
plt.hist(bias_raw,bins=bins, alpha=0.5, density=True)
plt.hist(bias_antenna_delay, bins=bins, alpha=0.5, density=True)
plt.hist(bias_fully_calib, bins=bins, alpha=0.5, density=True)
plt.grid()

plt.show()
# %%
