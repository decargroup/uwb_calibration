# %%
from pyuwbcalib.utils import load, set_plotting_env
from pyuwbcalib.uwbcalibrate import UwbCalibrate
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
# matplotlib.use('Qt5Agg')

# Set the plotting environment
set_plotting_env()

# Load the PostProcess object
data = load("data.pickle")

# Instantiate a UwbCalibrate object, and remove static extremes
calib = UwbCalibrate(data, rm_static=True)

# Compute the raw bias measurements
bias_raw = np.array(calib.df['bias'])

# Correct antenna delays
calib.calibrate_antennas()

# Compute the antenna-delay-corrected measurements
bias_antenna_delay = np.array(calib.df['bias'])

# Correct power-correlated bias
calib.fit_power_model()

# Compute the fully-calibrated measurements
bias_fully_calib = np.array(calib.df['bias'])
# %%
# Plot the measurements pre- and post-correction.
bins = np.linspace(-0.5,1,100)
plt.hist(bias_raw,bins=bins, alpha=0.5, density=True)
plt.hist(bias_antenna_delay, bins=bins, alpha=0.5, density=True)
plt.hist(bias_fully_calib, bins=bins, alpha=0.5, density=True)
# %%
