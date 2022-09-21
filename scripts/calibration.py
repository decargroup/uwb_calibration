# %%
from pyuwbcalib.utils import load, set_plotting_env
from pyuwbcalib.uwbcalibrate import UwbCalibrate
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
# matplotlib.use('Qt5Agg')

set_plotting_env()

data = load("data.pickle")

calib = UwbCalibrate(data, rm_static=True)
bias_raw = np.array(calib.df['bias'])

calib.calibrate_antennas()
bias_antenna_delay = np.array(calib.df['bias'])

calib.fit_power_model()
bias_fully_calib = np.array(calib.df['bias'])
# %%
bins = np.linspace(-0.5,1,100)
plt.hist(bias_raw,bins=bins, alpha=0.5, density=True)
plt.hist(bias_antenna_delay, bins=bins, alpha=0.5, density=True)
plt.hist(bias_fully_calib, bins=bins, alpha=0.5, density=True)
# %%
