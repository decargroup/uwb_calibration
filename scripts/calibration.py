# %%
from pyuwbcalib.postprocess import load
from pyuwbcalib.uwbcalibrate import UwbCalibrate
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# matplotlib.use('Qt5Agg')
from matplotlib import rc
rc('text', usetex=True)

sns.set_palette("colorblind")
sns.set_style('whitegrid')

plt.rc('figure', figsize=(16, 9))
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=35)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rc('legend', facecolor=[1,1,1])
plt.rc('legend', fontsize=30)
plt.rcParams['figure.constrained_layout.use'] = True

c = 299702547
data = load()

calib = UwbCalibrate(data, rm_static=True)
bias_raw = np.array(calib.df['bias'])

calib.calibrate_antennas()
bias_antenna_delay = np.array(calib.df['bias'])

calib.fit_power_model()
bias_fully_calib = np.array(calib.df['bias'])
# %%
plt.hist(bias_raw,bins=np.linspace(-0.5,1,100), alpha=0.5, density=True)
plt.hist(bias_antenna_delay,bins=np.linspace(-0.5,1,100), alpha=0.5, density=True)
plt.hist(bias_fully_calib,bins=np.linspace(-0.5,1,100), alpha=0.5, density=True)
# %%
