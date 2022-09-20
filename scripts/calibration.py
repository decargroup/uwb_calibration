# %%
from pyuwbcalib.postprocess import load
from pyuwbcalib.uwbcalibrate import UwbCalibrate
import numpy as np

c = 299702547
data = load()

calib = UwbCalibrate(data, rm_static=True)
calib.calibrate_antennas()

# %%
