from pyuwbcalib.utils import load
from pyuwbcalib.uwbcalibrate import ApplyCalibration
from pyuwbcalib.postprocess import PostProcess
# TODO: the passive listening calibration is untested. Could be more verbose in this example.

data: PostProcess = load("data.pickle")
calib_results = load("calib_results.pickle")

data.df = ApplyCalibration.antenna_delays(
    data.df, 
    calib_results['delays']
)
data.df_passive = ApplyCalibration.antenna_delays_passive(
    data.df_passive, 
    calib_results['delays']
)
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
