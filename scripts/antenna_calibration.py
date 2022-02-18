from pyuwbcalib.uwbcalibrate import UwbCalibrate

x = UwbCalibrate("datasets/synthetic_1.csv","datasets/synthetic_2.csv",[1,2,3],average=False)

print(x.calibrate_antennas())