from pyuwbcalib.uwbcalibrate import UwbCalibrate

x = UwbCalibrate("datasets/synthetic_1.csv","datasets/synthetic_2.csv",[1,2,3],average=False)
# x0 = UwbCalibrate("datasets/2022_02_21/formatted_ID1_twr0.csv",
#                  "datasets/2022_02_21/formatted_ID2_twr0.csv",
#                  [1,2,3],average=False)
x1 = UwbCalibrate("datasets/2022_02_21/formatted_ID1_twr1.csv",
                 "datasets/2022_02_21/formatted_ID2_twr1.csv",
                 [1,2,3],average=False)
x2 = UwbCalibrate("datasets/2022_02_21/formatted_ID1_twr2.csv",
                 "datasets/2022_02_21/formatted_ID2_twr2.csv",
                 [1,2,3],average=False)

# print(x.calibrate_antennas())
# print(x0.calibrate_antennas())
print(x1.calibrate_antennas())
print(x2.calibrate_antennas())