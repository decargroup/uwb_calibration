
from pyuwbcalib.uwbcalibrate import UwbCalibrate
import matplotlib.pyplot as plt


# x = UwbCalibrate("datasets/synthetic_1.csv","datasets/synthetic_2.csv",[1,2,3],average=False)
x = UwbCalibrate("datasets/2022_02_23/formatted_ID1_twr1.csv",
                 "datasets/2022_02_23/formatted_ID2_twr1.csv",
                 [1,2,3],average=False)

id1 = 2
id2 = 3

meas_old = x.compute_range_meas(id1,id2)

delays = x.calibrate_antennas()
print(delays)
x.correct_antenna_delay(1, delays['Module 1'])
x.correct_antenna_delay(2, delays['Module 2'])
x.correct_antenna_delay(3, delays['Module 3'])

meas_new = x.compute_range_meas(id1,id2)

#%%
fig, ax = plt.subplots()
ax.set_title("Measurements "+str(id1)+"->"+str(id2))
ax.set_xlabel("Measurement Number")
ax.set_ylabel("Range [m]")
ax.set_ylim(0,4)
plt.plot(meas_old,linewidth=1, label='Raw')
plt.plot(meas_new,linewidth=1, label='Calibrated')
plt.plot(x.data[str(id1)+"->"+str(id2)]['gt'],linewidth=3, label='GT')
ax.legend()
plt.show()

