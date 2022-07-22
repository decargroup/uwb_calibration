# standard libraries
import numpy as np
import matplotlib.pyplot as plt

# custom libraries
from pyuwbcalib.postprocess import PostProcess

n = 3000 # number of lines to read

def read_range(filename):
    # Read the file
    with open(filename,'r') as f:
        lines = f.readlines()

    range_data = np.zeros((n,))

    lv0=0
    for line in lines:
        if lv0>=n:
            break
        
        dict_temp = eval(line)
        if 'range' in dict_temp.keys():
            range_data[lv0] = dict_temp['range']
            lv0 += 1

    f.close()

    return range_data

# TWR 0 ----------------------------------------------------------------
filename = 'datasets/2022_03_16/raw_twr0/log_16_03_2022_14_50_36_ID1.txt'
range_twr0 = read_range(filename)

# TWR 1, original ------------------------------------------------------
filename = 'datasets/2022_03_16/raw_twr1/log_16_03_2022_14_52_35_ID1.txt'
range_twr1 = read_range(filename)

# TWR 1, no delay between rx1 and tx2 ----------------------------------
filename = 'datasets/2022_03_16/raw_twr1/log_16_03_2022_14_54_04_ID1.txt'
range_twr2 = read_range(filename)

# TWR 1, no delay between rx1 and tx2, delay between tx2 and tx3 x6 --
filename = 'datasets/2022_03_16/raw_twr1/log_16_03_2022_15_15_28_ID1.txt'
range_twr3 = read_range(filename)

# PLOTTING -------------------------------------------------------
# Create a 2x2 subplot
fig, axes = plt.subplots(nrows=2, ncols=2)
# Ensure all text is in Tex
plt.rcParams['text.usetex'] = True
# Use seaborn theme
plt.style.use('seaborn-darkgrid')
# Space out subplots
fig.tight_layout()
# Set to full screen
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
# Main title
fig.suptitle("Total number of measurements: "+str(n), fontsize=24)
# Move subplots down
fig.subplots_adjust(top=0.9)

ax = plt.subplot(2,2,1)
plt.plot(range_twr0-np.mean(range_twr0))
plt.ylim([-0.2, 0.2])
ax.set_title(r"Single-Sided TWR", fontsize=20)
ax.set_xlabel(r'Measurement Number', fontsize=16)
ax.set_ylabel(r'$d - \bar{d}$ [m]', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.text(0.9, 0.05,'Std: '+str(np.round(np.std(range_twr0),4)),
         fontsize=20,
         color='red', 
         horizontalalignment='center',
         verticalalignment='center',
         transform = ax.transAxes)

ax = plt.subplot(2,2,2)
plt.plot(range_twr1-np.mean(range_twr1))
plt.ylim([-0.2, 0.2])
ax.set_title(r"Double-Sided TWR", fontsize=20)
ax.set_xlabel(r'Measurement Number', fontsize=16)
ax.set_ylabel(r'$d - \bar{d}$ [m]', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.text(0.9, 0.05,'Std: '+str(np.round(np.std(range_twr1),4)),
         fontsize=20,
         color='red', 
         horizontalalignment='center',
         verticalalignment='center',
         transform = ax.transAxes)

ax = plt.subplot(2,2,3)
plt.plot(range_twr2-np.mean(range_twr2))
plt.ylim([-0.2, 0.2])
ax.set_title(r"Double-Sided TWR, no delay in tx2", fontsize=20)
ax.set_xlabel(r'Measurement Number', fontsize=16)
ax.set_ylabel(r'$d - \bar{d}$ [m]', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.text(0.9, 0.05,'Std: '+str(np.round(np.std(range_twr2),4)),
         fontsize=20,
         color='red', 
         horizontalalignment='center',
         verticalalignment='center',
         transform = ax.transAxes)

ax = plt.subplot(2,2,4)
plt.plot(range_twr3-np.mean(range_twr3))
plt.ylim([-0.2, 0.2])
ax.set_title(r"Double-Sided TWR, no delay in tx2 AND x6 longer delay tx3", fontsize=20)
ax.set_xlabel(r'Measurement Number', fontsize=16)
ax.set_ylabel(r'$d - \bar{d}$ [m]', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.text(0.9, 0.05,'Std: '+str(np.round(np.std(range_twr3),4)),
         fontsize=20,
         color='red', 
         horizontalalignment='center',
         verticalalignment='center',
         transform = ax.transAxes)

plt.show()