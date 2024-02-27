# UWB Calibration Package

**This package is still in development. Any contribution is welcome!**

A Python package to calibrate the UWB modules in order to improve ranging accuracy.

## Example

```python
from pyuwbcalib.machine import RosMachine
from pyuwbcalib.postprocess import PostProcess
from configparser import ConfigParser, ExtendedInterpolation

# The configuration file
config_file = 'config/ifo_3_drones_rosbag.config'

# Parse through the configuration file
parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read(config_file)

# Create a RosMachine object for every machine
machines = {}
for i,machine in enumerate(parser['MACHINES']):
    machine_id = parser['MACHINES'][machine]
    machines[machine_id] = RosMachine(parser, i)

# Process and merge the data from all the machines
data = PostProcess(machines)

# Instantiate a UwbCalibrate object, and remove static extremes
calib = UwbCalibrate(data, rm_static=True)

# Extract the raw bias measurements
bias_raw = np.array(calib.df['bias'])

# Correct antenna delays
calib.calibrate_antennas()

# Extract the antenna-delay-corrected measurements
bias_antenna_delay = np.array(calib.df['bias'])

# Correct power-correlated bias
calib.fit_power_model()

# Extract the fully-calibrated measurements
bias_fully_calib = np.array(calib.df['bias'])
```

## Installation
Python 3.6 or greater is required. Inside this repo's directory, you may run

    pip3 install .
or

    pip3 install -e .

which installs the package in-place, allowing you make changes to the code without having to reinstall every time. 

The documentation can be compiled using

    cd cdocs
    make html

The file `docs/build/html/index.html` can then be opened in a web browser.

## Citation
M. A. Shalaby, C. C. Cossette, J. R. Forbes and J. Le Ny, "Calibration and Uncertainty Characterization for Ultra-Wideband Two-Way-Ranging Measurements," 2023 IEEE International Conference on Robotics and Automation (ICRA), London, United Kingdom, 2023.
