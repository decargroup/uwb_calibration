# %%
from pyuwbcalib.machine import RosMachine
from pyuwbcalib.postprocess import PostProcess
from pyuwbcalib.utils import save
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
# %%
# Process and merge the data from all the machines
data = PostProcess(machines)

# Save the processed data using the pickle library
save(data, filename="data.pickle")
# %%
