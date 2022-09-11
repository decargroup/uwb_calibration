# %%
from pyuwbcalib.machine import RosMachine
from pyuwbcalib.postprocess import PostProcess
from configparser import ConfigParser, ExtendedInterpolation

config_file = 'config/ifo_3_drones_rosbag.config'

parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read(config_file)

ts_to_ns = 1e9 * (1.0 / 499.2e6 / 128.0) # DW time unit to nanoseconds

machines = {}
for i,machine in enumerate(parser['MACHINES']):
    machine_id = parser['MACHINES'][machine]
    machines[machine_id] = RosMachine(parser,
                                      i,
                                      ts_to_ns = ts_to_ns)
# %%
data = PostProcess(machines, compute_intervals=True)
data.save()
# %%
