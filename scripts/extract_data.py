# %%
from pyuwbcalib.machine import RosMachine
from pyuwbcalib.postprocess import PostProcess
from pyuwbcalib.utils import save
from configparser import ConfigParser, ExtendedInterpolation

config_file = 'config/ifo_3_drones_rosbag.config'

parser = ConfigParser(interpolation=ExtendedInterpolation())
parser.read(config_file)

machines = {}
for i,machine in enumerate(parser['MACHINES']):
    machine_id = parser['MACHINES'][machine]
    machines[machine_id] = RosMachine(parser, i)
# %%
data = PostProcess(machines)
save(data, filename="data.pickle")
# %%
