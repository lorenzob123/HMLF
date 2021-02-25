import os

from hmlf.a2c import A2C
from hmlf.ddpg import DDPG
from hmlf.dqn import DQN
from hmlf.her import HER
from hmlf.mpdqn import MPDQN
from hmlf.paddpg import PADDPG
from hmlf.pdqn import PDQN
from hmlf.ppo import PPO
from hmlf.sac import SAC
from hmlf.sddpg import SDDPG
from hmlf.td3 import TD3

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file, "r") as file_handler:
    __version__ = file_handler.read().strip()
