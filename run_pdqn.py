from hmlf.pdqn import PDQN
from hmlf.common.monitor import Monitor

import sys
sys.path.append("./environments")

from enviroments.obstacle_coursev2 import ObstacleCourse
env = ObstacleCourse()

from hmlf.common.preprocessing import get_action_dim
from hmlf.common.callbacks import EvalCallback


pdqn = PDQN("MlpPolicy", env, learning_rate=0.001)

pdqn.learn(total_timesteps=1e6, callback=EvalCallback(eval_env=Monitor(ObstacleCourse()), eval_freq = 10000, n_eval_episodes=50))


