from hmlf.paddpg.paddpg import PADDPG
from hmlf.pdqn import PDQN
from hmlf.pdqn.policies import build_action_space_q, build_simple_parameter_space, build_state_parameter_space, PDQNPolicy
from hmlf.common.utils import get_linear_fn

import torch

import sys
sys.path.append("./environments")

from enviroments.obstacle_course import ObstacleCourse
env = ObstacleCourse()

from hmlf.common.preprocessing import get_action_dim
from hmlf.common.callbacks import EvalCallback

print(f"action_dim={get_action_dim(env.action_space)}")

# from hmlf.paddpg import PADDPG
# pa = PADDPG("MlpPolicy", env)
# import ipdb; ipdb.set_trace()



# parameter_action_space = build_simple_parameter_space(env.action_space)
# q_observation_space = build_state_parameter_space(env.observation_space, env.action_space)
# q_action_space = build_action_space_q(env.action_space)
# schedule = get_linear_fn(1, 1e-3, 0.1)
# pol = PDQNPolicy(env.observation_space, env.action_space, schedule)
# sample = [env.observation_space.sample(), env.observation_space.sample()]
# sample_tensor = torch.Tensor(sample)

# action = pol.predict(sample)
# q_values = pol.forward(sample_tensor)

pqdn = PDQN("MlpPolicy", env, learning_starts=1000)

pqdn.learn(total_timesteps=1e6, callback=EvalCallback(eval_env=ObstacleCourse(), eval_freq = 10000, n_eval_episodes=20))


