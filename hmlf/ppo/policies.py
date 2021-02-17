# This file is here just to define MlpPolicy/CnnPolicy
# that work for PPO
from hmlf.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, register_policy

MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy

register_policy(MlpPolicy, "MlpPolicy", "PPO")
register_policy(CnnPolicy, "CnnPolicy", "PPO")
