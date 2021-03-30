# This file is here just to define MlpPolicy/CnnPolicy
# that work for PPO
from hmlf.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy
from hmlf.common.policy_register import register_policy

MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy


register_policy("PPO", "MlpPolicy", MlpPolicy)
register_policy("PPO", "CnnPolicy", CnnPolicy)
