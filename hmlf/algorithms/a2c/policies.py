# This file is here just to define MlpPolicy/CnnPolicy
# that work for A2C
from hmlf.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy
from hmlf.common.policy_register import register_policy

MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy

register_policy("A2C", "MlpPolicy", MlpPolicy)
register_policy("A2C", "CnnPolicy", CnnPolicy)
