# This file is here just to define MlpPolicy/CnnPolicy
# that work for A2C
from hmlf.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, register_policy

MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy

register_policy(MlpPolicy, "MlpPolicy", "A2C")
register_policy(CnnPolicy, "CnnPolicy", "A2C")
