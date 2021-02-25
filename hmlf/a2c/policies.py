# This file is here just to define MlpPolicy/CnnPolicy
# that work for A2C
from hmlf.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy

MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy
