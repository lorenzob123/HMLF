# This file is here just to define MlpPolicy/CnnPolicy
# that work for PPO
from hmlf.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy

MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy
