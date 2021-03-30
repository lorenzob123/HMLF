# DDPG can be view as a special case of TD3
from hmlf.algorithms.td3.policies import CnnPolicy, MlpPolicy  # noqa:F401
from hmlf.common.policy_register import register_policy

register_policy("PADDPG", "MlpPolicy", MlpPolicy)
register_policy("PADDPG", "CnnPolicy", CnnPolicy)
