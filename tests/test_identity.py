import numpy as np
import pytest

from hmlf.algorithms import A2C, DDPG, DQN, PPO, SAC, TD3
from hmlf.algorithms.a2c import MlpPolicy as MlpPolicyA2C
from hmlf.algorithms.ddpg import MlpPolicy as MlpPolicyDDPG
from hmlf.algorithms.dqn import MlpPolicy as MlpPolicyDQN
from hmlf.algorithms.ppo import MlpPolicy as MlpPolicyPPO
from hmlf.algorithms.sac import MlpPolicy as MlpPolicySAC
from hmlf.algorithms.td3 import MlpPolicy as MlpPolicyTD3
from hmlf.common.evaluation import evaluate_policy
from hmlf.common.noise import NormalActionNoise
from hmlf.environments.identity_env import IdentityEnv, IdentityEnvBox, IdentityEnvMultiBinary, IdentityEnvMultiDiscrete
from hmlf.environments.vec_env import DummyVecEnv

DIM = 4


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_class,policy_class",
    [
        (A2C, MlpPolicyA2C),
        (PPO, MlpPolicyPPO),
        (DQN, MlpPolicyDQN),
    ],
)
@pytest.mark.parametrize("env", [IdentityEnv(DIM), IdentityEnvMultiDiscrete(DIM), IdentityEnvMultiBinary(DIM)])
def test_discrete(model_class, policy_class, env):
    env_ = DummyVecEnv([lambda: env])
    kwargs = {}
    n_steps = 2000
    if model_class == DQN:
        kwargs = dict(learning_starts=0)
        n_steps = 2000
        # DQN only support discrete actions
        if isinstance(env, (IdentityEnvMultiDiscrete, IdentityEnvMultiBinary)):
            return

    model = model_class(policy_class, env_, gamma=0.4, seed=1, **kwargs).learn(n_steps)

    evaluate_policy(model, env_, n_eval_episodes=20, reward_threshold=70, warn=False)
    obs = env.reset()

    assert np.shape(model.predict(obs)[0]) == np.shape(obs)


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_class,policy_class",
    [(A2C, MlpPolicyA2C), (PPO, MlpPolicyPPO), (SAC, MlpPolicySAC), (TD3, MlpPolicyTD3), (DDPG, MlpPolicyDDPG)],
)
def test_continuous(model_class, policy_class):
    env = IdentityEnvBox(eps=0.5)

    n_steps = {A2C: 3500, PPO: 3000, SAC: 700, TD3: 500, DDPG: 500}[model_class]

    kwargs = dict(policy_kwargs=dict(net_arch=[64, 64]), seed=0, gamma=0.95)
    if model_class in [TD3]:
        n_actions = 1
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        kwargs["action_noise"] = action_noise

    model = model_class(policy_class, env, **kwargs).learn(n_steps)

    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90, warn=False)
