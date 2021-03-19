import gym
import numpy as np
import pytest

from hmlf import spaces
from hmlf.algorithms import A2C, DDPG, DQN, PPO, SAC, TD3
from hmlf.algorithms.a2c import MlpPolicy as MlpPolicyA2C
from hmlf.algorithms.ddpg import MlpPolicy as MlpPolicyDDPG
from hmlf.algorithms.dqn import MlpPolicy as MlpPolicyDQN
from hmlf.algorithms.ppo import MlpPolicy as MlpPolicyPPO
from hmlf.algorithms.sac import MlpPolicy as MlpPolicySAC
from hmlf.algorithms.td3 import MlpPolicy as MlpPolicyTD3
from hmlf.common.evaluation import evaluate_policy


class DummyMultiDiscreteSpace(gym.Env):
    def __init__(self, nvec):
        super(DummyMultiDiscreteSpace, self).__init__()
        self.observation_space = spaces.MultiDiscrete(nvec)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, {}


class DummyMultiBinary(gym.Env):
    def __init__(self, n):
        super(DummyMultiBinary, self).__init__()
        self.observation_space = spaces.MultiBinary(n)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, {}


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_class,policy_class",
    [
        (SAC, MlpPolicySAC),
        (TD3, MlpPolicyTD3),
        (DQN, MlpPolicyDQN),
    ],
)
@pytest.mark.parametrize("env", [DummyMultiDiscreteSpace([4, 3]), DummyMultiBinary(8)])
def test_identity_spaces(model_class, policy_class, env):
    """
    Additional tests for DQ/SAC/TD3 to check observation space support
    for MultiDiscrete and MultiBinary.
    """
    # DQN only support discrete actions
    if model_class == DQN:
        env.action_space = spaces.Discrete(4)

    env = gym.wrappers.TimeLimit(env, max_episode_steps=100)

    model = model_class(policy_class, env, gamma=0.5, seed=1, policy_kwargs=dict(net_arch=[64]))
    model.learn(total_timesteps=500)

    evaluate_policy(model, env, n_eval_episodes=5, warn=False)


@pytest.mark.parametrize(
    "model_class,policy_class",
    [
        (A2C, MlpPolicyA2C),
        (PPO, MlpPolicyPPO),
        (SAC, MlpPolicySAC),
        (TD3, MlpPolicyTD3),
        (DQN, MlpPolicyDQN),
        (DDPG, MlpPolicyDDPG),
    ],
)
@pytest.mark.parametrize("env", ["Pendulum-v0", "CartPole-v1"])
def test_action_spaces(model_class, policy_class, env):
    if model_class in [SAC, DDPG, TD3]:
        supported_action_space = env == "Pendulum-v0"
    elif model_class == DQN:
        supported_action_space = env == "CartPole-v1"
    elif model_class in [A2C, PPO]:
        supported_action_space = True

    if supported_action_space:
        model_class(policy_class, env)
    else:
        with pytest.raises(AssertionError):
            model_class(policy_class, env)
