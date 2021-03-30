import gym
import pytest

from hmlf.algorithms import A2C, DDPG, DQN, MPDQN, PADDPG, PDQN, PPO, SAC, SDDPG, TD3
from hmlf.algorithms.a2c import MlpPolicy as A2CMlpPolicy
from hmlf.algorithms.ddpg import MlpPolicy as DDPGMlpPolicy
from hmlf.algorithms.dqn import MlpPolicy as DQNMlpPolicy
from hmlf.algorithms.mpdqn import MlpPolicy as MPDQNMlpPolicy
from hmlf.algorithms.paddpg import MlpPolicy as PADDPGMlpPolicy
from hmlf.algorithms.pdqn import MlpPolicy as PDQNMlpPolicy
from hmlf.algorithms.ppo import MlpPolicy as PPOMlpPolicy
from hmlf.algorithms.sac import MlpPolicy as SACMlpPolicy
from hmlf.algorithms.sddpg import MlpPolicy as SDDPGMlpPolicy
from hmlf.algorithms.td3 import MlpPolicy as TD3MlpPolicy
from hmlf.environments.dummy_hybrid import DummyHybrid


@pytest.fixture
def dummy_hybrid_env():
    return DummyHybrid([1, 2, 3])


@pytest.fixture
def dummy_discrete_env():
    return gym.make("CartPole-v0")


@pytest.fixture
def dummy_box_env():
    return gym.make("Pendulum-v0")


@pytest.mark.parametrize(
    "algorithm, policy_class_mlp",
    [
        (DDPG, DDPGMlpPolicy),
        (MPDQN, MPDQNMlpPolicy),
        (PDQN, PDQNMlpPolicy),
        (PADDPG, PADDPGMlpPolicy),
        (PPO, PPOMlpPolicy),
        (SDDPG, SDDPGMlpPolicy),
        (TD3, TD3MlpPolicy),
    ],
)
def test_types_of_registered_hybrid_policys(algorithm, policy_class_mlp, dummy_hybrid_env):
    alg_mlp_object = algorithm(policy_class_mlp, dummy_hybrid_env)
    alg_mlp_str = algorithm("MlpPolicy", dummy_hybrid_env)
    assert alg_mlp_object.policy_class == alg_mlp_str.policy_class
    assert isinstance(alg_mlp_str.policy, policy_class_mlp)


@pytest.mark.parametrize(
    "algorithm, policy_class_mlp",
    [
        (A2C, A2CMlpPolicy),
        (DQN, DQNMlpPolicy),
        (PPO, PPOMlpPolicy),
    ],
)
def test_types_of_registered_discrete_policys(algorithm, policy_class_mlp, dummy_discrete_env):
    alg_mlp_object = algorithm(policy_class_mlp, dummy_discrete_env)
    alg_mlp_str = algorithm("MlpPolicy", dummy_discrete_env)
    assert alg_mlp_object.policy_class == alg_mlp_str.policy_class
    assert isinstance(alg_mlp_str.policy, policy_class_mlp)


@pytest.mark.parametrize(
    "algorithm, policy_class_mlp",
    [
        (A2C, A2CMlpPolicy),
        (DDPG, DDPGMlpPolicy),
        (PPO, PPOMlpPolicy),
        (SAC, SACMlpPolicy),
        (TD3, TD3MlpPolicy),
    ],
)
def test_types_of_registered_box_policys(algorithm, policy_class_mlp, dummy_box_env):
    alg_mlp_object = algorithm(policy_class_mlp, dummy_box_env)
    alg_mlp_str = algorithm("MlpPolicy", dummy_box_env)
    assert alg_mlp_object.policy_class == alg_mlp_str.policy_class
    assert isinstance(alg_mlp_str.policy, policy_class_mlp)
