import gym
import pytest

from hmlf import algorithms, spaces
from hmlf.environments import DummyHybrid, wrap_environment
from hmlf.environments.wrapper import OneHotWrapper, SequenceWrapper, SimpleHybridWrapper


@pytest.mark.parametrize(
    "algorithm",
    [
        "A2C",
        "a2C",
        algorithms.A2C,
        "DQN",
        "dqn",
        algorithms.DQN,
        "DDPG",
        algorithms.DDPG,
        "TD3",
        "td3",
        algorithms.TD3,
        "PPO",
        algorithms.PPO,
        "SAC",
        algorithms.SAC,
    ],
)
@pytest.mark.parametrize(
    "env",
    [
        gym.make("CartPole-v0"),
        gym.make("Pendulum-v0"),
    ],
)
def test_no_wrapper(algorithm, env):
    wrapped_env = wrap_environment(algorithm, env)
    assert wrapped_env is env


@pytest.fixture
def dummy_hybrid_env():
    return DummyHybrid([1, 2, 3])


@pytest.mark.parametrize(
    "algorithm",
    [
        "MPDQN",
        "MpdqN",
        algorithms.MPDQN,
        "PDQN",
        algorithms.PDQN,
        "PPO",
        "Ppo",
        algorithms.PPO,
    ],
)
def test_simple_hybrid_wrapper(algorithm, dummy_hybrid_env):
    wrapped_env = wrap_environment(algorithm, dummy_hybrid_env)
    assert isinstance(wrapped_env, SimpleHybridWrapper)


@pytest.mark.parametrize(
    "algorithm",
    [
        "PADDPG",
        algorithms.PADDPG,
    ],
)
def test_one_hot_wrapper(algorithm, dummy_hybrid_env):
    wrapped_env = wrap_environment(algorithm, dummy_hybrid_env)
    assert isinstance(wrapped_env, OneHotWrapper)


@pytest.mark.parametrize(
    "algorithm",
    [
        "SDDPG",
        algorithms.SDDPG,
    ],
)
def test_sequence_wrapper(algorithm, dummy_hybrid_env):
    sequence = [1, 0, 0, 1]
    wrapped_env = wrap_environment(algorithm, dummy_hybrid_env, sequence)
    assert isinstance(wrapped_env, SequenceWrapper)
    assert wrapped_env.sequence_curator.sequence == sequence

    with pytest.raises(AssertionError):
        wrap_environment(algorithm, dummy_hybrid_env, [])

    with pytest.raises(AssertionError):
        wrap_environment(algorithm, dummy_hybrid_env, "abc")


@pytest.mark.parametrize(
    "algorithm",
    [
        "Baumhaus",
        "",
        spaces.SimpleHybrid,
        spaces.OneHotHybrid,
    ],
)
def test_unknown(algorithm, dummy_hybrid_env):
    with pytest.raises(NotImplementedError):
        wrap_environment(algorithm, dummy_hybrid_env)
