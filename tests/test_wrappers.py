import gym
import numpy as np
import pytest

from hmlf.environments import DummyEnv, DummyHybrid
from hmlf.environments.stage_controller import OneStepPerStageController, StateDependentStageController
from hmlf.environments.wrapper import OneHotWrapper, SequenceWrapper, SimpleHybridWrapper
from hmlf.spaces import ContinuousParameters, OneHotHybrid, SimpleHybrid


@pytest.fixture
def simple_env():
    return DummyEnv()


@pytest.fixture
def simple_hybrid_env():
    return DummyHybrid([2, 7, 3, 10])


@pytest.fixture
def dummy_reward_function():
    return lambda x: 0.1


@pytest.mark.parametrize(
    "wrapper, action_space",
    [
        (SimpleHybridWrapper, SimpleHybrid),
    ],
)
def test_init_hybrid_simple(wrapper, action_space, simple_env):
    wrapped_env = wrapper(simple_env)

    assert isinstance(wrapped_env, gym.Wrapper)
    assert isinstance(wrapped_env.action_space, action_space)
    assert wrapped_env.observation_space == simple_env.observation_space
    assert wrapped_env.metadata == simple_env.metadata


@pytest.mark.parametrize(
    "wrapper, action_space",
    [
        (SequenceWrapper, ContinuousParameters),
        (OneHotWrapper, OneHotHybrid),
    ],
)
def test_init_hybrid_advanced(wrapper, action_space, simple_hybrid_env):
    if wrapper == SequenceWrapper:
        wrapped_env = wrapper(simple_hybrid_env, [0, 1, 0], OneStepPerStageController())
    else:
        wrapped_env = wrapper(simple_hybrid_env)
        assert wrapped_env.observation_space == simple_hybrid_env.observation_space

    assert isinstance(wrapped_env, gym.Wrapper)
    assert isinstance(wrapped_env.action_space, action_space)
    assert wrapped_env.metadata == simple_hybrid_env.metadata


@pytest.mark.parametrize(
    "wrapper",
    [
        (SimpleHybridWrapper),
    ],
)
def test_hybrid_simple_step_sample(wrapper, simple_env):
    wrapped_env = wrapper(simple_env)

    sample = simple_env.action_space.sample()
    wrapped_env.step(sample)

    sample = wrapped_env.action_space.sample()
    simple_env.step(sample)

    assert len(sample) == 3
    assert sample[0] == 1 or sample[0] == 0
    for i in range(1, len(sample)):
        assert np.all((sample[i] < simple_env.action_space[i].high) * (sample[i] > simple_env.action_space[i].low))


@pytest.mark.parametrize(
    "wrapper",
    [
        (OneHotWrapper),
    ],
)
def test_onehot_hybrid_step_sample(wrapper, simple_hybrid_env):
    wrapped_env = wrapper(simple_hybrid_env)

    sample = simple_hybrid_env.action_space.sample()
    sample = (np.array([0, 1]), *sample[1:])
    wrapped_env.step(sample)

    sample = wrapped_env.action_space.sample()
    sample = (1, *sample[1:])
    simple_hybrid_env.step(sample)

    sample = wrapped_env.action_space.sample()

    assert len(sample) == 5
    assert sample[0].shape == (4,)
    assert np.sum(sample[0] == 1) == 1 and np.sum(sample[0] == 0) == 3
    for i in range(1, len(sample)):
        assert np.all(
            (sample[i] < simple_hybrid_env.action_space[i].high) * (sample[i] > simple_hybrid_env.action_space[i].low)
        )


@pytest.mark.parametrize(
    "sequence",
    [
        [0, 1, 3, 2],
        [1, 1, 1, 3],
    ],
)
def test_init_sequence(sequence, simple_hybrid_env):
    wrapped_env = SequenceWrapper(simple_hybrid_env, sequence)
    assert isinstance(wrapped_env.stage_controller, OneStepPerStageController)

    wrapped_env = SequenceWrapper(simple_hybrid_env, sequence, StateDependentStageController([lambda x: True]))
    assert isinstance(wrapped_env.stage_controller, StateDependentStageController)


@pytest.mark.parametrize(
    "sequence",
    [
        [0, 1, 3, 2],
        [1, 1, 1, 3],
    ],
)
def test_sequence_hybrid_step_sample(sequence, simple_hybrid_env):
    wrapped_env = SequenceWrapper(simple_hybrid_env, sequence, OneStepPerStageController())
    for i in range(2):
        obs = wrapped_env.reset()
        for action in sequence:
            assert obs[0] == action
            obs, r, done, info = wrapped_env.step(wrapped_env.action_space.sample())
        assert done

    sample = wrapped_env.action_space.sample()

    assert len(sample) == 4
    for i in range(0, len(sample)):
        assert np.all(
            (sample[i] < simple_hybrid_env.action_space[i + 1].high) * (sample[i] > simple_hybrid_env.action_space[i + 1].low)
        )
