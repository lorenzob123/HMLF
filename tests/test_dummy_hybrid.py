import numpy as np
import pytest

from hmlf.environments import DummyHybrid
from hmlf.spaces import SimpleHybrid

N_MAX_STEPS = 50


def test_type_assertions():
    with pytest.raises(AssertionError):
        DummyHybrid("hello")
    with pytest.raises(AssertionError):
        DummyHybrid([1, 2, 3.4])
    with pytest.raises(AssertionError):
        DummyHybrid([1, 2], 2.3)
    with pytest.raises(ValueError):
        DummyHybrid([-1, 2], 2)
    with pytest.raises(AssertionError):
        DummyHybrid([1, 2], -2)


def test_empty_list():
    with pytest.raises(ValueError):
        DummyHybrid([])


@pytest.mark.parametrize(
    "parameter_dimensions",
    [
        [1],
        [1, 3, 4],
        [5, 6, 7, 1, 2],
    ],
)
def test__dimensions(parameter_dimensions):
    env = DummyHybrid(parameter_dimensions)
    assert env.n_parameter_spaces == len(parameter_dimensions)
    assert type(env.action_space) is SimpleHybrid
    assert env.action_space.get_dimension() == int(1 + np.sum(parameter_dimensions))


@pytest.mark.parametrize(
    "observation_dimension",
    [
        1,
        10,
        434,
    ],
)
def test_observation_dimensions(observation_dimension):
    env = DummyHybrid([1], observation_dimension)
    assert env.observation_space.shape[0] == observation_dimension


def test_step():
    observation_dimension = 10
    env = DummyHybrid([1, 3, 2], observation_dimension)
    for i in range(10):
        observation, reward, is_done, info = env.step(env.action_space.sample())
        assert type(observation) is np.ndarray
        assert len(observation) == observation_dimension
        assert type(reward) is float
        assert type(is_done) is bool
        assert type(info) is dict
        assert info == {}
        assert env.n_steps == i + 1


def test_reset():
    observation_dimension = 10
    env = DummyHybrid([1, 3, 2], observation_dimension)
    for _ in range(10):
        env.step(env.action_space.sample())
    observation = env.reset()
    assert type(observation) is np.ndarray
    assert len(observation) == observation_dimension
    assert env.n_steps == 0


@pytest.mark.parametrize(
    "parameters",
    [
        [-1, 1],
        [2, 2],
        [0, 0],
        [-1e-4, 0],
    ],
)
def test_reward_not_positive(parameters):
    env = DummyHybrid([2])
    _, reward, _, _ = env.step((0, parameters))
    assert reward <= 0


@pytest.mark.parametrize(
    "steps",
    [
        2,
        N_MAX_STEPS - 1,
        N_MAX_STEPS,
        N_MAX_STEPS + 1,
        N_MAX_STEPS + 10,
    ],
)
def test_is_done(steps):
    env = DummyHybrid([2])
    random_value = np.random.random(size=0)
    _, _, is_done, _ = env.step((0, [random_value, -random_value]))
    assert is_done
    env.reset()

    for _ in range(steps):
        _, _, is_done, _ = env.step((env.action_space.sample()))

    assert is_done == (steps >= N_MAX_STEPS)
