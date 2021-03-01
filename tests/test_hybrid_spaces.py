import numpy as np
import pytest
from gym.spaces import Box, Discrete

from hmlf.spaces import OneHotHybrid, SimpleHybrid


@pytest.mark.parametrize(
    "space_list",
    [
        (
            Discrete(5),
            Box(low=-1, high=143, shape=(1,)),
            Box(low=1, high=1.2, shape=(2,)),
            Box(low=11, high=13, shape=(3,)),
            Box(low=-1, high=1, shape=(4,)),
            Box(low=-1, high=1, shape=(5,)),
        ),
        (
            Discrete(2),
            Box(low=-1, high=143, shape=(1,)),
            Box(low=1, high=1.2, shape=(0,)),
        ),
    ],
)
def test_init_simple_hybrid(space_list):
    hybrid_space = SimpleHybrid(space_list)
    assert hybrid_space.discrete_dim == space_list[0].n
    sample = hybrid_space.sample()

    # check datatypes
    assert type(sample) == tuple
    assert type(sample[0]) == int
    for s in sample[1:]:
        assert type(s) == np.ndarray

    # check that sample and fomat_actions work consistently and correctly
    sample_list = [hybrid_space.sample(), hybrid_space.sample()]
    sample_array = np.vstack([np.hstack(sample_list[0]), np.hstack(sample_list[1])])
    new_sample_list = hybrid_space.format_action(sample_array)

    for sample, new_sample in zip(sample_list, new_sample_list):
        for action, new_action in zip(sample, new_sample):
            assert np.sum(np.abs(action - new_action)) < 1e-6


@pytest.mark.parametrize(
    "space_list",
    [
        (
            Discrete(5),
            Box(low=-1, high=143, shape=(1,)),
            Box(low=1, high=1.2, shape=(2,)),
            Box(low=11, high=13, shape=(3,)),
            Box(low=-1, high=1, shape=(4,)),
            Box(low=-1, high=1, shape=(5,)),
        ),
        (
            Discrete(2),
            Box(low=-1, high=143, shape=(1,)),
            Box(low=1, high=1.2, shape=(0,)),
        ),
    ],
)
def test_init_onehot_hybrid(space_list):
    hybrid_space = OneHotHybrid(space_list)
    assert hybrid_space.discrete_dim == space_list[0].n
    sample = hybrid_space.sample()

    # check type and dimensions
    assert type(sample) == np.ndarray
    for s in sample[: hybrid_space.discrete_dim]:
        assert s == 1 or s == 0
    assert len(sample) == hybrid_space.get_dimension()

    # check that sample and format_action work correctly
    sample_list = [hybrid_space.sample(), hybrid_space.sample()]
    new_sample_list = hybrid_space.format_action(np.vstack(sample_list))

    for sample, new_sample in zip(sample_list, new_sample_list):
        assert sample[new_sample[0]] == 1
        assert np.sum(np.abs(sample[hybrid_space.discrete_dim :] - np.hstack(new_sample[1:]))) < 1e-6

        for action_p, action_space in zip(new_sample[1:], space_list[1:]):
            assert action_p.shape == action_space.shape
            assert np.linalg.norm(action_p - np.clip(action_p, action_space.low, action_space.high)) < 1e-6
