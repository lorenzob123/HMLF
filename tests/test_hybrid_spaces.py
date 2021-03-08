import numpy as np
import pytest
from gym.spaces import Box

from hmlf.spaces import OneHotHybrid, SimpleHybrid


@pytest.mark.parametrize(
    "spaces",
    [
        [
            Box(low=-1, high=143, shape=(1,)),
            Box(low=1, high=1.2, shape=(2,)),
            Box(low=11, high=13, shape=(3,)),
            Box(low=-1, high=1, shape=(4,)),
            Box(low=-1, high=1, shape=(5,)),
        ],
        [
            Box(low=-1, high=143, shape=(1,)),
            Box(low=1, high=1.2, shape=(0,)),
        ],
    ],
)
def test_init_simple_hybrid(spaces):
    hybrid_space = SimpleHybrid(spaces)
    assert hybrid_space.get_n_discrete_options() == len(spaces)
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
    "spaces",
    [
        [
            Box(low=-1, high=143, shape=(1,)),
            Box(low=1, high=1.2, shape=(2,)),
            Box(low=11, high=13, shape=(3,)),
            Box(low=-1, high=1, shape=(4,)),
            Box(low=-1, high=1, shape=(5,)),
        ],
        [
            Box(low=-1, high=143, shape=(1,)),
            Box(low=1, high=1.2, shape=(2,)),
        ],
    ],
)
def test_init_onehot_hybrid(spaces):
    hybrid_space = OneHotHybrid(spaces)
    assert hybrid_space.get_n_discrete_options() == len(spaces)
    sample = hybrid_space.sample()

    # check type and dimensions
    assert type(sample) == tuple
    assert sum(sample[0]) == 1
    assert sum(sample[0] == 1) == 1
    assert sum(sample[0] == 0) == hybrid_space.get_n_discrete_options() - 1
    assert np.hstack(sample).shape[0] == hybrid_space.get_dimension()

    # check that sample and format_action work correctly
    sample_list = [np.hstack(hybrid_space.sample()), np.hstack(hybrid_space.sample())]
    new_sample_list = hybrid_space.format_action(np.vstack(sample_list))

    for sample, new_sample in zip(sample_list, new_sample_list):
        assert np.sum(np.abs(sample[hybrid_space.get_n_discrete_options() :] - np.hstack(new_sample[1:]))) < 1e-6

        for action_p, action_space in zip(new_sample[1:], spaces):
            assert action_p.shape == action_space.shape
            assert np.linalg.norm(action_p - np.clip(action_p, action_space.low, action_space.high)) < 1e-6
