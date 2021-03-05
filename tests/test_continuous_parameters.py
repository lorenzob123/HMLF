from typing import List, Optional

import numpy as np
import pytest

from hmlf.spaces import Box, ContinuousParameters, Discrete, SimpleHybrid, Tuple
from hmlf.spaces.hybrid_base import HybridBase


def make_box(low: Optional[List] = None, high: Optional[List] = None, shape: Optional[Tuple] = None) -> Box:
    if shape is None:
        if (low is None) and (high is None):
            raise ValueError("Some value needs to be not none")
        else:
            low = np.array(low)
            high = np.array(high)
            return Box(low, high)
    else:
        if low is None:
            low = -np.inf
        if high is None:
            high = np.inf
        return Box(low, high, shape)


def test_invalid_arguments():
    with pytest.raises(AssertionError):
        ContinuousParameters("string")
    with pytest.raises(AssertionError):
        ContinuousParameters(1.343)
    with pytest.raises(AssertionError):
        ContinuousParameters([])
    with pytest.raises(AssertionError):
        ContinuousParameters([1, 2])
    with pytest.raises(AssertionError):
        ContinuousParameters([make_box(shape=(1,)), 2])
    with pytest.raises(AssertionError):
        continuous_spaces = [
            make_box([-1, 2.3], [45, 4.3]),
            make_box([-10], [45]),
            make_box([50, 34, 0], [100, 120, 2]),
        ]
        ContinuousParameters([Discrete(1)] + continuous_spaces)


def test_dimensions():
    space = ContinuousParameters([make_box(shape=(1,)), make_box(shape=(3,)), make_box(shape=(2,))])

    assert space.get_n_discrete_spaces() == 0
    assert space.get_n_discrete_options() == 0

    assert isinstance(space._get_continuous_spaces(), list)
    assert len(space._get_continuous_spaces()) == 3

    assert space.get_n_continuous_spaces() == 3
    assert space.get_n_continuous_options() == (1 + 3 + 2)
    assert space._get_dimensions_of_continuous_spaces() == [1, 3, 2]

    assert np.array_equal(space.split_indices, [1, 4])
    assert space.get_dimension() == (0 + 1 + 3 + 2)


def test_build_action():
    continuous_spaces = [make_box([-1, 2.3], [45, 4.3]), make_box([-10], [45])]
    space = ContinuousParameters(continuous_spaces)

    discrete = np.array([2, 0, 1])
    parameters = np.array(
        [
            [0, 3, 2],
            [-10, 3, 63.1],
            [0, 3, 50],
        ]
    )

    action = space.build_action(discrete, parameters)
    print(action)

    assert isinstance(action, list)
    assert len(action) == 3
    assert len(action[0]) == 2
    assert isinstance(action[0], tuple)
    assert len(action[0]) == 2


def test_repr_does_not_throw_error():
    continuous_spaces = [make_box([-1, 2.3], [45, 4.3]), make_box([-10], [45])]
    space = ContinuousParameters(continuous_spaces)
    represensation_string = repr(space)
    represensation_string = represensation_string.replace(", float32", "")
    eval(represensation_string)
    assert "ContinuousParameters" in represensation_string


def test_comparison():
    continuous_spaces = [make_box([-1, 2.3], [45, 4.3]), make_box([-10], [45])]
    space = ContinuousParameters(continuous_spaces)
    continuous_spaces2 = [make_box([-1, 54], [45, 4.3]), make_box([-10], [445])]
    space2 = SimpleHybrid(continuous_spaces2)

    assert space == space
    assert space != "hi"
    assert space != space2
    assert issubclass(SimpleHybrid, HybridBase)
