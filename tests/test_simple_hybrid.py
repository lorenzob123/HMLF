import numpy as np
import pytest

from hmlf.spaces import Discrete, SimpleHybrid
from hmlf.spaces.hybrid_base import HybridBase

from .utils import make_box


def test_invalid_arguments():
    with pytest.raises(AssertionError):
        SimpleHybrid("string")
    with pytest.raises(AssertionError):
        SimpleHybrid(1.343)
    with pytest.raises(ValueError):
        SimpleHybrid([])
    with pytest.raises(AssertionError):
        SimpleHybrid([1, 2])
    with pytest.raises(AssertionError):
        continuous_spaces = [
            make_box([-1, 2.3], [45, 4.3]),
            make_box([-10], [45]),
            make_box([50, 34, 0], [100, 120, 2]),
        ]
        SimpleHybrid([Discrete(10)] + continuous_spaces)


@pytest.mark.parametrize(
    "spaces",
    [
        [make_box(shape=(1,)), make_box(shape=(3,)), make_box(shape=(2,))],
        [make_box(shape=(1,)), make_box(shape=(3,)), make_box(shape=(2,)), make_box(shape=(334,)), make_box(shape=(30,))],
    ],
)
def test_input_tuple(spaces):
    space_list = SimpleHybrid(spaces)
    space_tuple = SimpleHybrid(tuple(spaces))
    assert space_list == space_tuple


def test_dimensions():
    space = SimpleHybrid([make_box(shape=(1,)), make_box(shape=(3,)), make_box(shape=(2,))])

    assert space.n_discrete_options == 3
    assert space.get_n_discrete_spaces() == 1
    assert space.get_n_discrete_options() == 3

    assert isinstance(space._get_continuous_spaces(), list)
    assert len(space._get_continuous_spaces()) == 3

    assert space.get_n_continuous_spaces() == 3
    assert space.get_n_continuous_options() == (1 + 3 + 2)
    assert space._get_dimensions_of_continuous_spaces() == [1, 3, 2]

    assert np.array_equal(space.split_indices, [1, 4])
    assert space.get_dimension() == (1 + 1 + 3 + 2)


def test_low_high_concatination():
    continuous_spaces = [
        make_box([-1, 2.3], [45, 4.3]),
        make_box([-10], [45]),
        make_box([50, 34, 0], [100, 120, 2]),
    ]
    space = SimpleHybrid(continuous_spaces)

    print(space.continuous_low)
    print(space.continuous_high)
    assert np.allclose(space.continuous_low, [-1.0, 2.3, -10.0, 50.0, 34.0, 0.0])
    assert np.allclose(space.continuous_high, [45.0, 4.3, 45.0, 100.0, 120.0, 2.0])


def test_build_action():
    continuous_spaces = [make_box([-1, 2.3], [45, 4.3]), make_box([-10], [45])]
    space = SimpleHybrid(continuous_spaces)

    discrete = np.array([2, 0, 1])
    parameters = np.array(
        [
            [0, 3, 2],
            [-10, 3, 63.1],
            [0, 3, 50],
        ]
    )

    action = space.build_action(discrete, parameters)
    assert isinstance(action, list)
    assert len(action) == 3
    assert len(action[0]) == 3
    assert isinstance(action[0], tuple)
    assert action[0][0] == 2
    assert action[1][0] == 0
    assert action[2][0] == 1
    assert np.allclose(action[0][1], [0.0, 3.0])
    assert np.allclose(action[0][2], [2])
    assert np.allclose(action[1][1], [-1, 3])
    assert np.allclose(action[1][2], [45])


def test_repr_does_not_throw_error():
    from hmlf.spaces import Box

    continuous_spaces = [make_box([-1, 2.3], [45, 4.3]), make_box([-10], [45])]
    space = SimpleHybrid(continuous_spaces)
    represensation_string = repr(space)
    represensation_string = represensation_string.replace(", float32", "")
    eval(represensation_string)
    assert "SimpleHybrid" in represensation_string


def test_comparison():
    continuous_spaces = [make_box([-1, 2.3], [45, 4.3]), make_box([-10], [45])]
    space = SimpleHybrid(continuous_spaces)
    continuous_spaces2 = [make_box([-1, 54], [45, 4.3]), make_box([-10], [445])]
    space2 = SimpleHybrid(continuous_spaces2)

    assert space == space
    assert space != "hi"
    assert space != space2
    assert issubclass(SimpleHybrid, HybridBase)
