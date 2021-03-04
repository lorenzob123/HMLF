import typing
from typing import List

import numpy as np

from hmlf.spaces.gym import Box, Discrete, Space, Tuple


class SimpleHybrid(Tuple):
    """
    A tuple (i.e., product) of simpler spaces, where the first space is Discrete and the other are Box.
    Samples have the form (int, Box1.sample(), ..., BoxN.sample())
    Example usage:
    self.observation_space = spaces.Tuple((spaces.Discrete(2), hmlf.spaces.Box(np.array((0, 1)), np.array((2, 3)))))
    """

    def __init__(self, spaces_list: List[Space]):
        self.spaces = spaces_list
        self._validate_arguments()

        self.discrete_dim = self._get_discrete_dim()
        dims_continuous = self._get_continuous_dims()
        self.continuous_dim = np.sum(dims_continuous)
        self.split_indices = np.cumsum(dims_continuous[:-1])

        self.continuous_low = np.hstack(tuple(self.spaces[i].low for i in range(1, len(self.spaces))))
        self.continuous_high = np.hstack(tuple(self.spaces[i].high for i in range(1, len(self.spaces))))

    def _validate_arguments(self):
        assert isinstance(self.spaces, list), f"spaces_list arguments needs to of type list. Found {type(self.spaces)}."
        assert len(self.spaces) > 0, f"spaces_list arguments needs to be non empty. Found {self.spaces}."

        for i, space in enumerate(self.spaces):
            assert isinstance(space, Space), "Elements of the SimpleHybrid must be instances of hmlf.Space"
            if i == 0:
                assert isinstance(space, Discrete), "First element of SimpleHybrid has to be of type hmlf.spaces.Discrete"
            else:
                assert isinstance(
                    space, Box
                ), f"Later (index > 0) elements of SimpleHybrid has to be of type hmlf.spaces.Box. Failed for index {i}."

        discrete_dimension = self._get_discrete_dim()
        n_parameter_spaces = len(self.spaces) - 1
        assert (
            discrete_dimension == n_parameter_spaces
        ), f"Discrete dimension should be len(spaces_list) - 1. Found {discrete_dimension}, {n_parameter_spaces}."

    def _get_discrete_dim(self):
        return self.spaces[0].n

    def _get_continuous_dims(self) -> List[int]:
        # Since each space is one dimensional, shape[0] gets the dimension
        dims = [space.shape[0] for space in self.spaces[1:]]
        return dims

    def get_dimension(self) -> int:
        dims_continuous = self._get_continuous_dims()
        return 1 + np.sum(dims_continuous)

    def format_action(self, actions: np.ndarray) -> List[typing.Tuple]:
        discrete, parameters = actions[:, 0], actions[:, 1:]

        return self.build_action(discrete, parameters)

    def build_action(self, discrete: np.ndarray, parameters: np.ndarray) -> List[typing.Tuple]:
        # We clip the parameters
        parameters = np.clip(parameters, self.continuous_low, self.continuous_high)

        # We format the full action for each environment
        sample = []
        for i in range(discrete.shape[0]):
            sample.append(tuple([discrete[i]] + np.split(parameters[i], self.split_indices)))

        return sample

    def __repr__(self) -> str:
        return "SimpleHybrid([" + ", ".join([repr(s) for s in self.spaces]) + "])"

    def __eq__(self, other) -> bool:
        return isinstance(other, SimpleHybrid) and self.spaces == other.spaces
