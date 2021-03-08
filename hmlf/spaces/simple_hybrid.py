import typing
from typing import List, Tuple, Union

import numpy as np

from hmlf.spaces.gym import Box, Discrete, Space
from hmlf.spaces.hybrid_base import HybridBase


class SimpleHybrid(HybridBase):
    """
    A tuple (i.e., product) of simpler spaces, where the first space is Discrete and the other are Box.
    Samples have the form (int, Box1.sample(), ..., BoxN.sample())
    Example usage:
    self.observation_space = spaces.Tuple((spaces.Discrete(2), hmlf.spaces.Box(np.array((0, 1)), np.array((2, 3)))))
    """

    def __init__(self, spaces: Union[List[Space], Tuple[Space]]):
        self.n_discrete_options = len(spaces)
        spaces = [Discrete(self.n_discrete_options)] + spaces
        super().__init__(spaces)

        self.continuous_low = np.hstack(tuple(self.spaces[i].low for i in range(1, len(self.spaces))))
        self.continuous_high = np.hstack(tuple(self.spaces[i].high for i in range(1, len(self.spaces))))

    def _validate_arguments(self) -> None:
        super()._validate_arguments()
        assert len(self.spaces) > 0, f"You need to input at least one space. Found {len(self.spaces)}."
        for i, space in enumerate(self.spaces):
            if i == 0:
                assert isinstance(space, Discrete), "First element of SimpleHybrid has to be of type hmlf.spaces.Discrete"
            else:
                assert isinstance(space, Box), f"Spaces have to be of type hmlf.spaces.Box. Failed for index {i}."

    def get_n_discrete_spaces(self) -> int:
        return 1

    def get_n_discrete_options(self) -> int:
        return self.n_discrete_options

    def _get_continuous_spaces(self) -> List[Box]:
        return self.spaces[1:]

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
        return "SimpleHybrid([" + ", ".join([repr(space) for space in self._get_continuous_spaces()]) + "])"
