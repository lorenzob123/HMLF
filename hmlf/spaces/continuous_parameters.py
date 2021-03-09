import typing
from typing import List, Tuple, Union

import numpy as np

from hmlf.spaces.gym import Box, Space
from hmlf.spaces.hybrid_base import HybridBase


class ContinuousParameters(HybridBase):
    """
    A tuple (i.e., product) of simpler spaces, where the first space is Discrete and the other are Box.
    Samples have the form (int, Box1.sample(), ..., BoxN.sample())
    Example usage:
    self.observation_space = spaces.Tuple((spaces.Discrete(2), gym.spaces.Box(np.array((0, 1)), np.array((2, 3)))))
    """

    def __init__(self, spaces: Union[List[Space], Tuple[Space]]):
        super().__init__(spaces)

    def _validate_arguments(self) -> None:
        super()._validate_arguments()

        for i, space in enumerate(self.spaces):
            assert isinstance(space, Box), f"Spaces have to be of type hmlf.spaces.Box. Failed for index {i}."

    def get_n_discrete_spaces(self) -> int:
        return 0

    def get_n_discrete_options(self) -> int:
        return 0

    def _get_continuous_spaces(self) -> List[Box]:
        return self.spaces

    def format_action(self, actions: np.ndarray) -> List[typing.Tuple]:
        raise NotImplementedError("Not implemented for ContinuousParameters")

    def _build_single_action(self, current_discrete: int, current_parameters: List) -> Tuple:
        return tuple([x for x in np.split(current_parameters, self.split_indices)])

    def __repr__(self) -> str:
        return "ContinuousParameters([" + ", ".join([str(s) for s in self.spaces]) + "])"
