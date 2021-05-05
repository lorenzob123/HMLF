import typing
from typing import List, Tuple, Union

import numpy as np

from hmlf.spaces.gym import Box, Space
from hmlf.spaces.hybrid_base import HybridBase


class ContinuousParameters(HybridBase):
    """
    A hybrid action space in the form of a tuple (i.e., product) of simpler space, where all ssubspaces are of type Box.
        Samples have the form (Box1.sample(), ..., BoxN.sample())

    :param spaces: The base action spaces of type Box.
    """

    def __init__(self, spaces: Union[List[Space], Tuple[Space]]):
        super().__init__(spaces)
        self.low = np.hstack([space.low for space in spaces])
        self.high = np.hstack([space.high for space in spaces])

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
        sample = []
        for i in range(actions.shape[0]):
            sample.append(np.split(actions[i], self.split_indices))
        return sample

    def _build_single_action(self, current_discrete: int, current_parameters: List) -> Tuple:
        return tuple([x for x in np.split(current_parameters, self.split_indices)])

    def __repr__(self) -> str:
        return "ContinuousParameters([" + ", ".join([str(s) for s in self.spaces]) + "])"
