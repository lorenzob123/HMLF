import typing
from typing import List, Tuple, Union

import numpy as np

from hmlf.spaces.gym import Box, Discrete, Space
from hmlf.spaces.hybrid_base import HybridBase


class SimpleHybrid(HybridBase):
    """
    A hybrid action space in the form of a tuple (i.e., product) of simpler space,
        where the first space is Discrete and the other are Box.
        Samples have the form (int, Box1.sample(), ..., BoxN.sample())

    :param spaces: The base action spaces of type Box.
    """

    def __init__(self, spaces: Union[List[Space], Tuple[Space]]):
        assert isinstance(spaces, (list, tuple)), "spaces argument needs to be of type list/tuple"
        self.n_discrete_options = len(spaces)
        spaces = [Discrete(self.n_discrete_options)] + list(spaces)
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

    def _preprocess_parameters(self, parameters: np.ndarray) -> np.ndarray:
        return np.clip(parameters, self.continuous_low, self.continuous_high)

    def _build_single_action(self, current_discrete: int, current_parameters: List) -> Tuple:
        return tuple([current_discrete] + np.split(current_parameters, self.split_indices))

    def __repr__(self) -> str:
        return "SimpleHybrid([" + ", ".join([repr(space) for space in self._get_continuous_spaces()]) + "])"
