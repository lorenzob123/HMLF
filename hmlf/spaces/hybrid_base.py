import typing
from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np

from hmlf.spaces.gym import Box, Space, Tuple


class HybridBase(Tuple, metaclass=ABCMeta):
    def __init__(self, spaces: List[Space]):
        self.spaces = spaces
        self._validate_arguments()

        self.split_indices = self._get_split_indices_for_continuous_spaces()

    @abstractmethod
    def get_n_discrete_spaces(self) -> int:
        pass

    @abstractmethod
    def get_n_discrete_options(self) -> int:
        pass

    @abstractmethod
    def _get_continuous_spaces(self) -> List[Box]:
        pass

    @abstractmethod
    def format_action(self, actions: np.ndarray) -> List[typing.Tuple]:
        pass

    @abstractmethod
    def build_action(self, discrete: np.ndarray, parameters: np.ndarray) -> List[typing.Tuple]:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    def _validate_arguments(self) -> None:
        assert isinstance(self.spaces, list), f"spaces arguments needs to of type list. Found {type(self.spaces)}."
        assert len(self.spaces) > 0, f"spaces arguments needs to be non empty. Found {self.spaces}."
        for space in self.spaces:
            assert isinstance(space, Space), "Elements of spaces argument have to be subclasses of hmlf.spaces.Space"

    def _get_split_indices_for_continuous_spaces(self) -> np.ndarray:
        return np.cumsum(self._get_dimensions_of_continuous_spaces()[:-1])

    def get_n_continuous_spaces(self) -> int:
        return len(self._get_continuous_spaces())

    def get_n_continuous_options(self) -> int:
        return int(np.sum(self._get_dimensions_of_continuous_spaces()))

    def get_dimension(self) -> int:
        return self.get_n_discrete_spaces() + self.get_n_continuous_options()

    def _get_dimensions_of_continuous_spaces(self) -> List[int]:
        continuous_spaces = self._get_continuous_spaces()
        # Since each space is one dimensional, shape[0] gets the dimension
        continuous_dimensions = [int(space.shape[0]) for space in continuous_spaces]
        return continuous_dimensions

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.spaces == other.spaces
