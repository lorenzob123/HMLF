import typing
from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Union

import numpy as np

from hmlf.spaces.gym import Box, Space
from hmlf.spaces.gym import Tuple as GymTuple


class HybridBase(GymTuple, metaclass=ABCMeta):
    """
    Abstract base class for the hybrid action spaces.

    :param spaces: The base action spaces of type Box.
    """

    def __init__(self, spaces: Union[List[Space], Tuple[Space]]):
        if isinstance(spaces, tuple):
            spaces = list(spaces)
        self.spaces = spaces
        self._validate_arguments()

        self.split_indices = self._get_split_indices_for_continuous_spaces()

    @abstractmethod
    def get_n_discrete_spaces(self) -> int:
        """
        Calculates the number of discrete spaces.

        :return: The number of discrete spaces.
        """
        pass

    @abstractmethod
    def get_n_discrete_options(self) -> int:
        """
        Calculates the number of discrete options available to the algorithm.
            E.g. the number of continuous spaces for a simple hybrid algorithm.

        return: The number of discrete spaces.
        """
        pass

    @abstractmethod
    def _get_continuous_spaces(self) -> List[Box]:
        pass

    @abstractmethod
    def format_action(self, actions: np.ndarray) -> List[typing.Tuple]:
        pass

    @abstractmethod
    def _build_single_action(self, current_discrete: int, current_parameters: List) -> Tuple:
        """ "
        Part of the algorithm template `build_action`.
            Provides to build a single action in the format of the subclass out of the discrete actions and the parameters.

        :param current_discrete: The discrete part of the action.
        :param current_parameters: The parameter part of the action.

        :return: The formatted action.
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    def _validate_arguments(self) -> None:
        assert isinstance(
            self.spaces, (list, tuple)
        ), f"spaces arguments needs to of type list/tuple. Found {type(self.spaces)}."
        assert len(self.spaces) > 0, f"spaces arguments needs to be non empty. Found {self.spaces}."
        for space in self.spaces:
            assert isinstance(space, Space), "Elements of spaces argument have to be subclasses of hmlf.spaces.Space"

    def build_action(self, discrete: np.ndarray, parameters: np.ndarray) -> List[typing.Tuple]:
        """
        Builds actions in the format of the `HybridBase` subclass.

        :param discrete: The discrete parts of the actions.
        :param parameters: The parameter parts of the action.

        :return: The actions in the format of the respective hybrid space.
        """
        parameters = self._preprocess_parameters(parameters)
        sample = []
        for i in range(discrete.shape[0]):
            sample.append(self._build_single_action(discrete[i], parameters[i]))
        return sample

    def _preprocess_parameters(self, parameters: np.ndarray) -> np.ndarray:
        """
        Part of the algorithm template `build_action`.
            Provides the ability to preprocess the parameters - e.g. clip them.

        :param parameters: [description]

        :return: [description]
        """
        return parameters

    def _get_split_indices_for_continuous_spaces(self) -> np.ndarray:
        """
        Provides the slice indices for the continuous spaces.

        :return: The slice indices.
        """
        return np.cumsum(self._get_dimensions_of_continuous_spaces()[:-1])

    def get_n_continuous_spaces(self) -> int:
        """
        Retuns the number of continuous spaces.

        :return: The number of continuous spaces.
        """
        return len(self._get_continuous_spaces())

    def get_n_continuous_options(self) -> int:
        """
        Returns the number of continuous options.
            E.g. the sum of continuous action space dimensions.

        :return: The number of continuous options.
        """
        return int(np.sum(self._get_dimensions_of_continuous_spaces()))

    def get_dimension(self) -> int:
        """
        Returns the total dimension of the hybrid space.

        :return: The total dimension.
        """
        return self.get_n_discrete_spaces() + self.get_n_continuous_options()

    def _get_dimensions_of_continuous_spaces(self) -> List[int]:
        continuous_spaces = self._get_continuous_spaces()
        # Since each space is one dimensional, shape[0] gets the dimension
        continuous_dimensions = [int(space.shape[0]) for space in continuous_spaces]
        return continuous_dimensions

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.spaces == other.spaces
