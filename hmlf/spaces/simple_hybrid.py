from typing import List, Tuple

import numpy as np

from hmlf import spaces


class SimpleHybrid(spaces.Tuple):
    """
    A tuple (i.e., product) of simpler spaces, where the first space is Discrete and the other are Box.
    Samples have the form (int, Box1.sample(), ..., BoxN.sample())
    Example usage:
    self.observation_space = spaces.Tuple((spaces.Discrete(2), hmlf.spaces.Box(np.array((0, 1)), np.array((2, 3)))))
    """

    def __init__(self, spaces_list: List[spaces.Space]):
        self.spaces = spaces_list
        for i, space in enumerate(spaces_list):
            assert isinstance(space, spaces.Space), "Elements of the SimpleHybrid must be instances of hmlf.Space"
            if i == 0:
                assert isinstance(
                    space, spaces.Discrete
                ), "First element of SimpleHybrid has to be of type hmlf.spaces.Discrete"
            else:
                assert isinstance(
                    space, spaces.Box
                ), f"Later (index > 0) elements of SimpleHybrid has to be of type hmlf.spaces.Box. Failed for index {i}."

        self.discrete_dim = self.spaces[0].n
        dims_continuous = self._get_continuous_dims()
        self.continuous_dim = np.sum(dims_continuous)

        self.continuous_low = np.hstack(tuple(self.spaces[i].low for i in range(1, len(self.spaces))))
        self.continuous_high = np.hstack(tuple(self.spaces[i].high for i in range(1, len(self.spaces))))

    def _get_continuous_dims(self) -> List[int]:
        # Since each space is one dimensional, shape[0] gets the dimension
        dims = [space.shape[0] for space in self.spaces[1:]]
        return dims

    def get_dimension(self) -> int:
        dims_continuous = self._get_continuous_dims()
        return 1 + np.sum(dims_continuous)

    def format_action(self, actions: np.ndarray) -> Tuple:
        discrete, parameters = actions[:, 0], actions[:, 1:]

        return self.build_action(discrete, parameters)

    def build_action(self, discrete: np.array, parameters: np.ndarray) -> List[Tuple]:
        # We clip the parameters
        parameters = np.clip(parameters, self.continuous_low, self.continuous_high)

        # We prepare the split of the parameters for each discrete action
        dims_continuous = self._get_continuous_dims()
        split_indizes = np.cumsum(dims_continuous[:-1])

        # We format the full action for each environment
        sample = []
        for i in range(discrete.shape[0]):
            sample.append([discrete[i]] + np.split(parameters[i], split_indizes))

        return sample

    def __repr__(self) -> str:
        return "SimpleHybrid(" + ", ".join([str(s) for s in self.spaces]) + ")"

    def __eq__(self, other) -> bool:
        return isinstance(other, SimpleHybrid) and self.spaces == other.spaces
