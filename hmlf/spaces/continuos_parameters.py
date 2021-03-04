from typing import List, Tuple

import gym
import numpy as np

from hmlf import spaces


class ContinuosParameters(gym.spaces.Tuple):
    """
    A tuple (i.e., product) of Box spaces.
    Samples have the form (Box1.sample(), ..., BoxN.sample())
    Example usage:
    action_space = ContinuosParameters(spaces.Tuple(hmlf.spaces.Box(np.array((0, 1)), np.array((2, 3)))
                                                    hmlf.spaces.Box(0, 1, shape=(2,)))))
    action_space.sample() -> np.array([.2, 21.3, .3, .6])

    :param tuple of Box spaces
    """

    def __init__(self, spaces_list):
        self.spaces = spaces_list
        for i, space in enumerate(spaces_list):
            assert isinstance(
                space, spaces.Box
            ), f"Elements of ContinuosParameters have to be of type hmlf.spaces.Box. Failed for index {i}."

        dims_continous = self._get_continous_dims()
        self.continuous_dim = np.sum(dims_continous)

        self.continuous_low = np.hstack(tuple(self.spaces[i].low for i in range(len(self.spaces))))
        self.continuous_high = np.hstack(tuple(self.spaces[i].high for i in range(len(self.spaces))))

    def _get_continous_dims(self) -> List[int]:
        # Since each space is one dimensional, shape[0] gets the dimension
        dims = [space.shape[0] for space in self.spaces]
        return dims

    def get_dimension(self) -> int:
        dims_continous = self._get_continous_dims()
        return np.sum(dims_continous)

    def build_action(self, discrete, parameters: np.ndarray) -> List[Tuple]:
        # We clip the parameters
        # param_low = np.hstack(tuple(self.spaces[i].low for i in range(1, len(self.spaces))))
        # param_high = np.hstack(tuple(self.spaces[i].high for i in range(1, len(self.spaces))))
        # parameters = np.clip(parameters, param_low, param_high)

        # We prepare the split of the parameters for each discrete action
        dims_continous = self._get_continous_dims()
        split_indices = np.cumsum(dims_continous[:-1])

        # We format the full action for each environment
        sample = []
        for i in range(discrete.shape[0]):
            sample.append(np.split(parameters[i], split_indices))

        return sample

    def __repr__(self) -> str:
        return "ContinuosParameters(" + ", ".join([str(s) for s in self.spaces]) + ")"

    def __eq__(self, other) -> bool:
        return isinstance(other, ContinuosParameters) and self.spaces == other.spaces
