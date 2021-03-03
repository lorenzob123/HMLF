import typing
from typing import List

import numpy as np

from hmlf.spaces.simple_hybrid import SimpleHybrid


class OneHotHybrid(SimpleHybrid):
    """
    A tuple (i.e., product) of simpler spaces, where the first space is Discrete and the other are Box.
    Samples have the form (int, Box1.sample(), ..., BoxN.sample())
    Example usage:
    self.observation_space = spaces.Tuple((spaces.Discrete(2), hmlf.spaces.Box(np.array((0, 1)), np.array((2, 3)))))
    """

    def __init__(self, spaces):
        super().__init__(spaces)

    def get_dimension(self) -> int:
        dims_continuous = self._get_continuous_dims()
        return self.discrete_dim + np.sum(dims_continuous)

    def sample(self):
        discrete_action = np.zeros(self.spaces[0].n)
        np.put(discrete_action, self.spaces[0].sample(), 1)
        return np.hstack([discrete_action] + [space.sample() for space in self.spaces[1:]])

    def format_action(self, actions) -> List:
        discrete, parameters = actions[:, : self.discrete_dim], actions[:, self.discrete_dim :]
        discrete = np.argmax(discrete, axis=1)
        return self.build_action(discrete, parameters)

    def build_action(self, discrete: np.ndarray, parameters: np.ndarray) -> List[typing.Tuple]:
        # We clip the parameters
        param_low = np.hstack(tuple(self.spaces[i].low for i in range(1, len(self.spaces))))
        param_high = np.hstack(tuple(self.spaces[i].high for i in range(1, len(self.spaces))))
        parameters = np.clip(parameters, param_low, param_high)

        # We prepare the split of the parameters for each discrete action
        dims_continuous = self._get_continuous_dims()
        split_indices = np.cumsum(dims_continuous[:-1])

        # We format the full action for each environment
        sample = []
        for i in range(discrete.shape[0]):
            sample.append(tuple([discrete[i]] + np.split(parameters[i], split_indices)))

        return sample

    def __repr__(self) -> str:
        return "OneHotHybrid([" + ", ".join([str(s) for s in self.spaces]) + "])"

    def __eq__(self, other) -> bool:
        return isinstance(other, OneHotHybrid) and self.spaces == other.spaces
