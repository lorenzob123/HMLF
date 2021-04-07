from typing import List, Tuple, Union

import numpy as np

from hmlf.spaces.gym import Space
from hmlf.spaces.simple_hybrid import SimpleHybrid


class OneHotHybrid(SimpleHybrid):
    """
    A hybrid action space in the form of a tuple (i.e., product) of simpler space,
        where the first space is Discrete encoded as hot one econding and the other are of type Box.
        Samples have the form ((0, 0, ..., 1, 0, .., 0), Box1.sample(), ..., BoxN.sample())

    Args:
        spaces (Union[List[Space], Tuple]): The base action spaces of type Box.
    """

    def __init__(self, spaces: Union[List[Space], Tuple[Space]]):
        super().__init__(spaces)

    def get_n_discrete_spaces(self) -> int:
        return self.n_discrete_options

    def sample(self) -> Tuple:
        discrete_action = np.zeros(self.spaces[0].n)
        np.put(discrete_action, self.spaces[0].sample(), 1)
        return (discrete_action,) + tuple(space.sample() for space in self.spaces[1:])

    def format_action(self, actions) -> List[Tuple]:
        discrete, parameters = actions[:, : self.get_n_discrete_options()], actions[:, self.get_n_discrete_options() :]
        return self.build_action(discrete, parameters)

    def __repr__(self) -> str:
        return "OneHotHybrid([" + ", ".join([str(space) for space in self._get_continuous_spaces()]) + "])"
