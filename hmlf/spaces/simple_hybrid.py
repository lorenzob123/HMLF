from typing import List
from hmlf.spaces import Space
from hmlf.spaces import Tuple
from hmlf.spaces import Discrete

import numpy as np

class SimpleHybrid(Tuple):
    """
    A tuple (i.e., product) of simpler spaces, where the first space is Discrete and the other are Box.
    Samples have the form (int, Box1.sample(), ..., BoxN.sample())
    Example usage:
    self.observation_space = spaces.Tuple((spaces.Discrete(2), gym.spaces.Box(np.array((0, 1)), np.array((2, 3)))))
    """
    def __init__(self, spaces):
        self.spaces = spaces
        for i, space in enumerate(spaces):
            assert isinstance(space, Space), "Elements of the SimpleHybrid must be instances of gym.Space"
            if i == 0:
                assert isinstance(space, Discrete), "First element of SimpleHybrid has to be of type hmlf.spaces.Discrete"
            else:
                assert isinstance(space, Discrete), f"Later (index > 0) elements of SimpleHybrid has to be of type hmlf.spaces.Box. Failed for index {i}."
        
        super(SimpleHybrid, self).__init__(None, None)


    def _get_continous_dims(self) -> List[int]:
        # Since each space is one dimensional, shape[0] gets the dimension
        dims = [space.shape[0] for space in self.spaces[1:]]  
        return dims

    def get_dimension(self) -> int:
        dims_continous = self._get_continous_dims()
        return 1 + np.sum(dims_continous)

    def make_sample(self, discrete: int, parameters: np.ndarray) -> Tuple:
        dims_continous = self._get_continous_dims()
        # Skip the last one, because cumsum returns list with N values for N 
        split_indizes = np.cumsum(dims_continous[:-1])
        # Returns list with parameters for each action subspace

        split = np.split(parameters, split_indizes)
        for i, space_sample in enumerate(split):
            sample_action_space = self.spaces[1 + i] # First one is discrete
            # Clips action to subspace boundaries
            split[i] = np.clip(split[i], sample_action_space.low, sample_action_space.high)

        sample = tuple([discrete, *split])
        return sample

    def sample(self) -> Tuple:
        return tuple([space.sample() for space in self.spaces])

    def __repr__(self) -> str:
        return "SimpleTuple(" + ", ". join([str(s) for s in self.spaces]) + ")"

    def __len__(self) -> int:
        return len(self.spaces)
      
    def __eq__(self, other) -> bool:
        return isinstance(other, SimpleHybrid) and self.spaces == other.spaces