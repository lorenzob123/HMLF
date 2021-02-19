from typing import List
from hmlf.spaces import Space, Tuple, Discrete, Box, SimpleHybrid

import numpy as np

class OneHotHybrid(SimpleHybrid):
    """
    A tuple (i.e., product) of simpler spaces, where the first space is Discrete and the other are Box.
    Samples have the form (int, Box1.sample(), ..., BoxN.sample())
    Example usage:
    self.observation_space = spaces.Tuple((spaces.Discrete(2), hmlf.spaces.Box(np.array((0, 1)), np.array((2, 3)))))
    """
    def __init__(self, spaces):
        self.spaces = spaces
        for i, space in enumerate(spaces):
            assert isinstance(space, Space), "Elements of the SimpleHybrid must be instances of hmlf.Space"
            if i == 0:
                assert isinstance(space, Discrete), "First element of SimpleHybrid has to be of type hmlf.spaces.Discrete"
            else:
                assert isinstance(space, Box), f"Later (index > 0) elements of SimpleHybrid has to be of type hmlf.spaces.Box. Failed for index {i}."
        
        self.discrete_dim = self.spaces[0].n
        dims_continuous = self._get_continuous_dims()
        self.continuous_dim = np.sum(dims_continuous)


    def _get_continuous_dims(self) -> List[int]:
        # Since each space is one dimensional, shape[0] gets the dimension
        dims = [space.shape[0] for space in self.spaces[1:]]  
        return dims

    def get_dimension(self) -> int:
        dims_continuous = self._get_continuous_dims()
        return self.discrete_dim + np.sum(dims_continuous)

    def sample(self):
        discrete_action = np.zeros(self.spaces[0].n)
        np.put(discrete_action, self.spaces[0].sample(), 1)
        return np.hstack([discrete_action]+ [space.sample() for space in self.spaces[1:]])

    def format_action(self, actions) -> Tuple:

        discrete, parameters = actions[:, :self.discrete_dim], actions[:, self.discrete_dim:]

        discrete = np.argmax(discrete, axis=1)

        # We clip the parameters
        param_low = np.hstack(tuple(self.spaces[i].low for i in range(1, len(self.spaces))))
        param_high = np.hstack(tuple(self.spaces[i].high for i in range(1, len(self.spaces))))
        parameters = np.clip(parameters, param_low, param_high)
        
        # We prepare the split of the parameters for each discrete action
        dims_continuous = self._get_continuous_dims()
        split_indizes = np.cumsum(dims_continuous[:-1])

        # We format the full action for each environment
        sample = []
        for i in range(discrete.shape[0]):
            sample.append([discrete[i]] + np.split(parameters[i], split_indizes))

        return sample

    def __repr__(self) -> str:
        return "SimpleHybrid(" + ", ". join([str(s) for s in self.spaces]) + ")"
      
    def __eq__(self, other) -> bool:
        return isinstance(other, SimpleHybrid) and self.spaces == other.spaces


if __name__ == "__main__":

    action_space = OneHotHybrid((Discrete(2), Box(-1, 1, shape=(1, )), Box(-1, 1, shape=(1, ))))
    actions = np.vstack((action_space.sample(), action_space.sample()))
    print(action_space.format_action(actions))