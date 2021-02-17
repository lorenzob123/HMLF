import gym 
from hmlf.spaces import OneHotHybrid, SimpleHybrid, ContinuosParameters
import numpy as np


class OneHotWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = OneHotHybrid(env.action_space.spaces)


class SimpleHybridWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = SimpleHybrid(env.action_space.spaces)


class SequenceWrapper(gym.Wrapper):
    def __init__(self,  env, sequence):
        super().__init__(env)
        self.env = env
        
        # The action space does not need to include the discrete action
        self.action_space = ContinuosParameters(env.action_space.spaces[1:])

        # We append the discrete action as an obseration
        low = np.append(0, self.env.observation_space.low)
        high = np.append(env.action_space.spaces[0].n, self.env.observation_space.high)
        self.observation_space = gym.spaces.Box(low=low, high=high)
        self.sequence = sequence
        self.queque = list(self.sequence)
        self.current_action = None
    
    def reset(self):
        self.queque = list(self.sequence)
        obs = self.env.reset()
        self.current_action = self.queque.pop(0)
        return np.append(self.current_action, obs)

    def step(self, action):
        print (action)
        print((self.current_action, *action))
        obs, r, done, info = self.env.step((self.current_action, *action))
        if self.queque:
            self.current_action = self.queque.pop(0)
        else:
            done = True
            # default option
            self.current_action = 0

        return np.append(self.current_action, obs), r, done, info
        

