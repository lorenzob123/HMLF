import gym
import numpy as np


class PaAcBipedalWalker(gym.Wrapper):
    
    
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(2),
                                             gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                                             gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)))
    
    def step(self, pa_action):
        leg = {'left': 0, 'right':1}
        action = np.zeros(4)

        if pa_action[0] == leg['left']:
            action[:2] =  pa_action[1]
        elif pa_action[0] == leg['right']:
            action[2:] = pa_action[2]
        return self.env.step(action)