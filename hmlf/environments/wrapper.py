import gym
from hmlf.spaces import OneHotHybrid, SimpleHybrid


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
        print(self.action_space)
