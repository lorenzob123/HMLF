from hmlf.spaces.simple_hybrid import SimpleHybrid
from hmlf.environments.dummy_env import DummyEnv
from hmlf.common import callbacks
from hmlf.environments import ObstacleCourse_v2, OneHotWrapper, DummyEnv
from hmlf.spaces import OneHotHybrid, SimpleHybrid
from hmlf import PADDPG, PPO # , PDQN
from hmlf.common.callbacks import EvalCallback
import gym

class SimpleHybridWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = SimpleHybrid(env.action_space.spaces)
        print(self.action_space)



model = PPO("MlpPolicy", env=SimpleHybridWrapper(DummyEnv()), learning_rate=1e-4)
callback =callback=EvalCallback(eval_env=SimpleHybridWrapper(DummyEnv()),
                                eval_freq = 1000, n_eval_episodes=20)
model.learn(1e5, callback=callback)

