import gym

from hmlf import PPO
from hmlf.common.callbacks import EvalCallback
from hmlf.environments import DummyEnv
from hmlf.spaces import SimpleHybrid


class SimpleHybridWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = SimpleHybrid(env.action_space.spaces)
        print(self.action_space)


model = PPO("MlpPolicy", env=SimpleHybridWrapper(DummyEnv()), learning_rate=1e-4)
callback = EvalCallback(eval_env=SimpleHybridWrapper(DummyEnv()), eval_freq=1000, n_eval_episodes=20)
model.learn(1e5, callback=callback)
