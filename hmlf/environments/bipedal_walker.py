import gym
import numpy as np

from hmlf import spaces


class PaAcBipedalWalker(gym.Wrapper):
    """Wrapper for the BipedalWalker, where you can only move one leg per time step
    making the environment effectivly a hybrid space.

    """

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = spaces.Tuple(
            (
                spaces.Discrete(2),
                spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            )
        )

    def step(self, pa_action):
        leg = {"left": 0, "right": 1}
        action = np.zeros(4)

        if pa_action[0] == leg["left"]:
            action[:2] = pa_action[1]
        elif pa_action[0] == leg["right"]:
            action[2:] = pa_action[2]
        return self.env.step(action)
