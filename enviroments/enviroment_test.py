import gym
from gym import spaces
import numpy as np
from hmlf.common.vec_env import SubprocVecEnv


class DummyEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self):
        super(DummyEnv, self).__init__()

        # First entry in teh spaces.Tuple to be the selection of the skill
        # the remaining entries are the paramters for the specific skills
        # So if you have 2 skills the tuple will need 3 entries in total:
        # The first will be a a discrete space with n = 2
        # The second one will be the parameters for skill1
        # The third will be the parameters for skill2
        self.action_space = spaces.Tuple((spaces.Discrete(2),
                                            spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
                                            spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)))
        # self.action_space = spaces.Box(low=-1, high=1, shape=(1, ))
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(2,), dtype=np.float32)

    def step(self, action):
        observation = np.random.random(size=2).astype(np.float32)
        reward = np.random.rand()
        done = False if reward < .9 else True
        
        return observation, reward, done, {}

    def reset(self):
        observation = np.random.random(size=2)
        return observation.astype(np.float32)  # reward, done, info can't be included

    def render(self, mode='human'):
        pass

    def close (self):
        pass


if __name__ == "__main__":
    from hmlf import PADDPG, DDPG, PPO
    

    env = [lambda: DummyEnv() for ip in range(8)]
    env = SubprocVecEnv(env)
    print(env.action_space.dtype)

    model = PADDPG('MlpPolicy', DummyEnv(), verbose=1)
    model.learn(total_timesteps=1000000)
