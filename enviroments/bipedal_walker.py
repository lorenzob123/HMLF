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
        




if __name__ == "__main__":
    from hmlf import PADDPG, DDPG, PPO
    from hmlf.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

    hyperparams = {
                       "policy": 'MlpPolicy',
                       "gamma": 0.98,
                       "buffer_size": 200000,
                       "learning_starts": 10000,
                       "gradient_steps": -1,
                       "n_episodes_rollout": 1,
                       "learning_rate":  1e-3,
                       "policy_kwargs": dict(net_arch=[400, 300])}
    hyperparams["action_noise"] = NormalActionNoise(
            mean=np.zeros(4), sigma=.1 * np.ones(4))

    env = [lambda: PaAcBipedalWalker(gym.make('BipedalWalker-v3')) for ip in range(4)]
    env = SubprocVecEnv(env)
    print(env.action_space.dtype)

    model = DDPG(env=gym.make('BipedalWalker-v3'), **hyperparams,  verbose=1)
    model.learn(total_timesteps=1e6)