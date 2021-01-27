import gym
import numpy as np
from gym.spaces import Box, Discrete, Tuple


class ObstacleCourse(gym.Env):
    
    
    def __init__(self):


        self.max_move = .2
        self.max_jump = 0.05
        self.n_obstacles = 3
        self.goal_position = .95
        self.obstacle_thickness = 0.01
        self.jump_threshold =.1
        self.max_timesteps = 30
        self.goal_threshold = 0.05
        tmp = np.ones(1)
        self.action_space = Tuple((Discrete(2), Box(-self.max_move, self.max_move, (1,)), Box(0, 1, (1,))))
        self.observation_space = Box(low=np.zeros(7), high=np.ones(7))

        self.position = 0
        self.time = 0
        self.obstacle_position = np.zeros(self.n_obstacles)
        self.obstacle_target_height = np.zeros(self.n_obstacles)

    def reset(self):
        self.obstacle_position, self.obstacle_target_height = self._build_obstacles()
        self.position = 0
        self.time = 0
        return self.get_observation()

        
    def _build_obstacles(self):
        obstacle_position=np.zeros(3)
        while np.any(np.diff(obstacle_position) < self.max_jump):
            obstacle_position = np.random.rand(3)*(self.goal_position - 2* self.max_jump) +self.max_jump
            obstacle_position.sort()

        return obstacle_position, np.random.rand(3)


    def step(self, action):
        move, jump = 0, 1

        if action[0] == move:
            observation = self._move(action[1][0])
        elif action[0]== jump:
            observation = self._jump(action[2][0])
        r, done = self.compute_reward_done()
        return observation, r, done, {}


    def _move(self, movement):
        pre_relative_position = self.obstacle_position-self.position
        post_relative_position = self.obstacle_position-self.position - movement
        change = np.any((pre_relative_position * post_relative_position) < 0)

        if change:
            obstacle_idx = np.where(pre_relative_position * post_relative_position < 0)[0][0]
            if movement >= 0:
                self.position = self.obstacle_position[obstacle_idx] - self.obstacle_thickness
            elif movement < 0:
                self.position = self.obstacle_position[obstacle_idx] + self.obstacle_thickness

        else:
            self.position += movement


        self.position = np.clip(self.position, 0, 1)

        return self.get_observation()


    def _jump(self, height):
        pre_relative_position = self.obstacle_position-self.position
        post_relative_position = self.obstacle_position-self.position - self.max_jump
        change = np.any((pre_relative_position * post_relative_position) < 0)
        if change:
            obstacle_idx = np.where(pre_relative_position * post_relative_position < 0)[0][0]
            if abs(height-self.obstacle_target_height[obstacle_idx]) < self.jump_threshold:
                self.position += self.max_jump
        else:
            self.position += self.max_jump
        
        self.position = np.clip(self.position, 0, 1)
        
        return self.get_observation()


    def get_observation(self):
        return np.hstack((np.array([self.position]), self.obstacle_position, self.obstacle_target_height))


    def compute_reward_done(self):
        distance = abs(self.position - self.goal_position)
        if distance < self.goal_threshold:
            return 1., True
        
        self.time += 1

        return -distance, self.time >= self.max_timesteps



if __name__ == "__main__":
    from hmlf import PADDPG, DDPG, PPO
    from hmlf.common.vec_env import SubprocVecEnv
    from hmlf.common.callbacks import EvalCallback
    from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

    env = [lambda: ObstacleCourse() for ip in range(16)]
    env = SubprocVecEnv(env)

    model = PPO('MlpPolicy', env=env,  verbose=2)

    obs = ObstacleCourse().reset()

    model.learn(total_timesteps=1e6, callback=EvalCallback(eval_env=ObstacleCourse(), eval_freq = 1000, n_eval_episodes=20))