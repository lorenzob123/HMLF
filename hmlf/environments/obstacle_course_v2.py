import gym
import numpy as np
from hmlf.spaces import SimpleHybrid, Box, Discrete


class ObstacleCourse_v2(gym.Env):
    
    
    def __init__(self):


        self.max_move = np.float32(.4)
        self.max_jump = 0.05
        self.n_obstacles = 3
        self.goal_position = .95
        self.obstacle_thickness = 0.01
        self.jump_threshold =.1
        self.max_timesteps = 100
        self.goal_threshold = 0.05
        tmp = np.ones(1)
        self.action_space = SimpleHybrid((Discrete(2),
                                   Box(0, self.max_move, (1,)),
                                   Box(np.float32(0), np.float32(1), (1,))))
        self.observation_space = Box(low=np.zeros(3), high=np.ones(3), dtype=np.float32)

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
            self.position = self.obstacle_position[obstacle_idx] - self.obstacle_thickness


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
        next_obstacles = np.argwhere((self.obstacle_position - self.position) > 0)
        if next_obstacles.size > 0:
            distance_next = self.obstacle_position[next_obstacles[0]][0]
            height_next = self.obstacle_target_height[next_obstacles[0]][0]
        else:
            distance_next = 1
            height_next = 1
        # print(np.array([self.position, distance_next, height_next]))
        return np.array([self.position, distance_next, height_next])


    def compute_reward_done(self):
        distance = abs(self.position - self.goal_position)
        if distance < self.goal_threshold:
            return 1., True
        
        self.time += 1

        return -distance, self.time >= self.max_timesteps
