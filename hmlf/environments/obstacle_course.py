import gym
import numpy as np

from hmlf.spaces import Box, SimpleHybrid


class ObstacleCourse(gym.Env):
    """Simple environment for benchmarking hybrid methods. It is an obsticle course the agent has to traverse by either
    jumping of running. When running you move longer distances but you get blocked by obstacles, when jumping with the
    correct height you can skip over an obstacle. The objective for the agent is to traverse this obstacle course in
    the least time possible, while getting over 3 obstacles. These are placed randomly between the agent's starting position
    and the goal.

    The action space includes 2 tasks (jump, run) and one parameter for each task (jump height, run distance).
    The observation space includes :
       - current position
       - goal position x3
       - goal target height
    """

    def __init__(self):

        self.max_move = np.float32(0.2)
        self.max_jump = 0.05
        self.n_obstacles = 3
        self.goal_position = 0.95
        self.obstacle_thickness = 0.01
        self.jump_threshold = 0.1
        self.max_timesteps = 30
        self.goal_threshold = 0.05
        self.action_space = SimpleHybrid([Box(-self.max_move, self.max_move, (1,)), Box(np.float32(0), np.float32(1), (1,))])
        self.observation_space = Box(low=np.zeros(7, dtype=np.float32), high=np.ones(7, dtype=np.float32))

        self.position = 0
        self.time = 0
        self.obstacle_position = np.zeros(self.n_obstacles)
        self.obstacle_target_height = np.zeros(self.n_obstacles)

    def reset(self):
        # Set up obstacle course
        self.obstacle_position, self.obstacle_target_height = self._build_obstacles()
        self.position = 0
        self.time = 0
        return self.get_observation()

    def _build_obstacles(self):
        # Choose obstacle position at random within `[max_jump, goal_position - 2 max_jump] `
        # Discard obstacle positions if two obstacles are within a max jump from each other
        obstacle_position = np.zeros(3)
        while np.any(np.diff(obstacle_position) < self.max_jump):
            obstacle_position = np.random.rand(3) * (self.goal_position - 2 * self.max_jump) + self.max_jump
            obstacle_position.sort()

        return obstacle_position, np.random.rand(3)

    def step(self, action):
        move, jump = 0, 1

        if action[0] == move:
            observation = self._move(action[1][0])
        elif action[0] == jump:
            observation = self._jump(action[2][0])

        r, done = self.compute_reward_done()
        return observation, r, done, {}

    def _move(self, movement) -> np.ndarray:
        """Performs the move action

        Args:
            movement (float): how far the agent plans to move

        Returns:
            observations (np.ndarray): observations after performing move
        """
        # If the relative position to an obstacle would change if the agent moved with a step of `movement`, then
        # stop agent right before the obstacle
        pre_relative_position = self.obstacle_position - self.position
        post_relative_position = self.obstacle_position - self.position - movement
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
        """Performs the jump action

        Args:
            movement (float): how height the agent jumps

        Returns:
            observations (np.ndarray): observations after performing jump
        """
        # If the jump would cause a change in the relative poision to an obstacle, this function
        # checks if the height parameter is close enough to the obstacle's height target.
        pre_relative_position = self.obstacle_position - self.position
        post_relative_position = self.obstacle_position - self.position - self.max_jump
        change = np.any((pre_relative_position * post_relative_position) < 0)
        if change:
            obstacle_idx = np.where(pre_relative_position * post_relative_position < 0)[0][0]
            if abs(height - self.obstacle_target_height[obstacle_idx]) < self.jump_threshold:
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
            return 1.0, True
        self.time += 1
        return -distance, self.time >= self.max_timesteps
