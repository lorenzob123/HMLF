from typing import Dict, List, Optional, Tuple

import gym
import numpy as np

from hmlf.spaces import Box, SimpleHybrid


class DummyHybrid(gym.Env):
    def __init__(
        self,
        parameter_dimensions: List[int],
        observation_dimension: int = 3,
        reward_bias: Optional[np.ndarray] = None,
    ):
        self.LIMIT = 1000
        self.PARAMETER_SUM_TARGET = 0
        self.N_MAX_STEPS = 50
        self.REWARD_CUTOFF = 0.995 - 1

        self.n_steps = 0
        self.parameter_dimensions = parameter_dimensions
        self.observation_dimension = observation_dimension
        self.n_parameter_spaces = len(parameter_dimensions)
        if reward_bias is None:
            reward_bias = np.zeros(self.n_parameter_spaces)
        self.reward_bias = reward_bias

        self._validate_arguments()
        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()

    def _validate_arguments(self):
        assert isinstance(self.reward_bias, np.ndarray)
        assert len(self.reward_bias) == self.n_parameter_spaces
        assert (
            type(self.parameter_dimensions) is list
        ), f"Please input parameter_dimensions of type list. Found {type(self.parameter_dimensions)}"
        assert (
            type(self.observation_dimension) is int
        ), f"Please input observation_dimension of type int. Found {type(self.observation_dimension)}"
        for dimension in self.parameter_dimensions:
            assert type(dimension) is int, f"Please input dimension of type int. Found {type(dimension)}"
            if dimension <= 0:
                raise ValueError(f"Dimensions have to be > 0. Found {dimension}")
        if len(self.parameter_dimensions) == 0:
            raise ValueError("Please parameter_dimensions of length >0. Found 0")

        assert (
            self.observation_dimension > 0
        ), f"Observation dimension has to be greater than 0. Found {self.observation_dimension}"

    def _build_action_space(self) -> SimpleHybrid:
        parameter_spaces = self._build_parameter_spaces()
        spaces = parameter_spaces
        return SimpleHybrid(spaces)

    def _build_parameter_spaces(self) -> List[Box]:
        return [Box(-self.LIMIT, self.LIMIT, shape=(dimension,)) for dimension in self.parameter_dimensions]

    def _build_observation_space(self) -> Box:
        return Box(-self.LIMIT, self.LIMIT, shape=(self.observation_dimension,))

    def reset(self) -> np.ndarray:
        self.n_steps = 0
        return self._get_observation()

    def step(self, action: Tuple[int, List[np.ndarray]]) -> Tuple[np.ndarray, float, bool, Dict]:
        self.n_steps += 1

        observation = self._get_observation()
        reward = self._compute_reward(action)
        is_done = self._compute_is_done(reward, action)

        return observation, reward, is_done, {}

    def _get_observation(self) -> np.ndarray:
        return np.random.normal(size=(self.observation_dimension,))

    def _compute_reward(self, action: Tuple[int, List[np.ndarray]]) -> float:
        discrete_action = self._get_discrete(action)
        choosen_parameter = action[1 + discrete_action]
        choosen_parameter_sum = np.sum(choosen_parameter)
        difference_to_target = np.abs(self.PARAMETER_SUM_TARGET - choosen_parameter_sum)

        reward = np.exp(-difference_to_target)
        reward = reward - 1 + self.reward_bias[discrete_action]
        return float(reward)

    def _get_discrete(self, action: Tuple[int, List[np.ndarray]]) -> int:
        return action[0]

    def _compute_is_done(self, last_reward: float, action: Tuple[int, List[np.ndarray]]) -> bool:
        discrete_action = self._get_discrete(action)
        is_done_steps = self.n_steps >= self.N_MAX_STEPS

        reward_cutoff = self.REWARD_CUTOFF
        last_reward_without_bias = last_reward - self.reward_bias[discrete_action]
        is_done_reward = bool(last_reward_without_bias >= reward_cutoff)
        return is_done_steps or is_done_reward
