from typing import TYPE_CHECKING, List, Optional

import numpy as np
from gym import Wrapper

from hmlf import spaces

if TYPE_CHECKING:
    from hmlf.common.type_aliases import GymEnv

from hmlf.environments.sequence_curator import SequenceCurator
from hmlf.environments.stage_controller import BaseStageController, OneStepPerStageController
from hmlf.spaces import ContinuousParameters, OneHotHybrid, SimpleHybrid


class OneHotWrapper(Wrapper):
    """
    Wraps an environment with an `OneHotHybrid` action space.

    :param env: The environment to be wrapped.
    """

    def __init__(self, env: "GymEnv"):
        super().__init__(env)
        self.env = env
        self.action_space = OneHotHybrid(env.action_space.spaces[1:])

    def step(self, action):
        discrete = np.argmax(action[0])
        return self.env.step((discrete, *action[1:]))


class SimpleHybridWrapper(Wrapper):
    """
    Wraps an environment with a `SimpleHybrid` action space.

    :param env: The environment to be wrapped.
    """

    def __init__(self, env: "GymEnv"):
        super().__init__(env)
        self.env = env
        self.action_space = SimpleHybrid(env.action_space.spaces[1:])


class SequenceWrapper(Wrapper):
    """
    Wraps an environment with a sequence of skills for use with algorithms like SDDPG.
        The action space will be of type `ContinuousParameters`, because the current skill of the sequence
        is controller by the `SequenceWrapper`.

    :param env: The environment to be wrapped.
    :param sequence: The sequence of skills.
    :param stage_controller: If given, the `stage_controller` controls the transitions between stages. Defaults to None.
    """

    def __init__(self, env: "GymEnv", sequence: List[int], stage_controller: Optional[BaseStageController] = None):
        super().__init__(env)
        self.env = env

        self._set_up_action_space()
        self._set_up_observation_space()

        self.sequence_curator = SequenceCurator(sequence)
        if stage_controller is None:
            stage_controller = OneStepPerStageController()
        self.stage_controller = stage_controller

    def _set_up_action_space(self) -> None:
        # The action space does not need to include the discrete action
        self.action_space = ContinuousParameters(self.env.action_space.spaces[1:])

    def _set_up_observation_space(self) -> None:
        # We append the discrete action as an obseration
        low = np.append(0, self.env.observation_space.low)
        high = np.append(self.env.action_space.spaces[0].n, self.env.observation_space.high)
        self.observation_space = spaces.Box(low=low, high=high)

    def reset(self):
        self.sequence_curator.reset()
        self.stage_controller.reset()
        obs = self.env.reset()
        return self._preprend_current_stage_to_observation(self.sequence_curator.get_current(), obs)

    def _preprend_current_stage_to_observation(self, current_stage: int, obs: np.ndarray) -> np.ndarray:
        return np.append(current_stage, obs)

    def step(self, action):
        current_sequence_part = self.sequence_curator.get_current()
        obs, r, done, info = self.env.step((current_sequence_part, *action))

        if self.stage_controller.can_calculate_reward():
            r = self.stage_controller.calculate_reward(obs, current_sequence_part)

        current_stage_is_done = self.stage_controller.current_stage_is_done(obs, current_sequence_part)
        if current_stage_is_done:
            if self.sequence_curator.has_next():
                current_sequence_part = self.sequence_curator.next()
            else:
                done = True

        return self._preprend_current_stage_to_observation(current_sequence_part, obs), r, done, info
