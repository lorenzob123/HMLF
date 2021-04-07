from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from hmlf.common.type_aliases import RewardFunction, StageFunction


class BaseStageController(metaclass=ABCMeta):
    """
    Abstract baseclass for the stage controller classes.

     Args:
        reward_functions (Optional[List['RewardFunctions']]): Used to calculate different rewards for each stage.
            Defaults to None.
    """

    def __init__(self, reward_functions: Optional[List["RewardFunction"]] = None) -> None:
        self.reward_functions = reward_functions

    @abstractmethod
    def current_stage_is_done(self, obs: np.ndarray, current_stage: int) -> bool:
        """
        Calculates if the current stage is done. If so, the controller will move to the next skill in the sequence.

        Args:
            obs (np.ndarray): Current observation
            current_stage (int): Current stage/skill.

        Returns:
            bool: Whether the stage is done.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the stage controller to the initial state.
        """
        pass

    def can_calculate_reward(self) -> bool:
        """
        Used to detect if the controller can calculate custom rewards for each stage.

        Returns:
            bool: Whether the controller can calculate custom rewards for each stage.
        """
        return self.reward_functions is not None

    def calculate_reward(self, obs: np.ndarray, current_stage: int) -> float:
        """
        Calculates custom rewards using the provided reward functions.

        Args:
            obs (np.ndarray): Current observation
            current_stage (int): Current stage/skill.

        Returns:
            float: The reward.
        """
        current_reward_function = self.reward_functions[current_stage]
        return current_reward_function(obs)

    def _validate_arguments(self) -> None:
        if self.can_calculate_reward():
            assert isinstance(self.reward_functions, list)
            assert len(self.reward_functions) > 0


class OneStepPerStageController(BaseStageController):
    """
    Simple stage controller, that always returns true for `current_stage_is_done`. Thus each skill is executed only one time.

    Args:
        reward_functions (Optional[List['RewardFunctions']]): Used to calculate different rewards for each stage.
            Defaults to None.
    """

    def __init__(self, reward_functions: Optional[List["RewardFunction"]] = None) -> None:
        super().__init__(reward_functions)
        self._validate_arguments()

    def _validate_arguments(self) -> None:
        super()._validate_arguments()

    def current_stage_is_done(self, obs: np.ndarray, current_stage: int) -> bool:
        return True

    def reset(self) -> None:
        pass


class StateDependentStageController(BaseStageController):
    """
    Stage controller, that only moves to the next stage, if the criteria of the `stage_functions` are met
        (e.g. a goal position is reached).

    Args:
        stage_functions (List['StageFunction']): Used to detect, if a certain stage is done.
    reward_functions (Optional[List['RewardFunctions']]): Used to calculate different rewards for each stage.
        Defaults to None.

    """

    def __init__(
        self, stage_functions: List["StageFunction"], reward_functions: Optional[List["RewardFunction"]] = None
    ) -> None:
        super().__init__(reward_functions)
        self.stage_functions = stage_functions

        self._validate_arguments()

    def _validate_arguments(self) -> None:
        super()._validate_arguments()
        assert isinstance(self.stage_functions, list)
        assert len(self.stage_functions) > 0

    def current_stage_is_done(self, obs: np.ndarray, current_stage: int) -> bool:
        current_stage_function = self.stage_functions[current_stage]
        return current_stage_function(obs)

    def reset(self) -> None:
        pass
