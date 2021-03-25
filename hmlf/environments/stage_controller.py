from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, List, Optional

import numpy as np

if TYPE_CHECKING:
    from hmlf.common.type_aliases import RewardFunction, StageFunction


class BaseStageController(metaclass=ABCMeta):
    def __init__(self, reward_functions: Optional[List["RewardFunction"]] = None) -> None:
        self.reward_functions = reward_functions

    @abstractmethod
    def current_stage_is_done(self, obs: np.ndarray, current_stage: int) -> bool:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    def can_calculate_reward(self) -> bool:
        return self.reward_functions is not None

    def calculate_reward(self, obs: np.ndarray, current_stage: int) -> float:
        current_reward_function = self.reward_functions[current_stage]
        return current_reward_function(obs)

    def _validate_arguments(self) -> None:
        if self.can_calculate_reward():
            assert isinstance(self.reward_functions, list)
            assert len(self.reward_functions) > 0


class OneStepPerStageController(BaseStageController):
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
