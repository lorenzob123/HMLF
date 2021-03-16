from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from hmlf.common.type_aliases import StageFunction


class BaseStageController(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def current_stage_is_done(self, obs: np.ndarray, current_stage: int) -> bool:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    def _validate_arguments(self) -> None:
        pass


class OneStepPerStageController(BaseStageController):
    def __init__(self) -> None:
        super().__init__()

    def current_stage_is_done(self, obs: np.ndarray, current_stage: int) -> bool:
        return True

    def reset(self) -> None:
        pass


class StateDependentStageController(BaseStageController):
    def __init__(self, stage_functions: List["StageFunction"]) -> None:
        super().__init__()
        self.stage_functions = stage_functions

        self._validate_arguments()

    def _validate_arguments(self) -> None:
        assert isinstance(self.stage_functions, list)
        assert len(self.stage_functions) > 0

    def current_stage_is_done(self, obs: np.ndarray, current_stage: int) -> bool:
        current_stage_function = self.stage_functions[current_stage]
        return current_stage_function(obs)

    def reset(self) -> None:
        pass
