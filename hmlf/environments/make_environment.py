from inspect import isclass
from typing import TYPE_CHECKING, Callable, List, Optional, Type, Union

from hmlf import spaces
from hmlf.environments.stage_controller import BaseStageController
from hmlf.environments.wrapper import OneHotWrapper, SequenceWrapper, SimpleHybridWrapper

if TYPE_CHECKING:
    from hmlf.common.base_class import BaseAlgorithm
    from hmlf.common.type_aliases import GymEnv


_algorithm_wrappers = {}


def register_algorithm_for_make_environment(
    algorithm: Union[str, Type["BaseAlgorithm"]], wrapper_function: Callable[["GymEnv"], "GymEnv"]
) -> None:
    global _algorithm_wrappers
    algorithm_name = convert_algorithm_to_string(algorithm)
    _algorithm_wrappers[algorithm_name] = wrapper_function


def make_environment(
    algorithm: Union[str, Type["BaseAlgorithm"]],
    env: "GymEnv",
    sequence: Optional[List[int]] = None,
    stage_controller: Optional[BaseStageController] = None,
) -> "GymEnv":

    algorithm_name = convert_algorithm_to_string(algorithm)

    if algorithm_is_registered(algorithm_name):
        wrapper_function = get_from_registry(algorithm_name)
        return wrapper_function(env, sequence=sequence, stage_controller=stage_controller)
    else:
        raise NotImplementedError(f"Found unknown class {str(algorithm)} of name {algorithm_name}.")


def algorithm_is_registered(algorithm_name: str) -> bool:
    return algorithm_name in _algorithm_wrappers


def get_from_registry(algorithm_name: str) -> Callable:
    if algorithm_name in _algorithm_wrappers:
        return _algorithm_wrappers[algorithm_name]
    else:
        raise ValueError(f"Algorithm name {algorithm_name} not found in registry.")


def convert_algorithm_to_string(algorithm: Union[str, Type["BaseAlgorithm"]]) -> str:
    if isclass(algorithm):
        return convert_class_to_string(algorithm)
    else:
        return algorithm


def convert_class_to_string(algorithm: Type["BaseAlgorithm"]) -> str:
    representation = str(algorithm)
    sanitized_representation = representation.replace("'", "").replace(">", "")
    dot_separated_parts = sanitized_representation.split(".")
    class_name = dot_separated_parts[-1]
    return class_name


def wrap_one_hot(env: "GymEnv", **kwargs) -> OneHotWrapper:
    return OneHotWrapper(env)


def wrap_simple_hybrid(env: "GymEnv", **kwargs) -> SimpleHybridWrapper:
    return SimpleHybridWrapper(env)


def wrap_simple_hybrid_if_tuple_action_space(env: "GymEnv", **kwargs) -> SimpleHybridWrapper:
    if isinstance(env.action_space, spaces.Tuple):
        return wrap_simple_hybrid(env)
    else:
        return wrap_no_wrap(env)


def wrap_no_wrap(env: "GymEnv", **kwargs) -> "GymEnv":
    return env


def wrap_sequence(env: "GymEnv", **kwargs) -> SequenceWrapper:
    assert "sequence" in kwargs, "Sequence must be provided to wrap_sequence."
    sequence = kwargs["sequence"]
    if "stage_controller" in kwargs:
        stage_controller = kwargs["stage_controller"]
    assert isinstance(sequence, list), f"Provided sequence has to be of type list. Found {type(sequence)}"
    return SequenceWrapper(env, sequence, stage_controller)
