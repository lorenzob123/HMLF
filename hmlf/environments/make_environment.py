from inspect import isclass
from typing import TYPE_CHECKING, List, Optional, Type, Union

from hmlf import spaces
from hmlf.environments.stage_controller import BaseStageController
from hmlf.environments.wrapper import OneHotWrapper, SequenceWrapper, SimpleHybridWrapper

if TYPE_CHECKING:
    from hmlf.common.base_class import BaseAlgorithm
    from hmlf.common.type_aliases import GymEnv


def make_environment(
    algorithm: Union[str, Type["BaseAlgorithm"]],
    env: "GymEnv",
    sequence: Optional[List[int]] = None,
    stage_controller: Optional[BaseStageController] = None,
) -> "GymEnv":

    algorithm_name = convert_algorithm_to_string(algorithm)

    if algorithm_name == "PDQN":
        return wrap_simple_hybrid(env)
    elif algorithm_name == "MPDQN":
        return wrap_simple_hybrid(env)
    elif algorithm_name == "SDDPG":
        return wrap_sequence(env, sequence, stage_controller)
    elif algorithm_name == "PADDPG":
        return wrap_one_hot(env)
    elif algorithm_name == "PPO":
        return wrap_simple_hybrid_if_tuple_action_space(env)
    elif algorithm_name in ["TD3", "DQN", "SAC", "A2C", "DDPG"]:
        return wrap_no_wrap(env)
    else:
        raise NotImplementedError(f"Found unknown class {str(algorithm)} of name {algorithm_name}.")


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


def wrap_one_hot(env: "GymEnv") -> OneHotWrapper:
    return OneHotWrapper(env)


def wrap_simple_hybrid(env: "GymEnv") -> SimpleHybridWrapper:
    return SimpleHybridWrapper(env)


def wrap_simple_hybrid_if_tuple_action_space(env: "GymEnv") -> SimpleHybridWrapper:
    if isinstance(env.action_space, spaces.Tuple):
        return wrap_simple_hybrid(env)
    else:
        return wrap_no_wrap(env)


def wrap_no_wrap(env: "GymEnv") -> "GymEnv":
    return env


def wrap_sequence(
    env: "GymEnv", sequence: Optional[List[int]] = None, stage_controller: Optional[BaseStageController] = None
) -> SequenceWrapper:
    assert isinstance(sequence, list), f"Provided sequence has to be of type list. Found {type(sequence)}"
    return SequenceWrapper(env, sequence, stage_controller)
