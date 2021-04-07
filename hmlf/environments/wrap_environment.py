from typing import TYPE_CHECKING, Callable, List, Optional, Type, Union

from hmlf import spaces
from hmlf.common.utils import convert_algorithm_to_string
from hmlf.environments.stage_controller import BaseStageController

if TYPE_CHECKING:
    from hmlf.common.base_class import BaseAlgorithm
    from hmlf.common.type_aliases import GymEnv

from hmlf.environments.wrapper import OneHotWrapper, SequenceWrapper, SimpleHybridWrapper

_algorithm_wrapper_registry = {}


def register_algorithm_for_wrap_environment(
    algorithm: Union[str, Type["BaseAlgorithm"]], wrapper_function: Callable[["GymEnv"], "GymEnv"]
) -> None:
    """
    Registers a algorithm for usage with the `wrap_environment` function

    :param algorithm: The name of the algorithm. Class types are converted to strings.
    :param wrapper_function: The function, that will to the actual wrapping. Can take kwargs.
    """
    global _algorithm_wrapper_registry
    algorithm_name = convert_algorithm_to_string(algorithm)
    _algorithm_wrapper_registry[algorithm_name] = wrapper_function


def wrap_environment(
    algorithm: Union[str, Type["BaseAlgorithm"]],
    env: "GymEnv",
    sequence: Optional[List[int]] = None,
    stage_controller: Optional[BaseStageController] = None,
) -> "GymEnv":
    """
    Returns a wrapped version of a given environment with the hybrid wrappers needed for the algorithm.

    :param algorithm: The name of the algorithm. Class types are converted to strings.
    :param env: The environment to be wrapped.
    :param sequence: Passed to a SequenceWrapper if appropiate.
    :param stage_controller: Passed to a SequenceWrapper if appropiate.

    :return: The wrapped environment.
    """

    algorithm_name = convert_algorithm_to_string(algorithm)

    if algorithm_is_registered(algorithm_name):
        wrapper_function = get_from_algorithm_wrapper_registry(algorithm_name)
        return wrapper_function(env, sequence=sequence, stage_controller=stage_controller)
    else:
        raise NotImplementedError(f"Found unknown class {str(algorithm)} of name {algorithm_name}.")


def algorithm_is_registered(algorithm_name: str) -> bool:
    return algorithm_name in _algorithm_wrapper_registry


def get_from_algorithm_wrapper_registry(algorithm_name: str) -> Callable:
    """
    Gets a wrapper function from algorithm wrapper registry.

    :param algorithm_name: Algorithm for which the wrapper should be returned.

    :return: The previously registered wrapper function.
    """
    if algorithm_name in _algorithm_wrapper_registry:
        return _algorithm_wrapper_registry[algorithm_name]
    else:
        raise ValueError(f"Algorithm name {algorithm_name} not found in registry.")


def wrap_one_hot(env: "GymEnv", **kwargs) -> "OneHotWrapper":
    """
    Wraps a environment with a `OneHotWrapper`. Ignores kwargs.

    :param env: The environment to be wrapped.

    :return: The wrapped environment.
    """
    return OneHotWrapper(env)


def wrap_simple_hybrid(env: "GymEnv", **kwargs) -> "SimpleHybridWrapper":
    """
    Wraps a environment with a `SimpleHybridWrapper`. Ignores kwargs.

    :param env: The environment to be wrapped.

    :return: The wrapped environment.
    """
    return SimpleHybridWrapper(env)


def wrap_simple_hybrid_if_tuple_action_space(env: "GymEnv", **kwargs) -> "SimpleHybridWrapper":
    """
    Wraps a environment with a `SimpleHybridWrapper`, it it has an action space of type spaces.Tuple.
        For PPO. Ignores kwargs.

    :param env: The environment to be wrapped.

    :return: The wrapped environment.
    """
    if isinstance(env.action_space, spaces.Tuple):
        return wrap_simple_hybrid(env)
    else:
        return wrap_no_wrap(env)


def wrap_no_wrap(env: "GymEnv", **kwargs) -> "GymEnv":
    """
    Dummy wrapper that returns the given environment.. Ignores kwargs.

    :param env: The environment to be wrapped.

    :return: The orginal environment.
    """
    return env


def wrap_sequence(env: "GymEnv", **kwargs) -> "SequenceWrapper":
    """
    Wraps a environment with a `SequenceWrapper`. Needs a sequence kwarg.

    :param env The environment to be wrapped.
    :param sequence: The sequence for the SequenceWrapper.
    :param stage_controller: The (optional) StageController for the SequenceWrapper.

    :return: The wrapped environment.
    """
    assert "sequence" in kwargs, "Sequence must be provided to wrap_sequence."
    sequence = kwargs["sequence"]
    if "stage_controller" in kwargs:
        stage_controller = kwargs["stage_controller"]
    assert isinstance(sequence, list), f"Provided sequence has to be of type list. Found {type(sequence)}"
    return SequenceWrapper(env, sequence, stage_controller)
