import warnings
from typing import TYPE_CHECKING, Dict, Type, Union

if TYPE_CHECKING:
    from hmlf.common.base_class import BaseAlgorithm, BasePolicy

from hmlf.common.utils import convert_algorithm_to_string

_policy_registry: Dict[str, Dict] = {}


def register_policy(algorithm: Union[str, Type["BaseAlgorithm"]], policy_name: str, policy: Type["BasePolicy"]) -> None:
    """
    Inserts a policy into the registry.

    :param algorithm: The algorithm to which the policy belongs. Algorithms will be converted to strings.
    :param policy_name: The name of the policy - e.g. 'MlpPolicy'.
    :param policy: [description]
    """
    algorithm_name = convert_algorithm_to_string(algorithm)
    _insert_into_registry(algorithm_name, policy_name, policy)


def get_policy_from_registry(algorithm: Union[str, Type["BaseAlgorithm"]], policy_name: str) -> Type["BasePolicy"]:
    """
    Returns the policy for a given algorithm (e.g. `DDPG`) and policy name.

    :param: The algorithm to which the policy belongs. Algorithms will be converted to strings.
    :param policy_name: The name of the policy - e.g. 'MlpPolicy'.

    :return: The correct policy class.
    """
    algorithm_name = convert_algorithm_to_string(algorithm)
    return _get_from_registry(algorithm_name, policy_name)


def _insert_into_registry(algorithm_name: str, policy_name: str, policy: Type["BasePolicy"]) -> None:
    global _policy_registry
    if algorithm_name not in _policy_registry:
        _policy_registry[algorithm_name] = {}
    elif (algorithm_name in _policy_registry) and (policy_name in _policy_registry[algorithm_name]):
        warnings.warn(f"Overriding policy {policy} of name {policy_name} for algorith {algorithm_name}.")

    _policy_registry[algorithm_name][policy_name] = policy


def _get_from_registry(algorithm_name: str, policy_name: str) -> Type["BasePolicy"]:
    global _policy_registry
    if (algorithm_name in _policy_registry) and (policy_name in _policy_registry[algorithm_name]):
        return _policy_registry[algorithm_name][policy_name]
    else:
        raise ValueError(f"Algorithm {algorithm_name} not found in policy_registry. Found {_policy_registry.keys()}.")
