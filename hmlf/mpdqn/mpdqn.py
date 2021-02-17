from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from torch.nn import functional as F

from hmlf.common.type_aliases import GymEnv, Schedule
from hmlf.mpdqn.policies import MPDQNPolicy


from hmlf.pdqn import PDQN



class MPDQN(PDQN):
    """
    Deep Multi-Pass Parametrized Q-Network (MP-DQN)

    Paper: https://arxiv.org/abs/1810.06394
    Default hyperparameters are taken from the DQN-nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate_q: The learning rate for the Q-Network, it can be a function
        of the current progress remaining (from 1 to 0)
    :param learning_rate_parameter: The learning rate for the parameter network, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Set to `-1` to disable.
    :param gradient_steps: How many gradient steps to do after each rollout
        (see ``train_freq`` and ``n_episodes_rollout``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param n_episodes_rollout: Update the model every ``n_episodes_rollout`` episodes.
        Note that this cannot be used at the same time as ``train_freq``. Set to `-1` to disable.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[MPDQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate_q: Union[float, Schedule] = 1e-4,
        learning_rate_parameter: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1000000,
        learning_starts: int = 50000,
        batch_size: Optional[int] = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int = 4,
        gradient_steps: int = 1,
        n_episodes_rollout: int = -1,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(MPDQN, self).__init__(
            policy = policy, #: Union[str, Type[MPDQNPolicy]],
            policy_group = "MPDQN", # Need to add that for the P-DQN to use the correct policy
            env = env, #: Union[GymEnv, str],
            learning_rate_q = learning_rate_q, #: Union[float, Schedule] = 1e-4,
            learning_rate_parameter = learning_rate_parameter, #: Union[float, Schedule] = 1e-4,
            buffer_size = buffer_size, #: int = 1000000,
            learning_starts = learning_starts, #: int = 50000,
            batch_size = batch_size, #: Optional[int] = 32,
            tau = tau, #: float = 1.0,
            gamma = gamma, #: float = 0.99,
            train_freq = train_freq, #: int = 4,
            gradient_steps = gradient_steps, #: int = 1,
            n_episodes_rollout = n_episodes_rollout, #: int = -1,
            optimize_memory_usage = optimize_memory_usage, #: bool = False,
            target_update_interval = target_update_interval, #: int = 10000,
            exploration_fraction = exploration_fraction, #: float = 0.1,
            exploration_initial_eps = exploration_initial_eps, #: float = 1.0,
            exploration_final_eps = exploration_final_eps, #: float = 0.05,
            max_grad_norm = max_grad_norm, #: float = 10,
            tensorboard_log = tensorboard_log, #: Optional[str] = None,
            create_eval_env = create_eval_env, #: bool = False,
            policy_kwargs = policy_kwargs, #: Optional[Dict[str, Any]] = None,
            verbose = verbose, #: int = 0,
            seed = seed, #: Optional[int] = None,
            device = device, #: Union[th.device, str] = "auto",
            _init_setup_model = _init_setup_model, #: bool = True,
        )
