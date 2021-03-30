from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch as th
from torch import nn

from hmlf.algorithms.pdqn import MlpPolicy as PDQNPolicy
from hmlf.common.policy_register import register_policy
from hmlf.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN
from hmlf.common.type_aliases import Schedule
from hmlf.spaces import SimpleHybrid, Space


class MPDQNPolicy(PDQNPolicy):
    """
    Policy class with Q-Value Net and target net for MP-DQN.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule_q: Learning rate schedule for Q-Network (could be constant)
    :param lr_schedule_parameter: Learning rate schedule for parameter network (could be constant)
    :param net_arch_q: The specification of the Q-Network.
    :param net_arch_parameter: The specification of the parameter network.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: Space,
        action_space: SimpleHybrid,
        lr_schedule_q: Schedule,
        lr_schedule_parameter: Schedule,
        net_arch_q: Optional[List[int]] = None,
        net_arch_parameter: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):

        super(MPDQNPolicy, self).__init__(
            observation_space,  # :Space,
            action_space,  # :SimpleHybrid,
            lr_schedule_q,  # :Schedule,
            lr_schedule_parameter,  # :Schedule,
            net_arch_q,  # :Optional[List[int]] = None,
            net_arch_parameter,  # :Optional[List[int]] = None,
            activation_fn,  # :Type[nn.Module] = nn.ReLU,
            features_extractor_class,  # :Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs,  # :Optional[Dict[str, Any]] = None,
            normalize_images,  # :bool = True,
            optimizer_class,  # :Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs,  # :Optional[Dict[str, Any]] = None,
        )

        # For formatting inside forward Q
        self.discrete_action_size = self.action_space_q.n
        self.state_size = observation_space.shape[0]
        self.offsets = np.cumsum(action_space._get_dimensions_of_continuous_spaces())
        self.offsets = np.insert(self.offsets, 0, 0)

    def _format_q_observation(self, obs: th.Tensor, action_parameters: th.Tensor, batch_size: int) -> th.Tensor:
        # Sets up multi pass structure (see https://arxiv.org/pdf/1905.04388.pdf)
        # Shape is as in P-DQN, but all parameters are set to zero
        observations = th.cat((obs, th.zeros_like(action_parameters)), dim=1)
        # Repeat for each action
        observations = observations.repeat(self.discrete_action_size, 1)

        for i in range(self.discrete_action_size):
            row_from = i * batch_size  # Beginning of current batch
            row_to = (i + 1) * batch_size  # End of current batch
            col_from = self.state_size + self.offsets[i]  # Beginning of current parameter slice
            col_to = self.state_size + self.offsets[i + 1]  # End of current parameter slice
            observations[row_from:row_to, col_from:col_to] = action_parameters[:, self.offsets[i] : self.offsets[i + 1]]

        return observations

    def _format_q_output(self, q_values: th.Tensor, batch_size: int) -> th.Tensor:
        # TODO Documentation
        Q = []
        for i in range(self.discrete_action_size):
            Qi = q_values[i * batch_size : (i + 1) * batch_size, i]
            if len(Qi.shape) == 1:
                Qi = Qi.unsqueeze(1)
            Q.append(Qi)
        Q = th.cat(Q, dim=1)
        return Q

    def forward_q(self, obs: th.Tensor, action_parameters: th.Tensor, deterministic: bool = True) -> th.Tensor:
        batch_size = action_parameters.shape[0]
        observations = self._format_q_observation(obs, action_parameters, batch_size)

        q_values = self.q_net(observations)
        Q = self._format_q_output(q_values, batch_size)
        return Q

    def _forward_q_target(self, obs: th.Tensor, action_parameters: th.Tensor, deterministic: bool = True) -> th.Tensor:
        batch_size = action_parameters.shape[0]
        observations = self._format_q_observation(obs, action_parameters, batch_size)

        q_values = self.q_net_target(observations)
        Q = self._format_q_output(q_values, batch_size)
        return Q


MlpPolicy = MPDQNPolicy


class CnnPolicy(MPDQNPolicy):
    """
    Policy class for DQN when using images as input.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch_q: The specification of the Q-Network.
    :param net_arch_parameter: The specification of the parameter network.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: Space,
        action_space: SimpleHybrid,
        lr_schedule_q: Schedule,
        lr_schedule_parameter: Schedule,
        net_arch_q: Optional[List[int]] = None,
        net_arch_parameter: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(CnnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule_q,
            lr_schedule_parameter,
            net_arch_q,
            net_arch_parameter,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )


register_policy("MPDQN", "MlpPolicy", MlpPolicy)
register_policy("MPDQN", "CnnPolicy", CnnPolicy)
