from hmlf.spaces import SimpleHybrid, Box, Discrete, Space
from typing import Any, Dict, List, Optional, Type
from torch import nn

import torch as th
import numpy as np
import copy


from hmlf.common.policies import BasePolicy, register_policy
from hmlf.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN
from hmlf.common.type_aliases import Schedule

from hmlf.td3.policies import Actor
from hmlf.dqn.policies import QNetwork


def build_state_parameter_space(observation_space: Box, action_space: SimpleHybrid) -> Box:
    lows = np.hstack([observation_space.low, action_space.continuous_low])
    highs = np.hstack([observation_space.high, action_space.continuous_high])

    return Box(lows, highs)

class PDQNPolicy(BasePolicy):
    """
    Policy class with Q-Value Net and target net for P-DQN.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
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
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):

        assert isinstance(action_space, SimpleHybrid)

        self.action_space_parameter = Box(action_space.continuous_low, action_space.continuous_high)
        self.observation_space_q = build_state_parameter_space(observation_space, action_space)
        self.action_space_q = copy.copy(action_space[0])

        super(PDQNPolicy, self).__init__(
            observation_space,
            self.action_space_q,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        if net_arch is None:
            if features_extractor_class == FlattenExtractor:
                net_arch = [64, 64]
            else:
                net_arch = []

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.normalize_images = normalize_images

        self.net_args_q = {
            "observation_space": self.observation_space_q,
            "action_space": self.action_space_q,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        self.net_args_parameter = {
            "observation_space": self.observation_space,
            "action_space": self.action_space_parameter,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        self.q_net, self.q_net_target, self.parameter_net = None, None, None
        
        self._build(lr_schedule_q, lr_schedule_parameter)

    def _build(self, lr_schedule_q: Schedule, lr_schedule_parameter: Schedule) -> None:
        """
        Create the network and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.q_net = self._make_q_net()
        self.q_net_target = self._make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())

        self.parameter_net = self._make_parameter_net()

        #TODO: Separater arguments for parameter net?
        # Setup optimizer with initial learning rate
        print(self.optimizer_kwargs)
        self.optimizer_q_net = self.optimizer_class(self.q_net.parameters(), lr=lr_schedule_q(1), **self.optimizer_kwargs)
        self.optimizer_parameter_net = self.optimizer_class(self.parameter_net.parameters(), lr=lr_schedule_parameter(1), **self.optimizer_kwargs)

    def _make_q_net(self) -> QNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args_q, self.observation_space_q, features_extractor=None)
        return QNetwork(**net_args).to(self.device)

    def _make_parameter_net(self) -> Actor:
        net_args = self._update_features_extractor(self.net_args_parameter, self.observation_space, features_extractor=None)
        #TODO implement separate arguments
        return Actor(**net_args).to(self.device)

    def forward_parameters(self, obs: th.Tensor, deterministic: bool=True) -> th.Tensor:
        parameters = self.parameter_net(obs)
        return parameters

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        #Calculates q_values
        parameters = self.forward_parameters(obs)
        obs_q = th.cat([obs, parameters], dim=1) # appends parameters to observation
        q_values = self.q_net(obs_q)
        return q_values

    def forward_target(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        parameters = self.parameter_net(obs)
        obs_q = th.cat([obs, parameters], dim=1) # appends parameters to observation
        q_values = self.q_net_target(obs_q)
        return q_values

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        # Returns actions (index)
        q_values = self.forward(obs)
        return q_values.argmax(dim=1).reshape(-1)

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(
            dict(
                net_arch=self.net_args["net_arch"],
                activation_fn=self.net_args["activation_fn"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def _update_features_extractor(
        self, 
        net_kwargs: Dict[str, Any],
        observation_space: Space,
        features_extractor: Optional[BaseFeaturesExtractor] = None
    ) -> Dict[str, Any]:
        """
        Update the network keyword arguments and create a new features extractor object if needed.
        If a ``features_extractor`` object is passed, then it will be shared.

        :param net_kwargs: the base network keyword arguments, without the ones
            related to features extractor
        :param features_extractor: a features extractor object.
            If None, a new object will be created.
        :return: The updated keyword arguments
        """
        net_kwargs = net_kwargs.copy()
        if features_extractor is None:
            # The features extractor is not shared, create a new one
            features_extractor = self.make_features_extractor(observation_space)
        net_kwargs.update(dict(features_extractor=features_extractor, features_dim=features_extractor.features_dim))
        return net_kwargs

    def make_features_extractor(self, observation_space: Space) -> BaseFeaturesExtractor:
        """ Helper method to create a features extractor."""
        return self.features_extractor_class(observation_space, **self.features_extractor_kwargs)



MlpPolicy = PDQNPolicy


class CnnPolicy(PDQNPolicy):
    """
    Policy class for DQN when using images as input.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
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
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
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
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

register_policy("MlpPolicy", MlpPolicy)
register_policy("CnnPolicy", CnnPolicy)
