from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
from torch import nn

from hmlf.common.policies import BaseModel, BasePolicy, ContinuousCritic
from hmlf.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN, create_mlp, get_actor_critic_arch
from hmlf.common.type_aliases import Schedule
from hmlf.spaces.simple_hybrid import SimpleHybrid


class Actor(BasePolicy):
    """
    Actor network (policy) for TD3.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: SimpleHybrid,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super(Actor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.features_extractor = features_extractor
        self.normalize_images = normalize_images
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        action_dim = self.action_space._get_continous_dims()
        actor_list = []
        for action_param_dim in action_dim:
            actor_net = create_mlp(features_dim - 1, action_param_dim, net_arch, activation_fn, squash_output=True)
            actor_list.append(nn.Sequential(*actor_net))
        self.mu_list = nn.ModuleList(actor_list)

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        # TODO: implement a way to use extract_features
        # features = self.extract_features(obs)

        # TODO: this is pseudo code and it will not work because of the batched env
        actions = [th.zeros(obs.shape[0], dim).to(self.device) for dim in self.action_space._get_continous_dims()]

        for i in range(obs.shape[0]):
            discrete_i = int(obs[i, 0].item())
            param_i = self.mu_list[discrete_i](obs[i, 1:])
            actions[discrete_i][i, :] = param_i

        return th.cat(actions, dim=1)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.forward(observation, deterministic=deterministic)


class MetaCritic(BaseModel):
    def __init__(self, critic_kwargs):

        super().__init__(
            critic_kwargs["observation_space"],
            critic_kwargs["action_space"],
            features_extractor=critic_kwargs["features_extractor"],
            normalize_images=critic_kwargs["normalize_images"],
        )
        critic_list = []
        # action_dim = self.action_space._get_continous_dims()
        critic_kwargs["observation_space"] = gym.spaces.Box(
            critic_kwargs["observation_space"].low[1:], critic_kwargs["observation_space"].high[1:]
        )
        critic_kwargs["features_dim"] -= 1

        for param_space in self.action_space.spaces:
            critic_kwargs["action_space"] = param_space
            critic_list.append(ContinuousCritic(**critic_kwargs).to(self.device))
        self.critic_list = nn.ModuleList(critic_list)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor

        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        # TODO: implement a way to use extract_features
        # features = self.extract_features(obs)

        # TODO: this is pseudo code and it will not work because of the batched env
        dims_continous = self.action_space._get_continous_dims()
        indices = np.hstack((np.array([0]), np.cumsum(dims_continous)))
        q = []

        if len(obs.shape) == 1:
            obs = obs.view(1, -1)
        for i in range(obs.shape[0]):
            discrete_i = int(obs[i, 0].item())
            param_slice = slice(indices[discrete_i], indices[discrete_i + 1])
            # qvalue_input = th.cat((obs[[i], 1:], actions[[i], param_slice]), dim=1)
            q.append(self.critic_list[discrete_i](obs[[i], 1:], actions[[i], param_slice])[0])
        return (th.cat(q),)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        # with th.no_grad():
        #     features = self.extract_features(obs)
        return self.forward(obs, actions)[0]


class SDDPGPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for TD3.

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
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super(SDDPGPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )

        # Default network architecture, from the original paper
        if net_arch is None:
            if features_extractor_class == FlattenExtractor:
                net_arch = [400, 300]
            else:
                net_arch = []

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        # Create actor and target
        # the features extractor should not be shared
        self.actor = self.make_actor(features_extractor=None)
        self.actor_target = self.make_actor(features_extractor=None)
        # Initialize the target to have the same weights as the actor
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor.optimizer = self.optimizer_class(self.actor.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Critic target should not share the features extactor with critic
            # but it can share it with the actor target as actor and critic are sharing
            # the same features_extractor too
            # NOTE: as a result the effective poliak (soft-copy) coefficient for the features extractor
            # will be 2 * tau instead of tau (updated one time with the actor, a second time with the critic)
            self.critic_target = self.make_critic(features_extractor=self.actor_target.features_extractor)
        else:
            # Create new features extractor for each network
            self.critic = self.make_critic(features_extractor=None)
            self.critic_target = self.make_critic(features_extractor=None)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(self.critic.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                share_features_extractor=self.share_features_extractor,
            )
        )
        return data

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:

        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return MetaCritic(critic_kwargs).to(self.device)

    def forward(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self.actor(observation, deterministic=deterministic)


MlpPolicy = SDDPGPolicy


class CnnPolicy(SDDPGPolicy):
    """
    Policy class (with both actor and critic) for TD3.

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
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
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
            n_critics,
            share_features_extractor,
        )
