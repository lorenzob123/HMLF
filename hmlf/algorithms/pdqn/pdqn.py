from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from torch.nn import functional as F

from hmlf import spaces
from hmlf.algorithms.pdqn.policies import PDQNPolicy
from hmlf.common import logger
from hmlf.common.buffers import ReplayBuffer
from hmlf.common.noise import ActionNoise
from hmlf.common.off_policy_algorithm import OffPolicyAlgorithm
from hmlf.common.type_aliases import GymEnv, MaybeCallback, Schedule
from hmlf.common.utils import get_linear_fn, get_schedule_fn, is_vectorized_observation, polyak_update, update_learning_rate
from hmlf.environments.make_environment import register_algorithm_for_make_environment, wrap_simple_hybrid


class PDQN(OffPolicyAlgorithm):
    """
    Deep Parametrized Q-Network (P-DQN)

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
    param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
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
        policy: Type[PDQNPolicy],
        env: Union[GymEnv, str],
        learning_rate_q: Union[float, Schedule] = 1e-4,
        learning_rate_parameter: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1000000,
        learning_starts: int = 50000,
        batch_size: Optional[int] = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
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

        super(PDQN, self).__init__(
            policy,
            env,
            1,  # learning_rate. We set it up ourselves, because we have two networks.
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Tuple,),
        )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None

        # Set up learning_rates
        self.learning_rate = None
        self.learning_rate_q = learning_rate_q
        self.learning_rate_parameter = learning_rate_parameter
        self.lr_schedule_q = None  # type: Optional[Schedule]
        self.lr_schedule_parameter = None  # type: Optional[Schedule]

        self.q_net, self.q_net_target, self.parameter_net = None, None, None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        # Need to change policy initialization to implement two different learning_rates
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self.replay_buffer = ReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            optimize_memory_usage=self.optimize_memory_usage,
        )
        # PDQNPolicy or MP-DQNPolicy
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule_q,
            self.lr_schedule_parameter,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)
        # Additional P-DQN _setup_model
        self._create_aliases()
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps, self.exploration_final_eps, self.exploration_fraction
        )
        self._convert_train_freq()

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target
        self.parameter_net = self.policy.parameter_net

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        if self.num_timesteps % self.target_update_interval == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        logger.record("rollout/exploration rate", self.exploration_rate)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer_q_net, self.policy.optimizer_parameter_net)

        loss_q = []
        loss_parameter = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.policy.forward_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates
            current_q_values = self.policy.forward_q(replay_data.observations, replay_data.actions[:, 1:])
            # Retrieve the q-values for the actions from the replay buffer
            # Need to do actions[:, 0].reshape(-1, 1) to get the discrete and not include parameters.
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions[:, 0].reshape(-1, 1).long())

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)

            # Optimize the policy
            self.policy.optimizer_q_net.zero_grad()
            loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.q_net.parameters(), self.max_grad_norm)
            self.policy.optimizer_q_net.step()

            q_values = self.policy.forward(replay_data.observations)
            loss2 = -q_values.sum()
            self.policy.optimizer_parameter_net.zero_grad()
            loss2.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameter_net.parameters(), self.max_grad_norm)
            self.policy.optimizer_parameter_net.step()

            loss_q.append(loss.item())
            loss_parameter.append(loss2.item())

        # Increase update counter
        self._n_updates += gradient_steps

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/loss_q", np.mean(loss_q))
        logger.record("train/loss_parameter", np.mean(loss_parameter))

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if is_vectorized_observation(observation, self.observation_space):
                n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array([self.action_space.sample()])
        else:
            action, state = self.policy.predict(observation, state, mask, deterministic)
            obs_tensor = th.Tensor(observation).to(self.device)
            parameters = self.policy.forward_parameters(obs_tensor).detach().cpu().numpy()
            action = self.action_space.build_action(action, parameters)
        return action, state

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "PDQN",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(PDQN, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _excluded_save_params(self) -> List[str]:
        return super(PDQN, self)._excluded_save_params() + ["q_net", "q_net_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer_q_net", "policy.optimizer_parameter_net"]

        return state_dicts, []

    def _sample_action(
        self, learning_starts: int, action_noise: Optional[ActionNoise] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample()])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = self.predict(self._last_obs)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = np.hstack((np.array([unscaled_action[0][0]]), *unscaled_action[0][1:]))
            action = unscaled_action
        return action, buffer_action

    def _setup_lr_schedule(self) -> None:
        """Transform to callable if needed."""
        # Override from base class, to implement two learning_rates
        self.lr_schedule_q = get_schedule_fn(self.learning_rate_q)
        self.lr_schedule_parameter = get_schedule_fn(self.learning_rate_parameter)

    def _update_learning_rate(self, optimizer_q: th.optim.Optimizer, optimizer_parameter: th.optim.Optimizer) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).

        :param optimizers:
            An optimizer or a list of optimizers.
        """
        # Log the current learning rate
        logger.record("train/learning_rate_q", self.lr_schedule_q(self._current_progress_remaining))
        logger.record("train/learning_rate_parameter", self.lr_schedule_parameter(self._current_progress_remaining))

        update_learning_rate(optimizer_q, self.lr_schedule_q(self._current_progress_remaining))
        update_learning_rate(optimizer_parameter, self.lr_schedule_parameter(self._current_progress_remaining))


register_algorithm_for_make_environment(PDQN, wrap_simple_hybrid)
