import io
import os
import pathlib
import warnings
from collections import OrderedDict
from copy import deepcopy

import gym
import numpy as np
import pytest
import torch as th

from hmlf.algorithms import A2C, DDPG, DQN, PPO, SAC, TD3
from hmlf.algorithms.a2c import CnnPolicy as CnnPolicyA2C
from hmlf.algorithms.a2c import MlpPolicy as MlpPolicyA2C
from hmlf.algorithms.ddpg import CnnPolicy as CnnPolicyDDPG
from hmlf.algorithms.ddpg import MlpPolicy as MlpPolicyDDPG
from hmlf.algorithms.dqn import CnnPolicy as CnnPolicyDQN
from hmlf.algorithms.dqn import MlpPolicy as MlpPolicyDQN
from hmlf.algorithms.ppo import CnnPolicy as CnnPolicyPPO
from hmlf.algorithms.ppo import MlpPolicy as MlpPolicyPPO
from hmlf.algorithms.sac import CnnPolicy as CnnPolicySAC
from hmlf.algorithms.sac import MlpPolicy as MlpPolicySAC
from hmlf.algorithms.td3 import CnnPolicy as CnnPolicyTD3
from hmlf.algorithms.td3 import MlpPolicy as MlpPolicyTD3
from hmlf.common.base_class import BaseAlgorithm
from hmlf.common.save_util import load_from_pkl, open_path, save_to_pkl
from hmlf.common.utils import get_device
from hmlf.environments.identity_env import FakeImageEnv, IdentityEnv, IdentityEnvBox
from hmlf.environments.vec_env import DummyVecEnv

MODEL_LIST = [
    (A2C, MlpPolicyA2C),
    (PPO, MlpPolicyPPO),
    (SAC, MlpPolicySAC),
    (TD3, MlpPolicyTD3),
    (DQN, MlpPolicyDQN),
    (DDPG, MlpPolicyDDPG),
]

MODEL_LIST_CNN = [
    (A2C, CnnPolicyA2C),
    (PPO, CnnPolicyPPO),
    (SAC, CnnPolicySAC),
    (TD3, CnnPolicyTD3),
    (DQN, CnnPolicyDQN),
    (DDPG, CnnPolicyDDPG),
]

N_STEPS_SMALL = 120


def select_env(model_class: BaseAlgorithm) -> gym.Env:
    """
    Selects an environment with the correct action space as DQN only supports discrete action space
    """
    if model_class == DQN:
        return IdentityEnv(10)
    else:
        return IdentityEnvBox(10)


@pytest.mark.parametrize("model_class,policy_class", MODEL_LIST)
def test_save_load(tmp_path, model_class, policy_class):
    """
    Test if 'save' and 'load' saves and loads model correctly
    and if 'get_parameters' and 'set_parameters' and work correctly.

    ''warning does not test function of optimizer parameter load

    :param model_class: (BaseAlgorithm) A RL model
    """

    env = DummyVecEnv([lambda: select_env(model_class)])

    # create model
    model = model_class(policy_class, env, policy_kwargs=dict(net_arch=[16]), verbose=1)
    model.learn(total_timesteps=N_STEPS_SMALL)

    env.reset()
    observations = np.concatenate([env.step([env.action_space.sample()])[0] for _ in range(10)], axis=0)

    # Get parameters of different objects
    # deepcopy to avoid referencing to tensors we are about to modify
    original_params = deepcopy(model.get_parameters())

    # Test different error cases of set_parameters.
    # Test that invalid object names throw errors
    invalid_object_params = deepcopy(original_params)
    invalid_object_params["I_should_not_be_a_valid_object"] = "and_I_am_an_invalid_tensor"
    with pytest.raises(ValueError):
        model.set_parameters(invalid_object_params, exact_match=True)
    with pytest.raises(ValueError):
        model.set_parameters(invalid_object_params, exact_match=False)

    # Test that exact_match catches when something was missed.
    missing_object_params = dict((k, v) for k, v in list(original_params.items())[:-1])
    with pytest.raises(ValueError):
        model.set_parameters(missing_object_params, exact_match=True)

    # Test that exact_match catches when something inside state-dict
    # is missing but we have exact_match.
    missing_state_dict_tensor_params = {}
    for object_name in original_params:
        object_params = {}
        missing_state_dict_tensor_params[object_name] = object_params
        # Skip last item in state-dict
        for k, v in list(original_params[object_name].items())[:-1]:
            object_params[k] = v
    with pytest.raises(RuntimeError):
        # PyTorch load_state_dict throws RuntimeError if strict but
        # invalid state-dict.
        model.set_parameters(missing_state_dict_tensor_params, exact_match=True)

    # Test that parameters do indeed change.
    random_params = {}
    for object_name, params in original_params.items():
        # Do not randomize optimizer parameters (custom layout)
        if "optim" in object_name:
            random_params[object_name] = params
        else:
            # Again, skip the last item in state-dict
            random_params[object_name] = OrderedDict(
                (param_name, th.rand_like(param)) for param_name, param in list(params.items())[:-1]
            )

    # Update model parameters with the new random values
    model.set_parameters(random_params, exact_match=False)

    new_params = model.get_parameters()
    # Check that all params except the final item in each state-dict are different.
    for object_name in original_params:
        # Skip optimizers (no valid comparison with just th.allclose)
        if "optim" in object_name:
            continue
        # state-dicts use ordered dictionaries, so key order
        # is guaranteed.
        last_key = list(original_params[object_name].keys())[-1]
        for k in original_params[object_name]:
            if k == last_key:
                # Should be same as before
                assert th.allclose(
                    original_params[object_name][k], new_params[object_name][k]
                ), "Parameter changed despite not included in the loaded parameters."
            else:
                # Should be different
                assert not th.allclose(
                    original_params[object_name][k], new_params[object_name][k]
                ), "Parameters did not change as expected."

    params = new_params

    # get selected actions
    selected_actions, _ = model.predict(observations, deterministic=True)

    # Check
    model.save(tmp_path / "test_save.zip")
    del model

    # Check if the model loads as expected for every possible choice of device:
    for device in ["auto", "cpu", "cuda"]:
        model = model_class.load(str(tmp_path / "test_save.zip"), env=env, device=device)

        # check if the model was loaded to the correct device
        assert model.device.type == get_device(device).type
        assert model.policy.device.type == get_device(device).type

        # check if params are still the same after load
        new_params = model.get_parameters()

        # Check that all params are the same as before save load procedure now
        for object_name in new_params:
            # Skip optimizers (no valid comparison with just th.allclose)
            if "optim" in object_name:
                continue
            for key in params[object_name]:
                assert new_params[object_name][key].device.type == get_device(device).type
                assert th.allclose(
                    params[object_name][key].to("cpu"), new_params[object_name][key].to("cpu")
                ), "Model parameters not the same after save and load."

        # check if model still selects the same actions
        new_selected_actions, _ = model.predict(observations, deterministic=True)
        assert np.allclose(selected_actions, new_selected_actions, 1e-4)

        # check if learn still works
        model.learn(total_timesteps=N_STEPS_SMALL)

        del model

    # clear file from os
    os.remove(tmp_path / "test_save.zip")


@pytest.mark.parametrize("model_class,policy_class", MODEL_LIST)
def test_set_env(model_class, policy_class):
    """
    Test if set_env function does work correct
    :param model_class: (BaseAlgorithm) A RL model
    """

    # use discrete for DQN
    env = DummyVecEnv([lambda: select_env(model_class)])
    env2 = DummyVecEnv([lambda: select_env(model_class)])
    env3 = select_env(model_class)

    kwargs = {}
    if model_class in {DQN, DDPG, SAC, TD3}:
        kwargs = dict(learning_starts=0)
    elif model_class in {A2C, PPO}:
        kwargs = dict(n_steps=100)

    # create model
    model = model_class(policy_class, env, policy_kwargs=dict(net_arch=[16]), **kwargs)
    # learn
    model.learn(total_timesteps=N_STEPS_SMALL)

    # change env
    model.set_env(env2)
    # learn again
    model.learn(total_timesteps=N_STEPS_SMALL)

    # change env test wrapping
    model.set_env(env3)
    # learn again
    model.learn(total_timesteps=N_STEPS_SMALL)


@pytest.mark.parametrize("model_class,policy_class", MODEL_LIST)
def test_exclude_include_saved_params(tmp_path, model_class, policy_class):
    """
    Test if exclude and include parameters of save() work

    :param model_class: (BaseAlgorithm) A RL model
    """
    env = DummyVecEnv([lambda: select_env(model_class)])

    # create model, set verbose as 2, which is not standard
    model = model_class(policy_class, env, policy_kwargs=dict(net_arch=[16]), verbose=2)

    # Check if exclude works
    model.save(tmp_path / "test_save", exclude=["verbose"])
    del model
    model = model_class.load(str(tmp_path / "test_save.zip"))
    # check if verbose was not saved
    assert model.verbose != 2

    # set verbose as something different then standard settings
    model.verbose = 2
    # Check if include works
    model.save(tmp_path / "test_save", exclude=["verbose"], include=["verbose"])
    del model
    model = model_class.load(str(tmp_path / "test_save.zip"))
    assert model.verbose == 2

    # clear file from os
    os.remove(tmp_path / "test_save.zip")


@pytest.mark.parametrize(
    "model_class,policy_class",
    [
        (A2C, CnnPolicyA2C),
        (TD3, CnnPolicyTD3),
    ],
)
def test_save_load_env_cnn(tmp_path, model_class, policy_class):
    """
    Test loading with an env that requires a ``CnnPolicy``.
    This is to test wrapping and observation space check.
    We test one on-policy and one off-policy
    algorithm as the rest share the loading part.
    """
    env = FakeImageEnv(screen_height=40, screen_width=40, n_channels=2, discrete=False)
    kwargs = dict(policy_kwargs=dict(net_arch=[32]))
    if model_class == TD3:
        kwargs.update(dict(buffer_size=100, learning_starts=50))

    model = model_class(policy_class, env, **kwargs).learn(100)
    model.save(tmp_path / "test_save")
    # Test loading with env and continuing training
    model = model_class.load(str(tmp_path / "test_save.zip"), env=env).learn(100)
    # clear file from os
    os.remove(tmp_path / "test_save.zip")


@pytest.mark.parametrize(
    "model_class,policy_class",
    [
        (SAC, MlpPolicySAC),
        (TD3, MlpPolicyTD3),
        (DQN, MlpPolicyDQN),
    ],
)
def test_save_load_replay_buffer(tmp_path, model_class, policy_class):
    path = pathlib.Path(tmp_path / "logs/replay_buffer.pkl")
    path.parent.mkdir(exist_ok=True, parents=True)  # to not raise a warning
    model = model_class(
        policy_class, select_env(model_class), buffer_size=1000, policy_kwargs=dict(net_arch=[64]), learning_starts=200
    )
    model.learn(N_STEPS_SMALL)
    old_replay_buffer = deepcopy(model.replay_buffer)
    model.save_replay_buffer(path)
    model.replay_buffer = None
    model.load_replay_buffer(path)

    assert np.allclose(old_replay_buffer.observations, model.replay_buffer.observations)
    assert np.allclose(old_replay_buffer.actions, model.replay_buffer.actions)
    assert np.allclose(old_replay_buffer.rewards, model.replay_buffer.rewards)
    assert np.allclose(old_replay_buffer.dones, model.replay_buffer.dones)

    # test extending replay buffer
    model.replay_buffer.extend(
        old_replay_buffer.observations,
        old_replay_buffer.observations,
        old_replay_buffer.actions,
        old_replay_buffer.rewards,
        old_replay_buffer.dones,
    )


@pytest.mark.parametrize(
    "model_class,policy_class",
    [
        (SAC, MlpPolicySAC),
        (TD3, MlpPolicyTD3),
        (DQN, MlpPolicyDQN),
    ],
)
@pytest.mark.parametrize("optimize_memory_usage", [False, True])
def test_warn_buffer(recwarn, model_class, policy_class, optimize_memory_usage):
    """
    When using memory efficient replay buffer,
    a warning must be emitted when calling `.learn()`
    multiple times.
    See https://github.com/DLR-RM/stable-baselines3/issues/46
    """
    # remove gym warnings
    warnings.filterwarnings(action="ignore", category=DeprecationWarning)
    warnings.filterwarnings(action="ignore", category=UserWarning, module="gym")

    model = model_class(
        policy_class,
        select_env(model_class),
        buffer_size=100,
        optimize_memory_usage=optimize_memory_usage,
        policy_kwargs=dict(net_arch=[64]),
        learning_starts=10,
    )

    model.learn(N_STEPS_SMALL)

    model.learn(N_STEPS_SMALL, reset_num_timesteps=False)

    # Check that there is no warning
    assert len(recwarn) == 0

    model.learn(N_STEPS_SMALL)

    if optimize_memory_usage:
        assert len(recwarn) == 1
        warning = recwarn.pop(UserWarning)
        assert "The last trajectory in the replay buffer will be truncated" in str(warning.message)
    else:
        assert len(recwarn) == 0


@pytest.mark.parametrize("model_class,policy_class", MODEL_LIST + MODEL_LIST_CNN)
def test_save_load_policy(
    tmp_path,
    model_class,
    policy_class,
):
    """
    Test saving and loading policy only.

    :param model_class: (BaseAlgorithm) A RL model
    :param policy_str: (str) Name of the policy.
    """
    kwargs = dict(policy_kwargs=dict(net_arch=[16]))
    if "Cnn" not in str(policy_class):  # MlpPolicy
        env = select_env(model_class)
    else:
        if model_class in [SAC, TD3, DQN, DDPG]:
            # Avoid memory error when using replay buffer
            # Reduce the size of the features
            kwargs = dict(
                buffer_size=250, learning_starts=0, policy_kwargs=dict(features_extractor_kwargs=dict(features_dim=32))
            )
        env = FakeImageEnv(screen_height=40, screen_width=40, n_channels=2, discrete=model_class == DQN)

    env = DummyVecEnv([lambda: env])

    # create model
    model = model_class(policy_class, env, verbose=1, **kwargs)
    model.learn(total_timesteps=N_STEPS_SMALL)

    env.reset()
    observations = np.concatenate([env.step([env.action_space.sample()])[0] for _ in range(10)], axis=0)

    policy = model.policy
    policy_class = policy.__class__
    actor, actor_class = None, None
    if model_class in [SAC, TD3]:
        actor = policy.actor
        actor_class = actor.__class__

    # Get dictionary of current parameters
    params = deepcopy(policy.state_dict())

    # Modify all parameters to be random values
    random_params = dict((param_name, th.rand_like(param)) for param_name, param in params.items())

    # Update model parameters with the new random values
    policy.load_state_dict(random_params)

    new_params = policy.state_dict()
    # Check that all params are different now
    for k in params:
        assert not th.allclose(params[k], new_params[k]), "Parameters did not change as expected."

    params = new_params

    # get selected actions
    selected_actions, _ = policy.predict(observations, deterministic=True)
    # Should also work with the actor only
    if actor is not None:
        selected_actions_actor, _ = actor.predict(observations, deterministic=True)

    # Save and load policy
    policy.save(tmp_path / "policy.pkl")
    # Save and load actor
    if actor is not None:
        actor.save(tmp_path / "actor.pkl")

    del policy, actor

    policy = policy_class.load(tmp_path / "policy.pkl")
    if actor_class is not None:
        actor = actor_class.load(tmp_path / "actor.pkl")

    # check if params are still the same after load
    new_params = policy.state_dict()

    # Check that all params are the same as before save load procedure now
    for key in params:
        assert th.allclose(params[key], new_params[key]), "Policy parameters not the same after save and load."

    # check if model still selects the same actions
    new_selected_actions, _ = policy.predict(observations, deterministic=True)
    assert np.allclose(selected_actions, new_selected_actions, 1e-4)

    if actor_class is not None:
        new_selected_actions_actor, _ = actor.predict(observations, deterministic=True)
        assert np.allclose(selected_actions_actor, new_selected_actions_actor, 1e-4)
        assert np.allclose(selected_actions_actor, new_selected_actions, 1e-4)

    # clear file from os
    os.remove(tmp_path / "policy.pkl")
    if actor_class is not None:
        os.remove(tmp_path / "actor.pkl")


@pytest.mark.parametrize(
    "model_class,policy_class",
    [
        (DQN, MlpPolicyDQN),
        (DQN, CnnPolicyDQN),
    ],
)
def test_save_load_q_net(tmp_path, model_class, policy_class):
    """
    Test saving and loading q-network/quantile net only.

    :param model_class: (BaseAlgorithm) A RL model
    :param policy_str: (str) Name of the policy.
    """
    kwargs = dict(policy_kwargs=dict(net_arch=[16]))
    if "Cnn" not in str(policy_class):  # MlpPolicy
        env = select_env(model_class)
    else:
        if model_class in [DQN]:
            # Avoid memory error when using replay buffer
            # Reduce the size of the features
            kwargs = dict(
                buffer_size=250,
                learning_starts=100,
                policy_kwargs=dict(features_extractor_kwargs=dict(features_dim=32)),
            )
        env = FakeImageEnv(screen_height=40, screen_width=40, n_channels=2, discrete=model_class == DQN)

    env = DummyVecEnv([lambda: env])

    # create model
    model = model_class(policy_class, env, verbose=1, **kwargs)
    model.learn(total_timesteps=N_STEPS_SMALL)

    env.reset()
    observations = np.concatenate([env.step([env.action_space.sample()])[0] for _ in range(10)], axis=0)

    q_net = model.q_net
    q_net_class = q_net.__class__

    # Get dictionary of current parameters
    params = deepcopy(q_net.state_dict())

    # Modify all parameters to be random values
    random_params = dict((param_name, th.rand_like(param)) for param_name, param in params.items())

    # Update model parameters with the new random values
    q_net.load_state_dict(random_params)

    new_params = q_net.state_dict()
    # Check that all params are different now
    for k in params:
        assert not th.allclose(params[k], new_params[k]), "Parameters did not change as expected."

    params = new_params

    # get selected actions
    selected_actions, _ = q_net.predict(observations, deterministic=True)

    # Save and load q_net
    q_net.save(tmp_path / "q_net.pkl")

    del q_net

    q_net = q_net_class.load(tmp_path / "q_net.pkl")

    # check if params are still the same after load
    new_params = q_net.state_dict()

    # Check that all params are the same as before save load procedure now
    for key in params:
        assert th.allclose(params[key], new_params[key]), "Policy parameters not the same after save and load."

    # check if model still selects the same actions
    new_selected_actions, _ = q_net.predict(observations, deterministic=True)
    assert np.allclose(selected_actions, new_selected_actions, 1e-4)

    # clear file from os
    os.remove(tmp_path / "q_net.pkl")


@pytest.mark.parametrize("pathtype", [str, pathlib.Path])
def test_open_file_str_pathlib(tmp_path, pathtype):
    # check that suffix isn't added because we used open_path first
    with open_path(pathtype(f"{tmp_path}/t1"), "w") as fp1:
        save_to_pkl(fp1, "foo")
    assert fp1.closed
    with pytest.warns(None) as record:
        assert load_from_pkl(pathtype(f"{tmp_path}/t1")) == "foo"
    assert not record

    # test custom suffix
    with open_path(pathtype(f"{tmp_path}/t1.custom_ext"), "w") as fp1:
        save_to_pkl(fp1, "foo")
    assert fp1.closed
    with pytest.warns(None) as record:
        assert load_from_pkl(pathtype(f"{tmp_path}/t1.custom_ext")) == "foo"
    assert not record

    # test without suffix
    with open_path(pathtype(f"{tmp_path}/t1"), "w", suffix="pkl") as fp1:
        save_to_pkl(fp1, "foo")
    assert fp1.closed
    with pytest.warns(None) as record:
        assert load_from_pkl(pathtype(f"{tmp_path}/t1.pkl")) == "foo"
    assert not record

    # test that a warning is raised when the path doesn't exist
    with open_path(pathtype(f"{tmp_path}/t2.pkl"), "w") as fp1:
        save_to_pkl(fp1, "foo")
    assert fp1.closed
    with pytest.warns(None) as record:
        assert load_from_pkl(open_path(pathtype(f"{tmp_path}/t2"), "r", suffix="pkl")) == "foo"
    assert len(record) == 0

    with pytest.warns(None) as record:
        assert load_from_pkl(open_path(pathtype(f"{tmp_path}/t2"), "r", suffix="pkl", verbose=2)) == "foo"
    assert len(record) == 1

    fp = pathlib.Path(f"{tmp_path}/t2").open("w")
    fp.write("rubbish")
    fp.close()
    # test that a warning is only raised when verbose = 0
    with pytest.warns(None) as record:
        open_path(pathtype(f"{tmp_path}/t2"), "w", suffix="pkl", verbose=0).close()
        open_path(pathtype(f"{tmp_path}/t2"), "w", suffix="pkl", verbose=1).close()
        open_path(pathtype(f"{tmp_path}/t2"), "w", suffix="pkl", verbose=2).close()
    assert len(record) == 1


def test_open_file(tmp_path):

    # path must much the type
    with pytest.raises(TypeError):
        open_path(123, None, None, None)

    p1 = tmp_path / "test1"
    fp = p1.open("wb")

    # provided path must match the mode
    with pytest.raises(ValueError):
        open_path(fp, "r")
    with pytest.raises(ValueError):
        open_path(fp, "randomstuff")

    # test identity
    _ = open_path(fp, "w")
    assert _ is not None
    assert fp is _

    # Can't use a closed path
    with pytest.raises(ValueError):
        fp.close()
        open_path(fp, "w")

    buff = io.BytesIO()
    assert buff.writable()
    assert buff.readable() is ("w" == "w")
    _ = open_path(buff, "w")
    assert _ is buff
    with pytest.raises(ValueError):
        buff.close()
        open_path(buff, "w")
