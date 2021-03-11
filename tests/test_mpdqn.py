from typing import Type

import numpy as np
import pytest

from hmlf.algorithms import MPDQN, PDQN
from hmlf.algorithms.mpdqn import MlpPolicy as MPDQNMlpPolicy
from hmlf.algorithms.pdqn import MlpPolicy as PDQNMlpPolicy
from hmlf.common.evaluation import evaluate_policy
from hmlf.common.utils import get_linear_fn
from hmlf.environments import DummyHybrid
from hmlf.environments.vec_env.dummy_vec_env import DummyVecEnv
from hmlf.spaces import Box, OneHotHybrid
from hmlf.spaces.simple_hybrid import SimpleHybrid


@pytest.fixture
def simple_env():
    return DummyHybrid([2, 7, 3, 10])


@pytest.mark.parametrize(
    "model,policy_class",
    [
        (MPDQN, MPDQNMlpPolicy),
        (PDQN, PDQNMlpPolicy),
    ],
)
def test_init(model: PDQN, policy_class: PDQNMlpPolicy, simple_env: DummyHybrid):
    alg = model(policy_class, simple_env)

    assert alg.action_space == simple_env.action_space
    assert alg.observation_space == simple_env.observation_space
    assert type(alg.policy) is policy_class
    assert alg.q_net is not None
    assert alg.parameter_net is not None
    assert alg.q_net_target is not None
    assert alg.policy.optimizer_q_net is not None
    assert alg.policy.optimizer_parameter_net is not None


@pytest.mark.parametrize(
    "model,policy_class",
    [
        (MPDQN, MPDQNMlpPolicy),
        (PDQN, PDQNMlpPolicy),
    ],
)
def test_predict_format(model: PDQN, policy_class: PDQNMlpPolicy, simple_env: DummyHybrid):
    alg = model(policy_class, simple_env)
    action_space = simple_env.action_space
    n_actions = 5

    prediction = alg.predict([simple_env.observation_space.sample() for _ in range(n_actions)])

    assert type(prediction) is tuple
    obs, state = prediction
    assert state is None
    assert type(obs) is list
    assert len(obs) == n_actions

    for single_obs in obs:
        single_obs_len_expected = action_space.get_n_discrete_spaces() + action_space.get_n_continuous_spaces()
        assert len(single_obs) == single_obs_len_expected
        assert type(single_obs[0]) is np.int64
        continuous_parameter_lengths = action_space._get_dimensions_of_continuous_spaces()
        for i, parameter in enumerate(single_obs[1:]):
            assert len(parameter) == continuous_parameter_lengths[i]


def test_action_spaces(simple_env: SimpleHybrid):
    schedule = get_linear_fn(1, 0.1, 0.2)
    observation_space = Box(low=-1, high=143, shape=(3,))
    action_space_false = OneHotHybrid(
        [
            Box(low=-1, high=143, shape=(1,)),
            Box(low=1, high=1.2, shape=(2,)),
        ]
    )

    with pytest.raises(AssertionError):
        PDQNMlpPolicy(observation_space, action_space_false, schedule, schedule)
    with pytest.raises(AssertionError):
        MPDQNMlpPolicy(observation_space, action_space_false, schedule, schedule)


@pytest.mark.parametrize(
    "model,policy_class",
    [
        (MPDQN, MPDQNMlpPolicy),
        (PDQN, PDQNMlpPolicy),
    ],
)
def test_is_running(model: Type[PDQN], policy_class: Type[PDQNMlpPolicy], simple_env: DummyHybrid):
    alg = model(policy_class, simple_env, learning_starts=0)

    alg.learn(10)


@pytest.mark.slow
@pytest.mark.parametrize(
    "model,policy_class",
    [
        (MPDQN, MPDQNMlpPolicy),
        (PDQN, PDQNMlpPolicy),
    ],
)
def test_is_learning(model: Type[PDQN], policy_class: Type[PDQNMlpPolicy]):
    env = DummyHybrid([2])
    alg = model(policy_class, env, learning_rate_q=1e-3, learning_rate_parameter=1e-3, learning_starts=1000)
    env_ = DummyVecEnv([lambda: env])
    alg.learn(10000)

    evaluate_policy(alg, env_, n_eval_episodes=20, reward_threshold=-1, warn=False)
