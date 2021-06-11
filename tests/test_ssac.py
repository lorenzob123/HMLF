import numpy as np
import pytest

from hmlf.algorithms.ssac import SSAC, MlpPolicy
from hmlf.environments import DummyHybrid, SequenceWrapper


@pytest.fixture
def simple_env():
    return DummyHybrid([2, 7, 3, 10])


@pytest.mark.parametrize(
    "sequence",
    [
        ([0, 0, 0, 0, 0]),
        ([0, 1, 3, 2]),
        ([1]),
    ],
)
def test_init(sequence, simple_env):
    env = SequenceWrapper(simple_env, sequence)
    alg = SSAC(MlpPolicy, env)

    assert alg.action_space == env.action_space
    assert alg.observation_space == env.observation_space
    assert type(alg.policy) is MlpPolicy
    assert len(alg.policy.actor.mu_list) == 4
    assert len(alg.policy.critic.critic_list) == 4


@pytest.mark.parametrize(
    "sequence",
    [
        ([0, 0, 0, 0, 0]),
        ([0, 1, 3, 2]),
    ],
)
def test_predict_format(sequence, simple_env):
    env = SequenceWrapper(simple_env, sequence)
    alg = SSAC(MlpPolicy, env)
    n_actions = len(sequence) - 1
    env.reset()
    observations = [env.step(env.action_space.sample())[0].astype(np.float32) for _ in range(n_actions)]
    out_pred = alg.predict(observations)

    assert type(out_pred) is tuple
    prediction, state = out_pred
    assert state is None
    assert type(prediction) is list
    assert len(prediction) == n_actions

    for pred, obs in zip(prediction, observations):
        choice = int(obs[0])
        for i in range(len(pred)):
            if i != choice:
                assert np.sum(np.abs(pred[i])) == 0

            else:
                assert pred[choice].shape == env.action_space.spaces[choice].shape


@pytest.mark.parametrize(
    "sequence",
    [
        ([0, 0, 0, 0, 0]),
        ([0, 1, 3, 2]),
    ],
)
def test_is_running(sequence: list, simple_env: DummyHybrid):
    env = SequenceWrapper(simple_env, sequence)
    alg = SSAC(MlpPolicy, env)

    alg.learn(10)
