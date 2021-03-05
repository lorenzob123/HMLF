import pytest
import torch as th

from hmlf.algorithms import A2C, PPO
from hmlf.algorithms.a2c import MlpPolicy as MlpPolicyA2C
from hmlf.algorithms.ppo import MlpPolicy as MlpPolicyPPO
from hmlf.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    HybridDistribution,
    MultiCategoricalDistribution,
    SquashedDiagGaussianDistribution,
    StateDependentNoiseDistribution,
    TanhBijector,
)
from hmlf.common.utils import set_random_seed
from hmlf.spaces import SimpleHybrid
from hmlf.spaces.gym import Box,Discrete

N_ACTIONS = 2
N_FEATURES = 3
N_SAMPLES = int(5e6)


def test_bijector():
    """
    Test TanhBijector
    """
    actions = th.ones(5) * 2.0
    bijector = TanhBijector()

    squashed_actions = bijector.forward(actions)
    # Check that the boundaries are not violated
    assert th.max(th.abs(squashed_actions)) <= 1.0
    # Check the inverse method
    assert th.isclose(TanhBijector.inverse(squashed_actions), actions).all()


@pytest.mark.parametrize(
    "model_class,policy_class",
    [
        (A2C, MlpPolicyA2C),
        (PPO, MlpPolicyPPO),
    ],
)
def test_squashed_gaussian(model_class, policy_class):
    """
    Test run with squashed Gaussian (notably entropy computation)
    """
    model = model_class(policy_class, "Pendulum-v0", use_sde=True, n_steps=100, policy_kwargs=dict(squash_output=True))
    model.learn(500)

    gaussian_mean = th.rand(N_SAMPLES, N_ACTIONS)
    dist = SquashedDiagGaussianDistribution(N_ACTIONS)
    _, log_std = dist.proba_distribution_net(N_FEATURES)
    dist = dist.proba_distribution(gaussian_mean, log_std)
    actions = dist.get_actions()
    assert th.max(th.abs(actions)) <= 1.0


def test_sde_distribution():
    n_actions = 1
    deterministic_actions = th.ones(N_SAMPLES, n_actions) * 0.1
    state = th.ones(N_SAMPLES, N_FEATURES) * 0.3
    dist = StateDependentNoiseDistribution(n_actions, full_std=True, squash_output=False)

    set_random_seed(1)
    _, log_std = dist.proba_distribution_net(N_FEATURES)
    dist.sample_weights(log_std, batch_size=N_SAMPLES)

    dist = dist.proba_distribution(deterministic_actions, log_std, state)
    actions = dist.get_actions()

    assert th.allclose(actions.mean(), dist.distribution.mean.mean(), rtol=2e-3)
    assert th.allclose(actions.std(), dist.distribution.scale.mean(), rtol=2e-3)


# TODO: analytical form for squashed Gaussian?
@pytest.mark.parametrize(
    "dist",
    [
        DiagGaussianDistribution(N_ACTIONS),
        StateDependentNoiseDistribution(N_ACTIONS, squash_output=False),
    ],
)
def test_entropy(dist):
    # The entropy can be approximated by averaging the negative log likelihood
    # mean negative log likelihood == differential entropy
    set_random_seed(1)
    state = th.rand(N_SAMPLES, N_FEATURES)
    deterministic_actions = th.rand(N_SAMPLES, N_ACTIONS)
    _, log_std = dist.proba_distribution_net(N_FEATURES, log_std_init=th.log(th.tensor(0.2)))

    if isinstance(dist, DiagGaussianDistribution):
        dist = dist.proba_distribution(deterministic_actions, log_std)
    else:
        dist.sample_weights(log_std, batch_size=N_SAMPLES)
        dist = dist.proba_distribution(deterministic_actions, log_std, state)

    actions = dist.get_actions()
    entropy = dist.entropy()
    log_prob = dist.log_prob(actions)
    assert th.allclose(entropy.mean(), -log_prob.mean(), rtol=5e-3)


categorical_params = [
    (CategoricalDistribution(N_ACTIONS), N_ACTIONS),
    (MultiCategoricalDistribution([2, 3]), sum([2, 3])),
    (BernoulliDistribution(N_ACTIONS), N_ACTIONS),
]


@pytest.mark.parametrize("dist, CAT_ACTIONS", categorical_params)
def test_categorical(dist, CAT_ACTIONS):
    # The entropy can be approximated by averaging the negative log likelihood
    # mean negative log likelihood == entropy
    set_random_seed(1)
    action_logits = th.rand(N_SAMPLES, CAT_ACTIONS)
    dist = dist.proba_distribution(action_logits)
    actions = dist.get_actions()
    entropy = dist.entropy()
    log_prob = dist.log_prob(actions)
    assert th.allclose(entropy.mean(), -log_prob.mean(), rtol=5e-3)


@pytest.mark.parametrize(
    "action_space",
    [
        SimpleHybrid([
            Discrete(5),
            Box(low=-1, high=143, shape=(1,)),
            Box(low=1, high=1.2, shape=(2,)),
            Box(low=11, high=13, shape=(3,)),
            Box(low=-1, high=1, shape=(4,)),
            Box(low=-1, high=1, shape=(5,)),
        ]),

        SimpleHybrid([
            Discrete(2),
            Box(low=-1, high=143, shape=(1,)),
            Box(low=1, high=1.2, shape=(0,)),
        ]),
    ],
)
def test_hybrid(action_space):
    # The entropy can be approximated by averaging the negative log likelihood
    # mean negative log likelihood == entropy
    set_random_seed(1)
    dist = HybridDistribution(action_space)

    set_random_seed(1)
    #
    dim = action_space.continuous_dim + action_space.discrete_dim
    deterministic_actions = th.rand(N_SAMPLES, dim)
    _, log_std = dist.proba_distribution_net(N_FEATURES, log_std_init=th.log(th.tensor(0.2)))

    dist = dist.proba_distribution(deterministic_actions, log_std)

    actions = dist.get_actions()
    entropy = dist.entropy()
    log_prob = dist.log_prob(actions)
    assert th.allclose(entropy.mean(), -log_prob.mean(), rtol=5e-3)
