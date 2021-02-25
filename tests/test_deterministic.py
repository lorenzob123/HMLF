import pytest

from hmlf import A2C, DQN, PPO, SAC, TD3
from hmlf.a2c import MlpPolicy as MlpPolicyA2C
from hmlf.common.noise import NormalActionNoise
from hmlf.dqn import MlpPolicy as MlpPolicyDQN
from hmlf.ppo import MlpPolicy as MlpPolicyPPO
from hmlf.sac import MlpPolicy as MlpPolicySAC
from hmlf.td3 import MlpPolicy as MlpPolicyTD3

N_STEPS_TRAINING = 3000
SEED = 0


@pytest.mark.parametrize(
    "model_class,policy_class",
    [
        (A2C, MlpPolicyA2C),
        (PPO, MlpPolicyPPO),
        (SAC, MlpPolicySAC),
        (TD3, MlpPolicyTD3),
        (DQN, MlpPolicyDQN),
    ],
)
def test_deterministic_training_common(model_class, policy_class):
    results = [[], []]
    rewards = [[], []]
    # Smaller network
    kwargs = {"policy_kwargs": dict(net_arch=[64])}
    if model_class in [TD3, SAC]:
        env_id = "Pendulum-v0"
        kwargs.update({"action_noise": NormalActionNoise(0.0, 0.1), "learning_starts": 100})
    else:
        env_id = "CartPole-v1"
        if model_class == DQN:
            kwargs.update({"learning_starts": 100})

    for i in range(2):
        model = model_class(policy_class, env_id, seed=SEED, **kwargs)
        model.learn(N_STEPS_TRAINING)
        env = model.get_env()
        obs = env.reset()
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, _, _ = env.step(action)
            results[i].append(action)
            rewards[i].append(reward)
    assert sum(results[0]) == sum(results[1]), results
    assert sum(rewards[0]) == sum(rewards[1]), rewards
