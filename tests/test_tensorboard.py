import os

import pytest

from hmlf.algorithms import A2C, PPO, SAC, TD3
from hmlf.algorithms.a2c import MlpPolicy as MlpPolicyA2C
from hmlf.algorithms.ppo import MlpPolicy as MlpPolicyPPO
from hmlf.algorithms.sac import MlpPolicy as MlpPolicySAC
from hmlf.algorithms.td3 import MlpPolicy as MlpPolicyTD3

MODEL_LIST = [
    ("a2c", A2C, MlpPolicyA2C, "CartPole-v1"),
    ("ppo", PPO, MlpPolicyPPO, "CartPole-v1"),
    ("sac", SAC, MlpPolicySAC, "Pendulum-v0"),
    ("td3", TD3, MlpPolicyTD3, "Pendulum-v0"),
]
N_STEPS = 50


@pytest.mark.slow
@pytest.mark.parametrize("model_name,model_class,policy_class,env_name", MODEL_LIST)
def test_tensorboard(tmp_path, model_name, model_class, policy_class, env_name):
    # Skip if no tensorboard installed
    pytest.importorskip("tensorboard")

    logname = model_name.upper()
    model = model_class(policy_class, env_name, verbose=1, tensorboard_log=tmp_path)
    model.learn(N_STEPS)
    model.learn(N_STEPS, reset_num_timesteps=False)

    assert os.path.isdir(tmp_path / str(logname + "_1"))
    assert not os.path.isdir(tmp_path / str(logname + "_2"))

    logname = "tb_multiple_runs_" + model_name
    model.learn(N_STEPS, tb_log_name=logname)
    model.learn(N_STEPS, tb_log_name=logname)

    assert os.path.isdir(tmp_path / str(logname + "_1"))
    # Check that the log dir name increments correctly
    assert os.path.isdir(tmp_path / str(logname + "_2"))
