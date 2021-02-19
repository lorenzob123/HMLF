from hmlf.common.callbacks import EvalCallback
from hmlf.common.monitor import Monitor
from hmlf.environments import ObstacleCourse_v2
from hmlf.mpdqn import MPDQN, MlpPolicy

env = ObstacleCourse_v2()
pdqn = MPDQN(MlpPolicy, env, learning_rate_q=0.001, learning_rate_parameter=0.001, learning_starts=1)
pdqn.learn(
    total_timesteps=1e2, callback=EvalCallback(eval_env=Monitor(ObstacleCourse_v2()), eval_freq=10000, n_eval_episodes=50)
)
pdqn.save("test.zip")
