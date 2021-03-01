from hmlf import SDDPG
from hmlf.common.callbacks import EvalCallback
from hmlf.environments import ObstacleCourse_v2, SequenceWrapper
from hmlf.sddpg import MlpPolicy

env = SequenceWrapper(ObstacleCourse_v2(), [0, 1, 0, 1, 0, 1, 0])
obs = env.reset()
env.step(env.action_space.sample())

callback = EvalCallback(
    eval_env=SequenceWrapper(ObstacleCourse_v2(), [0, 1, 0, 1, 0, 1, 0, 0, 0]), eval_freq=100, n_eval_episodes=5
)

model = SDDPG(MlpPolicy, env=env, verbose=1, learning_starts=10000)
model.learn(1e5, callback=callback)
