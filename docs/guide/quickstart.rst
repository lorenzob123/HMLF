.. _quickstart:

===============
Getting Started
===============

This library tries to keep the API from stablebaselines3, while letting you use hybrid algorithms and parametrized action spaces

.. code-block:: python

  from hmlf.algorithms import SDDPG
  from hmlf.environments import wrap_environment, ObstacleCourse_v2

  env = wrap_environment(SDDPG, ObstacleCourse_v2(), [0, 1, 0, 1, 0, 1, 0])

  model = SDDPG("MlpPolicy", env=env,  learning_rate=1e-1, verbose=1, learning_starts=10000)
  model.learn(1e5, callback=callback)

  obs = env.reset()
  for i in range(1000):
      action, _state = model.predict(obs, deterministic=True)
      obs, reward, done, info = env.step(action)
      env.render()
      if done:
        obs = env.reset()

