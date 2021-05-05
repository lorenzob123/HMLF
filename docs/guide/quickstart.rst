.. _quickstart:

===============
Getting Started
===============
The HMLF is a fork of stablebaselines3. As such we try to keep the API of the algorithms as close as possible to its original counterpart, while granting the use of hybrid algorithms.
The major change form stablebaselines3 is the necessary step requiered to wrap the environment for the different hybrid algorithms. The function ``wrap_environment`` is used to do exaclty that. 

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

