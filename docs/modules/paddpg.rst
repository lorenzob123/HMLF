.. _paddpg:

.. automodule:: hmlf.algorithms.paddpg


PADDPG
======

.. A synchronous, deterministic variant of `Asynchronous Advantage Actor Critic (A3C) <https://arxiv.org/abs/1602.01783>`_.
.. It uses multiple workers to avoid the use of a replay buffer.




Notes
-----

-  Original paper:  https://arxiv.org/abs/1602.01783


Can I use?
----------

-  Recurrent policies: ❌
-  Multi processing: ❌
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ❌      ✔️
Box           ❌      ✔️
MultiDiscrete ❌      ✔️
MultiBinary   ❌      ✔️
Hybrid        ✔️       ✔️

============= ====== ===========



Parameters
----------

.. autoclass:: PADDPG
  :members:
  :inherited-members:


.. PADDPG Policies
.. -------------

.. .. autoclass:: MlpPolicy
..   :members:
..   :inherited-members:


