RL Algorithms
=============

This table displays the rl algorithms that are implemented in the Stable Baselines3 project,
along with some useful characteristics: support for discrete/continuous actions, multiprocessing.


============ =========== =========== ============ ================= =============== ================
Name         ``Hybrid``  ``Box``     ``Discrete`` ``MultiDiscrete`` ``MultiBinary`` Multi Processing
============ =========== =========== ============ ================= =============== ================
A2C          ✔️           ✔️           ✔️            ✔️                 ✔️               ✔️
DDPG         ❌           ✔️          ❌            ❌                ❌              ❌
DQN          ❌           ❌           ✔️           ❌                ❌              ❌
HER          ❌           ✔️            ✔️           ❌                ❌              ❌
MPDQN        ✔️           ❌            ❌           ❌                ❌              ❌
PADDPG       ✔️           ❌            ❌           ❌                ❌              ❌
PDQN         ✔️           ❌            ❌           ❌                ❌              ❌
PPO          ✔️           ✔️           ✔️            ✔️                 ✔️               ✔️
SAC          ❌           ✔️          ❌            ❌                ❌              ❌
SDDPG        ✔️           ❌            ❌           ❌                ❌              ❌
TD3          ❌           ✔️          ❌            ❌                ❌              ❌
============ =========== =========== ============ ================= =============== ================



Actions ``hmlf.spaces``:

-  ``Hybrid``: a Tuple encoding hybrid actions. The first entry is the
   discrete action executed and the remaining entries are the parameters
   for each of the discrete actions
-  ``Box``: A N-dimensional box that contains every point in the action
   space.
-  ``Discrete``: A list of possible actions, where each timestep only
   one of the actions can be used.
-  ``MultiDiscrete``: A list of possible actions, where each timestep only one action of each discrete set can be used.
- ``MultiBinary``: A list of possible actions, where each timestep any of the actions can be used in any combination.


Reproducibility
---------------

Completely reproducible results are not guaranteed across Python releases or different platforms.
Furthermore, results need not be reproducible between CPU and GPU executions, even when using identical seeds.

In order to make computations deterministics, on your specific problem on one specific platform,
you need to pass a ``seed`` argument at the creation of a model.
If you pass an environment to the model using ``set_env()``, then you also need to seed the environment first.


Credit: part of the *Reproducibility* section comes from `PyTorch Documentation <https://pytorch.org/docs/stable/notes/randomness.html>`_
