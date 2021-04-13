.. HMLF documentation master file, created by
   sphinx-quickstart on Thu Sep 26 11:06:54 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HMLF Docs - Hybrid Machine Learning Framework
========================================================================

`Hybrid Machine Learning Framework (HMLF) <https://github.tik.uni-stuttgart.de/IFF/HMLF>`_ is a library focuse on parameterized action spaces,
 with state of the art implementation of all major algorithms in the literature.


Github repository: https://github.tik.uni-stuttgart.de/IFF/HMLF



Main Features
--------------

- Paramterized action spaces support
- PEP8 compliant (unified code style)
- Documented functions and classes
- Tests, high code coverage and type hints
- Clean code

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/install
   guide/quickstart
   guide/rl_tips
   guide/rl
   guide/algos
   guide/examples
   guide/vec_envs
   guide/custom_env
   guide/custom_policy
   guide/callbacks
   guide/checking_nan
   guide/developer
   guide/save_format
   guide/export


.. toctree::
  :maxdepth: 1
  :caption: RL Algorithms

  modules/base
  modules/a2c
  modules/ddpg
  modules/dqn
  modules/her
  modules/mpdqn
  modules/paddpg
  modules/pdqn
  modules/ppo
  modules/sac
  modules/sddpg
  modules/td3

.. toctree::
  :maxdepth: 1
  :caption: Common

  common/atari_wrappers
  common/env_util
  common/distributions
  common/evaluation
  common/env_checker
  common/monitor
  common/logger
  common/noise
  common/utils


Citing HMLF
------------------------
To cite this project in publications:

.. code-block:: bibtex

    @misc{hmlf,
      author = {-},
      title = {HMLF - Hybrid Machine Learning Framework},
      year = {2021},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.tik.uni-stuttgart.de/IFF/HMLF}},
    }

Contributing
------------

To any interested in making the rl baselines better, there are still some improvements
that need to be done.
You can check issues in the `repo <https://github.tik.uni-stuttgart.de/IFF/HMLF>`_.

If you want to contribute, please read `CONTRIBUTING.md <https://github.tik.uni-stuttgart.de/IFF/HMLF/blob/master/CONTRIBUTING.md>`_ first.

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`search`
* :ref:`modindex`
