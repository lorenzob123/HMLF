.. _install:

Installation
============

Prerequisites
-------------

HMLF requires python 3.6+.

Windows 10
~~~~~~~~~~

We recommend using `Anaconda <https://conda.io/docs/user-guide/install/windows.html>`_ for Windows users for easier installation of Python packages and required libraries. You need an environment with Python version 3.6 or above.

For a quick start you can move straight to installing HMLF in the next step.

.. note::

	Trying to create Atari environments may result to vague errors related to missing DLL files and modules. This is an
	issue with atari-py package. `See this discussion for more information <https://github.com/openai/atari-py/issues/65>`_.


Bleeding-edge version
---------------------

.. code-block:: bash

	pip install git+https://github.tik.uni-stuttgart.de/IFF/HMLF


Development version
-------------------

To contribute to HMLF, with support for running tests and building the documentation.

.. code-block:: bash

    git clone https://github.tik.uni-stuttgart.de/IFF/HMLF && cd HMLF
    pip install -e .[docs,tests,extra]


Using Docker Images
-------------------

To use the docker image you will need an access token or a valid login and then running.

.. code-block:: bash

	docker build --build-arg GIT_ACCESS_TOKEN=<your-access-token> -dt hmlf .

This image uses the gpu acceleration by default.