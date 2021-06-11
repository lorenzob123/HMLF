[![Build Status](https://www.travis-ci.com/lorenzob123/HMLF.svg?branch=master)](https://www.travis-ci.com/lorenzob123/HMLF)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# Hmlf Project
<img src="docs/\_static/img/hmlf.png" align="right" width="40%"/>
This library is a fork of StableBaselines3 with the focus on parametrized action spaces.
The objective of this work is to make it easier to work with environments where the tasks to perform have a discrete components and a continuos ones, while maintaing state of the art implementations and user friendliness. 

# Algorithms available


**Standard Algorithms**
- A2C
- DDPG
- DQN
- HER
- PPO
- TD3

**Hybrid Algorithms**
- A2C
- MP-DQN
- PADDPG
- P-DQN
- PPO
- S-DDPG


## Installation
To install the library you need to clone it on your local machine

```git clone https://github.tik.uni-stuttgart.de/IFF/HMLF```

and then run the command

```pip install -e .```

and now you can already use the library. If you want extra functionalities for testing the code with `pytest` or building the documentation use this instead

```pip install -e .[extra]```


## Documentation

To build the documentation simply run the following commmand in the main folder
``` make doc```
This will create the html files for the documentation, that can be found in `docs/_build/html`.
