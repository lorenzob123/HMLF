---
name: "\U0001F916 Custom Gym Environment Issue"
about: How to report an issue when using a custom Gym environment
labels: question, custom gym env
---

### 🤖 Custom Gym Environment

**Please check your environment first using**:

```python
from hmlf.common.env_checker import check_env

env = CustomEnv(arg1, ...)
# It will check your custom environment and output additional warnings if needed
check_env(env)
```

### Describe the bug

A clear and concise description of what the bug is.

### Code example

Please try to provide a minimal example to reproduce the bug.
For a custom environment, you need to give at least the observation space, action space, `reset()` and `step()` methods
(see working example below).
Error messages and stack traces are also helpful.

Please use the [markdown code blocks](https://help.github.com/en/articles/creating-and-highlighting-code-blocks)
for both code and stack traces.

```python
import gym
import numpy as np

from hmlf import spaces
from hmlf import A2C
from hmlf.common.env_checker import check_env


class CustomEnv(gym.Env):

  def __init__(self):
    super(CustomEnv, self).__init__()
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,))
    self.action_space = spaces.Box(low=-1, high=1, shape=(6,))

  def reset(self):
    return self.observation_space.sample()

  def step(self, action):
    obs = self.observation_space.sample()
    reward = 1.0
    done = False
    info = {}
    return obs, reward, done, info

env = CustomEnv()
check_env(env)

model = A2C("MlpPolicy", env, verbose=1).learn(1000)
```

```bash
Traceback (most recent call last): File ...

```