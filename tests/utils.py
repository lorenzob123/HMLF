from typing import List, Optional, Tuple

import numpy as np

from hmlf.spaces import Box


def make_box(low: Optional[List] = None, high: Optional[List] = None, shape: Optional[Tuple] = None) -> Box:
    if shape is None:
        if (low is None) and (high is None):
            raise ValueError("Some value needs to be not none")
        else:
            low = np.array(low)
            high = np.array(high)
            return Box(low, high)
    else:
        if low is None:
            low = -np.inf
        if high is None:
            high = np.inf
        return Box(low, high, shape)
