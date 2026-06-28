import numpy as np
from typing import cast
from numpy.typing import NDArray

def test():
    a = np.array([1, 2, 3])
    # argmin returns intp (which can be int64 or int32 depending on platform)
    # the issue is that it says "redundant-cast". So we can just remove the cast!
    pass
