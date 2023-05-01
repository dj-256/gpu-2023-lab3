import sys

import numpy as np
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning
import warnings

from prefix_scan import scanGPU
from util import randomArray, randomArbitraryArray

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

if __name__ == "__main__":
    a = randomArbitraryArray(100)
    print(*a, sep=",")
    res = scanGPU(a)
    print("res", res)
    expected = np.concatenate(([0], np.cumsum(a)))[:-1]
