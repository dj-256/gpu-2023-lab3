import numpy as np

from prefix_scan import scanGPU
from util import randomArray, scanCPU, realExclusiveScan


def test_equal_size():
    a = randomArray(2, 0, 10)
    print("a", a)
    expected = realExclusiveScan(a)
    res = scanGPU(a)
    assert np.allclose(res, expected)
