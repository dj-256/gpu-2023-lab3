import numpy as np

from prefix_scan import scanGPU
from util import randomArray, scanCPU, realExclusiveScan, randomArbitraryArray


def test_equal_size():
    a = randomArray(4)
    expected = realExclusiveScan(a)
    res = scanGPU(a, 16)
    assert np.all(res == expected)

def test_twice_size():
    a = randomArray(5)
    expected = realExclusiveScan(a)
    res = scanGPU(a, 16)
    assert np.all(res == expected)

def test_arbitrary_size():
    a = randomArbitraryArray(100)
    expected = realExclusiveScan(a)
    res = scanGPU(a, 16)
    assert np.all(res == expected)
