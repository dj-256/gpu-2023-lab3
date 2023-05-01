import numpy as np


def randomArray(power_of_two, min=-100, max=100):
    return np.random.randint(min, max, 2 ** power_of_two, dtype=np.int32)


def scanCPU(a):
    n = len(a)
    m = np.log2(n).astype(np.int32)
    for d in range(m):
        for k in range(0, n, 2 ** (d + 1)):
            a[k + 2 ** (d + 1) - 1] += a[k + 2 ** d - 1]
    a[n - 1] = 0
    for d in range(m - 1, -1, -1):
        for k in range(0, n, 2 ** (d + 1)):
            t = a[k + 2 ** d - 1]
            a[k + 2 ** d - 1] = a[k + 2 ** (d + 1) - 1]
            a[k + 2 ** (d + 1) - 1] += t


def realExclusiveScan(a):
    return np.concatenate(([0], np.cumsum(a)))[:-1]
