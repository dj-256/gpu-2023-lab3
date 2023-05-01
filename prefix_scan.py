from math import log2

import numpy as np
from numba import cuda, int32


@cuda.jit
def scanKernel(d_a, d_a_size, block_sums):
    block_size = cuda.blockDim.x
    local_id = cuda.threadIdx.x
    global_id = cuda.grid(1)
    if global_id >= d_a_size:
        return
    shared_a = cuda.shared.array((16,), int32)
    shared_a[local_id] = d_a[global_id]
    cuda.syncthreads()

    n = block_size
    m = int(log2(n))
    k = local_id

    for d in range(m):
        if k < n and k % 2 ** (d + 1) == 0:
            shared_a[k + 2 ** (d + 1) - 1] += shared_a[k + 2 ** d - 1]
    shared_a[n - 1] = 0
    for d in range(m - 1, -1, -1):
        if k < n and k % 2 ** (d + 1) == 0:
            t = shared_a[k + 2 ** d - 1]
            shared_a[k + 2 ** d - 1] = shared_a[k + 2 ** (d + 1) - 1]
            shared_a[k + 2 ** (d + 1) - 1] += t
    d_a[global_id] = shared_a[local_id]
    block_sums[cuda.blockIdx.x] += shared_a[local_id]
    cuda.syncthreads()


def scanGPU(a, block_size=16):
    grid_size = (len(a) + block_size - 1) // block_size
    d_a = cuda.to_device(a)
    d_block_sums = cuda.to_device(np.zeros(len(a) // block_size + 1, dtype=np.int32))
    n = len(a)
    scanKernel[grid_size, block_size](d_a, n, d_block_sums)
    res = d_a.copy_to_host()
    return res
