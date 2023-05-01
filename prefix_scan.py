import argparse
import warnings
from math import log2

import numpy as np
from numba import cuda, int32, NumbaPerformanceWarning

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

@cuda.jit
def scanKernel(d_a, d_a_size, block_sums):
    block_size = cuda.blockDim.x
    local_id = cuda.threadIdx.x
    global_id = cuda.grid(1)
    if global_id >= d_a_size:
        return
    shared_a = cuda.shared.array((16,), int32)
    e = d_a[global_id]
    shared_a[local_id] = e
    cuda.atomic.add(block_sums, cuda.blockIdx.x, e)
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
    cuda.syncthreads()


def scanGPU(a, block_size=16):
    grid_size = (len(a) + block_size - 1) // block_size
    d_a = cuda.to_device(a)
    d_block_sums = cuda.device_array_like(np.zeros(grid_size, dtype=np.int32))
    n = len(a)
    scanKernel[grid_size, block_size](d_a, n, d_block_sums)
    res = d_a.copy_to_host()
    block_sums = d_block_sums.copy_to_host()
    if len(block_sums) > 1:
        block_sums = scanGPU(block_sums)
        for i in range(len(block_sums)):
            res[i * block_size:(i+1) * block_size] += block_sums[i]
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help="input file containing the array")
    parser.add_argument('--tb', type=int, default=16, help="thread block size")
    parser.add_argument('--independent', action='store_true', help="independent scan")
    parser.add_argument('--inclusive', action='store_true', help="inclusive scan")
    args = parser.parse_args()

    file_content = open(args.filename).read()
    a = np.array([int(x) for x in file_content.split(",")])
    res = scanGPU(a)
    print(*res, sep=",")
