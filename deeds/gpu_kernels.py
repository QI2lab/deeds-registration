import cupy as cp

# RawKernel to compute Hamming distance between two uint64 arrays
_popcount_source = r"""
extern "C" __global__
void popcount(const unsigned long long* a,
              const unsigned long long* b,
              float* out,
              int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        unsigned long long diff = a[idx] ^ b[idx];
        out[idx] = __popcll(diff);
    }
}
"""

popcount_kernel = cp.RawKernel(_popcount_source, 'popcount')


def hamming_distance_gpu(a: cp.ndarray, b: cp.ndarray) -> cp.ndarray:
    """Return per-element Hamming distance between two uint64 arrays."""
    assert a.dtype == cp.uint64 and b.dtype == cp.uint64
    assert a.shape == b.shape
    out = cp.empty_like(a, dtype=cp.float32)
    n = a.size
    block = 256
    grid = (n + block - 1) // block
    popcount_kernel((grid,), (block,), (a, b, out, n))
    return out


def hamming_distance_cpu(a, b):
    """CPU implementation of Hamming distance."""
    import numpy as np

    if isinstance(a, cp.ndarray):
        a = cp.asnumpy(a)
    if isinstance(b, cp.ndarray):
        b = cp.asnumpy(b)
    assert a.dtype == np.uint64 and b.dtype == np.uint64
    out = np.unpackbits(np.bitwise_xor(a, b).view(np.uint8), axis=-1).sum(axis=-1)
    return out.astype(np.float32)
