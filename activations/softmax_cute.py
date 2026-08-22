import cuda.bindings.driver as cuda
import cupy as cp
import cutlass
import numpy as np
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

# softmax(x_i) = exp(x_i - max_j x_j) / sum_k exp(x_k - max_j x_j)

M, N = 1024, 1024
THREADS = 256


class Softmax:
    @cute.kernel
    def kernel(self, mX: cute.Tensor, mO: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        row = bidx * THREADS + tidx

        m = mX[row, 0]
        for j in cutlass.range(N):
            v = mX[row, j]
            m = max(m, v)

        denom = cutlass.Float32(0.0)
        for j in cutlass.range(N):
            denom = denom + cute.exp(mX[row, j] - m)

        for j in cutlass.range(N):
            mO[row, j] = cute.exp(mX[row, j] - m) / denom

    @cute.jit
    def __call__(self, mX: cute.Tensor, mO: cute.Tensor, stream: cuda.CUstream):
        self.kernel(mX, mO).launch(
            grid=[M // THREADS, 1, 1],
            block=[THREADS, 1, 1],
            stream=stream,
        )


def main():
    print(f"Row-wise softmax over a {M}x{N} matrix (CuTe DSL)...")

    x_np = np.random.randn(M, N).astype(np.float32)
    x = cp.asarray(x_np)
    o = cp.zeros((M, N), dtype=cp.float32)

    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)
    Softmax()(from_dlpack(x), from_dlpack(o), stream)
    cp.cuda.get_current_stream().synchronize()

    ref = np.exp(x_np - x_np.max(axis=1, keepdims=True))
    ref /= ref.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(o.get(), ref, atol=1e-5, rtol=1e-4)
    print("SUCCESS! CuTe DSL softmax matched NumPy result.")


if __name__ == "__main__":
    main()
