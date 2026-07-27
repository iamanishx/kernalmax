import cuda.bindings.driver as cuda
import cupy as cp
import cutlass
import numpy as np
from cutlass import cute
from cutlass.cute.nvgpu import warp
from cutlass.cute.runtime import from_dlpack

M, N, K = 16, 8, 16


class TensorCoreMma:
    @cute.kernel
    def kernel(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
               tiled_mma: cute.TiledMma):
        tidx, _, _ = cute.arch.thread_idx()

        thr_mma = tiled_mma.get_slice(tidx)
        tCgA = thr_mma.partition_A(mA)
        tCgB = thr_mma.partition_B(mB)
        tCgC = thr_mma.partition_C(mC)

        # per-lane register fragments shaped exactly to feed the tensor core
        rA  = tiled_mma.make_fragment_A(tCgA)
        rB  = tiled_mma.make_fragment_B(tCgB)
        acc = tiled_mma.make_fragment_C(tCgC)

        cute.autovec_copy(tCgA, rA)
        cute.autovec_copy(tCgB, rB)

        acc.fill(0.0)


        # the whole warp issues one mma.sync, computing D = A @ B + 0 in fp32.
        cute.gemm(tiled_mma, acc, rA, rB, acc)


        cute.autovec_copy(acc, tCgC)

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
                 stream: cuda.CUstream):
        # the MMA atom: fp16 inputs, fp32 accumulator, shape 16x8x16 (one warp)
        op = warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (M, N, K))

        tiled_mma = cute.make_tiled_mma(op)
        self.kernel(mA, mB, mC, tiled_mma).launch(
            grid=[1, 1, 1],
            block=[32, 1, 1],
            stream=stream,
        )


def main():
    print(f"One warp, one tensor-core MMA: ({M}x{K}) @ ({K}x{N}) -> ({M}x{N}) fp32")

    a_np = np.random.randn(M, K).astype(np.float16)
    b_np = np.random.randn(K, N).astype(np.float16)
    a = cp.asarray(a_np)
    b = cp.asarray(b_np)
    c = cp.zeros((M, N), dtype=cp.float32)

    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)
    # B is passed as b.T -> view (N,K), because cute.gemm contracts trailing mode
    TensorCoreMma()(from_dlpack(a), from_dlpack(b.T), from_dlpack(c), stream)
    cp.cuda.get_current_stream().synchronize()

    ref = a_np.astype(np.float32) @ b_np.astype(np.float32)
    err = float(np.abs(c.get() - ref).max())
    print(f"max abs error vs NumPy fp32 reference: {err:.2e}")
    np.testing.assert_allclose(c.get(), ref, atol=1e-2, rtol=1e-2)
    print("SUCCESS! the tensor core produced a correct 16x8 fp32 output tile.")


if __name__ == "__main__":
    main()
