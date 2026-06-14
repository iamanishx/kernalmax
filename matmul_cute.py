import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import cupy as cp
import numpy as np
import cuda.bindings.driver as cuda

M, N, K = 256, 256, 256
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 8

# ---- Thread layout of the tiled MMA ------------------------------------------
# 16x16 = 256 threads. Each thread accumulates a small sub-block of the
# BLOCK_M x BLOCK_N output tile in registers.
NUM_THREADS = 16 * 16


class SimpleGemm:
    def __init__(self, bM=BLOCK_M, bN=BLOCK_N, bK=BLOCK_K):
        self.bM, self.bN, self.bK = bM, bN, bK

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
                 stream: cuda.CUstream):
        # MMA atom = one FMA in fp32. The tiled MMA arranges a 16x16 grid of
        # threads (last mode = 1 means a single atom along K).
        op = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
        atoms_layout = cute.make_layout((16, 16, 1), stride=(16, 1, 0))
        tiled_mma = cute.make_tiled_mma(op, atoms_layout)

        # One CTA per (BLOCK_M x BLOCK_N) tile of C.
        grid_m = (M + self.bM - 1) // self.bM
        grid_n = (N + self.bN - 1) // self.bN

        self.kernel(mA, mB, mC, tiled_mma).launch(
            grid=[grid_m, grid_n, 1],
            block=[NUM_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
               tiled_mma: cute.TiledMma):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        # ---- Slice the global tensors into this CTA's tiles -----------------
        # mA is (M, K): take rows [bidx*bM : +bM], keep all K -> (bM, bK, k_tiles)
        gA = cute.local_tile(mA, tiler=(self.bM, self.bK),
                             coord=(bidx, None))
        # mB is (N, K): take cols [bidy*bN : +bN], keep all K -> (bN, bK, k_tiles)
        # NOTE: cute.gemm contracts the *last* mode of both A and B, so B must be
        # presented as (N, K). We pass b.T on the host so this view is (N, K).
        gB = cute.local_tile(mB, tiler=(self.bN, self.bK),
                             coord=(bidy, None))
        # mC is (M, N): this CTA's output tile -> (bM, bN)
        gC = cute.local_tile(mC, tiler=(self.bM, self.bN),
                             coord=(bidx, bidy))

        thr_mma = tiled_mma.get_slice(tidx)

        tCgC = thr_mma.partition_C(gC)          
        acc = tiled_mma.make_fragment_C(tCgC)
        acc.fill(0.0)

        k_tiles = cute.size(gA, mode=[2])
        for k in cutlass.range(k_tiles):
            tCrA = thr_mma.partition_A(gA[None, None, k])
            tCrB = thr_mma.partition_B(gB[None, None, k])
            # acc += A_tile @ B_tile
            cute.gemm(tiled_mma, acc, tCrA, tCrB, acc)

        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),
                                        mC.element_type)
        cute.copy(copy_atom, acc, tCgC)


def main():
    print(f"Multiplying {M}x{K} and {K}x{N} matrices (CuTe DSL SIMT GEMM)...")


    a_np = np.random.randn(M, K).astype(np.float32)
    b_np = np.random.randn(K, N).astype(np.float32)
    a = cp.asarray(a_np)
    b = cp.asarray(b_np)
    c = cp.zeros((M, N), dtype=cp.float32)

    mA = from_dlpack(a)
    mB = from_dlpack(b.T)
    mC = from_dlpack(c)

    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)

    gemm = SimpleGemm()
    gemm(mA, mB, mC, stream)
    cp.cuda.get_current_stream().synchronize()

    expected = a_np @ b_np
    np.testing.assert_allclose(c.get(), expected, atol=1e-2, rtol=1e-3)
    print("SUCCESS! CuTe DSL matmul matched NumPy result.")


if __name__ == "__main__":
    main()
