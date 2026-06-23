import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.warp as warp
from cutlass.cute.runtime import from_dlpack

import cupy as cp
import numpy as np
import cuda.bindings.driver as cuda


M, N, K = 256, 256, 256

BM, BN, BK = 16, 8, 16
THREADS = 32 


class GemmTcSmem:
    @cute.kernel
    def kernel(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
               tiled_mma: cute.TiledMma,
               tcA: cute.TiledCopy, tcB: cute.TiledCopy):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        gA = cute.local_tile(mA, (BM, BK), (bidx, None))  
        gB = cute.local_tile(mB, (BN, BK), (bidy, None))   
        gC = cute.local_tile(mC, (BM, BN), (bidx, bidy))  

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(cutlass.Float16, cute.make_layout((BM, BK)),
                                  byte_alignment=16)
        sB = smem.allocate_tensor(cutlass.Float16, cute.make_layout((BN, BK)),
                                  byte_alignment=16)

        thr_mma = tiled_mma.get_slice(tidx)
        tCgC = thr_mma.partition_C(gC)
        acc = tiled_mma.make_fragment_C(tCgC)
        acc.fill(0.0)

        thrA = tcA.get_slice(tidx)
        thrB = tcB.get_slice(tidx)

        ktiles = cute.size(gA, mode=[2])
        for k in cutlass.range(ktiles):
            # 1) stage this K-chunk of A and B from GLOBAL into SMEM
            cute.copy(tcA, thrA.partition_S(gA[None, None, k]), thrA.partition_D(sA))
            cute.copy(tcB, thrB.partition_S(gB[None, None, k]), thrB.partition_D(sB))
            # 2) everyone must see the full SMEM tile before reading it
            cute.arch.sync_threads()

            # 3) load each thread's MMA fragment from SMEM into registers
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            rA = tiled_mma.make_fragment_A(tCsA)
            rB = tiled_mma.make_fragment_B(tCsB)
            cute.autovec_copy(tCsA, rA)
            cute.autovec_copy(tCsB, rB)
            cute.gemm(tiled_mma, acc, rA, rB, acc)
            cute.arch.sync_threads()

        cute.autovec_copy(acc, tCgC)

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
                 stream: cuda.CUstream):
        op = warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(op)

        atomA = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),
                                    cutlass.Float16, num_bits_per_copy=16)
        atomB = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),
                                    cutlass.Float16, num_bits_per_copy=16)
        tcA = cute.make_tiled_copy_tv(atomA,
                                      cute.make_layout((8, 4), stride=(4, 1)),
                                      cute.make_layout((2, 4), stride=(4, 1)))
        tcB = cute.make_tiled_copy_tv(atomB,
                                      cute.make_layout((8, 4), stride=(4, 1)),
                                      cute.make_layout((1, 4), stride=(4, 1)))

        self.kernel(mA, mB, mC, tiled_mma, tcA, tcB).launch(
            grid=[M // BM, N // BN, 1], 
            block=[THREADS, 1, 1],         
            stream=stream,
        )


def main():
    print(f"Tensor-core GEMM, SMEM-staged, K-loop: ({M}x{K}) @ ({K}x{N}) -> ({M}x{N})")
    print(f"CTA tile {BM}x{BN}, BK={BK}, {THREADS} threads (1 warp), "
          f"grid {M//BM}x{N//BN}, K-loop {K//BK} iters\n")

    a_np = np.random.randn(M, K).astype(np.float16)
    b_np = np.random.randn(K, N).astype(np.float16)
    a = cp.asarray(a_np)
    b = cp.asarray(b_np)
    c = cp.zeros((M, N), dtype=cp.float32)

    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)
    GemmTcSmem()(from_dlpack(a), from_dlpack(b.T), from_dlpack(c), stream)
    cp.cuda.get_current_stream().synchronize()

    ref = a_np.astype(np.float32) @ b_np.astype(np.float32)
    err = float(np.abs(c.get() - ref).max())
    rel = err / float(np.abs(ref).max())
    print(f"max abs err vs NumPy fp32: {err:.3e}   (relative: {rel:.3e})")
    np.testing.assert_allclose(c.get(), ref, atol=2e-2, rtol=2e-2)
    print("SUCCESS! Staged tensor-core GEMM matches the fp32 reference.")
    print("This is the keystone: SMEM staging + tensor-core MMA + K-loop, all composed.")


if __name__ == "__main__":
    main()
