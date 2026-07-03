
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.warp as warp
from cutlass.cute.runtime import from_dlpack

import torch
import cuda.bindings.driver as cuda


M, N, K = 256, 256, 256

BM, BN, BK = 32, 16, 16
THREADS = 128         

class GemmTcSmemV2:
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
            cute.copy(tcA, thrA.partition_S(gA[None, None, k]), thrA.partition_D(sA))
            cute.copy(tcB, thrB.partition_S(gB[None, None, k]), thrB.partition_D(sB))
            cute.arch.sync_threads()

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
        tiled_mma = cute.make_tiled_mma(op, cute.make_layout((2, 2, 1)))

        atomA = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),
                                    cutlass.Float16, num_bits_per_copy=16)
        atomB = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),
                                    cutlass.Float16, num_bits_per_copy=16)
        # A tile (32,16) over 128 threads: thr (16,8)=128, val (2,2)=4 -> 32x16
        tcA = cute.make_tiled_copy_tv(atomA,
                                      cute.make_layout((16, 8), stride=(8, 1)),
                                      cute.make_layout((2, 2), stride=(2, 1)))
        # B tile (16,16) over 128 threads: thr (16,8)=128, val (1,2)=2 -> 16x16
        tcB = cute.make_tiled_copy_tv(atomB,
                                      cute.make_layout((16, 8), stride=(8, 1)),
                                      cute.make_layout((1, 2), stride=(2, 1)))

        self.kernel(mA, mB, mC, tiled_mma, tcA, tcB).launch(
            grid=[M // BM, N // BN, 1],  
            block=[THREADS, 1, 1], 
            stream=stream,
        )


def main():
    print(f"MULTI-WARP tensor-core GEMM: ({M}x{K}) @ ({K}x{N}) -> ({M}x{N})")
    print(f"CTA tile {BM}x{BN}, BK={BK}, {THREADS} threads (4 warps as 2x2), "
          f"grid {M//BM}x{N//BN}\n")

    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    GemmTcSmemV2()(from_dlpack(a), from_dlpack(b.T), from_dlpack(c), stream)
    torch.cuda.synchronize()

    ref = a.float() @ b.float()
    err = (c - ref).abs().max().item()
    rel = err / ref.abs().max().item()
    print(f"max abs err vs torch fp32: {err:.3e}   (relative: {rel:.3e})")
    torch.testing.assert_close(c, ref, atol=2e-2, rtol=2e-2)
    print("SUCCESS! 4 warps cooperated on each 32x16 tile via make_tiled_mma((2,2,1)).")
    print("New skill unlocked: multi-warp tensor-core composition. Next: swizzle + ldmatrix (v3).")


if __name__ == "__main__":
    main()
