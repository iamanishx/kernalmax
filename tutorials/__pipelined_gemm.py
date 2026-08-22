import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

M = N = K = 128
BM = BN = 16          # output tile
BK = 16               # K-depth per step
THREADS = BM * BN     # 256
STAGES = 2            # double buffering 


class PipelinedGemm:
    @cute.kernel
    def kernel(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
               tcA: cute.TiledCopy, tcB: cute.TiledCopy):
        tid, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        row = tid // BN 
        col = tid % BN

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(cutlass.Float32,
                                  cute.make_layout((STAGES, BM, BK)), byte_alignment=16)
        sB = smem.allocate_tensor(cutlass.Float32,
                                  cute.make_layout((STAGES, BK, BN)), byte_alignment=16)

        gA = cute.local_tile(mA, (BM, BK), (bx, None))
        gB = cute.local_tile(mB, (BK, BN), (None, by))
        thrA = tcA.get_slice(tid)
        thrB = tcB.get_slice(tid)

        ktiles = K // BK

        # prefetch
        cute.copy(tcA, thrA.partition_S(gA[None, None, 0]), thrA.partition_D(sA[0, None, None]))
        cute.copy(tcB, thrB.partition_S(gB[None, None, 0]), thrB.partition_D(sB[0, None, None]))
        cute.arch.cp_async_commit_group()

        acc = cutlass.Float32(0.0)
        for kt in cutlass.range(ktiles):
            cur = kt % STAGES                 # current buffer
            nxt = (kt + 1) % STAGES           # next buffer 

            # prefetch next tile
            if kt + 1 < ktiles:
                cute.copy(tcA, thrA.partition_S(gA[None, None, kt + 1]),
                          thrA.partition_D(sA[nxt, None, None]))
                cute.copy(tcB, thrB.partition_S(gB[None, None, kt + 1]),
                          thrB.partition_D(sB[nxt, None, None]))
                cute.arch.cp_async_commit_group()

            cute.arch.cp_async_wait_group(1)
            cute.arch.sync_threads()

            for kk in cutlass.range(BK):
                acc = acc + sA[cur, row, kk] * sB[cur, kk, col]
            cute.arch.sync_threads()        
            
        mC[(bx * BM + row, by * BN + col)] = acc

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
                 stream: cuda.CUstream):
        atomA = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.Float32, num_bits_per_copy=32)
        atomB = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.Float32, num_bits_per_copy=32)
        tcA = cute.make_tiled_copy_tv(atomA, cute.make_layout((BM, BK), stride=(BK, 1)),
                                      cute.make_layout((1, 1)))
        tcB = cute.make_tiled_copy_tv(atomB, cute.make_layout((BK, BN), stride=(BN, 1)),
                                      cute.make_layout((1, 1)))
        self.kernel(mA, mB, mC, tcA, tcB).launch(
            grid=[M // BM, N // BN, 1], block=[THREADS, 1, 1], stream=stream)


def main():
    print(f"MULTI-STAGE PIPELINED GEMM ({M}x{K} @ {K}x{N}), STAGES={STAGES}")
    print("all warps load AND compute; next tile prefetched via cp.async\n")
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    PipelinedGemm()(from_dlpack(a), from_dlpack(b), from_dlpack(c), stream)
    torch.cuda.synchronize()
    err = (c - a @ b).abs().max().item()
    print(f"max abs err vs torch: {err:.3e}")
    torch.testing.assert_close(c, a @ b, atol=1e-2, rtol=1e-2)
    print("SUCCESS! next tile loads while current tile computes (latency hidden).")


if __name__ == "__main__":
    main()
