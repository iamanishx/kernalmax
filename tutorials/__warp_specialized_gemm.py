import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

M = N = K = 128
BM = BN = 16             
BK = 16                    # K-depth per step
NCONS = BM * BN            # 256 consumer threads (one per output element)
NPROD = 64                 # 2 producer warps
THREADS = NCONS + NPROD  
STAGES = 2            


class WarpSpecializedGemm:
    @cute.kernel
    def kernel(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(cutlass.Float32,
                                  cute.make_layout((STAGES, BM, BK)), byte_alignment=16)
        sB = smem.allocate_tensor(cutlass.Float32,
                                  cute.make_layout((STAGES, BK, BN)), byte_alignment=16)

        gA = cute.local_tile(mA, (BM, BK), (bx, None))    # (BM, BK, ktiles)
        gB = cute.local_tile(mB, (BK, BN), (None, by))    # (BK, BN, ktiles)

        prod = tid >= NCONS        # last NPROD threads are PRODUCERS
        cid = tid                  # consumer id  (valid when tid <  NCONS)
        pid = tid - NCONS          # producer id  (valid when tid >= NCONS)
        row = cid // BN            # consumer's output element
        col = cid % BN
        ktiles = K // BK

        if prod:
            for e in cutlass.range(pid, BM * BK, NPROD):
                sA[0, e // BK, e % BK] = gA[e // BK, e % BK, 0]
            for e in cutlass.range(pid, BK * BN, NPROD):
                sB[0, e // BN, e % BN] = gB[e // BN, e % BN, 0]
        cute.arch.sync_threads()

        acc = cutlass.Float32(0.0)
        for kt in cutlass.range(ktiles):
            cur = kt % STAGES
            nxt = (kt + 1) % STAGES

            if prod:
                if kt + 1 < ktiles:
                    for e in cutlass.range(pid, BM * BK, NPROD):
                        sA[nxt, e // BK, e % BK] = gA[e // BK, e % BK, kt + 1]
                    for e in cutlass.range(pid, BK * BN, NPROD):
                        sB[nxt, e // BN, e % BN] = gB[e // BN, e % BN, kt + 1]
            else:
                for kk in cutlass.range(BK):
                    acc = acc + sA[cur, row, kk] * sB[cur, kk, col]

            cute.arch.sync_threads() 

        if not prod:
            mC[(bx * BM + row, by * BN + col)] = acc

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
                 stream: cuda.CUstream):
        self.kernel(mA, mB, mC).launch(
            grid=[M // BM, N // BN, 1], block=[THREADS, 1, 1], stream=stream)


def main():
    print(f"WARP-SPECIALIZED GEMM ({M}x{K} @ {K}x{N})")
    print(f"{NPROD} producer threads (load) + {NCONS} consumer threads (compute), "
          f"STAGES={STAGES}\n")
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    WarpSpecializedGemm()(from_dlpack(a), from_dlpack(b), from_dlpack(c), stream)
    torch.cuda.synchronize()
    err = (c - a @ b).abs().max().item()
    print(f"max abs err vs torch: {err:.3e}")
    torch.testing.assert_close(c, a @ b, atol=1e-2, rtol=1e-2)
    print("SUCCESS! producer warps load the next tile while consumer warps compute.")
    print("(on Hopper: TMA + wgmma + mbarriers make this the FA-3 pattern)")


if __name__ == "__main__":
    main()
