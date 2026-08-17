import math

import cuda.bindings.driver as cuda
import cutlass
import torch
import torch.nn.functional as F
from cutlass import cute
from cutlass.cute.nvgpu import warp
from cutlass.cute.runtime import from_dlpack

BH, S, D = 2, 128, 64
BLK_M, BLK_N = 16, 16    
THREADS = 32                
NEG = -1.0e30
SCALE = 1.0 / math.sqrt(D)


class FlashAttentionTC:
    @cute.kernel
    def kernel(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
               mO: cute.Tensor, mma1: cute.TiledMma, mma2: cute.TiledMma,
               scale: cutlass.Constexpr):
        tid, _, _ = cute.arch.thread_idx()
        blk, _, _ = cute.arch.block_idx()
        ntile = S // BLK_M
        bh = blk // ntile                 # which (batch, head)
        q0 = (blk % ntile) * BLK_M        # first query row of this tile

        smem = cutlass.utils.SmemAllocator()
        sQ = smem.allocate_tensor(cutlass.Float16, cute.make_layout((BLK_M, D)), byte_alignment=16)
        sK = smem.allocate_tensor(cutlass.Float16, cute.make_layout((BLK_N, D)), byte_alignment=16)
        sV = smem.allocate_tensor(cutlass.Float16, cute.make_layout((D, BLK_N)), byte_alignment=16)  # V^T
        sP = smem.allocate_tensor(cutlass.Float16, cute.make_layout((BLK_M, BLK_N)), byte_alignment=16)
        sS = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BLK_M, BLK_N)), byte_alignment=16)
        sO = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BLK_M, D)), byte_alignment=16)
        sm = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BLK_M,)))
        sl = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BLK_M,)))
        sa = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BLK_M,)))

        # load Q tile once, zero O, init running m = -inf, l = 0
        for i in cutlass.range(tid, BLK_M * D, THREADS):
            sQ[(i // D, i % D)] = mQ[(bh, q0 + i // D, i % D)]
            sO[(i // D, i % D)] = cutlass.Float32(0.0)
        if tid < BLK_M:
            sm[tid] = cutlass.Float32(NEG)
            sl[tid] = cutlass.Float32(0.0)
        cute.arch.sync_threads()

        thr1 = mma1.get_slice(tid)
        thr2 = mma2.get_slice(tid)

        for kb in cutlass.range(S // BLK_N):
            k0 = kb * BLK_N
            # load K block, and V block transposed into sV[d, n] = V[n, d]
            for i in cutlass.range(tid, BLK_N * D, THREADS):
                r = i // D
                c = i % D
                sK[(r, c)] = mK[(bh, k0 + r, c)]
                sV[(c, r)] = mV[(bh, k0 + r, c)]
            cute.arch.sync_threads()

            # ---- GEMM 1: S = Q @ K^T  (tensor core, contract over D) ----
            accS = mma1.make_fragment_C(thr1.partition_C(sS))
            accS.fill(0.0)
            rQ = mma1.make_fragment_A(thr1.partition_A(sQ))
            rK = mma1.make_fragment_B(thr1.partition_B(sK))
            cute.autovec_copy(thr1.partition_A(sQ), rQ)
            cute.autovec_copy(thr1.partition_B(sK), rK)
            cute.gemm(mma1, accS, rQ, rK, accS)
            cute.autovec_copy(accS, thr1.partition_C(sS))   # S -> SMEM (plain layout)
            cute.arch.sync_threads()

            # ---- online softmax (threads 0..BLK_M-1, one query row each) ----
            if tid < BLK_M:
                bmax = cutlass.Float32(NEG)
                for j in cutlass.range(BLK_N):
                    s = sS[(tid, j)] * scale
                    bmax = max(bmax, s)
                mold = sm[tid]
                mnew = mold
                mnew = max(mnew, bmax)
                alpha = cute.arch.exp(mold - mnew)          # rescale factor
                bl = cutlass.Float32(0.0)
                for j in cutlass.range(BLK_N):
                    p = cute.arch.exp(sS[(tid, j)] * scale - mnew)
                    sP[(tid, j)] = p.to(cutlass.Float16)     # P as fp16 = GEMM2 A operand
                    bl = bl + p
                sl[tid] = sl[tid] * alpha + bl               # running denominator
                sm[tid] = mnew
                sa[tid] = alpha
            cute.arch.sync_threads()

            # ---- rescale O by alpha (the alpha*O part) ----
            for i in cutlass.range(tid, BLK_M * D, THREADS):
                r = i // D
                sO[(r, i % D)] = sO[(r, i % D)] * sa[r]
            cute.arch.sync_threads()

            # ---- GEMM 2: O = O + P @ V  (tensor core, accumulate into rescaled O) ----
            accD = mma2.make_fragment_C(thr2.partition_C(sO))
            cute.autovec_copy(thr2.partition_C(sO), accD)    # load rescaled O as C
            rP = mma2.make_fragment_A(thr2.partition_A(sP))
            rV = mma2.make_fragment_B(thr2.partition_B(sV))
            cute.autovec_copy(thr2.partition_A(sP), rP)
            cute.autovec_copy(thr2.partition_B(sV), rV)
            cute.gemm(mma2, accD, rP, rV, accD)              # C = C + P@V
            cute.autovec_copy(accD, thr2.partition_C(sO))
            cute.arch.sync_threads()

        # ---- final normalize: O = O / l ----
        for i in cutlass.range(tid, BLK_M * D, THREADS):
            r = i // D
            c = i % D
            mO[(bh, q0 + r, c)] = (sO[(r, c)] / sl[r]).to(cutlass.Float16)

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor,
                 mO: cute.Tensor, stream: cuda.CUstream):
        op = warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 16))
        mma1 = cute.make_tiled_mma(op)
        mma2 = cute.make_tiled_mma(op)
        self.kernel(mQ, mK, mV, mO, mma1, mma2, SCALE).launch(
            grid=[BH * (S // BLK_M), 1, 1],
            block=[THREADS, 1, 1],
            stream=stream,
        )


def _bench(fn, it=50, wu=20):
    for _ in range(wu):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / it * 1000.0


def main():
    print(f"FUSED tensor-core FlashAttention-2: BH={BH}, S={S}, D={D}")
    print(f"CTA tile BLK_M={BLK_M}, key block BLK_N={BLK_N}, one warp, tensor cores\n")

    torch.manual_seed(0)
    q = torch.randn(BH, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(BH, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(BH, S, D, device="cuda", dtype=torch.float16)
    o = torch.zeros(BH, S, D, device="cuda", dtype=torch.float16)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    mQ, mK, mV, mO = (from_dlpack(x) for x in (q, k, v, o))
    FlashAttentionTC()(mQ, mK, mV, mO, stream)
    torch.cuda.synchronize()

    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
    err = (o.float() - ref).abs().max().item()
    print(f"max abs err vs torch SDPA: {err:.3e}")
    torch.testing.assert_close(o.float(), ref, atol=2e-2, rtol=2e-2)
    print("correctness OK (fused, tensor cores, online softmax)\n")

    compiled = cute.compile(FlashAttentionTC(), mQ, mK, mV, mO, stream)
    ours = _bench(lambda: compiled(mQ, mK, mV, mO, stream))
    torch_t = _bench(lambda: F.scaled_dot_product_attention(q, k, v))
    print(f"ours (fused TC):  {ours:8.2f} us")
    print(f"torch SDPA:       {torch_t:8.2f} us")


if __name__ == "__main__":
    main()
