import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cuda.bindings.driver as cuda
import cutlass
import torch
from block_reduce import block_reduce_max, block_reduce_sum
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

BH, S, D = 2, 128, 64  
BLK_N = D   
THREADS = BLK_N
WARPS = THREADS // 32
NEG_INF = -3.4e38


class FlashAttention:
    @cute.kernel
    def kernel(self, mQ: cute.Tensor, mK: cute.Tensor,
               mV: cute.Tensor, mO: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        i, bh, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        qs = smem.allocate_tensor(cutlass.Float32, cute.make_layout(D), byte_alignment=4)
        ps = smem.allocate_tensor(cutlass.Float32, cute.make_layout(BLK_N), byte_alignment=4)
        red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(32), byte_alignment=4)
        acc = smem.allocate_tensor(cutlass.Float32, cute.make_layout(D), byte_alignment=4)

        if tid < D:
            qs[tid] = mQ[bh, i, tid]
            acc[tid] = cutlass.Float32(0.0) # fa numerator
        cute.arch.sync_threads()

        m = cutlass.Float32(NEG_INF)     # running max
        l = cutlass.Float32(0.0)         # running denominator (sum of exp)

        nblk = S // BLK_N
        for blk in cutlass.range(nblk):
            # thread tid scores KEY (blk*BLK_N + tid) against the query row
            kj = blk * BLK_N + tid
            s = cutlass.Float32(0.0)
            for d in cutlass.range(D):
                s = s + qs[d] * mK[bh, kj, d]

            blk_max = block_reduce_max(s, red, tid, WARPS)
            m_new = m
            m_new = max(m_new, blk_max)
            alpha = cute.exp(m - m_new)          # correction factor for old state

            p = cute.exp(s - m_new)              # softmax numerator
            ps[tid] = p
            blk_sum = block_reduce_sum(p, red, tid, WARPS)

            l = l * alpha + blk_sum              # rescale old denom, add this block
            cute.arch.sync_threads()             # ps[] fully written before PV read

            # rescale the running output by alpha, then add p @ V_block
            if tid < D:
                a = acc[tid] * alpha
                for j in cutlass.range(BLK_N):
                    a = a + ps[j] * mV[bh, blk * BLK_N + j, tid]
                acc[tid] = a

            m = m_new                            # advance the running max
            cute.arch.sync_threads()             # acc[] done before next block

        # ---- f O = acc / l ----
        if tid < D:
            mO[bh, i, tid] = acc[tid] / l

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor,
                 mV: cute.Tensor, mO: cute.Tensor, stream: cuda.CUstream):
        self.kernel(mQ, mK, mV, mO).launch(
            grid=[S, BH, 1],         
            block=[THREADS, 1, 1],
            stream=stream,
        )


def _bench(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters 


def main():
    print(f"FUSED FlashAttention (online softmax): Q,K,V = [{BH},{S},{D}]")
    print(f"grid [S={S}, BH={BH}], {THREADS} threads, K/V walked in "
          f"{S//BLK_N} blocks of {BLK_N}\n")

    torch.manual_seed(0)
    q = torch.randn(BH, S, D, device="cuda", dtype=torch.float32)
    k = torch.randn(BH, S, D, device="cuda", dtype=torch.float32)
    v = torch.randn(BH, S, D, device="cuda", dtype=torch.float32)
    o = torch.zeros(BH, S, D, device="cuda", dtype=torch.float32)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    flash = FlashAttention()
    flash(from_dlpack(q), from_dlpack(k), from_dlpack(v), from_dlpack(o), stream)
    torch.cuda.synchronize()

    # torch reference: plain softmax(Q@K^T)@V (no 1/sqrt(D) scale, matches our kernel)
    ref = torch.softmax(q @ k.transpose(-2, -1), dim=-1) @ v
    err = (o - ref).abs().max().item()
    print(f"max abs err vs torch softmax(Q@K^T)@V: {err:.3e}")
    torch.testing.assert_close(o, ref, atol=1e-4, rtol=1e-4)
    print("SUCCESS! Fused attention matches the reference, WITHOUT storing the full score row.")
    print("This is the FlashAttention-2 algorithm: online softmax with alpha rescaling.\n")

    # ---- benchmark: our  vs torch's fused SDPA (the real FlashAttention) ----
    # IMPORTANT: AOT-compile with cute.compile FIRST. Calling the @cute.jit
    # wrapper directly re-traces on every call (~1000x overhead!), so a naive
    # loop would time the Python tracer, not the GPU kernel.
    cq, ck, cv, co = (from_dlpack(q), from_dlpack(k), from_dlpack(v), from_dlpack(o))
    compiled = cute.compile(flash, cq, ck, cv, co, stream)
    def run_ours():
        compiled(cq, ck, cv, co, stream)

    # SDPA applies 1/sqrt(D); scale q up so it computes the same thing we do
    qs = q * (D ** 0.5)
    def run_sdpa():
        torch.nn.functional.scaled_dot_product_attention(qs, k, v)

    us_ours = _bench(run_ours)
    us_sdpa = _bench(run_sdpa)
    print("Benchmark (avg over 200 iters, AOT-compiled):")
    print(f"  ours  (scalar fp32, 1 CTA/row) : {us_ours:8.2f} us")
    print(f"  torch SDPA (fused, optimized)  : {us_sdpa:8.2f} us")
    print("  NOTE: this problem is TINY (BH=2, S=128), so both numbers are")
    print("  dominated by kernel-launch overhead, not real compute. A fair")
    print("  speed test needs big shapes + tensor cores (that is the v5 step).")


if __name__ == "__main__":
    main()
