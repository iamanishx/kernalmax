"""Multi-CTA GEMV ops — @cute.jit, warp-per-row coalesced, bf16-capable.

PERSISTENT MULTI-SM DESIGN
-------------------------
grid = [NUM_SMS] so every SM has a resident CTA. Work split:

  * Big GEMVs (Q/K/V/O/gate/up/down/lm_head) are split across CTAs by output
    row: global warp id `cta*NUM_WARPS + warp` strides over N by TOTAL_WARPS.
    Outputs go to GLOBAL memory so all CTAs can see them; a grid_barrier
    follows each one.
  * Small ops (rmsnorm/rope/gqa) are computed REDUNDANTLY by every CTA in its
    own SMEM. Cheaper than splitting + barriering (896 elems vs a barrier).

Coalescing: lanes read consecutive k → 1-2 transactions/warp instead of 32.
After warp_reduction all lanes hold the full sum, so any lane can write.
"""

import cutlass
from config import NUM_CTAS, NUM_THREADS
from cutlass import cute

THREADS = NUM_THREADS
NUM_WARPS = NUM_THREADS // 32
TOTAL_WARPS = NUM_CTAS * NUM_WARPS      # warps across the whole grid


@cute.jit
def g2s(gIn, sOut, tid, D: cutlass.Constexpr):
    """Copy a global activation vector into this CTA's SMEM."""
    for i in cutlass.range(tid, D, THREADS):
        sOut[i] = gIn[i]


@cute.jit
def gemv_bias_mc(sIn, w_all, b_all, gOut, tid, cta, layer: cutlass.Constexpr,
                 K: cutlass.Constexpr, N: cutlass.Constexpr):
    """gOut[n] = sum_k sIn[k]*w[layer,n,k] + b[layer,0,n]. Split across CTAs."""
    warp = tid // 32
    lane = tid % 32
    gwarp = cta * NUM_WARPS + warp
    for n in cutlass.range(gwarp, N, TOTAL_WARPS):
        acc = cutlass.Float32(0.0)
        for k in cutlass.range(lane, K, 32):
            acc = acc + sIn[k] * w_all[(layer, n, k)].to(cutlass.Float32)
        acc = cute.arch.warp_reduction(acc, lambda a, b: a + b, threads_in_group=32)
        gOut[n] = acc + b_all[(layer, 0, n)].to(cutlass.Float32)


@cute.jit
def gemv_residual_mc(sIn, w_all, gResid, tid, cta, layer: cutlass.Constexpr,
                     K: cutlass.Constexpr, N: cutlass.Constexpr):
    """gResid[n] += sum_k sIn[k]*w[layer,n,k]. Disjoint n slices per CTA."""
    warp = tid // 32
    lane = tid % 32
    gwarp = cta * NUM_WARPS + warp
    for n in cutlass.range(gwarp, N, TOTAL_WARPS):
        acc = cutlass.Float32(0.0)
        for k in cutlass.range(lane, K, 32):
            acc = acc + sIn[k] * w_all[(layer, n, k)].to(cutlass.Float32)
        acc = cute.arch.warp_reduction(acc, lambda a, b: a + b, threads_in_group=32)
        gResid[n] = gResid[n] + acc


@cute.jit
def mlp_gate_silu_mc(sN, w_gate, w_up, gOut, tid, cta, layer: cutlass.Constexpr,
                     K: cutlass.Constexpr, N: cutlass.Constexpr):
    """gOut[n] = SiLU(gate[n]) * up[n], both projections fused in one pass."""
    warp = tid // 32
    lane = tid % 32
    gwarp = cta * NUM_WARPS + warp
    for n in cutlass.range(gwarp, N, TOTAL_WARPS):
        g = cutlass.Float32(0.0)
        u = cutlass.Float32(0.0)
        for k in cutlass.range(lane, K, 32):
            xn = sN[k]
            g = g + xn * w_gate[(layer, n, k)].to(cutlass.Float32)
            u = u + xn * w_up[(layer, n, k)].to(cutlass.Float32)
        g = cute.arch.warp_reduction(g, lambda a, b: a + b, threads_in_group=32)
        u = cute.arch.warp_reduction(u, lambda a, b: a + b, threads_in_group=32)
        sig = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.arch.exp(-g))
        gOut[n] = g * sig * u


@cute.jit
def lm_head_mc(sIn, w_lm, mOut, tid, cta,
               K: cutlass.Constexpr, VOCAB: cutlass.Constexpr):
    """mOut[0,v] = sum_k sIn[k]*w_lm[v,k]. Largest GEMV (~27% of weight bytes)."""
    warp = tid // 32
    lane = tid % 32
    gwarp = cta * NUM_WARPS + warp
    for v in cutlass.range(gwarp, VOCAB, TOTAL_WARPS):
        acc = cutlass.Float32(0.0)
        for k in cutlass.range(lane, K, 32):
            acc = acc + sIn[k] * w_lm[(v, k)].to(cutlass.Float32)
        acc = cute.arch.warp_reduction(acc, lambda a, b: a + b, threads_in_group=32)
        mOut[(0, v)] = acc
