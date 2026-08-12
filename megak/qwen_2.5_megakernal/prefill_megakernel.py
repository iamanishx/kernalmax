import math

import cuda.bindings.driver as cuda
import cutlass
from config import (
    FFN_INTERMEDIATE,
    GROUP,
    HALF,
    HEAD_DIM,
    HIDDEN_DIM,
    KV_DIM,
    NUM_CTAS,
    NUM_KV_HEADS,
    NUM_LAYERS,
    NUM_Q_HEADS,
    Q_DIM,
    RMS_NORM_EPS,
    VOCAB_SIZE,
)
from cutlass import cute
from cutlass.cute.nvgpu import warp
from ops.reduce import grid_barrier

# PREFILL uses 128 threads = 4 warps, matching make_tiled_mma(op,(2,2,1)).
# TRAP: a (2,2,1) tiled MMA is a 4-warp op. Launching 512 threads gives
# threads >=128 invalid MMA partitions -> silently corrupt output.
THREADS = 128
NUM_WARPS = THREADS // 32      # 4
TOTAL_WARPS = NUM_CTAS * NUM_WARPS

BM, BN, BK = 32, 128, 16       # GEMM tile (validated in ops/gemm_prefill.py)
NEG = -1.0e30
ATTN_SCALE = 1.0 / math.sqrt(HEAD_DIM)
PF_BARRIERS = 9                # barrier gens per layer — MUST equal the
                               # number of grid_barrier calls emitted below
                               # (base+1 .. base+9). Trap: mismatch aliases
                               # generations across layers -> early release.

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


# ── tensor-core GEMM over the grid: gOut[M,N] = A[M,K] @ w[layer,N,K]^T ──
@cute.jit
def pf_gemm(mA, w_all, gOut, b_all, sA, sB, sC, mma, cta, tid, layer: cutlass.Constexpr,
            M: cutlass.Constexpr, N: cutlass.Constexpr, K: cutlass.Constexpr,
            residual: cutlass.Constexpr, has_bias: cutlass.Constexpr):
    n_mt = M // BM
    n_nt = N // BN
    total = n_mt * n_nt
    for blk in cutlass.range(cta, total, NUM_CTAS):
        mi = blk % n_mt
        ni = blk // n_mt
        m0 = mi * BM
        n0 = ni * BN
        # SMEM is allocated ONCE by the kernel and passed in.
        # TRAP: a second SmemAllocator() inside a helper restarts at the SMEM
        # base, so its buffers ALIAS the kernel's (e.g. sR) -> corruption.
        tm = mma.get_slice(tid)
        acc = mma.make_fragment_C(tm.partition_C(sC)); acc.fill(0.0)
        for k in cutlass.range(K // BK):
            k0 = k * BK
            for i in cutlass.range(tid, BM * BK, THREADS):
                r, c = i // BK, i % BK
                sA[(r, c)] = mA[(m0 + r, k0 + c)].to(cutlass.BFloat16)
            for i in cutlass.range(tid, BN * BK, THREADS):
                r, c = i // BK, i % BK
                sB[(r, c)] = w_all[(layer, n0 + r, k0 + c)]
            cute.arch.sync_threads()
            rA = mma.make_fragment_A(tm.partition_A(sA))
            rB = mma.make_fragment_B(tm.partition_B(sB))
            cute.autovec_copy(tm.partition_A(sA), rA)
            cute.autovec_copy(tm.partition_B(sB), rB)
            cute.gemm(mma, acc, rA, rB, acc)
            cute.arch.sync_threads()
        cute.autovec_copy(acc, tm.partition_C(sC)); cute.arch.sync_threads()
        # EPILOGUE: bias is added HERE, by the CTA that owns this n-slice.
        # TRAP: a separate row-split bias loop after the column-split GEMM races
        # (CTA0 would read columns CTA1 hasn't written) -> deterministic garbage.
        for i in cutlass.range(tid, BM * BN, THREADS):
            r, c = i // BN, i % BN
            val = sC[(r, c)]
            if cutlass.const_expr(has_bias):
                val = val + b_all[(layer, 0, n0 + c)].to(cutlass.Float32)
            if cutlass.const_expr(residual):
                gOut[(m0 + r, n0 + c)] = gOut[(m0 + r, n0 + c)] + val
            else:
                gOut[(m0 + r, n0 + c)] = val
        cute.arch.sync_threads()


# ── per-row rmsnorm over the grid (per-CTA block reduce via sR scratch) ──
@cute.jit
def pf_rmsnorm(gIn, w_norm, gOut, sR, cta, tid, layer: cutlass.Constexpr,
               M: cutlass.Constexpr, D: cutlass.Constexpr):
    # FIXED trip count for every CTA (rows_per), body guarded by m < M, so all
    # CTAs execute the same number of sync_threads() -> no desync.
    rows_per: cutlass.Constexpr = (M + NUM_CTAS - 1) // NUM_CTAS
    lane = tid % 32
    wid = tid // 32
    for it in cutlass.range_constexpr(rows_per):
        m = cta * rows_per + it
        p = cutlass.Float32(0.0)
        if m < M:
            for k in cutlass.range(tid, D, THREADS):
                x = gIn[(m, k)]
                p = p + x * x
        ws = cute.arch.warp_reduction(p, lambda a, b: a + b, threads_in_group=32)
        sR[lane] = cutlass.Float32(0.0)
        cute.arch.sync_threads()
        sR[wid] = ws
        cute.arch.sync_threads()
        tot = cute.arch.warp_reduction(sR[lane], lambda a, b: a + b, threads_in_group=32)
        inv = cute.math.rsqrt(tot / cutlass.Float32(D) + cutlass.Float32(RMS_NORM_EPS))
        if m < M:
            for k in cutlass.range(tid, D, THREADS):
                gOut[(m, k)] = gIn[(m, k)] * inv * w_norm[(layer, 0, k)].to(cutlass.Float32)
        cute.arch.sync_threads()


# ── per-row RoPE (position = row index) + write KV cache ──
@cute.jit
def pf_rope_kv(gQ, gK, gV, gKC, gVC, mRope, cta, tid, layer: cutlass.Constexpr,
               M: cutlass.Constexpr):
    rows_per: cutlass.Constexpr = (M + NUM_CTAS - 1) // NUM_CTAS
    for it in cutlass.range_constexpr(rows_per):
        m = cta * rows_per + it
        if m < M:
            for i in cutlass.range(tid, NUM_Q_HEADS * HALF, THREADS):
                h, d = i // HALF, i % HALF
                b = h * HEAD_DIM
                cs = mRope[(m, d, 0)]; sn = mRope[(m, d, 1)]
                x1 = gQ[(m, b + d)]; x2 = gQ[(m, b + d + HALF)]
                gQ[(m, b + d)] = x1 * cs - x2 * sn
                gQ[(m, b + d + HALF)] = x2 * cs + x1 * sn
            for i in cutlass.range(tid, NUM_KV_HEADS * HALF, THREADS):
                h, d = i // HALF, i % HALF
                b = h * HEAD_DIM
                cs = mRope[(m, d, 0)]; sn = mRope[(m, d, 1)]
                x1 = gK[(m, b + d)]; x2 = gK[(m, b + d + HALF)]
                gK[(m, b + d)] = x1 * cs - x2 * sn
                gK[(m, b + d + HALF)] = x2 * cs + x1 * sn
            # CRITICAL: the in-place RoPE above is written by thread i to dims
            # (b+d) and (b+d+HALF), but the copy below has thread k read dim k.
            # Those are DIFFERENT WARPS -> without this barrier a warp reads a
            # dim another warp hasn't RoPE'd yet. Symptom: dims 0-31 (warp 0's
            # own) correct, all other dims stale/un-roped.
            cute.arch.sync_threads()
            for k in cutlass.range(tid, KV_DIM, THREADS):
                gKC[(layer, m, k)] = gK[(m, k)]
                gVC[(layer, m, k)] = gV[(m, k)]
            cute.arch.sync_threads()


# ── causal attention: one warp per (q-head, query-row), online softmax ──
@cute.jit
def pf_attn(gK, gV, gQ, gA, cta, tid, layer: cutlass.Constexpr, M: cutlass.Constexpr):
    wid = tid // 32
    lane = tid % 32
    gw = cta * NUM_WARPS + wid
    ntasks = NUM_Q_HEADS * M
    for task in cutlass.range(gw, ntasks, TOTAL_WARPS):
        m = task % M
        h = task // M
        kvh = h // GROUP
        q0 = gQ[(m, h * HEAD_DIM + lane)]
        q1 = gQ[(m, h * HEAD_DIM + lane + HALF)]
        rm = cutlass.Float32(NEG); rl = cutlass.Float32(0.0)
        a0 = cutlass.Float32(0.0); a1 = cutlass.Float32(0.0)
        for j in cutlass.range(m + 1):     # causal: keys 0..m
            kk0 = gK[(j, kvh * HEAD_DIM + lane)]
            kk1 = gK[(j, kvh * HEAD_DIM + lane + HALF)]
            part = q0 * kk0 + q1 * kk1
            s = cute.arch.warp_reduction(part, lambda a, b: a + b, threads_in_group=32)
            s = s * ATTN_SCALE
            mnew = max(rm, s)
            alpha = cute.arch.exp(rm - mnew)
            p = cute.arch.exp(s - mnew)
            rl = rl * alpha + p
            vv0 = gV[(j, kvh * HEAD_DIM + lane)]
            vv1 = gV[(j, kvh * HEAD_DIM + lane + HALF)]
            a0 = a0 * alpha + p * vv0
            a1 = a1 * alpha + p * vv1
            rm = mnew
        gA[(m, h * HEAD_DIM + lane)] = a0 / rl
        gA[(m, h * HEAD_DIM + lane + HALF)] = a1 / rl


# ── fused MLP: SiLU(gate) * up -> gGu ──
@cute.jit
def pf_silu(gGu, gUp, cta, tid, M: cutlass.Constexpr):
    total = M * FFN_INTERMEDIATE
    for i in cutlass.range(cta * THREADS + tid, total, TOTAL_WARPS * 32):
        g = gGu[i]
        u = gUp[i]
        sig = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.arch.exp(-g))
        gGu[i] = g * sig * u


class PrefillMegakernel:
    @cute.kernel
    def kernel(self, mToks, mEmb, mRope,
               w_rms1, w_q, b_q, w_k, b_k, w_v, b_v, w_o, w_rms2,
               w_gate, w_up, w_down, w_fnorm, w_lm,
               gH, gX, gQ, gK, gV, gA, gGu, gUp, gKC, gVC, gBar, mLogits,
               mma, mMreal,
               M: cutlass.Constexpr, n_layers: cutlass.Constexpr):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        sR = smem.allocate_tensor(cutlass.Float32, cute.make_layout(32), byte_alignment=16)
        sA = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((BM, BK), stride=(BK, 1)), byte_alignment=16)
        sB = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((BN, BK), stride=(BK, 1)), byte_alignment=16)
        sC = smem.allocate_tensor(cutlass.Float32, cute.make_layout((BM, BN)), byte_alignment=16)

        Mreal = mMreal[0]
        rows_per: cutlass.Constexpr = (M + NUM_CTAS - 1) // NUM_CTAS

        # embed: gH[m,:] = emb[token[m]] (pad rows use token 0).
        # Fixed trip count per CTA + m < M guard (no divergent sync counts).
        for it in cutlass.range_constexpr(rows_per):
            m = cta * rows_per + it
            if m < M:
                tok = mToks[m] if m < Mreal else cutlass.Int32(0)
                for k in cutlass.range(tid, HIDDEN_DIM, THREADS):
                    gH[(m, k)] = mEmb[(tok, k)].to(cutlass.Float32)
        grid_barrier(gBar, tid, 1, NUM_CTAS)

        for layer in cutlass.range_constexpr(n_layers):
            base = layer * PF_BARRIERS + 1

            pf_rmsnorm(gH, w_rms1, gX, sR, cta, tid, layer, M, HIDDEN_DIM)
            grid_barrier(gBar, tid, base + 1, NUM_CTAS)

            pf_gemm(gX, w_q, gQ, b_q, sA, sB, sC, mma, cta, tid, layer, M, Q_DIM, HIDDEN_DIM, False, True)
            pf_gemm(gX, w_k, gK, b_k, sA, sB, sC, mma, cta, tid, layer, M, KV_DIM, HIDDEN_DIM, False, True)
            pf_gemm(gX, w_v, gV, b_v, sA, sB, sC, mma, cta, tid, layer, M, KV_DIM, HIDDEN_DIM, False, True)
            grid_barrier(gBar, tid, base + 2, NUM_CTAS)

            pf_rope_kv(gQ, gK, gV, gKC, gVC, mRope, cta, tid, layer, M)
            grid_barrier(gBar, tid, base + 3, NUM_CTAS)

            pf_attn(gK, gV, gQ, gA, cta, tid, layer, M)
            grid_barrier(gBar, tid, base + 4, NUM_CTAS)

            pf_gemm(gA, w_o, gH, b_q, sA, sB, sC, mma, cta, tid, layer, M, HIDDEN_DIM, HIDDEN_DIM, True, False)
            grid_barrier(gBar, tid, base + 5, NUM_CTAS)

            pf_rmsnorm(gH, w_rms2, gX, sR, cta, tid, layer, M, HIDDEN_DIM)
            grid_barrier(gBar, tid, base + 6, NUM_CTAS)

            pf_gemm(gX, w_gate, gGu, b_q, sA, sB, sC, mma, cta, tid, layer, M, FFN_INTERMEDIATE, HIDDEN_DIM, False, False)
            pf_gemm(gX, w_up, gUp, b_q, sA, sB, sC, mma, cta, tid, layer, M, FFN_INTERMEDIATE, HIDDEN_DIM, False, False)
            grid_barrier(gBar, tid, base + 7, NUM_CTAS)
            pf_silu(gGu, gUp, cta, tid, M)
            grid_barrier(gBar, tid, base + 8, NUM_CTAS)
            pf_gemm(gGu, w_down, gH, b_q, sA, sB, sC, mma, cta, tid, layer, M, HIDDEN_DIM, FFN_INTERMEDIATE, True, False)
            grid_barrier(gBar, tid, base + 9, NUM_CTAS)

        # final rmsnorm (last real row) + lm_head GEMV
        ml = Mreal - 1
        lane = tid % 32
        wid = tid // 32
        p = cutlass.Float32(0.0)
        for k in cutlass.range(tid, HIDDEN_DIM, THREADS):
            x = gH[(ml, k)]
            p = p + x * x
        ws = cute.arch.warp_reduction(p, lambda a, b: a + b, threads_in_group=32)
        sR[lane] = cutlass.Float32(0.0); cute.arch.sync_threads()
        sR[wid] = ws; cute.arch.sync_threads()
        tot = cute.arch.warp_reduction(sR[lane], lambda a, b: a + b, threads_in_group=32)
        inv = cute.math.rsqrt(tot / cutlass.Float32(HIDDEN_DIM) + cutlass.Float32(RMS_NORM_EPS))
        for k in cutlass.range(tid, HIDDEN_DIM, THREADS):
            gX[(0, k)] = gH[(ml, k)] * inv * w_fnorm[(0, k)].to(cutlass.Float32)
        grid_barrier(gBar, tid, n_layers * PF_BARRIERS + 2, NUM_CTAS)

        gw = cta * NUM_WARPS + wid
        for v in cutlass.range(gw, VOCAB_SIZE, TOTAL_WARPS):
            acc = cutlass.Float32(0.0)
            for k in cutlass.range(lane, HIDDEN_DIM, 32):
                acc = acc + gX[(0, k)] * w_lm[(v, k)].to(cutlass.Float32)
            acc = cute.arch.warp_reduction(acc, lambda a, b: a + b, threads_in_group=32)
            mLogits[(0, v)] = acc

    @cute.jit
    def __call__(self, mToks, mEmb, mRope,
                 w_rms1, w_q, b_q, w_k, b_k, w_v, b_v, w_o, w_rms2,
                 w_gate, w_up, w_down, w_fnorm, w_lm,
                 gH, gX, gQ, gK, gV, gA, gGu, gUp, gKC, gVC, gBar, mLogits,
                 mMreal, stream: cuda.CUstream,
                 M: cutlass.Constexpr = 32, n_layers: cutlass.Constexpr = NUM_LAYERS):
        op = warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16))
        mma = cute.make_tiled_mma(op, cute.make_layout((2, 2, 1)))
        self.kernel(mToks, mEmb, mRope,
                    w_rms1, w_q, b_q, w_k, b_k, w_v, b_v, w_o, w_rms2,
                    w_gate, w_up, w_down, w_fnorm, w_lm,
                    gH, gX, gQ, gK, gV, gA, gGu, gUp, gKC, gVC, gBar, mLogits,
                    mma, mMreal, M, n_layers).launch(
            grid=[NUM_CTAS, 1, 1], block=[THREADS, 1, 1], stream=stream)


PF_BARRIERS_TOTAL = NUM_LAYERS * PF_BARRIERS + 2
