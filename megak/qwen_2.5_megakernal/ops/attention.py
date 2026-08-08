import cutlass
from config import (
    GROUP,
    HALF,
    HEAD_DIM,
    NUM_CTAS,
    NUM_KV_HEADS,
    NUM_Q_HEADS,
    NUM_THREADS,
    SCALE,
)
from cutlass import cute

THREADS = NUM_THREADS
NUM_WARPS = NUM_THREADS // 32
TOTAL_WARPS = NUM_CTAS * NUM_WARPS
NEG_INF = -3.0e38


@cute.jit
def rope(sQ, sK, mRope, tid, position):
    """
    Apply RoPE in-place to Q [Q_DIM] and K [KV_DIM] in SMEM.
    mRope: [MAX_SEQ, HALF, 2] — [...,0]=cos, [...,1]=sin, indexed by freq d.
    `position` is RUNTIME so decode works at any step.

    CONVENTION: HF Llama/Qwen2 use `rotate_half`, NOT interleaved pairs.
        x1 = x[0:32], x2 = x[32:64]
        out[d]    = x1[d]*cos[d] - x2[d]*sin[d]
        out[d+32] = x2[d]*cos[d] + x1[d]*sin[d]
    Using the GPT-NeoX interleaved (2i, 2i+1) pairing instead gives
    plausible-but-wrong attention (degenerate/repetitive text).
    Each thread owns one (d, d+32) pair, so the in-place update is race-free.
    """
    for i in cutlass.range(tid, NUM_Q_HEADS * HALF, THREADS):
        head = i // HALF
        d = i % HALF
        base = head * HEAD_DIM
        cos = mRope[(position, d, 0)]
        sin = mRope[(position, d, 1)]
        x1 = sQ[base + d]
        x2 = sQ[base + d + HALF]
        sQ[base + d] = x1 * cos - x2 * sin
        sQ[base + d + HALF] = x2 * cos + x1 * sin
    for i in cutlass.range(tid, NUM_KV_HEADS * HALF, THREADS):
        head = i // HALF
        d = i % HALF
        base = head * HEAD_DIM
        cos = mRope[(position, d, 0)]
        sin = mRope[(position, d, 1)]
        x1 = sK[base + d]
        x2 = sK[base + d + HALF]
        sK[base + d] = x1 * cos - x2 * sin
        sK[base + d + HALF] = x2 * cos + x1 * sin


@cute.jit
def kv_cache_write(sK, sV, gKC, gVC, tid, layer: cutlass.Constexpr, position):
    """
    Append this token's K and V to the cache.
    gKC/gVC: [NUM_LAYERS, MAX_SEQ, KV_DIM] flattened as [L, S, KV_DIM].
    Only KV_DIM=128 elements per tensor — cheap.
    """
    for i in cutlass.range(tid, NUM_KV_HEADS * HEAD_DIM, THREADS):
        gKC[(layer, position, i)] = sK[i]
        gVC[(layer, position, i)] = sV[i]


@cute.jit
def gqa_attention(sQ, gKC, gVC, sA, tid, layer: cutlass.Constexpr, seq_len):
    """
    Real GQA with online softmax over cache positions 0..seq_len-1.

    sQ:  [Q_DIM] this token's Q (RoPE applied)
    gKC: [L, S, KV_DIM] key cache
    gVC: [L, S, KV_DIM] value cache
    sA:  [Q_DIM] output in SMEM (written)
    seq_len: RUNTIME valid cache length (= position + 1)

    Computed REDUNDANTLY by every CTA: one warp per Q head, striding by this
    CTA's warp count (16 warps >= 14 heads, so warps 0..13 each take one head).
    This is required because sA lives in per-CTA SMEM — splitting heads across
    the whole grid would leave 19 of 20 CTAs with an unwritten sA.

    Within a head, lane l owns output dims l and l+32. The score is reduced
    across all 32 lanes so every lane sees the same value, keeping the running
    max / denominator consistent.
    """
    warp = tid // 32
    lane = tid % 32

    for h in cutlass.range(warp, NUM_Q_HEADS, NUM_WARPS):
        kv_h = h // GROUP
        q_base = h * HEAD_DIM
        kv_base = kv_h * HEAD_DIM

        # running online-softmax state; each lane owns dims lane, lane+32
        m = cutlass.Float32(NEG_INF)
        l = cutlass.Float32(0.0)
        acc0 = cutlass.Float32(0.0)     # dim = lane
        acc1 = cutlass.Float32(0.0)     # dim = lane + 32

        for j in cutlass.range(seq_len):
            # ---- score s = (q · k) * SCALE, cooperatively across 32 lanes ----
            part = cutlass.Float32(0.0)
            for d in cutlass.range(lane, HEAD_DIM, 32):
                part = part + sQ[q_base + d] * gKC[(layer, j, kv_base + d)]
            s = cute.arch.warp_reduction(part, lambda a, b: a + b,
                                         threads_in_group=32)
            s = s * cutlass.Float32(SCALE)

            # ---- online softmax update (identical in every lane) ----
            m_new = cute.arch.fmax(m, s)
            alpha = cute.arch.exp(m - m_new)      # rescale old state
            p = cute.arch.exp(s - m_new)          # weight of this key
            l = l * alpha + p
            m = m_new

            # ---- accumulate p * V for this lane's two dims ----
            acc0 = acc0 * alpha + p * gVC[(layer, j, kv_base + lane)]
            acc1 = acc1 * alpha + p * gVC[(layer, j, kv_base + lane + 32)]

        # ---- normalize and write out ----
        inv = cutlass.Float32(1.0) / l
        sA[q_base + lane] = acc0 * inv
        sA[q_base + lane + 32] = acc1 * inv
