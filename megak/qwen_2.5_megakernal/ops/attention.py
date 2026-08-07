"""RoPE + GQA attention (decode, seq_len=1) — @cute.jit.

RoPE: rotates (2i, 2i+1) pairs within each head. Identity when position=0.
GQA decode: with a single key (self), softmax over 1 element = 1.0, so the
attention output equals V, broadcast from each KV head to its GROUP Q heads.
Full online-softmax over a KV cache comes with the prefill/decode-with-cache
version (see general_kernals/flash_attention_tc_cute.py for the pattern).
"""

import cutlass
from config import (
    GROUP,
    HALF,
    HEAD_DIM,
    NUM_KV_HEADS,
    NUM_Q_HEADS,
    NUM_THREADS,
    Q_DIM,
)
from cutlass import cute

THREADS = NUM_THREADS


@cute.jit
def rope(sQ, sK, mRope, tid, position: cutlass.Constexpr):
    """
    Apply RoPE to Q [Q_DIM] and K [KV_DIM] in SMEM.
    mRope: [MAX_SEQ, HALF, 2] — [...,0]=cos, [...,1]=sin.
    """
    for i in cutlass.range(tid, NUM_Q_HEADS * HALF, THREADS):
        head = i // HALF
        pair = i % HALF
        base = head * HEAD_DIM
        cos = mRope[(position, pair, 0)]
        sin = mRope[(position, pair, 1)]
        x = sQ[base + 2 * pair]
        y = sQ[base + 2 * pair + 1]
        sQ[base + 2 * pair]     = x * cos - y * sin
        sQ[base + 2 * pair + 1] = y * cos + x * sin
    for i in cutlass.range(tid, NUM_KV_HEADS * HALF, THREADS):
        head = i // HALF
        pair = i % HALF
        base = head * HEAD_DIM
        cos = mRope[(position, pair, 0)]
        sin = mRope[(position, pair, 1)]
        x = sK[base + 2 * pair]
        y = sK[base + 2 * pair + 1]
        sK[base + 2 * pair]     = x * cos - y * sin
        sK[base + 2 * pair + 1] = y * cos + x * sin


@cute.jit
def gqa_decode_first(sV, sOut, tid):
    """
    First-token GQA (seq_len=1): out = V broadcast across each KV head's group.
    Q head h uses KV head h // GROUP.
    """
    for i in cutlass.range(tid, Q_DIM, THREADS):
        head = i // HEAD_DIM
        d = i % HEAD_DIM
        sOut[i] = sV[(head // GROUP) * HEAD_DIM + d]
