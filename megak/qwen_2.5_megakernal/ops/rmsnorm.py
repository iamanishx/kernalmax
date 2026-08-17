import cutlass
from config import NUM_THREADS, RMS_NORM_EPS
from cutlass import cute

from ops.reduce import block_reduce_sum

THREADS = NUM_THREADS


@cute.jit
def rmsnorm(sIn, w_all, sOut, sR, tid, layer: cutlass.Constexpr, D: cutlass.Constexpr):
    """
    sOut = (sIn / rms(sIn)) * w_all[layer]
    w_all: [L, 1, D]  (per-layer RMS weight stacked)
    """
    p = cutlass.Float32(0.0)
    for i in cutlass.range(tid, D, THREADS):
        x = sIn[i]
        p = p + x * x
    tot = block_reduce_sum(p, sR, tid)
    rms = cute.math.rsqrt(tot / cutlass.Float32(D) + cutlass.Float32(RMS_NORM_EPS))
    for i in cutlass.range(tid, D, THREADS):
        sOut[i] = sIn[i] * rms * w_all[(layer, 0, i)].to(cutlass.Float32)


@cute.jit
def rmsnorm_final(sIn, w_fnorm, sOut, sR, tid, D: cutlass.Constexpr):
    """Final RMSNorm. w_fnorm: [1, D]."""
    p = cutlass.Float32(0.0)
    for i in cutlass.range(tid, D, THREADS):
        x = sIn[i]
        p = p + x * x
    tot = block_reduce_sum(p, sR, tid)
    rms = cute.math.rsqrt(tot / cutlass.Float32(D) + cutlass.Float32(RMS_NORM_EPS))
    for i in cutlass.range(tid, D, THREADS):
        sOut[i] = sIn[i] * rms * w_fnorm[(0, i)].to(cutlass.Float32)
