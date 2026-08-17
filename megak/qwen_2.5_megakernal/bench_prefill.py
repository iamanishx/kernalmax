"""Prefill vs decode throughput — ONE M bucket (prefill compile is ~210s/bucket).

Prefill is BATCHED (weights read once for all M tokens), decode is a GEMV per
token (weights read every token). So prefill tok/s should be far higher.

Usage: python3 bench_prefill.py [n_prompt_tokens]
"""
import os
import sys

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from megakernel import MegakernelEngine
from prefill_megakernel import BM as PF_BM


def timeit(fn, iters, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    eng = MegakernelEngine(verbose=True)
    ids = [100 + (i % 900) for i in range(n)]
    M = ((n + PF_BM - 1) // PF_BM) * PF_BM

    eng._build_prefill(M)                 # compile ONCE, outside timing
    print()

    eng.reset(); eng.prefill_mk(ids)
    dec_ms = timeit(lambda: eng.step(500, n), 30)
    eng.reset()
    pre_ms = timeit(lambda: eng.prefill_mk(ids), 10)

    dec_tps = 1000 / dec_ms
    pre_tps = n / (pre_ms / 1e3)
    print("=" * 60)
    print(f"{'phase':<12}{'ms/launch':>12}{'tokens':>9}{'tok/s':>12}")
    print("-" * 60)
    print(f"{'PREFILL':<12}{pre_ms:>12.2f}{n:>9}{pre_tps:>12.1f}")
    print(f"{'DECODE':<12}{dec_ms:>12.2f}{1:>9}{dec_tps:>12.1f}")
    print("-" * 60)
    print(f"=> prefill throughput is {pre_tps/dec_tps:.1f}x decode")
    print()
    print("WHY: prefill = ONE GEMM pass over M tokens (weights read once).")
    print("     decode  = one GEMV per token (weights read every token).")
    print("     Prefill arithmetic intensity scales with M; decode is pinned ~1.")


if __name__ == "__main__":
    main()
