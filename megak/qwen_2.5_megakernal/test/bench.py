"""A/B benchmark: megakernel vs HuggingFace, single-stream decode.

Fair comparison:
  * same model (Qwen2.5-0.5B-Instruct), same precision (bf16 weights)
  * both use a KV cache, both do single-token decode steps
  * both measured at the same sequence positions
  * benchmarked sequentially and freed in between (6 GB card)

HF eager is the common baseline. HF + torch.compile + static cache would be
faster than eager — noted in the output, not measured here.

Usage: python3 bench.py [n_decode_steps]
"""
import gc
import os
import sys

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT_TOKENS = 16


def timer(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms per call


def bench_megakernel(n_steps):
    from megakernel import MegakernelEngine
    print("─" * 62)
    print("MEGAKERNEL (CuTe DSL, one launch, 20 CTAs)")
    print("─" * 62)
    eng = MegakernelEngine(verbose=True)
    ids = list(range(100, 100 + PROMPT_TOKENS))

    eng.reset()
    eng.prefill(ids)
    pos = len(ids)

    # steady-state decode: one step at a fixed position (reset each call so the
    # cache length stays constant across timed iterations)
    def one_step():
        eng.step(500, pos)

    ms = timer(one_step, n_steps)
    weight_bytes = eng.weight_bytes
    del eng
    gc.collect()
    torch.cuda.empty_cache()
    return ms, weight_bytes


def bench_hf(n_steps):
    from transformers import AutoModelForCausalLM
    print("─" * 62)
    print("HUGGINGFACE eager (bf16, KV cache)")
    print("─" * 62)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, local_files_only=True).cuda().eval()
    wb = sum(p.numel() * p.element_size() for p in m.parameters())
    print(f"    weights {wb/1024**3:.2f} GB (bf16)")

    ids = torch.tensor([list(range(100, 100 + PROMPT_TOKENS))], device="cuda")
    with torch.no_grad():
        out = m(ids, use_cache=True)
    past = out.past_key_values
    nxt = torch.tensor([[500]], device="cuda")

    def one_step():
        with torch.no_grad():
            m(nxt, past_key_values=past, use_cache=True)

    ms = timer(one_step, n_steps)
    del m, past, out
    gc.collect()
    torch.cuda.empty_cache()
    return ms, wb


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    p = torch.cuda.get_device_properties(0)
    print("=" * 62)
    print(f"Single-stream decode benchmark — {p.name}")
    print(f"model {MODEL_ID}, prompt {PROMPT_TOKENS} tok, {n} timed steps")
    print("=" * 62)

    mk_ms, mk_bytes = bench_megakernel(n)
    hf_ms, hf_bytes = bench_hf(n)

    print()
    print("=" * 62)
    print("RESULTS (single-token decode, batch=1)")
    print("=" * 62)
    print(f"{'impl':<22}{'ms/tok':>10}{'tok/s':>10}{'GB/s':>10}")
    print("-" * 62)
    for name, ms, wb in (("megakernel (CuTe)", mk_ms, mk_bytes),
                         ("HF eager (bf16)", hf_ms, hf_bytes)):
        print(f"{name:<22}{ms:>10.2f}{1000/ms:>10.1f}{wb/(ms*1e-3)/1e9:>10.1f}")
    print("-" * 62)
    if hf_ms > mk_ms:
        print(f"megakernel is {hf_ms/mk_ms:.2f}x FASTER than HF eager")
    else:
        print(f"megakernel is {mk_ms/hf_ms:.2f}x SLOWER than HF eager")
    print()
    print(f"HBM roofline @ {mk_bytes/1024**3:.2f} GB/tok, ~216 GB/s peak:")
    print(f"  floor = {mk_bytes/216e9*1e3:.2f} ms/tok = {216e9/mk_bytes:.0f} tok/s")
    print()
    print("caveats: HF eager is the baseline; HF + torch.compile + static cache")
    print("would close much of the gap. At batch>1 HF/vLLM win decisively —")
    print("this kernel is hardcoded M=1 and cannot batch.")


if __name__ == "__main__":
    main()
