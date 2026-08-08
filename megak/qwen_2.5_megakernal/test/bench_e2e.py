"""End-to-end latency: same prompt → same generated tokens, megakernel vs HF.

Measures total wall-clock to produce a response (prefill + all decode steps),
which is what a user actually waits for. Both:
  * same model (Qwen2.5-0.5B-Instruct), same bf16 weights
  * same prompt, same chat template
  * greedy decoding, same token budget
  * KV cache on both sides

Reports TTFT (time to first token), total latency, and output tok/s.
Run sequentially and freed in between (6 GB card).

Usage: python3 bench_e2e.py ["prompt"] [max_new_tokens]
"""
import gc
import os
import sys
import time

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def chat_ids(tok, prompt):
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, tokenize=True)
    if isinstance(enc, dict) or hasattr(enc, "input_ids"):
        enc = enc["input_ids"]
    if len(enc) and isinstance(enc[0], (list, tuple)):
        enc = enc[0]
    return [int(t) for t in enc]


def run_megakernel(prompt, max_new):
    from megakernel import MegakernelEngine
    print("─" * 64)
    print("MEGAKERNEL (CuTe DSL — one launch per token)")
    print("─" * 64)
    eng = MegakernelEngine(verbose=True)
    ids = chat_ids(eng.tok, prompt)

    # WARM UP (match the HF path): first launch pays CUDA module load,
    # cold L2 and first-touch page faults — not steady-state cost.
    eng.reset()
    eng.prefill(ids[:4])
    eng.step(500, 4)
    torch.cuda.synchronize()

    eng.reset()
    torch.cuda.synchronize()
    t_start = time.perf_counter()

    # prefill
    logits = eng.prefill(ids)
    torch.cuda.synchronize()
    t_prefill_done = time.perf_counter()

    # decode
    out, pos = [], len(ids)
    ttft = None
    for _ in range(max_new):
        nxt = int(logits.argmax().item())
        if ttft is None:
            torch.cuda.synchronize()
            ttft = time.perf_counter() - t_start
        if nxt in eng.stop_ids:
            break
        out.append(nxt)
        logits = eng.step(nxt, pos)
        pos += 1
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    text = eng.tok.decode(out)
    res = dict(text=text, ids=out, n_prompt=len(ids), n_new=len(out),
               ttft=ttft, total=t_end - t_start,
               prefill=t_prefill_done - t_start,
               decode=t_end - t_prefill_done)
    del eng
    gc.collect(); torch.cuda.empty_cache()
    return res


def run_hf(prompt, max_new):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("─" * 64)
    print("HUGGINGFACE eager (bf16, KV cache)")
    print("─" * 64)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, local_files_only=True).cuda().eval()
    print(f"    weights {sum(p.numel()*p.element_size() for p in m.parameters())/1024**3:.2f} GB")

    ids = chat_ids(tok, prompt)
    inp = torch.tensor([ids], device="cuda")
    stops = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}

    # warm up kernels/autotune so we don't measure first-call overhead
    with torch.no_grad():
        m(inp, use_cache=True)
    torch.cuda.synchronize()

    t_start = time.perf_counter()
    with torch.no_grad():
        o = m(inp, use_cache=True)
    past = o.past_key_values
    torch.cuda.synchronize()
    t_prefill_done = time.perf_counter()

    out, ttft = [], None
    logits = o.logits[:, -1, :]
    for _ in range(max_new):
        nxt = int(logits.argmax().item())
        if ttft is None:
            torch.cuda.synchronize()
            ttft = time.perf_counter() - t_start
        if nxt in stops:
            break
        out.append(nxt)
        with torch.no_grad():
            o = m(torch.tensor([[nxt]], device="cuda"),
                  past_key_values=past, use_cache=True)
        past = o.past_key_values
        logits = o.logits[:, -1, :]
    torch.cuda.synchronize()
    t_end = time.perf_counter()

    res = dict(text=tok.decode(out), ids=out, n_prompt=len(ids), n_new=len(out),
               ttft=ttft, total=t_end - t_start,
               prefill=t_prefill_done - t_start,
               decode=t_end - t_prefill_done)
    del m, past, o
    gc.collect(); torch.cuda.empty_cache()
    return res


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Write one sentence about GPUs."
    max_new = int(sys.argv[2]) if len(sys.argv) > 2 else 64

    p = torch.cuda.get_device_properties(0)
    print("=" * 64)
    print(f"END-TO-END LATENCY — {p.name}")
    print(f"model {MODEL_ID} | greedy | max_new={max_new}")
    print(f"prompt: {prompt!r}")
    print("=" * 64)

    mk = run_megakernel(prompt, max_new)
    hf = run_hf(prompt, max_new)

    print()
    print("=" * 64)
    print("OUTPUTS")
    print("=" * 64)
    print(f"megakernel ({mk['n_new']} tok): {mk['text']}")
    print()
    print(f"HF eager   ({hf['n_new']} tok): {hf['text']}")
    same = mk["ids"] == hf["ids"]
    npref = sum(1 for a, b in zip(mk["ids"], hf["ids"]) if a == b)
    print(f"\nidentical output: {same}"
          f"  (matching leading tokens: {npref}/{min(len(mk['ids']), len(hf['ids']))})")

    print()
    print("=" * 64)
    print("LATENCY (wall clock, prompt in → response out)")
    print("=" * 64)
    hdr = f"{'impl':<20}{'TTFT ms':>10}{'prefill ms':>12}{'decode ms':>11}{'TOTAL ms':>10}{'out tok/s':>11}"
    print(hdr); print("-" * 64)
    for name, r in (("megakernel", mk), ("HF eager", hf)):
        tps = r["n_new"] / r["decode"] if r["decode"] > 0 else 0
        print(f"{name:<20}{r['ttft']*1e3:>10.0f}{r['prefill']*1e3:>12.0f}"
              f"{r['decode']*1e3:>11.0f}{r['total']*1e3:>10.0f}{tps:>11.1f}")
    print("-" * 64)

    # normalize by tokens produced, since outputs may differ in length
    mk_per = mk["total"] / max(mk["n_new"], 1)
    hf_per = hf["total"] / max(hf["n_new"], 1)
    print(f"per generated token: megakernel {mk_per*1e3:.1f} ms | "
          f"HF {hf_per*1e3:.1f} ms")
    if hf_per > mk_per:
        print(f"→ megakernel {hf_per/mk_per:.2f}x FASTER end-to-end (per token)")
    else:
        print(f"→ megakernel {mk_per/hf_per:.2f}x SLOWER end-to-end (per token)")

    print()
    print("note: our prefill is token-by-token (M=1); HF batches the whole")
    print("prompt into one GEMM, so HF wins prefill/TTFT and we win decode.")


if __name__ == "__main__":
    main()
