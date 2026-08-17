"""FINAL end-to-end: hybrid megakernel vs pure HuggingFace.

  HYBRID (ours):  prefill = torch batched (read-once) -> KV loaded into our
                  cache -> decode = megakernel (one launch/token, 95% roofline)
  HF BASELINE:    prefill + decode both in transformers eager (KV cache)

Same model (bf16), same prompt, greedy, same stop tokens. Both warmed.
Reported: TTFT (time to first generated token), decode tok/s, total latency.

Usage: python3 bench_final.py ["prompt"] [max_new]
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


def run_hybrid(prompt, max_new):
    from megakernel import MegakernelEngine
    eng = MegakernelEngine(verbose=True)
    ids = chat_ids(eng.tok, prompt)
    # WARM: compile the prefill megakernel for this padded M + warm decode
    eng.reset(); eng.prefill_mk(ids); eng.step(5, len(ids)); torch.cuda.synchronize()

    eng.reset(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    logits = eng.prefill_mk(ids)
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0
    out, pos = [], len(ids)
    for _ in range(max_new):
        nxt = int(logits.argmax())
        if nxt in eng.stop_ids:
            break
        out.append(nxt)
        logits = eng.step(nxt, pos); pos += 1
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    res = dict(text=eng.tok.decode(out), n_prompt=len(ids), n_new=len(out),
               ttft=ttft, total=total, decode_tok_s=len(out)/(total-ttft) if total > ttft else 0)
    del eng
    gc.collect(); torch.cuda.empty_cache()
    return res


def run_hf(prompt, max_new):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, local_files_only=True).cuda().eval()
    stops = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    ids = chat_ids(tok, prompt)
    inp = torch.tensor([ids], device="cuda")
    with torch.no_grad():                          # warm
        m(inp, use_cache=True)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        o = m(inp, use_cache=True)
    past = o.past_key_values
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0
    out = []
    logits = o.logits[:, -1, :]
    for _ in range(max_new):
        nxt = int(logits.argmax())
        if nxt in stops:
            break
        out.append(nxt)
        with torch.no_grad():
            o = m(torch.tensor([[nxt]], device="cuda"), past_key_values=past, use_cache=True)
        past = o.past_key_values
        logits = o.logits[:, -1, :]
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    res = dict(text=tok.decode(out), n_prompt=len(ids), n_new=len(out),
               ttft=ttft, total=total, decode_tok_s=len(out)/(total-ttft) if total > ttft else 0)
    del m, past
    gc.collect(); torch.cuda.empty_cache()
    return res


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Write one sentence about GPUs."
    max_new = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    print("=" * 68)
    print(f"FINAL END-TO-END — {MODEL_ID} | greedy | max_new={max_new}")
    print(f"prompt: {prompt!r}")
    print("=" * 68)

    hy = run_hybrid(prompt, max_new)
    hf = run_hf(prompt, max_new)

    print(f"\nMEGAKERNEL : {hy['text']}")
    print(f"HF     : {hf['text']}")
    npref = sum(1 for a, b in zip(hy["text"], hf["text"]) if a == b)

    print()
    print("=" * 68)
    print(f"{'impl':<22}{'TTFT ms':>10}{'decode tok/s':>14}{'TOTAL ms':>11}")
    print("-" * 68)
    for name, r in (("MEGAKERNEL (ours)", hy), ("HF eager", hf)):
        print(f"{name:<22}{r['ttft']*1e3:>10.1f}{r['decode_tok_s']:>14.1f}{r['total']*1e3:>11.0f}")
    print("-" * 68)
    hy_pt = hy["total"]/max(hy["n_new"],1); hf_pt = hf["total"]/max(hf["n_new"],1)
    print(f"per-token total: hybrid {hy_pt*1e3:.1f} ms | HF {hf_pt*1e3:.1f} ms")
    print(f"TTFT   : {hf['ttft']/hy['ttft']:.2f}x {'FASTER' if hy['ttft']<hf['ttft'] else 'slower'} (hybrid vs HF)")
    print(f"decode : {hy['decode_tok_s']/hf['decode_tok_s']:.2f}x FASTER (hybrid vs HF)")
    print(f"total  : {hf_pt/hy_pt:.2f}x FASTER (hybrid vs HF, per generated token)")


if __name__ == "__main__":
    main()
