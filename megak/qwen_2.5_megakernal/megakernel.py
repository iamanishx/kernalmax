"""
Qwen2.5-0.5B PERSISTENT MEGAKERNEL — full 24-layer decode in ONE launch,
across all 20 SMs of an RTX 4050.

ARCHITECTURE
------------
grid = [NUM_CTAS] (= #SMs) so every SM holds one resident CTA for the whole
forward pass. There are no kernel boundaries: 24 layers + LM head run inside a
single launch.

Work split:
  * Big GEMVs (Q/K/V/O/gate/up/down/lm_head) — split by output row across all
    320 warps in the grid. Results land in GLOBAL memory; a grid_barrier
    follows so every CTA sees the complete vector.
  * Small ops (rmsnorm/rope/gqa) — computed REDUNDANTLY by each CTA in its own
    SMEM. 896 elements of duplicate math is far cheaper than a barrier.

Cross-CTA sync uses an atomic counter in global memory (sync_threads only works
*within* a CTA). Safe because grid <= #SMs guarantees all CTAs are co-resident.

Optimizations applied:
  1. warp-per-row GEMV      → coalesced weight loads (was 32x uncoalesced)
  2. bf16 big weights       → halves HBM bytes on a bandwidth-bound decode
  3. 512 threads/CTA        → 16 warps, more memory requests in flight
  4. persistent 20-CTA grid → uses all SMs instead of 1
  5. fused gate+up+SiLU     → one pass, one SMEM buffer instead of two

Correctness: matches HuggingFace top-1 on every token tested.

NOTE: Q/K are computed and RoPE'd but unused at seq_len=1 (softmax over a
single key = 1, so attention output = V). Kept for structural correctness and
because a KV-cache version needs them.

Usage:  python megakernel.py
"""

import math
import os

import cuda.bindings.driver as cuda
import cutlass
import torch
from config import (
    FFN_INTERMEDIATE,
    HALF,
    HEAD_DIM,
    HIDDEN_DIM,
    KV_DIM,
    NUM_CTAS,
    NUM_LAYERS,
    NUM_THREADS,
    Q_DIM,
    ROPE_THETA,
    VOCAB_SIZE,
)
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from ops import (
    g2s,
    gemv_bias_mc,
    gemv_residual_mc,
    gqa_decode_first,
    grid_barrier,
    lm_head_mc,
    mlp_gate_silu_mc,
    rmsnorm,
    rmsnorm_final,
    rope,
)

THREADS = NUM_THREADS
BARRIERS_PER_LAYER = 4


class Qwen25Megakernel:
    @cute.kernel
    def kernel(self, mTok, mEmb, mRope,
               w_rms1, w_q, b_q, w_k, b_k, w_v, b_v, w_o, w_rms2,
               w_gate, w_up, w_down, w_fnorm, w_lm,
               gH, gQ, gK, gV, gG, gBar, mLogits,
               position: cutlass.Constexpr, n_layers: cutlass.Constexpr):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()

        # per-CTA SMEM working set (~35 KB of the ~99 KB budget)
        sH = smem.allocate_tensor(cutlass.Float32, cute.make_layout(HIDDEN_DIM), byte_alignment=16)
        sN = smem.allocate_tensor(cutlass.Float32, cute.make_layout(HIDDEN_DIM), byte_alignment=16)
        sQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout(Q_DIM), byte_alignment=16)
        sK = smem.allocate_tensor(cutlass.Float32, cute.make_layout(KV_DIM), byte_alignment=16)
        sV = smem.allocate_tensor(cutlass.Float32, cute.make_layout(KV_DIM), byte_alignment=16)
        sA = smem.allocate_tensor(cutlass.Float32, cute.make_layout(HIDDEN_DIM), byte_alignment=16)
        sG = smem.allocate_tensor(cutlass.Float32, cute.make_layout(FFN_INTERMEDIATE), byte_alignment=16)
        sR = smem.allocate_tensor(cutlass.Float32, cute.make_layout(32), byte_alignment=16)

        # ── embedding: every CTA loads the token row into its own SMEM, and
        #    all CTAs write the same values to gH (idempotent, no race) ──
        token = mTok[0]
        for i in cutlass.range(tid, HIDDEN_DIM, THREADS):
            e = mEmb[(token, i)].to(cutlass.Float32)
            sH[i] = e
            gH[i] = e
        grid_barrier(gBar, tid, 1, NUM_CTAS)

        for layer in cutlass.range_constexpr(n_layers):
            base = layer * BARRIERS_PER_LAYER + 1

            # everyone refreshes its SMEM copy of the hidden state
            g2s(gH, sH, tid, HIDDEN_DIM)
            cute.arch.sync_threads()

            # ═══ attention ═══
            rmsnorm(sH, w_rms1, sN, sR, tid, layer, HIDDEN_DIM)   # redundant
            cute.arch.sync_threads()

            gemv_bias_mc(sN, w_q, b_q, gQ, tid, cta, layer, HIDDEN_DIM, HIDDEN_DIM)
            gemv_bias_mc(sN, w_k, b_k, gK, tid, cta, layer, HIDDEN_DIM, KV_DIM)
            gemv_bias_mc(sN, w_v, b_v, gV, tid, cta, layer, HIDDEN_DIM, KV_DIM)
            grid_barrier(gBar, tid, base + 1, NUM_CTAS)

            g2s(gQ, sQ, tid, Q_DIM)
            g2s(gK, sK, tid, KV_DIM)
            g2s(gV, sV, tid, KV_DIM)
            cute.arch.sync_threads()

            rope(sQ, sK, mRope, tid, position)                     # redundant
            cute.arch.sync_threads()
            gqa_decode_first(sV, sA, tid)                          # redundant
            cute.arch.sync_threads()

            gemv_residual_mc(sA, w_o, gH, tid, cta, layer, HIDDEN_DIM, HIDDEN_DIM)
            grid_barrier(gBar, tid, base + 2, NUM_CTAS)

            # ═══ MLP ═══
            g2s(gH, sH, tid, HIDDEN_DIM)
            cute.arch.sync_threads()
            rmsnorm(sH, w_rms2, sN, sR, tid, layer, HIDDEN_DIM)    # redundant
            cute.arch.sync_threads()

            mlp_gate_silu_mc(sN, w_gate, w_up, gG, tid, cta, layer,
                             HIDDEN_DIM, FFN_INTERMEDIATE)
            grid_barrier(gBar, tid, base + 3, NUM_CTAS)

            g2s(gG, sG, tid, FFN_INTERMEDIATE)
            cute.arch.sync_threads()
            gemv_residual_mc(sG, w_down, gH, tid, cta, layer,
                             FFN_INTERMEDIATE, HIDDEN_DIM)
            grid_barrier(gBar, tid, base + 4, NUM_CTAS)

        # ═══ final norm + LM head ═══
        g2s(gH, sH, tid, HIDDEN_DIM)
        cute.arch.sync_threads()
        rmsnorm_final(sH, w_fnorm, sN, sR, tid, HIDDEN_DIM)        # redundant
        cute.arch.sync_threads()
        lm_head_mc(sN, w_lm, mLogits, tid, cta, HIDDEN_DIM, VOCAB_SIZE)

    @cute.jit
    def __call__(self, mTok, mEmb, mRope,
                 w_rms1, w_q, b_q, w_k, b_k, w_v, b_v, w_o, w_rms2,
                 w_gate, w_up, w_down, w_fnorm, w_lm,
                 gH, gQ, gK, gV, gG, gBar, mLogits, stream: cuda.CUstream,
                 position: cutlass.Constexpr = 0,
                 n_layers: cutlass.Constexpr = NUM_LAYERS):
        # NOTE: Constexpr args must come from defaults, never after `stream`
        # in the call tuple — doing so silently corrupts the launch.
        self.kernel(mTok, mEmb, mRope,
                    w_rms1, w_q, b_q, w_k, b_k, w_v, b_v, w_o, w_rms2,
                    w_gate, w_up, w_down, w_fnorm, w_lm,
                    gH, gQ, gK, gV, gG, gBar, mLogits,
                    position, n_layers).launch(
            grid=[NUM_CTAS, 1, 1], block=[THREADS, 1, 1], stream=stream)


# ═══════════════════════════ host ═══════════════════════════

def build_rope_table(max_seq, device="cuda"):
    t = torch.zeros(max_seq, HALF, 2, device=device, dtype=torch.float32)
    for pair in range(HALF):
        theta = 1.0 / (ROPE_THETA ** (2.0 * pair / HEAD_DIM))
        for pos in range(max_seq):
            t[pos, pair, 0] = math.cos(theta * pos)
            t[pos, pair, 1] = math.sin(theta * pos)
    return t


def _load_hf(model_id="Qwen/Qwen2.5-0.5B"):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32,
                                             local_files_only=True).eval()
    sd = {k: v.detach().clone() for k, v in m.named_parameters()}

    def st(key):
        return torch.stack([sd[key.format(i)] for i in range(NUM_LAYERS)])

    W = {
        "rms1": st("model.layers.{}.input_layernorm.weight").unsqueeze(1),
        "q":    st("model.layers.{}.self_attn.q_proj.weight"),
        "bq":   st("model.layers.{}.self_attn.q_proj.bias").unsqueeze(1),
        "k":    st("model.layers.{}.self_attn.k_proj.weight"),
        "bk":   st("model.layers.{}.self_attn.k_proj.bias").unsqueeze(1),
        "v":    st("model.layers.{}.self_attn.v_proj.weight"),
        "bv":   st("model.layers.{}.self_attn.v_proj.bias").unsqueeze(1),
        "o":    st("model.layers.{}.self_attn.o_proj.weight"),
        "rms2": st("model.layers.{}.post_attention_layernorm.weight").unsqueeze(1),
        "gate": st("model.layers.{}.mlp.gate_proj.weight"),
        "up":   st("model.layers.{}.mlp.up_proj.weight"),
        "down": st("model.layers.{}.mlp.down_proj.weight"),
        "embed": sd["model.embed_tokens.weight"],
        "fnorm": sd["model.norm.weight"].unsqueeze(0),
        "lm":   sd.get("lm_head.weight", sd["model.embed_tokens.weight"]),
    }
    return W, m


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    print("=" * 62)
    print("Qwen2.5-0.5B PERSISTENT Megakernel — 24 layers, 1 launch")
    print("=" * 62)
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | SMs: {props.multi_processor_count} | sm_{props.major}{props.minor}")
    print(f"grid=[{NUM_CTAS}] block=[{THREADS}] → {NUM_CTAS*THREADS//32} warps total")
    dev = torch.device("cuda")

    print("\n[1] Loading HF weights (offline)...")
    W, hf = _load_hf()
    for k in W:
        W[k] = W[k].to(dev).contiguous()
    if os.environ.get("BF16", "1") == "1":
        for k in ["q", "k", "v", "o", "gate", "up", "down", "embed", "lm"]:
            W[k] = W[k].to(torch.bfloat16).contiguous()
    gb = sum(v.numel() * v.element_size() for v in W.values()) / 1024**3
    print(f"    weight bytes: {gb:.2f} GB "
          f"({'bf16 big / fp32 norms' if os.environ.get('BF16','1')=='1' else 'fp32'})")

    print("[2] RoPE table + global activation buffers...")
    rope_t = build_rope_table(128, dev)
    z = lambda n: torch.zeros(n, device=dev, dtype=torch.float32)
    gH, gQ, gK, gV, gG = z(HIDDEN_DIM), z(Q_DIM), z(KV_DIM), z(KV_DIM), z(FFN_INTERMEDIATE)
    gBar = torch.zeros(1, device=dev, dtype=torch.int32)
    mLogits = torch.zeros(1, VOCAB_SIZE, device=dev, dtype=torch.float32)
    mTok = torch.zeros(1, device=dev, dtype=torch.int32)

    def ct(t): return from_dlpack(t, assumed_align=16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    mk = Qwen25Megakernel()

    print("[3] Compiling...")
    import time
    t0 = time.time()
    args = (ct(mTok), ct(W["embed"]), ct(rope_t),
            ct(W["rms1"]), ct(W["q"]), ct(W["bq"]), ct(W["k"]), ct(W["bk"]),
            ct(W["v"]), ct(W["bv"]), ct(W["o"]),
            ct(W["rms2"]), ct(W["gate"]), ct(W["up"]), ct(W["down"]),
            ct(W["fnorm"]), ct(W["lm"]),
            ct(gH), ct(gQ), ct(gK), ct(gV), ct(gG), ct(gBar), ct(mLogits), stream)
    compiled = cute.compile(mk, *args)
    print(f"    compiled in {time.time()-t0:.1f}s")

    def run(tok):
        # barrier counter must restart at 0 for every launch
        gBar.zero_()
        mTok[0] = tok
        compiled(*args)

    print("\n[4] Correctness vs HuggingFace (first token, position 0)...")
    npass = 0
    toks = [42, 100, 1000, 5000, 12345]
    for tok in toks:
        run(tok)
        torch.cuda.synchronize()
        with torch.no_grad():
            ref = hf(torch.tensor([[tok]])).logits[0, -1, :]
        our = mLogits[0].cpu()
        err = (our - ref).abs().max().item()
        ok = our.argmax().item() == ref.argmax().item()
        npass += ok
        print(f"    tok={tok:6d}: top-1 {'OK ' if ok else 'X  '} "
              f"max_err={err:.3e}  ours={our.argmax().item()} hf={ref.argmax().item()}")
    print(f"    → {npass}/{len(toks)} top-1 match")

    print("\n[5] Benchmark...")
    def bench(fn, it=30):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(it):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / it * 1000

    us = bench(lambda: run(42))
    weight_bytes = sum(v.numel() * v.element_size() for v in W.values())
    achieved = weight_bytes / (us * 1e-6) / 1e9
    print(f"    megakernel:      {us:9.1f} us/token")
    print(f"    effective BW:    {achieved:9.1f} GB/s  "
          f"(4050 peak ~216 GB/s → {achieved/216*100:.0f}% of peak)")
    print(f"    HBM floor:       {weight_bytes/216e9*1e6:9.1f} us "
          f"(if perfectly bandwidth-bound)")


if __name__ == "__main__":
    main()
