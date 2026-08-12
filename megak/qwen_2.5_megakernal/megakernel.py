import math
import os
import sys
import time

import cuda.bindings.driver as cuda
import cutlass
import torch
from config import (
    FFN_INTERMEDIATE,
    HALF,
    HEAD_DIM,
    HIDDEN_DIM,
    KV_DIM,
    MAX_SEQ_CACHE,
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
    gqa_attention,
    grid_barrier,
    kv_cache_write,
    lm_head_mc,
    mlp_gate_silu_mc,
    rmsnorm,
    rmsnorm_final,
    rope,
)
from prefill_megakernel import BM as PF_BM
from prefill_megakernel import PrefillMegakernel

THREADS = NUM_THREADS
BARRIERS_PER_LAYER = 4

# Instruct model: follows chat templates and emits <|im_end|> so generation
# terminates cleanly. The base model rambles — not supported here.
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


class Qwen25Megakernel:
    @cute.kernel
    def kernel(self, mTok, mPos, mEmb, mRope,
               w_rms1, w_q, b_q, w_k, b_k, w_v, b_v, w_o, w_rms2,
               w_gate, w_up, w_down, w_fnorm, w_lm,
               gH, gQ, gK, gV, gG, gKC, gVC, gBar, mLogits,
               n_layers: cutlass.Constexpr):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()

        sH = smem.allocate_tensor(cutlass.Float32, cute.make_layout(HIDDEN_DIM), byte_alignment=16)
        sN = smem.allocate_tensor(cutlass.Float32, cute.make_layout(HIDDEN_DIM), byte_alignment=16)
        sQ = smem.allocate_tensor(cutlass.Float32, cute.make_layout(Q_DIM), byte_alignment=16)
        sK = smem.allocate_tensor(cutlass.Float32, cute.make_layout(KV_DIM), byte_alignment=16)
        sV = smem.allocate_tensor(cutlass.Float32, cute.make_layout(KV_DIM), byte_alignment=16)
        sA = smem.allocate_tensor(cutlass.Float32, cute.make_layout(HIDDEN_DIM), byte_alignment=16)
        sG = smem.allocate_tensor(cutlass.Float32, cute.make_layout(FFN_INTERMEDIATE), byte_alignment=16)
        sR = smem.allocate_tensor(cutlass.Float32, cute.make_layout(32), byte_alignment=16)

        # runtime position / sequence length
        token = mTok[0]
        position = mPos[0]
        seq_len = position + cutlass.Int32(1)

        # ── embedding (all CTAs write identical values to gH) ──
        for i in cutlass.range(tid, HIDDEN_DIM, THREADS):
            e = mEmb[(token, i)].to(cutlass.Float32)
            sH[i] = e
            gH[i] = e
        grid_barrier(gBar, tid, 1, NUM_CTAS)

        for layer in cutlass.range_constexpr(n_layers):
            base = layer * BARRIERS_PER_LAYER + 1

            g2s(gH, sH, tid, HIDDEN_DIM)
            cute.arch.sync_threads()

            # ═══ attention block ═══
            rmsnorm(sH, w_rms1, sN, sR, tid, layer, HIDDEN_DIM)
            cute.arch.sync_threads()

            gemv_bias_mc(sN, w_q, b_q, gQ, tid, cta, layer, HIDDEN_DIM, HIDDEN_DIM)
            gemv_bias_mc(sN, w_k, b_k, gK, tid, cta, layer, HIDDEN_DIM, KV_DIM)
            gemv_bias_mc(sN, w_v, b_v, gV, tid, cta, layer, HIDDEN_DIM, KV_DIM)
            grid_barrier(gBar, tid, base + 1, NUM_CTAS)

            g2s(gQ, sQ, tid, Q_DIM)
            g2s(gK, sK, tid, KV_DIM)
            g2s(gV, sV, tid, KV_DIM)
            cute.arch.sync_threads()

            rope(sQ, sK, mRope, tid, position)
            cute.arch.sync_threads()

            # Each CTA writes the FULL KV row itself (redundant, identical
            # values) so it only needs to see its own write — no grid barrier.
            kv_cache_write(sK, sV, gKC, gVC, tid, layer, position)
            cute.arch.sync_threads()

            # real GQA + online softmax over cache[0 .. seq_len-1].
            # Redundant per CTA because sA is per-CTA SMEM.
            gqa_attention(sQ, gKC, gVC, sA, tid, layer, seq_len)
            cute.arch.sync_threads()

            gemv_residual_mc(sA, w_o, gH, tid, cta, layer, HIDDEN_DIM, HIDDEN_DIM)
            grid_barrier(gBar, tid, base + 2, NUM_CTAS)

            # ═══ MLP block ═══
            g2s(gH, sH, tid, HIDDEN_DIM)
            cute.arch.sync_threads()
            rmsnorm(sH, w_rms2, sN, sR, tid, layer, HIDDEN_DIM)
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
        rmsnorm_final(sH, w_fnorm, sN, sR, tid, HIDDEN_DIM)
        cute.arch.sync_threads()
        lm_head_mc(sN, w_lm, mLogits, tid, cta, HIDDEN_DIM, VOCAB_SIZE)

    @cute.jit
    def __call__(self, mTok, mPos, mEmb, mRope,
                 w_rms1, w_q, b_q, w_k, b_k, w_v, b_v, w_o, w_rms2,
                 w_gate, w_up, w_down, w_fnorm, w_lm,
                 gH, gQ, gK, gV, gG, gKC, gVC, gBar, mLogits,
                 stream: cuda.CUstream,
                 n_layers: cutlass.Constexpr = NUM_LAYERS):
        # NOTE: Constexpr args come from defaults — never pass them positionally
        # after `stream`, that silently corrupts the launch.
        self.kernel(mTok, mPos, mEmb, mRope,
                    w_rms1, w_q, b_q, w_k, b_k, w_v, b_v, w_o, w_rms2,
                    w_gate, w_up, w_down, w_fnorm, w_lm,
                    gH, gQ, gK, gV, gG, gKC, gVC, gBar, mLogits,
                    n_layers).launch(
            grid=[NUM_CTAS, 1, 1], block=[THREADS, 1, 1], stream=stream)



BARRIERS_TOTAL = NUM_LAYERS * BARRIERS_PER_LAYER + 1


class MegakernelEngine:
    
    def __init__(self, model_id=MODEL_ID, bf16=True, verbose=True):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.dev = torch.device("cuda")
        self.verbose = verbose

        if verbose:
            print(f"[1] loading weights: {model_id}")
        # fp32 copy ONLY for exact weight extraction; dropped after to save VRAM.
        hf = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float32, local_files_only=True).eval()
        self.tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        sd = {k: v.detach().clone() for k, v in hf.named_parameters()}
        self.hf = hf

        # Robust stop-token set. Qwen chat ends a turn with <|im_end|>, which is
        # NOT eos_token_id on the BASE model — without this the model rambles.
        stops = set()
        for tokstr in ("<|im_end|>", "<|endoftext|>"):
            i = self.tok.convert_tokens_to_ids(tokstr)
            if isinstance(i, int) and i >= 0:
                stops.add(i)
        for i in (self.tok.eos_token_id, self.tok.pad_token_id):
            if isinstance(i, int) and i >= 0:
                stops.add(i)
        self.stop_ids = stops

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
        del hf            # free the fp32 reference model (~2.5 GB)
        import gc; gc.collect()
        self.hf_bf16 = None        # built lazily on first batched prefill
        self._model_id = model_id
        for k in W:
            W[k] = W[k].to(self.dev).contiguous()
        if bf16:
            for k in ["q", "k", "v", "o", "gate", "up", "down", "embed", "lm"]:
                W[k] = W[k].to(torch.bfloat16).contiguous()
        self.W = W
        self.weight_bytes = sum(v.numel() * v.element_size() for v in W.values())

        # rope table + activation buffers + KV cache
        self.rope_t = self._rope_table(MAX_SEQ_CACHE)
        z = lambda *s: torch.zeros(*s, device=self.dev, dtype=torch.float32)
        self.gH, self.gQ = z(HIDDEN_DIM), z(Q_DIM)
        self.gK, self.gV = z(KV_DIM), z(KV_DIM)
        self.gG = z(FFN_INTERMEDIATE)
        self.gKC = z(NUM_LAYERS, MAX_SEQ_CACHE, KV_DIM)
        self.gVC = z(NUM_LAYERS, MAX_SEQ_CACHE, KV_DIM)
        self.gBar = torch.zeros(1, device=self.dev, dtype=torch.int32)
        self.mLogits = z(1, VOCAB_SIZE)
        self.mTok = torch.zeros(1, device=self.dev, dtype=torch.int32)
        self.mPos = torch.zeros(1, device=self.dev, dtype=torch.int32)

        kv_mb = (self.gKC.numel() + self.gVC.numel()) * 4 / 1024**2
        if verbose:
            print(f"    weights {self.weight_bytes/1024**3:.2f} GB"
                  f"{' (bf16 big)' if bf16 else ' (fp32)'}, KV cache {kv_mb:.0f} MB")
            print("[2] compiling megakernel...")

        ct = lambda t: from_dlpack(t, assumed_align=16)
        self.stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self.args = (
            ct(self.mTok), ct(self.mPos), ct(W["embed"]), ct(self.rope_t),
            ct(W["rms1"]), ct(W["q"]), ct(W["bq"]), ct(W["k"]), ct(W["bk"]),
            ct(W["v"]), ct(W["bv"]), ct(W["o"]),
            ct(W["rms2"]), ct(W["gate"]), ct(W["up"]), ct(W["down"]),
            ct(W["fnorm"]), ct(W["lm"]),
            ct(self.gH), ct(self.gQ), ct(self.gK), ct(self.gV), ct(self.gG),
            ct(self.gKC), ct(self.gVC), ct(self.gBar), ct(self.mLogits),
            self.stream,
        )
        t0 = time.time()
        self.compiled = cute.compile(Qwen25Megakernel(), *self.args)
        if verbose:
            print(f"    compiled in {time.time()-t0:.1f}s "
                  f"({BARRIERS_TOTAL} grid barriers)")

    @staticmethod
    def _rope_table(max_seq):
        t = torch.zeros(max_seq, HALF, 2, device="cuda", dtype=torch.float32)
        for pair in range(HALF):
            theta = 1.0 / (ROPE_THETA ** (2.0 * pair / HEAD_DIM))
            for pos in range(max_seq):
                t[pos, pair, 0] = math.cos(theta * pos)
                t[pos, pair, 1] = math.sin(theta * pos)
        return t

    def reset(self):
        self.gKC.zero_()
        self.gVC.zero_()

    def _maybe_log_mem(self, where):
        if os.environ.get("MEMLOG") == "1":
            a = torch.cuda.memory_allocated() / 1024**3
            r = torch.cuda.memory_reserved() / 1024**3
            print(f"    [mem:{where}] alloc={a:.2f}GB reserved={r:.2f}GB")

    def step(self, token_id, position):
        """One forward pass. Appends K/V at `position`, returns logits [VOCAB]."""
        self._maybe_log_mem("step")
        self.gBar.zero_()                 # barrier counter restarts each launch
        self.mTok[0] = token_id
        self.mPos[0] = position
        self.compiled(*self.args)
        return self.mLogits[0]

    def prefill(self, token_ids):
        logits = None
        for pos, t in enumerate(token_ids):
            logits = self.step(int(t), pos)
        torch.cuda.synchronize()
        return logits

    def _get_prefill_model(self):
        if self.hf_bf16 is None:
            from transformers import AutoModelForCausalLM
            self.hf_bf16 = AutoModelForCausalLM.from_pretrained(
                self._model_id, dtype=torch.bfloat16,
                local_files_only=True).cuda().eval()
        return self.hf_bf16

    def _build_prefill(self, M):
        """Compile the prefill megakernel for a padded length M (cached per M)."""
        if not hasattr(self, "_pf"):
            self._pf = {}
        if M in self._pf:
            return self._pf[M]
        dev = self.dev
        z = lambda *sh: torch.zeros(*sh, device=dev, dtype=torch.float32)
        b = dict(
            gH=z(M, HIDDEN_DIM), gX=z(M, HIDDEN_DIM), gQ=z(M, Q_DIM),
            gK=z(M, KV_DIM), gV=z(M, KV_DIM), gA=z(M, HIDDEN_DIM),
            gGu=z(M, FFN_INTERMEDIATE), gUp=z(M, FFN_INTERMEDIATE),
            mToks=torch.zeros(M, device=dev, dtype=torch.int32),
            mMreal=torch.zeros(1, device=dev, dtype=torch.int32),
        )
        W = self.W
        ct = lambda t: from_dlpack(t, assumed_align=16)
        args = (ct(b["mToks"]), ct(W["embed"]), ct(self.rope_t),
                ct(W["rms1"]), ct(W["q"]), ct(W["bq"]), ct(W["k"]), ct(W["bk"]),
                ct(W["v"]), ct(W["bv"]), ct(W["o"]), ct(W["rms2"]),
                ct(W["gate"]), ct(W["up"]), ct(W["down"]),
                ct(W["fnorm"]), ct(W["lm"]),
                ct(b["gH"]), ct(b["gX"]), ct(b["gQ"]), ct(b["gK"]), ct(b["gV"]),
                ct(b["gA"]), ct(b["gGu"]), ct(b["gUp"]),
                ct(self.gKC), ct(self.gVC), ct(self.gBar), ct(self.mLogits),
                ct(b["mMreal"]), self.stream)
        if self.verbose:
            print(f"[3] compiling PREFILL megakernel (M={M})...")
        t0 = time.time()
        compiled = cute.compile(PrefillMegakernel(), *args, M=M)
        if self.verbose:
            print(f"    compiled in {time.time()-t0:.0f}s")
        self._pf[M] = (compiled, args, b)
        return self._pf[M]

    @torch.no_grad()
    def prefill_mk(self, token_ids):
        """PREFILL MEGAKERNEL: whole prompt in ONE launch. Fills KV cache,
        returns logits [VOCAB] for the first generated token."""
        n = len(token_ids)
        M = ((n + PF_BM - 1) // PF_BM) * PF_BM
        compiled, args, b = self._build_prefill(M)
        b["mToks"].zero_()
        b["mToks"][:n] = torch.tensor(token_ids, device=self.dev, dtype=torch.int32)
        b["mMreal"][0] = n
        self.gBar.zero_()
        compiled(*args)
        torch.cuda.synchronize()
        return self.mLogits[0]

    @torch.no_grad()
    def prefill_batched(self, token_ids):
        """Batched prefill: one HF forward pass over the whole prompt, then copy
        the resulting K/V into our megakernel cache. Returns logits [VOCAB].

        The win vs token-by-token: weights are read ONCE for all M tokens
        (a GEMM) instead of M times (M GEMVs). See .agents/megakernel.md —
        prefill is bandwidth-bound at small L, so read-once is the whole game.
        """
        n = len(token_ids)
        ids = torch.tensor([token_ids], device=self.dev)
        out = self._get_prefill_model()(ids, use_cache=True)
        logits = out.logits[0, -1].float()          # [VOCAB]
        # transformers>=5 DynamicCache: pkv.layers[l].keys is [1, n_kv, seq, hd].
        # Flatten the 2 KV heads into [seq, KV_DIM] to match our cache layout.
        for layer in range(NUM_LAYERS):
            L = out.past_key_values.layers[layer]
            k = L.keys[0].transpose(0, 1).reshape(n, KV_DIM)    # [seq, KV_DIM]
            v = L.values[0].transpose(0, 1).reshape(n, KV_DIM)
            self.gKC[layer, :n] = k.float()
            self.gVC[layer, :n] = v.float()
        torch.cuda.synchronize()
        return logits

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=40, temperature=0.0, top_k=50,
                 chat=True, stream_out=True, fast_prefill=True):

        if chat:
            enc = self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, tokenize=True)
            if isinstance(enc, dict) or hasattr(enc, "input_ids"):
                ids = enc["input_ids"]
            else:
                ids = enc
            if len(ids) and isinstance(ids[0], (list, tuple)):
                ids = ids[0]
            ids = [int(t) for t in ids]
        else:
            ids = [int(t) for t in self.tok(prompt)["input_ids"]]

        if len(ids) + max_new_tokens > MAX_SEQ_CACHE:
            max_new_tokens = MAX_SEQ_CACHE - len(ids)

        self.reset()
        t0 = time.time()
        logits = self.prefill_mk(ids) if fast_prefill else self.prefill(ids)
        t_prefill = time.time() - t0

        out_ids, pos = [], len(ids)
        t0 = time.time()
        for _ in range(max_new_tokens):
            if temperature <= 0.0:
                nxt = int(logits.argmax().item())
            else:
                lg = logits.float() / temperature
                if top_k:
                    v, ix = lg.topk(min(top_k, lg.numel()))
                    probs = torch.softmax(v, dim=-1)
                    nxt = int(ix[torch.multinomial(probs, 1)].item())
                else:
                    nxt = int(torch.multinomial(torch.softmax(lg, -1), 1).item())

            if nxt in self.stop_ids:
                break
            out_ids.append(nxt)
            if stream_out:
                print(self.tok.decode([nxt]), end="", flush=True)

            logits = self.step(nxt, pos)
            pos += 1
            if pos >= MAX_SEQ_CACHE:
                break
        torch.cuda.synchronize()
        t_decode = time.time() - t0
        if stream_out:
            print()

        n = max(len(out_ids), 1)
        return {
            "text": self.tok.decode(out_ids),
            "n_prompt": len(ids),
            "n_new": len(out_ids),
            "prefill_s": t_prefill,
            "decode_s": t_decode,
            "prefill_tok_s": len(ids) / t_prefill if t_prefill else 0.0,
            "decode_tok_s": n / t_decode if t_decode else 0.0,
        }


# ═══════════════════════════ validation + demo ═══════════════════════════

def validate(eng, n_pos=6):
    """Check our logits against HF across several positions (tests KV cache)."""
    print("\n[3] correctness vs HuggingFace across positions...")
    ids = eng.tok("The capital of France is Paris, and the capital of Italy is")["input_ids"]
    ids = [int(t) for t in ids][:n_pos]
    eng.reset()
    hf_out = eng.hf(torch.tensor([ids]))          # full-sequence HF reference
    npass = 0
    for pos, t in enumerate(ids):
        our = eng.step(int(t), pos)
        torch.cuda.synchronize()
        ref = hf_out.logits[0, pos, :]
        ok = our.cpu().argmax().item() == ref.argmax().item()
        err = (our.cpu() - ref).abs().max().item()
        npass += ok
        print(f"    pos={pos} tok={t:6d}: top-1 {'OK ' if ok else 'X  '} "
              f"max_err={err:.3e}  ours={our.cpu().argmax().item()} hf={ref.argmax().item()}")
    print(f"    → {npass}/{len(ids)} positions match "
          f"({'KV cache + online softmax CORRECT' if npass==len(ids) else 'MISMATCH'})")
    return npass == len(ids)


def validate_generation(eng, prompt="The capital of France is", n=20):
    """Definitive end-to-end test: our greedy generation vs HF's, token-for-token."""
    print(f"\n[4] end-to-end greedy generation vs HF ('{prompt}', {n} tokens)...")
    ids = [int(t) for t in eng.tok(prompt)["input_ids"]]

    # ours
    eng.reset()
    logits = eng.prefill(ids)
    ours, pos = [], len(ids)
    for _ in range(n):
        nxt = int(logits.argmax().item())
        ours.append(nxt)
        logits = eng.step(nxt, pos)
        pos += 1

    # HF greedy
    with torch.no_grad():
        hf_ids = eng.hf.generate(torch.tensor([ids]), max_new_tokens=n,
                                 do_sample=False,
                                 pad_token_id=eng.tok.eos_token_id)
    theirs = [int(t) for t in hf_ids[0][len(ids):]]

    # find first divergence and report how close that decision was
    first_div = next((i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b), None)
    if first_div is not None:
        eng.reset()
        lg = eng.prefill(ids)
        for t in ours[:first_div]:
            lg = eng.step(t, len(ids) + ours[:first_div].index(t))
        top2 = lg.float().topk(2)
        gap = (top2.values[0] - top2.values[1]).item()
        print(f"    first divergence at token {first_div}: "
              f"top-2 logit gap = {gap:.4f}  "
              f"({'NEAR-TIE → bf16 rounding' if gap < 0.1 else 'large gap → real bug'})")

    match = sum(a == b for a, b in zip(ours, theirs))
    print(f"    ours  : {eng.tok.decode(ours)!r}")
    print(f"    HF    : {eng.tok.decode(theirs)!r}")
    print(f"    → {match}/{len(theirs)} tokens identical "
          f"({'EXACT MATCH' if match == len(theirs) else 'divergence'})")
    return match == len(theirs)


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__.strip())
        print("\nusage: python3 megakernel.py \"your prompt\" [max_new_tokens] [temperature]")
        print("       python3 megakernel.py --validate      # correctness checks only")
        sys.exit(0 if args else 1)

    validate_only = args[0] == "--validate"
    prompt = None if validate_only else args[0]
    max_new = int(args[1]) if len(args) > 1 and not validate_only else 128
    temp = float(args[2]) if len(args) > 2 and not validate_only else 0.0

    print("=" * 62)
    print("Qwen2.5-0.5B-Instruct Megakernel — prefill + decode, KV cache")
    print("=" * 62)
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name} | SMs: {p.multi_processor_count} | sm_{p.major}{p.minor}")
    print(f"grid=[{NUM_CTAS}] block=[{THREADS}] -> {NUM_CTAS*THREADS//32} warps\n")

    eng = MegakernelEngine()

    if validate_only:
        validate(eng)
        print(f"    stop tokens: {sorted(eng.stop_ids)}")
        validate_generation(eng)
        return

    print(f"\n{'='*62}\nPROMPT: {prompt}\n{'-'*62}")
    r = eng.generate(prompt, max_new_tokens=max_new, temperature=temp)
    print(f"{'-'*62}")
    print(f"prompt {r['n_prompt']} tok | generated {r['n_new']} tok")
    print(f"prefill {r['prefill_s']*1000:.0f} ms ({r['prefill_tok_s']:.1f} tok/s) | "
          f"decode {r['decode_s']*1000:.0f} ms ({r['decode_tok_s']:.1f} tok/s)")


if __name__ == "__main__":
    main()
