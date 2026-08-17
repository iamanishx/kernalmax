"""Qwen2.5-0.5B architecture constants and weight layout helpers.

All values from: https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json
"""

from dataclasses import dataclass

# ─── Architecture ───

NUM_LAYERS: int = 24
HIDDEN_DIM: int = 896
NUM_Q_HEADS: int = 14
NUM_KV_HEADS: int = 2
HEAD_DIM: int = HIDDEN_DIM // NUM_Q_HEADS  # 64
FFN_INTERMEDIATE: int = 4864
VOCAB_SIZE: int = 151936
MAX_SEQ_LEN: int = 32768
RMS_NORM_EPS: float = 1e-6
ROPE_THETA: float = 1_000_000.0
KV_DIM: int = NUM_KV_HEADS * HEAD_DIM  # 128
Q_DIM: int = NUM_Q_HEADS * HEAD_DIM  # 896
GROUP: int = NUM_Q_HEADS // NUM_KV_HEADS  # 7 Q heads per KV head
HALF: int = HEAD_DIM // 2  # 32 rope pairs

# ─── Kernel config (tunable) ───

BLOCK_SIZE: int = 512  # threads per CTA
NUM_THREADS: int = 512  # alias for BLOCK_SIZE
NUM_WARPS: int = BLOCK_SIZE // 32  # 16

# Matmul tile sizes (SIMT phase — tune later for tensor core phase)
TILE_M: int = 1       # single token
TILE_K: int = 64      # K tile size
TILE_N: int = 64      # N tile size

# Tiling config
CTA_TILE_Q: int = 1                 # Q heads per CTA (single token)
CTA_TILE_KV: int = 2               # KV heads per CTA (all KV heads for 0.5B)

# ─── Weight layout (matching HF Qwen2.5 state dict) ───

# Key naming convention from HF:
#   model.layers.{layer_idx}.input_layernorm.weight     → RMS1
#   model.layers.{layer_idx}.self_attn.q_proj.weight    → Q proj
#   model.layers.{layer_idx}.self_attn.k_proj.weight    → K proj
#   model.layers.{layer_idx}.self_attn.v_proj.weight    → V proj
#   model.layers.{layer_idx}.self_attn.o_proj.weight    → O proj
#   model.layers.{layer_idx}.post_attention_layernorm.weight → RMS2
#   model.layers.{layer_idx}.mlp.gate_proj.weight       → Gate
#   model.layers.{layer_idx}.mlp.up_proj.weight         → Up
#   model.layers.{layer_idx}.mlp.down_proj.weight       → Down
#   model.embed_tokens.weight                            → Embed
#   model.norm.weight                                     → Final RMS
#   lm_head.weight (tied with embed_tokens)               → LM Head

# Weight shapes per layer:
#   rms1:      [896]
#   q_proj:    [896, 896]      (14 heads × 64 dim)
#   k_proj:    [128, 896]      (2 KV heads × 64 dim)
#   v_proj:    [128, 896]
#   o_proj:    [896, 896]
#   rms2:      [896]
#   gate_proj: [4864, 896]
#   up_proj:   [4864, 896]
#   down_proj: [896, 4864]


@dataclass
class LayerWeights:
    """Pointers/shapes for one layer's weight tensors."""
    rms1: object      # [896] float32/bf16
    q_proj: object    # [896, 896]
    k_proj: object    # [128, 896]
    v_proj: object    # [128, 896]
    o_proj: object    # [896, 896]
    rms2: object      # [896]
    gate_proj: object # [4864, 896]
    up_proj: object   # [4864, 896]
    down_proj: object # [896, 4864]


# ─── GPU / launch config ───

# RTX 4050 (Ada Lovelace, sm_89)
# 20 SMs, 2560 CUDA cores, 6GB VRAM
NUM_SMS: int = 20
NUM_CTAS: int = 20  # persistent: one CTA per SM
SMEM_PER_SM: int = 98 * 1024  # ~98 KB (configurable split with L1)
MAX_CTAS_PER_SM: int = 16
MAX_THREADS_PER_SM: int = 1536

# For persistent scheduling: one CTA per SM
PERSISTENT_GRID: int = NUM_SMS

# attention scale 1/sqrt(head_dim)
import math as _m

SCALE: float = 1.0 / _m.sqrt(HEAD_DIM)
MAX_SEQ_CACHE: int = 512  # KV cache capacity
