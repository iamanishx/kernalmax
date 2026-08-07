"""Modular @cute.jit ops for the Qwen2.5-0.5B persistent megakernel.

@cute.jit is required so runtime grid-stride loops get AST-preprocessed
(plain Python functions don't — see ../GOTCHAS.md §1.7).
"""

from ops.attention import gqa_decode_first, rope
from ops.matmul import (
    g2s,
    gemv_bias_mc,
    gemv_residual_mc,
    lm_head_mc,
    mlp_gate_silu_mc,
)
from ops.reduce import block_reduce_sum, grid_barrier
from ops.rmsnorm import rmsnorm, rmsnorm_final

__all__ = [
    "block_reduce_sum",
    "g2s",
    "gemv_bias_mc",
    "gemv_residual_mc",
    "gqa_decode_first",
    "grid_barrier",
    "lm_head_mc",
    "mlp_gate_silu_mc",
    "rmsnorm",
    "rmsnorm_final",
    "rope",
]
