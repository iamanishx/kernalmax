"""
tma_v3.py -- TMA copy via the RAW tensor-map API (cutlass 4.7 experimental).

Same verified GMEM -> SMEM -> GMEM copy as tma.py, but built on the new
cutlass.experimental.cuda tensor-map API instead of make_tiled_tma_atom.

The two descriptor builders (same CUtensorMap out, different input style):

  create_tensor_map_tiled(...)            RAW args. You pass everything yourself:
      global_address=ptr, dtype, global_dims, global_strides, box_dims, swizzle
      - dims/box in TMA order: dims[0] = CONTIGUOUS axis (column-major)
      - global_strides in 16-BYTE units (N*4//16), not elements, not bytes

  create_tensor_map_tiled_from_view(t, box_dims=..., swizzle=...)   WRAPPER.
      Auto-detects address/dtype/dims/strides from the cute.Tensor layout.
      - box_dims in TENSOR order (BM, BN), it reorders internally
      - use this one for plain torch-backed tensors

Gotchas this file had to respect:
  - desc.global_tx_bytes() is a METHOD (parens!), not a property. It derives the
    mbarrier expect_tx byte count from descriptor metadata, no hand math to drift.
  - TMA coordinates are innermost-dim FIRST: (n_offset, m_offset) for row-major
    (M, N). coords are element indices, not bytes.
  - nvvm.cp_async_bulk_tensor_shared_cta_global does NOT self-elect. It must run
    inside elect_one() (unlike cute.copy / block_copy, which self-elect).
  - The descriptor arrives as cutlass.GridConstant[TensorMap] (= __grid_constant__,
    lives in param/constant space, shared by all CTAs).
  - cute.copy / block_copy CANNOT consume this raw TensorMap: those need a CuTe
    copy ATOM from make_tiled_tma_atom_*. This file builds no atom; the raw
    descriptor is consumed by the nvvm intrinsic directly. Two separate worlds:
      atom world: make_tiled_tma_atom -> cute.copy / block_copy
      raw world:  create_tensor_map_tiled* -> nvvm intrinsics / inline PTX

Lineage: tma.py (atom + tma_partition) -> tma_v2.py (atom + block_copy)
         -> tma_v3.py (raw descriptor + coords). Verified on B200, exact match.
"""

import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from cutlass.experimental.cuda import (
    TensorMap,
    TensorMapSwizzle,
    create_tensor_map_tiled_from_view,
)
from cutlass.experimental.primitives import nvvm_wrapper as nvvm

BM = 64
BN = 64
TPB = 128


class TmaCopyV3:
    @cute.jit
    def __call__(self, A: cute.Tensor, B: cute.Tensor):
        desc = create_tensor_map_tiled_from_view(
            A, box_dims=(BM, BN), swizzle=TensorMapSwizzle.none)
        M, N = A.shape
        self.kernel(desc, B).launch(
            grid=(N // BN, M // BM, 1), block=(TPB, 1, 1))

    @cute.kernel
    def kernel(self, desc: cutlass.GridConstant[TensorMap], B: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid_n, bid_m, _ = cute.arch.block_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(B.element_type,
                                  cute.make_layout((BM, BN), stride=(BN, 1)),
                                  byte_alignment=128)
        mbar = smem.allocate_array(cutlass.Int64, 1)

        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(mbar, 1)
        cute.arch.sync_threads()

        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbar, desc.global_tx_bytes())
                nvvm.cp_async_bulk_tensor_shared_cta_global(
                    sA.iterator,
                    desc.get_ptr(),
                    (bid_n * BN, bid_m * BM),
                    mbar)

        cute.arch.mbarrier_wait(mbar, 0)

        for i in cutlass.range_constexpr(BM * BN // TPB):
            idx = i * TPB + tid
            col = idx % BN
            row = idx // BN
            B[bid_m * BM + row, bid_n * BN + col] = sA[row, col]


def main():
    M, N = 512, 512
    a = torch.randn(M, N, device="cuda", dtype=torch.float32)
    b = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    TmaCopyV3()(from_dlpack(a, assumed_align=32), from_dlpack(b, assumed_align=32))
    torch.cuda.synchronize()

    print(f"TMA GMEM -> SMEM -> GMEM  {M}x{N} fp32, tile {BM}x{BN}   (raw tensormap API)")
    print(f"match: {torch.equal(a, b)}   max abs err: {(a - b).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
