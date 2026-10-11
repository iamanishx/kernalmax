"""
tma_v4.py -- TMA copy with HAND-WRITTEN PTX via llvm.inline_asm. For clarity + fun.

Same verified GMEM -> SMEM -> GMEM copy as tma.py / tma_v2.py / tma_v3.py, but every
hardware instruction is a Python wrapper over llvm.inline_asm carrying the exact PTX
string. Nothing between you and the instruction.

The four wrappers (compare each string to blackwell/tma.sm90.ptx, they match 1:1):
    mbar_init            mbarrier.init.shared::cta.b64
    mbar_arrive_expect   mbarrier.arrive.expect_tx.release.cta.shared::cta.b64
    tma_load_2d          cp.async.bulk.tensor.2d...mbarrier::complete_tx::bytes
    mbar_wait            mbarrier.try_wait.parity spin loop (%= = unique label per use)

Marshalling rules that make it work:
    - SMEM pointers (addrspace 6): ptr.toint() returns Int32 -> constraint "r"
    - the tensormap descriptor pointer: Int64 generic address -> constraint "l"
    - every operand needs .ir_value() to become an MLIR value
    - NO self-election at this level: everything runs under elect_one()
    - the host-side CUtensorMap still comes from create_tensor_map_tiled_from_view
      (a driver call; there is no inline-asm path to it, that floor never moves)
"""

import cutlass
import torch
from cutlass import Int32, cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.experimental.cuda import (
    TensorMap,
    TensorMapSwizzle,
    create_tensor_map_tiled_from_view,
)

BM = 64
BN = 64
TPB = 128


def _asm(asm: str, constraints: str, operands):
    llvm.inline_asm(
        None, operands, asm, constraints,
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def mbar_init(mbar_ptr, cnt):
    _asm("mbarrier.init.shared::cta.b64 [$0], $1;", "r,r",
         [mbar_ptr.toint().ir_value(), Int32(cnt).ir_value()])


def mbar_arrive_expect(mbar_ptr, nbytes):
    _asm("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [$0], $1;", "r,r",
         [mbar_ptr.toint().ir_value(), Int32(nbytes).ir_value()])


def tma_load_2d(smem_ptr, desc_ptr, x, y, mbar_ptr):
    _asm("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
         " [$0], [$1, {$2, $3}], [$4];",
         "r,l,r,r,r",
         [smem_ptr.toint().ir_value(), desc_ptr.toint().ir_value(),
          Int32(x).ir_value(), Int32(y).ir_value(), mbar_ptr.toint().ir_value()])


def mbar_wait(mbar_ptr, phase):
    _asm(
        "{\n"
        ".reg .pred P1;\n"
        "L1_v4:\n"
        "mbarrier.try_wait.parity.shared.b64 P1, [$0], $1, 10000000;\n"
        "@P1 bra.uni D1_v4;\n"
        "bra.uni L1_v4;\n"
        "D1_v4:\n"
        "}",
        "r,r",
        [mbar_ptr.toint().ir_value(), Int32(phase).ir_value()],
    )


class TmaCopyV4:
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
                mbar_init(mbar, 1)
        cute.arch.sync_threads()

        if warp_id == 0:
            with cute.arch.elect_one():
                mbar_arrive_expect(mbar, desc.global_tx_bytes())
                tma_load_2d(sA.iterator, desc.get_ptr(), bid_n * BN, bid_m * BM, mbar)

        mbar_wait(mbar, 0)

        for i in cutlass.range_constexpr(BM * BN // TPB):
            idx = i * TPB + tid
            col = idx % BN
            row = idx // BN
            B[bid_m * BM + row, bid_n * BN + col] = sA[row, col]


def main():
    M, N = 512, 512
    a = torch.randn(M, N, device="cuda", dtype=torch.float32)
    b = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    TmaCopyV4()(from_dlpack(a, assumed_align=32), from_dlpack(b, assumed_align=32))
    torch.cuda.synchronize()

    print(f"TMA GMEM -> SMEM -> GMEM  {M}x{N} fp32, tile {BM}x{BN}   (inline PTX)")
    print(f"match: {torch.equal(a, b)}   max abs err: {(a - b).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
