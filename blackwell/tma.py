import cutlass
import torch
from cutlass import cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

BM = 64
BN = 64
TPB = 128


class TmaCopy:
    @cute.jit
    def prepare_tma(self, tensor: cute.Tensor):
        tma_op = cpasync.CopyBulkTensorTileG2SOp()
        s_layout = cute.make_layout((BM, BN), stride=(BN, 1))
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            tma_op, tensor, s_layout, cta_tiler=(BM, BN))
        return tma_atom, tma_tensor, s_layout

    @cute.jit
    def __call__(self, A: cute.Tensor, B: cute.Tensor):
        A_args = self.prepare_tma(A)
        M, N = A.shape
        self.kernel(A_args, B).launch(
            grid=(N // BN, M // BM, 1), block=(TPB, 1, 1))

    @cute.kernel
    def kernel(self, A_args: tuple, B: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid_n, bid_m, _ = cute.arch.block_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)

        A_tma_atom, A_tma_tensor, sA_layout = A_args

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(B.element_type, sA_layout, byte_alignment=128)
        mbar = smem.allocate_array(cutlass.Int64, 1)

        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(mbar, 1)
        cute.arch.sync_threads()

        if warp_id == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(mbar, BM * BN * 4)

            src = cute.local_tile(A_tma_tensor, tiler=(BM, BN), coord=(bid_m, bid_n))
            tAsA, tAgA = cpasync.tma_partition(
                A_tma_atom,
                cta_coord=0,
                cta_layout=cute.make_layout(1),
                smem_tensor=cute.group_modes(sA, 0, 2),
                gmem_tensor=cute.group_modes(src, 0, 2),
            )
            cute.copy(A_tma_atom, tAgA, tAsA, tma_bar_ptr=mbar)

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

    TmaCopy()(from_dlpack(a, assumed_align=32), from_dlpack(b, assumed_align=32))
    torch.cuda.synchronize()

    print(f"TMA GMEM -> SMEM -> GMEM  {M}x{N} fp32, tile {BM}x{BN}")
    print(f"match: {torch.equal(a, b)}   max abs err: {(a - b).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
