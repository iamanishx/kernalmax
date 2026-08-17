import cuda.bindings.driver as cuda
import cutlass
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import cute
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05 import CtaGroup
from cutlass.cute.runtime import from_dlpack


class Tcgen05Minimal:
    def __init__(self):
        self.ab_dtype = cutlass.BFloat16
        self.c_dtype = cutlass.Float32
        self.acc_dtype = cutlass.Float32
        self.mma_tiler_mn = (128, 128)       # one tcgen05 instruction covers (128,128) output
        self.cta_group = CtaGroup.ONE
        self.threads_per_cta = 128            # one warpgroup; only warp 0 drives TMA+MMA

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
                 stream: cuda.CUstream):
        # detect major-ness from the actual strides (K contiguous here)
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(mC)

        # the MMA "constructor": describes tcgen05.mma (idesc-level info)
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.ab_dtype, self.a_major_mode, self.b_major_mode,
            self.acc_dtype, self.cta_group, self.mma_tiler_mn)

        # K tile = 4 x the instruction's fixed 32 bytes = 64 bf16 elements
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])   # 16
        self.mma_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], mma_inst_shape_k * 4)
        self.cta_tile_shape_mnk = self.mma_tiler                      # 1-CTA: no split
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,))
        self.epi_tile = self.cta_tile_shape_mnk[:2]

        # ONE stage only. The swizzle is chosen inside here (built-in).
        a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.ab_dtype, 1)
        b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.ab_dtype, 1)

        # EDUCATIONAL PRINTS (trace time, host side): look for S<3,4,3>
        print("A smem layout (1 stage):", a_smem_layout_staged)
        print("B smem layout (1 stage):", b_smem_layout_staged)

        # TMA constructors (the CUtensorMap descriptors), MMA-aware variants
        a_op = sm100_utils.cluster_shape_to_tma_atom_A((1, 1), tiled_mma.thr_id)
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, mA, a_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape, internal_type=None)
        
        b_op = sm100_utils.cluster_shape_to_tma_atom_B((1, 1), tiled_mma.thr_id)
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, mB, b_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape, internal_type=None)

        num_tma_bytes = (
            cute.size_in_bytes(self.ab_dtype, a_smem_layout)
            + cute.size_in_bytes(self.ab_dtype, b_smem_layout)
        ) * cute.size(tiled_mma.thr_id.shape)

        M, N = mA.shape[0], mB.shape[0]
        grid = (M // self.cta_tile_shape_mnk[0], N // self.cta_tile_shape_mnk[1], 1)

        self.kernel(tiled_mma, tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b, mC,
                    self.cluster_layout_vmnk, a_smem_layout_staged, b_smem_layout_staged,
                    num_tma_bytes).launch(
            grid=grid, block=(self.threads_per_cta, 1, 1), cluster=(1, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, tiled_mma, tma_atom_a, mA_mkl, tma_atom_b, mB_nkl, mC_mnl,
               cluster_layout_vmnk, a_smem_layout_staged, b_smem_layout_staged,
               num_tma_bytes: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)        # = 0 (1-CTA)
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(0)
        mma_tile_coord_mnl = (bidx // cute.size(tiled_mma.thr_id.shape), bidy, bidz)

        # SMEM: one A tile, one B tile, two barriers, one TMEM mailbox
        @cute.struct
        class SharedStorage:
            tmem_holding_buf: cutlass.Int32
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = smem.allocate_tensor(self.ab_dtype, a_smem_layout_staged.outer,
                                  byte_alignment=128, swizzle=a_smem_layout_staged.inner)
        sB = smem.allocate_tensor(self.ab_dtype, b_smem_layout_staged.outer,
                                  byte_alignment=128, swizzle=b_smem_layout_staged.inner)
        tma_bar = smem.allocate_array(cutlass.Int64, 1)   # TMA-done barrier (bytes)
        mma_bar = smem.allocate_array(cutlass.Int64, 1)   # MMA-done barrier (commit)

        # both barriers expect exactly ONE arrival per phase
        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(tma_bar, 1)
                cute.arch.mbarrier_init(mma_bar, 1)
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
        cute.arch.sync_threads()

        # global tiles for the whole tensor, partitioned through the MMA atom
        gA_mkl = cute.local_tile(mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gB_nkl = cute.local_tile(mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        gC_mnl = cute.local_tile(mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, block_in_cluster_coord_vmnk[2], a_cta_layout,
            cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3))
        
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, block_in_cluster_coord_vmnk[1], b_cta_layout,
            cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3))

        tAgA = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
        tBgB = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

        # SMEM descriptors for the MMA (a-desc / b-desc tensors)
        tCrA = tiled_mma.make_fragment_A(sA)     # (MMA, MMA_M, MMA_K, STAGE=1)
        tCrB = tiled_mma.make_fragment_B(sB)

        # TMEM accumulator: layout from fake fragment, storage from the allocator
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        num_tmem_cols = sm100_utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        tmem_bar = cutlass.pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads_per_cta)
        tmem = cutlass.utils.TmemAllocator(storage.tmem_holding_buf,
                                           barrier_for_retrieve=tmem_bar, is_two_cta=False)
        tmem.allocate(num_tmem_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        num_kblks = cute.size(tCrA, mode=[2])     # 4: BK=64 / instr-K=16
        phase = cutlass.Int32(0)                  # one parity var; both barriers flip together

        # ================= SERIAL MAINLOOP (educational, zero overlap) =================
        for kt in cutlass.range(k_tile_cnt):

            # 1) TMA load A tile + B tile into the single SMEM stage (one thread fires)
            if warp_idx == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(tma_bar, num_tma_bytes)
                cute.copy(tma_atom_a, tAgA[(None, kt)], tAsA[(None, 0)], tma_bar_ptr=tma_bar)
                cute.copy(tma_atom_b, tBgB[(None, kt)], tBsB[(None, 0)], tma_bar_ptr=tma_bar)

            # 2) wait until all TMA bytes landed in SMEM
            cute.arch.mbarrier_wait(tma_bar, phase)

            # 3) one thread issues 4 async tcgen05.mma (one per 16-element K slab)
            if warp_idx == 0:
                for kb in cutlass.range_constexpr(num_kblks):
                    kblk_crd = (None, None, kb, 0)
                    cute.gemm(tiled_mma, tCtAcc, tCrA[kblk_crd], tCrB[kblk_crd], tCtAcc)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)   # D = A@B + D after 1st
                with cute.arch.elect_one():
                    tcgen05.commit(mma_bar)    # "arrive on mma_bar when these MMAs finish"

            # 4) wait until the tensor core is done BEFORE next TMA overwrites sA/sB
            cute.arch.mbarrier_wait(mma_bar, phase)

            phase = 1 - phase

        # after the last mma_bar wait, the TMEM accumulator is final
        tmem.relinquish_alloc_permit()

        # ================= EPILOGUE: TMEM -> registers -> GMEM =================
        tCgC_this = tCgC[(None, None, None,
                          mma_tile_coord_mnl[0], mma_tile_coord_mnl[1], mma_tile_coord_mnl[2])]
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk, self.c_layout, self.c_dtype, self.acc_dtype,
            self.epi_tile, False)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc)
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc)
        tTR_gC = thr_copy_t2r.partition_D(tCgC_this)
        tTR_rAcc = cute.make_rmem_tensor(tTR_gC.shape, self.acc_dtype)
        cute.copy(tiled_copy_t2r, tTR_tAcc, tTR_rAcc)      # tcgen05.ld
        cute.autovec_copy(tTR_rAcc, tTR_gC)                # registers -> GMEM

        cute.arch.sync_threads()
        tmem.free(tmem_ptr)


def main():
    m = n = k = 8192
    print(f"MINIMAL tcgen05 GEMM {m}x{k} @ {k}x{n} (bf16, fp32 acc, serial, 1 stage)\n")

    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16).unsqueeze(-1)   # (M,K,L=1)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).unsqueeze(-1)   # (N,K,L=1)
    c = torch.zeros(m, n, device="cuda", dtype=torch.float32).unsqueeze(-1)    # (M,N,L=1)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    Tcgen05Minimal()(from_dlpack(a, assumed_align=32),
                     from_dlpack(b, assumed_align=32),
                     from_dlpack(c, assumed_align=32), stream)
    torch.cuda.synchronize()
    ref = torch.einsum("mkl,nkl->mnl", a.float(), b.float())
    print(f"max abs err vs torch: {(c - ref).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
