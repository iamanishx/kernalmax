import time

import cupy as cp
import cutlass
from cutlass import cute
from cutlass.backend import utils
from cutlass.cute import from_dlpack


class SGemm:
    def __init__(
        self,
        cta_tiler: tuple[int, int, int] = (64, 64, 32),
        num_stages: int = 3,
        num_threads: int = 128,
    ):
        self._cta_tiler = cta_tiler
        self._num_stages = num_stages
        self._num_threads = num_threads
        assert num_threads > 0, "needs at least one thread"
        assert num_threads % 16 == 0, "multiples of 16 required for MMA thread layout"

        self._bM, self._bN, self._bK = self._cta_tiler
        assert self._bM % 16 == 0, "multiple of 16 required for tile dimension M"
        assert self._bN % 16 == 0, "multiple of 16 required for tile dimension N"
        assert self._num_stages >= 3, "num_stages must be greater than or equal to 3"

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA)
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB)
        self.c_major_mode = utils.LayoutEnum.from_tensor(mC)

        padding_a = 4 if self.a_major_mode == utils.LayoutEnum.ROW_MAJOR else 0
        padding_b = 4 if self.b_major_mode == utils.LayoutEnum.ROW_MAJOR else 0
        
        sA_layout = cute.make_layout(
            (self._bM, self._bK, self._num_stages),
            stride=(1, (self._bM + padding_a), self._bK * (self._bM + padding_a)),
        )
        sB_layout = cute.make_layout(
            (self._bN, self._bK, self._num_stages),
            stride=(1, (self._bN + padding_b), self._bK * (self._bN + padding_b)),
        )

        smem_size = cute.size_in_bytes(mA.element_type, sA_layout) + cute.size_in_bytes(
            mB.element_type, sB_layout
        )

        if self.a_major_mode == utils.LayoutEnum.COL_MAJOR:
            num_vectorized = 4 if (mA.layout.max_alignment % 16 == 0) else 1
            atom_async_copy_A = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mA.element_type,
                num_bits_per_copy=mA.element_type.width * num_vectorized,
            )
            major_mode_size = self._bM // num_vectorized
            tA = cute.make_layout(
                (major_mode_size, self._num_threads // major_mode_size),
                stride=(1, major_mode_size),
            )
            vA = cute.make_layout((num_vectorized, 1))
        else:
            num_vectorized = 4 if (mA.layout.max_alignment % 16 == 0) else 1
            atom_async_copy_A = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mA.element_type,
                num_bits_per_copy=mA.element_type.width * num_vectorized,
            )
            major_mode_size = self._bK // num_vectorized
            tA = cute.make_layout(
                (self._num_threads // major_mode_size, major_mode_size),
                stride=(major_mode_size, 1),
            )
            vA = cute.make_layout((1, num_vectorized))

        if self.b_major_mode == utils.LayoutEnum.COL_MAJOR:
            num_vectorized = 4 if (mB.layout.max_alignment % 16 == 0) else 1
            atom_async_copy_B = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mB.element_type,
                num_bits_per_copy=mB.element_type.width * num_vectorized,
            )
            major_mode_size = self._bN // num_vectorized
            tB = cute.make_layout(
                (major_mode_size, self._num_threads // major_mode_size),
                stride=(1, major_mode_size),
            )
            vB = cute.make_layout((num_vectorized, 1))
        else:
            num_vectorized = 4 if (mB.layout.max_alignment % 16 == 0) else 1
            atom_async_copy_B = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mB.element_type,
                num_bits_per_copy=mB.element_type.width * num_vectorized,
            )
            major_mode_size = self._bK // num_vectorized
            tB = cute.make_layout(
                (self._num_threads // major_mode_size, major_mode_size),
                stride=(major_mode_size, 1),
            )
            vB = cute.make_layout((1, num_vectorized))

        tiled_copy_A = cute.make_tiled_copy_tv(atom_async_copy_A, tA, vA)
        tiled_copy_B = cute.make_tiled_copy_tv(atom_async_copy_B, tB, vB)

        atoms_layout = cute.make_layout(
            (self._num_threads // 16, 16, 1), stride=(16, 1, 0)
        )
        if self.c_major_mode == utils.LayoutEnum.COL_MAJOR:
            atoms_layout = cute.make_layout(
                (16, self._num_threads // 16, 1), stride=(1, 16, 0)
            )
            
        op = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
        permutation_tiler_M = cute.make_layout(
            (atoms_layout.shape[0], 4), stride=(4, 1)
        )
        permutation_tiler_N = cute.make_layout(
            (atoms_layout.shape[1], 4), stride=(4, 1)
        )
        tiled_mma = cute.make_tiled_mma(
            op,
            atoms_layout,
            permutation_mnk=(permutation_tiler_M, permutation_tiler_N, None),
        )

        grid_dim = *cute.ceil_div(mC.shape, (self._bM, self._bN)), 1

        self.kernel(
            mA,
            mB,
            mC,
            sA_layout,
            sB_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_mma,
            epilogue_op,
        ).launch(
            grid=grid_dim,
            block=[cute.size(atoms_layout), 1, 1],
            smem=smem_size,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sA_layout: cute.Layout,
        sB_layout: cute.Layout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        tidx, tidy, tidz = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        tiler_coord = (bidx, bidy, None)
        thr_mma = tiled_mma.get_slice(tidx)

        gA = cute.local_tile(
            mA, tiler=self._cta_tiler, coord=tiler_coord, proj=(1, None, 1)
        )
        gB = cute.local_tile(
            mB, tiler=self._cta_tiler, coord=tiler_coord, proj=(None, 1, 1)
        )
        gC = cute.local_tile(
            mC, tiler=self._cta_tiler, coord=tiler_coord, proj=(1, 1, None)
        )

        residue_k = mA.shape[1] - cutlass.Int32(self._bK) * gA.shape[2]
        gA = cute.domain_offset((0, residue_k, 0), gA)
        gB = cute.domain_offset((0, residue_k, 0), gB)

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
        sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)
        
        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        tBgB = thr_copy_B.partition_S(gB)
        tBsB = thr_copy_B.partition_D(sB)

        mcA = cute.make_identity_tensor(mA.shape)
        mcB = cute.make_identity_tensor(mB.shape)
        cA = cute.local_tile(
            mcA, tiler=self._cta_tiler, coord=tiler_coord, proj=(1, None, 1)
        )
        cB = cute.local_tile(
            mcB, tiler=self._cta_tiler, coord=tiler_coord, proj=(None, 1, 1)
        )
        cA = cute.domain_offset((0, residue_k, 0), cA)
        cB = cute.domain_offset((0, residue_k, 0), cB)
        
        tAcA = thr_copy_A.partition_S(cA)
        tBcB = thr_copy_B.partition_S(cB)

        tApA = cute.make_fragment(
            cute.make_layout(
                (
                    tAsA.shape[0][1],
                    cute.size(tAsA, mode=[1]),
                    cute.size(tAsA, mode=[2]),
                ),
                stride=(cute.size(tAsA, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        tBpB = cute.make_fragment(
            cute.make_layout(
                (
                    tBsB.shape[0][1],
                    cute.size(tBsB, mode=[1]),
                    cute.size(tBsB, mode=[2]),
                ),
                stride=(cute.size(tBsB, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        tApA_residue_k = cute.make_fragment(
            cute.make_layout(
                (
                    tAsA.shape[0][1],
                    cute.size(tAsA, mode=[1]),
                    cute.size(tAsA, mode=[2]),
                ),
                stride=(
                    cute.size(tAsA, mode=[1]) * cute.size(tAsA, mode=[2]),
                    cute.size(tAsA, mode=[2]),
                    1,
                ),
            ),
            cutlass.Boolean,
        )
        tBpB_residue_k = cute.make_fragment(
            cute.make_layout(
                (
                    tBsB.shape[0][1],
                    cute.size(tBsB, mode=[1]),
                    cute.size(tBsB, mode=[2]),
                ),
                stride=(
                    cute.size(tBsB, mode=[1]) * cute.size(tBsB, mode=[2]),
                    cute.size(tBsB, mode=[2]),
                    1,
                ),
            ),
            cutlass.Boolean,
        )

        for rest_v in range(tApA.shape[0]):
            for m in range(tApA.shape[1]):
                tApA[rest_v, m, 0] = cute.elem_less(
                    tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0]
                )
        for rest_v in range(tBpB.shape[0]):
            for n in range(tBpB.shape[1]):
                tBpB[rest_v, n, 0] = cute.elem_less(
                    tBcB[(0, rest_v), n, 0, 0][0], mB.shape[0]
                )

        for rest_v in range(tApA_residue_k.shape[0]):
            for m in range(tApA_residue_k.shape[1]):
                for k in range(tApA_residue_k.shape[2]):
                    coord_A = tAcA[(0, rest_v), m, k, 0]
                    tApA_residue_k[rest_v, m, k] = cute.elem_less(
                        (coord_A[0], cutlass.Int32(-1)), (mA.shape[0], coord_A[1])
                    )
        for rest_v in range(tBpB_residue_k.shape[0]):
            for n in range(tBpB_residue_k.shape[1]):
                for k in range(tBpB_residue_k.shape[2]):
                    coord_B = tBcB[(0, rest_v), n, k, 0]
                    tBpB_residue_k[rest_v, n, k] = cute.elem_less(
                        (coord_B[0], cutlass.Int32(-1)), (mB.shape[0], coord_B[1])
                    )

        k_pipe_max = cute.size(tAsA, mode=[3])
        k_tile_count = cute.size(tAgA, mode=[3])
        gmem_pipe_read = cutlass.Int32(0)
        
        cute.copy(
            tiled_copy_A,
            tAgA[None, None, None, gmem_pipe_read],
            tAsA[None, None, None, 0],
            pred=tApA_residue_k,
        )
        cute.copy(
            tiled_copy_B,
            tBgB[None, None, None, gmem_pipe_read],
            tBsB[None, None, None, 0],
            pred=tBpB_residue_k,
        )
        cute.arch.cp_async_commit_group()
        gmem_pipe_read = (
            gmem_pipe_read + 1
            if gmem_pipe_read + 1 < k_tile_count
            else cutlass.Int32(0)
        )

        for k_tile in range(1, k_pipe_max - 1):
            if k_tile < k_tile_count:
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, None, gmem_pipe_read],
                    tAsA[None, None, None, k_tile],
                    pred=tApA,
                )
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, None, gmem_pipe_read],
                    tBsB[None, None, None, k_tile],
                    pred=tBpB,
                )

            gmem_pipe_read = (
                gmem_pipe_read + 1
                if gmem_pipe_read + 1 < k_tile_count
                else cutlass.Int32(0)
            )
            cute.arch.cp_async_commit_group()

        if k_tile_count < k_pipe_max:
            for rest_v in range(tApA.shape[0]):
                for m in range(tApA.shape[1]):
                    tApA[rest_v, m, 0] = cutlass.Boolean(0)
            for rest_v in range(tBpB.shape[0]):
                for n in range(tBpB.shape[1]):
                    tBpB[rest_v, n, 0] = cutlass.Boolean(0)

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)

        smem_pipe_read = cutlass.Int32(0)
        smem_pipe_write = cutlass.Int32(k_pipe_max - 1)

        tCsA_p = tCsA[None, None, None, smem_pipe_read]
        tCsB_p = tCsB[None, None, None, smem_pipe_read]

        k_block_max = cute.size(tCrA, mode=[2])

        if k_block_max > 1:
            cute.arch.cp_async_wait_group(k_pipe_max - 2)
            cute.arch.barrier()
            cute.autovec_copy(tCsA_p[None, None, 0], tCrA[None, None, 0])
            cute.autovec_copy(tCsB_p[None, None, 0], tCrB[None, None, 0])

        for _ in cutlass.range_dynamic(k_tile_count, unroll=1):
            for k_block in range(k_block_max):
                if k_block == k_block_max - 1:
                    tCsA_p = tCsA[None, None, None, smem_pipe_read]
                    tCsB_p = tCsB[None, None, None, smem_pipe_read]
                    cute.arch.cp_async_wait_group(k_pipe_max - 2)
                    cute.arch.barrier()

                k_block_next = (k_block + 1) % k_block_max
                cute.autovec_copy(
                    tCsA_p[None, None, k_block_next],
                    tCrA[None, None, k_block_next],
                )
                cute.autovec_copy(
                    tCsB_p[None, None, k_block_next],
                    tCrB[None, None, k_block_next],
                )

                if k_block == 0:
                    cute.copy(
                        tiled_copy_A,
                        tAgA[None, None, None, gmem_pipe_read],
                        tAsA[None, None, None, smem_pipe_write],
                        pred=tApA,
                    )

                cute.gemm(
                    tiled_mma,
                    tCrC,
                    tCrA[None, None, k_block],
                    tCrB[None, None, k_block],
                    tCrC,
                )

                if k_block == 0:
                    cute.copy(
                        tiled_copy_B,
                        tBgB[None, None, None, gmem_pipe_read],
                        tBsB[None, None, None, smem_pipe_write],
                        pred=tBpB,
                    )
                    cute.arch.cp_async_commit_group()
                    smem_pipe_write = smem_pipe_read
                    smem_pipe_read = smem_pipe_read + 1
                    if smem_pipe_read == k_pipe_max:
                        smem_pipe_read = cutlass.Int32(0)
                    gmem_pipe_read = (
                        gmem_pipe_read + 1
                        if gmem_pipe_read + 1 < k_tile_count
                        else cutlass.Int32(1)
                    )

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()
        
        tCrC.store(epilogue_op(tCrC.load()))

        cC = cute.make_identity_tensor(gC.shape)
        tCpC = thr_mma.partition_C(cC)
        predC = cute.make_fragment(tCrC.layout, cutlass.Boolean)
        residue_m = mC.shape[0] - cutlass.Int32(self._bM) * bidx
        residue_n = mC.shape[1] - cutlass.Int32(self._bN) * bidy
        
        for i in range(cute.size(tCrC.shape)):
            predC[i] = cute.elem_less(tCpC[i], (residue_m, residue_n))
            
        numIterM = cute.size(tCrC, mode=[1])
        numIterN = cute.size(tCrC, mode=[2])
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
        cute.copy(atom, tCrC, tCgC, pred=predC)

def main():
    M, N, K = 256, 256, 256
    print(f"Multiplying {M}x{K} and {K}x{N} matrices using CuTe DSL...")

    a = cp.random.randn(M, K).astype(cp.float16)
    b = cp.random.randn(K, N).astype(cp.float16)
    c = cp.zeros((M, N), dtype=cp.float32)

    a_tensor = from_dlpack(a)
    b_tensor = from_dlpack(b)
    c_tensor = from_dlpack(c)

    # Compile the kernel
    sgemm = SGemm(cta_tiler=(64, 64, 32), num_stages=3, num_threads=128)
    print("Compiling CuTe kernel JIT...")
    start_compile = time.time()
    gemm_fn = cute.compile(sgemm, a_tensor, b_tensor, c_tensor)
    print(f"Compiled in {time.time() - start_compile:.4f} seconds.")

    # Launch the kernel
    print("Launching CuTe kernel...")
    gemm_fn(a_tensor, b_tensor, c_tensor)
    cp.cuda.get_current_stream().synchronize()
    print("Execution complete.")

    # Verify against numpy on CPU
    a_np = a.get().astype('float32')
    b_np = b.get().astype('float32')
    expected = a_np @ b_np

    cp.testing.assert_allclose(c, expected, atol=1e-1, rtol=1e-2)
    print("SUCCESS! CuTe DSL Matmul matched NumPy reference.")

if __name__ == "__main__":
    main()
