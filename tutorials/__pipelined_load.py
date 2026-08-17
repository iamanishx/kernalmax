import cupy as cp
import cutlass
import cutlass.cute.nvgpu.cpasync as cpa
import numpy as np
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

N = 1024
THREADS = 32
VPT = 4
TILE = THREADS * VPT       # 128 elements per K-tile
NUM_TILES = N // TILE      # 8
STAGES = 3                 # ring depth; need >= 2; 3 is the typical sweet spot


class PipelinedY2XPlus1:
    @cute.kernel
    def kernel(self, mX: cute.Tensor, mO: cute.Tensor,
               tc_load: cute.TiledCopy, tc_store: cute.TiledCopy):
        tidx, _, _ = cute.arch.thread_idx()

        gX = cute.local_tile(mX, tiler=(TILE,), coord=(None,))
        gO = cute.local_tile(mO, tiler=(TILE,), coord=(None,))

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(cutlass.Float32,
                                  cute.make_layout((TILE, STAGES)),
                                  byte_alignment=16)

        # tXgX shape: ((vals_per_copy, copies), 1, NUM_TILES)
        # tXsX shape: ((vals_per_copy, copies), 1, STAGES)
        # tOgO shape: ((vals_per_copy, copies), 1, NUM_TILES)
        # The third mode is what we cycle through.
        thr_l = tc_load.get_slice(tidx)
        thr_s = tc_store.get_slice(tidx)
        tXgX = thr_l.partition_S(gX)
        tXsX = thr_l.partition_D(sX)
        tOgO = thr_s.partition_D(gO)

        # PROLOGUE: kick off STAGES-1 = 2 loads BEFORE entering the loop.
        # After this, two cp.async batches are in flight.
        for k in cutlass.range_constexpr(STAGES - 1):
            cute.copy(tc_load, tXgX[None, None, k], tXsX[None, None, k])
            cute.arch.cp_async_commit_group()

        # one iteration per K-tile.
        for k in cutlass.range_constexpr(NUM_TILES):
            # Wait until at most STAGES-2 = 1 batch is still pending.
            cute.arch.cp_async_wait_group(STAGES - 2)
            cute.arch.sync_threads()

            stage = k % STAGES

            f = cute.make_fragment_like(tXsX[None, None, 0])
            cute.autovec_copy(tXsX[None, None, stage], f)
            v = f.load()
            f.store(v * 2.0 + 1.0)               
            cute.copy(tc_store, f, tOgO[None, None, k])

            # ---- PREFETCH the tile that's two steps ahead, into the ring ----
            next_k = k + STAGES - 1
            if next_k < NUM_TILES:
                cute.copy(tc_load,
                          tXgX[None, None, next_k],
                          tXsX[None, None, next_k % STAGES])
            cute.arch.cp_async_commit_group()

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

    @cute.jit
    def __call__(self, mX: cute.Tensor, mO: cute.Tensor):
        # Two atoms because the directions differ:
        # global -> shared uses cp.async; register -> global is plain store.
        atom_load  = cute.make_copy_atom(cpa.CopyG2SOp(),
                                         cutlass.Float32, num_bits_per_copy=32)
        atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),
                                         cutlass.Float32, num_bits_per_copy=32)
        tc_load  = cute.make_tiled_copy_tv(atom_load,
                                           cute.make_layout(THREADS),
                                           cute.make_layout(VPT))
        tc_store = cute.make_tiled_copy_tv(atom_store,
                                           cute.make_layout(THREADS),
                                           cute.make_layout(VPT))
        self.kernel(mX, mO, tc_load, tc_store).launch(
            grid=[1, 1, 1], block=[THREADS, 1, 1])


def main():
    print(f"Pipelined y = 2*x + 1, N={N} (= {NUM_TILES} tiles of {TILE})")
    print(f"Ring depth: STAGES={STAGES}  →  {STAGES-1} loads in flight at all times\n")

    x_np = np.random.randn(N).astype(np.float32)
    x = cp.asarray(x_np)
    o = cp.zeros(N, dtype=cp.float32)

    PipelinedY2XPlus1()(from_dlpack(x), from_dlpack(o))
    cp.cuda.get_current_stream().synchronize()

    err = float(np.abs(o.get() - (2.0 * x_np + 1.0)).max())
    print(f"max abs err vs NumPy: {err:.2e}")
    np.testing.assert_allclose(o.get(), 2.0 * x_np + 1.0, atol=1e-6)
    print("\nSUCCESS! Three-stage cp.async ring buffer hid the load latency behind compute.")
    print("Real GEMMs apply this exact pattern to the K-loop, just with bigger tiles and tensor-core MMAs as the 'compute'.")


if __name__ == "__main__":
    main()
