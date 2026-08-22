import cuda.bindings.driver as cuda
import cupy as cp
import cutlass
import numpy as np
from cutlass import cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

N = 1024
THREADS = 256
VPT = N // THREADS

class CpAsyncDemo:
    @cute.kernel
    def kernel(self, mX: cute.Tensor, mO: cute.Tensor, tiled_copy: cute.TiledCopy):
        tidx, _, _ = cute.arch.thread_idx()

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(cutlass.Float32, cute.make_layout(N), byte_alignment=4)

        thr = tiled_copy.get_slice(tidx)
        gS = thr.partition_S(mX)        
        sD = thr.partition_D(sX)  

        # ---- the async dance ----
        cute.copy(tiled_copy, gS, sD)        # 1) fire async global -> shared
        cute.arch.cp_async_commit_group()    # 2) bundle into a group
        cute.arch.cp_async_wait_group(0)     # 3) wait until ALL groups finish
        cute.arch.sync_threads()             # 4) block barrier: smem visible to all

        # ---- consume the loaded data ----
        # write 2 * sX back to global; if cp.async didn't actually land, this fails
        for v in cutlass.range_constexpr(VPT):
            i = tidx + v * THREADS
            mO[i] = sX[i] * 2.0

    @cute.jit
    def __call__(self, mX: cute.Tensor, mO: cute.Tensor, stream: cuda.CUstream):
        # ONLY DIFFERENCE from __copy_atom.py: CopyG2SOp instead of CopyUniversalOp.
        # Same atom builder, same tiled-copy builder, same num_bits knob.
        atom = cute.make_copy_atom(cpasync.CopyG2SOp(),
                                   cutlass.Float32, num_bits_per_copy=32)
        tiled_copy = cute.make_tiled_copy_tv(
            atom,
            cute.make_layout(THREADS),   
            cute.make_layout(VPT), 
        )
        self.kernel(mX, mO, tiled_copy).launch(
            grid=[1, 1, 1], block=[THREADS, 1, 1], stream=stream)


def main():
    x_np = np.random.randn(N).astype(np.float32)
    x = cp.asarray(x_np)
    o = cp.zeros(N, dtype=cp.float32)

    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)
    CpAsyncDemo()(from_dlpack(x), from_dlpack(o), stream)
    cp.cuda.get_current_stream().synchronize()

    err = float(np.abs(o.get() - 2.0 * x_np).max())
    print(f"cp.async global->shared, then 2x writeback. max error: {err}")
    np.testing.assert_allclose(o.get(), 2.0 * x_np, atol=1e-6)
    print("SUCCESS! cp.async load committed before the read.")


if __name__ == "__main__":
    main()
