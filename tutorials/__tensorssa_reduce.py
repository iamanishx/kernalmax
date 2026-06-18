import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import cupy as cp
import numpy as np
import cuda.bindings.driver as cuda

N = 1024
THREADS = 256
VPT = N // THREADS        
WARPS = THREADS // 32      


class SsaReduce:
    @cute.kernel
    def kernel(self, mX: cute.Tensor, mO: cute.Tensor, mSum: cute.Tensor,
               tiled_copy: cute.TiledCopy):
        tidx, _, _ = cute.arch.thread_idx()

        thr = tiled_copy.get_slice(tidx)
        gS  = thr.partition_S(mX)
        gD  = thr.partition_D(mO)
        f   = cute.make_fragment_like(gS)
        cute.copy(tiled_copy, gS, f)

        v = f.load()                 
        v = v * 2.0 + 1.0              
        f.store(v)                    
        cute.copy(tiled_copy, f, gD)   

        my_sum = v.reduce(cute.ReductionOp.ADD,
                          cutlass.Float32(0.0),
                          reduction_profile=0)

        offset = 16
        while offset > 0:
            my_sum = my_sum + cute.arch.shuffle_sync_down(my_sum, offset)
            offset //= 2

        smem = cutlass.utils.SmemAllocator()
        sWarpSums = smem.allocate_tensor(cutlass.Float32, cute.make_layout(32),
                                         byte_alignment=4)
        lane = tidx % 32
        warp = tidx // 32
        if lane == 0:
            sWarpSums[warp] = my_sum
        # zero-pad slots [WARPS..31] so warp 0 can shuffle-reduce across all 32
        if tidx >= WARPS and tidx < 32:
            sWarpSums[tidx] = cutlass.Float32(0.0)
        cute.arch.sync_threads()

        if warp == 0:
            x = sWarpSums[lane]
            offset = 16
            while offset > 0:
                x = x + cute.arch.shuffle_sync_down(x, offset)
                offset //= 2
            if lane == 0:
                mSum[0] = x          

    @cute.jit
    def __call__(self, mX: cute.Tensor, mO: cute.Tensor, mSum: cute.Tensor,
                 stream: cuda.CUstream):
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),
                                   cutlass.Float32, num_bits_per_copy=32)
        tc = cute.make_tiled_copy_tv(atom,
                                     cute.make_layout(THREADS),
                                     cute.make_layout(VPT))
        self.kernel(mX, mO, mSum, tc).launch(
            grid=[1, 1, 1], block=[THREADS, 1, 1], stream=stream)


def main():
    print(f"TensorSSA: load -> v*2+1 -> store, then BLOCK-SUM all {N} results.")
    x_np = np.random.randn(N).astype(np.float32)
    x = cp.asarray(x_np)
    o = cp.zeros(N, dtype=cp.float32)
    s = cp.zeros(1, dtype=cp.float32)

    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)
    SsaReduce()(from_dlpack(x), from_dlpack(o), from_dlpack(s), stream)
    cp.cuda.get_current_stream().synchronize()

    ref_o = 2.0 * x_np + 1.0
    ref_s = float(ref_o.sum())
    err_o = float(np.abs(o.get() - ref_o).max())
    err_s = abs(float(s.get()[0]) - ref_s)
    print(f"transform max err : {err_o}")
    print(f"block sum gpu={float(s.get()[0]):.6f}  ref={ref_s:.6f}  diff={err_s:.2e}")
    np.testing.assert_allclose(o.get(), ref_o, atol=1e-6)
    np.testing.assert_allclose(float(s.get()[0]), ref_s, atol=1e-3)
    print("SUCCESS! TensorSSA arithmetic + .reduce() + shuffle/smem combine all match.")


if __name__ == "__main__":
    main()
