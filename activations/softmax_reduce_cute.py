# shifted_z = z - np.max(z)
# exp_z = np.exp(shifted_z)
# return exp_z / np.sum(exp_z)

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import cupy as cp
import numpy as np
import cuda.bindings.driver as cuda

M, N = 1024, 1024
THREADS = 256
VPT = N // THREADS        
WARPS = THREADS // 32      
NEG_INF = -3.4e38

class SoftmaxReduce:
    @cute.kernel
    def kernel(self, mX: cute.Tensor, mO: cute.Tensor, tiled_copy: cute.TiledCopy):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        gX = mX[(bidx, None)]
        gO = mO[(bidx, None)]

        thr = tiled_copy.get_slice(tidx)
        gS = thr.partition_S(gX)
        gD = thr.partition_D(gO)

        f = cute.make_fragment_like(gS)
        cute.copy(tiled_copy, gS, f)


        smem = cutlass.utils.SmemAllocator()
        s = smem.allocate_tensor(cutlass.Float32, cute.make_layout(32),
                                 byte_alignment=4)

        lane = tidx % 32
        warp = tidx // 32

        # PASS 1: ROW MAX  (reduce(MAX) -> warp shuffle MAX -> SMEM combine -> broadcast)
        v = f.load()                                                       # TensorSSA (4 fp32)
        my_max = v.reduce(cute.ReductionOp.MAX,
                          cutlass.Float32(NEG_INF),
                          reduction_profile=0)                             

        offset = 16
        while offset > 0:
            other = cute.arch.shuffle_sync_down(my_max, offset)
            if other > my_max:
                my_max = other
            offset //= 2

        if lane == 0:
            s[warp] = my_max
        if tidx >= WARPS and tidx < 32:
            s[tidx] = cutlass.Float32(NEG_INF)
        cute.arch.sync_threads()

        if warp == 0:
            x = s[lane]
            offset = 16
            while offset > 0:
                other = cute.arch.shuffle_sync_down(x, offset)
                if other > x:
                    x = other
                offset //= 2
            if lane == 0:
                s[0] = x                                                   
        cute.arch.sync_threads()

        row_max = s[0]                                                    
        cute.arch.sync_threads()                                          
        
        # PASS 2: ROW SUM of exp(v - row_max)  (reduce(ADD) -> ... -> broadcast)
        v_exp = cute.exp(v - row_max)                                    
        my_sum = v_exp.reduce(cute.ReductionOp.ADD,
                              cutlass.Float32(0.0),
                              reduction_profile=0)                        

        offset = 16
        while offset > 0:
            my_sum = my_sum + cute.arch.shuffle_sync_down(my_sum, offset)
            offset //= 2

        if lane == 0:
            s[warp] = my_sum
        if tidx >= WARPS and tidx < 32:
            s[tidx] = cutlass.Float32(0.0)
        cute.arch.sync_threads()

        if warp == 0:
            x = s[lane]
            offset = 16
            while offset > 0:
                x = x + cute.arch.shuffle_sync_down(x, offset)
                offset //= 2
            if lane == 0:
                s[0] = x
        cute.arch.sync_threads()

        row_sum = s[0]

        f.store(v_exp / row_sum)                                          
        cute.copy(tiled_copy, f, gD)                                     

    @cute.jit
    def __call__(self, mX: cute.Tensor, mO: cute.Tensor, stream: cuda.CUstream):
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),
                                   cutlass.Float32, num_bits_per_copy=32)
        tc = cute.make_tiled_copy_tv(atom,
                                     cute.make_layout(THREADS),
                                     cute.make_layout(VPT))
        self.kernel(mX, mO, tc).launch(
            grid=[M, 1, 1],
            block=[THREADS, 1, 1],
            stream=stream,
        )


def main():
    print(f"softmax (reduction) over a {M}x{N} matrix, 1 CTA per row, 256 threads/row...")
    x_np = np.random.randn(M, N).astype(np.float32)
    x = cp.asarray(x_np)
    o = cp.zeros((M, N), dtype=cp.float32)

    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)
    SoftmaxReduce()(from_dlpack(x), from_dlpack(o), stream)
    cp.cuda.get_current_stream().synchronize()

    ref = np.exp(x_np - x_np.max(axis=1, keepdims=True))
    ref /= ref.sum(axis=1, keepdims=True)
    err = float(np.abs(o.get() - ref).max())
    print(f"max abs error vs NumPy: {err:.2e}")
    np.testing.assert_allclose(o.get(), ref, atol=1e-5, rtol=1e-4)
    print("SUCCESS! row-wise softmax matches NumPy. Threads cooperated correctly.")


if __name__ == "__main__":
    main()
