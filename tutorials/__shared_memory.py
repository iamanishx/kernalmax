"""
SHARED MEMORY  --  the block-wide scratchpad, and the write/sync/read dance

How this differs from the warp shuffle (__shfl_down_sync.py)
-----------------------------------------------------------
    warp shuffle : register -> register, NO memory, but ONLY within one warp (32 lanes)
    shared memory: every thread in the BLOCK can read/write a common on-chip buffer,
                   ANY block size, but you MUST use a barrier (sync_threads) between
                   a write and a read so everyone sees the finished data.

Shared memory lives on-chip (much faster than global), is private to one CTA
(thread block), and is programmer-managed. It is how threads that are NOT in the
same warp communicate.

---------------
    write to shared  -->  cute.arch.sync_threads()  -->  read from shared

Skip that barrier and you get a race: a thread may read a slot before the owning
thread has written it. Almost every "sometimes wrong" shared-memory bug is a
missing sync_threads.

This tutorial shows two uses in one kernel:
  1. REVERSE  : thread tidx writes s[tidx], then reads s[N-1-tidx]  (cross-thread read)
  2. BLOCK SUM: a shared-memory tree reduction over the whole block (the multi-warp
                version of the shuffle fold; here it works for any block size)
"""

import cuda.bindings.driver as cuda
import cupy as cp
import cutlass
import numpy as np
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

N = 256  


class SharedMemDemo:
    @cute.kernel
    def kernel(self, mX: cute.Tensor, mRev: cute.Tensor, mSum: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()

        smem = cutlass.utils.SmemAllocator()
        s = smem.allocate_tensor(cutlass.Float32, cute.make_layout(N), byte_alignment=4)

        s[tidx] = mX[tidx]  
        cute.arch.sync_threads() 
        mRev[tidx] = s[N - 1 - tidx]


        stride = N // 2
        while stride > 0:
            if tidx < stride:                       
                s[tidx] = s[tidx] + s[tidx + stride]
            cute.arch.sync_threads()               
            stride //= 2

        if tidx == 0:
            mSum[0] = s[0]

    @cute.jit
    def __call__(self, mX: cute.Tensor, mRev: cute.Tensor, mSum: cute.Tensor,
                 stream: cuda.CUstream):
        self.kernel(mX, mRev, mSum).launch(
            grid=[1, 1, 1],
            block=[N, 1, 1],
            stream=stream,
        )


def main():
    x_np = np.random.randn(N).astype(np.float32)
    x = cp.asarray(x_np)
    rev = cp.zeros(N, dtype=cp.float32)
    out_sum = cp.zeros(1, dtype=cp.float32)

    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)
    SharedMemDemo()(from_dlpack(x), from_dlpack(rev), from_dlpack(out_sum), stream)
    cp.cuda.get_current_stream().synchronize()

    rev_ok = np.array_equal(rev.get(), x_np[::-1])
    print(f"reverse correct : {rev_ok}")
    print(f"block sum gpu={float(out_sum.get()[0]):.4f}  numpy={x_np.sum():.4f}")

    assert rev_ok, "reverse failed (likely a missing sync_threads)"
    np.testing.assert_allclose(out_sum.get()[0], x_np.sum(), atol=1e-3)
    print("SUCCESS! shared-memory reverse and block reduction match NumPy.")


if __name__ == "__main__":
    main()
