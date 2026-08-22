"""
WARP SHUFFLE REDUCTION  --  __shfl_down_sync, the CuTe DSL way

idea
--------
The 32 lanes of a warp can read each other's REGISTERS directly. No shared
memory, no sync_threads, no round trip through any memory. One instruction.

    cute.arch.shuffle_sync_down(v, d)
        -> returns the value of `v` held by lane (myLane + d).
           Lanes near the top (myLane + d >= 32) just keep their own v.

A reduction over 32 lanes is a log2(32) = 5 step "fold". Each step folds the
upper half of the still-active lanes down onto the lower half, so the result
ends up in LANE 0.

    SUM fold:                          MAX fold:
        v += shfl_down(v, 16)              v = max(v, shfl_down(v, 16))
        v += shfl_down(v,  8)              v = max(v, shfl_down(v,  8))
        v += shfl_down(v,  4)              v = max(v, shfl_down(v,  4))
        v += shfl_down(v,  2)              v = max(v, shfl_down(v,  2))
        v += shfl_down(v,  1)              v = max(v, shfl_down(v,  1))
        # lane 0 now holds the sum         # lane 0 now holds the max

Why it matters for DL: this is the inner engine of every fast softmax,
LayerNorm/RMSNorm, and attention reduction. It replaces the shared-memory
tree you wrote in softmax_reduce_cute.py with something faster and barrier-free.

Run:  python3 tutorials/__shfl_down_sync.py
"""

import cuda.bindings.driver as cuda
import cupy as cp
import numpy as np
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

WARP = 32


class WarpReduce:
    @cute.kernel
    def kernel(self, mX: cute.Tensor, mSum: cute.Tensor, mMax: cute.Tensor):
        lane, _, _ = cute.arch.thread_idx()

        v = mX[lane]            
        m = mX[lane]            

        offset = WARP // 2 
        while offset > 0:
            v = v + cute.arch.shuffle_sync_down(v, offset)

            other = cute.arch.shuffle_sync_down(m, offset)
            m = max(m, other)

            if lane == 0:
                cute.printf("  after offset %2d : lane0 sum=%8.4f  max=%8.4f\n",
                            offset, v, m)
            offset //= 2

        if lane == 0:
            mSum[0] = v
            mMax[0] = m

    @cute.jit
    def __call__(self, mX: cute.Tensor, mSum: cute.Tensor, mMax: cute.Tensor,
                 stream: cuda.CUstream):
        self.kernel(mX, mSum, mMax).launch(
            grid=[1, 1, 1],
            block=[WARP, 1, 1],
            stream=stream,
        )


def main():
    x_np = np.random.randn(WARP).astype(np.float32)
    x = cp.asarray(x_np)
    out_sum = cp.zeros(1, dtype=cp.float32)
    out_max = cp.zeros(1, dtype=cp.float32)

    print("32 input values (one per lane):")
    print(np.array2string(x_np, precision=3, max_line_width=80))
    print("watch lane 0 build the answer step by step:")

    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)
    WarpReduce()(from_dlpack(x), from_dlpack(out_sum), from_dlpack(out_max), stream)
    cp.cuda.get_current_stream().synchronize()

    gpu_sum, gpu_max = float(out_sum.get()[0]), float(out_max.get()[0])
    print(f"\ngpu  sum={gpu_sum:.4f}  max={gpu_max:.4f}")
    print(f"numpy sum={x_np.sum():.4f}  max={x_np.max():.4f}")

    np.testing.assert_allclose(gpu_sum, x_np.sum(), atol=1e-4)
    np.testing.assert_allclose(gpu_max, x_np.max(), atol=1e-6)
    print("SUCCESS! warp-shuffle reduction matches NumPy.")


if __name__ == "__main__":
    main()
