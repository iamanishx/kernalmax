"""
COPY ATOMS and TILED COPY  --  the way to move data in CuTe

The pieces
----------
  copy_atom  = make_copy_atom(op, dtype, num_bits_per_copy=W)
                 ONE copy operation. W = bits moved per instruction
                 (32 = one fp32; 128 = four fp32 at once, "vectorized").
  tiled_copy = make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
                 spread that atom over THREADS x VALUES-per-thread. The TV layout
                 it builds decides WHICH elements each thread touches.
  thr        = tiled_copy.get_slice(tidx)         # this thread's view
  tS = thr.partition_S(src)                        # this thread's slice of source
  tD = thr.partition_D(dst)                        # this thread's slice of dest
  frag = make_fragment_like(tS)                    # register buffer of that shape
  cute.copy(tiled_copy, tS, frag)                  # global -> registers
  cute.copy(tiled_copy, frag, tD)                  # registers -> global

COALESCING (the key performance idea)
-------------------------------------
The TV layout this builds is (256,4):(1,256): thread stride 1. That means
CONSECUTIVE THREADS touch CONSECUTIVE ADDRESSES. When a warp's 32 lanes all read
32 neighboring addresses, the hardware merges them into ONE memory transaction.
That is "coalescing", and it is the single most important global-memory rule.

  thread 0 owns elements 0, 256, 512, 768
  thread 1 owns elements 1, 257, 513, 769
  thread 2 owns elements 2, 258, 514, 770
  ...
  so at step 0 the warp reads addresses 0,1,2,...,31  -> one fat transaction. Good.

VECTORIZATION (the other axis, and the tradeoff)
------------------------------------------------
A 128-bit atom copies 4 fp32 per instruction, but only if a thread's 4 elements
are CONTIGUOUS in memory (static stride 1). In the coalesced layout above, a
thread's 4 elements are 256 apart, NOT contiguous, so a 128-bit copy is rejected.
To vectorize you need a layout where each thread owns a contiguous chunk, which
trades against simple coalescing. Real kernels arrange a 2D tiling so a warp is
both coalesced AND vectorized. For now: know that num_bits_per_copy is the knob,
and that coalescing (threads) and vectorization (per-thread contiguity) are
different requirements on the layout.

Run:  python3 tutorials/__copy_atom.py
"""

import cuda.bindings.driver as cuda
import cupy as cp
import cutlass
import cutlass.cute as cute
import numpy as np
from cutlass.cute.runtime import from_dlpack

N = 1024
THREADS = 256
VPT = N // THREADS


class CopyDemo:
    @cute.kernel
    def kernel(self, mX: cute.Tensor, mO: cute.Tensor, tiled_copy: cute.TiledCopy):
        tidx, _, _ = cute.arch.thread_idx()

        thr = tiled_copy.get_slice(tidx)
        tXgX = thr.partition_S(mX)
        _tXgO = thr.partition_D(mO)

        frag = cute.make_fragment_like(tXgX)
        cute.copy(tiled_copy, tXgX, frag)

        frag.store(frag.load() * 2.0)

           # REGISTERS -> GLOBAL (coalesced store)

    @cute.jit
    def __call__(self, mX: cute.Tensor, mO: cute.Tensor, stream: cuda.CUstream):
        # one 32-bit copy per element, tiled over 256 threads x 4 values each
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=32
        )
        tiled_copy = cute.make_tiled_copy_tv(
            atom,
            cute.make_layout(THREADS),
            cute.make_layout(VPT),
        )
        self.kernel(mX, mO, tiled_copy).launch(
            grid=[1, 1, 1], block=[THREADS, 1, 1], stream=stream
        )


def main():
    x_np = np.random.randn(N).astype(np.float32)
    x = cp.asarray(x_np)
    o = cp.zeros(N, dtype=cp.float32)

    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)
    CopyDemo()(from_dlpack(x), from_dlpack(o), stream)
    cp.cuda.get_current_stream().synchronize()

    err = float(np.abs(o.get() - 2.0 * x_np).max())
    print(f"tiled copy + scale-by-2 max error: {err}")
    np.testing.assert_allclose(o.get(), 2.0 * x_np, atol=1e-6)
    print("SUCCESS! global -> registers -> global via tiled copy matched NumPy.")


if __name__ == "__main__":
    main()
