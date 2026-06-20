import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import cupy as cp
import numpy as np

N = 1000
THREADS = 128
VPT = 8
TILE = THREADS * VPT


class Y2XPlus1:
    @cute.kernel
    def kernel(self, mX: cute.Tensor, mO: cute.Tensor,
               tc: cute.TiledCopy, n_valid: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        thr = tc.get_slice(tidx)
        gS = thr.partition_S(mX)
        gD = thr.partition_D(mO)       
        
        cX = cute.make_identity_tensor(mX.shape)

        tCc = thr.partition_S(cX)

        pred = cute.make_fragment_like(gS, cutlass.Boolean)
        for i in cutlass.range_constexpr(cute.size(pred)):
            pred[i] = cute.elem_less(tCc[i][0], n_valid)

        f = cute.make_fragment_like(gS)
        f.fill(0.0)              
        cute.copy(tc, gS, f, pred=pred)

        v = f.load()
        f.store(v * 2.0 + 1.0)

        cute.copy(tc, f, gD, pred=pred)

    @cute.jit
    def __call__(self, mX: cute.Tensor, mO: cute.Tensor,
                 n_valid: cutlass.Constexpr):
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),
                                   cutlass.Float32, num_bits_per_copy=32)
        tc = cute.make_tiled_copy_tv(atom,
                                     cute.make_layout(THREADS),
                                     cute.make_layout(VPT))
        self.kernel(mX, mO, tc, n_valid).launch(
            grid=[1, 1, 1], block=[THREADS, 1, 1])


def main():
    print(f"y = 2*x + 1 over N={N} valid elements, single tile of {TILE}")
    print(f"(boundary case: TILE - N = {TILE - N} OOB elements predicated out)\n")

    x_np = np.random.randn(TILE).astype(np.float32)
    x = cp.asarray(x_np)
    o = cp.zeros(TILE, dtype=cp.float32)

    Y2XPlus1()(from_dlpack(x), from_dlpack(o), N)
    cp.cuda.get_current_stream().synchronize()

    out = o.get()
    ref = 2.0 * x_np[:N] + 1.0
    err = float(np.abs(out[:N] - ref).max())
    tail_max = float(np.abs(out[N:]).max())

    print(f"valid region [0, {N}):    max abs err vs NumPy = {err:.2e}")
    print(f"tail region   [{N}, {TILE}):  max abs value     = {tail_max:.2e}   (should be 0)")
    np.testing.assert_allclose(out[:N], ref, atol=1e-6)
    assert tail_max == 0.0, "tail not 0 — predicated store leaked"
    print("\nSUCCESS! predication kept the tail untouched while the valid region computed correctly.")


if __name__ == "__main__":
    main()
