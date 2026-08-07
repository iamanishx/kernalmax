"""Verify grid_barrier works across CTAs before restructuring the megakernel."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from ops.reduce import grid_barrier

NCTA = 20
THREADS = 128


class BarrierTest:
    """Each CTA writes its id to gBuf[cta], barriers, then CTA 0 sums all.
    Without a working barrier, CTA 0 would read zeros for late CTAs."""
    @cute.kernel
    def kernel(self, gBuf, gBar, gSum):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()

        if tid == 0:
            gBuf[cta] = cutlass.Float32(cta + 1)

        grid_barrier(gBar, tid, 1, NCTA)

        # every CTA sums the whole buffer — all must see all writes
        if tid == 0:
            acc = cutlass.Float32(0.0)
            for i in cutlass.range(NCTA):
                acc = acc + gBuf[i]
            gSum[cta] = acc

    @cute.jit
    def __call__(self, gBuf, gBar, gSum, stream: cuda.CUstream):
        self.kernel(gBuf, gBar, gSum).launch(
            grid=[NCTA, 1, 1], block=[THREADS, 1, 1], stream=stream)


def main():
    dev = torch.device("cuda")
    gBuf = torch.zeros(NCTA, device=dev, dtype=torch.float32)
    gBar = torch.zeros(1, device=dev, dtype=torch.int32)
    gSum = torch.zeros(NCTA, device=dev, dtype=torch.float32)

    def ct(t): return from_dlpack(t, assumed_align=16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    print(f"compiling grid_barrier test ({NCTA} CTAs)...")
    c = cute.compile(BarrierTest(), ct(gBuf), ct(gBar), ct(gSum), stream)
    print("running...")
    c(ct(gBuf), ct(gBar), ct(gSum), stream)
    torch.cuda.synchronize()

    expect = NCTA * (NCTA + 1) / 2          # 1+2+...+20 = 210
    ok = bool((gSum == expect).all().item())
    print(f"expected {expect} in every slot")
    print(f"got: {gSum.tolist()}")
    print(f"barrier {'WORKS' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
