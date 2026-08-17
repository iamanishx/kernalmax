import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

TPB = 256


@cute.kernel
def relu_kernel(mA: cute.Tensor, mC: cute.Tensor,
                cols: cutlass.Int32, tpb: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx() 
    for j in cutlass.range(tidx, cols, tpb):
        mC[(bidx, j)] = cute.arch.fmax(mA[(bidx, j)], cutlass.Float32(0.0))


@cute.jit
def solution(input: cute.Tensor, output: cute.Tensor,
             n: cutlass.Int64, m: cutlass.Int64):
    relu_kernel(input, output, cutlass.Int32(m), TPB).launch(
        grid=(cutlass.Int32(n), 1, 1),
        block=(TPB, 1, 1),
    )


def _bench(fn, it=200, wu=50):
    for _ in range(wu):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / it


def main():
    sizes = [(4096, 4096), (6144, 4096), (4096, 7168), (4096, 8192), (8192, 8192)]
    print("ReLU  C = max(0, A)   (memory-bound; goal = saturate HBM)\n")

    for (M, N) in sizes:
        torch.manual_seed(0)
        a = torch.randn(M, N, device="cuda", dtype=torch.float32)
        c = torch.empty(M, N, device="cuda", dtype=torch.float32)
        mi = from_dlpack(a, assumed_align=16)
        mo = from_dlpack(c, assumed_align=16)

        comp = cute.compile(solution, mi, mo, cutlass.Int64(M), cutlass.Int64(N))
        comp(mi, mo, cutlass.Int64(M), cutlass.Int64(N))
        torch.cuda.synchronize()

        err = (c - torch.relu(a)).abs().max().item()
        gb = 2 * M * N * 4 / 1e9         
        us = _bench(lambda: comp(mi, mo, cutlass.Int64(M), cutlass.Int64(N))) * 1000
        ut = _bench(lambda: torch.relu(a)) * 1000
        print(f"  {M}x{N}: err {err:.0e} | ours {us:7.1f}us {gb/(us/1e6):6.0f} GB/s"
              f" | torch {ut:7.1f}us {gb/(ut/1e6):6.0f} GB/s"
              f" | {ut/us:.2f}x")

    print("\nnote: on B200 the same structure scales to HBM3e; tune TPB (128/256/512).")


if __name__ == "__main__":
    main()
