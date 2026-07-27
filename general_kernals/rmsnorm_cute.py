import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from torch import nn


@cute.kernel
def rms_norm_kernel(mX: cute.Tensor, mW: cute.Tensor, mY: cute.Tensor,
                    threads_per_block: cutlass.Constexpr,
                    num_tokens: cutlass.Constexpr,
                    hidden_dim: cutlass.Constexpr,
                    epsilon: cutlass.Constexpr):
    allocator = cutlass.utils.SmemAllocator()
    sdata = allocator.allocate_tensor(cutlass.Float32,
                                      cute.make_layout(threads_per_block),
                                      byte_alignment=16, swizzle=None)
    squared_reduce = allocator.allocate_tensor(cutlass.Float32,
                                               cute.make_layout(1))

    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    block_sum = 0.0
    for i in range(tidx, hidden_dim, threads_per_block, unroll_full=True):
        x_ = mX[(bidx, i)]
        block_sum += x_ * x_
    sdata[tidx] = block_sum
    cute.arch.sync_threads()

    if tidx < 128:
        sdata[tidx] += sdata[tidx + 128]
    cute.arch.sync_threads()
    if tidx < 64:
        sdata[tidx] += sdata[tidx + 64]
    cute.arch.sync_threads()
    if tidx < 32:
        sdata[tidx] += sdata[tidx + 32]
        res = cute.arch.warp_reduction_sum(sdata[tidx], threads_in_group=32)
        if tidx == 0:
            squared_reduce[0] = cute.math.rsqrt(res / hidden_dim + epsilon,
                                                fastmath=True)
    cute.arch.sync_threads()

    rms = squared_reduce[0]
    for i in range(tidx, hidden_dim, threads_per_block, unroll_full=True):
        mY[(bidx, i)] = mX[(bidx, i)] * rms * mW[i]


@cute.jit
def rms_norm_(mX: cute.Tensor, mW: cute.Tensor, mY: cute.Tensor,
              num_tokens: cutlass.Constexpr, hidden_dim: cutlass.Constexpr,
              epsilon: cutlass.Constexpr):
    threads_per_block = 256
    rms_norm_kernel(mX, mW, mY, threads_per_block,
                    num_tokens, hidden_dim, epsilon).launch(
        grid=(num_tokens, 1, 1),
        block=(threads_per_block, 1, 1),
    )


def _bench(fn, iters=100, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def main():
    num_tokens, hidden_dim, eps = 65536, 1024, 1e-5
    print(f"RMSNorm: X = [{num_tokens}, {hidden_dim}]  (real-sized, memory-bound)\n")

    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.float32)
    w = torch.randn(hidden_dim, device="cuda", dtype=torch.float32)
    y = torch.zeros(num_tokens, hidden_dim, device="cuda", dtype=torch.float32)

    mX = from_dlpack(x, assumed_align=16)
    mW = from_dlpack(w, assumed_align=16)
    mY = from_dlpack(y, assumed_align=16)

    rms_norm_(mX, mW, mY, num_tokens, hidden_dim, eps)
    torch.cuda.synchronize()
    ref_layer = nn.RMSNorm(hidden_dim, eps=eps).cuda()
    ref_layer.weight.data = w
    y_ref = ref_layer(x)
    err = (y - y_ref).abs().max().item()
    print(f"max abs err vs torch RMSNorm: {err:.3e}")
    torch.testing.assert_close(y, y_ref, atol=1e-4, rtol=1e-4)
    print("correctness OK\n")

    # --- 3-way benchmark ---
    # 1) ours WITHOUT cute.compile (the @cute.jit wrapper re-traces each call)
    def ours_nojit():
        rms_norm_(mX, mW, mY, num_tokens, hidden_dim, eps)

    # 2) ours WITH cute.compile (AOT once)
    # constexpr args (num_tokens, hidden_dim, eps) get BAKED IN at compile time,
    # so the compiled callable only takes the runtime tensor args.
    compiled = cute.compile(rms_norm_, mX, mW, mY, num_tokens, hidden_dim, eps)
    def ours_compiled():
        compiled(mX, mW, mY)

    # 3) torch RMSNorm
    def torch_rms():
        ref_layer(x)

    us_nojit = _bench(ours_nojit)
    us_comp = _bench(ours_compiled)
    us_torch = _bench(torch_rms)

    # achieved bandwidth for the compiled kernel (1 read + 1 write of X)
    gb = num_tokens * hidden_dim * 2 * 4 / 1e9
    bw = gb / (us_comp / 1e6)

    print("Benchmark (avg over 100 iters):")
    print(f"  ours WITHOUT cute.compile : {us_nojit:9.2f} us   (re-traces every call!)")
    print(f"  ours WITH    cute.compile : {us_comp:9.2f} us   ({bw:6.1f} GB/s)")
    print(f"  torch RMSNorm             : {us_torch:9.2f} us")
    print()
    print(f"  cute.compile speedup over no-compile: {us_nojit / us_comp:8.1f}x")
    print(f"  compiled vs torch:                    {us_torch / us_comp:8.2f}x "
          f"({'ours faster' if us_comp < us_torch else 'torch faster'})")


if __name__ == "__main__":
    main()
