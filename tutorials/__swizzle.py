import cupy as cp
import cutlass
import numpy as np
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

N = 32
THREADS = 32


def show_bank_pattern():
    print("Bank pattern when 32 threads simultaneously write sA[tidx, 0]:")
    print()
    print("  PLAIN layout  (offset = t*32 + j  ->  bank = j):")
    plain_banks = [(t * 32 + 0) % 32 for t in range(32)]
    print(f"    threads 0..31 banks:  {plain_banks}")
    print(f"    distinct banks: {len(set(plain_banks))}  -> {32 // len(set(plain_banks))}-way CONFLICT")
    print()
    print("  SWIZZLED layout  (offset = t*32 + (j XOR t)  ->  bank = j XOR t):")
    swz_banks = [((t * 32 + 0) ^ t) % 32 for t in range(32)]
    print(f"    threads 0..31 banks:  {swz_banks}")
    print(f"    distinct banks: {len(set(swz_banks))}  -> CONFLICT-FREE")
    print()



class TransposePlain:
    """Transpose via plain row-major SMEM. SMEM writes have 32-way bank conflict."""

    @cute.kernel
    def kernel(self, mIn: cute.Tensor, mOut: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()

        smem = cutlass.utils.SmemAllocator()
        smem_layout = cute.make_layout((N, N), stride=(N, 1))   # row-major, stride 32
        sA = smem.allocate_tensor(cutlass.Float32, smem_layout, byte_alignment=16)

        for j in cutlass.range_constexpr(N):
            sA[tidx, j] = mIn[tidx, j]
        cute.arch.sync_threads()

        for i in cutlass.range_constexpr(N):
            mOut[tidx, i] = sA[i, tidx]

    @cute.jit
    def __call__(self, mIn: cute.Tensor, mOut: cute.Tensor):
        self.kernel(mIn, mOut).launch(grid=[1, 1, 1], block=[THREADS, 1, 1])


class TransposeSwizzled:
    """Same kernel, swizzled SMEM layout. SMEM writes are conflict-free."""

    @cute.kernel
    def kernel(self, mIn: cute.Tensor, mOut: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()

        smem = cutlass.utils.SmemAllocator()
        plain = cute.make_layout((N, N), stride=(N, 1))
        # Swizzle<5, 0, 5>: take 5 bits of offset starting at bit 5 (the row
        # index) and XOR them into bits at position 0 (the bank index).
        # Result: sA[i, j] still gives the same value, but the underlying
        # bank pattern is rotated row-by-row.
        smem_layout = cute.make_composed_layout(cute.make_swizzle(5, 0, 5), 0, plain)
        sA = smem.allocate_tensor(cutlass.Float32, smem_layout, byte_alignment=16)

        for j in cutlass.range_constexpr(N):
            sA[tidx, j] = mIn[tidx, j]
        cute.arch.sync_threads()

        for i in cutlass.range_constexpr(N):
            mOut[tidx, i] = sA[i, tidx]

    @cute.jit
    def __call__(self, mIn: cute.Tensor, mOut: cute.Tensor):
        self.kernel(mIn, mOut).launch(grid=[1, 1, 1], block=[THREADS, 1, 1])


def main():
    print(f"32x32 fp32 transpose via SMEM, {THREADS} threads in one warp.\n")
    show_bank_pattern()

    a_np = np.random.randn(N, N).astype(np.float32)
    ref = a_np.T
    a = cp.asarray(a_np)
    o_plain = cp.zeros((N, N), dtype=cp.float32)
    o_swz = cp.zeros((N, N), dtype=cp.float32)

    TransposePlain()(from_dlpack(a), from_dlpack(o_plain))
    cp.cuda.get_current_stream().synchronize()
    err_plain = float(np.abs(o_plain.get() - ref).max())

    TransposeSwizzled()(from_dlpack(a), from_dlpack(o_swz))
    cp.cuda.get_current_stream().synchronize()
    err_swz = float(np.abs(o_swz.get() - ref).max())

    print(f"plain layout    transpose err: {err_plain:.2e}")
    print(f"swizzled layout transpose err: {err_swz:.2e}")
    np.testing.assert_allclose(o_plain.get(), ref)
    np.testing.assert_allclose(o_swz.get(), ref)


if __name__ == "__main__":
    main()
