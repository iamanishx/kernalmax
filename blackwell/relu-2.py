import cutlass
import cutlass.cute as cute

TPB = 256
VEC = 4
ILP = 4


@cute.kernel
def relu_kernel(tA: cute.Tensor, tC: cute.Tensor, ncv: cutlass.Int32, tpb: cutlass.Constexpr):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    for base in cutlass.range(tidx, ncv, tpb * ILP):
        for u in cutlass.range_constexpr(ILP):
            v = base + u * tpb
            if v < ncv:
                r = cute.make_fragment((VEC,), cutlass.Float32)
                cute.autovec_copy(tA[None, (bidx, v)], r)
                for j in cutlass.range_constexpr(VEC):
                    r[j] = cute.arch.fmax(r[j], cutlass.Float32(0.0))
                cute.autovec_copy(r, tC[None, (bidx, v)])


@cute.jit
def solution(input: cute.Tensor, output: cute.Tensor, n: cute.Int64, m: cute.Int64):
    tA = cute.zipped_divide(input, (1, VEC))
    tC = cute.zipped_divide(output, (1, VEC))
    relu_kernel(tA, tC, cutlass.Int32(m // VEC), TPB).launch(
        grid=(cutlass.Int32(n), 1, 1), block=(TPB, 1, 1))
