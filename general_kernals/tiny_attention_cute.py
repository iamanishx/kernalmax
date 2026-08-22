import cuda.bindings.driver as cuda
import cupy as cp
import cutlass
import numpy as np
from block_reduce import block_reduce_max, block_reduce_sum
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

BH, S, D = 2, 128, 64
THREADS = S
WARPS = THREADS // 32


class TinyAttention:
    @cute.kernel
    def kernel(self, mQ: cute.Tensor, mK: cute.Tensor,
               mV: cute.Tensor, mO: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        i, bh, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        qs = smem.allocate_tensor(cutlass.Float32, cute.make_layout(D), byte_alignment=4)
        ss = smem.allocate_tensor(cutlass.Float32, cute.make_layout(S), byte_alignment=4)
        red = smem.allocate_tensor(cutlass.Float32, cute.make_layout(32), byte_alignment=4)

        lane = tid % 32
        # ---- 1) load this query row Q[bh, i, :] into SMEM ----
        if tid < D:
            qs[tid] = mQ[bh, i, tid]
        cute.arch.sync_threads()

        # ---- 2) thread j(=tid) computes score[j] = dot(Q[i], K[j]) ----
        acc = cutlass.Float32(0.0)
        for d in cutlass.range(D):
            acc = acc + qs[d] * mK[bh, tid, d]

        # ---- 3) softmax over the S scores (REUSED block-reduction helpers) ----
        # 3a) ROW MAX: one helper call replaces the whole warp-shuffle dance
        row_max = block_reduce_max(acc, red, tid, WARPS)

        # 3b) exp(score - max), stash prob in SMEM, and ROW SUM (same helper)
        p = cute.exp(acc - row_max)
        ss[tid] = p
        row_sum = block_reduce_sum(p, red, tid, WARPS)

        # 3c) normalize: ss now holds the probabilities P[j]
        ss[tid] = ss[tid] / row_sum
        cute.arch.sync_threads()

        # ---- 4) output O[bh,i,d] = sum_j P[j] * V[j,d]  (threads d<D) ----
        if tid < D:
            o = cutlass.Float32(0.0)
            for j in cutlass.range(S):
                o = o + ss[j] * mV[bh, j, tid]
            mO[bh, i, tid] = o

    @cute.jit
    def __call__(self, mQ: cute.Tensor, mK: cute.Tensor,
                 mV: cute.Tensor, mO: cute.Tensor, stream: cuda.CUstream):
        self.kernel(mQ, mK, mV, mO).launch(
            grid=[S, BH, 1],          # one CTA per (query row, batch*head)
            block=[THREADS, 1, 1],    # one thread per key
            stream=stream,
        )


def main():
    print(f"Unfused attention: Q,K,V = [{BH},{S},{D}]  ->  O = softmax(Q@K^T)@V")
    print(f"grid [S={S}, BH={BH}], {THREADS} threads (1 per key), "
          f"one CTA per query row\n")

    q = np.random.randn(BH, S, D).astype(np.float32)
    k = np.random.randn(BH, S, D).astype(np.float32)
    v = np.random.randn(BH, S, D).astype(np.float32)
    dq, dk, dv = cp.asarray(q), cp.asarray(k), cp.asarray(v)
    do = cp.zeros((BH, S, D), dtype=cp.float32)

    stream = cuda.CUstream(cp.cuda.get_current_stream().ptr)
    TinyAttention()(from_dlpack(dq), from_dlpack(dk), from_dlpack(dv),
                    from_dlpack(do), stream)
    cp.cuda.get_current_stream().synchronize()

    sc = q @ k.transpose(0, 2, 1)
    sc = sc - sc.max(-1, keepdims=True)
    p = np.exp(sc)
    p = p / p.sum(-1, keepdims=True)
    ref = p @ v

    err = float(np.abs(do.get() - ref).max())
    print(f"max abs err vs NumPy: {err:.3e}")
    np.testing.assert_allclose(do.get(), ref, atol=1e-4, rtol=1e-4)
    print("SUCCESS! Unfused attention wired end to end: Q@K^T -> softmax -> @V.")
    print("Next step: FUSE it (online softmax) so we never store the full score row.")


if __name__ == "__main__":
    main()
