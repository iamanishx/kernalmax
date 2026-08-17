"""Grid-wide barrier + block reduction — @cute.jit.

Grid barrier (sense-reversing counter):
  Cross-CTA sync is REQUIRED for a persistent multi-SM megakernel because
  `sync_threads()` only synchronizes within one CTA. Op N+1 needs ALL of op N's
  output, which lives in global memory written by all CTAs.

  Protocol: thread 0 of each CTA atomically increments a global counter, then
  every CTA spins until the counter reaches gen*num_ctas. `gen` is a
  compile-time constant that increments at each barrier site, so barriers never
  alias — no sense-reversal bookkeeping needed.

  SAFE ONLY IF all CTAs are co-resident (grid <= num_SMs), otherwise a
  non-resident CTA can never arrive → deadlock. We launch grid=[NUM_SMS].
"""

import cutlass
from cutlass import cute


@cute.jit
def block_reduce_sum(partial, sR, tid):
    """Block-wide sum via zero-padded 32-slot SMEM buffer (no dynamic-if)."""
    warp = tid // 32
    lane = tid % 32
    ws = cute.arch.warp_reduction(partial, lambda a, b: a + b, threads_in_group=32)
    sR[lane] = cutlass.Float32(0.0)
    cute.arch.sync_threads()
    sR[warp] = ws
    cute.arch.sync_threads()
    blk = cute.arch.warp_reduction(sR[lane], lambda a, b: a + b, threads_in_group=32)
    return blk


@cute.jit
def grid_barrier(mBar, tid, gen: cutlass.Constexpr, num_ctas: cutlass.Constexpr):
    """
    Grid-wide barrier with GPU-scope memory ordering.

    CRITICAL: GPU L1 caches are NOT coherent across SMs. Without the fences,
    a CTA can read a stale L1 line for data another CTA just wrote, producing
    wrong-but-plausible numbers. The release fence publishes this CTA's writes
    before it arrives; the acquire fence after the spin makes peer writes
    visible to this CTA.

    `gen` must be a unique increasing compile-time integer per barrier site.
    All CTAs must be co-resident (grid <= #SMs) or this deadlocks.
    """
    cute.arch.sync_threads()               # this CTA's threads are done
    cute.arch.fence_acq_rel_gpu()          # publish our writes GPU-wide
    target = cutlass.Int32(gen * num_ctas)
    if tid == 0:
        cute.arch.atomic_add(mBar.iterator, cutlass.Int32(1),
                             sem="release", scope="gpu")
        # spin with an atomic (bypasses L1) until all CTAs arrive
        while cute.arch.atomic_add(mBar.iterator, cutlass.Int32(0),
                                   sem="acquire", scope="gpu") < target:
            pass
    cute.arch.sync_threads()               # release the rest of the CTA
    cute.arch.fence_acq_rel_gpu()          # make peer writes visible to us
