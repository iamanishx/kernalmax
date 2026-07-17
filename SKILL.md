---
name: cutedsl-kernels
description: Write, debug, validate, and optimize NVIDIA CuTe DSL (nvidia-cutlass-dsl) GPU kernels in Python. Use for CuTe layout algebra, TV layouts, tiled copies, predication, shared/register/tensor memory, cp.async and TMA pipelines, mbarriers, warp specialization, warp/block/cluster reductions, MMA on SM80 (tensor cores), SM90 (WGMMA), and SM100/Blackwell (tcgen05/UMMA, TMEM, block-scaled FP4/FP8), tile schedulers, torch interop via from_dlpack, PTX inspection, and diagnosing alignment/vectorization/synchronization failures.
version: 1.0.0
author: Nous Research
license: MIT
dependencies: [nvidia-cutlass-dsl>=4.5.2, torch]
metadata:
  hermes:
    tags: [CUDA, GPU Kernels, CUTLASS, CuTe DSL, Blackwell, Hopper, Performance, TMA, Tensor Cores]
    related_skills: [flash-attention]
---

# CuTe DSL kernel engineering

CuTe DSL is the Python-native CUTLASS DSL (`pip install nvidia-cutlass-dsl`). Kernels are Python functions JIT-compiled to PTX/CUBIN via MLIR. The same CuTe abstractions as CUTLASS C++ apply: layouts, tensors, atoms, tiled copies/MMAs, pipelines. Standard imports:

```python
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp, warpgroup, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass import Int32, Float32, Boolean, const_expr
```

Verify API availability against the installed version (`cutlass.__version__`); this document matches 4.x.

## Method: work from the algorithm down

1. State the mathematical output contract and which CTA/thread owns every output.
2. Choose CTA ownership so required reductions stay as local as possible (thread < warp < block < cluster < grid).
3. Express data movement with layouts, tilers, copy atoms, and partitions — one TV mapping drives partitioning, coordinates, predicates, and copies.
4. Add asynchrony (cp.async, TMA, pipelines) only after identifying independent work that overlaps its latency.
5. Validate against an independent reference before interpreting benchmarks or PTX.
6. Change one performance mechanism at a time; keep a simple known-correct kernel variant.

Do not translate scalar CUDA indexing into CuTe syntax and call it layout algebra. Make the thread/value mapping explicit.

## Compilation model

- `@cute.jit` — host-side JIT function: builds layouts/atoms, then launches kernels. Shape-dependent tilers and layouts belong here; pass them into the kernel as arguments.
- `@cute.kernel` — device kernel. Launch from inside a `@cute.jit` function:

```python
self.kernel(mX, mO, tv_layout, tiler_mn).launch(
    grid=[num_blocks, 1, 1],
    block=[num_threads, 1, 1],
    cluster=[1, cluster_n, 1] if const_expr(cluster_n > 1) else None,  # SM90+
    smem=smem_bytes,        # optional; inferred from SmemAllocator when omitted
    stream=stream,          # cuda.CUstream
)
```

- `cute.compile(fn, *args)` compiles ahead of time and returns a callable; calling a `@cute.jit` function directly compiles on first use. Retain PTX with `cute.compile[cute.KeepPTX()](fn, *args)` then read `compiled.__ptx__`. Env vars in most setups: `CUTE_DSL_KEEP_PTX=1`, `CUTE_CUBIN_PATH`, `CUTE_DSL_LINEINFO=1`.
- **Static vs dynamic**: Python ints/floats/bools and `cutlass.Constexpr`-annotated parameters are compile-time constants that specialize the kernel (layouts, unrolling, feature flags). Runtime scalars are `cutlass.Int32`, `cutlass.Float32`, `cutlass.Boolean`, etc. Every distinct Constexpr combination = separate compilation; cache compiled callables keyed on (dtypes, static shapes, flags).
- `cutlass.const_expr(x)` guards compile-time branches: `if const_expr(self.is_causal): ...`. Use it for any `if`/`is None` test on Constexpr values; a plain dynamic `if` on runtime values generates predicated device code and only works for simple scalar assignment patterns.
- Loops: `cutlass.range(n)` / plain `range(n)` over static bounds unroll; `cutlass.range_constexpr(n)` forces compile-time iteration (required when the body indexes Python lists/tuples); `cutlass.range(n, unroll_full=True)` forces full unroll; `cutlass.range_dynamic(start, stop)` emits a runtime loop.
- Torch interop: `mX = from_dlpack(x, assumed_align=16)`. `assumed_align` is a *contract* — claim 32-byte alignment only if the allocation and all views guarantee it. For dynamic offsets tell the compiler divisibility facts with `k = cute.assume(k, divby=32)`. Mark dynamic layout modes with `.mark_layout_dynamic(leading_dim=...)` or `.mark_compact_shape_dynamic(...)` where supported to avoid one compilation per shape.
- Compile without a GPU / without real data using fake tensors: `cute.runtime.make_fake_tensor(dtype, shape, stride=..., assumed_align=...)`, `cute.sym_int(...)` for symbolic dims, `cute.runtime.make_fake_stream(...)`.

## The object model

- **Layout** = shape + stride: a function from logical coordinates to linear offsets. `cute.make_layout((128, 64), stride=(64, 1))`; `cute.make_ordered_layout((m, n), order=(1, 0))` builds compact strides with a chosen fastest mode (order `(1,0)` ⇒ mode 1, i.e. columns within a row, is contiguous). Layouts are hierarchical: modes can be tuples, e.g. shape `((2, 64), 8)`.
- **Tensor** = iterator (pointer) + layout. A tensor's shape proves nothing about physical contiguity or alignment — the strides and iterator do.
- **Tiler** — the logical tile a cooperative operation covers.
- **TV layout** — maps (thread, value) → position in tile. `tiler, tv_layout = cute.make_layout_tv(thr_layout, val_layout)`. This one object is the source of truth for ownership.
- **Atom** — one hardware instruction (copy or MMA) with dtype and width.
- **TiledCopy / TiledMMA** — atom bound to a TV layout over a tiler.
- **Coordinate tensor** — `cute.make_identity_tensor(shape)` yields logical coordinates; tile and partition it exactly like the data to derive predicates and row/column tests.
- **Fragment** — register tensor: `cute.make_fragment_like(partition, dtype=None)` (ownership-compatible with a partition) or `cute.make_rmem_tensor(...)`.

Layout algebra toolbox (all under `cute.`): `local_tile(tensor, tiler, coord)` extracts a CTA tile; `zipped_divide(tensor, tiler)` → `((TileShape), (RestShape))`; `logical_divide`, `flat_divide`, `logical_product`, `composition`, `coalesce`, `select(layout, mode=[...])`, `group_modes(t, b, e)`, `slice_`, `domain_offset(coord, tensor)` (shifts the iterator), `recast_tensor` / `recast_ptr` / `recast_layout` (bit-width reinterpretation, e.g. viewing 8×FP32 as packed words), `tile_to_shape(atom, shape, order)`, `make_composed_layout(swizzle, offset, layout)`, `make_swizzle(B, M, S)`, `filter_zeros` (drop stride-0 modes), `size`, `cosize`, `rank`, `shape`, `ceil_div`, `round_up`. Naming convention for partitioned tensors: `tXgY` = thread-partitioned global tensor Y under copy X; likewise `tXsY` (shared), `tXrY` (register), `tXcY` (coordinates). Keep the letters honest — they encode the memory space and provenance.

When a partition's shape surprises you, print it (`cute.printf` works on layouts at trace time, and Python `print` shows static shapes during compilation). Do not guess nested indexing.

## Copies

Atom construction: `cute.make_copy_atom(op, dtype, num_bits_per_copy=bits)` with ops:

| Op | Direction | Notes |
|---|---|---|
| `cute.nvgpu.CopyUniversalOp()` | any | synchronous ld/st, any width up to 128b |
| `cpasync.CopyG2SOp()` | gmem→smem | cp.async, 32/64/128-bit, SM80+; takes `LoadCacheMode` |
| `warp.LdMatrix8x8x16bOp(...)` / `StMatrix...` | smem↔rmem | ldmatrix/stmatrix for MMA operand layouts |
| `cpasync.CopyBulkTensorTileG2SOp(cta_group)` | gmem→smem | TMA load (SM90+); `...S2GOp` store; `...MulticastOp` cluster multicast |
| `tcgen05.Ld32x32bOp/...`, `St...` | tmem↔rmem | Blackwell TMEM access |

Building and using a tiled copy:

```python
atom = cute.make_copy_atom(cpasync.CopyG2SOp(), dtype, num_bits_per_copy=128)
thr_layout = cute.make_ordered_layout((num_threads // tpr, tpr), order=(1, 0))
val_layout = cute.make_layout((1, elems_per_thread))
tiled_copy = cute.make_tiled_copy_tv(atom, thr_layout, val_layout)

thr_copy = tiled_copy.get_slice(tidx)
tXgX = thr_copy.partition_S(gX)          # source partition
tXsX = thr_copy.partition_D(sX)          # destination partition
tXcX = thr_copy.partition_S(cX)          # identity/coordinate partition
cute.copy(tiled_copy, tXgX, tXsX, pred=tXpX)   # pred optional
cute.autovec_copy(tXsX, tXrX)            # layout-aware smem<->rmem, auto-vectorized
```

`partition_S`/`partition_D` apply the same TV map to both sides; compatible partitioned shapes are the key invariant. MMA-matched copies: `cute.make_tiled_copy_A/B(atom, tiled_mma)`, `cute.make_tiled_copy_C_atom(atom, tiled_mma)`, and `thr_copy.retile(frag)` to re-view a fragment under another copy's ownership.

**Predication recipe** (ragged tiles): partition an identity tensor alongside the data, then build a Boolean tensor testing coordinates against limits. A partitioned tensor has nested shape `((vec, rest_v), m_iters, k_iters)` — index it with the full coordinate, e.g. `tXcX[(0, rest_v), m, k]` picks the first element of one vector:

```python
cX = cute.local_tile(cute.make_identity_tensor(mX.shape), tiler_mn, (bidx, 0))
tXcX = thr_copy.partition_S(cX)
# one predicate per vector transaction; stride 0 in the m mode broadcasts across rows
tXpX = cute.make_rmem_tensor(
    cute.make_layout(
        (cute.size(tXcX, mode=[0, 1]), cute.size(tXcX, mode=[1]), cute.size(tXcX, mode=[2])),
        stride=(cute.size(tXcX, mode=[2]), 0, 1)),
    cutlass.Boolean)
for rest_v in cutlass.range_constexpr(tXpX.shape[0]):
    for rest_k in cutlass.range_constexpr(tXpX.shape[2]):
        tXpX[rest_v, 0, rest_k] = cute.elem_less(tXcX[(0, rest_v), 0, rest_k][1], limit)
cute.copy(tiled_copy, tXgX, tXsX, pred=tXpX)
row = tXcX[(0, 0), 0, 0][0]        # this thread's first row coordinate, for m-mode guards
```

Predicate at the copy atom's vector granularity: one predicate per whole vector transaction (test the vector's first coordinate). If partial vectors are possible at a boundary, use a narrower atom or a scalar tail path. After a predicated load into smem, fill out-of-bounds slots with the reduction identity (0 for sums, −inf for maxes) before computing:

```python
fill = cute.make_rmem_tensor_like(tXsX[(None, 0), None, 0])   # one vector's worth
fill.fill(-Float32.inf)
for rest_v in cutlass.range_constexpr(tXsX.shape[0][1]):
    for rest_k in cutlass.range_constexpr(tXsX.shape[2]):
        if not tXpX[rest_v, 0, rest_k]:
            cute.autovec_copy(fill, tXsX[(None, rest_v), None, rest_k])
```

**Vector-width legality** — prove all of these before widening num_bits_per_copy:

1. Innermost value mode contiguous in both source and destination.
2. Each thread owns that many adjacent elements.
3. Iterator + every tile/domain offset preserves the byte alignment (a 32B-aligned allocation does not make every derived view 32B-aligned; `domain_offset`, dynamic offsets, and slices weaken provable alignment).
4. The leading dimension preserves alignment for every row that can start a vector.
5. Predicates operate at vector granularity.
6. The op supports that width and direction.

A practical default: `vecsize = gcd(N, 128 // dtype.width)` elements per thread, 128-bit transactions. Use 256-bit only when statically provable and only after PTX confirms the instruction and the kernel measurably improves.

## Memory spaces and shared memory

Declare shared storage with a struct and allocator inside the kernel:

```python
@cute.struct
class SharedStorage:
    sX: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, tile_m * tile_n], 16]
    reduction_buf: cute.struct.MemRange[cutlass.Float32, num_warps]
    mbar: cute.struct.MemRange[cutlass.Int64, num_stages]

smem = cutlass.utils.SmemAllocator()
storage = smem.allocate(SharedStorage)
sX = storage.sX.get_tensor(sX_layout)                    # or .get_tensor(layout.outer, swizzle=layout.inner)
# ad-hoc: smem.allocate_tensor(dtype, layout, byte_alignment=16), smem.allocate_array(dtype, n)
```

Size the struct with `SharedStorage.size_in_bytes()` when passing `smem=` explicitly. Avoid smem bank conflicts for MMA operand tiles by using the arch smem layout atoms (`warpgroup.make_smem_layout_atom(...)`, `tcgen05.make_smem_layout_atom(...)`, or `cutlass.utils.make_smem_layout_a/b/epi`) rather than hand-rolled swizzles.

After `cute.autovec_copy(tXrX, tXsX)`, smem holds a copy; the register fragment is still live and authoritative for the thread. Track which copy is authoritative — a classic bug is updating registers then accidentally re-reading stale smem (or vice versa). Insert `cute.arch.sync_threads()` only around producer/consumer transitions through smem, not around a thread reusing its own registers.

## Device-side compute

- Thread identity: `tidx, tidy, tidz = cute.arch.thread_idx()`; `cute.arch.block_idx()`, `grid_dim()`, `block_dim()`, `cute.arch.warp_idx()`, `cute.arch.lane_idx()`, `cute.arch.block_idx_in_cluster()`. Wrap warp-uniform values: `warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())`.
- Bulk register math via TensorSSA: `x = tXrX.load().to(cutlass.Float32)` gives a value you can apply arithmetic to elementwise (`y = x * x + 1.0`), reduce (`x.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0)`), then `tXrO.store(y.to(out_dtype))`. Always accumulate in fp32; convert at the store.
- Math: `cute.math.exp2/exp/log/log2/rsqrt/sqrt/tanh(..., fastmath=True)`, `cute.arch.rcp_approx`, `cute.arch.exp2`. For softmax-style code use exp2: `exp(x) = exp2(x * log2(e))`, folding `log2(e)` into the scale.
- Scalar elementwise loops over a fragment: `for i in cutlass.range_constexpr(cute.size(tXrX)): tXrX[i] = ...`.
- Printing: `cute.printf("...", ...)` (guard with `if tidx == 0:`); Python `print` shows static values at trace time.

## Reductions: narrowest scope first

```python
# warp: butterfly shuffle
val = cute.arch.warp_reduction(val, lambda a, b: a + b, threads_in_group=32)
# or manually: for i in [1,2,4,8,16]: val += cute.arch.shuffle_sync_bfly(val, offset=i)

# block: one partial per warp through smem, one warp finishes
if lane == 0: reduction_buf[warp_id] = val
cute.arch.sync_threads()
if warp_id == 0:
    v = reduction_buf[lane] if lane < num_warps else identity
    v = cute.arch.warp_reduction(v, op, threads_in_group=num_warps)
```

Cluster-scope reductions (row wider than one CTA, SM90+): write partials to peer-CTA smem with `st.async.shared::cluster` (`mapa` to translate the pointer; helpers exist in kernel libraries as `store_shared_remote(val, remote_smem_ptr, mbar_ptr, peer_cta_rank)`), count arrivals with an mbarrier, then each CTA reduces all peers' partials locally. Initialize cluster mbarriers before `cute.arch.cluster_arrive_relaxed()` / `cluster_wait()`. Keep the lane↔data mapping derivable from the TV layout, and assert/document any shuffle distance assumption next to the layout construction.

Online softmax pattern (the canonical fused-reduction idiom): track `row_max` and `row_sum` fragments; per tile compute `new_max = max(old_max, tile_max)`, `correction = exp2((old_max - new_max) * scale_log2)`, rescale `row_sum` and the output accumulator by `correction`, accumulate `exp2(x*scale_log2 - new_max*scale_log2)`. Guard `-inf` rows (fully masked) by substituting 0 for the max before subtraction.

## Asynchrony tier 1: cp.async (SM80+)

```python
cute.copy(tiled_copy_g2s, tXgX, tXsX, pred=tXpX)   # issue (async atom)
cute.arch.cp_async_commit_group()
cute.arch.cp_async_wait_group(0)                    # 0 = all outstanding groups done
cute.arch.sync_threads()                            # make smem visible CTA-wide
```

Multi-stage manual ring: issue stages `0..S-2`, commit each; per iteration `wait_group(S-2)` → sync → compute stage `i % S` → refill it with tile `i+S-1` → commit; drain at the tail. The overlap that matters is between the refill of one stage and arithmetic on another. `cp.async` followed immediately by `wait_group(0)` every tile hides nothing — verify with a profile that real work sits between issue and wait. Manual groups are fine for a small single-agent pipeline; switch to mbarrier `Pipeline*` classes when multiple agents (warps) produce/consume.

## Asynchrony tier 2: TMA + mbarriers (SM90+)

TMA moves whole tiles with one instruction issued by one thread; completion is tracked by an mbarrier with a transaction byte count.

```python
# host (@cute.jit): build atom + TMA-view tensor
tma_atom_A, tma_mA = cpasync.make_tiled_tma_atom(
    cpasync.CopyBulkTensorTileG2SOp(), mA, sA_layout_one_stage, (tile_m, tile_k))
# for MMA operands prefer nvgpu.make_tiled_tma_atom_A / _B which match the tiled MMA

# kernel: partition smem/gmem for TMA
tAsA, tAgA = cpasync.tma_partition(
    tma_atom_A, cta_coord_in_cluster, cta_layout,
    cute.group_modes(sA, 0, 2), cute.group_modes(gA_tiles, 0, 2))

# producer thread (one per CTA / leader CTA):
with cute.arch.elect_one():
    cute.arch.mbarrier_init(mbar + stage, 1)  # once, then mbarrier_init_fence + sync
    cute.arch.mbarrier_arrive_and_expect_tx(mbar + stage, tma_bytes)
    cute.copy(tma_atom_A, tAgA[None, k_tile], tAsA[None, stage], tma_bar_ptr=mbar + stage,
              mcast_mask=mask)   # mcast_mask only for multicast atoms
# consumer:
cute.arch.mbarrier_wait(mbar + stage, phase)   # phase toggles 0/1 each ring pass
```

Rules that prevent hangs: `expect_tx` bytes must equal exactly what the TMA writes to that barrier (multiply by `cta_group_size` for 2-CTA MMAs); every consumer tracks `(index, phase)` and flips phase each wrap; prefetch descriptors once per kernel (`cpasync.prefetch_descriptor(tma_atom)`, warp 0); TMA store completion uses `cute.arch.cp_async_bulk_commit_group()` / `cp_async_bulk_wait_group(n, read=True)`; multicast masks come from `cpasync.create_tma_multicast_mask(...)`.

Prefer `cutlass.pipeline` over raw mbarriers once there are ≥2 agents:

```python
from cutlass import pipeline
p = pipeline.PipelineTmaAsync.create(
    barrier_storage=storage.mbar.data_ptr(), num_stages=S,
    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, ...),
    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, ...),
    tx_count=bytes_per_stage)
ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, S)
cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, S)
# producer: p.producer_acquire(ps); issue TMA with tma_bar_ptr=p.producer_get_barrier(ps); ps.advance()
# consumer: p.consumer_wait(cs); ...use smem...; p.consumer_release(cs); cs.advance()
# tail: p.producer_tail(ps)  # drain before exiting / reusing smem
```

Variants: `PipelineCpAsync` (cp.async agents), `PipelineTmaUmma` (TMA producer → Blackwell MMA consumer; leader-CTA arrives for 2-CTA), `PipelineUmmaAsync` (MMA producer → threads consumer), `PipelineTmaStore` (epilogue stores). `pipeline_init_arrive/pipeline_init_wait` synchronize barrier initialization cluster-wide.

Choose TMA over cp.async for large regular 2D/3D tiles, multicast across a cluster, or producer/consumer warp specialization. For small irregular tiles the descriptor setup and barrier protocol can cost more than cp.async — measure.

## MMA

Atom → tiled MMA → partitions → `cute.gemm`:

```python
tiled_mma = cute.make_tiled_mma(atom, atom_layout_mnk)     # or arch helper below
thr_mma = tiled_mma.get_slice(tidx)
tCsA = thr_mma.partition_A(sA); tCrA = tiled_mma.make_fragment_A(tCsA)
tCsB = thr_mma.partition_B(sB); tCrB = tiled_mma.make_fragment_B(tCsB)
acc  = thr_mma.make_fragment_C(thr_mma.partition_shape_C((tile_m, tile_n)))
acc.fill(0.0)
for k in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
    cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
```

- **SM80**: `warp.MmaF16BF16Op(dtype, acc_dtype, (16, 8, 16))`; operands come to registers via ldmatrix copies. Multi-stage cp.async mainloop.
- **SM90**: `warpgroup.MmaF16BF16Op / MmaF8Op` — 128-thread warpgroup MMA reading operands directly from smem (`OperandSource.SMEM`). Helper: `cutlass.utils.hopper_helpers` / `make_trivial_tiled_mma`. Protocol around every gemm group: `warpgroup.fence(); [gemm calls]; warpgroup.commit_group(); warpgroup.wait_group(N)`. Toggle accumulate with `tiled_mma.set(warpgroup.Field.ACCUMULATE, True/False)`. Typical structure: producer warp(s) run TMA, two consumer warpgroups ping-pong mainloop/epilogue; rebalance registers with `cute.arch.setmaxregister_decrease(n)` (producers) / `setmaxregister_increase(n)` (consumers), multiples of 8.
- **SM100 (Blackwell)**: `tcgen05.MmaF16BF16Op / MmaFP8Op / MmaF8F6F4Op / MmaMXF4NVF4Op(...)` with `tcgen05.CtaGroup.ONE|TWO`; one thread issues the MMA (`with cute.arch.elect_one():`), accumulator lives in **TMEM**, operands in smem (descriptors) or TMEM. Helpers: `cutlass.utils.make_trivial_tiled_mma(...)`, `cutlass.utils.blackwell_helpers`.

TMEM lifecycle (SM100):

```python
cute.arch.alloc_tmem(num_cols, storage.tmem_holding_buf)        # one warp allocates; cols ≤ 512
tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, alignment=16,
                                       ptr_to_buffer_holding_addr=storage.tmem_holding_buf)
acc_tmem = cute.make_tensor(tmem_ptr, acc_layout)               # offsets partition the 512 cols
# after MMA: move TMEM→registers with a tmem copy
tiled_t2r = tcgen05.make_tmem_copy(cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(r)), Float32), acc_tmem)
# ... epilogue ... then, after all consumers signal via mbarrier:
cute.arch.relinquish_tmem_alloc_permit()
cute.arch.dealloc_tmem(tmem_ptr, num_cols)
```

TMEM budgeting is manual: you place accumulators at column offsets and must prove no live buffers overlap. `tcgen05.find_tmem_tensor_col_offset(t)` helps compute offsets. Fences when mixing proxies: `cute.arch.fence_view_async_tmem_load/store()`.

Block-scaled narrow precision (SM100): FP4 (`Float4E2M1FN`) / MXFP8 with per-16/32-element scale factors (`Float8E4M3FN` UE4M3 or `Float8E8M0FNU` UE8M0). Build with `cutlass.utils.make_blockscaled_trivial_tiled_mma(a_dtype, a_major, b_major, sf_dtype, sf_vec_size, cta_group, shape_mn)`; scale-factor smem layouts via `cutlass.utils.make_smem_layout_sfa/sfb`, staged into TMEM with `tcgen05.make_s2t_copy`; set per-K scale pointers with `mma_atom.set(tcgen05.Field.SFA, ...)`. B operands must be K-major for FP4.

## Warp specialization and schedulers

Structure large kernels as cooperating agents keyed on `warp_idx`: e.g. load warp (TMA), MMA warp, softmax/compute warp groups, correction/epilogue warps. Each pair of agents communicates through one pipeline or named barrier; draw the full barrier graph before coding — every producer arrival must have a matching consumer wait with correct thread counts.

Named barriers for warp-group sync (barrier 0 is `sync_threads`; use an IntEnum starting at 1): `cute.arch.barrier(barrier_id=ID, number_of_threads=N)`, `cute.arch.barrier_arrive(...)`.

Persistent kernels: launch ~`HardwareInfo().get_device_multiprocessor_count()` CTAs; each loops over work tiles from a scheduler (`cutlass.utils.StaticPersistentTileScheduler` or a custom one). Order tiles for L2 reuse (swizzle groups of tiles sharing operands into consecutive scheduling slots); for load-imbalanced problems (causal attention, grouped GEMM), schedule longest work first (LPT) and consider a dynamic atomic-counter scheduler. On SM100, cluster launch control (CLC) schedulers exist as `cutlass.utils.ClcDynamicPersistentTileScheduler`.

## Cluster features (SM90+)

`cluster=[cx, cy, cz]` at launch; `cute.arch.cluster_arrive_relaxed()` / `cluster_wait()` bracket cluster-wide setup; `block_idx_in_cluster` locates the CTA. Use clusters for TMA multicast (halve L2 traffic for shared operands), cross-CTA smem access (`st.async` reductions), and 2-CTA MMAs. Do not add clusters because the target supports them — use a second CTA only when the work per exchange amortizes the synchronization and the algorithm has no serial cross-CTA reduction in its critical path.

## Canonical skeleton: memory-bound rowwise kernel

The recipe behind rmsnorm/softmax/cross-entropy class kernels — one CTA (or cluster) per row-block:

```python
@cute.kernel
def kernel(self, mX, mO, tv_layout, tiler_mn):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    smem = cutlass.utils.SmemAllocator()
    sX = smem.allocate_tensor(mX.element_type, sX_layout, byte_alignment=16)

    gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
    gO = cute.local_tile(mO, tiler_mn, (bidx, 0))
    cX = cute.local_tile(cute.make_identity_tensor(mX.shape), tiler_mn, (bidx, 0))

    thr_copy = tiled_copy.get_slice(tidx)
    tXgX, tXsX = thr_copy.partition_S(gX), thr_copy.partition_D(sX)
    tXgO, tXcX = thr_copy.partition_D(gO), thr_copy.partition_S(cX)
    tXrX, tXrO = cute.make_fragment_like(tXgX), cute.make_fragment_like(tXgO)

    tXpX = <vector-granularity predicate from tXcX, as in the predication recipe>
    row = tXcX[(0, 0), 0, 0][0]
    if row < mX.shape[0]:
        cute.copy(tiled_copy_g2s, tXgX, tXsX, pred=tXpX)  # cp.async
    cute.arch.cp_async_commit_group(); cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()
    # fill OOB smem with reduction identity if ragged
    cute.autovec_copy(tXsX, tXrX)
    x = tXrX.load().to(cutlass.Float32)
    r = x.reduce(cute.ReductionOp.ADD, 0.0, 0)            # thread-local
    r = cute.arch.warp_reduction(r, lambda a, b: a + b, threads_in_group=tpr)
    # block/cluster combine if threads_per_row spans warps
    y = f(x, r)
    tXrO.store(y.to(mO.element_type))
    if row < mX.shape[0]:
        cute.copy(tiled_copy_r2g, tXrO, tXgO, pred=tXpX)
```

This skeleton (as a full softmax with max/sum reductions, `cute.arch.fmax` in the warp reduction, ragged-N predicates, fp32 and bf16) compiles and validates against `torch.softmax` verbatim on cutlass 4.5.2.

Sizing heuristics: 128–256 threads; `threads_per_row` grows with N (8 for N≤64 up to 256 for N>16k); vectorize at `gcd(N, 128 // dtype.width)`; add `cluster_n > 1` only when one CTA cannot hold a row.

## Validation and benchmarking

- Validate against an independent reference (torch/NumPy) before reading any benchmark. Include: ragged shapes, degenerate inputs (zeros, -inf rows, rank-deficient), nonzero tile origins, non-contiguous strides, and every dtype you claim. A single seeded dense input proves nothing.
- Keep compilation, input reset, and correctness copies outside the timed region. Time with CUDA events over many iterations after warmup; report median and min. For memory-bound kernels sanity-check against achievable bandwidth (bytes moved / time vs. peak); for compute-bound, against tensor-core peak.
- Beware L2-resident microbenchmarks: rotate inputs through a buffer larger than L2 or use CUDA-graph L2-rotation benchmarking when comparing configs.
- Inspect PTX (KeepPTX) for: transaction widths (`ld.global.v4`, `cp.async.cg.128`), local-memory spills (`st.local`), barrier placement, whether vectorization survived lowering. Inspect SASS for register counts. Use `compute-sanitizer --tool=racecheck` for shared-memory races (expect false positives with raw TMA `cp.async.bulk`) and `--tool=memcheck` for OOB.
- Debug hangs by printf-bisection with thread guards (`if tidx == 0 and bidx == 0:`), checking each pipeline's (index, phase) discipline and every mbarrier's arrival count against its init count.

## Failure → diagnosis table

| Symptom | Check |
|---|---|
| Copy-alignment verifier error | Derived layout + iterator alignment of the *source view* (offsets/slices weaken it), not just the allocation; `assumed_align` truthfulness; `cute.assume` on dynamic offsets |
| Wrong values at tile edges | Predicate shape must match the copy atom's value modes; predicate from the identity tensor partitioned identically to data |
| Stale smem reads | Missing `sync_threads` after wait; stage reused before consumers done (release/acquire order, phase flip) |
| Wrong stores after register update | Authoritative copy is in rmem — don't re-read smem after modifying only registers (or vice versa) |
| Hang | Mismatched mbarrier arrive/wait counts; wrong expect_tx bytes (2-CTA: ×cta_group); phase not flipped on ring wrap; a warp exited before `producer_tail` |
| Async gives no speedup | Not enough independent work between issue and wait — restructure before adding stages |
| Register spills | Replace small dynamic-indexed fragments with scalars, reduce unroll, lower per-thread values, rebalance with setmaxregister |
| Slow small-batch | Reduce per-CTA resources or add independent tiles before splitting dependent work across CTAs |
| One compilation per shape | Mark dynamic layout modes / use sym ints; keep shapes out of Constexpr unless they control layouts |

## Feature-selection cheat sheet

| Situation | Use |
|---|---|
| Elementwise / rowwise memory-bound | TV-layout tiled copy, cp.async, warp+block reduction; 128-bit vectors |
| Row too wide for one CTA | Cluster + mbarrier cross-CTA reduction |
| Large regular tiles, multiple agents | TMA + `cutlass.pipeline`, warp specialization |
| GEMM-scale matmul on SM90 | WGMMA from smem, TMA multicast, persistent scheduler |
| GEMM-scale matmul on SM100 | tcgen05 + TMEM, `PipelineTmaUmma`, consider 2-CTA for ≥256-wide M tiles |
| Sub-byte / block-scaled dtypes | SM100 block-scaled MMA with SF layouts; pack/unpack via `recast_tensor` |
| Small serial recurrences | Plain registers and shuffles — TMA/TMEM/UMMA machinery will not help |

Borrow patterns from proven kernels (CUTLASS `examples/python/CuTeDSL`, the quack kernel library, FlashAttention-4 `flash_attn/cute`) but re-derive every layout for your tensor shapes — constants copied from attention or GEMM kernels are wrong for anything else.