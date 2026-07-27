import cutlass
from cutlass import cute

NEG_INF = -3.4e38


@cute.jit
def block_reduce_max(val: cutlass.Float32, red: cute.Tensor,
                     tid: cutlass.Int32, warps: cutlass.Constexpr) -> cutlass.Float32:
    """Reduce one `val` per thread to the block MAX. Returns it broadcast to all threads.
    `red` is a 32-float SMEM scratch tensor. `warps` = THREADS // 32."""
    lane = tid % 32
    warp = tid // 32
    offset = 16
    while offset > 0:
        o = cute.arch.shuffle_sync_down(val, offset)
        val = max(val, o)
        offset //= 2
    if lane == 0:
        red[warp] = val
    if tid >= warps and tid < 32:
        red[tid] = cutlass.Float32(NEG_INF)
    cute.arch.sync_threads()
    if warp == 0:
        x = red[lane]
        offset = 16
        while offset > 0:
            o = cute.arch.shuffle_sync_down(x, offset)
            x = max(x, o)
            offset //= 2
        if lane == 0:
            red[0] = x
    cute.arch.sync_threads()
    result = red[0]
    cute.arch.sync_threads()
    return result


@cute.jit
def block_reduce_sum(val: cutlass.Float32, red: cute.Tensor,
                     tid: cutlass.Int32, warps: cutlass.Constexpr) -> cutlass.Float32:
  
    lane = tid % 32
    warp = tid // 32
    offset = 16
    while offset > 0:
        val = val + cute.arch.shuffle_sync_down(val, offset)
        offset //= 2
    if lane == 0:
        red[warp] = val
    if tid >= warps and tid < 32:
        red[tid] = cutlass.Float32(0.0)
    cute.arch.sync_threads()
    if warp == 0:
        x = red[lane]
        offset = 16
        while offset > 0:
            x = x + cute.arch.shuffle_sync_down(x, offset)
            offset //= 2
        if lane == 0:
            red[0] = x
    cute.arch.sync_threads()
    result = red[0]
    cute.arch.sync_threads()
    return result
