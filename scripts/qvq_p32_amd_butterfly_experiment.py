"""Experimental Hadamard kernels; never imported by production dispatch."""

import triton
import triton.language as tl


@triton.jit
def fht128_kernel(X, Y, ROWS: tl.constexpr, BLOCK_R: tl.constexpr):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    c = tl.arange(0, 128)
    v = tl.load(X + r[:, None] * 128 + c[None, :], r[:, None] < ROWS, 0)
    for stage in tl.static_range(7):
        other = tl.gather(
            v, tl.broadcast_to((c ^ (1 << stage))[None, :], (BLOCK_R, 128)), 1
        )
        v = tl.where((c[None, :] & (1 << stage)) == 0, v + other, other - v)
    tl.store(Y + r[:, None] * 128 + c[None, :], v, r[:, None] < ROWS)


@triton.jit
def fht128_split_kernel(X, Y, ROWS: tl.constexpr, BLOCK_R: tl.constexpr):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    c = tl.arange(0, 128)
    v = tl.load(X + r[:, None] * 128 + c[None, :], r[:, None] < ROWS, 0)
    for stage in tl.static_range(7):
        grouped = tl.reshape(v, (BLOCK_R, 128 // (2 << stage), 2, 1 << stage))
        lo, hi = tl.split(tl.permute(grouped, (0, 1, 3, 2)))
        v = tl.reshape(
            tl.permute(tl.join(lo + hi, lo - hi), (0, 1, 3, 2)), (BLOCK_R, 128)
        )
    tl.store(Y + r[:, None] * 128 + c[None, :], v, r[:, None] < ROWS)
