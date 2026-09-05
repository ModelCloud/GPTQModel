"""Experimental Hadamard kernels; never imported by production dispatch."""

import triton
import triton.language as tl


@triton.jit
def composite_trim_kernel(
    pre_hadamard_ptr, power_ptr, base_ptr, sv_ptr, output_ptr,
    size_n: tl.constexpr, base_size: tl.constexpr, base_pad: tl.constexpr,
    power_width: tl.constexpr, block_p: tl.constexpr, inv_sqrt_n: tl.constexpr,
):
    """Split the 40 real base rows into 32+8, padding only the tail to 16."""
    row = tl.program_id(0)
    position = tl.program_id(1) * block_p + tl.arange(0, block_p)
    out = tl.arange(0, base_pad)
    lo = tl.arange(0, 32)
    hi = 32 + tl.arange(0, 16)
    p = tl.arange(0, power_width)
    power = tl.load(power_ptr + p[:, None] * power_width + position[None, :])
    x_lo = tl.load(pre_hadamard_ptr + row * size_n + lo[:, None] * power_width + p[None, :])
    x_hi = tl.load(
        pre_hadamard_ptr + row * size_n + hi[:, None] * power_width + p[None, :],
        hi[:, None] < base_size, 0.0,
    )
    stage_lo = tl.dot(x_lo, power, input_precision="ieee")
    stage_hi = tl.dot(x_hi, power, input_precision="ieee")
    base_lo = tl.load(base_ptr + out[:, None] * base_pad + lo[None, :])
    base_hi = tl.load(base_ptr + out[:, None] * base_pad + hi[None, :])
    value = tl.dot(base_lo, stage_lo, input_precision="ieee")
    value = tl.dot(base_hi, stage_hi, value, input_precision="ieee")
    columns = out[:, None] * power_width + position[None, :]
    scale = tl.load(sv_ptr + columns, out[:, None] < base_size, 0.0)
    tl.store(output_ptr + row * size_n + columns, value * inv_sqrt_n * scale, out[:, None] < base_size)


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
