"""Experimental Hadamard kernels; never imported by production dispatch."""

import triton
import triton.language as tl


@triton.jit
def folded_gemv_dot2_kernel(
    input_ptr, weight_ptr, residual_ptr, output_ptr,
    size_k: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
    use_residual: tl.constexpr,
):
    """FP16 packed products with FP32 accumulation, not FP16 product rounding."""
    if use_residual:
        folded_gemv_full_k_kernel(
            input_ptr, weight_ptr, residual_ptr, output_ptr, size_k, block_n, block_k, use_residual,
        )
    else:
        pair = tl.arange(0, block_k // 2)
        n = tl.program_id(0) * block_n + tl.arange(0, block_n)
        x = tl.load(input_ptr.to(tl.pointer_type(tl.int32)) + pair, pair < size_k // 2, 0)
        w = tl.load(
            weight_ptr.to(tl.pointer_type(tl.int32)) + n[:, None] * (size_k // 2) + pair[None, :],
            pair[None, :] < size_k // 2, 0,
        )
        dot = tl.inline_asm_elementwise(
            "v_dot2c_f32_f16 $0, $2, $3", "=v,0,v,v",
            [tl.full((block_n, block_k // 2), 0, tl.float32), w,
             tl.broadcast_to(x[None, :], (block_n, block_k // 2))],
            dtype=tl.float32, is_pure=True, pack=1,
        )
        tl.store(output_ptr + n, tl.sum(dot, 1))


@triton.jit
def folded_gemv_split_k_kernel(
    input_ptr, weight_ptr, residual_ptr, output_ptr,
    size_k: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
    use_residual: tl.constexpr,
):
    """Remove full-K power-of-two padding with two exact power-of-two pieces."""
    tl.static_assert(size_k == 5120 or size_k == 6144)
    n = tl.program_id(0) * block_n + tl.arange(0, block_n)
    k0 = tl.arange(0, 4096)
    k1 = 4096 + tl.arange(0, size_k - 4096)
    x0 = tl.load(input_ptr + k0).to(tl.float32)
    x1 = tl.load(input_ptr + k1).to(tl.float32)
    w0 = tl.load(weight_ptr + n[:, None] * size_k + k0[None, :]).to(tl.float32)
    w1 = tl.load(weight_ptr + n[:, None] * size_k + k1[None, :]).to(tl.float32)
    if use_residual:
        w0 += tl.load(residual_ptr + n[:, None] * size_k + k0[None, :]).to(tl.float32)
        w1 += tl.load(residual_ptr + n[:, None] * size_k + k1[None, :]).to(tl.float32)
    total = tl.sum(w0 * x0[None, :], 1) + tl.sum(w1 * x1[None, :], 1)
    tl.store(output_ptr + n, total)


@triton.jit
def folded_gemv_full_k_kernel(
    input_ptr, weight_ptr, residual_ptr, output_ptr,
    size_k: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
    use_residual: tl.constexpr,
):
    """Single full-K reduction instead of reducing and accumulating each K tile."""
    n = tl.program_id(0) * block_n + tl.arange(0, block_n)
    k = tl.arange(0, block_k)
    x = tl.load(input_ptr + k, k < size_k, 0).to(tl.float32)
    w = tl.load(weight_ptr + n[:, None] * size_k + k[None, :], k[None, :] < size_k, 0).to(tl.float32)
    if use_residual:
        w += tl.load(residual_ptr + n[:, None] * size_k + k[None, :], k[None, :] < size_k, 0).to(tl.float32)
    tl.store(output_ptr + n, tl.sum(w * x[None, :], 1))


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
