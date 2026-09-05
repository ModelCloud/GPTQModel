"""Experimental Hadamard kernels; never imported by production dispatch."""

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import (
    BlockedLayout,
    DotOperandLayout,
    SliceLayout,
)
from triton.experimental.gluon.language.amd import AMDMFMALayout
from triton.experimental.gluon.language.amd.cdna4 import mfma


@gluon.jit
def folded_residual_gemm_gluon_kernel(
    x_ptr, high_ptr, low_ptr, output_ptr,
    size_m: gl.constexpr, size_n: gl.constexpr, size_k: gl.constexpr,
    block_m: gl.constexpr, block_n: gl.constexpr, block_k: gl.constexpr,
    interleave: gl.constexpr = False,
):
    """Share X loads between two FP32-accumulating GEMMs; reassociation experiment."""
    gl.static_assert(size_k % block_k == 0)
    a_layout: gl.constexpr = BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    b_layout: gl.constexpr = BlockedLayout([4, 1], [16, 4], [1, 4], [0, 1])
    mma_layout: gl.constexpr = AMDMFMALayout(4, [16, 16, 32], False, [2, 2])
    rm = gl.program_id(0) * block_m + gl.arange(0, block_m, layout=SliceLayout(1, a_layout))
    ak = gl.arange(0, block_k, layout=SliceLayout(0, a_layout))
    bk = gl.arange(0, block_k, layout=SliceLayout(1, b_layout))
    rn = gl.program_id(1) * block_n + gl.arange(0, block_n, layout=SliceLayout(0, b_layout))
    primary = gl.full((block_m, block_n), 0, gl.float32, mma_layout)
    correction = gl.full((block_m, block_n), 0, gl.float32, mma_layout)
    for offset in range(size_k // block_k):
        x = gl.load(x_ptr + rm[:, None] * size_k + (offset * block_k + ak[None, :]),
                    rm[:, None] < size_m, 0)
        indices = offset * block_k + bk[:, None] + rn[None, :] * size_k
        high = gl.load(high_ptr + indices, rn[None, :] < size_n, 0)
        low = gl.load(low_ptr + indices, rn[None, :] < size_n, 0)
        a = gl.convert_layout(x, DotOperandLayout(0, mma_layout, 8))
        b = gl.convert_layout(high, DotOperandLayout(1, mma_layout, 8))
        c = gl.convert_layout(low, DotOperandLayout(1, mma_layout, 8))
        primary = mfma(a, b, primary)
        if interleave:
            primary = mfma(a, c, primary)
        else:
            correction = mfma(a, c, correction)
    out_layout: gl.constexpr = BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    if interleave:
        result = gl.convert_layout(primary, out_layout)
    else:
        result = gl.convert_layout(primary + correction, out_layout)
    om = gl.program_id(0) * block_m + gl.arange(0, block_m, layout=SliceLayout(1, out_layout))
    on = gl.program_id(1) * block_n + gl.arange(0, block_n, layout=SliceLayout(0, out_layout))
    gl.store(output_ptr + om[:, None] * size_n + on[None, :], result,
             (om[:, None] < size_m) & (on[None, :] < size_n))


@gluon.jit
def folded_gemv_dot2_gluon_kernel(
    input_ptr, weight_ptr, residual_ptr, output_ptr,
    size_k: gl.constexpr, block_n: gl.constexpr, block_k: gl.constexpr,
    use_residual: gl.constexpr,
):
    """Explicit wave ownership; experimental FP32 reassociation, not a production path."""
    gl.static_assert(size_k % block_k == 0)
    layout: gl.constexpr = BlockedLayout([1, 4], [1, 64], [4, 1], [1, 0])
    n = gl.program_id(0) * block_n + gl.arange(0, block_n, layout=SliceLayout(1, layout))
    if use_residual:
        k = gl.arange(0, block_k, layout=SliceLayout(0, layout))
        partial = gl.full((block_n, block_k), 0, gl.float32, layout)
        for offset in range(size_k // block_k):
            c = offset * block_k + k
            x = gl.load(input_ptr + c).to(gl.float32)
            w = gl.load(weight_ptr + n[:, None] * size_k + c[None, :]).to(gl.float32)
            w += gl.load(residual_ptr + n[:, None] * size_k + c[None, :]).to(gl.float32)
            partial += w * x[None, :]
    else:
        pair = gl.arange(0, block_k // 2, layout=SliceLayout(0, layout))
        partial = gl.full((block_n, block_k // 2), 0, gl.float32, layout)
        for offset in range(size_k // block_k):
            c = offset * (block_k // 2) + pair
            x = gl.load(input_ptr.to(gl.pointer_type(gl.int32)) + c)
            w = gl.load(weight_ptr.to(gl.pointer_type(gl.int32)) + n[:, None] * (size_k // 2) + c[None, :])
            # Conservative dependency workaround validated on gfx950. The minimum
            # delay is not established; removing it corrupts the all-ones test.
            partial = gl.inline_asm_elementwise(
                "v_dot2c_f32_f16 $0, $2, $3\n s_nop 7", "=v,0,v,v",
                [partial, w, x[None, :]],
                dtype=gl.float32, is_pure=True, pack=1,
            )
    gl.store(output_ptr + n, gl.sum(partial, 1))


@triton.jit
def folded_gemv_dot2_loop_kernel(
    input_ptr, weight_ptr, residual_ptr, output_ptr,
    size_k: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
    use_residual: tl.constexpr,
):
    """Accumulate packed FP32 partials across K tiles, then reduce once."""
    tl.static_assert(size_k % block_k == 0)
    n = tl.program_id(0) * block_n + tl.arange(0, block_n)
    if use_residual:
        k = tl.arange(0, block_k)
        partial = tl.full((block_n, block_k), 0, tl.float32)
        for offset in range(size_k // block_k):
            c = offset * block_k + k
            x = tl.load(input_ptr + c).to(tl.float32)
            w = tl.load(weight_ptr + n[:, None] * size_k + c[None, :]).to(tl.float32)
            w += tl.load(residual_ptr + n[:, None] * size_k + c[None, :]).to(tl.float32)
            partial += w * x[None, :]
    else:
        pair = tl.arange(0, block_k // 2)
        partial = tl.full((block_n, block_k // 2), 0, tl.float32)
        for offset in range(size_k // block_k):
            c = offset * (block_k // 2) + pair
            x = tl.load(input_ptr.to(tl.pointer_type(tl.int32)) + c)
            w = tl.load(weight_ptr.to(tl.pointer_type(tl.int32)) + n[:, None] * (size_k // 2) + c[None, :])
            # Match the validated conservative delay in the Gluon experiment.
            partial = tl.inline_asm_elementwise(
                "v_dot2c_f32_f16 $0, $2, $3\n s_nop 7", "=v,0,v,v",
                [partial, w, tl.broadcast_to(x[None, :], (block_n, block_k // 2))],
                dtype=tl.float32, is_pure=True, pack=1,
            )
    tl.store(output_ptr + n, tl.sum(partial, 1))


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
