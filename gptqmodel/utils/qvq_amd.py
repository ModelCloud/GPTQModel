# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""ROCm P32 inference kernels specialized for AMD Instinct MI355X."""

from __future__ import annotations

import math
import threading
from operator import index

import torch
import triton
import triton.language as tl

from ..quantization.qvq_rates import (
    normalize_qvq_rate,
    qvq_transition_bits,
    qvq_words_per_tile,
)

_P32_RATES = (2.0, 2.5, 3.0, 3.5)
_GFX950_ARCH = "gfx950"
_QWEN38_27B_FOLDED_SHAPES = frozenset(
    {
        (5120, 12288),
        (5120, 1024),
        (5120, 10240),
        (5120, 6144),
        (6144, 5120),
        (5120, 17408),
        (17408, 5120),
    }
)
_QWEN38_27B_FOLDED_M_LIMITS = {
    (6144, 5120): 4096,
    (5120, 17408): 512,
    (17408, 5120): 1024,
}
_QWEN38_27B_RESIDUAL_FOLDED_SHAPES = frozenset({(17408, 5120), (6144, 5120)})
_QWEN38_27B_COMPOSITE_RECOVERY_SHAPE = (17408, 5120)
_COMPOSITE_HADAMARD_CACHE: dict[
    torch.device, tuple[torch.Tensor, torch.Tensor, int, int]
] = {}
_COMPOSITE_HADAMARD_CACHE_LOCK = threading.Lock()


def qvq_p32_amd_supported(device: torch.device | str) -> bool:
    """Return whether ``device`` is the measured gfx950 ROCm target."""

    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available() or torch.version.hip is None:
        return False
    try:
        properties = torch.cuda.get_device_properties(target)
    except (AssertionError, RuntimeError):
        return False
    return str(getattr(properties, "gcnArchName", "")).split(":", 1)[0] == _GFX950_ARCH


def qvq_p32_amd_folded_shape_supported(in_features: int, out_features: int) -> bool:
    """Return whether the full-layer folded cache is measured for this geometry."""

    return (in_features, out_features) in _QWEN38_27B_FOLDED_SHAPES


def qvq_p32_amd_folded_case_supported(m: int, in_features: int, out_features: int) -> bool:
    """Return whether a measured geometry is accurate at this batch size."""

    shape = (in_features, out_features)
    if m <= 0 or shape not in _QWEN38_27B_FOLDED_SHAPES:
        return False
    return m <= _QWEN38_27B_FOLDED_M_LIMITS.get(shape, m)


def qvq_p32_amd_folded_prefers_fp32_output(m: int, in_features: int, out_features: int) -> bool:
    """Keep FP32 output where gfx950's direct-FP16 GEMM algorithm regresses."""

    return ((in_features, out_features) == (5120, 1024) and m == 4096) or (
        (in_features, out_features) == (5120, 6144) and m == 512
    )


def _bank_mask(transition_bits: int, bank_alt_id: int) -> int:
    masks = {
        4: (0x0000, 0x5A5A, 0x3C3C, 0xC3C3),
        5: (0x0000, 0x9696, 0x3C3C, 0xC3C3),
        6: (0x0000, 0x6969, 0x5A5A, 0x3C3C),
        7: (0x0000, 0xC3C3, 0x9696, 0x5A5A),
    }
    return masks[transition_bits][bank_alt_id]


def _launch_config(m: int, n: int = 4096, k: int = 4096) -> tuple[int, int, int]:
    """Choose an MFMA tile without materializing a shape Cartesian product."""

    if m <= 16:
        return 16, 64, 8
    if m == 64 and n >= 10240:
        return 64, 64, 8
    if m <= 64:
        return 32, 64, 8
    if m <= 256 and n <= 1024:
        return 32, 64, 8
    if m <= 1024 and n <= 1024:
        return 64, 64, 8
    if m == 128 and k <= 6144 and n <= 6144:
        return 64, 64, 8
    if m == 1024 and 8192 < n <= 12288:
        return 1024, 64, 8
    if m == 1024 and n > 1024:
        return 512, 64, 8
    if m == 512 and 8192 < n <= 12288:
        return 512, 64, 8
    if m == 512 and n > 1024:
        return 256, 64, 8
    if m == 256 and n >= 10240:
        return 256, 64, 8
    if m == 2048 and n <= 1024:
        return 128, 64, 8
    if m >= 4096 and n <= 1024:
        return 256, 64, 8
    if m >= 2048:
        return 512, 64, 8
    return 128, 64, 8


def _use_gemv(m: int, n: int) -> bool:
    num_pid_n = triton.cdiv(n, 64)
    return m == 1 or (m <= 4 and m * num_pid_n <= 256)


def _integer_argument(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        return index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


@triton.jit
def _qvq_p32_predecode_gfx950_kernel(  # pragma: no cover - compiled and exercised on the GPU
    trellis_ptr,
    levels_ptr,
    bank_ids_ptr,
    dense_ptr,
    size_k: tl.constexpr,
    size_n: tl.constexpr,
    transition_bits: tl.constexpr,
    words_per_tile: tl.constexpr,
    alternate_mask: tl.constexpr,
    block_n: tl.constexpr,
):
    """Expand four adjacent K16xN16 tiles into a preshuffled N-by-K operand."""

    pid = tl.program_id(0)
    n_blocks = tl.cdiv(size_n, block_n)
    k_tile = pid // n_blocks
    n_block = pid % n_blocks
    local_k = tl.arange(0, 16)[:, None]
    pair_columns = n_block * (block_n // 2) + tl.arange(0, block_n // 2)
    pair_column_mask = pair_columns < size_n // 2
    local_pair_n = pair_columns[None, :] & 7
    pair = local_k * 8 + local_pair_n
    tile = k_tile * (size_n // 16) + pair_columns[None, :] // 8
    bit_position = (127 - pair) * transition_bits
    first_word = bit_position >> 5
    shift = bit_position & 31
    next_word = tl.where(first_word + 1 == words_per_tile, 0, first_word + 1)
    word_base = tile * words_per_tile
    low = tl.load(
        trellis_ptr + word_base + first_word,
        mask=pair_column_mask[None, :],
        other=0,
    ).to(tl.uint32)
    high_mask = pair_column_mask[None, :]
    if transition_bits == 4:
        high_mask &= shift > 16
    high = tl.load(trellis_ptr + word_base + next_word, mask=high_mask, other=0).to(tl.uint32)
    state = tl.inline_asm_elementwise(
        "v_alignbit_b32 $0, $2, $1, $3",
        "=v,v,v,v",
        [low, high, shift],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ) & 0xFFFF
    packed_bank = tl.load(bank_ids_ptr + tile, mask=pair_column_mask[None, :], other=0)
    selected = (packed_bank >> (pair >> 4)) & 1
    mixed = state ^ (state >> 8)
    mixed ^= selected * (alternate_mask ^ (alternate_mask >> 8))
    mixed = tl.inline_asm_elementwise(
        "v_mad_u32_u24 $0, $1, $2, $3",
        "=v,v,s,v",
        [mixed, 40503, 17011],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ) & 0xFFFF
    mixed ^= mixed >> 7
    even_weight = tl.load(levels_ptr + (mixed >> 8)).to(tl.float16)
    odd_weight = tl.load(levels_ptr + (mixed & 0xFF)).to(tl.float16)
    weight = tl.interleave(even_weight, odd_weight)
    columns = n_block * block_n + tl.arange(0, block_n)
    offsets = columns[None, :] * size_k + (
        k_tile * 16 + tl.arange(0, 16)
    )[:, None]
    tl.store(dense_ptr + offsets, weight, mask=columns[None, :] < size_n)


def _qvq_p32_predecoded_weight(
    window: torch.Tensor,
    levels: torch.Tensor,
    bank_ids: torch.Tensor,
    *,
    bits: float,
    size_k: int,
    size_n: int,
    transition_bits: int,
    words_per_tile: int,
    bank_alt_id: int,
) -> torch.Tensor:
    """Return a mutation-aware transient FP16 cache for repeated inference."""

    key = (
        window._version,
        levels,
        levels._version,
        bank_ids,
        bank_ids._version,
        bits,
        size_k,
        size_n,
        bank_alt_id,
    )
    cached = getattr(window, "_qvq_p32_amd_dense_cache", None)
    if cached is not None:
        cached_key, dense, _ = cached
        if all(
            current is recorded
            if isinstance(current, torch.Tensor)
            else current == recorded
            for current, recorded in zip(key, cached_key, strict=True)
        ):
            return dense

    block_n = 64
    dense = torch.empty((size_n, size_k), dtype=torch.float16, device=window.device)
    grid = ((size_k // 16) * triton.cdiv(size_n, block_n),)
    _qvq_p32_predecode_gfx950_kernel[grid](
        window,
        levels,
        bank_ids,
        dense,
        size_k=size_k,
        size_n=size_n,
        transition_bits=transition_bits,
        words_per_tile=words_per_tile,
        alternate_mask=_bank_mask(transition_bits, bank_alt_id),
        block_n=block_n,
        num_warps=4,
        num_stages=1,
        waves_per_eu=0,
    )
    window._qvq_p32_amd_dense_cache = (key, dense, dense.T)
    return dense


def _qvq_p32_folded_weight(
    window: torch.Tensor,
    levels: torch.Tensor,
    bank_ids: torch.Tensor,
    su: torch.Tensor,
    sv: torch.Tensor,
    *,
    bits: float,
    size_k: int,
    size_n: int,
    transition_bits: int,
    words_per_tile: int,
    bank_alt_id: int,
    input_hadamard: bool,
    output_hadamard: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, tuple | None]:
    """Fold immutable QVQ axes into mutation-aware N-by-K FP16 cache operands."""

    key = (
        window._version,
        levels,
        levels._version,
        bank_ids,
        bank_ids._version,
        bits,
        size_k,
        size_n,
        bank_alt_id,
        su,
        su._version,
        sv,
        sv._version,
        input_hadamard,
        output_hadamard,
    )
    cached = getattr(window, "_qvq_p32_amd_folded_cache", None)
    if cached is not None:
        cached_key, _, operand, _, residual_operand, composite_recovery = cached
        if all(
            current is recorded
            if isinstance(current, torch.Tensor)
            else current == recorded
            for current, recorded in zip(key, cached_key, strict=True)
        ):
            # A caller may have populated the ordinary predecode cache through
            # forward_pretransformed after this folded entry was built.
            window._qvq_p32_amd_dense_cache = None
            return operand, residual_operand, composite_recovery

    dense = _qvq_p32_predecoded_weight(
        window,
        levels,
        bank_ids,
        bits=bits,
        size_k=size_k,
        size_n=size_n,
        transition_bits=transition_bits,
        words_per_tile=words_per_tile,
        bank_alt_id=bank_alt_id,
    )
    folded_fp32 = dense.to(torch.float32)
    composite_recovery = None
    use_composite_recovery = (
        (size_k, size_n) == _QWEN38_27B_COMPOSITE_RECOVERY_SHAPE
        and not input_hadamard
        and output_hadamard
    )
    if input_hadamard or (output_hadamard and not use_composite_recovery):
        from ..quantization.rotation.hadamard_utils import matmul_hadU

        # dense is W^T. Input folding right-multiplies it by H_K^T;
        # output folding transposes around the row-oriented H_N helper.
        if input_hadamard:
            folded_fp32 = matmul_hadU(folded_fp32, transpose=True)
        if output_hadamard and not use_composite_recovery:
            folded_fp32 = matmul_hadU(folded_fp32.T.contiguous()).T.contiguous()
    folded_fp32.mul_(su.to(torch.float32)[None, :])
    if not use_composite_recovery:
        folded_fp32.mul_(sv.to(torch.float32)[:, None])
    if use_composite_recovery:
        # hipBLASLt selects a much faster small-M path when its K-by-N input is
        # physically contiguous.  Keep the public/cache N-by-K view without
        # paying for a second persistent copy.
        operand = folded_fp32.T.to(torch.float16).contiguous()
        folded = operand.T
    else:
        folded = folded_fp32.to(torch.float16).contiguous()
        operand = folded.T
    residual = None
    residual_operand = None
    if use_composite_recovery:
        composite_recovery = (*_qvq_p32_composite_hadamard_constants(window.device), sv)
    elif (size_k, size_n) in _QWEN38_27B_RESIDUAL_FOLDED_SHAPES:
        # This shape narrowly misses the end-to-end 2e-3 error budget when the
        # FP32 folded matrix is rounded once.  A second FP16 expansion term
        # preserves that accuracy while remaining much cheaper than decoding
        # P32 weights on every token. Attention output needs the second term at
        # M>=64; keep it cached across mixed decode/prefill calls. This doubles
        # these layers' persistent folded-cache footprint.
        residual = (folded_fp32 - folded.to(torch.float32)).to(torch.float16).contiguous()
        residual_operand = residual.T
    # The predecoded matrix is only a construction intermediate here. Keeping
    # it would double the persistent dense-cache footprint for every layer.
    window._qvq_p32_amd_dense_cache = None
    window._qvq_p32_amd_folded_cache = (
        key,
        folded,
        operand,
        residual,
        residual_operand,
        composite_recovery,
    )
    return operand, residual_operand, composite_recovery


def _qvq_p32_composite_hadamard_constants(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Return shared FP32 matrices for the 40-by-128 Qwen down transform."""

    with _COMPOSITE_HADAMARD_CACHE_LOCK:
        cached = _COMPOSITE_HADAMARD_CACHE.get(device)
        if cached is not None:
            return cached
        from ..quantization.rotation.hadamard_utils import get_hadK

        base, base_size = get_hadK(5120)
        if base is None or base_size != 40:
            raise RuntimeError("Qwen3.8 down recovery requires the canonical K=40 Hadamard base")
        base_pad = 64
        padded_base = torch.zeros((base_pad, base_pad), dtype=torch.float32, device=device)
        padded_base[:base_size, :base_size] = base.to(device=device, dtype=torch.float32)
        power_width = 5120 // base_size
        power = torch.tensor(
            [
                [
                    1.0 if (row & column).bit_count() % 2 == 0 else -1.0
                    for column in range(power_width)
                ]
                for row in range(power_width)
            ],
            dtype=torch.float32,
            device=device,
        )
        cached = (padded_base, power, base_size, power_width)
        _COMPOSITE_HADAMARD_CACHE[device] = cached
        return cached


@triton.jit
def _qvq_p32_composite_recovery_gfx950_kernel(  # pragma: no cover - compiled and exercised on the GPU
    pre_hadamard_ptr,
    power_ptr,
    base_ptr,
    sv_ptr,
    output_ptr,
    size_n: tl.constexpr,
    base_size: tl.constexpr,
    base_pad: tl.constexpr,
    power_width: tl.constexpr,
    block_p: tl.constexpr,
    inv_sqrt_n: tl.constexpr,
):
    """Fuse both composite output-Hadamard factors, SV scale, and cast."""

    row = tl.program_id(0)
    position = tl.program_id(1) * block_p + tl.arange(0, block_p)
    output_base = tl.arange(0, base_pad)
    power_input = tl.arange(0, power_width)
    pre_hadamard = tl.load(
        pre_hadamard_ptr
        + row * size_n
        + output_base[:, None] * power_width
        + power_input[None, :],
        mask=output_base[:, None] < base_size,
        other=0.0,
    )
    power = tl.load(
        power_ptr + power_input[:, None] * power_width + position[None, :]
    )
    staged = tl.dot(pre_hadamard, power, input_precision="ieee")
    base = tl.load(
        base_ptr + output_base[:, None] * base_pad + output_base[None, :]
    )
    transformed = tl.dot(base, staged, input_precision="ieee")
    columns = output_base[:, None] * power_width + position[None, :]
    scale = tl.load(sv_ptr + columns, mask=output_base[:, None] < base_size, other=0.0)
    tl.store(
        output_ptr + row * size_n + columns,
        transformed * inv_sqrt_n * scale,
        mask=output_base[:, None] < base_size,
    )


@triton.jit
def _qvq_p32_composite_base_gfx950_kernel(  # pragma: no cover - compiled and exercised on the GPU
    staged_ptr,
    base_ptr,
    sv_ptr,
    output_ptr,
    size_n: tl.constexpr,
    base_size: tl.constexpr,
    base_pad: tl.constexpr,
    power_width: tl.constexpr,
    block_p: tl.constexpr,
    inv_sqrt_n: tl.constexpr,
):
    """Apply the small composite base after rocBLAS handles the power factor."""

    row = tl.program_id(0)
    position = tl.program_id(1) * block_p + tl.arange(0, block_p)
    output_base = tl.arange(0, base_pad)
    reduce_low = tl.arange(0, 32)
    base_low = tl.load(
        base_ptr + output_base[:, None] * base_pad + reduce_low[None, :]
    )
    staged_low = tl.load(
        staged_ptr
        + row * size_n
        + reduce_low[:, None] * power_width
        + position[None, :]
    )
    transformed = tl.dot(base_low, staged_low, input_precision="ieee")
    reduce_high = 32 + tl.arange(0, 16)
    base_high = tl.load(
        base_ptr + output_base[:, None] * base_pad + reduce_high[None, :]
    )
    staged_high = tl.load(
        staged_ptr
        + row * size_n
        + reduce_high[:, None] * power_width
        + position[None, :],
        mask=reduce_high[:, None] < base_size,
        other=0.0,
    )
    transformed += tl.dot(base_high, staged_high, input_precision="ieee")
    columns = output_base[:, None] * power_width + position[None, :]
    scale = tl.load(sv_ptr + columns, mask=output_base[:, None] < base_size, other=0.0)
    tl.store(
        output_ptr + row * size_n + columns,
        transformed * inv_sqrt_n * scale,
        mask=output_base[:, None] < base_size,
    )


@triton.jit
def _qvq_p32_folded_gemv_gfx950_kernel(  # pragma: no cover - compiled and exercised on the GPU
    input_ptr,
    weight_ptr,
    residual_ptr,
    output_ptr,
    size_k: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    use_residual: tl.constexpr,
):
    """Multiply one row by the cached physical N-by-K folded weight."""

    rows = tl.program_id(0) * block_n + tl.arange(0, block_n)
    accumulator = tl.zeros((block_n,), dtype=tl.float32)
    for k_block in range(size_k // block_k):
        columns = k_block * block_k + tl.arange(0, block_k)
        activation = tl.load(input_ptr + columns).to(tl.float32)
        weight = tl.load(weight_ptr + rows[:, None] * size_k + columns[None, :]).to(tl.float32)
        if use_residual:
            weight += tl.load(
                residual_ptr + rows[:, None] * size_k + columns[None, :]
            ).to(tl.float32)
        accumulator += tl.sum(weight * activation[None, :], axis=1)
    tl.store(output_ptr + rows, accumulator)


def _qvq_p32_folded_execute(
    x: torch.Tensor,
    operand: torch.Tensor,
    residual_operand: torch.Tensor | None = None,
    composite_recovery: tuple | None = None,
    *,
    out_features: int,
    output_fp32: bool,
) -> torch.Tensor:
    """Run the measured folded-cache kernel after its operands are validated."""

    m, k = x.shape
    n = out_features
    if composite_recovery is not None:
        padded_base, power, base_size, power_width, sv = composite_recovery
        pre_hadamard = torch.mm(x, operand, out_dtype=torch.float32)
        output = torch.empty(
            (m, n),
            device=x.device,
            dtype=torch.float32 if output_fp32 else x.dtype,
        )
        block_p = 32 if m <= 64 else 64
        if m == 1024:
            staged = torch.mm(
                pre_hadamard.view(m * base_size, power_width), power
            ).view(m, base_size, power_width)
            _qvq_p32_composite_base_gfx950_kernel[(m, power_width // block_p)](
                staged,
                padded_base,
                sv,
                output,
                size_n=n,
                base_size=base_size,
                base_pad=padded_base.shape[0],
                power_width=power_width,
                block_p=block_p,
                inv_sqrt_n=1.0 / math.sqrt(n),
                num_warps=8,
                num_stages=1,
                waves_per_eu=0,
                matrix_instr_nonkdim=16,
                kpack=1,
            )
            return output
        _qvq_p32_composite_recovery_gfx950_kernel[(m, power_width // block_p)](
            pre_hadamard,
            power,
            padded_base,
            sv,
            output,
            size_n=n,
            base_size=base_size,
            base_pad=padded_base.shape[0],
            power_width=power_width,
            block_p=block_p,
            inv_sqrt_n=1.0 / math.sqrt(n),
            num_warps=8,
            num_stages=1,
            waves_per_eu=0,
            matrix_instr_nonkdim=16,
            kpack=1,
        )
        return output
    if (k, n) == (6144, 5120) and m <= 32:
        # Preserve the retained small-M arithmetic and single-weight read even
        # when the same module previously populated its large-M residual cache.
        residual_operand = None
    if m == 1 and not output_fp32 and qvq_p32_amd_folded_shape_supported(k, n):
        output = torch.empty((1, n), device=x.device, dtype=x.dtype)
        if n == 1024:
            block_n, block_k = 16, 256
        elif n == 12288:
            block_n, block_k = 8, 256
        elif n == 10240:
            block_n, block_k = 8, 512
        else:
            block_n, block_k = 4, 512
        _qvq_p32_folded_gemv_gfx950_kernel[(n // block_n,)](
            x,
            # operand is a K-by-N transpose view over physical N-by-K storage.
            operand,
            operand if residual_operand is None else residual_operand,
            output,
            size_k=k,
            block_n=block_n,
            block_k=block_k,
            use_residual=residual_operand is not None,
            num_warps=4,
            num_stages=1,
            waves_per_eu=0,
        )
        return output
    if residual_operand is not None:
        primary = torch.mm(x, operand, out_dtype=torch.float32)
        output = torch.addmm(
            primary,
            x,
            residual_operand,
            out_dtype=torch.float32,
        )
        return output if output_fp32 else output.to(x.dtype)
    return torch.mm(x, operand, out_dtype=torch.float32 if output_fp32 else x.dtype)


@triton.jit
def _qvq_p32_gfx950_kernel(  # pragma: no cover - compiled and exercised on the GPU
    input_ptr,
    trellis_ptr,
    levels_ptr,
    bank_ids_ptr,
    output_ptr,
    size_m: tl.constexpr,
    size_k: tl.constexpr,
    size_n: tl.constexpr,
    transition_bits: tl.constexpr,
    words_per_tile: tl.constexpr,
    alternate_mask: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    num_pid_n: tl.constexpr,
    xcd_swizzle: tl.constexpr,
):
    pid = tl.program_id(0)
    if xcd_swizzle:
        num_pid = tl.num_programs(0)
        pid = (pid % 8) * (num_pid // 8) + pid // 8
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    rows = pid_m * block_m + tl.arange(0, block_m)
    columns = pid_n * block_n + tl.arange(0, block_n)
    row_mask = rows < size_m
    column_mask = columns < size_n
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    n_tiles = size_n // 16

    for k_block in range(size_k // block_k):
        local_k = tl.arange(0, block_k)[:, None]
        pair_columns = pid_n * (block_n // 2) + tl.arange(0, block_n // 2)
        pair_column_mask = pair_columns < size_n // 2
        local_pair_n = pair_columns[None, :] & 7
        pair = (local_k & 15) * 8 + local_pair_n
        tile = (
            (k_block * (block_k // 16) + (local_k >> 4)) * n_tiles
            + pair_columns[None, :] // 8
        )
        bit_position = (127 - pair) * transition_bits
        first_word = bit_position >> 5
        shift = bit_position & 31
        next_word = tl.where(first_word + 1 == words_per_tile, 0, first_word + 1)
        word_base = tile * words_per_tile
        low = tl.load(
            trellis_ptr + word_base + first_word,
            mask=pair_column_mask[None, :],
            other=0,
        ).to(tl.uint32)
        high_mask = pair_column_mask[None, :]
        if transition_bits == 4:
            high_mask &= shift > 16
        high = tl.load(
            trellis_ptr + word_base + next_word,
            mask=high_mask,
            other=0,
        ).to(tl.uint32)
        if size_m >= 32 and size_m <= 512:
            state = tl.inline_asm_elementwise(
                "v_alignbit_b32 $0, $2, $1, $3",
                "=v,v,v,v",
                [low, high, shift],
                dtype=tl.uint32,
                is_pure=True,
                pack=1,
            ) & 0xFFFF
        else:
            state = tl.where(
                shift == 0,
                low,
                (low >> shift) | (high << ((32 - shift) & 31)),
            ) & 0xFFFF

        packed_bank = tl.load(bank_ids_ptr + tile, mask=pair_column_mask[None, :], other=0)
        selected = (packed_bank >> (pair >> 4)) & 1
        mixed = state ^ (state >> 8)
        mixed = mixed ^ (selected * (alternate_mask ^ (alternate_mask >> 8)))
        mixed = tl.inline_asm_elementwise(
            "v_mad_u32_u24 $0, $1, $2, $3",
            "=v,v,s,v",
            [mixed, 40503, 17011],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        ) & 0xFFFF
        mixed = mixed ^ (mixed >> 7)
        even_weight = tl.load(levels_ptr + (mixed >> 8)).to(tl.float16)
        odd_weight = tl.load(levels_ptr + (mixed & 0xFF)).to(tl.float16)
        weight = tl.interleave(even_weight, odd_weight)
        input_offsets = rows[:, None] * size_k + k_block * block_k + tl.arange(0, block_k)[None, :]
        activation = tl.load(input_ptr + input_offsets, mask=row_mask[:, None], other=0.0)
        accumulator = tl.dot(activation, weight, accumulator)

    output_offsets = rows[:, None] * size_n + columns[None, :]
    tl.store(output_ptr + output_offsets, accumulator, mask=row_mask[:, None] & column_mask[None, :])


@triton.jit
def _qvq_p32_gemv_gfx950_kernel(  # pragma: no cover - compiled and exercised on the GPU
    input_ptr,
    trellis_ptr,
    levels_ptr,
    bank_ids_ptr,
    output_ptr,
    size_m: tl.constexpr,
    size_k: tl.constexpr,
    size_n: tl.constexpr,
    transition_bits: tl.constexpr,
    words_per_tile: tl.constexpr,
    alternate_mask: tl.constexpr,
    block_n: tl.constexpr,
    num_pid_n: tl.constexpr,
    xcd_swizzle: tl.constexpr,
):
    pid = tl.program_id(0)
    if xcd_swizzle:
        num_pid = tl.num_programs(0)
        pid = (pid % 8) * (num_pid // 8) + pid // 8
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    columns = pid_n * block_n + tl.arange(0, block_n)
    column_mask = columns < size_n
    accumulator = tl.zeros((block_n,), dtype=tl.float32)
    n_tiles = size_n // 16

    for k_tile in range(size_k // 16):
        local_k = tl.arange(0, 16)[:, None]
        pair_columns = pid_n * (block_n // 2) + tl.arange(0, block_n // 2)
        pair_column_mask = pair_columns < size_n // 2
        pair = local_k * 8 + (pair_columns[None, :] & 7)
        tile = k_tile * n_tiles + pair_columns[None, :] // 8
        bit_position = (127 - pair) * transition_bits
        first_word = bit_position >> 5
        shift = bit_position & 31
        next_word = tl.where(first_word + 1 == words_per_tile, 0, first_word + 1)
        word_base = tile * words_per_tile
        low = tl.load(
            trellis_ptr + word_base + first_word,
            mask=pair_column_mask[None, :],
            other=0,
        ).to(tl.uint32)
        high_mask = pair_column_mask[None, :]
        if transition_bits == 4:
            high_mask &= shift > 16
        high = tl.load(
            trellis_ptr + word_base + next_word,
            mask=high_mask,
            other=0,
        ).to(tl.uint32)
        state = tl.inline_asm_elementwise(
            "v_alignbit_b32 $0, $2, $1, $3",
            "=v,v,v,v",
            [low, high, shift],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        ) & 0xFFFF
        packed_bank = tl.load(bank_ids_ptr + tile, mask=pair_column_mask[None, :], other=0)
        selected = (packed_bank >> (pair >> 4)) & 1
        mixed = state ^ (state >> 8)
        mixed = mixed ^ (selected * (alternate_mask ^ (alternate_mask >> 8)))
        mixed = tl.inline_asm_elementwise(
            "v_mad_u32_u24 $0, $1, $2, $3",
            "=v,v,s,v",
            [mixed, 40503, 17011],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        ) & 0xFFFF
        mixed = mixed ^ (mixed >> 7)
        even_weight = tl.load(levels_ptr + (mixed >> 8)).to(tl.float16)
        odd_weight = tl.load(levels_ptr + (mixed & 0xFF)).to(tl.float16)
        weight = tl.interleave(even_weight, odd_weight)
        activation = tl.load(input_ptr + pid_m * size_k + k_tile * 16 + tl.arange(0, 16))
        accumulator += tl.sum(activation[:, None].to(tl.float32) * weight.to(tl.float32), axis=0)

    tl.store(output_ptr + pid_m * size_n + columns, accumulator, mask=column_mask)


def qvq_p32_amd(
    x: torch.Tensor,
    window: torch.Tensor,
    levels: torch.Tensor,
    bank_ids: torch.Tensor,
    bits: float,
    *,
    out_features: int,
    bank_alt_id: int,
    output_fp32: bool = True,
    cache_weight: bool = True,
) -> torch.Tensor:
    """Multiply FP16 activations by continuous-window V2B2-P32 tiles on gfx950.

    The default inference path lazily expands immutable P32 weights into a
    transient FP16 GEMM cache. Set ``cache_weight=False`` to retain the fused,
    storage-neutral decoder when runtime VRAM matters more than throughput.
    """

    bits = normalize_qvq_rate(bits)
    if bits not in _P32_RATES:
        raise ValueError("AMD P32 supports rates W2, W2.5, W3, and W3.5")
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    out_features = _integer_argument("out_features", out_features)
    bank_alt_id = _integer_argument("bank_alt_id", bank_alt_id)
    if not isinstance(output_fp32, bool):
        raise TypeError("output_fp32 must be boolean")
    if not isinstance(cache_weight, bool):
        raise TypeError("cache_weight must be boolean")

    # The first invocation performs the complete public-API validation below.
    # Repeated inference calls can safely avoid device-property queries, shape
    # reconstruction, and a fresh transpose view when the mutation-aware cache
    # still describes every fixed operand. HIP events include the host launch
    # gap, so keeping this path lean materially improves small-M latency.
    cached = getattr(window, "_qvq_p32_amd_dense_cache", None)
    if (
        cache_weight
        and isinstance(cached, tuple)
        and len(cached) == 3
        and type(bits) in (int, float)
        and not isinstance(bits, bool)
        and type(out_features) is int
        and type(bank_alt_id) is int
        and x.ndim == 2
        and x.dtype == torch.float16
        and x.is_contiguous()
    ):
        cached_key, _, operand = cached
        if (
            isinstance(cached_key, tuple)
            and len(cached_key) == 9
            and x.device == window.device
            and cached_key[0] == window._version
            and cached_key[1] is levels
            and cached_key[2] == levels._version
            and cached_key[3] is bank_ids
            and cached_key[4] == bank_ids._version
            and cached_key[5] == bits
            and cached_key[6] == x.shape[1]
            and cached_key[7] == out_features
            and cached_key[8] == bank_alt_id
        ):
            output_dtype = torch.float32 if output_fp32 else x.dtype
            return torch.mm(x, operand, out_dtype=output_dtype)

    if not qvq_p32_amd_supported(x.device):
        raise RuntimeError("AMD P32 requires a ROCm gfx950 device")
    if x.ndim != 2 or window.ndim != 2:
        raise ValueError("AMD P32 expects 2D input and window tensors")
    if x.dtype != torch.float16:
        raise TypeError("AMD P32 currently requires float16 input")
    if window.dtype != torch.int32:
        raise TypeError("AMD P32 requires int32 continuous-window words")
    if levels.dtype != torch.float16 or tuple(levels.shape) != (256,):
        raise TypeError("AMD P32 requires the canonical 256-entry float16 PGC16 table")
    if bank_ids.dtype != torch.uint8 or bank_ids.ndim != 1:
        raise TypeError("AMD P32 requires packed uint8 binary bank selectors")
    if any(tensor.device != x.device for tensor in (window, levels, bank_ids)):
        raise ValueError("AMD P32 tensors must share one device")
    if any(not tensor.is_contiguous() for tensor in (x, window, levels, bank_ids)):
        raise ValueError("AMD P32 tensors must be contiguous")
    if not 1 <= bank_alt_id <= 3:
        raise ValueError("AMD P32 alternative bank ID must be in [1, 3]")

    m, k = x.shape
    n = out_features
    if not m or k <= 0 or n <= 0 or k % 16 or n % 16:
        raise ValueError(f"AMD P32 requires positive M and positive K/N divisible by 16, got M={m}, K={k}, N={n}")
    tile_count = (k // 16) * (n // 16)
    expected_window = (tile_count, qvq_words_per_tile(bits, vector_size=2))
    if tuple(window.shape) != expected_window:
        raise ValueError(f"AMD P32 window must have shape {expected_window}")
    if tuple(bank_ids.shape) != (tile_count,):
        raise ValueError(f"AMD P32 selectors must have shape {(tile_count,)}")

    block_m, block_n, num_warps = _launch_config(m, n, k)
    block_k = 64 if m >= 128 else (32 if m == 64 and n >= 10240 else 16)
    output_dtype = torch.float32 if output_fp32 else x.dtype
    if cache_weight:
        dense = _qvq_p32_predecoded_weight(
            window,
            levels,
            bank_ids,
            bits=bits,
            size_k=k,
            size_n=n,
            transition_bits=transition_bits,
            words_per_tile=expected_window[1],
            bank_alt_id=bank_alt_id,
        )
        return torch.mm(x, dense.T, out_dtype=output_dtype)

    output = torch.empty((m, n), device=x.device, dtype=output_dtype)
    alternate_mask = _bank_mask(transition_bits, bank_alt_id)
    if _use_gemv(m, n):
        block_n = 64
        num_pid_n = triton.cdiv(n, block_n)
        grid = (m * num_pid_n,)
        _qvq_p32_gemv_gfx950_kernel[grid](
            x,
            window,
            levels,
            bank_ids,
            output,
            size_m=m,
            size_k=k,
            size_n=n,
            transition_bits=transition_bits,
            words_per_tile=expected_window[1],
            alternate_mask=alternate_mask,
            block_n=block_n,
            num_pid_n=num_pid_n,
            xcd_swizzle=(m * num_pid_n) % 8 == 0,
            num_warps=8,
            num_stages=1,
            waves_per_eu=0,
        )
        return output

    num_pid_m = triton.cdiv(m, block_m)
    num_pid_n = triton.cdiv(n, block_n)
    num_pid = num_pid_m * num_pid_n
    grid = (num_pid,)
    _qvq_p32_gfx950_kernel[grid](
        x,
        window,
        levels,
        bank_ids,
        output,
        size_m=m,
        size_k=k,
        size_n=n,
        transition_bits=transition_bits,
        words_per_tile=expected_window[1],
        alternate_mask=alternate_mask,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_pid_n=num_pid_n,
        xcd_swizzle=num_pid % 8 == 0,
        num_warps=num_warps,
        num_stages=1 if block_m == 1024 else (3 if m <= 64 else 2),
        waves_per_eu=0,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    return output


def qvq_p32_amd_folded(
    x: torch.Tensor,
    window: torch.Tensor,
    levels: torch.Tensor,
    bank_ids: torch.Tensor,
    su: torch.Tensor,
    sv: torch.Tensor,
    bits: float,
    *,
    out_features: int,
    bank_alt_id: int,
    input_hadamard: bool,
    output_hadamard: bool,
    output_fp32: bool = True,
) -> torch.Tensor:
    """Apply a complete linear QVQ layer through one folded-cache GEMM."""

    bits = normalize_qvq_rate(bits)
    if bits not in _P32_RATES:
        raise ValueError("AMD folded P32 supports rates W2, W2.5, W3, and W3.5")
    out_features = _integer_argument("out_features", out_features)
    bank_alt_id = _integer_argument("bank_alt_id", bank_alt_id)
    if not isinstance(input_hadamard, bool) or not isinstance(output_hadamard, bool):
        raise TypeError("AMD folded P32 transform-axis flags must be boolean")
    if not isinstance(output_fp32, bool):
        raise TypeError("output_fp32 must be boolean")

    # The first invocation performs the full public-API validation below.
    # A mutation-aware cache hit can avoid device-property queries, shape
    # reconstruction, and a second cache-key tuple/zip on every token.
    cached = getattr(window, "_qvq_p32_amd_folded_cache", None)
    if (
        isinstance(cached, tuple)
        and len(cached) == 6
        and x.ndim == 2
        and x.dtype == torch.float16
        and x.is_contiguous()
    ):
        cached_key, _, operand, _, residual_operand, composite_recovery = cached
        if (
            isinstance(cached_key, tuple)
            and len(cached_key) == 15
            and x.device == window.device
            and cached_key[0] == window._version
            and cached_key[1] is levels
            and cached_key[2] == levels._version
            and cached_key[3] is bank_ids
            and cached_key[4] == bank_ids._version
            and cached_key[5] == bits
            and cached_key[6] == x.shape[1]
            and cached_key[7] == out_features
            and cached_key[8] == bank_alt_id
            and cached_key[9] is su
            and cached_key[10] == su._version
            and cached_key[11] is sv
            and cached_key[12] == sv._version
            and cached_key[13] == input_hadamard
            and cached_key[14] == output_hadamard
        ):
            window._qvq_p32_amd_dense_cache = None
            return _qvq_p32_folded_execute(
                x,
                operand,
                residual_operand,
                composite_recovery,
                out_features=out_features,
                output_fp32=output_fp32,
            )

    transition_bits = qvq_transition_bits(bits, vector_size=2)
    if not qvq_p32_amd_supported(x.device):
        raise RuntimeError("AMD folded P32 requires a ROCm gfx950 device")
    if x.ndim != 2 or window.ndim != 2:
        raise ValueError("AMD folded P32 expects 2D input and window tensors")
    if x.dtype != torch.float16:
        raise TypeError("AMD folded P32 currently requires float16 input")
    if window.dtype != torch.int32:
        raise TypeError("AMD folded P32 requires int32 continuous-window words")
    if levels.dtype != torch.float16 or tuple(levels.shape) != (256,):
        raise TypeError("AMD folded P32 requires the canonical 256-entry float16 PGC16 table")
    if bank_ids.dtype != torch.uint8 or bank_ids.ndim != 1:
        raise TypeError("AMD folded P32 requires packed uint8 binary bank selectors")
    if su.dtype != torch.float16 or tuple(su.shape) != (x.shape[1],):
        raise TypeError("AMD folded P32 requires one float16 SU value per input feature")
    if sv.dtype != torch.float16 or tuple(sv.shape) != (out_features,):
        raise TypeError("AMD folded P32 requires one float16 SV value per output feature")
    if any(tensor.device != x.device for tensor in (window, levels, bank_ids, su, sv)):
        raise ValueError("AMD folded P32 tensors must share one device")
    if any(not tensor.is_contiguous() for tensor in (x, window, levels, bank_ids, su, sv)):
        raise ValueError("AMD folded P32 tensors must be contiguous")
    if not 1 <= bank_alt_id <= 3:
        raise ValueError("AMD folded P32 alternative bank ID must be in [1, 3]")

    m, k = x.shape
    n = out_features
    if not m or k <= 0 or n <= 0 or k % 16 or n % 16:
        raise ValueError(
            f"AMD folded P32 requires positive M and positive K/N divisible by 16, got M={m}, K={k}, N={n}"
        )
    tile_count = (k // 16) * (n // 16)
    words_per_tile = qvq_words_per_tile(bits, vector_size=2)
    expected_window = (tile_count, words_per_tile)
    if tuple(window.shape) != expected_window:
        raise ValueError(f"AMD folded P32 window must have shape {expected_window}")
    if tuple(bank_ids.shape) != (tile_count,):
        raise ValueError(f"AMD folded P32 selectors must have shape {(tile_count,)}")

    operand, residual_operand, composite_recovery = _qvq_p32_folded_weight(
        window,
        levels,
        bank_ids,
        su,
        sv,
        bits=bits,
        size_k=k,
        size_n=n,
        transition_bits=transition_bits,
        words_per_tile=words_per_tile,
        bank_alt_id=bank_alt_id,
        input_hadamard=input_hadamard,
        output_hadamard=output_hadamard,
    )
    return _qvq_p32_folded_execute(
        x,
        operand,
        residual_operand,
        composite_recovery,
        out_features=out_features,
        output_fp32=output_fp32,
    )


__all__ = [
    "qvq_p32_amd",
    "qvq_p32_amd_folded",
    "qvq_p32_amd_folded_case_supported",
    "qvq_p32_amd_folded_prefers_fp32_output",
    "qvq_p32_amd_folded_shape_supported",
    "qvq_p32_amd_supported",
]
