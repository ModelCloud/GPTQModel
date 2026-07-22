# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from ...utils.env import env_flag
from ...utils.logger import setup_logger


LAYOUT_GPTQ = "gptq"
LAYOUT_AWQ = "awq"

_LAYOUT_GPTQ_ID = 0
_LAYOUT_AWQ_ID = 1
_BITS = 3
_TRILIN_GROUP_SIZE = 128
_PACK_BLOCK_VALUES = 32
_PACK_WORDS = 3
_ZERO = 1 << (_BITS - 1)
_UINT32_MASK = (1 << 32) - 1
_MARLIN_N_ALIGNMENT = 64
_MARLIN_GROUP_SIZES = frozenset({32, 64, 128})
_TRILIN_NATIVE_GROUP_SIZES = frozenset({16, 32, 64, 96, 128, 192, 256, 384, 512, 1024})
_TRILIN_LORA_ENV = "GPTQMODEL_TRILIN_LORA"
_TRILIN_LORA_SIZE = 4096
_TRILIN_LORA_RANKS = frozenset({32, 64, 128, 256})


log = setup_logger()


@dataclass(frozen=True)
class Triton3BitLaunchConfig:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int

    def __post_init__(self) -> None:
        for name, value in (
            ("block_m", self.block_m),
            ("block_n", self.block_n),
            ("block_k", self.block_k),
        ):
            if value < 16 or value > 128 or value & (value - 1):
                raise ValueError(f"{name} must be a power of two in [16, 128], got {value}")
        if self.num_warps not in {1, 2, 4, 8}:
            raise ValueError(f"num_warps must be one of 1, 2, 4, or 8, got {self.num_warps}")
        if self.num_stages not in {1, 2, 3, 4}:
            raise ValueError(f"num_stages must be in [1, 4], got {self.num_stages}")


@dataclass(frozen=True)
class Marlin3BitState:
    qweight: torch.Tensor
    scales: torch.Tensor
    workspace: torch.Tensor
    empty: torch.Tensor


def _normalize_axis(axis: int, ndim: int) -> int:
    normalized = axis + ndim if axis < 0 else axis
    if normalized < 0 or normalized >= ndim:
        raise IndexError(f"axis {axis} is out of range for a {ndim}-dimensional tensor")
    return normalized


def pack_3bit(values: torch.Tensor, *, axis: int) -> torch.Tensor:
    """Pack contiguous groups of 32 unsigned 3-bit values into three int32 words."""
    if values.ndim == 0:
        raise ValueError("pack_3bit expects at least one tensor dimension")
    axis = _normalize_axis(axis, values.ndim)
    count = values.shape[axis]
    if count % _PACK_BLOCK_VALUES != 0:
        raise ValueError(f"pack_3bit axis length must be divisible by 32, got {count}")
    if values.is_floating_point() or values.is_complex():
        raise TypeError(f"pack_3bit expects an integer tensor, got {values.dtype}")
    if values.numel() and (torch.any(values < 0) or torch.any(values > 7)):
        raise ValueError("pack_3bit values must be in the inclusive range [0, 7]")

    moved = values.movedim(axis, -1).contiguous().to(torch.int64)
    blocks = moved.reshape(*moved.shape[:-1], count // _PACK_BLOCK_VALUES, _PACK_BLOCK_VALUES)
    packed = torch.zeros(
        (*blocks.shape[:-1], _PACK_WORDS),
        dtype=torch.int64,
        device=values.device,
    )
    for index in range(_PACK_BLOCK_VALUES):
        bit = index * _BITS
        word = bit // 32
        shift = bit % 32
        value = blocks[..., index]
        packed[..., word] |= (value << shift) & _UINT32_MASK
        if shift > 32 - _BITS:
            packed[..., word + 1] |= value >> (32 - shift)

    packed = packed.reshape(*moved.shape[:-1], count // _PACK_BLOCK_VALUES * _PACK_WORDS)
    return packed.to(torch.int32).movedim(-1, axis).contiguous()


def unpack_3bit(packed: torch.Tensor, *, axis: int, count: int) -> torch.Tensor:
    """Unpack three-word continuous 3-bit blocks along ``axis``."""
    if packed.ndim == 0:
        raise ValueError("unpack_3bit expects at least one tensor dimension")
    axis = _normalize_axis(axis, packed.ndim)
    if packed.dtype != torch.int32:
        raise TypeError(f"unpack_3bit expects torch.int32 storage, got {packed.dtype}")
    if count <= 0 or count % _PACK_BLOCK_VALUES != 0:
        raise ValueError(f"unpack_3bit count must be a positive multiple of 32, got {count}")
    expected_words = count // _PACK_BLOCK_VALUES * _PACK_WORDS
    if packed.shape[axis] != expected_words:
        raise ValueError(
            f"unpack_3bit expected {expected_words} packed words on axis {axis}, got {packed.shape[axis]}"
        )

    moved = packed.movedim(axis, -1).contiguous()
    words = (moved.to(torch.int64) & _UINT32_MASK).reshape(
        *moved.shape[:-1],
        count // _PACK_BLOCK_VALUES,
        _PACK_WORDS,
    )
    values = torch.empty(
        (*words.shape[:-1], _PACK_BLOCK_VALUES),
        dtype=torch.int32,
        device=packed.device,
    )
    for index in range(_PACK_BLOCK_VALUES):
        bit = index * _BITS
        word = bit // 32
        shift = bit % 32
        value = words[..., word] >> shift
        if shift > 32 - _BITS:
            value |= words[..., word + 1] << (32 - shift)
        values[..., index] = (value & 0x7).to(torch.int32)

    values = values.reshape(*moved.shape[:-1], count)
    return values.movedim(-1, axis).contiguous()


def _normalize_group_size(group_size: int, k: int) -> int:
    if group_size == -1:
        return k
    if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size <= 0:
        raise ValueError(f"3-bit group_size must be -1 or a positive integer, got {group_size!r}")
    if k % group_size != 0:
        raise ValueError(f"3-bit group_size={group_size} must divide K={k}")
    return group_size


def dequantize_3bit(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    layout: str,
    group_size: int = _TRILIN_GROUP_SIZE,
) -> torch.Tensor:
    """Portable Torch fallback/reference for symmetric grouped 3-bit weights."""
    if layout == LAYOUT_GPTQ:
        k = qweight.shape[0] // _PACK_WORDS * _PACK_BLOCK_VALUES
        codes = unpack_3bit(qweight, axis=0, count=k)
    elif layout == LAYOUT_AWQ:
        k = qweight.shape[0]
        n = qweight.shape[1] // _PACK_WORDS * _PACK_BLOCK_VALUES
        codes = unpack_3bit(qweight, axis=1, count=n)
    else:
        raise ValueError(f"unknown 3-bit layout `{layout}`; expected `{LAYOUT_GPTQ}` or `{LAYOUT_AWQ}`")

    group_size = _normalize_group_size(group_size, k)
    n = codes.shape[1]
    if tuple(scales.shape) != (k // group_size, n):
        raise ValueError(
            "3-bit scales must have shape "
            f"({k // group_size}, {n}), got {tuple(scales.shape)}"
        )
    expanded_scales = scales.repeat_interleave(group_size, dim=0)
    return (codes.to(scales.dtype) - _ZERO) * expanded_scales


def repack_awq_to_gptq_3bit(qweight: torch.Tensor) -> torch.Tensor:
    """Convert continuous N-packed AWQ storage to the coalesced K-packed runtime layout."""
    if qweight.ndim != 2 or qweight.dtype != torch.int32:
        raise TypeError(f"3-bit AWQ repack expects a 2D int32 tensor, got {qweight.dtype}")
    k = qweight.shape[0]
    if k <= 0 or k % _PACK_BLOCK_VALUES != 0:
        raise ValueError(f"3-bit AWQ repack requires positive K divisible by 32, got {k}")
    if qweight.shape[1] <= 0 or qweight.shape[1] % _PACK_WORDS != 0:
        raise ValueError(
            "3-bit AWQ repack requires a positive packed-N dimension divisible by 3, "
            f"got {qweight.shape[1]}"
        )
    n = qweight.shape[1] // _PACK_WORDS * _PACK_BLOCK_VALUES
    codes = unpack_3bit(qweight, axis=1, count=n)
    return pack_3bit(codes, axis=0)


@triton.jit
def _expand_gptq_3bit_to_uint4b8_kernel(
    qweight_ptr,
    output_ptr,
    packed_rows,
    n_size,
    stride_qwk,
    stride_qwn,
    stride_ok,
    stride_on,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < packed_rows * n_size
    output_k = offsets // n_size
    offsets_n = offsets % n_size
    base_k = output_k * 8
    packed = tl.zeros((BLOCK_SIZE,), dtype=tl.uint32)

    for index in tl.static_range(0, 8):
        offsets_k = base_k + index
        bit = (offsets_k % 32) * 3
        word = bit // 32
        shift = bit % 32
        packed_k = (offsets_k // 32) * 3 + word
        low_word = tl.load(
            qweight_ptr + packed_k * stride_qwk + offsets_n * stride_qwn,
            mask=mask,
            other=0,
        ).to(tl.uint32)
        high_word = tl.load(
            qweight_ptr + (packed_k + 1) * stride_qwk + offsets_n * stride_qwn,
            mask=mask & (shift > 29),
            other=0,
        ).to(tl.uint32)
        high_shift = (32 - shift) & 31
        code = ((low_word >> shift) | (high_word << high_shift)) & 0x7
        packed |= (code + 4) << (index * 4)

    tl.store(
        output_ptr + output_k * stride_ok + offsets_n * stride_on,
        packed,
        mask=mask,
    )


def expand_gptq_3bit_to_uint4b8(qweight: torch.Tensor) -> torch.Tensor:
    """Expand K-packed 3-bit codes into exact signed-Marlin 4-bit nibbles."""
    if qweight.ndim != 2 or qweight.dtype != torch.int32:
        raise TypeError(f"3-bit Marlin expansion expects a 2D int32 tensor, got {qweight.dtype}")
    if qweight.device.type != "cuda":
        raise ValueError(f"3-bit Marlin expansion requires a CUDA tensor, got {qweight.device}")
    if qweight.shape[0] <= 0 or qweight.shape[0] % _PACK_WORDS != 0:
        raise ValueError(
            "3-bit Marlin expansion requires a positive packed-K dimension divisible by 3, "
            f"got {qweight.shape[0]}"
        )
    if qweight.shape[1] <= 0:
        raise ValueError(f"3-bit Marlin expansion requires positive N, got {qweight.shape[1]}")
    if qweight.stride(1) != 1:
        raise ValueError("3-bit Marlin expansion requires contiguous qweight rows")

    k = qweight.shape[0] // _PACK_WORDS * _PACK_BLOCK_VALUES
    n = qweight.shape[1]
    output = torch.empty((k // 8, n), dtype=torch.int32, device=qweight.device)
    block_size = 256
    grid = (triton.cdiv(output.numel(), block_size),)
    with torch.cuda.device(qweight.device):
        _expand_gptq_3bit_to_uint4b8_kernel[grid](
            qweight,
            output,
            output.shape[0],
            n,
            qweight.stride(0),
            qweight.stride(1),
            output.stride(0),
            output.stride(1),
            BLOCK_SIZE=block_size,
            num_warps=4,
            num_stages=1,
        )
    return output


def marlin_3bit_eligible(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = _TRILIN_GROUP_SIZE,
) -> bool:
    """Return whether tensors satisfy the guarded FP16 native-cache contract."""
    if not env_flag("GPTQMODEL_TRILIN_NATIVE", default=True):
        return False
    if qweight.device.type != "cuda" or scales.device != qweight.device:
        return False
    if qweight.ndim != 2 or scales.ndim != 2 or qweight.dtype != torch.int32:
        return False
    if scales.dtype != torch.float16 or qweight.stride(1) != 1 or scales.stride(1) != 1:
        return False
    if qweight.shape[0] <= 0 or qweight.shape[0] % _PACK_WORDS != 0:
        return False
    k = qweight.shape[0] // _PACK_WORDS * _PACK_BLOCK_VALUES
    n = qweight.shape[1]
    try:
        group_size = _normalize_group_size(group_size, k)
    except ValueError:
        return False
    if group_size not in _MARLIN_GROUP_SIZES and group_size != k:
        return False
    if n % _MARLIN_N_ALIGNMENT != 0:
        return False
    if tuple(scales.shape) != (k // group_size, n):
        return False
    return torch.cuda.get_device_capability(qweight.device) >= (8, 0)


def _trilin_native_3bit_eligible(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> bool:
    """Return whether tensors satisfy the continuous-3-bit native CUDA contract."""
    if group_size not in _TRILIN_NATIVE_GROUP_SIZES or not env_flag("GPTQMODEL_TRILIN_NATIVE", default=True):
        return False
    if qweight.device.type != "cuda" or scales.device != qweight.device:
        return False
    if qweight.ndim != 2 or scales.ndim != 2 or qweight.dtype != torch.int32:
        return False
    if scales.dtype != torch.float16 or qweight.stride(1) != 1 or scales.stride(1) != 1:
        return False
    if qweight.shape[0] <= 0 or qweight.shape[0] % _PACK_WORDS != 0:
        return False
    k = qweight.shape[0] // _PACK_WORDS * _PACK_BLOCK_VALUES
    n = qweight.shape[1]
    if k % group_size != 0 or n % _MARLIN_N_ALIGNMENT != 0:
        return False
    if tuple(scales.shape) != (k // group_size, n):
        return False
    return torch.cuda.get_device_capability(qweight.device) >= (8, 0)


def prepare_trilin_3bit(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = _TRILIN_GROUP_SIZE,
) -> bool:
    """Load the true continuous-3-bit CUDA path when the layer is eligible."""
    if group_size not in {-1, _TRILIN_GROUP_SIZE}:
        log.warn.once(
            "Kernel: Trilin 3-bit received a non-primary group size "
            f"({group_size}); group_size=128 remains the fully optimized contract and latency may be higher."
        )
    if not _trilin_native_3bit_eligible(qweight, scales, group_size):
        return False
    k = qweight.shape[0] // _PACK_WORDS * _PACK_BLOCK_VALUES
    if k % _TRILIN_GROUP_SIZE != 0:
        return False

    try:
        from ...utils.trilin import trilin_runtime_available, trilin_runtime_error

        if not trilin_runtime_available():
            log.warn.once(
                "Trilin native 3-bit CUDA path is unavailable; using Marlin/Triton fallbacks: "
                f"{trilin_runtime_error()}"
            )
            return False
        log.info.once(
            "Kernel: Trilin true continuous-3-bit FP16/BF16 path enabled for positive group sizes >=16 "
            "(GEMV M=1, WMMA M=2..16)."
        )
        return True
    except Exception as exc:
        log.warn.once(
            "Trilin native 3-bit CUDA initialization failed; using Marlin/Triton fallbacks: "
            f"{exc}"
        )
        return False


def matmul_trilin_3bit(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    group_size: int = _TRILIN_GROUP_SIZE,
) -> torch.Tensor:
    """Run the original continuous 3-bit tensor through native Trilin CUDA."""
    from ...utils.trilin import trilin_matmul

    return trilin_matmul(input, qweight, scales, bias, group_size)


def prepare_trilin_lora_3bit(
    adapter,
    *,
    device: torch.device,
    in_features: int,
    out_features: int,
    group_size: int,
) -> torch.Tensor | None:
    """Allocate a disjoint payload for an exact supported Ampere decode specialization."""
    if not env_flag(_TRILIN_LORA_ENV, default=True):
        return None
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (8, 0):
        return None
    if (
        group_size != _TRILIN_GROUP_SIZE
        or in_features != _TRILIN_LORA_SIZE
        or out_features != _TRILIN_LORA_SIZE
    ):
        return None
    has_compressed_lora = getattr(adapter, "_has_compressed_lora", None)
    if callable(has_compressed_lora) and has_compressed_lora():
        return None
    lora_a = getattr(adapter, "lora_A", None)
    lora_b = getattr(adapter, "lora_B", None)
    if not callable(getattr(adapter, "_forward_lora_tensors", None)):
        return None
    if not isinstance(lora_a, torch.Tensor) or not isinstance(lora_b, torch.Tensor):
        return None
    if lora_a.dim() != 2 or lora_b.dim() != 2:
        return None
    rank = lora_a.shape[1]
    if rank not in _TRILIN_LORA_RANKS:
        return None
    if tuple(lora_a.shape) != (_TRILIN_LORA_SIZE, rank):
        return None
    if tuple(lora_b.shape) != (rank, _TRILIN_LORA_SIZE):
        return None
    if not lora_a.is_contiguous():
        adapter.lora_A = lora_a.contiguous()
    if not lora_b.is_contiguous():
        adapter.lora_B = lora_b.contiguous()
    log.info.once("Kernel: Ampere cooperative TriLin+LoRA ranks 32/64/128/256 decode inference is enabled.")
    workspace = torch.empty((rank,), dtype=torch.float32, device=device)
    stream = torch.cuda.current_stream(device)
    adapter._trilin_lora_stream_workspaces = {
        (workspace.get_device(), stream.cuda_stream): workspace,
    }
    return workspace


def _trilin_lora_workspace_for_current_stream(
    adapter,
    input: torch.Tensor,
    fallback: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """Return scratch exclusively owned by this adapter and CUDA stream."""
    stream = torch.cuda.current_stream(input.device)
    key = (input.get_device(), stream.cuda_stream)
    workspaces = getattr(adapter, "_trilin_lora_stream_workspaces", None)
    if not isinstance(workspaces, dict):
        workspaces = {}
        adapter._trilin_lora_stream_workspaces = workspaces

    def valid(candidate) -> bool:
        return (
            isinstance(candidate, torch.Tensor)
            and candidate.device == input.device
            and candidate.dtype == torch.float32
            and candidate.is_contiguous()
            and candidate.numel() >= rank
        )

    workspace = workspaces.get(key)
    if valid(workspace):
        return workspace
    if valid(fallback) and not any(candidate is fallback for candidate in workspaces.values()):
        workspace = fallback
    else:
        workspace = torch.empty((rank,), dtype=torch.float32, device=input.device)
    workspaces[key] = workspace
    return workspace


def matmul_trilin_lora_3bit(
    adapter,
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    workspace: torch.Tensor | None,
    *,
    bias: torch.Tensor | None = None,
    group_size: int = _TRILIN_GROUP_SIZE,
) -> torch.Tensor | None:
    """Try the one-launch TriLin+LoRA specialization and return ``None`` for the established fallback."""
    if not env_flag(_TRILIN_LORA_ENV, default=True) or workspace is None or group_size != _TRILIN_GROUP_SIZE:
        return None
    if input.device.type != "cuda" or input.dtype not in (torch.float16, torch.bfloat16):
        return None
    if tuple(input.shape) != (1, _TRILIN_LORA_SIZE) or not input.is_contiguous():
        return None
    if torch.cuda.get_device_capability(input.device) != (8, 0) or torch.cuda.is_current_stream_capturing():
        return None
    has_compressed_lora = getattr(adapter, "_has_compressed_lora", None)
    if callable(has_compressed_lora) and has_compressed_lora():
        return None

    forward_lora_tensors = getattr(adapter, "_forward_lora_tensors", None)
    if not callable(forward_lora_tensors):
        return None
    lora_a, lora_b = forward_lora_tensors(input)
    if not isinstance(lora_a, torch.Tensor) or not isinstance(lora_b, torch.Tensor):
        return None
    if lora_a.dim() != 2 or lora_b.dim() != 2:
        return None
    rank = lora_a.shape[1]
    if rank not in _TRILIN_LORA_RANKS:
        return None
    if tuple(lora_a.shape) != (_TRILIN_LORA_SIZE, rank):
        return None
    if tuple(lora_b.shape) != (rank, _TRILIN_LORA_SIZE):
        return None
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (input, lora_a, lora_b)):
        return None
    if not lora_a.is_contiguous():
        adapter.lora_A = lora_a = lora_a.contiguous()
    if not lora_b.is_contiguous():
        adapter.lora_B = lora_b = lora_b.contiguous()
    if (
        lora_a.dtype != input.dtype
        or lora_b.dtype != input.dtype
        or lora_a.device != input.device
        or lora_b.device != input.device
    ):
        return None
    workspace = _trilin_lora_workspace_for_current_stream(adapter, input, workspace, rank)

    try:
        from ...utils.trilin import trilin_matmul_lora

        return trilin_matmul_lora(input, qweight, scales, lora_a, lora_b, workspace, bias)
    except Exception as exc:
        log.warn.once(f"Integrated TriLin+LoRA inference failed; using the standard adapter path: {exc}")
        return None


def prepare_marlin_3bit(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = _TRILIN_GROUP_SIZE,
) -> Marlin3BitState | None:
    """Build a non-serialized native Marlin cache for an eligible 3-bit layer."""
    if not marlin_3bit_eligible(qweight, scales, group_size):
        return None

    try:
        from ...utils.marlin import (
            gptq_marlin_repack,
            marlin_make_workspace_new,
            marlin_permute_scales,
            marlin_runtime_available,
            marlin_runtime_error,
        )

        if not marlin_runtime_available(torch.float16):
            log.warn.once(
                "Trilin expanded-Marlin FP16 cache is unavailable; using the fused Triton 3-bit fallback: "
                f"{marlin_runtime_error(torch.float16)}"
            )
            return None

        empty = torch.empty(0, dtype=torch.int32, device=qweight.device)
        k = qweight.shape[0] // _PACK_WORDS * _PACK_BLOCK_VALUES
        n = qweight.shape[1]
        group_size = _normalize_group_size(group_size, k)
        expanded_qweight = expand_gptq_3bit_to_uint4b8(qweight)
        marlin_qweight = gptq_marlin_repack(
            expanded_qweight,
            empty,
            size_k=k,
            size_n=n,
            num_bits=4,
            dtype=torch.float16,
        )
        marlin_scales = marlin_permute_scales(
            scales.contiguous(),
            size_k=k,
            size_n=n,
            group_size=group_size,
        )
        workspace = marlin_make_workspace_new(qweight.device)
        log.info.once("Kernel: Trilin expanded-Marlin FP16 fallback enabled from exact signed 3-bit values.")
        return Marlin3BitState(
            qweight=marlin_qweight,
            scales=marlin_scales,
            workspace=workspace,
            empty=empty,
        )
    except Exception as exc:
        log.warn.once(
            "Trilin expanded-Marlin FP16 cache initialization failed; using the fused Triton 3-bit fallback: "
            f"{exc}"
        )
        return None


def matmul_marlin_3bit(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    workspace: torch.Tensor,
    empty: torch.Tensor,
    *,
    k: int,
    n: int,
    bias: torch.Tensor | None = None,
    use_fp32_reduce: bool = True,
    use_packed_prefill: bool = False,
    packed_prefill_config: int = 0,
) -> torch.Tensor:
    """Run the exact 3-bit values through the expanded native Marlin cache."""
    if input.ndim != 2 or input.dtype != torch.float16 or input.device.type != "cuda":
        raise ValueError(
            "3-bit Marlin matmul requires a 2D CUDA FP16 activation, "
            f"got shape={tuple(input.shape)}, dtype={input.dtype}, device={input.device}"
        )
    if input.shape[1] != k:
        raise ValueError(f"3-bit Marlin matmul expected activation K={k}, got {input.shape[1]}")
    if any(tensor.device != input.device for tensor in (qweight, scales, workspace, empty)):
        raise ValueError("3-bit Marlin activation and runtime-cache tensors must be on the same CUDA device")

    from ...utils.marlin import apply_gptq_marlin_linear
    from ...utils.marlin_scalar_type import scalar_types

    return apply_gptq_marlin_linear(
        input=input,
        weight=qweight,
        weight_scale=scales,
        weight_zp=empty,
        g_idx=empty,
        g_idx_sort_indices=empty,
        workspace=workspace,
        wtype=scalar_types.uint4b8,
        output_size_per_partition=n,
        input_size_per_partition=k,
        is_k_full=True,
        bias=bias,
        use_fp32_reduce=use_fp32_reduce,
        use_packed_prefill=use_packed_prefill,
        packed_prefill_config=packed_prefill_config,
    )


@triton.jit
def _matmul_3bit_kernel(
    a_ptr,
    qweight_ptr,
    scales_ptr,
    output_ptr,
    m_size,
    n_size,
    k_size,
    stride_am,
    stride_ak,
    stride_qwk,
    stride_qwn,
    stride_sg,
    stride_sn,
    stride_om,
    stride_on,
    LAYOUT: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(n_size, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(k_size, BLOCK_K)):
        mask_k = offsets_k < k_size
        a = tl.load(
            a_ptr + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak,
            mask=(offsets_m[:, None] < m_size) & mask_k[None, :],
            other=0.0,
        )

        if LAYOUT == 0:
            bit = (offsets_k % 32) * 3
            word = bit // 32
            shift = bit % 32
            packed_k = (offsets_k // 32) * 3 + word
            packed_ptrs = qweight_ptr + packed_k[:, None] * stride_qwk + offsets_n[None, :] * stride_qwn
            packed_mask = mask_k[:, None] & (offsets_n[None, :] < n_size)
            low_word = tl.load(packed_ptrs, mask=packed_mask, other=0)
            high_word = tl.load(
                packed_ptrs + stride_qwk,
                mask=packed_mask & (shift[:, None] > 29),
                other=0,
            )
            high_shift = (32 - shift) & 31
            low_fragment = (low_word >> shift[:, None]) & 0x7
            low_fragment = tl.where(shift[:, None] == 30, low_fragment & 0x3, low_fragment)
            low_fragment = tl.where(shift[:, None] == 31, low_fragment & 0x1, low_fragment)
            codes = (low_fragment | (high_word << high_shift[:, None])) & 0x7
        else:
            bit = (offsets_n % 32) * 3
            word = bit // 32
            shift = bit % 32
            packed_n = (offsets_n // 32) * 3 + word
            packed_ptrs = qweight_ptr + offsets_k[:, None] * stride_qwk + packed_n[None, :] * stride_qwn
            packed_mask = mask_k[:, None] & (offsets_n[None, :] < n_size)
            low_word = tl.load(packed_ptrs, mask=packed_mask, other=0)
            high_word = tl.load(
                packed_ptrs + stride_qwn,
                mask=packed_mask & (shift[None, :] > 29),
                other=0,
            )
            high_shift = (32 - shift) & 31
            low_fragment = (low_word >> shift[None, :]) & 0x7
            low_fragment = tl.where(shift[None, :] == 30, low_fragment & 0x3, low_fragment)
            low_fragment = tl.where(shift[None, :] == 31, low_fragment & 0x1, low_fragment)
            codes = (low_fragment | (high_word << high_shift[None, :])) & 0x7

        groups = offsets_k // GROUP_SIZE
        scales = tl.load(
            scales_ptr + groups[:, None] * stride_sg + offsets_n[None, :] * stride_sn,
            mask=mask_k[:, None] & (offsets_n[None, :] < n_size),
            other=0.0,
        )
        dequantized = (codes.to(tl.float32) - 4) * scales.to(tl.float32)
        dequantized = dequantized.to(a.dtype)
        accumulator = tl.dot(a, dequantized, accumulator, out_dtype=tl.float32)
        offsets_k += BLOCK_K

    output_ptrs = output_ptr + offsets_m[:, None] * stride_om + offsets_n[None, :] * stride_on
    output_mask = (offsets_m[:, None] < m_size) & (offsets_n[None, :] < n_size)
    tl.store(output_ptrs, accumulator, mask=output_mask)


def _select_launch_config(m: int) -> Triton3BitLaunchConfig:
    if m <= 4:
        return Triton3BitLaunchConfig(block_m=16, block_n=32, block_k=32, num_warps=4, num_stages=1)
    if m <= 16:
        return Triton3BitLaunchConfig(block_m=16, block_n=64, block_k=32, num_warps=4, num_stages=1)
    if m <= 64:
        return Triton3BitLaunchConfig(block_m=32, block_n=64, block_k=32, num_warps=4, num_stages=1)
    return Triton3BitLaunchConfig(block_m=64, block_n=32, block_k=32, num_warps=8, num_stages=1)


def _validate_matmul_inputs(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    layout: str,
    group_size: int,
) -> tuple[int, int, int, int, int]:
    if layout not in {LAYOUT_GPTQ, LAYOUT_AWQ}:
        raise ValueError(f"unknown 3-bit layout `{layout}`; expected `{LAYOUT_GPTQ}` or `{LAYOUT_AWQ}`")
    if input.ndim != 2:
        raise ValueError(f"3-bit Triton matmul expects a 2D activation, got shape {tuple(input.shape)}")
    if input.device.type != "cuda":
        raise ValueError(f"3-bit Triton matmul requires CUDA activations, got {input.device}")
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"3-bit Triton matmul requires FP16 or BF16 activations, got {input.dtype}")
    if input.stride(1) != 1:
        raise ValueError("3-bit Triton matmul requires a contiguous activation K dimension")
    if qweight.dtype != torch.int32:
        raise TypeError(f"3-bit Triton matmul requires int32 qweight, got {qweight.dtype}")
    if scales.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"3-bit Triton matmul requires FP16 or BF16 scales, got {scales.dtype}")
    if qweight.device != input.device or scales.device != input.device:
        raise ValueError("activation, qweight, and scales must be on the same CUDA device")
    if qweight.ndim != 2 or scales.ndim != 2:
        raise ValueError("3-bit Triton matmul requires 2D qweight and scales tensors")
    if qweight.stride(1) != 1 or scales.stride(1) != 1:
        raise ValueError("3-bit Triton matmul requires contiguous qweight/scales last dimensions")

    m, k = input.shape
    groups, n = scales.shape
    if k <= 0 or k % _PACK_BLOCK_VALUES != 0:
        raise ValueError(f"3-bit Triton matmul requires positive K divisible by 32, got {k}")
    group_size = _normalize_group_size(group_size, k)
    if n <= 0 or n % _PACK_BLOCK_VALUES != 0:
        raise ValueError(f"3-bit Triton matmul requires positive N divisible by 32, got {n}")
    if groups != k // group_size:
        raise ValueError(f"3-bit scales expected {k // group_size} groups, got {groups}")

    expected_qweight = (
        (k // _PACK_BLOCK_VALUES * _PACK_WORDS, n)
        if layout == LAYOUT_GPTQ
        else (k, n // _PACK_BLOCK_VALUES * _PACK_WORDS)
    )
    if tuple(qweight.shape) != expected_qweight:
        raise ValueError(
            f"3-bit {layout} qweight expected shape {expected_qweight}, got {tuple(qweight.shape)}"
        )
    capability = torch.cuda.get_device_capability(input.device)
    if capability < (8, 0):
        raise RuntimeError(
            "fused 3-bit Triton matmul requires compute capability >= 8.0; "
            f"device {input.device} reports {capability[0]}.{capability[1]}"
        )
    return m, n, k, _LAYOUT_GPTQ_ID if layout == LAYOUT_GPTQ else _LAYOUT_AWQ_ID, group_size


def matmul_3bit(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    layout: str,
    group_size: int = _TRILIN_GROUP_SIZE,
    launch_config: Triton3BitLaunchConfig | None = None,
) -> torch.Tensor:
    """Fused symmetric grouped 3-bit matmul for GPTQ or AWQ continuous packing."""
    m, n, k, layout_id, group_size = _validate_matmul_inputs(input, qweight, scales, layout, group_size)
    output = torch.empty((m, n), dtype=input.dtype, device=input.device)
    if m == 0:
        return output

    config = launch_config or _select_launch_config(m)
    grid = (
        triton.cdiv(m, config.block_m) * triton.cdiv(n, config.block_n),
    )
    with torch.cuda.device(qweight.device):
        _matmul_3bit_kernel[grid](
            input,
            qweight,
            scales,
            output,
            m,
            n,
            k,
            input.stride(0),
            input.stride(1),
            qweight.stride(0),
            qweight.stride(1),
            scales.stride(0),
            scales.stride(1),
            output.stride(0),
            output.stride(1),
            LAYOUT=layout_id,
            GROUP_SIZE=group_size,
            BLOCK_M=config.block_m,
            BLOCK_N=config.block_n,
            BLOCK_K=config.block_k,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
    return output


__all__ = [
    "LAYOUT_AWQ",
    "LAYOUT_GPTQ",
    "Marlin3BitState",
    "Triton3BitLaunchConfig",
    "dequantize_3bit",
    "expand_gptq_3bit_to_uint4b8",
    "marlin_3bit_eligible",
    "matmul_3bit",
    "matmul_marlin_3bit",
    "matmul_trilin_3bit",
    "pack_3bit",
    "prepare_marlin_3bit",
    "prepare_trilin_3bit",
    "repack_awq_to_gptq_3bit",
    "unpack_3bit",
]
