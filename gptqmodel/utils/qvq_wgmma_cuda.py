# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Hopper-only CuTe RS-WGMMA kernels for exact standard-P32 payloads."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from ..quantization.qvq_rates import qvq_transition_bits
from .cpp import (
    TorchOpsJitExtension,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
    is_nvcc_compatible,
)
from .machete import _ensure_cutlass_source

_QVQ_WGMMA_NAME = "gptqmodel_qvq_wgmma_ops"
_QVQ_WGMMA_NAMESPACE = "gptqmodel_qvq_wgmma"
_SM90A_FLAGS = (
    "-gencode=arch=compute_90a,code=sm_90a",
    "-gencode=arch=compute_90a,code=compute_90a",
)
_TORCH_NVCC_UNDEFINES = (
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
)
_P32_TRANSITION_BITS = {2: 4, 2.5: 5, 3: 6, 3.5: 7}
_MAX_GROUPED_P32_SEGMENTS = 3


@dataclass(frozen=True)
class QVQHopperP32SegmentPlan:
    """One child projection inside a grouped Hopper P32 launch."""

    output_tile_start: int
    output_tile_count: int
    out_features: int
    bank_alt_id: int
    split_count: int


@dataclass(frozen=True)
class QVQHopperGroupedP32Plan:
    """A segmented Hopper plan retaining every child's native policy."""

    in_features: int
    transition_bits: int
    segments: tuple[QVQHopperP32SegmentPlan, ...]

    @property
    def out_features(self) -> int:
        return sum(segment.out_features for segment in self.segments)


@dataclass(frozen=True)
class QVQHopperGroupedP32Payload:
    """Cached K-major continuous-window payload for grouped Hopper execution."""

    trellis: torch.Tensor
    bank_ids: torch.Tensor
    plan: QVQHopperGroupedP32Plan


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source() -> list[str]:
    return [str(_project_root() / "gptqmodel_ext" / "qvq" / "qvq_wgmma_cuda.cu")]


def _include_paths() -> list[str]:
    cutlass_root = _ensure_cutlass_source()
    return [str((cutlass_root / "include").resolve())]


def _cuda_flags() -> list[str]:
    flags = [
        *_TORCH_NVCC_UNDEFINES,
        *default_jit_cuda_cflags(
            enable_bf16=True,
            include_lineinfo=True,
            include_nvcc_threads=True,
            nvcc_threads="2",
            include_split_compile=True,
            include_ptxas_optimizations=True,
            include_ptxas_verbosity=False,
            include_fatbin_compression=True,
            include_diag_suppress=True,
        ),
        *_SM90A_FLAGS,
    ]
    if is_nvcc_compatible():
        flags.insert(0, "-static-global-template-stub=false")
    return flags


_QVQ_WGMMA_EXTENSION = TorchOpsJitExtension(
    name=_QVQ_WGMMA_NAME,
    namespace=_QVQ_WGMMA_NAMESPACE,
    required_ops=(
        "fp16_to_fp8_e5m2_clamped",
        "p32_window_w3_m16",
        "p32_window_w3_m16_tma",
        "p32_window_m16_tma",
        "p32_window_m16_tma_ordered_split",
        "p32_window_m16_tma_ordered_partials",
        "p32_window_m16_tma_grouped",
        "p32_window_m16_tma_grouped_ordered_split",
        "p32_window_m16_tma_grouped_ordered_partials",
        "p32_window_m32_tma_grouped_reuse2",
        "p32_window_m32_tma_grouped_ordered_reuse2",
        "p32_window_m64_tma_grouped_reuse4",
        "p32_window_m64_tma_grouped_ordered_reuse4",
        "p32_window_decode_grouped_fp16",
        "p32_window_prepare_grouped_fp8",
        "p32_window_prepare_grouped_fp8_half_fold",
        "p32_window_prepare_grouped_fp16",
    ),
    sources=_source,
    build_root_env="GPTQMODEL_QVQ_WGMMA_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("qvq_wgmma"),
    display_name="QVQ exact-P32 Hopper RS-WGMMA kernels",
    extra_cflags=lambda: default_jit_cflags(enable_bf16=True),
    extra_cuda_cflags=_cuda_flags,
    extra_include_paths=_include_paths,
    extra_ldflags=("-lcuda",),
    force_rebuild_env="GPTQMODEL_QVQ_WGMMA_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
    merge_visible_cuda_arch_override=False,
)


def qvq_fp16_to_fp8_e5m2_clamped(input: torch.Tensor) -> torch.Tensor:
    """Convert finite FP16 values to E5M2 without overflowing to infinity."""

    if input.device.type != "cuda" or input.dtype != torch.float16:
        raise ValueError("Hopper FP8 prefill conversion requires FP16 CUDA input")
    return _QVQ_WGMMA_EXTENSION.op("fp16_to_fp8_e5m2_clamped")(
        input.contiguous()
    )


def _resolve_transition_bits(bits: float) -> int:
    try:
        transition_bits = None if isinstance(bits, bool) else _P32_TRANSITION_BITS[bits]
    except (KeyError, TypeError):
        transition_bits = None
    if transition_bits is None:
        transition_bits = qvq_transition_bits(bits, vector_size=2)
    if transition_bits not in (4, 5, 6, 7):
        raise ValueError("QVQ P32 TMA WGMMA supports W2 through W3.5")
    return transition_bits


def _resolve_hopper_split_count(
    *, input: torch.Tensor, transition_bits: int, out_features: int, split_count: int
) -> int:
    if split_count < 0:
        raise ValueError("QVQ P32 TMA WGMMA split_count must be non-negative")
    if split_count:
        return int(split_count)
    shape = (int(input.shape[1]), int(out_features))
    return {
        4: {
            (5120, 1024): 20,
            (5120, 6144): 20,
            (5120, 10240): 10,
            (5120, 12288): 10,
            (5120, 17408): 10,
            (6144, 5120): 8,
            (17408, 5120): 34,
        },
        5: {
            (5120, 1024): 20,
            (5120, 6144): 20,
            (5120, 10240): 10,
            (5120, 12288): 10,
            (5120, 17408): 10,
            (6144, 5120): 8,
            (17408, 5120): 34,
        },
        6: {
            (5120, 1024): 20,
            (5120, 6144): 20,
            (5120, 10240): 4,
            (5120, 12288): 10,
            (5120, 17408): 10,
            (6144, 5120): 8,
            (17408, 5120): 34,
        },
        7: {
            (5120, 1024): 20,
            (5120, 6144): 4,
            (5120, 10240): 4,
            (5120, 12288): 4,
            (5120, 17408): 5,
            (6144, 5120): 8,
            (17408, 5120): 34,
        },
    }[transition_bits].get(shape, 1)


def qvq_h100_ordered_split_count(
    *,
    device_name: str,
    compute_capability: tuple[int, int],
    logical_rows: int,
    in_features: int,
    out_features: int,
    transition_bits: int,
) -> int:
    """Return a measured H100 ordered split, or zero for the ordinary path."""

    if (
        "H100" in device_name
        and compute_capability == (9, 0)
        and logical_rows in (1, 2, 4, 8, 16)
        and (in_features, out_features) == (8192, 2048)
        and transition_bits in (4, 5, 6, 7)
    ):
        return 16
    return 0


def qvq_h100_large_m_ordered_split_count(
    *,
    device_name: str,
    compute_capability: tuple[int, int],
    logical_rows: int,
    in_features: int,
    out_features: int,
) -> int:
    """Return the measured H100 large-M split for narrow Llama down."""

    if (
        device_name != "NVIDIA H100"
        or compute_capability != (9, 0)
        or (in_features, out_features) != (8192, 2048)
        or logical_rows <= 16
    ):
        return 1
    if logical_rows <= 64:
        return 8
    if logical_rows <= 128:
        return 4
    if logical_rows <= 256:
        return 2
    return 1


def qvq_h100_grouped_ordered_split_counts(
    *,
    device_name: str,
    compute_capability: tuple[int, int],
    in_features: int,
    out_features: Sequence[int],
    transition_bits: int,
) -> tuple[int, ...] | None:
    """Return a measured H100 grouped split policy, or no override.

    Keep this helper geometry-only: architecture code decides which children
    form a legal group, while the Hopper backend owns the measured execution
    schedule for that geometry.  The Llama 3.2 1B query/key/value sweep found
    split 8 for every child to be the winner at M1, M2, M4, M8, and M16 for
    every supported P32 rate.
    """

    widths = tuple(int(value) for value in out_features)
    if (
        "H100" in device_name
        and compute_capability == (9, 0)
        and int(in_features) == 2048
        and widths == (2048, 512, 512)
        and int(transition_bits) in (4, 5, 6, 7)
    ):
        return (8, 8, 8)
    if (
        "H100" in device_name
        and compute_capability == (9, 0)
        and int(in_features) == 5120
        and int(transition_bits) in (4, 5, 6, 7)
    ):
        # Official Qwen3.8-27B projection geometry.  Preserve the same
        # per-child split schedule selected by the independently benchmarked
        # kernels instead of choosing one split from concatenated N.
        qwen_splits = {
            (12288, 1024, 1024): {
                4: (10, 20, 20),
                5: (10, 20, 20),
                6: (10, 20, 20),
                7: (4, 20, 20),
            },
            (10240, 6144): {
                4: (10, 20),
                5: (10, 20),
                6: (4, 20),
                7: (4, 4),
            },
            (17408, 17408): {
                # The complete fused-MLP sweep found split five faster than
                # split ten at every M1--M16 row count.  Four K256 stages per
                # CTA amortize the grouped decoder setup better than the
                # previous two-stage split-ten schedule.
                4: (5, 5),
                5: (5, 5),
                6: (5, 5),
                7: (5, 5),
            },
        }
        return qwen_splits.get(widths, {}).get(int(transition_bits))
    return None


def qvq_p32_window_wgmma_w3_m16(
    input: torch.Tensor,
    trellis: torch.Tensor,
    levels: torch.Tensor,
    bank_ids: torch.Tensor,
    *,
    out_features: int,
    bank_alt_id: int = 3,
    split_count: int = 1,
) -> torch.Tensor:
    """Run the direct-window standard-P32 W3 RS-WGMMA prototype."""

    return _QVQ_WGMMA_EXTENSION.op("p32_window_w3_m16")(
        input,
        trellis,
        levels,
        bank_ids,
        out_features,
        bank_alt_id,
        split_count,
    )


def qvq_p32_window_wgmma_w3_m16_tma(
    input: torch.Tensor,
    trellis: torch.Tensor,
    levels: torch.Tensor,
    bank_ids: torch.Tensor,
    *,
    out_features: int,
    bank_alt_id: int = 3,
    split_count: int = 0,
) -> torch.Tensor:
    """Run the two-stage TMA direct-window P32 W3 RS-WGMMA prototype."""

    if split_count == 0:
        shape = (int(input.shape[1]), int(out_features))
        split_count = {
            (5120, 1024): 20,
            (5120, 6144): 20,
            (5120, 10240): 4,
            (5120, 12288): 10,
            (5120, 17408): 10,
            (6144, 5120): 8,
            (17408, 5120): 34,
        }.get(shape, 1)

    return _QVQ_WGMMA_EXTENSION.op("p32_window_w3_m16_tma")(
        input,
        trellis,
        levels,
        bank_ids,
        out_features,
        bank_alt_id,
        split_count,
    )


def qvq_p32_window_wgmma_m16_tma(
    input: torch.Tensor,
    trellis: torch.Tensor,
    levels: torch.Tensor,
    bank_ids: torch.Tensor,
    bits: float,
    *,
    out_features: int,
    bank_alt_id: int = 3,
    split_count: int = 0,
) -> torch.Tensor:
    """Run the two-stage TMA direct-window P32 RS-WGMMA kernel at W2-W3.5."""

    transition_bits = _resolve_transition_bits(bits)
    split_count = _resolve_hopper_split_count(
        input=input,
        transition_bits=transition_bits,
        out_features=int(out_features),
        split_count=int(split_count),
    )
    return _QVQ_WGMMA_EXTENSION.op("p32_window_m16_tma")(
        input,
        trellis,
        levels,
        bank_ids,
        transition_bits,
        out_features,
        bank_alt_id,
        split_count,
    )


def qvq_p32_window_wgmma_m16_tma_ordered_split(
    input: torch.Tensor,
    trellis: torch.Tensor,
    levels: torch.Tensor,
    bank_ids: torch.Tensor,
    bits: float,
    *,
    out_features: int,
    bank_alt_id: int = 3,
    split_count: int,
) -> torch.Tensor:
    """Run split-K through disjoint FP32 planes and an ordered reducer."""

    transition_bits = _resolve_transition_bits(bits)
    split_count = _resolve_hopper_split_count(
        input=input,
        transition_bits=transition_bits,
        out_features=int(out_features),
        split_count=int(split_count),
    )
    return _QVQ_WGMMA_EXTENSION.op("p32_window_m16_tma_ordered_split")(
        input,
        trellis,
        levels,
        bank_ids,
        transition_bits,
        out_features,
        bank_alt_id,
        split_count,
    )


def qvq_p32_window_wgmma_m16_tma_ordered_partials(
    input: torch.Tensor,
    trellis: torch.Tensor,
    levels: torch.Tensor,
    bank_ids: torch.Tensor,
    bits: float,
    *,
    out_features: int,
    bank_alt_id: int = 3,
    split_count: int,
) -> torch.Tensor:
    """Return child-local ordered FP32 split planes without reducing them."""

    transition_bits = _resolve_transition_bits(bits)
    split_count = _resolve_hopper_split_count(
        input=input,
        transition_bits=transition_bits,
        out_features=int(out_features),
        split_count=int(split_count),
    )
    if split_count <= 1:
        raise ValueError("ordered partial output requires split_count greater than one")
    return _QVQ_WGMMA_EXTENSION.op("p32_window_m16_tma_ordered_partials")(
        input,
        trellis,
        levels,
        bank_ids,
        transition_bits,
        out_features,
        bank_alt_id,
        split_count,
    )


def qvq_p32_window_wgmma_group_plan(
    input: torch.Tensor,
    trellises: Sequence[torch.Tensor],
    levels: torch.Tensor,
    bank_ids: Sequence[torch.Tensor],
    bits: float,
    *,
    out_features: Sequence[int],
    bank_alt_ids: Sequence[int],
    split_counts: Sequence[int] | None = None,
) -> QVQHopperGroupedP32Plan:
    """Resolve a Hopper launch plan without deriving policy from total N."""

    trellises = tuple(trellises)
    bank_ids = tuple(bank_ids)
    widths = tuple(int(value) for value in out_features)
    alt_ids = tuple(int(value) for value in bank_alt_ids)
    segment_count = len(trellises)
    if not 1 <= segment_count <= _MAX_GROUPED_P32_SEGMENTS:
        raise ValueError("grouped Hopper P32 requires between one and three segments")
    if not (
        len(bank_ids) == segment_count
        and len(widths) == segment_count
        and len(alt_ids) == segment_count
    ):
        raise ValueError("grouped Hopper P32 segment metadata lengths must match")
    if split_counts is None:
        requested_splits = (0,) * segment_count
    else:
        requested_splits = tuple(int(value) for value in split_counts)
        if len(requested_splits) != segment_count:
            raise ValueError("grouped Hopper P32 split_counts length must match")
    if (
        input.ndim != 2
        or input.shape[0] < 16
        or input.shape[0] > 4096
        or input.shape[0] % 16
    ):
        raise ValueError(
            "grouped Hopper P32 requires M in [16, 4096] and divisible by 16"
        )
    if input.shape[1] <= 0 or input.shape[1] % 256:
        raise ValueError(
            "grouped Hopper P32 input K must be a positive multiple of 256"
        )
    for width, alt_id in zip(widths, alt_ids, strict=True):
        if width <= 0 or width % 256:
            raise ValueError(
                "grouped Hopper P32 output widths must be positive multiples of 256"
            )
        if not 0 <= alt_id <= 3:
            raise ValueError("grouped Hopper P32 bank IDs must be in [0, 3]")

    transition_bits = _resolve_transition_bits(bits)
    resolved_splits = tuple(
        _resolve_hopper_split_count(
            input=input,
            transition_bits=transition_bits,
            out_features=width,
            split_count=requested_split,
        )
        for width, requested_split in zip(widths, requested_splits, strict=True)
    )
    k_tiles = int(input.shape[1]) // 16
    for split in resolved_splits:
        if not 1 <= split <= 64:
            raise ValueError("grouped Hopper P32 split counts must be in [1, 64]")
        if k_tiles % split or (k_tiles // split) % 16:
            raise ValueError(
                "grouped Hopper P32 split partitions must contain a multiple "
                "of sixteen K16 tiles"
            )

    output_tile_start = 0
    segments = []
    for width, alt_id, split in zip(widths, alt_ids, resolved_splits, strict=True):
        output_tile_count = width // 16
        segments.append(
            QVQHopperP32SegmentPlan(
                output_tile_start=output_tile_start,
                output_tile_count=output_tile_count,
                out_features=width,
                bank_alt_id=alt_id,
                split_count=split,
            )
        )
        output_tile_start += output_tile_count
    return QVQHopperGroupedP32Plan(
        in_features=int(input.shape[1]),
        transition_bits=transition_bits,
        segments=tuple(segments),
    )


def qvq_pack_p32_window_hopper_group(
    trellises: Sequence[torch.Tensor],
    bank_ids: Sequence[torch.Tensor],
    plan: QVQHopperGroupedP32Plan,
) -> QVQHopperGroupedP32Payload:
    """Losslessly concatenate child windows along the P32 N16-tile axis."""

    trellises = tuple(trellises)
    bank_ids = tuple(bank_ids)
    if len(trellises) != len(plan.segments) or len(bank_ids) != len(plan.segments):
        raise ValueError("grouped Hopper P32 payload lengths must match the plan")
    k_tiles = plan.in_features // 16
    words_per_tile = 4 * plan.transition_bits
    trellis_parts = []
    bank_parts = []
    for trellis, selectors, segment in zip(
        trellises, bank_ids, plan.segments, strict=True
    ):
        expected_tiles = k_tiles * segment.output_tile_count
        if (
            trellis.dtype != torch.int32
            or trellis.numel() != expected_tiles * words_per_tile
        ):
            raise ValueError(
                "grouped Hopper P32 trellis has the wrong dtype or word count"
            )
        if selectors.dtype != torch.uint8 or selectors.numel() != expected_tiles:
            raise ValueError(
                "grouped Hopper P32 bank ids have the wrong dtype or length"
            )
        if (
            trellis.device != trellises[0].device
            or selectors.device != trellises[0].device
        ):
            raise ValueError("grouped Hopper P32 payloads must share one device")
        trellis_parts.append(
            trellis.reshape(k_tiles, segment.output_tile_count, words_per_tile)
        )
        bank_parts.append(selectors.reshape(k_tiles, segment.output_tile_count))
    return QVQHopperGroupedP32Payload(
        trellis=torch.cat(trellis_parts, dim=1)
        .reshape(-1, words_per_tile)
        .contiguous(),
        bank_ids=torch.cat(bank_parts, dim=1).reshape(-1).contiguous(),
        plan=plan,
    )


def qvq_p32_window_wgmma_grouped_packed(
    input: torch.Tensor,
    payload: QVQHopperGroupedP32Payload,
    levels: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Run one segmented Hopper TMA/RS-WGMMA grid for a cached payload."""

    plan = payload.plan
    rows = int(input.shape[0]) if input.ndim == 2 else 0
    if (
        input.ndim != 2
        or input.shape[1] != plan.in_features
        or rows < 16
        or rows > 4096
        or rows % 16
    ):
        raise ValueError("grouped Hopper P32 input does not match its row-tiled plan")
    if any(segment.split_count != 1 for segment in plan.segments):
        raise ValueError(
            "grouped Hopper P32 fusion requires split-1 children for exact "
            "reduction order"
        )
    widths = [segment.out_features for segment in plan.segments]
    output = _QVQ_WGMMA_EXTENSION.op("p32_window_m16_tma_grouped")(
        input,
        payload.trellis,
        levels,
        payload.bank_ids,
        plan.transition_bits,
        widths,
        [segment.bank_alt_id for segment in plan.segments],
        [segment.split_count for segment in plan.segments],
    )
    return tuple(
        child.reshape(rows, width)
        for child, width in zip(
            torch.split(output, [rows * width for width in widths]),
            widths,
            strict=True,
        )
    )


def qvq_p32_window_wgmma_grouped_ordered_packed(
    input: torch.Tensor,
    payload: QVQHopperGroupedP32Payload,
    levels: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Run a flattened grouped grid with child-local ordered split reduction."""

    plan = payload.plan
    rows = int(input.shape[0]) if input.ndim == 2 else 0
    if (
        input.ndim != 2
        or input.shape[1] != plan.in_features
        or rows < 16
        or rows > 4096
        or rows % 16
    ):
        raise ValueError("grouped Hopper P32 input does not match its row-tiled plan")
    widths = [segment.out_features for segment in plan.segments]
    output = _QVQ_WGMMA_EXTENSION.op("p32_window_m16_tma_grouped_ordered_split")(
        input,
        payload.trellis,
        levels,
        payload.bank_ids,
        plan.transition_bits,
        widths,
        [segment.bank_alt_id for segment in plan.segments],
        [segment.split_count for segment in plan.segments],
    )
    return tuple(
        child.reshape(rows, width)
        for child, width in zip(
            torch.split(output, [rows * width for width in widths]),
            widths,
            strict=True,
        )
    )


def qvq_p32_window_wgmma_grouped_ordered_partials_packed(
    input: torch.Tensor,
    payload: QVQHopperGroupedP32Payload,
    levels: torch.Tensor,
) -> torch.Tensor:
    """Run the grouped grid and return child-major ordered partial planes."""

    plan = payload.plan
    rows = int(input.shape[0]) if input.ndim == 2 else 0
    if (
        input.ndim != 2
        or input.shape[1] != plan.in_features
        or rows < 16
        or rows > 4096
        or rows % 16
    ):
        raise ValueError("grouped Hopper P32 input does not match its row-tiled plan")
    if not any(segment.split_count > 1 for segment in plan.segments):
        raise ValueError("ordered grouped partials require at least one split child")
    return _QVQ_WGMMA_EXTENSION.op("p32_window_m16_tma_grouped_ordered_partials")(
        input,
        payload.trellis,
        levels,
        payload.bank_ids,
        plan.transition_bits,
        [segment.out_features for segment in plan.segments],
        [segment.bank_alt_id for segment in plan.segments],
        [segment.split_count for segment in plan.segments],
    )


def qvq_p32_window_wgmma_grouped_reuse2_packed(
    input: torch.Tensor,
    payload: QVQHopperGroupedP32Payload,
    levels: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Decode once for each pair of M16 row tiles in a grouped Hopper grid."""

    plan = payload.plan
    rows = int(input.shape[0]) if input.ndim == 2 else 0
    if (
        input.ndim != 2
        or input.shape[1] != plan.in_features
        or rows < 32
        or rows > 4096
        or rows % 32
    ):
        raise ValueError(
            "grouped Hopper P32 row-reuse input requires M in [32, 4096] "
            "and divisible by 32"
        )
    widths = [segment.out_features for segment in plan.segments]
    ordered = any(segment.split_count != 1 for segment in plan.segments)
    op_name = (
        "p32_window_m32_tma_grouped_ordered_reuse2"
        if ordered
        else "p32_window_m32_tma_grouped_reuse2"
    )
    output = _QVQ_WGMMA_EXTENSION.op(op_name)(
        input,
        payload.trellis,
        levels,
        payload.bank_ids,
        plan.transition_bits,
        widths,
        [segment.bank_alt_id for segment in plan.segments],
        [segment.split_count for segment in plan.segments],
    )
    return tuple(
        child.reshape(rows, width)
        for child, width in zip(
            torch.split(output, [rows * width for width in widths]),
            widths,
            strict=True,
        )
    )


def qvq_p32_window_wgmma_grouped_reuse4_packed(
    input: torch.Tensor,
    payload: QVQHopperGroupedP32Payload,
    levels: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Decode once for each group of four M16 row tiles on Hopper."""

    plan = payload.plan
    rows = int(input.shape[0]) if input.ndim == 2 else 0
    if (
        input.ndim != 2
        or input.shape[1] != plan.in_features
        or rows < 64
        or rows > 4096
        or rows % 64
    ):
        raise ValueError(
            "grouped Hopper P32 reuse-4 input requires M in [64, 4096] "
            "and divisible by 64"
        )
    widths = [segment.out_features for segment in plan.segments]
    ordered = any(segment.split_count != 1 for segment in plan.segments)
    op_name = (
        "p32_window_m64_tma_grouped_ordered_reuse4"
        if ordered
        else "p32_window_m64_tma_grouped_reuse4"
    )
    output = _QVQ_WGMMA_EXTENSION.op(op_name)(
        input,
        payload.trellis,
        levels,
        payload.bank_ids,
        plan.transition_bits,
        widths,
        [segment.bank_alt_id for segment in plan.segments],
        [segment.split_count for segment in plan.segments],
    )
    return tuple(
        child.reshape(rows, width)
        for child, width in zip(
            torch.split(output, [rows * width for width in widths]),
            widths,
            strict=True,
        )
    )


def qvq_p32_window_decode_grouped_fp16_packed(
    payload: QVQHopperGroupedP32Payload,
    levels: torch.Tensor,
) -> torch.Tensor:
    """Decode a grouped canonical P32 payload once into temporary FP16 KxN."""

    plan = payload.plan
    if levels.device.type != "cuda" or levels.dtype != torch.float16:
        raise ValueError("grouped P32 FP16 decoding requires FP16 CUDA levels")
    if (
        payload.trellis.device != levels.device
        or payload.bank_ids.device != levels.device
    ):
        raise ValueError("grouped P32 FP16 decode tensors must share one device")
    return _QVQ_WGMMA_EXTENSION.op("p32_window_decode_grouped_fp16")(
        payload.trellis,
        levels.contiguous(),
        payload.bank_ids,
        plan.transition_bits,
        plan.in_features,
        [segment.out_features for segment in plan.segments],
        [segment.bank_alt_id for segment in plan.segments],
    )


def qvq_p32_window_grouped_prefill_fp16_packed(
    input: torch.Tensor,
    payload: QVQHopperGroupedP32Payload,
    levels: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Decode P32 once, then run one grouped FP16-by-FP16 prefill GEMM."""

    plan = payload.plan
    if (
        input.ndim != 2
        or input.device.type != "cuda"
        or input.dtype != torch.float16
        or input.shape[1] != plan.in_features
    ):
        raise ValueError("grouped P32 FP16 prefill requires a matching FP16 CUDA matrix")
    decoded = qvq_p32_window_decode_grouped_fp16_packed(payload, levels)
    output = torch.mm(input.contiguous(), decoded, out_dtype=torch.float32)
    widths = [segment.out_features for segment in plan.segments]
    return tuple(torch.split(output, widths, dim=1))


def qvq_p32_window_prepare_grouped_fp16_packed(
    payload: QVQHopperGroupedP32Payload,
    levels: torch.Tensor,
    input_scale: torch.Tensor,
    output_scales: Sequence[torch.Tensor],
    output_hadamards: Sequence[bool],
) -> torch.Tensor:
    """Decode and fold grouped P32 into one temporary effective FP16 weight."""

    plan = payload.plan
    if (
        levels.device.type != "cuda"
        or levels.dtype != torch.float16
        or input_scale.device != levels.device
        or input_scale.dtype != torch.float32
        or input_scale.shape != (plan.in_features,)
    ):
        raise ValueError(
            "grouped folded-FP16 preparation requires matching CUDA FP16 levels "
            "and contiguous FP32 input scale [K]"
        )
    if len(output_scales) != len(plan.segments) or len(output_hadamards) != len(
        plan.segments
    ):
        raise ValueError(
            "grouped folded-FP16 preparation requires one output scale and "
            "Hadamard flag per segment"
        )
    for scale, segment in zip(output_scales, plan.segments, strict=True):
        if (
            scale.device != levels.device
            or scale.dtype != torch.float32
            or scale.shape != (segment.out_features,)
        ):
            raise ValueError(
                "grouped folded-FP16 output scales must be matching FP32 CUDA vectors"
            )
    return _QVQ_WGMMA_EXTENSION.op("p32_window_prepare_grouped_fp16")(
        payload.trellis,
        levels.contiguous(),
        payload.bank_ids,
        input_scale.contiguous(),
        [scale.contiguous() for scale in output_scales],
        plan.transition_bits,
        plan.in_features,
        [segment.out_features for segment in plan.segments],
        [segment.bank_alt_id for segment in plan.segments],
        [int(enabled) for enabled in output_hadamards],
    )


def qvq_p32_window_prepare_grouped_fp8_packed(
    payload: QVQHopperGroupedP32Payload,
    levels: torch.Tensor,
    input_scale: torch.Tensor,
    output_scales: Sequence[torch.Tensor],
    output_hadamards: Sequence[bool],
    weight_scale: torch.Tensor,
    *,
    half_fold: bool = False,
) -> torch.Tensor:
    """Decode/fold P32 directly to column-major E4M3 for cuBLASLt."""

    plan = payload.plan
    if (
        levels.device.type != "cuda"
        or levels.dtype != torch.float16
        or input_scale.device != levels.device
        or input_scale.dtype != torch.float32
        or input_scale.shape != (plan.in_features,)
        or weight_scale.device != levels.device
        or weight_scale.dtype != torch.float32
        or weight_scale.numel() != 1
    ):
        raise ValueError(
            "grouped folded-FP8 preparation requires matching CUDA levels, "
            "FP32 scales, and one FP32 weight scale"
        )
    if len(output_scales) != len(plan.segments) or len(output_hadamards) != len(
        plan.segments
    ):
        raise ValueError(
            "grouped folded-FP8 preparation requires one output scale and "
            "Hadamard flag per segment"
        )
    for scale, segment in zip(output_scales, plan.segments, strict=True):
        if (
            scale.device != levels.device
            or scale.dtype != torch.float32
            or scale.shape != (segment.out_features,)
        ):
            raise ValueError(
                "grouped folded-FP8 output scales must be matching FP32 CUDA vectors"
            )
    op_name = (
        "p32_window_prepare_grouped_fp8_half_fold"
        if half_fold
        else "p32_window_prepare_grouped_fp8"
    )
    transposed = _QVQ_WGMMA_EXTENSION.op(op_name)(
        payload.trellis,
        levels.contiguous(),
        payload.bank_ids,
        input_scale.contiguous(),
        [scale.contiguous() for scale in output_scales],
        weight_scale.contiguous(),
        plan.transition_bits,
        plan.in_features,
        [segment.out_features for segment in plan.segments],
        [segment.bank_alt_id for segment in plan.segments],
        [int(enabled) for enabled in output_hadamards],
    )
    return transposed.t()


def qvq_p32_window_wgmma_single_large_m_packed(
    input: torch.Tensor,
    trellis: torch.Tensor,
    levels: torch.Tensor,
    bank_ids: torch.Tensor,
    bits: float,
    *,
    out_features: int,
    bank_alt_id: int,
    split_count: int = 1,
) -> torch.Tensor:
    """Run one canonical P32 child through the large-M grouped row grid.

    This is a zero-copy plan wrapper: the canonical child's prepared window
    payload and selectors become the single segment. Ordinary ``QVQLinear``
    modules can therefore use M32/M64 decode reuse without manufacturing a
    grouped checkpoint or retaining another weight representation.
    """

    plan = qvq_p32_window_wgmma_group_plan(
        input,
        (trellis,),
        levels,
        (bank_ids,),
        bits,
        out_features=(out_features,),
        bank_alt_ids=(bank_alt_id,),
        split_counts=(split_count,),
    )
    payload = QVQHopperGroupedP32Payload(
        trellis=trellis,
        bank_ids=bank_ids,
        plan=plan,
    )
    rows = int(input.shape[0])
    if rows >= 64 and rows % 64 == 0:
        output = qvq_p32_window_wgmma_grouped_reuse4_packed(input, payload, levels)
    elif rows >= 32 and rows % 32 == 0:
        output = qvq_p32_window_wgmma_grouped_reuse2_packed(input, payload, levels)
    else:
        output = (
            qvq_p32_window_wgmma_grouped_ordered_packed(input, payload, levels)
            if split_count != 1
            else qvq_p32_window_wgmma_grouped_packed(input, payload, levels)
        )
    return output[0]


def qvq_p32_window_wgmma_grouped(
    input: torch.Tensor,
    trellises: Sequence[torch.Tensor],
    levels: torch.Tensor,
    bank_ids: Sequence[torch.Tensor],
    bits: float,
    *,
    out_features: Sequence[int],
    bank_alt_ids: Sequence[int],
    split_counts: Sequence[int] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Plan, losslessly pack, and run grouped Hopper P32 execution."""

    trellises = tuple(trellises)
    bank_ids = tuple(bank_ids)
    plan = qvq_p32_window_wgmma_group_plan(
        input,
        trellises,
        levels,
        bank_ids,
        bits,
        out_features=out_features,
        bank_alt_ids=bank_alt_ids,
        split_counts=split_counts,
    )
    if any(segment.split_count != 1 for segment in plan.segments):
        return tuple(
            qvq_p32_window_wgmma_m16_tma(
                input,
                trellis,
                levels,
                selectors,
                bits,
                out_features=segment.out_features,
                bank_alt_id=segment.bank_alt_id,
                split_count=segment.split_count,
            )
            for trellis, selectors, segment in zip(
                trellises, bank_ids, plan.segments, strict=True
            )
        )
    payload = qvq_pack_p32_window_hopper_group(trellises, bank_ids, plan)
    return qvq_p32_window_wgmma_grouped_packed(input, payload, levels)


__all__ = [
    "QVQHopperGroupedP32Payload",
    "QVQHopperGroupedP32Plan",
    "QVQHopperP32SegmentPlan",
    "qvq_fp16_to_fp8_e5m2_clamped",
    "qvq_h100_grouped_ordered_split_counts",
    "qvq_h100_large_m_ordered_split_count",
    "qvq_h100_ordered_split_count",
    "qvq_p32_window_decode_grouped_fp16_packed",
    "qvq_p32_window_grouped_prefill_fp16_packed",
    "qvq_p32_window_prepare_grouped_fp8_packed",
    "qvq_p32_window_prepare_grouped_fp16_packed",
    "qvq_p32_window_wgmma_group_plan",
    "qvq_p32_window_wgmma_grouped",
    "qvq_p32_window_wgmma_grouped_ordered_packed",
    "qvq_p32_window_wgmma_grouped_ordered_partials_packed",
    "qvq_p32_window_wgmma_grouped_packed",
    "qvq_p32_window_wgmma_grouped_reuse2_packed",
    "qvq_p32_window_wgmma_grouped_reuse4_packed",
    "qvq_p32_window_wgmma_m16_tma",
    "qvq_p32_window_wgmma_m16_tma_ordered_split",
    "qvq_p32_window_wgmma_single_large_m_packed",
    "qvq_p32_window_wgmma_w3_m16",
    "qvq_p32_window_wgmma_w3_m16_tma",
    "qvq_pack_p32_window_hopper_group",
]
