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
        "p32_window_w3_m16",
        "p32_window_w3_m16_tma",
        "p32_window_m16_tma",
        "p32_window_m16_tma_ordered_split",
        "p32_window_m16_tma_ordered_partials",
        "p32_window_m16_tma_grouped",
        "p32_window_m16_tma_grouped_ordered_split",
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
    if input.ndim != 2 or input.shape[0] != 16:
        raise ValueError("grouped Hopper P32 requires an M16 input")
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
    if tuple(input.shape) != (16, plan.in_features):
        raise ValueError("grouped Hopper P32 input does not match its M16 plan")
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
        child.reshape(16, width)
        for child, width in zip(
            torch.split(output, [16 * width for width in widths]),
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
    if tuple(input.shape) != (16, plan.in_features):
        raise ValueError("grouped Hopper P32 input does not match its M16 plan")
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
        child.reshape(16, width)
        for child, width in zip(
            torch.split(output, [16 * width for width in widths]),
            widths,
            strict=True,
        )
    )


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
    "qvq_h100_grouped_ordered_split_counts",
    "qvq_h100_ordered_split_count",
    "qvq_p32_window_wgmma_group_plan",
    "qvq_p32_window_wgmma_grouped",
    "qvq_p32_window_wgmma_grouped_ordered_packed",
    "qvq_p32_window_wgmma_grouped_packed",
    "qvq_p32_window_wgmma_m16_tma",
    "qvq_p32_window_wgmma_m16_tma_ordered_split",
    "qvq_p32_window_wgmma_w3_m16",
    "qvq_p32_window_wgmma_w3_m16_tma",
    "qvq_pack_p32_window_hopper_group",
]
