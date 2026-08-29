# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Native CUDA inner GEMV for the planar QVQ trellis format."""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Callable
from operator import index
from pathlib import Path

import torch

from ..quantization.qvq_codecs import PGC16_CODEBOOK_VERSION, pgc16_levels_for_version
from ..quantization.qvq_pruning import VITERBI_PRUNING_AUTO
from ..quantization.qvq_rates import (
    QVQ_BITS,
    normalize_qvq_rate,
    qvq_transition_bits,
    qvq_words_per_tile,
)
from .cpp import (
    TorchOpsJitExtension,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
)

QVQ_CUDA_BITS = QVQ_BITS
_QVQ_CUDA_OPS_NAME = "gptqmodel_qvq_cuda_ops"
_QVQ_CUDA_NAMESPACE = "gptqmodel_qvq"
_QVQ_CUDA_OP: Callable | None = None
_QVQ_CUDA_LR_OP: Callable | None = None
_QVQ_CUDA_VITERBI_OP: Callable | None = None
_QVQ_CUDA_VITERBI_TRUSTED_OP: Callable | None = None
_QVQ_CUDA_VITERBI_TAIL_TRUSTED_OP: Callable | None = None
_QVQ_CUDA_VITERBI_V4_OP: Callable | None = None
_QVQ_CUDA_VITERBI_BANKED_OP: Callable | None = None
_QVQ_CUDA_VITERBI_V2_SEGMENT_BANKED_OP: Callable | None = None
_QVQ_CUDA_VITERBI_V2_SEGMENT_G_OP: Callable | None = None
_QVQ_CUDA_VITERBI_V2_SEGMENT_GRID_OP: Callable | None = None
_QVQ_CUDA_VITERBI_V2_SEGMENT_GRID_TRUSTED_OP: Callable | None = None
_QVQ_CUDA_VITERBI_V2_SEGMENT_TAIL_TRUSTED_OP: Callable | None = None
_QVQ_CUDA_VITERBI_V2_SEGMENT_MIDPOINT_TRUSTED_OP: Callable | None = None
_QVQ_CUDA_VITERBI_V2_SEGMENT_FAMILY_GRID_TRUSTED_OP: Callable | None = None


def _validate_viterbi_distance_range(
    sequences: torch.Tensor,
    codebook: torch.Tensor,
    *,
    vector_size: int,
    step_weights: torch.Tensor | None = None,
) -> None:
    """Reject finite inputs whose native FP32 emission arithmetic can overflow."""

    if sequences.numel() == 0:
        return
    maximum_weight = 1.0 if step_weights is None else float(step_weights.detach().abs().amax().item())
    accumulation_terms = max(1.0, max(1, int(sequences.shape[-2])) * maximum_weight)
    safe_magnitude = math.sqrt(torch.finfo(torch.float32).max / accumulation_terms) / (2.0 * math.sqrt(vector_size))
    maximum = torch.maximum(sequences.detach().abs().amax(), codebook.detach().abs().amax())
    if bool(maximum > safe_magnitude):
        raise ValueError(
            "QVQ CUDA Viterbi sequence/codebook magnitudes are too large for finite FP32 "
            "squared-distance arithmetic"
        )
_QVQ_CUDA_HADAMARD_OP: Callable | None = None
_QVQ_CUDA_YAQA_FEEDBACK_OP: Callable | None = None
_QVQ_CUDA_YAQA_FEEDBACK_UPDATE_OP: Callable | None = None
_QVQ_CUDA_SWIGLU_PROXY_SCALES_OP: Callable | None = None
_QVQ_CUDA_OP_LOCK = threading.Lock()
_PGC16_LEVELS: dict[tuple[torch.device, str], torch.Tensor] = {}
_PGC16_LEVELS_LOCK = threading.Lock()


def _qvq_cuda_root() -> Path:
    return Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "qvq"


def _qvq_cuda_sources() -> list[str]:
    return [
        str(_qvq_cuda_root() / "qvq_gemv_cuda.cu"),
        str(_qvq_cuda_root() / "qvq_viterbi_cuda.cu"),
        str(_qvq_cuda_root() / "qvq_hadamard_cuda.cu"),
        str(_qvq_cuda_root() / "qvq_yaqa_cuda.cu"),
        str(_qvq_cuda_root() / "qvq_swiglu_cuda.cu"),
    ]


_QVQ_CUDA_TORCH_OPS_EXTENSION = TorchOpsJitExtension(
    name=_QVQ_CUDA_OPS_NAME,
    namespace=_QVQ_CUDA_NAMESPACE,
    required_ops=(
        "gemv",
        "gemv_lr",
        "gemv_v4",
        "viterbi",
        "viterbi_trusted",
        "viterbi_tail_trusted",
        "viterbi_v4",
        "viterbi_banked",
        "viterbi_v2_segment_banked",
        "viterbi_v2_segment_g",
        "viterbi_v2_segment_grid",
        "viterbi_v2_segment_grid_trusted",
        "viterbi_v2_segment_tail_trusted",
        "viterbi_v2_segment_midpoint_trusted",
        "viterbi_v2_segment_family_grid_trusted",
        "hadamard",
        "yaqa_feedback",
        "yaqa_feedback_update_",
        "norm_rank_telemetry_snapshot",
        "norm_rank_cache_size",
        "norm_cache_size",
        "swiglu_proxy_scales",
    ),
    sources=_qvq_cuda_sources,
    build_root_env="GPTQMODEL_QVQ_CUDA_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("qvq_cuda"),
    display_name="QVQ CUDA kernels",
    extra_cflags=lambda: default_jit_cflags(enable_bf16=True),
    extra_cuda_cflags=lambda: default_jit_cuda_cflags(
        enable_bf16=True,
        include_lineinfo=True,
        include_nvcc_threads=True,
        nvcc_threads=os.getenv("GPTQMODEL_QVQ_NVCC_THREADS", os.getenv("NVCC_THREADS", "1")),
        include_split_compile=True,
        include_fast_compile=True,
        include_ptxas_optimizations=True,
        include_ptxas_verbosity=False,
        include_fatbin_compression=True,
        include_diag_suppress=True,
    ),
    force_rebuild_env="GPTQMODEL_QVQ_CUDA_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
)


def _extension_api():
    from gptqmodel import extension as extension_api

    return extension_api


def qvq_cuda_norm_rank_telemetry_snapshot(device: torch.device | str | int) -> dict[str, int | float | bool | str]:
    """Return cumulative exact-pruning work counters for one CUDA device."""

    resolved = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    with torch.cuda.device(resolved):
        values = _extension_api().op("qvq_cuda", "norm_rank_telemetry_snapshot")()
        cache_entries = int(_extension_api().op("qvq_cuda", "norm_rank_cache_size")())
    dispatches, baseline_fallbacks, evaluated, possible = (int(value) for value in values)
    skipped = max(0, possible - evaluated)
    reduction = 0.0 if possible == 0 else skipped / possible
    return {
        "strategy": "norm_band",
        "exact": True,
        "eligible_dispatches": dispatches,
        "baseline_fallbacks": baseline_fallbacks,
        "candidates_evaluated": evaluated,
        "baseline_candidates_possible": possible,
        "candidates_skipped": skipped,
        "candidate_reduction": reduction,
        "norm_rank_cache_entries": cache_entries,
    }


def _qvq_cuda_op() -> Callable:
    """Resolve the operator once without locking steady-state launches."""

    global _QVQ_CUDA_OP
    if _QVQ_CUDA_OP is None:
        with _QVQ_CUDA_OP_LOCK:
            if _QVQ_CUDA_OP is None:
                _QVQ_CUDA_OP = _extension_api().op("qvq_cuda", "gemv")
    return _QVQ_CUDA_OP


def _qvq_cuda_lr_op() -> Callable:
    """Resolve the native K32 x N8 local-ring GEMV operator once."""

    global _QVQ_CUDA_LR_OP
    if _QVQ_CUDA_LR_OP is None:
        with _QVQ_CUDA_OP_LOCK:
            if _QVQ_CUDA_LR_OP is None:
                _QVQ_CUDA_LR_OP = _extension_api().op("qvq_cuda", "gemv_lr")
    return _QVQ_CUDA_LR_OP


def _qvq_cuda_viterbi_op() -> Callable:
    """Resolve the native Viterbi operator once without locking launches."""

    global _QVQ_CUDA_VITERBI_OP
    if _QVQ_CUDA_VITERBI_OP is None:
        with _QVQ_CUDA_OP_LOCK:
            if _QVQ_CUDA_VITERBI_OP is None:
                _QVQ_CUDA_VITERBI_OP = _extension_api().op("qvq_cuda", "viterbi")
    return _QVQ_CUDA_VITERBI_OP


def _qvq_cuda_viterbi_trusted_op() -> Callable:
    """Resolve YAQA's structurally checked, value-prevalidated V2 operator."""

    global _QVQ_CUDA_VITERBI_TRUSTED_OP
    if _QVQ_CUDA_VITERBI_TRUSTED_OP is None:
        with _QVQ_CUDA_OP_LOCK:
            if _QVQ_CUDA_VITERBI_TRUSTED_OP is None:
                _QVQ_CUDA_VITERBI_TRUSTED_OP = _extension_api().op("qvq_cuda", "viterbi_trusted")
    return _QVQ_CUDA_VITERBI_TRUSTED_OP


def _qvq_cuda_viterbi_tail_trusted_op() -> Callable:
    """Resolve YAQA's fused canonical two-pass V2 operator."""

    global _QVQ_CUDA_VITERBI_TAIL_TRUSTED_OP
    if _QVQ_CUDA_VITERBI_TAIL_TRUSTED_OP is None:
        with _QVQ_CUDA_OP_LOCK:
            if _QVQ_CUDA_VITERBI_TAIL_TRUSTED_OP is None:
                _QVQ_CUDA_VITERBI_TAIL_TRUSTED_OP = _extension_api().op(
                    "qvq_cuda", "viterbi_tail_trusted"
                )
    return _QVQ_CUDA_VITERBI_TAIL_TRUSTED_OP


def _qvq_cuda_hadamard_op() -> Callable:
    """Resolve the fused Hadamard operator once without locking steady-state launches."""

    global _QVQ_CUDA_HADAMARD_OP
    if _QVQ_CUDA_HADAMARD_OP is None:
        with _QVQ_CUDA_OP_LOCK:
            if _QVQ_CUDA_HADAMARD_OP is None:
                _QVQ_CUDA_HADAMARD_OP = _extension_api().op("qvq_cuda", "hadamard")
    return _QVQ_CUDA_HADAMARD_OP


def _qvq_cuda_yaqa_feedback_op() -> Callable:
    """Resolve the fused factored-YAQA feedback operator once."""

    global _QVQ_CUDA_YAQA_FEEDBACK_OP
    if _QVQ_CUDA_YAQA_FEEDBACK_OP is None:
        with _QVQ_CUDA_OP_LOCK:
            if _QVQ_CUDA_YAQA_FEEDBACK_OP is None:
                _QVQ_CUDA_YAQA_FEEDBACK_OP = _extension_api().op("qvq_cuda", "yaqa_feedback")
    return _QVQ_CUDA_YAQA_FEEDBACK_OP


def _qvq_cuda_yaqa_feedback_update_op() -> Callable:
    """Resolve the fused in-place factored-YAQA cache update operator once."""

    global _QVQ_CUDA_YAQA_FEEDBACK_UPDATE_OP
    if _QVQ_CUDA_YAQA_FEEDBACK_UPDATE_OP is None:
        with _QVQ_CUDA_OP_LOCK:
            if _QVQ_CUDA_YAQA_FEEDBACK_UPDATE_OP is None:
                _QVQ_CUDA_YAQA_FEEDBACK_UPDATE_OP = _extension_api().op(
                    "qvq_cuda", "yaqa_feedback_update_"
                )
    return _QVQ_CUDA_YAQA_FEEDBACK_UPDATE_OP


def qvq_cuda_swiglu_proxy_scales(
    gate: torch.Tensor,
    up: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    group_size: int,
    scale_min: float,
    scale_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run fused CUDA Smooth-SwiGLU statistics and analytical group solve."""
    global _QVQ_CUDA_SWIGLU_PROXY_SCALES_OP
    if _QVQ_CUDA_SWIGLU_PROXY_SCALES_OP is None:
        with _QVQ_CUDA_OP_LOCK:
            if _QVQ_CUDA_SWIGLU_PROXY_SCALES_OP is None:
                _QVQ_CUDA_SWIGLU_PROXY_SCALES_OP = _extension_api().op(
                    "qvq_cuda", "swiglu_proxy_scales"
                )
    return _QVQ_CUDA_SWIGLU_PROXY_SCALES_OP(
        gate, up, up_weight, down_weight, group_size, scale_min, scale_max
    )


def _qvq_cuda_viterbi_v4_op() -> Callable:
    """Resolve the native vector-size-four Viterbi operator once."""

    global _QVQ_CUDA_VITERBI_V4_OP
    if _QVQ_CUDA_VITERBI_V4_OP is None:
        with _QVQ_CUDA_OP_LOCK:
            if _QVQ_CUDA_VITERBI_V4_OP is None:
                _QVQ_CUDA_VITERBI_V4_OP = _extension_api().op("qvq_cuda", "viterbi_v4")
    return _QVQ_CUDA_VITERBI_V4_OP


def _qvq_cuda_viterbi_banked_op() -> Callable:
    """Resolve the bank-batched native Viterbi operator once."""

    global _QVQ_CUDA_VITERBI_BANKED_OP
    if _QVQ_CUDA_VITERBI_BANKED_OP is None:
        with _QVQ_CUDA_OP_LOCK:
            if _QVQ_CUDA_VITERBI_BANKED_OP is None:
                _QVQ_CUDA_VITERBI_BANKED_OP = _extension_api().op("qvq_cuda", "viterbi_banked")
    return _QVQ_CUDA_VITERBI_BANKED_OP


def _qvq_cuda_viterbi_v2_segment_banked_op() -> Callable:
    """Resolve the coupled segmented-bank V2 operator once."""

    global _QVQ_CUDA_VITERBI_V2_SEGMENT_BANKED_OP
    if _QVQ_CUDA_VITERBI_V2_SEGMENT_BANKED_OP is None:
        with _QVQ_CUDA_OP_LOCK:
            if _QVQ_CUDA_VITERBI_V2_SEGMENT_BANKED_OP is None:
                _QVQ_CUDA_VITERBI_V2_SEGMENT_BANKED_OP = _extension_api().op(
                    "qvq_cuda", "viterbi_v2_segment_banked"
                )
    return _QVQ_CUDA_VITERBI_V2_SEGMENT_BANKED_OP


def _qvq_cuda_viterbi_v2_segment_g_op() -> Callable:
    """Resolve the G-only segmented V2 operator once."""

    global _QVQ_CUDA_VITERBI_V2_SEGMENT_G_OP
    if _QVQ_CUDA_VITERBI_V2_SEGMENT_G_OP is None:
        with _QVQ_CUDA_OP_LOCK:
            if _QVQ_CUDA_VITERBI_V2_SEGMENT_G_OP is None:
                _QVQ_CUDA_VITERBI_V2_SEGMENT_G_OP = _extension_api().op(
                    "qvq_cuda", "viterbi_v2_segment_g"
                )
    return _QVQ_CUDA_VITERBI_V2_SEGMENT_G_OP


def _qvq_cuda_viterbi_v2_segment_grid_op() -> Callable:
    """Resolve the grid-parallel G-only segmented V2 operator once."""

    global _QVQ_CUDA_VITERBI_V2_SEGMENT_GRID_OP
    if _QVQ_CUDA_VITERBI_V2_SEGMENT_GRID_OP is None:
        with _QVQ_CUDA_OP_LOCK:
            if _QVQ_CUDA_VITERBI_V2_SEGMENT_GRID_OP is None:
                _QVQ_CUDA_VITERBI_V2_SEGMENT_GRID_OP = _extension_api().op(
                    "qvq_cuda", "viterbi_v2_segment_grid"
                )
    return _QVQ_CUDA_VITERBI_V2_SEGMENT_GRID_OP


def _qvq_cuda_viterbi_v2_segment_grid_trusted_op() -> Callable:
    """Resolve YAQA's prevalidated grid-parallel segmented V2 operator."""

    global _QVQ_CUDA_VITERBI_V2_SEGMENT_GRID_TRUSTED_OP
    if _QVQ_CUDA_VITERBI_V2_SEGMENT_GRID_TRUSTED_OP is None:
        with _QVQ_CUDA_OP_LOCK:
            if _QVQ_CUDA_VITERBI_V2_SEGMENT_GRID_TRUSTED_OP is None:
                _QVQ_CUDA_VITERBI_V2_SEGMENT_GRID_TRUSTED_OP = _extension_api().op(
                    "qvq_cuda", "viterbi_v2_segment_grid_trusted"
                )
    return _QVQ_CUDA_VITERBI_V2_SEGMENT_GRID_TRUSTED_OP


def _qvq_cuda_viterbi_v2_segment_tail_trusted_op() -> Callable:
    """Resolve YAQA's fused two-pass segmented V2 operator."""

    global _QVQ_CUDA_VITERBI_V2_SEGMENT_TAIL_TRUSTED_OP
    if _QVQ_CUDA_VITERBI_V2_SEGMENT_TAIL_TRUSTED_OP is None:
        with _QVQ_CUDA_OP_LOCK:
            if _QVQ_CUDA_VITERBI_V2_SEGMENT_TAIL_TRUSTED_OP is None:
                _QVQ_CUDA_VITERBI_V2_SEGMENT_TAIL_TRUSTED_OP = _extension_api().op(
                    "qvq_cuda", "viterbi_v2_segment_tail_trusted"
                )
    return _QVQ_CUDA_VITERBI_V2_SEGMENT_TAIL_TRUSTED_OP


def _qvq_cuda_viterbi_v2_segment_midpoint_trusted_op() -> Callable:
    """Resolve YAQA's midpoint-only provisional segmented V2 operator."""

    global _QVQ_CUDA_VITERBI_V2_SEGMENT_MIDPOINT_TRUSTED_OP
    if _QVQ_CUDA_VITERBI_V2_SEGMENT_MIDPOINT_TRUSTED_OP is None:
        with _QVQ_CUDA_OP_LOCK:
            if _QVQ_CUDA_VITERBI_V2_SEGMENT_MIDPOINT_TRUSTED_OP is None:
                _QVQ_CUDA_VITERBI_V2_SEGMENT_MIDPOINT_TRUSTED_OP = _extension_api().op(
                    "qvq_cuda", "viterbi_v2_segment_midpoint_trusted"
                )
    return _QVQ_CUDA_VITERBI_V2_SEGMENT_MIDPOINT_TRUSTED_OP


def _qvq_cuda_viterbi_v2_segment_family_grid_trusted_op() -> Callable:
    """Resolve the family-batched B2-P32 segmented V2 operator."""

    global _QVQ_CUDA_VITERBI_V2_SEGMENT_FAMILY_GRID_TRUSTED_OP
    if _QVQ_CUDA_VITERBI_V2_SEGMENT_FAMILY_GRID_TRUSTED_OP is None:
        with _QVQ_CUDA_OP_LOCK:
            if _QVQ_CUDA_VITERBI_V2_SEGMENT_FAMILY_GRID_TRUSTED_OP is None:
                _QVQ_CUDA_VITERBI_V2_SEGMENT_FAMILY_GRID_TRUSTED_OP = _extension_api().op(
                    "qvq_cuda", "viterbi_v2_segment_family_grid_trusted"
                )
    return _QVQ_CUDA_VITERBI_V2_SEGMENT_FAMILY_GRID_TRUSTED_OP


def qvq_cuda_viterbi(
    sequences: torch.Tensor,
    codebook: torch.Tensor,
    bits: float,
    overlap: torch.Tensor | None = None,
    step_weights: torch.Tensor | None = None,
    vector_size: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the exact L16 Viterbi recurrence in one persistent block per sequence."""

    bits = normalize_qvq_rate(bits)
    if vector_size not in (2, 4):
        raise ValueError("QVQ CUDA Viterbi vector_size must be 2 or 4")
    transition_bits = qvq_transition_bits(bits, vector_size=vector_size)
    if vector_size == 4 and transition_bits not in (4, 6, 8, 10, 12, 14, 16):
        raise ValueError("QVQ CUDA V4 Viterbi supports only even transition widths from 4 through 16")
    if sequences.ndim != 3 or sequences.shape[2] != vector_size:
        raise ValueError(f"QVQ CUDA Viterbi expects sequences with shape [batch, steps, {vector_size}]")
    if tuple(codebook.shape) != (1 << 16, vector_size):
        raise ValueError(f"QVQ CUDA Viterbi expects a [65536, {vector_size}] codebook")
    if sequences.device.type != "cuda" or codebook.device != sequences.device:
        raise ValueError("QVQ CUDA Viterbi tensors must share one CUDA device")
    if sequences.dtype != torch.float32 or codebook.dtype not in (torch.float16, torch.float32):
        raise TypeError("QVQ CUDA Viterbi requires float32 sequences and float16 or float32 codebook")
    if any(not tensor.is_contiguous() for tensor in (sequences, codebook)):
        raise ValueError("QVQ CUDA Viterbi tensors must be contiguous")
    sequence_alignment = 16 if vector_size == 4 else 4
    codebook_alignment = 16 if vector_size == 4 and codebook.dtype == torch.float32 else 4
    if sequences.data_ptr() % sequence_alignment or codebook.data_ptr() % codebook_alignment:
        raise ValueError(
            "QVQ CUDA Viterbi tensors must satisfy the native vector-load alignment contract "
            f"(sequence={sequence_alignment}, codebook={codebook_alignment} bytes)"
        )
    if not torch.isfinite(sequences).all() or not torch.isfinite(codebook).all():
        raise ValueError("QVQ CUDA Viterbi sequences and codebook must be finite")
    _validate_viterbi_distance_range(sequences, codebook, vector_size=vector_size, step_weights=step_weights)
    if overlap is not None and (
        overlap.device != sequences.device or overlap.dtype != torch.int64 or not overlap.is_contiguous()
    ):
        raise ValueError("QVQ CUDA Viterbi overlap must be contiguous int64 on the sequence device")
    if overlap is not None:
        if tuple(overlap.shape) != (sequences.shape[0],):
            raise ValueError("QVQ CUDA Viterbi overlap must have shape [batch]")
        overlap_limit = 1 << (16 - transition_bits)
        if torch.any((overlap < 0) | (overlap >= overlap_limit)):
            raise ValueError(f"QVQ CUDA Viterbi overlap must be in [0, {overlap_limit})")
    if step_weights is not None:
        if tuple(step_weights.shape) != tuple(sequences.shape[:2]):
            raise ValueError("QVQ CUDA Viterbi step weights must have shape [batch, steps]")
        if step_weights.device != sequences.device or step_weights.dtype != torch.float32:
            raise ValueError("QVQ CUDA Viterbi step weights must be float32 on the sequence device")
        if not step_weights.is_contiguous():
            raise ValueError("QVQ CUDA Viterbi step weights must be contiguous")
        if not torch.isfinite(step_weights).all() or torch.any(step_weights < 0):
            raise ValueError("QVQ CUDA Viterbi step weights must be finite and nonnegative")
    if torch.cuda.get_device_capability(sequences.device) < (8, 0):
        raise RuntimeError("QVQ CUDA Viterbi requires a compute capability >= 8.0 device")
    op = _qvq_cuda_viterbi_op() if vector_size == 2 else _qvq_cuda_viterbi_v4_op()
    return op(sequences, codebook, transition_bits, overlap, step_weights)


def _qvq_cuda_viterbi_trusted(
    sequences: torch.Tensor,
    codebook: torch.Tensor,
    bits: float,
    overlap: torch.Tensor | None = None,
    step_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run V2 after YAQA has deferred all dynamic value/range checks."""

    transition_bits = qvq_transition_bits(normalize_qvq_rate(bits), vector_size=2)
    return _qvq_cuda_viterbi_trusted_op()(sequences, codebook, transition_bits, overlap, step_weights)


def qvq_cuda_viterbi_banked(
    sequences: torch.Tensor,
    codebooks: torch.Tensor,
    bits: float,
    overlap: torch.Tensor | None = None,
    step_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one native V4 Viterbi launch over all active banks and source sequences.

    The normal YAQA path supplies four banks. Propagation candidate generation
    may intentionally supply only banks 1--3 because canonical bank 0 is
    retained as a separately computed oracle.
    """

    bits = normalize_qvq_rate(bits)
    transition_bits = qvq_transition_bits(bits, vector_size=4)
    if transition_bits not in (4, 6, 8, 10, 12, 14, 16):
        raise ValueError("QVQ banked CUDA Viterbi supports only even transition widths from 4 through 16")
    if sequences.ndim not in (3, 4) or sequences.shape[-1] != 4:
        raise ValueError("QVQ banked Viterbi expects sequences with shape [batch, steps, 4] or [banks, batch, steps, 4]")
    if sequences.ndim == 4 and sequences.shape[0] != codebooks.shape[0]:
        raise ValueError("bank-specific QVQ sequences must have one batch per codebook bank")
    if codebooks.ndim != 3 or codebooks.shape[0] not in (1, 2, 3, 4) or tuple(codebooks.shape[1:]) != (1 << 16, 4):
        raise ValueError("QVQ banked Viterbi expects one to four codebooks with shape [banks, 65536, 4]")
    if sequences.device.type != "cuda" or codebooks.device != sequences.device:
        raise ValueError("QVQ banked Viterbi tensors must share one CUDA device")
    if sequences.dtype != torch.float32 or codebooks.dtype not in (torch.float16, torch.float32):
        raise TypeError("QVQ banked Viterbi requires float32 sequences and float16 or float32 codebooks")
    if not sequences.is_contiguous() or not codebooks.is_contiguous():
        raise ValueError("QVQ banked Viterbi tensors must be contiguous")
    codebook_alignment = 16 if codebooks.dtype == torch.float32 else 4
    if sequences.data_ptr() % 16 or codebooks.data_ptr() % codebook_alignment:
        raise ValueError(
            "QVQ banked Viterbi tensors must satisfy the native vector-load alignment contract "
            f"(sequence=16, codebooks={codebook_alignment} bytes)"
        )
    if not torch.isfinite(sequences).all() or not torch.isfinite(codebooks).all():
        raise ValueError("QVQ banked Viterbi sequences and codebooks must be finite")
    _validate_viterbi_distance_range(sequences, codebooks, vector_size=4, step_weights=step_weights)
    bank_count = codebooks.shape[0]
    batch = sequences.shape[1] if sequences.ndim == 4 else sequences.shape[0]
    steps = sequences.shape[2] if sequences.ndim == 4 else sequences.shape[1]
    if overlap is not None:
        if overlap.device != sequences.device or overlap.dtype != torch.int64:
            raise ValueError("QVQ banked Viterbi overlap must be CUDA int64")
        if overlap.numel() != bank_count * batch or not overlap.is_contiguous():
            raise ValueError("QVQ banked Viterbi overlap must contain bank_count * batch entries")
        overlap_limit = 1 << (16 - transition_bits)
        if torch.any((overlap < 0) | (overlap >= overlap_limit)):
            raise ValueError(f"QVQ banked Viterbi overlap must be in [0, {overlap_limit})")
        overlap = overlap.reshape(-1).contiguous()
    if step_weights is not None:
        if tuple(step_weights.shape) != (batch, steps) or step_weights.device != sequences.device:
            raise ValueError("QVQ banked Viterbi step weights must have shape [batch, steps] on CUDA")
        if step_weights.dtype != torch.float32 or not step_weights.is_contiguous():
            raise ValueError("QVQ banked Viterbi step weights must be contiguous float32")
        if not torch.isfinite(step_weights).all() or torch.any(step_weights < 0):
            raise ValueError("QVQ banked Viterbi step weights must be finite and nonnegative")
    if torch.cuda.get_device_capability(sequences.device) < (8, 0):
        raise RuntimeError("QVQ banked Viterbi requires compute capability >= 8.0")
    states, squared_error = _qvq_cuda_viterbi_banked_op()(
        sequences, codebooks, transition_bits, overlap, step_weights
    )
    return states.reshape(bank_count, batch, steps), squared_error.reshape(bank_count, batch)


def qvq_cuda_viterbi_v2_segment_banked(
    sequences: torch.Tensor,
    codebooks: torch.Tensor,
    bits: float,
    segment_steps: int,
    overlap: torch.Tensor | None = None,
    step_weights: torch.Tensor | None = None,
    pruning_policy: int = VITERBI_PRUNING_AUTO,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the exact coupled V2 bank recurrence for P32 or P64 selectors.

    ``pruning_policy`` is the native exact survivor-pruning policy code from
    :mod:`gptqmodel.quantization.qvq_pruning`. It defaults to ``auto``, so
    direct low-level callers keep the historical automatic behavior.
    """

    bits = normalize_qvq_rate(bits)
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    if transition_bits not in (2, 3, 4, 5, 6, 7):
        raise ValueError("QVQ segmented-bank CUDA V2 supports only rates W1 through W3.5")
    if tuple(sequences.shape[1:]) != (128, 2) or sequences.ndim != 3:
        raise ValueError("QVQ segmented-bank V2 expects sequences with shape [batch, 128, 2]")
    if codebooks.ndim != 3 or codebooks.shape[0] not in (2, 4) or tuple(codebooks.shape[1:]) != (1 << 16, 2):
        raise ValueError("QVQ segmented-bank V2 expects codebooks with shape [2|4, 65536, 2]")
    bank_count = int(codebooks.shape[0])
    if (bank_count, segment_steps) not in ((2, 16), (4, 32)):
        raise ValueError("QVQ segmented-bank V2 requires two P32 banks or four P64 banks")
    if sequences.device.type != "cuda" or codebooks.device != sequences.device:
        raise ValueError("QVQ segmented-bank V2 tensors must share one CUDA device")
    if sequences.dtype != torch.float32 or codebooks.dtype not in (torch.float16, torch.float32):
        raise TypeError("QVQ segmented-bank V2 requires float32 sequences and float16 or float32 codebooks")
    if not sequences.is_contiguous() or not codebooks.is_contiguous():
        raise ValueError("QVQ segmented-bank V2 tensors must be contiguous")
    if sequences.data_ptr() % 4 or codebooks.data_ptr() % 4:
        raise ValueError("QVQ segmented-bank V2 tensors must satisfy four-byte native alignment")
    if not torch.isfinite(sequences).all() or not torch.isfinite(codebooks).all():
        raise ValueError("QVQ segmented-bank V2 sequences and codebooks must be finite")
    _validate_viterbi_distance_range(sequences, codebooks, vector_size=2, step_weights=step_weights)
    batch = int(sequences.shape[0])
    if overlap is not None:
        if (
            overlap.device != sequences.device
            or overlap.dtype != torch.int64
            or tuple(overlap.shape) != (batch,)
            or not overlap.is_contiguous()
        ):
            raise ValueError("QVQ segmented-bank V2 overlap must be contiguous CUDA int64 with shape [batch]")
        overlap_limit = 1 << (16 - transition_bits)
        if torch.any((overlap < 0) | (overlap >= overlap_limit)):
            raise ValueError(f"QVQ segmented-bank V2 overlap must be in [0, {overlap_limit})")
    if step_weights is not None:
        if (
            step_weights.device != sequences.device
            or step_weights.dtype != torch.float32
            or tuple(step_weights.shape) != (batch, 128)
            or not step_weights.is_contiguous()
        ):
            raise ValueError(
                "QVQ segmented-bank V2 step weights must be contiguous CUDA float32 with shape [batch, 128]"
            )
        if not torch.isfinite(step_weights).all() or torch.any(step_weights < 0):
            raise ValueError("QVQ segmented-bank V2 step weights must be finite and nonnegative")
    if torch.cuda.get_device_capability(sequences.device) < (8, 0):
        raise RuntimeError("QVQ segmented-bank V2 requires compute capability >= 8.0")
    return _qvq_cuda_viterbi_v2_segment_grid_op()(
        sequences,
        codebooks,
        transition_bits,
        segment_steps,
        overlap,
        step_weights,
        int(pruning_policy),
    )


def qvq_cuda_hadamard(
    x: torch.Tensor,
    *,
    pre_scale: torch.Tensor | None = None,
    post_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    scale_mode: int = 0,
) -> torch.Tensor:
    """Apply the fast Walsh-Hadamard transform on the last dim in one fused launch.

    One kernel launch: optional per-column pre-scale, butterfly stages in float
    with output rounding, then optional per-column post-scale and bias. Two
    normalization modes mirror the Python references bitwise:
      - scale_mode 0: normalize FIRST (matmul_hadU_stable, fp16(sqrtf(n)) divisor)
      - scale_mode 1: normalize LAST (matmul_hadU, float sqrtf(n) divisor)
      - scale_mode 2: fuse pre-scale and normalization before the first narrow store
      - scale_mode 3/4: FP32 storage with finite FP16-rounding emulation for mode 0/1
    Validated bitwise-identical to the corresponding Python butterfly on CUDA for
    power-of-two dims.
    """

    if x.device.type != "cuda":
        raise ValueError("QVQ CUDA Hadamard requires a CUDA input")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"QVQ CUDA Hadamard requires float16, bfloat16, or float32 x, got {x.dtype}")
    if x.dim() < 1 or not x.is_contiguous():
        raise ValueError("QVQ CUDA Hadamard requires a contiguous tensor with rank >= 1")
    n = x.shape[-1]
    if n < 2 or n & (n - 1) or n > 16384:
        raise ValueError(f"QVQ CUDA Hadamard requires a power-of-two last dim in [2, 16384], got {n}")
    if scale_mode not in (0, 1, 2, 3, 4):
        raise ValueError("QVQ CUDA Hadamard scale_mode must be one of 0, 1, 2, 3, or 4")
    if scale_mode == 2 and x.dtype != torch.float16:
        raise TypeError("QVQ CUDA Hadamard range-safe pre-scale mode 2 requires float16 x")
    if scale_mode >= 3 and x.dtype != torch.float32:
        raise TypeError("QVQ CUDA Hadamard FP16-emulation modes 3/4 require float32 x")
    if torch.cuda.get_device_capability(x.device) < (8, 0):
        raise RuntimeError("QVQ CUDA Hadamard requires a compute capability >= 8.0 device")
    return _qvq_cuda_hadamard_op()(x, pre_scale, post_scale, bias, scale_mode)


def qvq_cuda_device_supported(device: torch.device | str) -> bool:
    """Return whether one concrete device can execute the native QVQ kernels."""

    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    try:
        return torch.cuda.get_device_capability(target) >= (8, 0)
    except (AssertionError, RuntimeError):
        return False


def _sm80_or_newer_device_available() -> bool:
    return any(
        qvq_cuda_device_supported(torch.device("cuda", device_index))
        for device_index in range(torch.cuda.device_count())
    )


def qvq_cuda_supported() -> bool:
    return _sm80_or_newer_device_available()


def qvq_cuda_available() -> bool:
    return qvq_cuda_supported() and _extension_api().is_available("qvq_cuda")


def qvq_cuda_error() -> str:
    if not torch.cuda.is_available():
        return "QVQ CUDA requires CUDA."
    if torch.version.hip is not None:
        return "QVQ CUDA requires NVIDIA CUDA; ROCm is not supported."
    if not _sm80_or_newer_device_available():
        return "QVQ CUDA requires a compute capability >= 8.0 device."
    return _extension_api().error("qvq_cuda")


def prewarm_qvq_cuda() -> bool:
    return _extension_api().load(name="qvq_cuda")["qvq_cuda"]


def _integer_argument(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    try:
        return index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from exc


def _pgc16_levels(
    device: torch.device,
    codebook_version: str,
) -> torch.Tensor:
    """Return the fixed FP16 PGC16-v1 table for one CUDA device.

    FP16 bit patterns are part of the PGC16 format. In particular, rounding the
    table to BF16 changes the decoded weight before accumulation and is not a
    valid dtype specialization of the codec.
    """

    key = (device, str(codebook_version).strip().lower())
    levels = _PGC16_LEVELS.get(key)
    if levels is None:
        with _PGC16_LEVELS_LOCK:
            levels = _PGC16_LEVELS.get(key)
            if levels is None:
                levels = pgc16_levels_for_version(key[1]).to(device=device).contiguous()
                _PGC16_LEVELS[key] = levels
    return levels


def qvq_cuda_gemv(
    x: torch.Tensor,
    trellis: torch.Tensor,
    bits: float,
    *,
    out_features: int,
    codebook_version: str = PGC16_CODEBOOK_VERSION,
    output_fp32: bool = False,
    vector_size: int = 2,
    bank_ids: torch.Tensor | None = None,
    v2b4_p64: bool = False,
    v2b2_p32: bool = False,
    v2b2_p32_lr: bool = False,
    bank_alt_id: int = 0,
    lr_split_count: int = 0,
) -> torch.Tensor:
    """Multiply transformed activations by planar QVQ tiles on the current CUDA stream.

    The kernel decodes tiles directly and owns no persistent dequantized cache.
    """

    bits = normalize_qvq_rate(bits)
    if vector_size not in (2, 4):
        raise ValueError("QVQ CUDA vector_size must be 2 or 4")
    if not isinstance(v2b4_p64, bool) or not isinstance(v2b2_p32, bool) or not isinstance(v2b2_p32_lr, bool):
        raise TypeError("QVQ CUDA segmented-bank format flags must be bools")
    if sum((v2b4_p64, v2b2_p32, v2b2_p32_lr)) > 1:
        raise ValueError("QVQ CUDA V2B4-P64, V2B2-P32, and V2B2-P32-LR are mutually exclusive")
    if (v2b4_p64 or v2b2_p32 or v2b2_p32_lr) and vector_size != 2:
        raise ValueError("QVQ CUDA segmented-bank formats require vector_size=2")
    if bank_ids is not None and vector_size != 4 and not (v2b4_p64 or v2b2_p32 or v2b2_p32_lr):
        raise ValueError("QVQ CUDA bank selectors require V4 or a segmented-bank V2 format")
    if vector_size == 2 and (v2b4_p64 or v2b2_p32 or v2b2_p32_lr) != (bank_ids is not None):
        raise ValueError("QVQ CUDA segmented-bank formats require exactly one packed selector byte per tile")
    bank_alt_id = _integer_argument("bank_alt_id", bank_alt_id)
    lr_split_count = _integer_argument("lr_split_count", lr_split_count)
    if lr_split_count < 0:
        raise ValueError("QVQ CUDA lr_split_count must be non-negative")
    if lr_split_count > 64:
        raise ValueError("QVQ CUDA lr_split_count must be in [1, 64] or 0 for automatic selection")
    if lr_split_count and not v2b2_p32_lr:
        raise ValueError("QVQ CUDA lr_split_count is valid only for V2B2-P32-LR")
    if v2b2_p32 or v2b2_p32_lr:
        if not 1 <= bank_alt_id <= 3:
            raise ValueError("QVQ CUDA V2B2-P32 bank_alt_id must be in [1, 3]")
    elif bank_alt_id != 0:
        raise ValueError("QVQ CUDA bank_alt_id is valid only for V2B2-P32")
    transition_bits = qvq_transition_bits(bits, vector_size=vector_size)
    if (v2b4_p64 or v2b2_p32 or v2b2_p32_lr) and transition_bits > 7:
        raise ValueError("QVQ CUDA segmented-bank V2 inference supports only rates W1 through W3.5")
    out_features = _integer_argument("out_features", out_features)
    if x.ndim != 2 or trellis.ndim != 2:
        raise ValueError("QVQ CUDA expects 2D x and trellis tensors")
    if x.device.type != "cuda" or trellis.device != x.device:
        raise ValueError("QVQ CUDA tensors must share one CUDA device")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"QVQ CUDA requires float16 or bfloat16 x, got {x.dtype}")
    if trellis.dtype != torch.int32:
        raise TypeError("QVQ CUDA requires int32 planar trellis words")
    if any(not tensor.is_contiguous() for tensor in (x, trellis)):
        raise ValueError("QVQ CUDA tensors must be contiguous")

    m, k = x.shape
    n = out_features
    if k <= 0 or n <= 0 or k % (32 if v2b2_p32_lr else 16) or n % (8 if v2b2_p32_lr else 16):
        divisor = "K32/N8" if v2b2_p32_lr else "16"
        raise ValueError(f"QVQ CUDA requires positive dimensions divisible by {divisor}, got K={k}, N={n}")
    tile_count = (k // 32) * (n // 8) if v2b2_p32_lr else (k // 16) * (n // 16)
    expected = (tile_count, qvq_words_per_tile(bits, vector_size=vector_size))
    if tuple(trellis.shape) != expected:
        raise ValueError(f"QVQ planar trellis must have shape {expected}, got {tuple(trellis.shape)}")
    if not isinstance(output_fp32, bool):
        raise TypeError("QVQ CUDA output_fp32 must be boolean")
    if bank_ids is not None:
        if bank_ids.device != x.device or bank_ids.dtype != torch.uint8:
            raise TypeError("QVQ CUDA bank selectors must be contiguous uint8 on the input CUDA device")
        if not bank_ids.is_contiguous():
            raise ValueError("QVQ CUDA bank selectors must be contiguous")
        if tuple(bank_ids.shape) != (tile_count,):
            raise ValueError(f"QVQ CUDA bank selectors must have shape {(tile_count,)}")
        if vector_size == 4 and torch.any(bank_ids > 3):
            raise ValueError("QVQ V4 bank selectors must be in [0, 3]")
    levels = _pgc16_levels(x.device, codebook_version)
    if m == 0:
        return torch.empty((0, n), dtype=torch.float32 if output_fp32 else x.dtype, device=x.device)
    if max(m, k, n) > 2**31 - 1:
        raise ValueError("QVQ CUDA dimensions exceed the int32 kernel limit")
    if torch.cuda.get_device_capability(x.device) < (8, 0):
        raise RuntimeError("QVQ CUDA requires a compute capability >= 8.0 device")
    if v2b2_p32_lr:
        return _qvq_cuda_lr_op()(
            x, trellis, levels, transition_bits, n, output_fp32, bank_ids, bank_alt_id, lr_split_count
        )
    op = _qvq_cuda_op()
    if vector_size == 4:
        return torch.ops.gptqmodel_qvq.gemv_v4(
            x, trellis, levels, transition_bits, n, output_fp32, bank_ids
        )
    bank_mode = 2 if v2b4_p64 else 3 if v2b2_p32 else 0
    return op(x, trellis, levels, transition_bits, n, output_fp32, bank_ids, bank_mode, bank_alt_id)


__all__ = [
    "QVQ_CUDA_BITS",
    "prewarm_qvq_cuda",
    "qvq_cuda_available",
    "qvq_cuda_device_supported",
    "qvq_cuda_error",
    "qvq_cuda_gemv",
    "qvq_cuda_hadamard",
    "qvq_cuda_supported",
    "qvq_cuda_viterbi",
    "qvq_cuda_viterbi_banked",
    "qvq_cuda_viterbi_v2_segment_banked",
]
