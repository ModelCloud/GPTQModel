# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Clean-room QVQ reference math derived from the published QTIP and YAQA papers."""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Callable, Iterator

import torch

from ..utils.planar_packing import planar_pack_rows, planar_unpack_rows
from ..utils.qvq_cpu import qvq_cpu_supported
from .qvq_codecs import (
    PGC16_CODEBOOK_VERSION,
    PGC16_NORMALIZATION_RMS,
    pgc16_codebook,
    pgc16_codebook_v2_bank,
    pgc16_codebook_v4,
    pgc16_codebook_v4_bank,
    pgc16_decode_states,
    pgc16_decode_states_v2_banked,
    pgc16_decode_states_v4,
    pgc16_decode_states_v4_banked,
    pgc16_levels_for_version,
    pgc16_scale_factor,
    pgc18_codebook_v4,
    pgc18_decode_states_v4,
)
from .qvq_pruning import (
    reject_viterbi_pruning_fallback_if_strict,
    resolve_viterbi_pruning_policy,
    viterbi_pruning_dispatch_code,
)
from .qvq_rates import (
    QVQ_BITS as _QVQ_BITS,
)
from .qvq_rates import (
    normalize_qvq_rate,
    qvq_transition_bits,
    qvq_words_per_tile,
)
from .qvq_yaqa import YAQA_DEFAULT_REGULARIZATION, YAQA_PAPER_REGULARIZATION
from .rotation.hadamard_utils import matmul_hadU

QVQ_BITS = _QVQ_BITS
QVQ_V2B4_P64_SEGMENT_WEIGHTS = 64
QVQ_V2B4_P64_SEGMENTS_PER_TILE = 4
QVQ_V2B4_P64_STEPS_PER_SEGMENT = QVQ_V2B4_P64_SEGMENT_WEIGHTS // 2
QVQ_V2B2_P32_SEGMENT_WEIGHTS = 32
QVQ_V2B2_P32_SEGMENTS_PER_TILE = 8
QVQ_V2B2_P32_STEPS_PER_SEGMENT = QVQ_V2B2_P32_SEGMENT_WEIGHTS // 2
# LR32 keeps the same eight 32-weight binary-bank rings, but makes each ring
# one output column over a K32 x N8 logical tile.  The serialized payload is
# still 256 weights and therefore has the same word count/BPW as P32.
QVQ_V2B2_P32_LR_RINGS_PER_TILE = 8
QVQ_V2B2_P32_LR_RING_STEPS = 16
QVQ_V2B2_P32_LR_TILE_ROWS = 32
QVQ_V2B2_P32_LR_TILE_COLS = 8
QVQ_YAQA_SAMPLE_TILE_COUNTS = {
    "full": None,
    "32_16x16": 32,
    "64_16x16": 64,
    "96_16x16": 96,
    "128_16x16": 128,
    "256_16x16": 256,
}


def _yaqa_sample_tile_indices(tile_count: int, sample_strategy: str) -> torch.Tensor:
    """Return deterministic, evenly spaced 16x16 tile indices for a sampled YAQA family screen."""

    if isinstance(tile_count, bool) or not isinstance(tile_count, int) or tile_count < 1:
        raise ValueError("YAQA sampled family selection requires a positive tile count.")
    requested_count = QVQ_YAQA_SAMPLE_TILE_COUNTS.get(sample_strategy)
    if requested_count is None:
        if sample_strategy == "full":
            raise ValueError("YAQA `full` selection evaluates complete module candidates and does not sample tiles.")
        raise ValueError("Unknown YAQA sample strategy.")
    sample_count = min(requested_count, tile_count)
    return torch.linspace(0, tile_count - 1, sample_count, dtype=torch.float64).round().to(torch.long)
_QVQ_LOW_RATE_OUTPUT_SCALE_STRENGTH = 0.5
_QVQ_PROPAGATION_DELTA_CACHE_BYTES = 256 * 1024 * 1024
_QVQ_CODEBOOK_CACHE_LOCK = threading.Lock()
_QVQ_CODEBOOK_CACHE: dict[tuple[str, int, int, float, str, torch.dtype], torch.Tensor] = {}
_QVQ_V4_BANK_CACHE: dict[tuple[str, float, str, torch.dtype], tuple[torch.Tensor, ...]] = {}
_QVQ_V4_BANK_STACK_CACHE: dict[tuple[str, float, str, torch.dtype], torch.Tensor] = {}
_QVQ_V2B4_BANK_CACHE: dict[tuple[str, float, str, torch.dtype], tuple[torch.Tensor, ...]] = {}
_QVQ_V2B4_BANK_STACK_CACHE: dict[tuple[str, float, str, torch.dtype], torch.Tensor] = {}
_QVQ_V2B2_PAIR_STACK_CACHE: dict[tuple[str, float, str, torch.dtype], tuple[torch.Tensor, ...]] = {}
_QVQ_YAQA_FAMILY_STREAM_LOCK = threading.Lock()
_QVQ_YAQA_FAMILY_STREAMS: dict[tuple[str, int], tuple[torch.cuda.Stream, ...]] = {}
_QVQ_YAQA_SCHEDULE_LOCK = threading.Lock()
_QVQ_YAQA_SCHEDULE_CACHE: dict[
    tuple[str, int, int, int, int],
    tuple[
        tuple[
            tuple[tuple[int, int], ...],
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        ...,
    ],
] = {}


def _yaqa_family_streams(device: torch.device, count: int) -> tuple[torch.cuda.Stream, ...]:
    """Return persistent side streams for independent B2 family candidates."""

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    key = (device.type, device_index)
    with _QVQ_YAQA_FAMILY_STREAM_LOCK:
        streams = _QVQ_YAQA_FAMILY_STREAMS.get(key)
        if streams is None or len(streams) < count:
            streams = tuple(torch.cuda.Stream(device=device) for _ in range(count))
            _QVQ_YAQA_FAMILY_STREAMS[key] = streams
    return streams[:count]


def _yaqa_anti_diagonal_schedule(
    device: torch.device,
    input_blocks: int,
    output_blocks: int,
    tile_rows: int = 16,
    tile_cols: int = 16,
) -> tuple[
    tuple[
        tuple[tuple[int, int], ...],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    ...,
]:
    """Reuse immutable YAQA coordinates and device index tensors by geometry."""

    key = (str(device), input_blocks, output_blocks, tile_rows, tile_cols)
    with _QVQ_YAQA_SCHEDULE_LOCK:
        schedule = _QVQ_YAQA_SCHEDULE_CACHE.get(key)
        if schedule is None:
            entries = []
            for diagonal in range(input_blocks + output_blocks - 2, -1, -1):
                first_input_block = max(0, diagonal - output_blocks + 1)
                last_input_block = min(input_blocks - 1, diagonal)
                coordinates = tuple(
                    (input_block, diagonal - input_block)
                    for input_block in range(first_input_block, last_input_block + 1)
                )
                input_indices = torch.tensor(
                    [coordinate[0] for coordinate in coordinates], dtype=torch.long, device=device
                )
                output_indices = torch.tensor(
                    [coordinate[1] for coordinate in coordinates], dtype=torch.long, device=device
                )
                entries.append(
                    (
                        coordinates,
                        input_indices,
                        output_indices,
                        input_indices * output_blocks + output_indices,
                        (
                            input_indices[:, None] * tile_rows
                            + torch.arange(tile_rows, dtype=torch.long, device=device)[None]
                        ).reshape(-1),
                        (
                            output_indices[:, None] * tile_cols
                            + torch.arange(tile_cols, dtype=torch.long, device=device)[None]
                        ),
                    )
                )
            schedule = tuple(entries)
            _QVQ_YAQA_SCHEDULE_CACHE[key] = schedule
    return schedule


def _yaqa_segmented_batch_size(
    logical_batch_size: int,
    requested_batch_size: int,
    bits: float,
    *,
    apple_host_feedback: bool,
    cuda_feedback: bool,
) -> int:
    """Choose an occupancy-aware YAQA tile batch without overriding explicit CUDA tuning."""

    if apple_host_feedback:
        apple_auto_batch = 32 if bits <= 2 or bits == 3 else 128
        return max(requested_batch_size, min(logical_batch_size, apple_auto_batch))
    if cuda_feedback and requested_batch_size == 16:
        cuda_auto_batch = 32 if bits == 1 else 64
        return min(logical_batch_size, cuda_auto_batch)
    return requested_batch_size


def _yaqa_viterbi_batch_size(
    logical_batch_size: int,
    requested_batch_size: int,
    bits: float,
    *,
    cuda_feedback: bool,
) -> int:
    """Fill canonical CUDA Viterbi launches while retaining explicit caller policy."""

    if cuda_feedback and requested_batch_size == 16:
        cuda_auto_batch = 32 if bits <= 1.5 else 128
        return min(logical_batch_size, cuda_auto_batch)
    return requested_batch_size


def _canonical_qvq_codebook(
    *,
    device: torch.device,
    vector_size: int,
    trellis_window: int = 16,
    bits: float = 2.0,
    codebook_version: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Reuse one immutable production codebook per CUDA device and format.

    Quantizing a MoE model creates thousands of module-local QVQ tasks, but the
    canonical PGC16 table is identical for every task. Keeping the cache at the
    Python layer ensures the native norm cache sees one stable data pointer
    instead of retaining one table and norm buffer per module.
    """

    # L16's canonical codebook is rate-independent. L18 embeds the rate-keyed
    # bank masks in its reconstruction manifold and therefore needs one cache
    # entry per rate.
    rate_key = float(bits) if trellis_window == 18 else 0.0
    key = (str(device), vector_size, trellis_window, rate_key, codebook_version, dtype)
    with _QVQ_CODEBOOK_CACHE_LOCK:
        cached = _QVQ_CODEBOOK_CACHE.get(key)
        if cached is not None:
            return cached
        levels = pgc16_levels_for_version(codebook_version).to(device=device)
        if trellis_window == 18:
            if vector_size != 4:
                raise ValueError("QVQ L18 currently requires vector_size=4.")
            codebook = pgc18_codebook_v4(bits=bits, device=device, dtype=dtype, levels=levels)
        else:
            codebook = (
                pgc16_codebook(device=device, dtype=dtype, levels=levels)
                if vector_size == 2
                else pgc16_codebook_v4(device=device, dtype=dtype, levels=levels)
            )
        _QVQ_CODEBOOK_CACHE[key] = codebook
        return codebook


def _canonical_qvq_v4_banks(
    *,
    device: torch.device,
    bits: float,
    codebook_version: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    """Return one immutable, rate-keyed four-bank tuple per device."""

    key = (str(device), float(bits), codebook_version, dtype)
    with _QVQ_CODEBOOK_CACHE_LOCK:
        cached = _QVQ_V4_BANK_CACHE.get(key)
        if cached is not None:
            return cached
        bank_stack = torch.stack(tuple(
            pgc16_codebook_v4_bank(
                bank,
                bits=bits,
                device=device,
                dtype=dtype,
                levels=pgc16_levels_for_version(codebook_version).to(device=device),
            ).detach()
            for bank in range(4)
        )).contiguous()
        # Keep bank views backed by one immutable allocation. This gives the
        # native norm cache one stable bank-major pointer across all modules.
        banks = tuple(bank_stack[bank] for bank in range(4))
        _QVQ_V4_BANK_CACHE[key] = banks
        _QVQ_V4_BANK_STACK_CACHE[key] = bank_stack
        return banks


def _canonical_qvq_v2b4_banks(
    *,
    device: torch.device,
    bits: float,
    codebook_version: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    """Return the immutable rate-keyed V2B4 graph portfolio."""

    key = (str(device), float(bits), codebook_version, dtype)
    with _QVQ_CODEBOOK_CACHE_LOCK:
        cached = _QVQ_V2B4_BANK_CACHE.get(key)
        if cached is not None:
            return cached
        levels = pgc16_levels_for_version(codebook_version).to(device=device)
        stack = torch.stack(
            tuple(
                pgc16_codebook_v2_bank(bank, bits=bits, device=device, dtype=dtype, levels=levels).detach()
                for bank in range(4)
            )
        ).contiguous()
        banks = tuple(stack[bank] for bank in range(4))
        _QVQ_V2B4_BANK_CACHE[key] = banks
        _QVQ_V2B4_BANK_STACK_CACHE[key] = stack
        return banks


def _canonical_qvq_v2b4_bank_stack(
    *,
    device: torch.device,
    bits: float,
    codebook_version: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the globally shared contiguous bank-major V2 codebook tensor."""

    key = (str(device), float(bits), codebook_version, dtype)
    with _QVQ_CODEBOOK_CACHE_LOCK:
        stack = _QVQ_V2B4_BANK_STACK_CACHE.get(key)
    if stack is None:
        _canonical_qvq_v2b4_banks(
            device=device,
            bits=bits,
            codebook_version=codebook_version,
            dtype=dtype,
        )
        with _QVQ_CODEBOOK_CACHE_LOCK:
            stack = _QVQ_V2B4_BANK_STACK_CACHE[key]
    return stack


def _canonical_qvq_v2b2_pair_stacks(
    *,
    device: torch.device,
    bits: float,
    codebook_version: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    """Return stable canonical/alternative stacks for all three B2 searches."""

    key = (str(device), float(bits), codebook_version, dtype)
    with _QVQ_CODEBOOK_CACHE_LOCK:
        cached = _QVQ_V2B2_PAIR_STACK_CACHE.get(key)
    if cached is not None:
        return cached
    banks = _canonical_qvq_v2b4_banks(
        device=device,
        bits=bits,
        codebook_version=codebook_version,
        dtype=dtype,
    )
    pair_stacks = tuple(torch.stack((banks[0], banks[alt_id])).contiguous() for alt_id in range(1, 4))
    with _QVQ_CODEBOOK_CACHE_LOCK:
        return _QVQ_V2B2_PAIR_STACK_CACHE.setdefault(key, pair_stacks)


def _canonical_qvq_v4_bank_stack(
    *,
    device: torch.device,
    bits: float,
    codebook_version: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the globally shared contiguous bank-major V4 codebook tensor."""

    key = (str(device), float(bits), codebook_version, dtype)
    with _QVQ_CODEBOOK_CACHE_LOCK:
        stack = _QVQ_V4_BANK_STACK_CACHE.get(key)
    if stack is None:
        _canonical_qvq_v4_banks(
            device=device,
            bits=bits,
            codebook_version=codebook_version,
            dtype=dtype,
        )
        with _QVQ_CODEBOOK_CACHE_LOCK:
            stack = _QVQ_V4_BANK_STACK_CACHE[key]
    return stack


@dataclass(frozen=True)
class TrellisQuantizationResult:
    """Minimum-distortion path returned by :func:`viterbi_quantize`."""

    states: torch.Tensor
    values: torch.Tensor
    squared_error: torch.Tensor


@dataclass(frozen=True)
class BankedTrellisQuantizationResult(TrellisQuantizationResult):
    """V2 path plus one selected decoder bank for each fixed-size segment."""

    segment_bank_ids: torch.Tensor


@dataclass(frozen=True)
class QVQLinearQuantizationResult:
    """Pack-ready tensors and dense reconstruction for one quantized linear."""

    trellis: torch.Tensor
    SU: torch.Tensor
    SV: torch.Tensor
    bias: torch.Tensor | None
    inner_weight: torch.Tensor
    weight: torch.Tensor
    proxy_loss: torch.Tensor
    baseline_proxy_loss: torch.Tensor
    output_scale_optimized_channels: int
    hessian_viterbi_selected: bool
    hessian_viterbi_candidate_relative_improvement: float | None
    rounding: str = "block_ldlq"
    kronecker_proxy_loss: torch.Tensor | None = None
    module_scale_search_selected: bool = False
    module_scale_multiplier: float = 1.0
    module_scale_reencoded: bool = False
    telemetry: dict[str, object] | None = None
    serialization_allowed: bool = True
    bank_ids: torch.Tensor | None = None
    bank_selector_bits: int = 2
    bank_alt_id: torch.Tensor | None = None
    yaqa_bank_fallback_to_v2: bool | None = None
    yaqa_selector_churn: float | None = None
    yaqa_family_changed: bool | None = None
    yaqa_block_family_id: int | None = None
    yaqa_spectral_selected: bool | None = None
    yaqa_spectral_method: str | None = None
    yaqa_spectral_rank: int | None = None
    yaqa_spectral_lambda: float | None = None
    yaqa_spectral_alpha: float | None = None
    yaqa_spectral_svd_device: str | None = None
    yaqa_spectral_concentration: dict[str, float] | None = None
    yaqa_spectral_oracle_losses: dict[str, float] | None = None
    yaqa_spectral_candidates: dict[str, dict[str, object]] | None = None
    yaqa_spectral_absorption_efficiency: float | None = None
    yaqa_spectral_selector_churn: float | None = None
    yaqa_spectral_family_changed: bool | None = None

    def serialized_tensors(self) -> dict[str, torch.Tensor]:
        """Return checkpoint tensors, rejecting research-only codebooks."""

        if not self.serialization_allowed:
            raise RuntimeError(
                "QVQ results produced with an experimental codebook are evaluation-only and cannot be serialized."
            )
        tensors = {"trellis": self.trellis, "SU": self.SU, "SV": self.SV}
        if self.bias is not None:
            tensors["bias"] = self.bias
        if self.bank_ids is not None:
            tensors["bank_ids"] = (
                pack_qvq_binary_bank_ids(self.bank_ids)
                if self.bank_selector_bits == 1
                else pack_qvq_bank_ids(self.bank_ids)
            )
        if self.bank_alt_id is not None:
            tensors["bank_alt_id"] = self.bank_alt_id
        return tensors


@dataclass(frozen=True)
class QVQSerializedCandidate:
    """One exact, serialized QVQ alternative for deferred model refinement.

    ``decoded_weight`` is retained for diagnostics only.  Installers must use
    ``serialized_tensors()`` (trellis, SU, SV, and packed bank selectors) so a
    candidate is evaluated exactly as it will reload from a checkpoint.
    """

    module_name: str
    trellis: torch.Tensor
    packed_bank_ids: torch.Tensor | None
    SU: torch.Tensor
    SV: torch.Tensor
    decoded_weight: torch.Tensor | None = None
    format: str = "qvq_v4"
    bits: float = 2.0
    vector_size: int = 4
    bank_count: int = 4
    codebook_version: str = PGC16_CODEBOOK_VERSION
    local_metrics: dict[str, float] = field(default_factory=dict)
    is_noop: bool = False
    _artifact_hash: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name in ("trellis", "SU", "SV"):
            tensor = getattr(self, name)
            if not isinstance(tensor, torch.Tensor) or not tensor.is_contiguous() or tensor.device.type != "cpu":
                raise ValueError(f"{name} must be a contiguous CPU tensor snapshot")
        if self.packed_bank_ids is not None and (
            self.packed_bank_ids.device.type != "cpu"
            or self.packed_bank_ids.dtype != torch.uint8
            or not self.packed_bank_ids.is_contiguous()
        ):
            raise ValueError("packed_bank_ids must be a contiguous CPU uint8 snapshot")
        if self.vector_size != 4 or self.bank_count != 4:
            raise ValueError("serialized bank candidates currently require V4 with four banks")
        if self.bits < 1 or self.bits > 4:
            raise ValueError("serialized candidate bits must be in [1, 4]")
        digest = hashlib.sha256()
        for tensor in (self.trellis, self.SU, self.SV, self.packed_bank_ids):
            if tensor is not None:
                digest.update(tensor.numpy().tobytes())
        object.__setattr__(self, "_artifact_hash", digest.hexdigest())

    @classmethod
    def from_result(
        cls,
        module_name: str,
        result: "QVQLinearQuantizationResult",
        *,
        bits: float,
        local_metrics=None,
        retain_decoded_weight: bool = False,
    ):
        serialized = result.serialized_tensors()
        return cls(
            module_name=module_name,
            trellis=serialized["trellis"].detach().to(device="cpu", copy=True).contiguous(),
            packed_bank_ids=None
            if result.bank_ids is None
            else pack_qvq_bank_ids(result.bank_ids).detach().to(device="cpu", copy=True).contiguous(),
            SU=serialized["SU"].detach().to(device="cpu", copy=True).contiguous(),
            SV=serialized["SV"].detach().to(device="cpu", copy=True).contiguous(),
            decoded_weight=(
                result.weight.detach().to(device="cpu", copy=True).contiguous() if retain_decoded_weight else None
            ),
            local_metrics={} if local_metrics is None else dict(local_metrics),
            bits=float(bits),
        )

    @property
    def artifact_hash(self) -> str:
        return self._artifact_hash

    def serialized_tensors(self) -> dict[str, torch.Tensor]:
        tensors = {"trellis": self.trellis.clone(), "SU": self.SU.clone(), "SV": self.SV.clone()}
        if self.packed_bank_ids is not None:
            tensors["bank_ids"] = self.packed_bank_ids.clone()
        return tensors


QVQCandidate = QVQSerializedCandidate


class QVQPropagationRefiner:
    """Greedy post-quantization selector for exact full-model candidates.

    Candidate generation happens during baseline quantization, but selection is
    deliberately separate: ``install`` must load the serialized candidate into
    the complete quantized model before ``evaluate`` runs.  ``evaluate`` returns
    a lower-is-better final-logit/task loss.  Rejected candidates are restored
    atomically through ``restore``.
    """

    def __init__(
        self,
        evaluate: Callable[[], float],
        install: Callable[[QVQCandidate], None],
        snapshot: Callable[[], object],
        restore: Callable[[object], None],
    ) -> None:
        self._evaluate = evaluate
        self._install = install
        self._snapshot = snapshot
        self._restore = restore

    def refine(
        self,
        candidates_by_module: dict[str, list[QVQCandidate]],
        *,
        baseline_score: float | None = None,
        max_sweeps: int = 1,
    ) -> tuple[float, list[QVQCandidate]]:
        """Select candidates by conditional full-model improvement.

        Every candidate is installed from its serialized tensors before
        evaluation.  A rejected candidate is rolled back before the next one;
        accepted candidates become the baseline for subsequent modules.
        """

        if max_sweeps < 1:
            raise ValueError("max_sweeps must be positive")
        score = float(self._evaluate() if baseline_score is None else baseline_score)
        if not math.isfinite(score):
            raise ValueError("baseline propagation score must be finite")
        selected_by_module: dict[str, QVQCandidate] = {}
        module_items = tuple(candidates_by_module.items())
        for _ in range(max_sweeps):
            changed = False
            for module_name, candidates in module_items:
                for candidate in candidates:
                    if candidate.module_name != module_name:
                        raise ValueError("candidate module_name does not match its candidate group")
                    if candidate.is_noop:
                        continue
                    checkpoint = self._snapshot()
                    try:
                        # The installer is required to consume
                        # candidate.serialized_tensors(), not decoded_weight.
                        self._install(candidate)
                        candidate_score = float(self._evaluate())
                    except BaseException:
                        self._restore(checkpoint)
                        raise
                    if math.isfinite(candidate_score) and candidate_score < score:
                        score = candidate_score
                        selected_by_module[module_name] = candidate
                        changed = True
                    else:
                        self._restore(checkpoint)
            if not changed:
                break
        return score, list(selected_by_module.values())


def pack_qvq_bank_ids(bank_ids: torch.Tensor) -> torch.Tensor:
    """Pack four generic QVQ 2-bit bank selectors into each byte."""

    if bank_ids.ndim != 1 or bank_ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("QVQ bank selectors must be a one-dimensional integer tensor.")
    values = bank_ids.to(torch.int64)
    if torch.any((values < 0) | (values >= 4)):
        raise ValueError("QVQ bank selectors must be in [0, 3].")
    pad = (-values.numel()) % 4
    if pad:
        values = torch.cat((values, torch.zeros(pad, device=values.device, dtype=torch.int64)))
    values = values.reshape(-1, 4)
    shifts = torch.arange(4, device=values.device, dtype=torch.int64) * 2
    return (values << shifts).sum(dim=1).to(torch.uint8).contiguous()


def unpack_qvq_bank_ids(packed: torch.Tensor, tile_count: int) -> torch.Tensor:
    """Unpack dense or four-per-byte selectors to ``tile_count`` entries."""

    if packed.ndim != 1 or packed.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("QVQ bank selectors must be a one-dimensional integer tensor.")
    if tile_count < 1:
        raise ValueError("QVQ tile_count must be positive.")
    if packed.numel() == tile_count:
        dense = packed.to(torch.int64)
    elif packed.numel() == (tile_count + 3) // 4:
        values = packed.to(torch.int64).unsqueeze(1)
        shifts = torch.arange(4, device=packed.device, dtype=torch.int64) * 2
        dense = ((values >> shifts) & 3).reshape(-1)[:tile_count]
    else:
        raise ValueError("QVQ bank selector payload has an invalid tile count.")
    if torch.any((dense < 0) | (dense >= 4)):
        raise ValueError("QVQ bank selectors must be in [0, 3].")
    return dense.to(torch.uint8).contiguous()


def pack_qvq_binary_bank_ids(bank_ids: torch.Tensor) -> torch.Tensor:
    """Pack eight B2 selectors into each byte."""

    if bank_ids.ndim != 1 or bank_ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("QVQ binary bank selectors must be a one-dimensional integer tensor.")
    values = bank_ids.to(torch.int64)
    if torch.any((values < 0) | (values >= 2)):
        raise ValueError("QVQ binary bank selectors must be in [0, 1].")
    pad = (-values.numel()) % 8
    if pad:
        values = torch.cat((values, torch.zeros(pad, device=values.device, dtype=torch.int64)))
    values = values.reshape(-1, 8)
    shifts = torch.arange(8, device=values.device, dtype=torch.int64)
    return (values << shifts).sum(dim=1).to(torch.uint8).contiguous()


def unpack_qvq_binary_bank_ids(packed: torch.Tensor, selector_count: int) -> torch.Tensor:
    """Unpack dense or eight-per-byte B2 selectors."""

    if packed.ndim != 1 or packed.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("QVQ binary bank selectors must be a one-dimensional integer tensor.")
    if selector_count < 1:
        raise ValueError("QVQ selector_count must be positive.")
    if packed.numel() == selector_count:
        dense = packed.to(torch.int64)
    elif packed.numel() == (selector_count + 7) // 8:
        values = packed.to(torch.int64).unsqueeze(1)
        shifts = torch.arange(8, device=packed.device, dtype=torch.int64)
        dense = ((values >> shifts) & 1).reshape(-1)[:selector_count]
    else:
        raise ValueError("QVQ binary bank selector payload has an invalid selector count.")
    if torch.any((dense < 0) | (dense >= 2)):
        raise ValueError("QVQ binary bank selectors must be in [0, 1].")
    return dense.to(torch.uint8).contiguous()


@dataclass
class QVQQuantizationTelemetry:
    """Opt-in host/CUDA phase telemetry for one QVQ module quantization."""

    host_dispatch_seconds: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _cuda_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = field(default_factory=list, repr=False)
    _device: torch.device | None = field(default=None, repr=False)
    _finalized: bool = field(default=False, repr=False)

    @contextmanager
    def phase(self, name: str, device: torch.device) -> Iterator[None]:
        """Record host dispatch and CUDA stream time without a per-phase sync."""

        if self._finalized:
            raise RuntimeError("QVQ telemetry cannot record phases after finalization.")
        start_event = end_event = None
        if device.type == "cuda":
            if self._device is not None and self._device != device:
                raise ValueError("QVQ telemetry cannot span multiple CUDA devices.")
            self._device = device
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        started = time.perf_counter()
        try:
            yield
        finally:
            self.host_dispatch_seconds[name] += time.perf_counter() - started
            self.calls[name] += 1
            if end_event is not None:
                end_event.record()
                self._cuda_events.append((name, start_event, end_event))

    def count(self, name: str, value: int = 1) -> None:
        if self._finalized:
            raise RuntimeError("QVQ telemetry cannot record counters after finalization.")
        self.counters[name] += value

    def finalize(self) -> dict[str, object]:
        """Synchronize once and return stable aggregate telemetry."""

        if self._finalized:
            raise RuntimeError("QVQ telemetry may be finalized only once.")
        self._finalized = True
        gpu_ms: dict[str, float] = defaultdict(float)
        if self._device is not None:
            torch.cuda.synchronize(self._device)
            for name, start, end in self._cuda_events:
                gpu_ms[name] += start.elapsed_time(end)
        phases = {
            name: {
                "calls": self.calls[name],
                "host_dispatch_ms": self.host_dispatch_seconds[name] * 1000.0,
                "gpu_ms": gpu_ms.get(name),
            }
            for name in self.calls
        }
        result: dict[str, object] = {"phases": phases, "counters": dict(self.counters)}
        if self._device is not None and self._device.type == "cuda":
            from ..utils.qvq_cuda import qvq_cuda_norm_rank_telemetry_snapshot

            result["viterbi_pruning"] = qvq_cuda_norm_rank_telemetry_snapshot(self._device)
        return result


def _qvq_phase(
    telemetry: QVQQuantizationTelemetry | None,
    name: str,
    device: torch.device,
):
    return nullcontext() if telemetry is None else telemetry.phase(name, device)


_QVQ_MPS_TRELLIS_BATCH_SIZES = {
    # The persistent low-rate Metal recurrence reaches its throughput knee at
    # 96 independent trellises. Quantization workspace is deliberately larger
    # than the former eager batch-16 path; checkpoint and inference storage do
    # not change.
    1: 96,
    1.5: 96,
    2: 32,
    2.5: 32,
    3: 32,
    3.5: 32,
    4: 64,
    4.5: 64,
    5: 64,
    5.5: 64,
    6: 64,
    6.5: 64,
    7: 256,
    7.5: 256,
    8: 512,
}
_QVQ_CUDA_TRELLIS_BATCH_SIZES = {
    # Lossless byte backpointers keep four 124-block waves near 1.2 GiB while
    # reaching the measured low-rate throughput knee on the local sm_80 host.
    1: 496,
    # W1.5 uses half the transient backpointer storage of W1. Four waves reach
    # its throughput knee at approximately the same 2-GiB peak allocation.
    1.5: 496,
    2: 512,
    # Multiples of 496 align four independent waves to this host's 124 sm_80
    # SMs. Rate-specific knees below were measured with exact PGC16 paths and
    # keep transient quantization workspace at or below 2 GiB.
    2.5: 3968,
    3: 3968,
    3.5: 3968,
    4: 3968,
    4.5: 3968,
    5: 3968,
    5.5: 3968,
    6: 3968,
    6.5: 3968,
    7: 3968,
    7.5: 3968,
    # W8 carries only one scalar recurrence per tile and needs no cost or
    # backpointer workspace, so a larger launch reaches its throughput knee.
    8: 2976,
}


def default_qvq_trellis_batch_size(
    bits: float,
    device: torch.device | str,
    *,
    trellis_window: int = 16,
) -> int:
    """Return a backend/rate/state-width batch size for reference quantization.

    The measured backend tables are L16 baselines.  Transient Viterbi costs
    and traceback state scale with ``2**L``, so preserve approximately the
    same workspace budget by dividing the batch by ``2**(L - 16)``.
    """

    bits = normalize_qvq_rate(bits)
    if isinstance(trellis_window, bool) or not isinstance(trellis_window, int):
        raise TypeError("QVQ trellis window must be an integer.")
    if trellis_window not in (16, 18):
        raise ValueError("QVQ default batch policy supports trellis windows 16 and 18.")
    if trellis_window == 18 and bits > 2.5:
        raise ValueError("QVQ L18 default batch policy supports only rates W1 through W2.5.")
    target_device = torch.device(device)
    if target_device.type == "mps":
        l16_batch_size = _QVQ_MPS_TRELLIS_BATCH_SIZES[bits]
    elif target_device.type == "cuda":
        l16_batch_size = _QVQ_CUDA_TRELLIS_BATCH_SIZES[bits]
    else:
        l16_batch_size = 16
    return max(1, l16_batch_size // (1 << (trellis_window - 16)))


def bitshift_next_state(
    state: torch.Tensor,
    edge: torch.Tensor,
    *,
    bits: float,
    vector_size: int,
    trellis_window: int,
) -> torch.Tensor:
    """Advance an ``L``-bit state by the QVQ bitshift-trellis rule.

    A transition shifts the state left by the exact integer transition width
    ``rate * vector_size`` and appends one edge value of that width. Inputs may
    be broadcast-compatible tensors.
    """

    shift = _validate_trellis_shape(
        bits=bits,
        vector_size=vector_size,
        trellis_window=trellis_window,
    )
    state_mask = (1 << trellis_window) - 1
    edge_mask = (1 << shift) - 1
    state_i64 = state.to(torch.int64)
    edge_i64 = edge.to(device=state.device, dtype=torch.int64)
    if torch.any((state_i64 < 0) | (state_i64 > state_mask)):
        raise ValueError(f"QVQ trellis states must be in `[0, {state_mask}]`.")
    if torch.any((edge_i64 < 0) | (edge_i64 > edge_mask)):
        raise ValueError(f"QVQ edge values must be in `[0, {edge_mask}]`.")
    return ((state_i64 << shift) & state_mask) | edge_i64


def pack_trellis_states(
    states: torch.Tensor,
    *,
    bits: float,
    vector_size: int = 2,
    trellis_window: int = 16,
) -> torch.Tensor:
    """Pack a tail-biting path into Pangolin-style planar int32 words.

    The low ``rate * vector_size`` bits of each state are one logical transition
    edge. Every edge uses one planar code, including odd transition widths for
    half-step rates. The circular stream has no initial-state overhead.
    """

    shift = _validate_trellis_shape(
        bits=bits,
        vector_size=vector_size,
        trellis_window=trellis_window,
    )
    if states.ndim < 1 or states.shape[-1] < 1:
        raise ValueError("QVQ states must have a non-empty final dimension.")
    edge_count = states.shape[-1]
    if edge_count % 32:
        raise ValueError("QVQ planar trellis streams must contain a whole number of 32-edge blocks.")

    state_mask = (1 << trellis_window) - 1
    states_i64 = states.to(torch.int64)
    if torch.any((states_i64 < 0) | (states_i64 > state_mask)):
        raise ValueError(f"QVQ trellis states must be in `[0, {state_mask}]`.")

    edge_mask = (1 << shift) - 1
    edges = states_i64 & edge_mask
    columns = edges.reshape(-1, edge_count).transpose(0, 1).contiguous()
    packed = planar_pack_rows(columns, shift)
    packed = packed.transpose(0, 1).reshape(*states.shape[:-1], -1).contiguous()
    recovered = unpack_trellis_states(
        packed,
        bits=bits,
        vector_size=vector_size,
        trellis_window=trellis_window,
    )
    if not torch.equal(recovered, states_i64):
        raise ValueError("QVQ states must form one transition-consistent tail-biting path.")
    return packed


def unpack_trellis_states(
    trellis: torch.Tensor,
    *,
    bits: float,
    vector_size: int = 2,
    trellis_window: int = 16,
) -> torch.Tensor:
    """Recover every tail-biting state from planar int32 words."""

    shift = _validate_trellis_shape(
        bits=bits,
        vector_size=vector_size,
        trellis_window=trellis_window,
    )
    if trellis.dtype != torch.int32:
        raise TypeError(f"QVQ planar trellis words must use torch.int32, got {trellis.dtype}.")
    if trellis.ndim < 1 or trellis.shape[-1] < 1:
        raise ValueError("QVQ trellis must have a non-empty final dimension.")

    if trellis.shape[-1] % shift:
        raise ValueError("QVQ planar trellis word count must be divisible by the transition width.")
    edge_count = (trellis.shape[-1] // shift) * 32
    columns = trellis.reshape(-1, trellis.shape[-1]).transpose(0, 1).contiguous()
    edges = planar_unpack_rows(columns, shift).transpose(0, 1)
    edges = edges.reshape(*trellis.shape[:-1], edge_count).to(torch.int64)
    total_bits = edge_count * shift
    steps = edge_count
    step_ids = torch.arange(steps, dtype=torch.int64, device=trellis.device)
    state_bits = torch.arange(trellis_window, dtype=torch.int64, device=trellis.device)
    offsets = ((step_ids[:, None] + 1) * shift - trellis_window + state_bits[None, :]) % total_bits
    selected = edges[..., offsets // shift]
    bits_i64 = (selected >> (shift - 1 - offsets % shift)) & 1
    state_shifts = torch.arange(trellis_window - 1, -1, -1, dtype=torch.int64, device=trellis.device)
    return (bits_i64.to(torch.int64) << state_shifts).sum(dim=-1).contiguous()


_QVQ_P32_TILE_EDGES = QVQ_V2B2_P32_SEGMENTS_PER_TILE * QVQ_V2B2_P32_STEPS_PER_SEGMENT
_QVQ_UINT32_MASK = (1 << 32) - 1


def _validate_p32_window_words(trellis: torch.Tensor, *, bits: float, layout: str) -> tuple[int, int]:
    """Validate one canonical or continuous-window P32 payload tensor."""

    transition_bits = _validate_trellis_shape(bits=bits, vector_size=2, trellis_window=16)
    if transition_bits > 7:
        raise ValueError("QVQ P32 continuous-window layout supports only rates W1 through W3.5.")
    if trellis.dtype != torch.int32:
        raise TypeError(f"QVQ P32 {layout} words must use torch.int32, got {trellis.dtype}.")
    if trellis.ndim < 1:
        raise ValueError(f"QVQ P32 {layout} words must have at least one dimension.")
    expected_words = qvq_words_per_tile(bits, weight_count=256, vector_size=2)
    if trellis.shape[-1] != expected_words:
        raise ValueError(
            f"QVQ P32 {layout} payload must contain {expected_words} words per 256-weight tile, "
            f"got shape {tuple(trellis.shape)}."
        )
    return transition_bits, expected_words


def repack_p32_planar_to_window(trellis: torch.Tensor, *, bits: float) -> torch.Tensor:
    """Losslessly repack canonical planar P32 into direct state-window words.

    The logical P32 path remains one circular 128-transition history. Physical
    transitions are stored in reverse chronological order, low bits first, so
    every logical 16-bit state is one circular 16-bit window beginning at the
    current transition. The payload has exactly the same word count as planar
    P32 and carries no initial-state or segment-boundary trailer.

    Non-contiguous inputs are accepted at this high-level boundary; the
    returned inference payload is always contiguous for low-level consumers.
    """

    transition_bits, words_per_tile = _validate_p32_window_words(trellis, bits=bits, layout="planar")
    flat = trellis.reshape(-1, words_per_tile).contiguous()
    columns = flat.transpose(0, 1).contiguous()
    edges = planar_unpack_rows(columns, transition_bits).transpose(0, 1).to(torch.int64)
    reversed_edges = edges.flip(1)

    bit_positions = torch.arange(_QVQ_P32_TILE_EDGES, device=trellis.device, dtype=torch.int64) * transition_bits
    word_ids = bit_positions >> 5
    shifts = bit_positions & 31
    packed = torch.zeros((flat.shape[0], words_per_tile), device=trellis.device, dtype=torch.int64)
    expanded_word_ids = word_ids.unsqueeze(0).expand(flat.shape[0], -1)
    packed.scatter_add_(1, expanded_word_ids, (reversed_edges << shifts) & _QVQ_UINT32_MASK)

    crosses_word = shifts + transition_bits > 32
    crossing_word_ids = word_ids[crosses_word] + 1
    crossing_shifts = 32 - shifts[crosses_word]
    packed.scatter_add_(
        1,
        crossing_word_ids.unsqueeze(0).expand(flat.shape[0], -1),
        reversed_edges[:, crosses_word] >> crossing_shifts,
    )
    return (packed & _QVQ_UINT32_MASK).to(torch.int32).reshape(trellis.shape).contiguous()


def repack_p32_window_to_planar(window_words: torch.Tensor, *, bits: float) -> torch.Tensor:
    """Restore canonical planar P32 words from the continuous-window layout."""

    transition_bits, words_per_tile = _validate_p32_window_words(
        window_words,
        bits=bits,
        layout="continuous-window",
    )
    flat = window_words.reshape(-1, words_per_tile).contiguous().to(torch.int64) & _QVQ_UINT32_MASK
    bit_positions = torch.arange(_QVQ_P32_TILE_EDGES, device=window_words.device, dtype=torch.int64) * transition_bits
    word_ids = bit_positions >> 5
    shifts = bit_positions & 31
    reversed_edges = flat[:, word_ids] >> shifts

    crosses_word = shifts + transition_bits > 32
    crossing_word_ids = word_ids[crosses_word] + 1
    crossing_shifts = 32 - shifts[crosses_word]
    reversed_edges[:, crosses_word] |= flat[:, crossing_word_ids] << crossing_shifts
    edges = (reversed_edges & ((1 << transition_bits) - 1)).flip(1)
    packed = planar_pack_rows(edges.transpose(0, 1).contiguous(), transition_bits)
    return packed.transpose(0, 1).reshape(window_words.shape).contiguous()


def unpack_p32_window_states(window_words: torch.Tensor, *, bits: float) -> torch.Tensor:
    """Decode every standard P32 state directly from circular bit windows."""

    transition_bits, words_per_tile = _validate_p32_window_words(
        window_words,
        bits=bits,
        layout="continuous-window",
    )
    flat = window_words.reshape(-1, words_per_tile).contiguous().to(torch.int64) & _QVQ_UINT32_MASK
    pair_ids = torch.arange(_QVQ_P32_TILE_EDGES, device=window_words.device, dtype=torch.int64)
    bit_positions = (_QVQ_P32_TILE_EDGES - 1 - pair_ids) * transition_bits
    word_ids = bit_positions >> 5
    shifts = bit_positions & 31
    states = flat[:, word_ids] >> shifts

    crosses_word = shifts > 16
    next_word_ids = (word_ids[crosses_word] + 1) % words_per_tile
    states[:, crosses_word] |= flat[:, next_word_ids] << (32 - shifts[crosses_word])
    return (states & 0xffff).reshape(*window_words.shape[:-1], _QVQ_P32_TILE_EDGES).contiguous()


def decode_p32_window_tiles(
    window_words: torch.Tensor,
    *,
    bits: float,
    bank_ids: torch.Tensor,
    bank_alt_id: torch.Tensor,
    codebook_version: str = PGC16_CODEBOOK_VERSION,
) -> torch.Tensor:
    """Reference-decode continuous-window P32 tiles to 256 row-major values."""

    states = unpack_p32_window_states(window_words, bits=bits)
    if bank_ids.device != window_words.device or bank_alt_id.device != window_words.device:
        raise ValueError("QVQ P32 continuous-window payload and bank metadata must share one device.")
    if bank_alt_id.numel() != 1 or bank_alt_id.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ValueError("QVQ P32 continuous-window alternative bank ID must be one integer scalar.")
    alt_id = int(bank_alt_id.item())
    if not 1 <= alt_id < 4:
        raise ValueError("QVQ P32 continuous-window alternative bank ID must be in [1, 3].")

    tile_count = states.numel() // _QVQ_P32_TILE_EDGES
    binary_ids = unpack_qvq_binary_bank_ids(
        bank_ids,
        tile_count * QVQ_V2B2_P32_SEGMENTS_PER_TILE,
    )
    state_bank_ids = (
        binary_ids.reshape(tile_count, QVQ_V2B2_P32_SEGMENTS_PER_TILE)
        .repeat_interleave(QVQ_V2B2_P32_STEPS_PER_SEGMENT, dim=1)
        .mul(alt_id)
    )
    levels = pgc16_levels_for_version(codebook_version).to(device=window_words.device)
    decoded = pgc16_decode_states_v2_banked(
        states.reshape(tile_count, _QVQ_P32_TILE_EDGES),
        state_bank_ids,
        bits=bits,
        levels=levels,
    )
    return decoded.reshape(*window_words.shape[:-1], 256).to(torch.float32).contiguous()


def reconstruct_p32_window_inner_weight(
    window_words: torch.Tensor,
    *,
    bits: float,
    in_features: int,
    out_features: int,
    bank_ids: torch.Tensor,
    bank_alt_id: torch.Tensor,
    codebook_version: str = PGC16_CODEBOOK_VERSION,
) -> torch.Tensor:
    """Materialize the standard K16-by-N16 P32 matrix from window words."""

    if in_features <= 0 or out_features <= 0 or in_features % 16 or out_features % 16:
        raise ValueError("QVQ P32 continuous-window dimensions must be positive and divisible by 16.")
    tile_count = (in_features // 16) * (out_features // 16)
    expected_words = qvq_words_per_tile(bits, weight_count=256, vector_size=2)
    if window_words.ndim != 2 or tuple(window_words.shape) != (tile_count, expected_words):
        raise ValueError(
            f"QVQ P32 continuous-window words must have shape {(tile_count, expected_words)}, "
            f"got {tuple(window_words.shape)}."
        )
    decoded = decode_p32_window_tiles(
        window_words,
        bits=bits,
        bank_ids=bank_ids,
        bank_alt_id=bank_alt_id,
        codebook_version=codebook_version,
    )
    return (
        decoded.view(in_features // 16, out_features // 16, 16, 16)
        .permute(0, 2, 1, 3)
        .reshape(in_features, out_features)
        .contiguous()
    )


_QVQ_P32_ANCHOR4_COUNT = _QVQ_P32_TILE_EDGES // 4


def _validate_p32_anchor4_words(trellis: torch.Tensor, *, bits: float, layout: str) -> tuple[int, int]:
    """Validate one canonical or Anchor-4 P32 payload tensor."""

    transition_bits, words_per_tile = _validate_p32_window_words(trellis, bits=bits, layout=layout)
    if transition_bits < 4:
        raise ValueError("QVQ P32 Anchor-4 layout supports only rates W2 through W3.5.")
    return transition_bits, words_per_tile


def _pack_p32_anchor4_values(anchors: torch.Tensor, *, transition_bits: int) -> torch.Tensor:
    """Pack 32 unsigned ``4 * transition_bits``-wide anchors into int32 words."""

    anchor_bits = 4 * transition_bits
    words_per_tile = anchor_bits
    bit_positions = torch.arange(
        _QVQ_P32_ANCHOR4_COUNT,
        device=anchors.device,
        dtype=torch.int64,
    ) * anchor_bits
    word_ids = bit_positions >> 5
    shifts = bit_positions & 31
    packed = torch.zeros((anchors.shape[0], words_per_tile), device=anchors.device, dtype=torch.int64)
    packed.scatter_add_(
        1,
        word_ids.unsqueeze(0).expand(anchors.shape[0], -1),
        (anchors << shifts) & _QVQ_UINT32_MASK,
    )

    crosses_word = shifts + anchor_bits > 32
    packed.scatter_add_(
        1,
        (word_ids[crosses_word] + 1).unsqueeze(0).expand(anchors.shape[0], -1),
        anchors[:, crosses_word] >> (32 - shifts[crosses_word]),
    )
    return (packed & _QVQ_UINT32_MASK).to(torch.int32)


def _unpack_p32_anchor4_values(anchor_words: torch.Tensor, *, transition_bits: int) -> torch.Tensor:
    """Unpack one row of 32 storage-neutral Anchor-4 records per P32 tile."""

    anchor_bits = 4 * transition_bits
    words_per_tile = anchor_bits
    flat = anchor_words.reshape(-1, words_per_tile).contiguous().to(torch.int64) & _QVQ_UINT32_MASK
    bit_positions = torch.arange(
        _QVQ_P32_ANCHOR4_COUNT,
        device=anchor_words.device,
        dtype=torch.int64,
    ) * anchor_bits
    word_ids = bit_positions >> 5
    shifts = bit_positions & 31
    anchors = flat[:, word_ids] >> shifts

    crosses_word = shifts + anchor_bits > 32
    anchors[:, crosses_word] |= flat[:, word_ids[crosses_word] + 1] << (32 - shifts[crosses_word])
    return anchors & ((1 << anchor_bits) - 1)


def _unpack_p32_anchor4_edges(anchor_words: torch.Tensor, *, transition_bits: int) -> torch.Tensor:
    """Recover the canonical 128-transition circular stream from Anchor-4."""

    anchors = _unpack_p32_anchor4_values(anchor_words, transition_bits=transition_bits)
    history_count = (16 + transition_bits - 1) // transition_bits
    history_bits = history_count * transition_bits
    edge_mask = (1 << transition_bits) - 1
    values = []
    for history_index in range(history_count):
        shift = (history_count - 1 - history_index) * transition_bits
        values.append((anchors >> shift) & edge_mask)
    for extra_index in range(4 - history_count):
        values.append((anchors >> (history_bits + extra_index * transition_bits)) & edge_mask)
    grouped_edges = torch.stack(values, dim=-1)

    base_ids = torch.arange(
        0,
        _QVQ_P32_TILE_EDGES,
        4,
        device=anchor_words.device,
        dtype=torch.int64,
    )
    edge_ids = (base_ids[:, None] - history_count + 1 + torch.arange(4, device=anchor_words.device))
    edge_ids %= _QVQ_P32_TILE_EDGES
    edges = torch.empty(
        (anchors.shape[0], _QVQ_P32_TILE_EDGES),
        device=anchor_words.device,
        dtype=torch.int64,
    )
    edges.scatter_(
        1,
        edge_ids.reshape(1, -1).expand(anchors.shape[0], -1),
        grouped_edges.reshape(anchors.shape[0], -1),
    )
    return edges


def repack_p32_planar_to_anchor4(trellis: torch.Tensor, *, bits: float) -> torch.Tensor:
    """Losslessly repack canonical planar P32 into storage-neutral Anchor-4.

    Each record represents four consecutive states using one 16-bit state,
    the omitted high history bits, and (for W3/W3.5) one following raw
    transition. Its width is exactly four transition codes, so 32 records
    occupy precisely the original P32 payload with no metadata or padding.
    """

    transition_bits, words_per_tile = _validate_p32_anchor4_words(trellis, bits=bits, layout="planar")
    flat = trellis.reshape(-1, words_per_tile).contiguous()
    columns = flat.transpose(0, 1).contiguous()
    edges = planar_unpack_rows(columns, transition_bits).transpose(0, 1).to(torch.int64)

    history_count = (16 + transition_bits - 1) // transition_bits
    base_ids = torch.arange(
        0,
        _QVQ_P32_TILE_EDGES,
        4,
        device=trellis.device,
        dtype=torch.int64,
    )
    anchor_values = torch.zeros(
        (flat.shape[0], _QVQ_P32_ANCHOR4_COUNT),
        device=trellis.device,
        dtype=torch.int64,
    )
    for history_index in range(history_count):
        edge_ids = (base_ids - history_count + 1 + history_index) % _QVQ_P32_TILE_EDGES
        shift = (history_count - 1 - history_index) * transition_bits
        anchor_values |= edges[:, edge_ids] << shift
    history_bits = history_count * transition_bits
    for extra_index in range(4 - history_count):
        edge_ids = (base_ids + 1 + extra_index) % _QVQ_P32_TILE_EDGES
        anchor_values |= edges[:, edge_ids] << (history_bits + extra_index * transition_bits)

    packed = _pack_p32_anchor4_values(anchor_values, transition_bits=transition_bits)
    return packed.reshape(trellis.shape).contiguous()


def repack_p32_anchor4_to_planar(anchor_words: torch.Tensor, *, bits: float) -> torch.Tensor:
    """Restore canonical planar P32 words from storage-neutral Anchor-4."""

    transition_bits, _ = _validate_p32_anchor4_words(anchor_words, bits=bits, layout="Anchor-4")
    edges = _unpack_p32_anchor4_edges(anchor_words, transition_bits=transition_bits)
    packed = planar_pack_rows(edges.transpose(0, 1).contiguous(), transition_bits)
    return packed.transpose(0, 1).reshape(anchor_words.shape).contiguous()


def unpack_p32_anchor4_states(anchor_words: torch.Tensor, *, bits: float) -> torch.Tensor:
    """Directly decode every exact P32 state from four-state anchors."""

    transition_bits, _ = _validate_p32_anchor4_words(
        anchor_words,
        bits=bits,
        layout="Anchor-4",
    )
    anchors = _unpack_p32_anchor4_values(anchor_words, transition_bits=transition_bits)
    edges = _unpack_p32_anchor4_edges(anchor_words, transition_bits=transition_bits)
    history_count = (16 + transition_bits - 1) // transition_bits
    history_mask = (1 << (history_count * transition_bits)) - 1
    base_states = (anchors & history_mask) & 0xffff
    base_ids = torch.arange(
        0,
        _QVQ_P32_TILE_EDGES,
        4,
        device=anchor_words.device,
        dtype=torch.int64,
    )
    states = torch.empty(
        (anchors.shape[0], _QVQ_P32_TILE_EDGES),
        device=anchor_words.device,
        dtype=torch.int64,
    )
    states[:, base_ids] = base_states
    current = base_states
    for step in range(1, 4):
        edge_ids = (base_ids + step) % _QVQ_P32_TILE_EDGES
        current = ((current << transition_bits) & 0xffff) | edges[:, edge_ids]
        states[:, edge_ids] = current
    return states.reshape(*anchor_words.shape[:-1], _QVQ_P32_TILE_EDGES).contiguous()


def decode_p32_anchor4_tiles(
    anchor_words: torch.Tensor,
    *,
    bits: float,
    bank_ids: torch.Tensor,
    bank_alt_id: torch.Tensor,
    codebook_version: str = PGC16_CODEBOOK_VERSION,
) -> torch.Tensor:
    """Reference-decode Anchor-4 P32 tiles to 256 row-major values."""

    states = unpack_p32_anchor4_states(anchor_words, bits=bits)
    if bank_ids.device != anchor_words.device or bank_alt_id.device != anchor_words.device:
        raise ValueError("QVQ P32 Anchor-4 payload and bank metadata must share one device.")
    if bank_alt_id.numel() != 1 or bank_alt_id.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ValueError("QVQ P32 Anchor-4 alternative bank ID must be one integer scalar.")
    alt_id = int(bank_alt_id.item())
    if not 1 <= alt_id < 4:
        raise ValueError("QVQ P32 Anchor-4 alternative bank ID must be in [1, 3].")

    tile_count = states.numel() // _QVQ_P32_TILE_EDGES
    binary_ids = unpack_qvq_binary_bank_ids(
        bank_ids,
        tile_count * QVQ_V2B2_P32_SEGMENTS_PER_TILE,
    )
    state_bank_ids = (
        binary_ids.reshape(tile_count, QVQ_V2B2_P32_SEGMENTS_PER_TILE)
        .repeat_interleave(QVQ_V2B2_P32_STEPS_PER_SEGMENT, dim=1)
        .mul(alt_id)
    )
    levels = pgc16_levels_for_version(codebook_version).to(device=anchor_words.device)
    decoded = pgc16_decode_states_v2_banked(
        states.reshape(tile_count, _QVQ_P32_TILE_EDGES),
        state_bank_ids,
        bits=bits,
        levels=levels,
    )
    return decoded.reshape(*anchor_words.shape[:-1], 256).to(torch.float32).contiguous()


def reconstruct_p32_anchor4_inner_weight(
    anchor_words: torch.Tensor,
    *,
    bits: float,
    in_features: int,
    out_features: int,
    bank_ids: torch.Tensor,
    bank_alt_id: torch.Tensor,
    codebook_version: str = PGC16_CODEBOOK_VERSION,
) -> torch.Tensor:
    """Materialize the exact K16-by-N16 P32 matrix from Anchor-4 words."""

    if in_features <= 0 or out_features <= 0 or in_features % 16 or out_features % 16:
        raise ValueError("QVQ P32 Anchor-4 dimensions must be positive and divisible by 16.")
    tile_count = (in_features // 16) * (out_features // 16)
    expected_words = qvq_words_per_tile(bits, weight_count=256, vector_size=2)
    if anchor_words.ndim != 2 or tuple(anchor_words.shape) != (tile_count, expected_words):
        raise ValueError(
            f"QVQ P32 Anchor-4 words must have shape {(tile_count, expected_words)}, "
            f"got {tuple(anchor_words.shape)}."
        )
    decoded = decode_p32_anchor4_tiles(
        anchor_words,
        bits=bits,
        bank_ids=bank_ids,
        bank_alt_id=bank_alt_id,
        codebook_version=codebook_version,
    )
    return (
        decoded.view(in_features // 16, out_features // 16, 16, 16)
        .permute(0, 2, 1, 3)
        .reshape(in_features, out_features)
        .contiguous()
    )


def unpack_local_ring_edges(
    trellis: torch.Tensor,
    *,
    bits: float,
) -> torch.Tensor:
    """Unpack an LR32 planar tile into ``[..., 8, 16]`` local-ring edges.

    LR32 deliberately retains the P32 planar byte layout.  The only semantic
    difference is that the 128 edge stream is interpreted as eight adjacent
    16-edge rings instead of one 128-edge circular stream.
    """

    shift = qvq_transition_bits(bits, vector_size=2)
    if trellis.dtype != torch.int32:
        raise TypeError(f"QVQ LR32 planar trellis words must use torch.int32, got {trellis.dtype}.")
    if trellis.ndim < 1 or trellis.shape[-1] != qvq_words_per_tile(bits, weight_count=256, vector_size=2):
        raise ValueError("QVQ LR32 trellis must contain the canonical 256-weight tile payload.")
    columns = trellis.reshape(-1, trellis.shape[-1]).transpose(0, 1).contiguous()
    edges = planar_unpack_rows(columns, shift).transpose(0, 1)
    edges = edges.reshape(*trellis.shape[:-1], 128).to(torch.int64)
    return edges.reshape(*trellis.shape[:-1], QVQ_V2B2_P32_LR_RINGS_PER_TILE, QVQ_V2B2_P32_LR_RING_STEPS)


def local_ring_states_from_edges(
    edges: torch.Tensor,
    *,
    bits: float,
) -> torch.Tensor:
    """Build L16 states independently for each LR32 ring."""

    shift = qvq_transition_bits(bits, vector_size=2)
    if edges.ndim < 2 or tuple(edges.shape[-2:]) != (
        QVQ_V2B2_P32_LR_RINGS_PER_TILE,
        QVQ_V2B2_P32_LR_RING_STEPS,
    ):
        raise ValueError("QVQ LR32 edges must have shape [..., 8, 16].")
    if edges.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError("QVQ LR32 edges must use an integer dtype.")
    edge_mask = (1 << shift) - 1
    edges_i64 = edges.to(torch.int64)
    if torch.any((edges_i64 < 0) | (edges_i64 > edge_mask)):
        raise ValueError(f"QVQ LR32 edges must be in [0, {edge_mask}].")
    states = _states_from_circular_edges(
        edges_i64.reshape(-1, QVQ_V2B2_P32_LR_RING_STEPS),
        shift=shift,
        trellis_window=16,
    )
    return states.reshape(*edges.shape[:-2], QVQ_V2B2_P32_LR_RINGS_PER_TILE, QVQ_V2B2_P32_LR_RING_STEPS)


def pack_local_ring_states(states: torch.Tensor, *, bits: float) -> torch.Tensor:
    """Pack ``[..., 8, 16]`` LR32 states into the canonical planar words."""

    shift = qvq_transition_bits(bits, vector_size=2)
    if states.ndim < 2 or tuple(states.shape[-2:]) != (
        QVQ_V2B2_P32_LR_RINGS_PER_TILE,
        QVQ_V2B2_P32_LR_RING_STEPS,
    ):
        raise ValueError("QVQ LR32 states must have shape [..., 8, 16].")
    state_mask = (1 << 16) - 1
    states_i64 = states.to(torch.int64)
    if torch.any((states_i64 < 0) | (states_i64 > state_mask)):
        raise ValueError("QVQ LR32 states must be in [0, 65535].")
    edges = states_i64 & ((1 << shift) - 1)
    flat_edges = edges.reshape(-1, 128)
    packed = planar_pack_rows(flat_edges.transpose(0, 1).contiguous(), shift)
    packed = packed.transpose(0, 1).reshape(*states.shape[:-2], -1).contiguous()
    recovered_edges = unpack_local_ring_edges(packed, bits=bits)
    recovered_states = local_ring_states_from_edges(recovered_edges, bits=bits)
    if not torch.equal(recovered_states, states_i64):
        raise ValueError("QVQ LR32 states must form eight transition-consistent local rings.")
    return packed


def unpack_local_ring_states(trellis: torch.Tensor, *, bits: float) -> torch.Tensor:
    """Recover all eight independent L16 state paths from LR32 words."""

    return local_ring_states_from_edges(unpack_local_ring_edges(trellis, bits=bits), bits=bits)


def decode_local_ring_states(trellis: torch.Tensor, *, bits: float) -> torch.Tensor:
    """Public readable alias for the LR32 state decoder."""

    return unpack_local_ring_states(trellis, bits=bits)


def reconstruct_local_ring_inner_weight(
    trellis: torch.Tensor,
    *,
    bits: float,
    in_features: int,
    out_features: int,
    codebook_version: str = PGC16_CODEBOOK_VERSION,
    bank_ids: torch.Tensor | None,
    bank_alt_id: torch.Tensor | None,
) -> torch.Tensor:
    """Decode LR32 tiles into the transformed ``[in_features, out_features]`` matrix.

    LR32 stores eight independent 16-transition rings per serialized tile.  A
    ring decodes to one 32-value output column of the logical K32 x N8 tile;
    this reshape is the part that must remain separate from the legacy P32
    16x16 mapping.
    """

    if (
        isinstance(in_features, bool)
        or not isinstance(in_features, int)
        or isinstance(out_features, bool)
        or not isinstance(out_features, int)
        or in_features <= 0
        or out_features <= 0
        or in_features % QVQ_V2B2_P32_LR_TILE_ROWS
        or out_features % QVQ_V2B2_P32_LR_TILE_COLS
    ):
        raise ValueError("QVQ LR32 in/out features must be positive and divisible by K32/N8.")
    tile_count = (in_features // QVQ_V2B2_P32_LR_TILE_ROWS) * (
        out_features // QVQ_V2B2_P32_LR_TILE_COLS
    )
    expected_words = qvq_words_per_tile(bits, weight_count=256, vector_size=2)
    if trellis.ndim != 2 or tuple(trellis.shape) != (tile_count, expected_words):
        raise ValueError(
            f"QVQ LR32 trellis must have shape `{(tile_count, expected_words)}`, got `{tuple(trellis.shape)}`."
        )
    decoded = decode_trellis_tiles(
        trellis,
        bits=bits,
        vector_size=2,
        trellis_window=16,
        codebook_version=codebook_version,
        bank_ids=bank_ids,
        v2b2_p32_lr=True,
        bank_alt_id=bank_alt_id,
    )
    k_blocks = in_features // QVQ_V2B2_P32_LR_TILE_ROWS
    n_blocks = out_features // QVQ_V2B2_P32_LR_TILE_COLS
    return (
        decoded.reshape(k_blocks, n_blocks, QVQ_V2B2_P32_LR_RINGS_PER_TILE, 32)
        .permute(0, 3, 1, 2)
        .reshape(in_features, out_features)
        .contiguous()
    )


def decode_local_ring_tiles(
    trellis: torch.Tensor,
    *,
    bits: float,
    codebook_version: str = PGC16_CODEBOOK_VERSION,
    bank_ids: torch.Tensor,
    bank_alt_id: torch.Tensor,
) -> torch.Tensor:
    """Decode LR32 tiles to ``[..., 8, 16, 2]`` value pairs."""

    return decode_trellis_tiles(
        trellis,
        bits=bits,
        vector_size=2,
        trellis_window=16,
        codebook_version=codebook_version,
        bank_ids=bank_ids,
        v2b2_p32_lr=True,
        bank_alt_id=bank_alt_id,
    )


def _states_from_circular_edges(edges: torch.Tensor, *, shift: int, trellis_window: int) -> torch.Tensor:
    """Recover circular states from one logical edge stream."""

    edge_count = edges.shape[-1]
    total_bits = edge_count * shift
    step_ids = torch.arange(edge_count, dtype=torch.int64, device=edges.device)
    state_bits = torch.arange(trellis_window, dtype=torch.int64, device=edges.device)
    offsets = ((step_ids[:, None] + 1) * shift - trellis_window + state_bits[None, :]) % total_bits
    selected = edges[..., offsets // shift]
    bits_i64 = (selected >> (shift - 1 - offsets % shift)) & 1
    state_shifts = torch.arange(trellis_window - 1, -1, -1, dtype=torch.int64, device=edges.device)
    return (bits_i64.to(torch.int64) << state_shifts).sum(dim=-1).contiguous()


def unpack_dual_v2_states(trellis: torch.Tensor, *, bits: float) -> torch.Tensor:
    """Recover two interleaved circular L16/V2 state paths from one planar tile."""

    shift = qvq_transition_bits(bits, vector_size=2)
    if trellis.dtype != torch.int32:
        raise TypeError(f"QVQ planar trellis words must use torch.int32, got {trellis.dtype}.")
    if trellis.ndim < 1 or trellis.shape[-1] != qvq_words_per_tile(bits, vector_size=2):
        raise ValueError("QVQ Dual-V2 trellis has an invalid planar word count.")
    columns = trellis.reshape(-1, trellis.shape[-1]).transpose(0, 1).contiguous()
    edges = planar_unpack_rows(columns, shift).transpose(0, 1)
    edges = edges.reshape(*trellis.shape[:-1], 128).to(torch.int64)
    first = _states_from_circular_edges(edges[..., 0::2], shift=shift, trellis_window=16)
    second = _states_from_circular_edges(edges[..., 1::2], shift=shift, trellis_window=16)
    states = torch.empty((*edges.shape[:-1], 128), dtype=torch.int64, device=trellis.device)
    states[..., 0::2] = first
    states[..., 1::2] = second
    return states


def pack_dual_v2_states(states: torch.Tensor, *, bits: float) -> torch.Tensor:
    """Pack two interleaved circular L16/V2 paths without adding payload bits."""

    shift = qvq_transition_bits(bits, vector_size=2)
    if states.ndim < 1 or states.shape[-1] != 128:
        raise ValueError("QVQ Dual-V2 state streams must contain exactly 128 interleaved pair states per tile.")
    states_i64 = states.to(torch.int64)
    if torch.any((states_i64 < 0) | (states_i64 >= 1 << 16)):
        raise ValueError("QVQ Dual-V2 states must be in [0, 65535].")
    edges = states_i64 & ((1 << shift) - 1)
    packed = planar_pack_rows(edges.reshape(-1, 128).transpose(0, 1).contiguous(), shift)
    packed = packed.transpose(0, 1).reshape(*states.shape[:-1], -1).contiguous()
    if not torch.equal(unpack_dual_v2_states(packed, bits=bits), states_i64):
        raise ValueError("QVQ Dual-V2 states must form two transition-consistent tail-biting paths.")
    return packed


def decode_trellis_tiles(
    trellis: torch.Tensor,
    *,
    bits: float,
    vector_size: int = 2,
    trellis_window: int = 16,
    codebook_version: str = PGC16_CODEBOOK_VERSION,
    bank_ids: torch.Tensor | None = None,
    dual_v2: bool = False,
    v2b4_p64: bool = False,
    v2b2_p32: bool = False,
    v2b2_p32_lr: bool = False,
    bank_alt_id: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode packed PGC16 QVQ tiles to scalar values in row-major order."""

    if not isinstance(dual_v2, bool):
        raise TypeError("QVQ dual_v2 must be a bool.")
    if not isinstance(v2b4_p64, bool):
        raise TypeError("QVQ v2b4_p64 must be a bool.")
    if not isinstance(v2b2_p32, bool):
        raise TypeError("QVQ v2b2_p32 must be a bool.")
    if not isinstance(v2b2_p32_lr, bool):
        raise TypeError("QVQ v2b2_p32_lr must be a bool.")
    if sum((dual_v2, v2b4_p64, v2b2_p32, v2b2_p32_lr)) > 1:
        raise ValueError("QVQ Dual-V2, V2B4-P64, V2B2-P32, and V2B2-P32-LR are mutually exclusive.")
    if dual_v2 and (vector_size != 2 or trellis_window != 16 or bank_ids is not None):
        raise ValueError("QVQ Dual-V2 requires vector_size=2, trellis_window=16, and no bank_ids.")
    if v2b4_p64 and (vector_size != 2 or trellis_window != 16 or bank_ids is None or bits > 3.5):
        raise ValueError("QVQ V2B4-P64 requires vector_size=2, trellis_window=16, bank selectors, and W1-W3.5.")
    if v2b2_p32 and (
        vector_size != 2
        or trellis_window != 16
        or bank_ids is None
        or bank_alt_id is None
        or bits > 3.5
    ):
        raise ValueError(
            "QVQ V2B2-P32 requires vector_size=2, trellis_window=16, binary selectors, an alternative bank, "
            "and W1-W3.5."
        )
    if v2b2_p32_lr and (
        vector_size != 2
        or trellis_window != 16
        or bank_ids is None
        or bank_alt_id is None
        or bits > 3.5
    ):
        raise ValueError(
            "QVQ V2B2-P32-LR requires vector_size=2, trellis_window=16, binary selectors, an alternative bank, "
            "and W1-W3.5."
        )
    if vector_size not in (2, 4) or trellis_window not in (16, 18):
        raise ValueError(
            "PGC16 requires vector_size 2 or 4 with trellis_window=16; L18/V4 requires vector_size=4."
        )
    if trellis_window == 18 and vector_size != 4:
        raise ValueError("QVQ L18 decoding requires vector_size=4.")
    if trellis_window == 18 and qvq_transition_bits(bits, vector_size=4) > 10:
        raise ValueError("QVQ L18 decoding supports only rates W1 through W2.5.")
    if trellis_window == 18 and bank_ids is not None:
        raise ValueError("QVQ L18 uses implicit history-selected banks and rejects serialized bank_ids.")

    levels = pgc16_levels_for_version(codebook_version).to(device=trellis.device)
    if v2b2_p32_lr:
        states = unpack_local_ring_states(trellis, bits=bits)
        if bank_alt_id.numel() != 1 or bank_alt_id.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("QVQ V2B2-P32-LR alternative bank ID must be one integer scalar.")
        if bank_alt_id.device != trellis.device:
            raise ValueError("QVQ V2B2-P32-LR alternative bank ID must be on the trellis device.")
        alt_id = int(bank_alt_id.item())
        if not 1 <= alt_id < 4:
            raise ValueError("QVQ V2B2-P32-LR alternative bank ID must be in [1, 3].")
        tile_count = states.numel() // (QVQ_V2B2_P32_LR_RINGS_PER_TILE * QVQ_V2B2_P32_LR_RING_STEPS)
        binary_ids = unpack_qvq_binary_bank_ids(bank_ids, tile_count * QVQ_V2B2_P32_LR_RINGS_PER_TILE)
        state_bank_ids = (
            binary_ids.reshape(tile_count, QVQ_V2B2_P32_LR_RINGS_PER_TILE)
            .repeat_interleave(QVQ_V2B2_P32_LR_RING_STEPS, dim=1)
            .mul(alt_id)
        )
        decoded = pgc16_decode_states_v2_banked(
            states.reshape(tile_count, -1),
            state_bank_ids,
            bits=bits,
            levels=levels,
        ).reshape(
            *trellis.shape[:-1],
            QVQ_V2B2_P32_LR_RINGS_PER_TILE,
            QVQ_V2B2_P32_LR_RING_STEPS,
            2,
        )
        return decoded.to(torch.float32).contiguous()

    states = (
        unpack_dual_v2_states(trellis, bits=bits)
        if dual_v2
        else unpack_trellis_states(
            trellis,
            bits=bits,
            vector_size=vector_size,
            trellis_window=trellis_window,
        )
    )
    if trellis_window == 18:
        decoded = pgc18_decode_states_v4(states, bits=bits, levels=levels)
    elif v2b4_p64:
        tile_count = states.numel() // states.shape[-1]
        segment_ids = unpack_qvq_bank_ids(bank_ids, tile_count * QVQ_V2B4_P64_SEGMENTS_PER_TILE)
        state_bank_ids = segment_ids.reshape(tile_count, QVQ_V2B4_P64_SEGMENTS_PER_TILE).repeat_interleave(
            QVQ_V2B4_P64_STEPS_PER_SEGMENT,
            dim=1,
        )
        decoded = pgc16_decode_states_v2_banked(
            states.reshape(tile_count, -1),
            state_bank_ids,
            bits=bits,
            levels=levels,
        ).reshape(*states.shape, 2)
    elif v2b2_p32:
        if bank_alt_id.numel() != 1 or bank_alt_id.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("QVQ V2B2-P32 alternative bank ID must be one integer scalar.")
        if bank_alt_id.device != trellis.device:
            raise ValueError("QVQ V2B2-P32 alternative bank ID must be on the trellis device.")
        alt_id = int(bank_alt_id.item())
        if not 1 <= alt_id < 4:
            raise ValueError("QVQ V2B2-P32 alternative bank ID must be in [1, 3].")
        tile_count = states.numel() // states.shape[-1]
        binary_ids = unpack_qvq_binary_bank_ids(
            bank_ids,
            tile_count * QVQ_V2B2_P32_SEGMENTS_PER_TILE,
        )
        state_bank_ids = (
            binary_ids.reshape(tile_count, QVQ_V2B2_P32_SEGMENTS_PER_TILE)
            .repeat_interleave(QVQ_V2B2_P32_STEPS_PER_SEGMENT, dim=1)
            .mul(alt_id)
        )
        decoded = pgc16_decode_states_v2_banked(
            states.reshape(tile_count, -1),
            state_bank_ids,
            bits=bits,
            levels=levels,
        ).reshape(*states.shape, 2)
    elif vector_size == 2:
        decoded = pgc16_decode_states(states, levels=levels)
    elif bank_ids is None:
        decoded = pgc16_decode_states_v4(states, levels=levels)
    else:
        decoded = pgc16_decode_states_v4_banked(states, bank_ids, bits=bits, levels=levels)
    decoded = decoded.to(torch.float32)
    return decoded.reshape(*trellis.shape[:-1], -1).contiguous()


def reconstruct_qvq_inner_weight(
    trellis: torch.Tensor,
    *,
    bits: float,
    in_features: int,
    out_features: int,
    tile_rows: int = 16,
    tile_cols: int = 16,
    vector_size: int = 2,
    trellis_window: int = 16,
    codebook_version: str = PGC16_CODEBOOK_VERSION,
    bank_ids: torch.Tensor | None = None,
    dual_v2: bool = False,
    v2b4_p64: bool = False,
    v2b2_p32: bool = False,
    v2b2_p32_lr: bool = False,
    bank_alt_id: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialize the transformed ``[in_features, out_features]`` weight."""

    if v2b2_p32_lr:
        return reconstruct_local_ring_inner_weight(
            trellis,
            bits=bits,
            in_features=in_features,
            out_features=out_features,
            codebook_version=codebook_version,
            bank_ids=bank_ids,
            bank_alt_id=bank_alt_id,
        )
    if in_features <= 0 or out_features <= 0 or in_features % tile_rows or out_features % tile_cols:
        raise ValueError("QVQ in/out features must be positive and divisible by the tile dimensions.")
    tile_count = (in_features // tile_rows) * (out_features // tile_cols)
    expected_words = qvq_words_per_tile(
        bits,
        weight_count=tile_rows * tile_cols,
        vector_size=vector_size,
    )
    if trellis.ndim != 2 or tuple(trellis.shape) != (tile_count, expected_words):
        raise ValueError(
            f"QVQ trellis must have shape `{(tile_count, expected_words)}`, got `{tuple(trellis.shape)}`."
        )
    if bank_ids is not None:
        if bank_ids.device != trellis.device:
            raise ValueError("QVQ bank selectors must be on the trellis device.")
        if v2b2_p32:
            selector_count = tile_count * QVQ_V2B2_P32_SEGMENTS_PER_TILE
            bank_ids = unpack_qvq_binary_bank_ids(bank_ids, selector_count)
        else:
            selector_count = tile_count * QVQ_V2B4_P64_SEGMENTS_PER_TILE if v2b4_p64 else tile_count
            if vector_size != 4 and not (v2b4_p64 and vector_size == 2):
                raise ValueError("QVQ bank selectors require V4, V2B4-P64, or V2B2-P32.")
            bank_ids = unpack_qvq_bank_ids(bank_ids, selector_count)
    decoded = decode_trellis_tiles(
        trellis,
        bits=bits,
        vector_size=vector_size,
        trellis_window=trellis_window,
        codebook_version=codebook_version,
        bank_ids=bank_ids,
        dual_v2=dual_v2,
        v2b4_p64=v2b4_p64,
        v2b2_p32=v2b2_p32,
        v2b2_p32_lr=v2b2_p32_lr,
        bank_alt_id=bank_alt_id,
    )
    return (
        decoded.view(in_features // tile_rows, out_features // tile_cols, tile_rows, tile_cols)
        .permute(0, 2, 1, 3)
        .reshape(in_features, out_features)
        .contiguous()
    )


def viterbi_quantize(
    sequence: torch.Tensor,
    codebook: torch.Tensor,
    *,
    bits: float,
    step_weights: torch.Tensor | None = None,
) -> TrellisQuantizationResult:
    """Find the minimum-squared-error path through a bitshift trellis.

    ``sequence`` has shape ``[steps, V]`` and ``codebook`` has shape
    ``[2**L, V]``. This is the unconstrained reference Viterbi pass; QVQ's
    two-pass approximate tail-biting policy belongs in the processor layer.
    The implementation favors clarity and deterministic validation over speed.
    """

    if sequence.ndim != 2:
        raise ValueError(f"QVQ sequence must have shape `[steps, V]`, got `{tuple(sequence.shape)}`.")
    if codebook.ndim != 2:
        raise ValueError(f"QVQ codebook must have shape `[2**L, V]`, got `{tuple(codebook.shape)}`.")
    if sequence.shape[0] < 1:
        raise ValueError("QVQ sequence must contain at least one vector.")
    if sequence.shape[1] != codebook.shape[1]:
        raise ValueError("QVQ sequence and codebook vector sizes must match.")
    if sequence.device != codebook.device:
        raise ValueError("QVQ sequence and codebook must be on the same device.")
    if not sequence.is_floating_point() or not codebook.is_floating_point():
        raise TypeError("QVQ sequence and codebook must use floating-point dtypes.")
    if not torch.isfinite(sequence).all() or not torch.isfinite(codebook).all():
        raise ValueError("QVQ sequence and codebook must contain only finite values.")

    if step_weights is not None:
        if tuple(step_weights.shape) != (sequence.shape[0],):
            raise ValueError("QVQ Viterbi step weights must have shape `[steps]`.")
        step_weights = step_weights.unsqueeze(0)
    result = batched_viterbi_quantize(
        sequence.unsqueeze(0),
        codebook,
        bits=bits,
        step_weights=step_weights,
    )
    return TrellisQuantizationResult(
        states=result.states[0],
        values=result.values[0],
        squared_error=result.squared_error[0],
    )


def _validate_viterbi_distance_range(
    sequences: torch.Tensor,
    codebook: torch.Tensor,
    *,
    work_dtype: torch.dtype,
    step_weights: torch.Tensor | None = None,
) -> None:
    """Reject finite values whose squared-distance arithmetic would overflow.

    The Viterbi recurrence compares squared distances.  Saturating an overflowed
    distance would turn distinct states into artificial ties, while the expanded
    ``||x||² + ||c||² - 2 x·c`` form can otherwise produce ``inf - inf``.  The
    bound is conservative: it guarantees that every coordinate difference and
    the sum of all vector coordinates fit in ``work_dtype``.
    """

    vector_size = sequences.shape[-1]
    dtype_limit = torch.finfo(work_dtype).max
    maximum_weight = 1.0 if step_weights is None else float(step_weights.detach().abs().amax().item())
    accumulation_terms = max(1.0, max(1, int(sequences.shape[1])) * maximum_weight)
    safe_bound = math.sqrt(dtype_limit / accumulation_terms) / (2.0 * math.sqrt(vector_size))
    maximum = torch.maximum(sequences.detach().abs().amax(), codebook.detach().abs().amax())
    if bool(maximum > safe_bound):
        raise ValueError(
            "QVQ Viterbi sequence/codebook magnitudes are too large for finite squared-distance arithmetic; "
            "rescale the inputs instead of relying on clamped losses"
        )


def _validate_fp32_representable(tensor: torch.Tensor, *, name: str) -> None:
    """Reject finite higher-precision values that would narrow to FP32 infinity."""

    if tensor.dtype != torch.float32 and not torch.isfinite(tensor.to(torch.float32)).all():
        raise ValueError(f"{name} cannot be represented as finite FP32 values")


def batched_viterbi_quantize(
    sequences: torch.Tensor,
    codebook: torch.Tensor,
    *,
    bits: float,
    overlap: torch.Tensor | None = None,
    step_weights: torch.Tensor | None = None,
    _cuda_values_prevalidated: bool = False,
) -> TrellisQuantizationResult:
    """Quantize a batch of sequences with compressed bitshift backpointers.

    ``sequences`` is ``[batch, steps, V]``.  When ``overlap`` is supplied, its
    ``L-transition_bits`` bits constrain the high bits of the first state and the low bits
    of the last state.  This is exactly the bitshift-trellis tail-biting
    condition.  Only the winning predecessor prefix for each shared suffix is
    retained, rather than a full predecessor tensor for every next state.
    """

    if not isinstance(_cuda_values_prevalidated, bool):
        raise TypeError("QVQ CUDA prevalidation flag must be bool.")
    if _cuda_values_prevalidated and sequences.device.type != "cuda":
        raise ValueError("QVQ CUDA prevalidation may be used only with CUDA sequences.")
    if sequences.ndim != 3:
        raise ValueError(f"QVQ sequences must have shape `[batch, steps, V]`, got `{tuple(sequences.shape)}`.")
    if codebook.ndim != 2:
        raise ValueError(f"QVQ codebook must have shape `[2**L, V]`, got `{tuple(codebook.shape)}`.")
    if sequences.shape[0] < 1 or sequences.shape[1] < 1:
        raise ValueError("QVQ sequences must contain at least one batch and one vector.")
    if sequences.shape[2] != codebook.shape[1]:
        raise ValueError("QVQ sequences and codebook vector sizes must match.")
    if sequences.device != codebook.device:
        raise ValueError("QVQ sequences and codebook must be on the same device.")
    if not sequences.is_floating_point() or not codebook.is_floating_point():
        raise TypeError("QVQ sequences and codebook must use floating-point dtypes.")
    if not _cuda_values_prevalidated and (
        not torch.isfinite(sequences).all() or not torch.isfinite(codebook).all()
    ):
        raise ValueError("QVQ sequences and codebook must contain only finite values.")
    if step_weights is not None:
        if tuple(step_weights.shape) != tuple(sequences.shape[:2]):
            raise ValueError("QVQ Viterbi step weights must have shape `[batch, steps]`.")
        if step_weights.device != sequences.device:
            raise ValueError("QVQ Viterbi step weights must share the sequence device.")
        if not step_weights.is_floating_point():
            raise TypeError("QVQ Viterbi step weights must use a floating-point dtype.")
        if not _cuda_values_prevalidated and (
            not torch.isfinite(step_weights).all() or torch.any(step_weights < 0)
        ):
            raise ValueError("QVQ Viterbi step weights must be finite and nonnegative.")

    state_count = int(codebook.shape[0])
    trellis_window = int(math.log2(state_count)) if state_count > 0 else -1
    if state_count != 1 << trellis_window:
        raise ValueError("QVQ codebook row count must be a power of two.")
    vector_size = int(codebook.shape[1])
    if _cuda_values_prevalidated and vector_size != 2:
        raise ValueError("QVQ trusted CUDA Viterbi currently supports only V2 sequences.")
    shift = _validate_trellis_shape(
        bits=bits,
        vector_size=vector_size,
        trellis_window=trellis_window,
    )

    use_float64 = sequences.device.type != "mps" and any(
        tensor.dtype == torch.float64 for tensor in (sequences, codebook, step_weights) if tensor is not None
    )
    work_dtype = torch.float64 if use_float64 else torch.float32
    if not _cuda_values_prevalidated:
        _validate_viterbi_distance_range(sequences, codebook, work_dtype=work_dtype, step_weights=step_weights)

    batch_size, step_count, _ = sequences.shape
    overlap_bits = trellis_window - shift
    overlap_i64 = None
    if overlap is not None:
        if overlap.device != sequences.device:
            raise ValueError("QVQ tail-biting overlap must share the sequence device.")
        if overlap.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError("QVQ tail-biting overlap must use an integer dtype.")
        if overlap.ndim != 1 or overlap.shape[0] != batch_size:
            raise ValueError("QVQ tail-biting overlap must have shape `[batch]`.")
        if overlap_bits == 0:
            overlap_i64 = torch.zeros(batch_size, dtype=torch.long, device=sequences.device)
        else:
            overlap_i64 = overlap.to(device=sequences.device, dtype=torch.long)
            overlap_limit = 1 << overlap_bits
            if not _cuda_values_prevalidated and torch.any(
                (overlap_i64 < 0) | (overlap_i64 >= overlap_limit)
            ):
                raise ValueError(f"QVQ tail-biting overlap must be in `[0, {overlap_limit - 1}]`.")

    if (
        sequences.device.type == "cuda"
        and state_count == 1 << 16
        and vector_size in (2, 4)
        and sequences.dtype == torch.float32
        and codebook.dtype in (torch.float16, torch.float32)
        and sequences.is_contiguous()
        and codebook.is_contiguous()
        and torch.cuda.get_device_capability(sequences.device) >= (8, 0)
    ):
        from ..utils.qvq_cuda import _qvq_cuda_viterbi_trusted, qvq_cuda_viterbi

        native_step_weights = None
        if step_weights is not None:
            native_step_weights = step_weights.to(torch.float32).contiguous()
        cuda_viterbi = _qvq_cuda_viterbi_trusted if _cuda_values_prevalidated else qvq_cuda_viterbi
        states, squared_error = cuda_viterbi(
            sequences,
            codebook,
            bits,
            overlap_i64,
            native_step_weights,
            **({"vector_size": vector_size} if not _cuda_values_prevalidated else {}),
        )
        return TrellisQuantizationResult(states=states, values=codebook[states], squared_error=squared_error)

    if (
        sequences.device.type == "mps"
        and state_count == 1 << 16
        and vector_size == 2
        and shift in range(2, 17)
        and sequences.dtype == torch.float32
        and codebook.dtype == torch.float32
        and sequences.is_contiguous()
        and codebook.is_contiguous()
    ):
        from ..utils.qvq_mps import qvq_mps_viterbi

        native_step_weights = None if step_weights is None else step_weights.to(torch.float32).contiguous()
        states, squared_error = qvq_mps_viterbi(
            sequences,
            codebook,
            bits,
            overlap_i64,
            native_step_weights,
            _trusted_inputs=True,
        )
        return TrellisQuantizationResult(states=states, values=codebook[states], squared_error=squared_error)

    if (
        sequences.device.type == "cpu"
        and state_count == 1 << 16
        and vector_size in (2, 4)
        and shift in range(2, 17)
        and sequences.dtype == torch.float32
        and codebook.dtype == torch.float32
        and sequences.is_contiguous()
        and codebook.is_contiguous()
    ):
        from ..utils.qvq_cpu import qvq_cpu_supported, qvq_cpu_viterbi

        if qvq_cpu_supported():
            native_step_weights = None if step_weights is None else step_weights.to(torch.float32).contiguous()
            states, squared_error = qvq_cpu_viterbi(
                sequences,
                codebook,
                shift,
                overlap=overlap_i64,
                step_weights=native_step_weights,
            )
            return TrellisQuantizationResult(states=states, values=codebook[states], squared_error=squared_error)

    work_sequence = sequences.to(work_dtype)
    work_codebook = codebook.to(work_dtype)
    work_step_weights = None if step_weights is None else step_weights.to(work_dtype)
    state_ids = torch.arange(state_count, dtype=torch.long, device=sequences.device)
    codebook_norm = work_codebook.square().sum(dim=-1)

    def emission(step: int) -> torch.Tensor:
        target = work_sequence[:, step]
        if work_dtype == torch.float64:
            # The expanded quadratic form catastrophically cancels for large
            # finite values even in FP64.  Float64 is the explicit precision
            # contract for this fallback, so compute the distance directly.
            distance = (target.unsqueeze(1) - work_codebook.unsqueeze(0)).square().sum(dim=-1)
        else:
            # Clamp only the tiny negative roundoff possible in the quadratic
            # expansion; this is algebraically the same squared Euclidean metric.
            distance = (
                target.square().sum(dim=-1, keepdim=True)
                + codebook_norm.unsqueeze(0)
                - 2 * target @ work_codebook.transpose(0, 1)
            ).clamp_min_(0)
        if work_step_weights is not None:
            distance = distance * work_step_weights[:, step].unsqueeze(1)
        return distance

    costs = emission(0)
    backpointers: list[torch.Tensor] = []

    predecessor_prefix_count = 1 << shift
    predecessor_suffix_count = 1 << (trellis_window - shift)
    predecessor_suffix = state_ids >> shift
    repeat_contiguous_suffix_costs = sequences.device.type in ("cuda", "mps")

    if overlap is not None:
        assert overlap_i64 is not None
        allowed_start = (state_ids.unsqueeze(0) >> shift) == overlap_i64.unsqueeze(1)
        costs = costs.masked_fill(~allowed_start, torch.inf)

    for step in range(1, step_count):
        predecessor_costs = costs.reshape(batch_size, predecessor_prefix_count, predecessor_suffix_count)
        best_cost, best_prefix = predecessor_costs.min(dim=1)
        if repeat_contiguous_suffix_costs:
            # State ids enumerate each suffix in one contiguous run. CUDA and
            # MPS expand those runs faster than loading an int64 gather map.
            transitioned = best_cost.repeat_interleave(predecessor_prefix_count, dim=1)
        else:
            transitioned = best_cost[:, predecessor_suffix]
        costs = transitioned + emission(step)
        # The predecessor prefix is bounded by 2**shift.  Keep the common
        # low-rate traceback in 16 bits; V4-W4 (shift=16) retains int32
        # because 65535 exceeds signed int16.  This is temporary workspace
        # only—the serialized trellis remains int32 planar words.
        traceback_dtype = torch.int16 if shift <= 15 else torch.int32
        backpointers.append(best_prefix.to(traceback_dtype))

    if overlap is not None and overlap_bits:
        overlap_mask = (1 << overlap_bits) - 1
        allowed_end = (state_ids.unsqueeze(0) & overlap_mask) == overlap_i64.unsqueeze(1)
        costs = costs.masked_fill(~allowed_end, torch.inf)

    end_state = costs.argmin(dim=1)
    path = torch.empty((batch_size, step_count), dtype=torch.long, device=sequences.device)
    path[:, -1] = end_state
    batch_ids = torch.arange(batch_size, dtype=torch.long, device=sequences.device)
    for step in range(step_count - 1, 0, -1):
        suffix = path[:, step] >> shift
        prefix = backpointers[step - 1][batch_ids, suffix].to(torch.long)
        path[:, step - 1] = prefix * predecessor_suffix_count + suffix

    return TrellisQuantizationResult(
        states=path,
        values=codebook[path],
        squared_error=costs[batch_ids, end_state],
    )


def _batched_v2_banked_viterbi_quantize(
    sequences: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    bits: float,
    segment_steps: int,
    overlap: torch.Tensor | None = None,
    step_weights: torch.Tensor | None = None,
    mlx_codebooks=None,
    viterbi_pruning: object | None = None,
    _cuda_values_prevalidated: bool = False,
) -> BankedTrellisQuantizationResult:
    """Exact min-sum recurrence over banked V2 segments.

    The bank is fixed for ``segment_steps`` V2 transitions. At each
    segment boundary the recurrence minimizes over both predecessor prefix and
    prior bank, then merges back to one survivor per 16-bit state and current
    bank. This exposes every bank without factorizing or enumerating complete
    selector schedules.
    """

    if sequences.ndim != 3 or sequences.shape[1:] != (128, 2):
        raise ValueError("QVQ banked V2 sequences must have shape `[batch, 128, 2]`.")
    if codebooks.ndim != 3 or codebooks.shape[0] not in (2, 4) or tuple(codebooks.shape[1:]) != (1 << 16, 2):
        raise ValueError("QVQ banked V2 codebooks must have shape `[2|4, 65536, 2]`.")
    if isinstance(segment_steps, bool) or not isinstance(segment_steps, int) or segment_steps < 1:
        raise ValueError("QVQ banked V2 segment_steps must be a positive integer.")
    if sequences.shape[1] % segment_steps:
        raise ValueError("QVQ banked V2 segment length must divide the 128-step tile.")
    if sequences.device != codebooks.device:
        raise ValueError("QVQ banked V2 sequences and codebooks must share one device.")
    if not sequences.is_floating_point() or not codebooks.is_floating_point():
        raise TypeError("QVQ banked V2 sequences and codebooks must be floating point.")
    if not _cuda_values_prevalidated and (
        not torch.isfinite(sequences).all() or not torch.isfinite(codebooks).all()
    ):
        raise ValueError("QVQ banked V2 sequences and codebooks must be finite.")
    if _cuda_values_prevalidated and sequences.device.type != "cuda":
        raise ValueError("QVQ prevalidated segmented V2 execution is internal to CUDA YAQA.")
    if step_weights is not None:
        if tuple(step_weights.shape) != tuple(sequences.shape[:2]):
            raise ValueError("QVQ banked V2 step weights must have shape `[batch, 128]`.")
        if step_weights.device != sequences.device or not step_weights.is_floating_point():
            raise TypeError("QVQ banked V2 step weights must be floating point on the sequence device.")
        if not torch.isfinite(step_weights).all() or torch.any(step_weights < 0):
            raise ValueError("QVQ banked V2 step weights must be finite and nonnegative.")

    shift = _validate_trellis_shape(bits=bits, vector_size=2, trellis_window=16)
    if shift > 7:
        raise ValueError("QVQ banked V2 supports only rates W1 through W3.5.")
    # Resolve the pruning policy before ANY dispatch decision.  The MLX, native
    # CPU, and eager recurrences below never reach the native CUDA op that
    # enforces the strict policies, so a non-CUDA call must be refused here or
    # `mode="required"` / `mode="auto"`+`fallback="error"` would silently use
    # the baseline recurrence.
    pruning_policy = viterbi_pruning_dispatch_code(viterbi_pruning)
    if sequences.device.type != "cuda":
        reject_viterbi_pruning_fallback_if_strict(
            pruning_policy,
            reason=f"the call runs on device `{sequences.device.type}`, not the CUDA fast path",
        )
    if (
        sequences.device.type == "mps"
        and sequences.dtype == torch.float32
        and codebooks.dtype == torch.float32
        and sequences.is_contiguous()
        and codebooks.is_contiguous()
        and (step_weights is None or step_weights.dtype == torch.float32)
    ):
        try:
            from ..utils.qvq_mlx import qvq_mlx_v2_banked_viterbi_from_torch_mps

            states, segment_bank_ids, squared_error = qvq_mlx_v2_banked_viterbi_from_torch_mps(
                sequences,
                codebooks,
                bits,
                segment_steps=segment_steps,
                overlap=overlap,
                step_weights=None if step_weights is None else step_weights.contiguous(),
                mlx_codebooks=mlx_codebooks,
            )
        except ModuleNotFoundError:
            pass
        else:
            path_banks = segment_bank_ids.repeat_interleave(segment_steps, dim=1).to(torch.long)
            return BankedTrellisQuantizationResult(
                states=states,
                values=codebooks[path_banks, states],
                squared_error=squared_error,
                segment_bank_ids=segment_bank_ids,
            )
    if (
        sequences.device.type == "cpu"
        and sequences.dtype == torch.float32
        and codebooks.dtype == torch.float32
        and sequences.is_contiguous()
        and codebooks.is_contiguous()
        and codebooks.shape[1] == (1 << 16)
        and codebooks.shape[2] == 2
        and (step_weights is None or step_weights.dtype == torch.float32)
    ):
        from ..utils.qvq_cpu import qvq_cpu_supported, qvq_cpu_viterbi_banked

        if qvq_cpu_supported():
            native_weights = None if step_weights is None else step_weights.contiguous()
            native_overlap = None if overlap is None else overlap.to(torch.int64).contiguous()
            states, squared_error, segment_bank_ids = qvq_cpu_viterbi_banked(
                sequences,
                codebooks,
                shift,
                segment_steps,
                overlap=native_overlap,
                step_weights=native_weights,
            )
            path_banks = segment_bank_ids.repeat_interleave(segment_steps, dim=1).to(torch.long)
            return BankedTrellisQuantizationResult(
                states=states,
                values=codebooks[path_banks, states],
                squared_error=squared_error,
                segment_bank_ids=segment_bank_ids,
            )
    use_float64 = sequences.device.type != "mps" and any(
        tensor.dtype == torch.float64 for tensor in (sequences, codebooks, step_weights) if tensor is not None
    )
    work_dtype = torch.float64 if use_float64 else torch.float32
    if not _cuda_values_prevalidated:
        for bank in codebooks:
            _validate_viterbi_distance_range(sequences, bank, work_dtype=work_dtype, step_weights=step_weights)
    work_sequences = sequences.to(work_dtype)
    work_codebooks = codebooks.to(work_dtype)
    work_weights = None if step_weights is None else step_weights.to(work_dtype)
    codebook_norms = work_codebooks.square().sum(dim=-1)
    batch_size = sequences.shape[0]
    bank_count = codebooks.shape[0]
    state_count = 1 << 16
    state_ids = torch.arange(state_count, dtype=torch.long, device=sequences.device)
    prefix_count = 1 << shift
    suffix_count = 1 << (16 - shift)
    predecessor_suffix = state_ids >> shift
    repeat_contiguous = sequences.device.type in ("cuda", "mps")

    def emission(step: int) -> torch.Tensor:
        target = work_sequences[:, step]
        if work_dtype == torch.float64:
            distance = (target[:, None, None, :] - work_codebooks[None]).square().sum(dim=-1)
        else:
            distance = (
                target.square().sum(dim=-1)[:, None, None]
                + codebook_norms[None]
                - 2 * torch.einsum("bv,ksv->bks", target, work_codebooks)
            ).clamp_min_(0)
        if work_weights is not None:
            distance = distance * work_weights[:, step, None, None]
        return distance

    overlap_bits = 16 - shift
    overlap_i64 = None
    if overlap is not None:
        if overlap.device != sequences.device or overlap.ndim != 1 or overlap.shape[0] != batch_size:
            raise ValueError("QVQ banked V2 overlap must have shape `[batch]` on the sequence device.")
        if overlap.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("QVQ banked V2 overlap must use an integer dtype.")
        overlap_i64 = overlap.to(torch.long)
        if torch.any((overlap_i64 < 0) | (overlap_i64 >= 1 << overlap_bits)):
            raise ValueError("QVQ banked V2 overlap is outside the legal retained-state range.")

    native_cuda_dispatch = (
        sequences.device.type == "cuda"
        and work_dtype == torch.float32
        and sequences.dtype == torch.float32
        and codebooks.dtype in (torch.float16, torch.float32)
        and sequences.is_contiguous()
        and codebooks.is_contiguous()
        and torch.cuda.get_device_capability(sequences.device) >= (8, 0)
    )
    if not native_cuda_dispatch:
        # Enforce the strict policies BEFORE the eager recurrence: the native
        # op never sees a call that fails these outer guards, so without this
        # check a CPU/MPS call (or any other non-native path) would silently
        # use the baseline recurrence despite `mode="required"` or
        # `mode="auto"` with `fallback="error"`.
        if sequences.device.type != "cuda":
            guard_reason = f"the call runs on device `{sequences.device.type}`, not the CUDA fast path"
        elif work_dtype != torch.float32 or sequences.dtype != torch.float32:
            guard_reason = "the call does not use FP32 sequences with the FP32 working dtype"
        elif codebooks.dtype not in (torch.float16, torch.float32):
            guard_reason = f"codebook dtype `{codebooks.dtype}` is not supported by the native op"
        elif not sequences.is_contiguous() or not codebooks.is_contiguous():
            guard_reason = "the sequences or codebooks are not contiguous"
        else:
            guard_reason = "the CUDA device is below compute capability 8.0"
        reject_viterbi_pruning_fallback_if_strict(pruning_policy, reason=guard_reason)
    if native_cuda_dispatch:
        from ..utils.qvq_cuda import (
            _qvq_cuda_viterbi_v2_segment_grid_trusted_op,
            qvq_cuda_viterbi_v2_segment_banked,
        )

        native_weights = None if step_weights is None else step_weights.to(torch.float32).contiguous()
        native_overlap = None if overlap_i64 is None else overlap_i64.contiguous()
        if _cuda_values_prevalidated:
            native_states, native_loss, segment_bank_ids = _qvq_cuda_viterbi_v2_segment_grid_trusted_op()(
                sequences,
                codebooks,
                shift,
                segment_steps,
                native_overlap,
                native_weights,
                pruning_policy,
            )
        else:
            native_states, native_loss, segment_bank_ids = qvq_cuda_viterbi_v2_segment_banked(
                sequences,
                codebooks,
                bits,
                segment_steps,
                native_overlap,
                native_weights,
                pruning_policy,
            )
        path_banks = segment_bank_ids.to(torch.long).repeat_interleave(segment_steps, dim=1)
        return BankedTrellisQuantizationResult(
            states=native_states,
            values=codebooks[path_banks, native_states],
            squared_error=native_loss,
            segment_bank_ids=segment_bank_ids,
        )

    costs = emission(0)
    if overlap_i64 is not None:
        allowed_start = (state_ids.unsqueeze(0) >> shift) == overlap_i64.unsqueeze(1)
        costs = costs.masked_fill(~allowed_start[:, None, :], torch.inf)

    prefix_pointers: list[torch.Tensor] = []
    boundary_pointers: list[torch.Tensor | None] = []
    for step in range(1, sequences.shape[1]):
        at_boundary = step % segment_steps == 0
        reshaped = costs.reshape(batch_size, bank_count, prefix_count, suffix_count)
        if at_boundary:
            candidates = reshaped.permute(0, 3, 1, 2).reshape(
                batch_size,
                suffix_count,
                bank_count * prefix_count,
            )
            best_cost, best_flat = candidates.min(dim=-1)
            transitioned = (
                best_cost.repeat_interleave(prefix_count, dim=1)
                if repeat_contiguous
                else best_cost[:, predecessor_suffix]
            )
            costs = transitioned[:, None, :] + emission(step)
            prefix_pointers.append(torch.empty(0, dtype=torch.int16, device=sequences.device))
            boundary_pointers.append(best_flat.to(torch.int16))
        else:
            best_cost, best_prefix = reshaped.min(dim=2)
            transitioned = (
                best_cost.repeat_interleave(prefix_count, dim=2)
                if repeat_contiguous
                else best_cost[:, :, predecessor_suffix]
            )
            costs = transitioned + emission(step)
            prefix_pointers.append(best_prefix.to(torch.int16))
            boundary_pointers.append(None)

    if overlap_i64 is not None:
        allowed_end = (state_ids.unsqueeze(0) & ((1 << overlap_bits) - 1)) == overlap_i64.unsqueeze(1)
        costs = costs.masked_fill(~allowed_end[:, None, :], torch.inf)

    flat_end = costs.reshape(batch_size, -1).argmin(dim=1)
    current_bank = flat_end // state_count
    current_state = flat_end % state_count
    step_count = sequences.shape[1]
    path = torch.empty((batch_size, step_count), dtype=torch.long, device=sequences.device)
    path_banks = torch.empty((batch_size, step_count), dtype=torch.long, device=sequences.device)
    path[:, -1] = current_state
    path_banks[:, -1] = current_bank
    batch_ids = torch.arange(batch_size, dtype=torch.long, device=sequences.device)
    for step in range(step_count - 1, 0, -1):
        suffix = current_state >> shift
        boundary = boundary_pointers[step - 1]
        if boundary is not None:
            flat_predecessor = boundary[batch_ids, suffix].to(torch.long)
            previous_bank = flat_predecessor // prefix_count
            prefix = flat_predecessor % prefix_count
        else:
            prefix = prefix_pointers[step - 1][batch_ids, current_bank, suffix].to(torch.long)
            previous_bank = current_bank
        current_state = prefix * suffix_count + suffix
        current_bank = previous_bank
        path[:, step - 1] = current_state
        path_banks[:, step - 1] = current_bank

    segment_ids = path_banks[:, ::segment_steps].to(torch.uint8)
    expanded_segment_ids = segment_ids.repeat_interleave(segment_steps, dim=1)
    if not torch.equal(expanded_segment_ids.to(path_banks.dtype), path_banks):
        raise RuntimeError("QVQ banked V2 traceback changed banks inside a segment.")
    values = work_codebooks[path_banks, path].to(dtype=codebooks.dtype)
    selected_loss = costs.reshape(batch_size, -1)[batch_ids, flat_end]
    return BankedTrellisQuantizationResult(
        states=path,
        values=values,
        squared_error=selected_loss,
        segment_bank_ids=segment_ids,
    )


def batched_v2b4_p64_viterbi_quantize(
    sequences: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    bits: float,
    overlap: torch.Tensor | None = None,
    step_weights: torch.Tensor | None = None,
    viterbi_pruning: object | None = None,
) -> BankedTrellisQuantizationResult:
    """Exact coupled four-bank V2 recurrence with P64 switching."""

    if codebooks.ndim != 3 or codebooks.shape[0] != 4:
        raise ValueError("QVQ V2B4-P64 requires four codebooks.")
    return _batched_v2_banked_viterbi_quantize(
        sequences,
        codebooks,
        bits=bits,
        segment_steps=QVQ_V2B4_P64_STEPS_PER_SEGMENT,
        overlap=overlap,
        step_weights=step_weights,
        viterbi_pruning=viterbi_pruning,
    )


def fixed_boundary_v2b2_p32_segment_quantize(
    sequences: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    bits: float,
    entry_states: torch.Tensor,
    exit_states: torch.Tensor,
) -> BankedTrellisQuantizationResult:
    """Quantize one P32 segment while preserving its surrounding V2 history.

    ``entry_states`` is the state immediately before the segment and
    ``exit_states`` is the segment's required final state.  Fixing both makes
    the replacement composable with every untouched segment: the first state
    remains a legal successor of the original prefix and the following
    segment observes the exact same predecessor state.
    """

    if sequences.ndim != 3 or tuple(sequences.shape[1:]) != (QVQ_V2B2_P32_STEPS_PER_SEGMENT, 2):
        raise ValueError("QVQ fixed-boundary P32 sequences must have shape [batch, 16, 2].")
    if tuple(codebooks.shape) != (2, 1 << 16, 2):
        raise ValueError("QVQ fixed-boundary P32 codebooks must have shape [2, 65536, 2].")
    if sequences.device != codebooks.device or not sequences.is_floating_point() or not codebooks.is_floating_point():
        raise TypeError("QVQ fixed-boundary P32 inputs must be floating point on one device.")
    if not torch.isfinite(sequences).all() or not torch.isfinite(codebooks).all():
        raise ValueError("QVQ fixed-boundary P32 inputs must be finite.")
    batch_size = sequences.shape[0]
    for name, states in (("entry", entry_states), ("exit", exit_states)):
        if states.device != sequences.device or states.ndim != 1 or states.shape[0] != batch_size:
            raise ValueError(f"QVQ fixed-boundary P32 {name} states must have shape [batch] on the sequence device.")
        if states.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError(f"QVQ fixed-boundary P32 {name} states must use an integer dtype.")
        if torch.any((states.to(torch.long) < 0) | (states.to(torch.long) >= 1 << 16)):
            raise ValueError(f"QVQ fixed-boundary P32 {name} states must be unsigned 16-bit values.")

    shift = _validate_trellis_shape(bits=bits, vector_size=2, trellis_window=16)
    if shift > 7:
        raise ValueError("QVQ fixed-boundary P32 supports only rates W1 through W3.5.")
    if (
        sequences.device.type == "cpu"
        and sequences.dtype == torch.float32
        and codebooks.dtype == torch.float32
        and sequences.is_contiguous()
        and codebooks.is_contiguous()
    ):
        from ..utils.qvq_cpu import qvq_cpu_supported, qvq_cpu_viterbi_banked

        if qvq_cpu_supported():
            states, squared_error, segment_bank_ids = qvq_cpu_viterbi_banked(
                sequences,
                codebooks,
                shift,
                sequences.shape[1],
                entry_states=entry_states,
                exit_states=exit_states,
            )
            path_banks = segment_bank_ids.repeat_interleave(sequences.shape[1], dim=1).to(torch.long)
            return BankedTrellisQuantizationResult(
                states=states,
                values=codebooks[path_banks, states],
                squared_error=squared_error,
                segment_bank_ids=segment_bank_ids,
            )
    work_dtype = torch.float64 if sequences.device.type != "mps" and (
        sequences.dtype == torch.float64 or codebooks.dtype == torch.float64
    ) else torch.float32
    for bank in codebooks:
        _validate_viterbi_distance_range(sequences, bank, work_dtype=work_dtype)
    work_sequences = sequences.to(work_dtype)
    work_codebooks = codebooks.to(work_dtype)
    codebook_norms = work_codebooks.square().sum(dim=-1)
    state_count = 1 << 16
    bank_count = 2
    prefix_count = 1 << shift
    suffix_count = 1 << (16 - shift)
    state_ids = torch.arange(state_count, dtype=torch.long, device=sequences.device)
    predecessor_suffix = state_ids >> shift
    entry_i64 = entry_states.to(torch.long)
    exit_i64 = exit_states.to(torch.long)

    def emission(step: int) -> torch.Tensor:
        target = work_sequences[:, step]
        if work_dtype == torch.float64:
            return (target[:, None, None, :] - work_codebooks[None]).square().sum(dim=-1)
        return (
            target.square().sum(dim=-1)[:, None, None]
            + codebook_norms[None]
            - 2 * torch.einsum("bv,ksv->bks", target, work_codebooks)
        ).clamp_min_(0)

    overlap_mask = suffix_count - 1
    allowed_start = (state_ids.unsqueeze(0) >> shift) == (entry_i64 & overlap_mask).unsqueeze(1)
    costs = emission(0).masked_fill(~allowed_start[:, None, :], torch.inf)
    prefix_pointers: list[torch.Tensor] = []
    repeat_contiguous = sequences.device.type in ("cuda", "mps")
    for step in range(1, QVQ_V2B2_P32_STEPS_PER_SEGMENT):
        reshaped = costs.reshape(batch_size, bank_count, prefix_count, suffix_count)
        best_cost, best_prefix = reshaped.min(dim=2)
        transitioned = (
            best_cost.repeat_interleave(prefix_count, dim=2)
            if repeat_contiguous
            else best_cost[:, :, predecessor_suffix]
        )
        costs = transitioned + emission(step)
        prefix_pointers.append(best_prefix.to(torch.int16))

    batch_ids = torch.arange(batch_size, dtype=torch.long, device=sequences.device)
    exit_costs = costs[batch_ids, :, exit_i64]
    selected_banks = exit_costs.argmin(dim=1)
    selected_loss = exit_costs[batch_ids, selected_banks]
    if not torch.isfinite(selected_loss).all():
        raise RuntimeError("QVQ fixed-boundary P32 search found no legal path between its boundary states.")

    path = torch.empty(
        (batch_size, QVQ_V2B2_P32_STEPS_PER_SEGMENT),
        dtype=torch.long,
        device=sequences.device,
    )
    current_state = exit_i64
    path[:, -1] = current_state
    for step in range(QVQ_V2B2_P32_STEPS_PER_SEGMENT - 1, 0, -1):
        suffix = current_state >> shift
        prefix = prefix_pointers[step - 1][batch_ids, selected_banks, suffix].to(torch.long)
        current_state = prefix * suffix_count + suffix
        path[:, step - 1] = current_state

    if not torch.equal(path[:, -1], exit_i64):
        raise RuntimeError("QVQ fixed-boundary P32 traceback changed the required exit state.")
    if not torch.equal(path[:, 0] >> shift, entry_i64 & overlap_mask):
        raise RuntimeError("QVQ fixed-boundary P32 traceback disconnected from the required entry state.")
    path_banks = selected_banks[:, None].expand(-1, QVQ_V2B2_P32_STEPS_PER_SEGMENT)
    return BankedTrellisQuantizationResult(
        states=path,
        values=codebooks[path_banks, path],
        squared_error=selected_loss,
        segment_bank_ids=selected_banks.to(torch.uint8).unsqueeze(1),
    )


def batched_v2b2_p32_viterbi_quantize(
    sequences: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    bits: float,
    overlap: torch.Tensor | None = None,
    step_weights: torch.Tensor | None = None,
    viterbi_pruning: object | None = None,
) -> BankedTrellisQuantizationResult:
    """Exact coupled binary-bank V2 recurrence with P32 switching."""

    if codebooks.ndim != 3 or codebooks.shape[0] != 2:
        raise ValueError("QVQ V2B2-P32 requires two codebooks.")
    return _batched_v2_banked_viterbi_quantize(
        sequences,
        codebooks,
        bits=bits,
        segment_steps=QVQ_V2B2_P32_STEPS_PER_SEGMENT,
        overlap=overlap,
        step_weights=step_weights,
        viterbi_pruning=viterbi_pruning,
    )


def _tail_biting_v2_banked_quantize(
    sequences: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    bits: float,
    segment_steps: int,
    step_weights: torch.Tensor | None = None,
    candidate_count: int = 1,
    mlx_codebooks=None,
    viterbi_pruning: object | None = None,
    _cuda_values_prevalidated: bool = False,
) -> BankedTrellisQuantizationResult:
    """Apply the canonical two-pass tail-biting approximation to banked V2."""

    if candidate_count != 1:
        raise ValueError("QVQ banked V2 initially supports exactly one tail-biting candidate.")
    midpoint = sequences.shape[1] // 2
    if midpoint % segment_steps:
        raise ValueError("QVQ banked V2 tail rotation must preserve segment boundaries.")
    shift = qvq_transition_bits(bits, vector_size=2)
    if (
        _cuda_values_prevalidated
        and sequences.device.type == "cuda"
        and torch.cuda.get_device_capability(sequences.device) == (8, 0)
        and shift in (3, 4, 5, 6)
    ):
        from ..utils.qvq_cuda import _qvq_cuda_viterbi_v2_segment_tail_trusted_op

        states, squared_error, segment_bank_ids = _qvq_cuda_viterbi_v2_segment_tail_trusted_op()(
            sequences,
            codebooks,
            shift,
            segment_steps,
            None if step_weights is None else step_weights.to(torch.float32).contiguous(),
            viterbi_pruning_dispatch_code(viterbi_pruning),
        )
        path_banks = segment_bank_ids.to(torch.long).repeat_interleave(segment_steps, dim=1)
        return BankedTrellisQuantizationResult(
            states=states,
            values=codebooks[path_banks, states],
            squared_error=squared_error,
            segment_bank_ids=segment_bank_ids,
        )
    rotated = torch.roll(sequences, shifts=midpoint, dims=1)
    rotated_weights = None if step_weights is None else torch.roll(step_weights, shifts=midpoint, dims=1)
    if (
        _cuda_values_prevalidated
        and sequences.device.type == "cuda"
        and torch.cuda.get_device_capability(sequences.device) == (8, 0)
        # Midpoint-only traceback wins at the two edge rates on SM80.  The
        # complete provisional result remains faster for the middle rates.
        and shift in (2, 7)
    ):
        from ..utils.qvq_cuda import _qvq_cuda_viterbi_v2_segment_midpoint_trusted_op

        overlap = _qvq_cuda_viterbi_v2_segment_midpoint_trusted_op()(
            rotated.contiguous(),
            codebooks,
            shift,
            segment_steps,
            None if rotated_weights is None else rotated_weights.to(torch.float32).contiguous(),
        )
    else:
        provisional = _batched_v2_banked_viterbi_quantize(
            rotated,
            codebooks,
            bits=bits,
            segment_steps=segment_steps,
            step_weights=rotated_weights,
            mlx_codebooks=mlx_codebooks,
            viterbi_pruning=viterbi_pruning,
            _cuda_values_prevalidated=_cuda_values_prevalidated,
        )
        overlap = provisional.states[:, midpoint - 1] & ((1 << (16 - shift)) - 1)
        if not _cuda_values_prevalidated and not torch.equal(overlap, provisional.states[:, midpoint] >> shift):
            raise RuntimeError("QVQ banked V2 provisional path violates the V2 transition rule.")
    return _batched_v2_banked_viterbi_quantize(
        sequences,
        codebooks,
        bits=bits,
        segment_steps=segment_steps,
        overlap=overlap,
        step_weights=step_weights,
        mlx_codebooks=mlx_codebooks,
        viterbi_pruning=viterbi_pruning,
        _cuda_values_prevalidated=_cuda_values_prevalidated,
    )


def tail_biting_v2b4_p64_quantize(
    sequences: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    bits: float,
    step_weights: torch.Tensor | None = None,
    candidate_count: int = 1,
    viterbi_pruning: object | None = None,
) -> BankedTrellisQuantizationResult:
    """Apply the canonical two-pass tail-biting approximation to V2B4-P64."""

    if codebooks.ndim != 3 or codebooks.shape[0] != 4:
        raise ValueError("QVQ V2B4-P64 requires four codebooks.")
    return _tail_biting_v2_banked_quantize(
        sequences,
        codebooks,
        bits=bits,
        segment_steps=QVQ_V2B4_P64_STEPS_PER_SEGMENT,
        step_weights=step_weights,
        candidate_count=candidate_count,
        viterbi_pruning=viterbi_pruning,
    )


def tail_biting_v2b2_p32_quantize(
    sequences: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    bits: float,
    step_weights: torch.Tensor | None = None,
    candidate_count: int = 1,
    viterbi_pruning: object | None = None,
) -> BankedTrellisQuantizationResult:
    """Apply the canonical two-pass tail-biting approximation to V2B2-P32."""

    if codebooks.ndim != 3 or codebooks.shape[0] != 2:
        raise ValueError("QVQ V2B2-P32 requires two codebooks.")
    return _tail_biting_v2_banked_quantize(
        sequences,
        codebooks,
        bits=bits,
        segment_steps=QVQ_V2B2_P32_STEPS_PER_SEGMENT,
        step_weights=step_weights,
        candidate_count=candidate_count,
        viterbi_pruning=viterbi_pruning,
    )


def tail_biting_viterbi_quantize(
    sequences: torch.Tensor,
    codebook: torch.Tensor,
    *,
    bits: float,
    step_weights: torch.Tensor | None = None,
    candidate_count: int = 1,
    _cuda_values_prevalidated: bool = False,
) -> TrellisQuantizationResult:
    """Apply QTIP Algorithm 4 with a non-regressing overlap candidate list.

    A fixed-rate ``V``-value transition still has exactly ``2**transition_bits``
    successors.  ``candidate_count`` does not alter that format invariant.
    Instead, it widens the approximate tail-biting boundary search: the
    original two-pass overlap is always evaluated first, additional overlaps
    are ranked by an exact forward/backward min-sum score on the rotated
    sequence, and the lowest-cost constrained circular path is retained.
    """

    if sequences.ndim != 3:
        raise ValueError("QVQ tail-biting sequences must have shape `[batch, steps, V]`.")
    if sequences.shape[1] < 2:
        raise ValueError("QVQ tail-biting requires at least two vectors.")
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or candidate_count < 1:
        raise ValueError("QVQ tail-biting candidate count must be a positive integer.")

    state_count = int(codebook.shape[0]) if codebook.ndim == 2 else 0
    trellis_window = int(math.log2(state_count)) if state_count > 0 else -1
    vector_size = int(codebook.shape[1]) if codebook.ndim == 2 else 0
    shift = _validate_trellis_shape(bits=bits, vector_size=vector_size, trellis_window=trellis_window)
    overlap_bits = trellis_window - shift
    if overlap_bits == 0:
        # The bitshift consumes the complete state, so no tail constraint can
        # cross a vector boundary.  The provisional pass cannot affect the
        # final path and would repeat the full dynamic program unnecessarily.
        return batched_viterbi_quantize(
            sequences,
            codebook,
            bits=bits,
            step_weights=step_weights,
            _cuda_values_prevalidated=_cuda_values_prevalidated,
        )

    if (
        candidate_count == 1
        and _cuda_values_prevalidated
        and sequences.device.type == "cuda"
        and vector_size == 2
        and torch.cuda.get_device_capability(sequences.device) >= (8, 0)
    ):
        from ..utils.qvq_cuda import _qvq_cuda_viterbi_tail_trusted_op

        states, squared_error = _qvq_cuda_viterbi_tail_trusted_op()(
            sequences,
            codebook,
            shift,
            None if step_weights is None else step_weights.to(torch.float32).contiguous(),
        )
        return TrellisQuantizationResult(
            states=states,
            values=codebook[states],
            squared_error=squared_error,
        )

    midpoint = sequences.shape[1] // 2
    rotated = torch.roll(sequences, shifts=midpoint, dims=1)
    rotated_weights = None if step_weights is None else torch.roll(step_weights, shifts=midpoint, dims=1)
    provisional = batched_viterbi_quantize(
        rotated,
        codebook,
        bits=bits,
        step_weights=rotated_weights,
        _cuda_values_prevalidated=_cuda_values_prevalidated,
    )
    overlap_mask = (1 << overlap_bits) - 1
    overlap = provisional.states[:, midpoint - 1] & overlap_mask
    if not _cuda_values_prevalidated and not torch.equal(overlap, provisional.states[:, midpoint] >> shift):
        raise RuntimeError("QVQ provisional Viterbi path violates the bitshift transition rule.")

    candidate_count = min(candidate_count, 1 << overlap_bits)
    overlaps = overlap.unsqueeze(1)
    if candidate_count > 1:
        overlap_scores = _tail_biting_overlap_scores(
            rotated,
            codebook,
            bits=bits,
            boundary=midpoint,
            step_weights=rotated_weights,
        )
        ranking_overlap = overlap
        if overlap_scores.device.type == "mps":
            # Metal's generic top-k has very high fixed cost for the W1
            # 16,384-entry overlap axis. Only up to a handful of integer
            # indices cross this boundary; rank them on the P cores after the
            # persistent Metal recurrence instead of launching a slow MPS sort.
            overlap_scores = overlap_scores.cpu()
            ranking_overlap = overlap.cpu()
        overlap_scores = overlap_scores.scatter(
            1,
            ranking_overlap.unsqueeze(1),
            torch.inf,
        )
        alternatives = overlap_scores.topk(
            candidate_count - 1,
            dim=1,
            largest=False,
            sorted=True,
        ).indices
        alternatives = alternatives.to(overlap.device)
        overlaps = torch.cat((overlaps, alternatives), dim=1)

    if sequences.device.type == "mps" and shift in (2, 3) and candidate_count > 1 and sequences.shape[0] <= 32:
        # Candidates are independent constrained recurrences. Flatten them
        # into the batch dimension so Metal schedules all threadgroups in one
        # launch instead of paying one launch and synchronization per overlap.
        # Above 32 source tiles the unexpanded batch already fills this Apple
        # GPU and the enlarged workspace becomes bandwidth-bound, so retain
        # the serial candidate loop for that measured production regime.
        # Candidate-major order inside each tile preserves the historical
        # strict-< tie rule when argmin selects the first equal loss.
        expanded_sequences = sequences.repeat_interleave(candidate_count, dim=0).contiguous()
        expanded_overlaps = overlaps.reshape(-1).contiguous()
        expanded_weights = (
            None
            if step_weights is None
            else step_weights.repeat_interleave(candidate_count, dim=0).contiguous()
        )
        candidates = batched_viterbi_quantize(
            expanded_sequences,
            codebook,
            bits=bits,
            overlap=expanded_overlaps,
            step_weights=expanded_weights,
            _cuda_values_prevalidated=_cuda_values_prevalidated,
        )
        batch_size, step_count, vector_size = sequences.shape
        candidate_losses = candidates.squared_error.reshape(batch_size, candidate_count)
        selected = candidate_losses.argmin(dim=1)
        batch_ids = torch.arange(batch_size, device=sequences.device)
        candidate_states = candidates.states.reshape(batch_size, candidate_count, step_count)
        candidate_values = candidates.values.reshape(batch_size, candidate_count, step_count, vector_size)
        return TrellisQuantizationResult(
            states=candidate_states[batch_ids, selected],
            values=candidate_values[batch_ids, selected],
            squared_error=candidate_losses[batch_ids, selected],
        )

    best = None
    for candidate_index in range(candidate_count):
        candidate = batched_viterbi_quantize(
            sequences,
            codebook,
            bits=bits,
            # Selecting one column from the candidate matrix is strided when
            # the list has more than one entry. The native CUDA operator
            # deliberately rejects such overlap tensors at its ABI boundary.
            overlap=overlaps[:, candidate_index].contiguous(),
            step_weights=step_weights,
            _cuda_values_prevalidated=_cuda_values_prevalidated,
        )
        if best is None:
            best = candidate
            continue
        improved = candidate.squared_error < best.squared_error
        best = TrellisQuantizationResult(
            states=torch.where(improved.unsqueeze(1), candidate.states, best.states),
            values=torch.where(improved[:, None, None], candidate.values, best.values),
            squared_error=torch.where(improved, candidate.squared_error, best.squared_error),
        )
    assert best is not None
    return best


def _tail_biting_overlap_scores(
    sequences: torch.Tensor,
    codebook: torch.Tensor,
    *,
    bits: float,
    boundary: int,
    step_weights: torch.Tensor | None,
) -> torch.Tensor:
    """Score the best unconstrained path through every boundary overlap.

    The caller first runs :func:`batched_viterbi_quantize`, which establishes
    the public tensor contract.  This helper performs min-sum forward and
    backward recurrences without traceback storage.  Its result has shape
    ``[batch, 2**(L - transition_bits)]``.
    """

    state_count = int(codebook.shape[0])
    trellis_window = int(math.log2(state_count))
    vector_size = int(codebook.shape[1])
    shift = _validate_trellis_shape(bits=bits, vector_size=vector_size, trellis_window=trellis_window)
    if boundary < 0 or boundary >= sequences.shape[1]:
        raise ValueError("QVQ tail-biting score boundary must index the sequence.")
    if (
        sequences.device.type == "mps"
        and state_count == 1 << 16
        and vector_size == 2
        and shift in (2, 3)
        and sequences.dtype == torch.float32
        and codebook.dtype == torch.float32
        and sequences.is_contiguous()
        and codebook.is_contiguous()
    ):
        from ..utils.qvq_mps import qvq_mps_overlap_scores

        native_step_weights = None if step_weights is None else step_weights.to(torch.float32).contiguous()
        return qvq_mps_overlap_scores(
            sequences,
            codebook,
            bits,
            boundary,
            native_step_weights,
            _trusted_inputs=True,
        )

    use_float64 = sequences.device.type != "mps" and any(
        tensor.dtype == torch.float64 for tensor in (sequences, codebook, step_weights) if tensor is not None
    )
    work_dtype = torch.float64 if use_float64 else torch.float32
    _validate_viterbi_distance_range(sequences, codebook, work_dtype=work_dtype, step_weights=step_weights)
    work_sequence = sequences.to(work_dtype)
    work_codebook = codebook.to(work_dtype)
    work_step_weights = None if step_weights is None else step_weights.to(work_dtype)
    batch_size = sequences.shape[0]
    state_ids = torch.arange(state_count, dtype=torch.long, device=sequences.device)
    codebook_norm = work_codebook.square().sum(dim=-1)

    def emission(step: int) -> torch.Tensor:
        target = work_sequence[:, step]
        if work_dtype == torch.float64:
            distance = (target.unsqueeze(1) - work_codebook.unsqueeze(0)).square().sum(dim=-1)
        else:
            distance = (
                target.square().sum(dim=-1, keepdim=True)
                + codebook_norm.unsqueeze(0)
                - 2 * target @ work_codebook.transpose(0, 1)
            ).clamp_min_(0)
        if work_step_weights is not None:
            distance = distance * work_step_weights[:, step].unsqueeze(1)
        return distance

    prefix_count = 1 << shift
    overlap_count = 1 << (trellis_window - shift)
    predecessor_overlap = state_ids >> shift
    successor_overlap = state_ids & (overlap_count - 1)
    repeat_contiguous = sequences.device.type in ("cuda", "mps")

    forward = emission(0)
    for step in range(1, boundary + 1):
        best_predecessor = forward.reshape(batch_size, prefix_count, overlap_count).min(dim=1).values
        if repeat_contiguous:
            forward = best_predecessor.repeat_interleave(prefix_count, dim=1)
        else:
            forward = best_predecessor[:, predecessor_overlap]
        forward = forward + emission(step)

    backward = torch.zeros_like(forward)
    for step in range(sequences.shape[1] - 1, boundary, -1):
        next_cost = emission(step) + backward
        best_successor = next_cost.reshape(batch_size, overlap_count, prefix_count).min(dim=2).values
        backward = best_successor[:, successor_overlap]

    return (forward + backward).reshape(batch_size, overlap_count, prefix_count).min(dim=2).values


def yaqa_sketch_b(weight_gradients: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return YAQA v3 Sketch-B input/output Hessian factors.

    ``weight_gradients`` contains one full-model-loss gradient per independent
    sequence with shape ``[sequences, out_features, in_features]``.  Gradients
    must not be averaged across sequences before this function: doing so adds
    cross-sequence terms that are absent from the Fisher estimator.

    The returned factors follow the paper's identity-initialized power step:
    ``H_I = E[G.T @ G] / out_features`` and
    ``H_O = E[G @ G.T] / in_features``.
    """

    if weight_gradients.ndim != 3:
        raise ValueError("YAQA Sketch B gradients must have shape `[sequences, out_features, in_features]`.")
    if min(weight_gradients.shape) < 1:
        raise ValueError("YAQA Sketch B requires non-empty sequence and weight dimensions.")
    if not weight_gradients.is_floating_point():
        raise TypeError("YAQA Sketch B gradients must use a floating-point dtype.")
    if not torch.isfinite(weight_gradients).all():
        raise ValueError("YAQA Sketch B gradients must contain only finite values.")

    # MPS does not implement float64.  Keep the high-precision CPU/CUDA
    # reference while retaining a usable FP32 Apple calibration path.
    accumulation_dtype = torch.float32 if weight_gradients.device.type == "mps" else torch.float64
    gradients = weight_gradients.to(accumulation_dtype)
    sequence_count, out_features, in_features = gradients.shape
    input_hessian = torch.einsum("soi,soj->ij", gradients, gradients)
    output_hessian = torch.einsum("soi,spi->op", gradients, gradients)
    input_hessian /= sequence_count * out_features
    output_hessian /= sequence_count * in_features
    if not torch.isfinite(input_hessian).all() or not torch.isfinite(output_hessian).all():
        raise ValueError("YAQA Sketch B Gram accumulation overflowed")
    return input_hessian, output_hessian


@dataclass(frozen=True)
class BlockLDLFactorization:
    """One provenance-checked block-LDL factorization and its stabilized Hessian."""

    hessian: torch.Tensor
    L: torch.Tensor
    D: torch.Tensor
    block_size: int
    hessian_version: int | None
    effective_damping: torch.Tensor
    retry_count: int


@dataclass(frozen=True)
class QVQInputHessianPreparation:
    """Provenance-checked reusable Block-LDLQ input geometry."""

    hessian: torch.Tensor
    factorization: tuple[torch.Tensor, torch.Tensor]
    damping: torch.Tensor
    source_hessian: torch.Tensor
    source_data_ptr: int
    source_version: int | None
    transformed_version: int | None
    factor_versions: tuple[int | None, int | None]
    block_size: int
    seed: int
    damp_percent: float


def _safe_tensor_version(tensor: torch.Tensor) -> int | None:
    """Return a tensor version when available, including inference-mode tensors.

    Inference tensors intentionally do not track version counters.  Identity and
    shape/device provenance checks remain available for those tensors, while
    ordinary mutable tensors retain the version-based mutation check.
    """

    try:
        return int(tensor._version)
    except RuntimeError as error:
        if "Inference tensors do not track version counter" not in str(error):
            raise
        return None


def _validate_block_ldl_input(H: torch.Tensor, block_size: int) -> None:

    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("QVQ Hessian must be a square matrix.")
    if not H.is_floating_point():
        raise TypeError("QVQ Hessian must use a floating-point dtype.")
    if not torch.isfinite(H).all():
        raise ValueError("QVQ Hessian must contain only finite values.")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("QVQ block size must be a positive integer.")
    if H.shape[0] % block_size:
        raise ValueError("QVQ Hessian width must be divisible by the block size.")


def _block_ldl_from_cholesky(H: torch.Tensor, chol: torch.Tensor, *, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build block factors from an already successful Cholesky decomposition."""

    L = torch.zeros_like(chol)
    blocks = H.shape[0] // block_size
    D = torch.zeros_like(H)
    for index in range(blocks):
        start = index * block_size
        stop = start + block_size
        diagonal = chol[start:stop, start:stop]
        L[start:, start:stop] = torch.linalg.solve_triangular(
            diagonal.transpose(0, 1),
            chol[start:, start:stop].transpose(0, 1),
            upper=True,
            left=True,
        ).transpose(0, 1)
        D[start:stop, start:stop] = diagonal @ diagonal.transpose(0, 1)
    return L, D


def block_ldl_factor(H: torch.Tensor, *, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unit block-lower ``L`` and block-diagonal ``D`` for ``H=L D Lᵀ``."""

    _validate_block_ldl_input(H, block_size)
    return _block_ldl_from_cholesky(H, torch.linalg.cholesky(H), block_size=block_size)


def prepare_qvq_input_hessian(
    H: torch.Tensor,
    *,
    seed: int,
    damp_percent: float = 0.01,
) -> QVQInputHessianPreparation:
    """Transform, damp, and factor one shared Block-LDLQ input Hessian."""

    _validate_block_ldl_input(H, 16)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("QVQ seed must be an integer.")
    if isinstance(damp_percent, bool) or not isinstance(damp_percent, (int, float)):
        raise TypeError("QVQ damping percent must be a real scalar.")
    damp_percent = float(damp_percent)
    if not math.isfinite(damp_percent) or damp_percent < 0:
        raise ValueError("QVQ damping percent must be finite and nonnegative.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(0, 2, (H.shape[0],), generator=generator, dtype=torch.int8).mul_(2).sub_(1)
    signs = signs.to(device=H.device, dtype=torch.float32)
    transformed = rht_preprocess_hessian(H, signs)
    transformed = (transformed + transformed.transpose(0, 1)) * 0.5
    damping = torch.maximum(
        transformed.diagonal().abs().mean() * damp_percent,
        torch.tensor(torch.finfo(torch.float32).eps, device=H.device),
    )
    transformed.diagonal().add_(damping)
    factorization = block_ldl_factor(transformed.to(torch.float32), block_size=16)
    return QVQInputHessianPreparation(
        hessian=transformed,
        factorization=factorization,
        damping=damping,
        source_hessian=H,
        source_data_ptr=H.data_ptr(),
        source_version=_safe_tensor_version(H),
        transformed_version=_safe_tensor_version(transformed),
        factor_versions=(_safe_tensor_version(factorization[0]), _safe_tensor_version(factorization[1])),
        block_size=16,
        seed=seed,
        damp_percent=damp_percent,
    )


def stabilized_block_ldl_factor(
    H: torch.Tensor,
    *,
    block_size: int,
    retry_damping: torch.Tensor,
    max_retries: int = 6,
) -> BlockLDLFactorization:
    """Factor a PSD YAQA Gram matrix with bounded isotropic FP32 recovery.

    The successful Cholesky is reused to construct block-LDL factors. No
    probe-then-refactor duplication occurs, and off-diagonal Fisher geometry
    is never clamped or rewritten.
    """

    _validate_block_ldl_input(H, block_size)
    if retry_damping.ndim != 0 or retry_damping.device != H.device or not retry_damping.is_floating_point():
        raise ValueError("YAQA retry damping must be a floating-point scalar on the Hessian device.")
    if not torch.isfinite(retry_damping) or retry_damping <= 0:
        raise ValueError("YAQA retry damping must be finite and positive.")
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("YAQA maximum damping retries must be a nonnegative integer.")

    working = H
    effective_damping = torch.zeros((), dtype=H.dtype, device=H.device)
    retry_count = 0
    while True:
        chol, info = torch.linalg.cholesky_ex(working, check_errors=False)
        if int(info.item()) == 0:
            break
        if retry_count >= max_retries:
            raise ValueError("YAQA Hessian remained non-positive-definite after bounded damping retries")
        if retry_count == 0:
            # Preserve the caller's nominally regularized matrix when no
            # recovery is needed; only a failed factorization allocates a
            # stabilized copy.
            working = H.clone()
        increment = retry_damping * (2**retry_count)
        working.diagonal().add_(increment)
        effective_damping = effective_damping + increment
        retry_count += 1

    L, D = _block_ldl_from_cholesky(working, chol, block_size=block_size)
    return BlockLDLFactorization(
        hessian=working,
        L=L,
        D=D,
        block_size=block_size,
        hessian_version=_safe_tensor_version(working),
        effective_damping=effective_damping,
        retry_count=retry_count,
    )


def _dual_v2_split(values: torch.Tensor) -> torch.Tensor:
    """Split interleaved pair steps into two independent batch-major chains."""

    if values.ndim < 2 or values.shape[1] != 128:
        raise ValueError("QVQ Dual-V2 expects 128 interleaved pair steps per tile.")
    return torch.cat((values[:, 0::2], values[:, 1::2]), dim=0).contiguous()


def _dual_v2_merge(values: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Merge two batch-major 64-step chains back into planar pair order."""

    if values.shape[0] != 2 * batch_size or values.shape[1] != 64:
        raise ValueError("QVQ Dual-V2 chain output has an invalid shape.")
    merged = torch.empty((batch_size, 128, *values.shape[2:]), dtype=values.dtype, device=values.device)
    merged[:, 0::2] = values[:batch_size]
    merged[:, 1::2] = values[batch_size:]
    return merged


def block_ldlq_inner(
    inner_weight: torch.Tensor,
    H: torch.Tensor,
    codebook: torch.Tensor,
    *,
    bits: float,
    tile_rows: int = 16,
    tile_cols: int = 16,
    trellis_batch_size: int = 16,
    viterbi_objective: str = "euclidean",
    tail_biting_candidates: int = 1,
    telemetry: QVQQuantizationTelemetry | None = None,
    factorization: tuple[torch.Tensor, torch.Tensor] | None = None,
    dual_v2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize transformed ``[in, out]`` weights with QVQ BlockLDLQ.

    This is Algorithm 5 written in the runtime's transposed orientation.  It
    returns the reconstructed inner weight and one tail-biting state stream per
    planar ``tile_rows × tile_cols`` tile.
    """

    if inner_weight.ndim != 2 or not inner_weight.is_floating_point():
        raise ValueError("QVQ inner weight must be a floating-point matrix.")
    if not torch.isfinite(inner_weight).all():
        raise ValueError("QVQ inner weight must contain only finite values.")
    if isinstance(tile_rows, bool) or not isinstance(tile_rows, int) or tile_rows < 1:
        raise ValueError("QVQ tile rows must be a positive integer.")
    if isinstance(tile_cols, bool) or not isinstance(tile_cols, int) or tile_cols < 1:
        raise ValueError("QVQ tile columns must be a positive integer.")
    if codebook.ndim != 2 or codebook.shape[0] < 1 or codebook.shape[1] < 1:
        raise ValueError("QVQ codebook must be a non-empty matrix.")
    if not codebook.is_floating_point():
        raise TypeError("QVQ codebook must use a floating-point dtype.")
    if inner_weight.device != H.device or inner_weight.device != codebook.device:
        raise ValueError("QVQ BlockLDLQ tensors must share one device.")
    if not torch.isfinite(codebook).all():
        raise ValueError("QVQ codebook must contain only finite values.")
    in_features, out_features = inner_weight.shape
    if in_features < 1 or out_features < 1:
        raise ValueError("QVQ inner-weight dimensions must be positive.")
    if in_features % tile_rows or out_features % tile_cols:
        raise ValueError("QVQ inner-weight dimensions must be divisible by the tile dimensions.")
    if tuple(H.shape) != (in_features, in_features):
        raise ValueError("QVQ Hessian shape must match the inner-weight input dimension.")
    if tile_rows * tile_cols % codebook.shape[1]:
        raise ValueError("QVQ tile size must be divisible by the codebook vector size.")
    if isinstance(trellis_batch_size, bool) or not isinstance(trellis_batch_size, int) or trellis_batch_size < 1:
        raise ValueError("QVQ trellis batch size must be a positive integer.")
    if (
        isinstance(tail_biting_candidates, bool)
        or not isinstance(tail_biting_candidates, int)
        or tail_biting_candidates < 1
    ):
        raise ValueError("QVQ tail-biting candidate count must be a positive integer.")
    if viterbi_objective not in {"euclidean", "hessian_diagonal"}:
        raise ValueError("QVQ Viterbi objective must be `euclidean` or `hessian_diagonal`.")
    if not isinstance(dual_v2, bool):
        raise TypeError("QVQ dual_v2 must be a bool.")
    if dual_v2 and (codebook.shape[0] != 1 << 16 or codebook.shape[1] != 2):
        raise ValueError("QVQ Dual-V2 requires the canonical [65536, 2] codebook.")

    if factorization is None:
        with _qvq_phase(telemetry, "block_ldl_factor", H.device):
            L, D = block_ldl_factor(H.to(torch.float32), block_size=tile_rows)
        # The factor is not needed after forming the strict-lower feedback.
        # Reuse it in place instead of materializing another K x K FP32 clone.
        feedback = L
        feedback.diagonal().sub_(1)
    else:
        L, D = factorization
        if L.shape != (in_features, in_features) or D.shape != (in_features, in_features):
            raise ValueError("QVQ BlockLDLQ factorization shape must match the Hessian.")
        if L.device != H.device or D.device != H.device:
            raise ValueError("QVQ BlockLDLQ factorization must share the Hessian device.")
        # Callers may share this factorization with another pass, so preserve
        # its diagonal and only clone when a caller-owned factor is supplied.
        feedback = L.clone()
        feedback.diagonal().sub_(1)
    with _qvq_phase(telemetry, "block_ldl_setup", L.device):
        source = inner_weight.to(device=L.device, dtype=torch.float32)
        error = source.clone()
        quantized = torch.zeros_like(source)
        tile_states = torch.empty(
            (
                in_features // tile_rows,
                out_features // tile_cols,
                tile_rows * tile_cols // codebook.shape[1],
            ),
            dtype=torch.long,
            device=L.device,
        )

    for block in range(in_features // tile_rows - 1, -1, -1):
        start = block * tile_rows
        stop = start + tile_rows
        with _qvq_phase(telemetry, "block_ldl_feedback", L.device):
            corrected = source[start:stop] + feedback[start:, start:stop].transpose(0, 1) @ (
                error[start:]
            )
            sequences = (
                corrected.reshape(tile_rows, out_features // tile_cols, tile_cols)
                .permute(1, 0, 2)
                .reshape(out_features // tile_cols, -1, codebook.shape[1])
            )
        step_weights = None
        if viterbi_objective == "hessian_diagonal":
            diagonal_weights = D[start:stop, start:stop].diagonal().clamp_min(0)
            diagonal_mean = diagonal_weights.mean()
            if not torch.isfinite(diagonal_mean) or diagonal_mean <= torch.finfo(diagonal_weights.dtype).eps:
                raise RuntimeError("QVQ conditioned Hessian diagonal must have positive finite mean.")
            diagonal_weights = diagonal_weights / diagonal_mean
            weights_per_row = tile_cols // codebook.shape[1]
            step_weights = (
                diagonal_weights.repeat_interleave(weights_per_row).unsqueeze(0).expand(sequences.shape[0], -1)
            )
        logical_batch_size = sequences.shape[0]
        if dual_v2:
            sequences = _dual_v2_split(sequences)
            if step_weights is not None:
                step_weights = _dual_v2_split(step_weights)
        reconstructed_chunks = []
        state_chunks = []
        sequence_chunks = sequences.split(trellis_batch_size)
        weight_chunks = (
            (None,) * len(sequence_chunks) if step_weights is None else step_weights.split(trellis_batch_size)
        )
        with _qvq_phase(telemetry, "block_ldl_viterbi", L.device):
            for chunk, chunk_weights in zip(sequence_chunks, weight_chunks, strict=True):
                result = tail_biting_viterbi_quantize(
                    chunk,
                    codebook,
                    bits=bits,
                    step_weights=chunk_weights,
                    candidate_count=tail_biting_candidates,
                )
                reconstructed_chunks.append(result.values)
                state_chunks.append(result.states)
                if telemetry is not None:
                    telemetry.count("tail_biting_chunks")
                    overlap_bits = int(math.log2(codebook.shape[0])) - qvq_transition_bits(
                        bits,
                        vector_size=codebook.shape[1],
                    )
                    recurrence_passes = 1 if overlap_bits == 0 else 1 + tail_biting_candidates
                    telemetry.count("viterbi_recurrence_passes", recurrence_passes)
                    if (
                        chunk.device.type == "cuda"
                        and chunk.dtype == torch.float32
                        and codebook.dtype in (torch.float16, torch.float32)
                        and chunk.is_contiguous()
                        and codebook.is_contiguous()
                        and torch.cuda.get_device_capability(chunk.device) >= (8, 0)
                    ):
                        telemetry.count("native_viterbi_launches", recurrence_passes)
        with _qvq_phase(telemetry, "block_ldl_reconstruct", L.device):
            reconstructed_values = torch.cat(reconstructed_chunks, dim=0)
            reconstructed_states = torch.cat(state_chunks, dim=0)
            if dual_v2:
                reconstructed_values = _dual_v2_merge(reconstructed_values, logical_batch_size)
                reconstructed_states = _dual_v2_merge(reconstructed_states, logical_batch_size)
            reconstructed = reconstructed_values.reshape(out_features // tile_cols, tile_rows, tile_cols)
            reconstructed = reconstructed.permute(1, 0, 2).reshape(tile_rows, out_features)
            quantized[start:stop] = reconstructed
            error[start:stop] = source[start:stop] - reconstructed
            tile_states[block] = reconstructed_states

    return quantized.to(dtype=inner_weight.dtype), tile_states.reshape(-1, tile_states.shape[-1])


def _block_ldlq_inner_v2_banked(
    inner_weight: torch.Tensor,
    H: torch.Tensor,
    codebooks: tuple[torch.Tensor, ...],
    *,
    bits: float,
    segment_steps: int,
    tile_rows: int = 16,
    tile_cols: int = 16,
    trellis_batch_size: int = 1,
    viterbi_objective: str = "euclidean",
    tail_biting_candidates: int = 1,
    telemetry: QVQQuantizationTelemetry | None = None,
    factorization: tuple[torch.Tensor, torch.Tensor] | None = None,
    bank0_oracle: tuple[torch.Tensor, torch.Tensor] | None = None,
    viterbi_pruning: object | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sequential Block-LDLQ using a coupled segmented V2 recurrence.

    Bank zero is also quantized as a complete independent V2 artifact. The
    mixed result is accepted only when its full input-Hessian proxy improves;
    otherwise the function atomically returns the exact V2 path and all-zero
    selectors.
    """

    if len(codebooks) not in (2, 4) or any(tuple(codebook.shape) != (1 << 16, 2) for codebook in codebooks):
        raise ValueError("QVQ banked V2 requires two or four `[65536, 2]` codebooks.")
    if any(codebook.device != inner_weight.device for codebook in codebooks):
        raise ValueError("QVQ banked V2 codebooks must share the weight device.")
    if bits > 3.5:
        raise ValueError("QVQ banked V2 supports only rates W1 through W3.5.")
    if tail_biting_candidates != 1:
        raise ValueError("QVQ banked V2 initially supports one tail-biting candidate.")
    if tile_rows != 16 or tile_cols != 16:
        raise ValueError("QVQ banked V2 requires 16x16 tiles.")
    if viterbi_objective not in {"euclidean", "hessian_diagonal"}:
        raise ValueError("QVQ banked V2 objective must be `euclidean` or `hessian_diagonal`.")
    if isinstance(segment_steps, bool) or not isinstance(segment_steps, int) or 128 % segment_steps:
        raise ValueError("QVQ banked V2 segment steps must divide 128.")
    in_features, out_features = inner_weight.shape
    if in_features % tile_rows or out_features % tile_cols or tuple(H.shape) != (in_features, in_features):
        raise ValueError("QVQ banked V2 weight/Hessian shapes must be divisible and aligned.")
    if not torch.isfinite(inner_weight).all() or not torch.isfinite(H).all():
        raise ValueError("QVQ banked V2 weight and Hessian must be finite.")

    shared_factorization = (
        block_ldl_factor(H.to(torch.float32), block_size=tile_rows) if factorization is None else factorization
    )
    if bank0_oracle is None:
        bank0_weight, bank0_states = block_ldlq_inner(
            inner_weight,
            H,
            codebooks[0],
            bits=bits,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            trellis_batch_size=trellis_batch_size,
            viterbi_objective=viterbi_objective,
            tail_biting_candidates=tail_biting_candidates,
            telemetry=telemetry,
            factorization=shared_factorization,
        )
    else:
        bank0_weight, bank0_states = bank0_oracle
        if tuple(bank0_weight.shape) != tuple(inner_weight.shape):
            raise ValueError("QVQ banked V2 bank-zero oracle weight has an invalid shape.")
        expected_state_shape = ((in_features // tile_rows) * (out_features // tile_cols), 128)
        if tuple(bank0_states.shape) != expected_state_shape:
            raise ValueError("QVQ banked V2 bank-zero oracle states have an invalid shape.")
    L, D = shared_factorization
    feedback = L.clone()
    feedback.diagonal().sub_(1)
    source = inner_weight.to(device=L.device, dtype=torch.float32)
    error = source.clone()
    quantized = torch.zeros_like(source)
    input_tiles = in_features // tile_rows
    output_tiles = out_features // tile_cols
    tile_states = torch.empty((input_tiles, output_tiles, 128), dtype=torch.long, device=source.device)
    segments_per_tile = 128 // segment_steps
    segment_bank_ids = torch.zeros(
        (input_tiles, output_tiles, segments_per_tile),
        dtype=torch.uint8,
        device=source.device,
    )
    bank_stack = torch.stack(codebooks).contiguous()

    for block in range(input_tiles - 1, -1, -1):
        start, stop = block * tile_rows, (block + 1) * tile_rows
        corrected = source[start:stop] + feedback[start:, start:stop].transpose(0, 1) @ error[start:]
        sequences = (
            corrected.reshape(tile_rows, output_tiles, tile_cols)
            .permute(1, 0, 2)
            .reshape(output_tiles, 128, 2)
        )
        step_weights = None
        if viterbi_objective == "hessian_diagonal":
            diagonal = D[start:stop, start:stop].diagonal().clamp_min(0)
            diagonal_mean = diagonal.mean()
            if not torch.isfinite(diagonal_mean) or diagonal_mean <= torch.finfo(diagonal.dtype).eps:
                raise RuntimeError("QVQ banked V2 conditioned Hessian diagonal must be positive and finite.")
            step_weights = (diagonal / diagonal_mean).repeat_interleave(tile_cols // 2)
            step_weights = step_weights.unsqueeze(0).expand(output_tiles, -1)

        reconstructed_chunks = []
        state_chunks = []
        selector_chunks = []
        sequence_chunks = sequences.split(trellis_batch_size)
        weight_chunks = (
            (None,) * len(sequence_chunks) if step_weights is None else step_weights.split(trellis_batch_size)
        )
        with _qvq_phase(telemetry, "block_ldl_v2_banked", source.device):
            for chunk, chunk_weights in zip(sequence_chunks, weight_chunks, strict=True):
                result = _tail_biting_v2_banked_quantize(
                    chunk,
                    bank_stack,
                    bits=bits,
                    segment_steps=segment_steps,
                    step_weights=chunk_weights,
                    candidate_count=tail_biting_candidates,
                    viterbi_pruning=viterbi_pruning,
                )
                reconstructed_chunks.append(result.values)
                state_chunks.append(result.states)
                selector_chunks.append(result.segment_bank_ids)
                if telemetry is not None:
                    telemetry.count("v2_banked_chunks")
                    telemetry.count("viterbi_recurrence_passes", 2)
        reconstructed_values = torch.cat(reconstructed_chunks, dim=0)
        reconstructed_states = torch.cat(state_chunks, dim=0)
        reconstructed_selectors = torch.cat(selector_chunks, dim=0)
        reconstructed = reconstructed_values.reshape(output_tiles, tile_rows, tile_cols)
        reconstructed = reconstructed.permute(1, 0, 2).reshape(tile_rows, out_features)
        quantized[start:stop] = reconstructed
        error[start:stop] = source[start:stop] - reconstructed
        tile_states[block] = reconstructed_states
        segment_bank_ids[block] = reconstructed_selectors

    mixed_error = quantized - source
    bank0_error = bank0_weight.to(torch.float32) - source
    mixed_loss = torch.sum((H.to(torch.float32) @ mixed_error) * mixed_error)
    bank0_loss = torch.sum((H.to(torch.float32) @ bank0_error) * bank0_error)
    if not torch.isfinite(mixed_loss) or mixed_loss >= bank0_loss:
        return (
            bank0_weight.to(dtype=inner_weight.dtype),
            bank0_states,
            torch.zeros(
                bank0_states.shape[0] * segments_per_tile,
                dtype=torch.uint8,
                device=inner_weight.device,
            ),
        )
    return (
        quantized.to(dtype=inner_weight.dtype),
        tile_states.reshape(-1, 128),
        segment_bank_ids.reshape(-1),
    )


def block_ldlq_inner_v2b4_p64(
    inner_weight: torch.Tensor,
    H: torch.Tensor,
    codebooks: tuple[torch.Tensor, ...],
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Block-LDLQ wrapper for four-bank P64 V2."""

    if len(codebooks) != 4:
        raise ValueError("QVQ V2B4-P64 requires four codebooks.")
    return _block_ldlq_inner_v2_banked(
        inner_weight,
        H,
        codebooks,
        segment_steps=QVQ_V2B4_P64_STEPS_PER_SEGMENT,
        **kwargs,
    )


def _block_ldlq_v2b2_family_batch_cuda(
    inner_weight: torch.Tensor,
    H: torch.Tensor,
    family_stacks: torch.Tensor,
    bank0_oracle: tuple[torch.Tensor, torch.Tensor],
    factorization: BlockLDLFactorization,
    *,
    bits: float,
    trellis_batch_size: int,
    family_batch_size: int | None,
    viterbi_objective: str,
    telemetry: QVQQuantizationTelemetry | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run three independent B2 Block-LDLQ histories in one CUDA work grid."""

    from ..utils.qvq_cuda import _qvq_cuda_viterbi_v2_segment_family_grid_trusted_op

    families = family_stacks.shape[0]
    in_features, out_features = inner_weight.shape
    input_tiles, output_tiles = in_features // 16, out_features // 16
    source = inner_weight.to(torch.float32)
    hessian = H.to(torch.float32)
    L, D = factorization
    feedback = L.clone()
    feedback.diagonal().sub_(1)
    errors = source.unsqueeze(0).expand(families, -1, -1).clone()
    quantized = torch.zeros_like(errors)
    states = torch.empty((families, input_tiles, output_tiles, 128), device=source.device, dtype=torch.long)
    selectors = torch.empty((families, input_tiles, output_tiles, 8), device=source.device, dtype=torch.uint8)
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    midpoint = 64
    overlap_mask = (1 << (16 - transition_bits)) - 1
    family_viterbi = _qvq_cuda_viterbi_v2_segment_family_grid_trusted_op()
    family_indices = torch.arange(families, device=source.device).view(families, 1, 1)
    # Six CTAs are generated per logical tile (three families x two banks).
    # A 128-tile window amortizes segment barriers and won at every W1--W3.5
    # rate on the local 124-SM device. Explicit larger caller batches survive.
    batch_size = (
        max(trellis_batch_size, min(output_tiles, 128))
        if family_batch_size is None
        else min(output_tiles, family_batch_size)
    )
    if batch_size <= 0:
        raise ValueError("QVQ B2 family batch size must be positive.")

    for block in range(input_tiles - 1, -1, -1):
        start, stop = block * 16, (block + 1) * 16
        left = feedback[start:, start:stop].transpose(0, 1)
        corrected = source[start:stop].unsqueeze(0) + torch.bmm(
            left.unsqueeze(0).expand(families, -1, -1), errors[:, start:]
        )
        sequences = corrected.reshape(families, 16, output_tiles, 16).permute(0, 2, 1, 3)
        sequences = sequences.reshape(families, output_tiles, 128, 2)
        step_weights = None
        if viterbi_objective == "hessian_diagonal":
            diagonal = D[start:stop, start:stop].diagonal().clamp_min(0)
            diagonal_mean = diagonal.mean()
            if not torch.isfinite(diagonal_mean) or diagonal_mean <= torch.finfo(diagonal.dtype).eps:
                raise RuntimeError("QVQ banked V2 conditioned Hessian diagonal must be positive and finite.")
            step_weights = (diagonal / diagonal_mean).repeat_interleave(8)

        block_values, block_states, block_selectors = [], [], []
        with _qvq_phase(telemetry, "block_ldl_v2_banked", source.device):
            for chunk_start in range(0, output_tiles, batch_size):
                chunk = sequences[:, chunk_start : chunk_start + batch_size].contiguous()
                chunk_count = chunk.shape[1]
                chunk_weights = None
                if step_weights is not None:
                    chunk_weights = step_weights.view(1, 1, 128).expand(families, chunk_count, -1).contiguous()
                provisional, _, _ = family_viterbi(
                    torch.roll(chunk, shifts=midpoint, dims=2).contiguous(),
                    family_stacks,
                    transition_bits,
                    QVQ_V2B2_P32_STEPS_PER_SEGMENT,
                    None,
                    None
                    if chunk_weights is None
                    else torch.roll(chunk_weights, shifts=midpoint, dims=2).contiguous(),
                )
                overlaps = (provisional[:, :, midpoint - 1] & overlap_mask).contiguous()
                if telemetry is not None:
                    family_sequences = families * chunk_count
                    telemetry.count("viterbi_family_grid_calls", 2)
                    telemetry.count("viterbi_logical_solve_ids", 2)
                    telemetry.count("viterbi_unique_logical_solve_ids", 2)
                    telemetry.count("viterbi_family_grid_sequences", family_sequences * 2)
                    telemetry.count("viterbi_family_state_steps", family_sequences * 128 * 2)
                    telemetry.count("viterbi_provisional_states_produced", family_sequences * 128)
                    telemetry.count("viterbi_provisional_states_consumed", family_sequences)
                    telemetry.count("viterbi_provisional_losses_discarded", family_sequences)
                    telemetry.count("viterbi_provisional_selectors_discarded", family_sequences * 8)
                    # Each block/chunk is generated once and feedback changes
                    # before the next block, so no call has identical inputs.
                    telemetry.count("viterbi_exact_reuse_candidates", 0)
                    telemetry.count("viterbi_reselection_revisits", 0)
                chunk_states, _, chunk_selectors = family_viterbi(
                    chunk,
                    family_stacks,
                    transition_bits,
                    QVQ_V2B2_P32_STEPS_PER_SEGMENT,
                    overlaps,
                    chunk_weights,
                )
                path_banks = chunk_selectors.to(torch.long).repeat_interleave(
                    QVQ_V2B2_P32_STEPS_PER_SEGMENT, dim=2
                )
                block_values.append(family_stacks[family_indices, path_banks, chunk_states])
                block_states.append(chunk_states)
                block_selectors.append(chunk_selectors)
                if telemetry is not None:
                    telemetry.count("v2_banked_family_chunks")
                    telemetry.count("viterbi_recurrence_passes", 2)
        reconstructed_states = torch.cat(block_states, dim=1)
        reconstructed_selectors = torch.cat(block_selectors, dim=1)
        reconstructed = torch.cat(block_values, dim=1).reshape(families, output_tiles, 16, 16)
        reconstructed = reconstructed.permute(0, 2, 1, 3).reshape(families, 16, out_features).to(torch.float32)
        quantized[:, start:stop] = reconstructed
        errors[:, start:stop] = source[start:stop].unsqueeze(0) - reconstructed
        states[:, block] = reconstructed_states
        selectors[:, block] = reconstructed_selectors

    bank0_weight, bank0_states = bank0_oracle
    bank0_error = bank0_weight.to(torch.float32) - source
    bank0_loss = torch.sum((hessian @ bank0_error) * bank0_error)
    for family in range(families):
        mixed_error = quantized[family] - source
        mixed_loss = torch.sum((hessian @ mixed_error) * mixed_error)
        if not torch.isfinite(mixed_loss) or mixed_loss >= bank0_loss:
            quantized[family] = bank0_weight.to(torch.float32)
            states[family] = bank0_states.reshape(input_tiles, output_tiles, 128)
            selectors[family].zero_()
    return quantized.to(inner_weight.dtype), states.reshape(families, -1, 128), selectors.reshape(families, -1)


def block_ldlq_inner_v2b2_p32(
    inner_weight: torch.Tensor,
    H: torch.Tensor,
    codebook_library: tuple[torch.Tensor, ...],
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select one complementary V2 family per module, then search P32 selectors.

    The four-entry library contains canonical V2 followed by three candidate
    alternatives. Every candidate is compared as a complete Block-LDLQ
    artifact under the full input-Hessian proxy. Exact ties retain standalone
    V2 and alternative ID 1, whose all-zero selector stream decodes bank zero.
    """

    if len(codebook_library) != 4:
        raise ValueError("QVQ V2B2-P32 requires canonical V2 plus three complementary candidates.")
    factorization = kwargs.pop("factorization", None)
    pair_stacks = kwargs.pop("bank_codebook_pair_stacks", None)
    family_batch = kwargs.pop("_family_batch", True)
    family_batch_size = kwargs.pop("_family_batch_size", None)
    if factorization is None:
        factorization = block_ldl_factor(H.to(torch.float32), block_size=16)
    bank0_weight, bank0_states = block_ldlq_inner(
        inner_weight,
        H,
        codebook_library[0],
        bits=kwargs["bits"],
        tile_rows=kwargs.get("tile_rows", 16),
        tile_cols=kwargs.get("tile_cols", 16),
        trellis_batch_size=kwargs.get("trellis_batch_size", 1),
        viterbi_objective=kwargs.get("viterbi_objective", "euclidean"),
        tail_biting_candidates=kwargs.get("tail_biting_candidates", 1),
        telemetry=kwargs.get("telemetry"),
        factorization=factorization,
    )
    source = inner_weight.to(torch.float32)
    hessian = H.to(torch.float32)

    def full_loss(candidate: torch.Tensor) -> torch.Tensor:
        error = candidate.to(torch.float32) - source
        return torch.sum((hessian @ error) * error)

    best_weight = bank0_weight
    best_states = bank0_states
    best_selectors = torch.zeros(
        bank0_states.shape[0] * QVQ_V2B2_P32_SEGMENTS_PER_TILE,
        dtype=torch.uint8,
        device=inner_weight.device,
    )
    best_alt_id = 1
    best_loss = full_loss(bank0_weight)
    if family_batch and inner_weight.device.type == "cuda" and kwargs.get("tail_biting_candidates", 1) == 1:
        family_stacks = torch.stack(
            tuple(
                torch.stack((codebook_library[0], codebook_library[alt_id])).contiguous()
                if pair_stacks is None
                else pair_stacks[alt_id - 1]
                for alt_id in range(1, 4)
            )
        ).contiguous()
        candidate_weights, candidate_states_batch, candidate_selectors_batch = (
            _block_ldlq_v2b2_family_batch_cuda(
                inner_weight,
                H,
                family_stacks,
                (bank0_weight, bank0_states),
                factorization,
                bits=kwargs["bits"],
                trellis_batch_size=kwargs.get("trellis_batch_size", 1),
                family_batch_size=family_batch_size,
                viterbi_objective=kwargs.get("viterbi_objective", "euclidean"),
                telemetry=kwargs.get("telemetry"),
            )
        )
        candidates = zip(candidate_weights, candidate_states_batch, candidate_selectors_batch, strict=True)
    else:
        candidates = (
            _block_ldlq_inner_v2_banked(
                inner_weight,
                H,
                (codebook_library[0], codebook_library[alt_id]),
                segment_steps=QVQ_V2B2_P32_STEPS_PER_SEGMENT,
                factorization=factorization,
                bank0_oracle=(bank0_weight, bank0_states),
                **kwargs,
            )
            for alt_id in range(1, 4)
        )
    for alt_id, (candidate_weight, candidate_states, candidate_selectors) in enumerate(candidates, start=1):
        candidate_loss = full_loss(candidate_weight)
        if torch.isfinite(candidate_loss) and candidate_loss < best_loss:
            best_weight = candidate_weight
            best_states = candidate_states
            best_selectors = candidate_selectors
            best_alt_id = alt_id
            best_loss = candidate_loss
    return (
        best_weight,
        best_states,
        best_selectors,
        torch.tensor([best_alt_id], dtype=torch.uint8, device=inner_weight.device),
    )


def block_ldlq_inner_banked(
    inner_weight: torch.Tensor,
    H: torch.Tensor,
    codebooks: tuple[torch.Tensor, ...],
    *,
    bank_codebook_stack: torch.Tensor | None = None,
    bits: float,
    tile_rows: int = 16,
    tile_cols: int = 16,
    trellis_batch_size: int = 16,
    viterbi_objective: str = "euclidean",
    tail_biting_candidates: int = 1,
    telemetry: QVQQuantizationTelemetry | None = None,
    factorization: tuple[torch.Tensor, torch.Tensor] | None = None,
    return_bank0_oracle: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Search fixed V4 banks per tile and return the winning selector.

    Each bank is quantized with the exact Block-LDLQ/Viterbi reference, then
    the tile winner is selected by the configured reconstructed-weight error.
    This reference implementation intentionally favors correctness and clear
    bank semantics over throughput; the batched CUDA selector is a later
    optimization and must match these outputs exactly.
    """

    if len(codebooks) != 4:
        raise ValueError("QVQ banked Block-LDLQ requires exactly four codebooks.")
    if inner_weight.device.type == "meta":
        raise ValueError("QVQ banked Block-LDLQ does not support meta tensors.")
    for codebook in codebooks:
        if codebook.device != inner_weight.device or codebook.shape != (1 << 16, 4):
            raise ValueError("QVQ bank codebooks must have shape (65536, 4) on the weight device.")
    if bank_codebook_stack is not None:
        expected_strides = ((1 << 16) * 4, 4, 1)
        if (
            tuple(bank_codebook_stack.shape) != (4, 1 << 16, 4)
            or bank_codebook_stack.device != inner_weight.device
            or bank_codebook_stack.dtype != codebooks[0].dtype
            or not bank_codebook_stack.is_contiguous()
            or tuple(bank_codebook_stack.stride()) != expected_strides
            or any(
                not bank.is_contiguous()
                or tuple(bank.stride()) != (4, 1)
                or bank.data_ptr()
                != bank_codebook_stack.data_ptr() + bank_index * expected_strides[0] * bank.element_size()
                for bank_index, bank in enumerate(codebooks)
            )
        ):
            raise ValueError("QVQ bank codebook stack must share storage with bank codebooks.")
    # Keep canonical bank 0 as the rollback oracle.  The selected path below
    # is generated sequentially so every winning tile is visible to the
    # Block-LDLQ correction of the next input block.
    shared_factorization = (
        block_ldl_factor(H.to(torch.float32), block_size=tile_rows) if factorization is None else factorization
    )
    bank0_weight, bank0_states = block_ldlq_inner(
        inner_weight,
        H,
        codebooks[0],
        bits=bits,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        trellis_batch_size=trellis_batch_size,
        viterbi_objective=viterbi_objective,
        tail_biting_candidates=tail_biting_candidates,
        telemetry=telemetry,
        factorization=shared_factorization,
    )
    input_tiles = inner_weight.shape[0] // tile_rows
    output_tiles = inner_weight.shape[1] // tile_cols
    tile_count = input_tiles * output_tiles
    L, D = shared_factorization
    # ``shared_factorization`` is still needed by the canonical bank-0 pass;
    # keep its factor immutable while avoiding a second factorization.
    feedback = L.clone()
    feedback.diagonal().sub_(1)
    source = inner_weight.to(device=L.device, dtype=torch.float32)
    native_banked = source.device.type == "cuda" and tail_biting_candidates == 1
    banked_codebooks = (
        bank_codebook_stack if bank_codebook_stack is not None and native_banked else
        torch.stack(codebooks).contiguous() if native_banked else None
    )
    selected_weight = torch.zeros_like(source)
    selected_error = source.clone()
    selected_states = torch.empty(
        (input_tiles, output_tiles, bank0_states.shape[-1]), dtype=bank0_states.dtype, device=source.device
    )
    bank_ids = torch.zeros(tile_count, device=inner_weight.device, dtype=torch.uint8)
    for block in range(input_tiles - 1, -1, -1):
        start, stop = block * tile_rows, (block + 1) * tile_rows
        corrected = source[start:stop] + feedback[start:, start:stop].transpose(0, 1) @ (
            selected_error[start:]
        )
        sequences = (
            corrected.reshape(tile_rows, output_tiles, tile_cols)
            .permute(1, 0, 2)
            .reshape(output_tiles, -1, codebooks[0].shape[1])
        )
        diagonal = D[start:stop, start:stop].diagonal().clamp_min(0)
        diagonal = diagonal / diagonal.mean().clamp_min(torch.finfo(torch.float32).eps)
        candidate_values = []
        candidate_paths = []
        if banked_codebooks is not None:
            from ..utils.qvq_cuda import qvq_cuda_viterbi_banked

            transition_bits = qvq_transition_bits(bits, vector_size=codebooks[0].shape[1])
            banked_paths = None
            banked_losses = None
            sequence_offset = 0
            for chunk in sequences.split(trellis_batch_size):
                chunk_weights = None
                if viterbi_objective == "hessian_diagonal":
                    chunk_weights = diagonal.repeat_interleave(tile_cols // codebooks[0].shape[1]).unsqueeze(0)
                    chunk_weights = chunk_weights.expand(chunk.shape[0], -1).contiguous()
                if transition_bits == 16:
                    states, chunk_losses = qvq_cuda_viterbi_banked(
                        chunk.contiguous(), banked_codebooks, bits, step_weights=chunk_weights
                    )
                    recurrence_passes = 1
                else:
                    midpoint = chunk.shape[1] // 2
                    rotated = torch.roll(chunk, shifts=midpoint, dims=1).contiguous()
                    rotated_weights = None if chunk_weights is None else torch.roll(chunk_weights, shifts=midpoint, dims=1)
                    provisional, _ = qvq_cuda_viterbi_banked(
                        rotated, banked_codebooks, bits, step_weights=rotated_weights
                    )
                    overlap_bits = 16 - transition_bits
                    overlap_mask = (1 << overlap_bits) - 1
                    overlaps = (provisional[:, :, midpoint - 1] & overlap_mask).contiguous()
                    states, chunk_losses = qvq_cuda_viterbi_banked(
                        chunk.contiguous(), banked_codebooks, bits, overlap=overlaps, step_weights=chunk_weights
                    )
                    recurrence_passes = 2
                if telemetry is not None:
                    telemetry.count("tail_biting_chunks")
                    telemetry.count("viterbi_recurrence_passes", recurrence_passes)
                    telemetry.count("native_viterbi_launches", recurrence_passes)
                if banked_paths is None:
                    banked_paths = torch.empty(
                        (states.shape[0], sequences.shape[0], states.shape[2]),
                        dtype=states.dtype,
                        device=states.device,
                    )
                    banked_losses = torch.empty(
                        (chunk_losses.shape[0], sequences.shape[0]),
                        dtype=chunk_losses.dtype,
                        device=chunk_losses.device,
                    )
                chunk_size = chunk.shape[0]
                banked_paths[:, sequence_offset : sequence_offset + chunk_size] = states
                banked_losses[:, sequence_offset : sequence_offset + chunk_size] = chunk_losses
                sequence_offset += chunk_size
        else:
            for codebook in codebooks:
                values, paths = [], []
                for chunk in sequences.split(trellis_batch_size):
                    result = tail_biting_viterbi_quantize(
                        chunk,
                        codebook,
                        bits=bits,
                        step_weights=(
                            diagonal.repeat_interleave(tile_cols // codebook.shape[1]).unsqueeze(0).expand(
                                chunk.shape[0], -1
                            )
                            if viterbi_objective == "hessian_diagonal"
                            else None
                        ),
                        candidate_count=tail_biting_candidates,
                    )
                    values.append(result.values)
                    paths.append(result.states)
                candidate_values.append(torch.cat(values, dim=0))
                candidate_paths.append(torch.cat(paths, dim=0))
        if banked_codebooks is not None:
            # Native Viterbi has already accumulated the exact Euclidean or
            # diagonal-weighted emission loss for every bank and tile. Reuse
            # it instead of gathering all four codebooks and materializing a
            # second [bank, tile, step, vector] FP32 error tensor.
            losses = banked_losses
            winners = losses.argmin(dim=0)
            output_indices = torch.arange(output_tiles, device=source.device)
            selected_states_block = banked_paths[winners, output_indices]
            selected_block = banked_codebooks[winners[:, None], selected_states_block]
        else:
            candidates = torch.stack(candidate_values).to(torch.float32)
            candidate_error = candidates - sequences.unsqueeze(0)
            if viterbi_objective == "hessian_diagonal":
                row_weights = diagonal.repeat_interleave(tile_cols // codebooks[0].shape[1])
                losses = candidate_error.square().sum(dim=3).mul(row_weights).sum(dim=2)
            else:
                losses = candidate_error.square().sum(dim=(2, 3))
            winners = losses.argmin(dim=0)
            output_indices = torch.arange(output_tiles, device=source.device)
            selected_states_block = torch.stack(candidate_paths, dim=0)[winners, output_indices]
            selected_block = candidates[winners, output_indices]
        selected_weight[start:stop] = selected_block.permute(1, 0, 2).reshape(tile_rows, -1)
        selected_error[start:stop] = source[start:stop] - selected_weight[start:stop]
        tile_base = block * output_tiles
        bank_ids[tile_base : tile_base + output_tiles] = winners.to(torch.uint8)
        selected_states[block] = selected_states_block
    # Bank decisions are provisional.  The independent bank paths have
    # different Block-LDLQ histories, so local tile SSE can produce a mixed
    # path that is worse under the actual input Hessian.  Never emit such a
    # regression: compare the complete mixed reconstruction with canonical
    # bank 0 and fall back atomically when its full proxy is not improved.
    mixed_error = selected_weight.to(torch.float32) - source
    bank0_error = bank0_weight.to(torch.float32) - source
    hessian = H.to(torch.float32)
    mixed_loss = torch.sum((hessian @ mixed_error) * mixed_error)
    bank0_loss = torch.sum((hessian @ bank0_error) * bank0_error)
    if not torch.isfinite(mixed_loss) or mixed_loss >= bank0_loss:
        selected_weight = bank0_weight.to(dtype=inner_weight.dtype)
        selected_states = bank0_states
        bank_ids = torch.zeros(tile_count, device=inner_weight.device, dtype=torch.uint8)
    result = (selected_weight.to(dtype=inner_weight.dtype), selected_states.reshape(-1, selected_states.shape[-1]), bank_ids)
    if return_bank0_oracle:
        return (*result, bank0_weight, bank0_states)
    return result


def _block_ldlq_inner_banked_candidates_native(
    inner_weight: torch.Tensor,
    H: torch.Tensor,
    codebooks: tuple[torch.Tensor, ...],
    *,
    bank_codebook_stack: torch.Tensor | None,
    bits: float,
    tile_rows: int,
    tile_cols: int,
    trellis_batch_size: int,
    viterbi_objective: str,
    telemetry: QVQQuantizationTelemetry | None,
    factorization: tuple[torch.Tensor, torch.Tensor],
    baseline_weight: torch.Tensor | None,
    baseline_states: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run independent noncanonical Block-LDLQ histories through banked native Viterbi."""

    from ..utils.qvq_cuda import qvq_cuda_viterbi_banked

    L, D = factorization
    feedback = L.clone()
    feedback.diagonal().sub_(1)
    source = inner_weight.to(device=L.device, dtype=torch.float32)
    use_baseline = baseline_weight is not None
    active_codebooks = codebooks[1:] if use_baseline else codebooks
    bank_count = len(active_codebooks)
    input_tiles = inner_weight.shape[0] // tile_rows
    output_tiles = inner_weight.shape[1] // tile_cols
    steps = tile_rows * tile_cols // active_codebooks[0].shape[1]
    errors = source.unsqueeze(0).expand(bank_count, -1, -1).clone()
    quantized = torch.zeros((4, *source.shape), dtype=source.dtype, device=source.device)
    states = torch.empty((4, input_tiles * output_tiles, steps), dtype=torch.long, device=source.device)
    if use_baseline:
        quantized[0] = baseline_weight.to(torch.float32)
        states[0] = baseline_states
    bank_stack = bank_codebook_stack[1:] if use_baseline else bank_codebook_stack
    if bank_stack is None:
        bank_stack = torch.stack(active_codebooks).contiguous()
    transition_bits = qvq_transition_bits(bits, vector_size=active_codebooks[0].shape[1])
    bank_indices = torch.arange(bank_count, device=source.device)[:, None, None]

    for block in range(input_tiles - 1, -1, -1):
        start, stop = block * tile_rows, (block + 1) * tile_rows
        left = feedback[start:, start:stop].transpose(0, 1)
        corrected_feedback = torch.bmm(
            left.unsqueeze(0).expand(bank_count, -1, -1), errors[:, start:]
        )
        corrected = source[start:stop].unsqueeze(0) + corrected_feedback
        sequences = corrected.reshape(bank_count, tile_rows, output_tiles, tile_cols)
        sequences = sequences.permute(0, 2, 1, 3).reshape(bank_count, output_tiles, steps, active_codebooks[0].shape[1])
        step_weights = None
        if viterbi_objective == "hessian_diagonal":
            diagonal = D[start:stop, start:stop].diagonal().clamp_min(0)
            diagonal = diagonal / diagonal.mean().clamp_min(torch.finfo(torch.float32).eps)
            step_weights = diagonal.repeat_interleave(tile_cols // active_codebooks[0].shape[1]).unsqueeze(0)
            step_weights = step_weights.expand(output_tiles, -1).contiguous()

        state_chunks, value_chunks = [], []
        for chunk in sequences.split(trellis_batch_size, dim=1):
            chunk = chunk.contiguous()
            chunk_weights = None if step_weights is None else step_weights[: chunk.shape[1]]
            recurrence_passes = 1
            if transition_bits == 16:
                chunk_states, _ = qvq_cuda_viterbi_banked(
                    chunk, bank_stack, bits, step_weights=chunk_weights
                )
            else:
                midpoint = chunk.shape[2] // 2
                rotated = torch.roll(chunk, shifts=midpoint, dims=2).contiguous()
                rotated_weights = None if chunk_weights is None else torch.roll(chunk_weights, shifts=midpoint, dims=1)
                provisional, _ = qvq_cuda_viterbi_banked(
                    rotated, bank_stack, bits, step_weights=rotated_weights
                )
                overlap_bits = 16 - transition_bits
                overlap_mask = (1 << overlap_bits) - 1
                overlaps = (provisional[:, :, midpoint - 1] & overlap_mask).contiguous()
                chunk_states, _ = qvq_cuda_viterbi_banked(
                    chunk, bank_stack, bits, overlap=overlaps, step_weights=chunk_weights
                )
                recurrence_passes = 2
            if telemetry is not None:
                telemetry.count("tail_biting_chunks")
                telemetry.count("viterbi_recurrence_passes", recurrence_passes if transition_bits != 16 else 1)
                telemetry.count("native_viterbi_launches", recurrence_passes if transition_bits != 16 else 1)
            state_chunks.append(chunk_states)
            bank_values = bank_stack[bank_indices, chunk_states]
            value_chunks.append(bank_values)
        block_states = torch.cat(state_chunks, dim=1)
        block_values = torch.cat(value_chunks, dim=1).reshape(bank_count, output_tiles, tile_rows, tile_cols)
        block_values = block_values.permute(0, 2, 1, 3).reshape(bank_count, tile_rows, -1)
        quantized[1 if use_baseline else 0 :, start:stop] = block_values
        errors[:, start:stop] = source[start:stop].unsqueeze(0) - block_values
        states[1 if use_baseline else 0 :, block * output_tiles : (block + 1) * output_tiles] = block_states

    return quantized.to(dtype=inner_weight.dtype), states


def block_ldlq_inner_banked_candidates(
    inner_weight: torch.Tensor,
    H: torch.Tensor,
    codebooks: tuple[torch.Tensor, ...],
    *,
    bank_codebook_stack: torch.Tensor | None = None,
    bits: float,
    tile_rows: int = 16,
    tile_cols: int = 16,
    trellis_batch_size: int = 16,
    viterbi_objective: str = "euclidean",
    tail_biting_candidates: int = 1,
    telemetry: QVQQuantizationTelemetry | None = None,
    factorization: tuple[torch.Tensor, torch.Tensor] | None = None,
    _trusted_inputs: bool = False,
    baseline_weight: torch.Tensor | None = None,
    baseline_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all exact bank reconstructions for propagation-aware selection."""

    if len(codebooks) != 4:
        raise ValueError("QVQ banked Block-LDLQ requires exactly four codebooks.")
    if (baseline_weight is None) != (baseline_states is None):
        raise ValueError("QVQ banked baseline weight and states must be supplied together.")
    if inner_weight.ndim != 2 or not inner_weight.is_floating_point():
        raise ValueError("QVQ inner weight must be a floating-point matrix.")
    if inner_weight.device.type == "meta":
        raise ValueError("QVQ banked Block-LDLQ does not support meta tensors.")
    if not _trusted_inputs and not torch.isfinite(inner_weight).all():
        raise ValueError("QVQ inner weight must contain only finite values.")
    if H.device != inner_weight.device or H.ndim != 2 or H.shape != (inner_weight.shape[0], inner_weight.shape[0]):
        raise ValueError("QVQ Hessian shape and device must match the inner-weight input dimension.")
    if not H.is_floating_point() or (not _trusted_inputs and not torch.isfinite(H).all()):
        raise ValueError("QVQ Hessian must be a finite floating-point matrix.")
    if isinstance(tile_rows, bool) or not isinstance(tile_rows, int) or tile_rows < 1:
        raise ValueError("QVQ tile rows must be a positive integer.")
    if isinstance(tile_cols, bool) or not isinstance(tile_cols, int) or tile_cols < 1:
        raise ValueError("QVQ tile columns must be a positive integer.")
    if inner_weight.shape[0] % tile_rows or inner_weight.shape[1] % tile_cols:
        raise ValueError("QVQ inner-weight dimensions must be divisible by the tile dimensions.")
    if isinstance(trellis_batch_size, bool) or not isinstance(trellis_batch_size, int) or trellis_batch_size < 1:
        raise ValueError("QVQ trellis batch size must be a positive integer.")
    if (
        isinstance(tail_biting_candidates, bool)
        or not isinstance(tail_biting_candidates, int)
        or tail_biting_candidates < 1
    ):
        raise ValueError("QVQ tail-biting candidate count must be a positive integer.")
    if viterbi_objective not in {"euclidean", "hessian_diagonal"}:
        raise ValueError("QVQ Viterbi objective must be `euclidean` or `hessian_diagonal`.")
    for codebook in codebooks:
        if (
            codebook.device != inner_weight.device
            or codebook.shape != (1 << 16, 4)
            or not codebook.is_floating_point()
            or not codebook.is_contiguous()
            or (not _trusted_inputs and not torch.isfinite(codebook).all())
        ):
            raise ValueError("QVQ bank codebooks must have shape (65536, 4) on the weight device.")
    if bank_codebook_stack is not None:
        expected_strides = ((1 << 16) * 4, 4, 1)
        if (
            tuple(bank_codebook_stack.shape) != (4, 1 << 16, 4)
            or bank_codebook_stack.device != inner_weight.device
            or bank_codebook_stack.dtype != codebooks[0].dtype
            or not bank_codebook_stack.is_contiguous()
            or tuple(bank_codebook_stack.stride()) != expected_strides
            or any(
                not bank.is_contiguous()
                or tuple(bank.stride()) != (4, 1)
                or bank.data_ptr()
                != bank_codebook_stack.data_ptr() + bank_index * expected_strides[0] * bank.element_size()
                for bank_index, bank in enumerate(codebooks)
            )
        ):
            raise ValueError("QVQ bank codebook stack must share storage with bank codebooks.")
    shared_factorization = (
        block_ldl_factor(H.to(torch.float32), block_size=tile_rows) if factorization is None else factorization
    )
    if inner_weight.device.type == "cuda" and tail_biting_candidates == 1:
        return _block_ldlq_inner_banked_candidates_native(
            inner_weight,
            H,
            codebooks,
            bits=bits,
            bank_codebook_stack=bank_codebook_stack,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            trellis_batch_size=trellis_batch_size,
            viterbi_objective=viterbi_objective,
            telemetry=telemetry,
            factorization=shared_factorization,
            baseline_weight=baseline_weight,
            baseline_states=baseline_states,
        )
    if baseline_weight is None:
        first_weight, first_states = block_ldlq_inner(
            inner_weight,
            H,
            codebooks[0],
            bits=bits,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            trellis_batch_size=trellis_batch_size,
            viterbi_objective=viterbi_objective,
            tail_biting_candidates=tail_biting_candidates,
            telemetry=telemetry,
            factorization=shared_factorization,
        )
    else:
        expected_state_shape = (
            (inner_weight.shape[0] // tile_rows) * (inner_weight.shape[1] // tile_cols),
            tile_rows * tile_cols // 4,
        )
        if baseline_weight.shape != inner_weight.shape or baseline_states.shape != expected_state_shape:
            raise ValueError("QVQ banked baseline tensors have incompatible shapes.")
        first_weight, first_states = baseline_weight, baseline_states
    candidate_weights = torch.empty(
        (len(codebooks), *first_weight.shape), dtype=first_weight.dtype, device=first_weight.device
    )
    candidate_states = torch.empty(
        (len(codebooks), *first_states.shape), dtype=first_states.dtype, device=first_states.device
    )
    candidate_weights[0].copy_(first_weight)
    candidate_states[0].copy_(first_states)
    del first_weight, first_states
    for bank, codebook in enumerate(codebooks[1:], start=1):
        bank_weight, bank_states = block_ldlq_inner(
            inner_weight,
            H,
            codebook,
            bits=bits,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            trellis_batch_size=trellis_batch_size,
            viterbi_objective=viterbi_objective,
            tail_biting_candidates=tail_biting_candidates,
            telemetry=telemetry,
            factorization=shared_factorization,
        )
        candidate_weights[bank].copy_(bank_weight)
        candidate_states[bank].copy_(bank_states)
        del bank_weight, bank_states
    return candidate_weights, candidate_states


def select_banked_tiles_by_output_error(
    candidate_weights: torch.Tensor,
    inputs: torch.Tensor,
    target_output: torch.Tensor,
    *,
    tile_rows: int = 16,
    tile_cols: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedily select bank tiles against held-out module outputs.

    Candidate weights use the QVQ inner orientation ``[banks, in, out]``.
    ``inputs`` and ``target_output`` must come from a disjoint held-out replay.
    After each tile decision the residual is updated, so later choices account
    for errors introduced by earlier tiles. Bank zero wins exact ties.
    """

    if candidate_weights.ndim != 3 or candidate_weights.shape[0] != 4:
        raise ValueError("banked candidate weights must have shape [4, in_features, out_features].")
    if inputs.ndim != 2 or target_output.ndim != 2 or inputs.shape[0] != target_output.shape[0]:
        raise ValueError("held-out inputs and outputs must be rank-2 tensors with matching rows.")
    if candidate_weights.device != inputs.device or target_output.device != inputs.device:
        raise ValueError("banked candidates, held-out inputs, and outputs must share one device.")
    in_features = candidate_weights.shape[1]
    out_features = candidate_weights.shape[2]
    if inputs.shape[1] != in_features or target_output.shape[1] != out_features:
        raise ValueError("held-out geometry does not match banked candidate weights.")
    if in_features % tile_rows or out_features % tile_cols:
        raise ValueError("banked candidate dimensions must be divisible by tile dimensions.")
    if not all(torch.isfinite(value).all() for value in (candidate_weights, inputs, target_output)):
        raise ValueError("banked candidate selection requires finite tensors.")

    selected = candidate_weights[0].clone()
    residual = target_output - inputs @ selected
    bank_ids = torch.zeros(
        (in_features // tile_rows) * (out_features // tile_cols),
        dtype=torch.uint8,
        device=inputs.device,
    )
    tile_index = 0
    for input_start in range(0, in_features, tile_rows):
        input_slice = slice(input_start, input_start + tile_rows)
        input_chunk = inputs[:, input_slice]
        for output_start in range(0, out_features, tile_cols):
            output_slice = slice(output_start, output_start + tile_cols)
            current_tile = selected[input_slice, output_slice]
            candidate_tiles = candidate_weights[:, input_slice, output_slice]
            deltas = input_chunk.unsqueeze(0) @ (candidate_tiles - current_tile)
            candidate_residual = residual[:, output_slice].unsqueeze(0) - deltas
            losses = candidate_residual.square().sum(dim=(1, 2))
            winner = int(losses.argmin().item())
            selected[input_slice, output_slice] = candidate_tiles[winner]
            residual[:, output_slice] = candidate_residual[winner]
            bank_ids[tile_index] = winner
            tile_index += 1
    return selected, bank_ids, residual


def yaqa_inner(
    inner_weight: torch.Tensor,
    input_hessian: torch.Tensor,
    output_hessian: torch.Tensor,
    codebook: torch.Tensor,
    *,
    bits: float,
    tile_rows: int = 16,
    tile_cols: int = 16,
    trellis_batch_size: int = 16,
    tail_biting_candidates: int = 1,
    bank_codebooks: tuple[torch.Tensor, ...] | None = None,
    bank_codebook_stack: torch.Tensor | None = None,
    segmented_bank_stack: torch.Tensor | None = None,
    dual_v2: bool = False,
    v2b4_p64: bool = False,
    v2b2_p32: bool = False,
    factorization: tuple[BlockLDLFactorization, BlockLDLFactorization] | None = None,
    telemetry: QVQQuantizationTelemetry | None = None,
    _bank0_oracle: tuple[torch.Tensor, torch.Tensor] | None = None,
    _rounding_bias: torch.Tensor | None = None,
    _diagnostics: dict[str, object] | None = None,
    _defer_segmented_cuda_checks: bool = False,
    _incremental_cuda_feedback: bool = False,
    _incremental_cuda_factored_feedback: bool = False,
    _incremental_cpu_factored_feedback: bool = False,
    _trusted_inputs: bool = False,
    viterbi_pruning: object | None = None,
) -> tuple[torch.Tensor, ...]:
    """Quantize QVQ's ``[in, out]`` weight with YAQA v3 feedback.

    This is the paper's two-sided fixed point transposed into QVQ runtime
    orientation.  Let ``A = W.T`` and ``E = A - Q(A)``.  Each tile rounds

    ``A + L_I'.T E L_O' + L_I'.T E + E L_O'``.

    Strictly block-lower feedback makes tiles on one anti-diagonal independent,
    so every tile is quantized exactly once from bottom-right to top-left.
    V4 banks select one complete decoder per tile. Segmented V2 banks instead
    run the exact additive P32/P64 recurrence on the YAQA-corrected tile and
    retain its complete selector schedule. Both paths feed the committed error
    into later anti-diagonals; checkpoint packing and inference are unchanged.
    """

    if not isinstance(_trusted_inputs, bool):
        raise TypeError("YAQA trusted-input flag must be bool.")
    if inner_weight.ndim != 2 or not inner_weight.is_floating_point():
        raise ValueError("YAQA inner weight must be a floating-point matrix.")
    if not _trusted_inputs and not torch.isfinite(inner_weight).all():
        raise ValueError("YAQA inner weight must contain only finite values.")
    if _rounding_bias is not None:
        if (
            not isinstance(_rounding_bias, torch.Tensor)
            or tuple(_rounding_bias.shape) != tuple(inner_weight.shape)
            or not _rounding_bias.is_floating_point()
            or _rounding_bias.device != inner_weight.device
        ):
            raise ValueError("YAQA rounding bias must match the floating-point inner-weight geometry and device.")
        if not _trusted_inputs and not torch.isfinite(_rounding_bias).all():
            raise ValueError("YAQA rounding bias must contain only finite values.")
        if not _trusted_inputs:
            _validate_fp32_representable(_rounding_bias, name="YAQA rounding bias")
    if isinstance(tile_rows, bool) or not isinstance(tile_rows, int) or tile_rows < 1:
        raise ValueError("YAQA tile rows must be a positive integer.")
    if isinstance(tile_cols, bool) or not isinstance(tile_cols, int) or tile_cols < 1:
        raise ValueError("YAQA tile columns must be a positive integer.")
    if not all(isinstance(flag, bool) for flag in (dual_v2, v2b4_p64, v2b2_p32)):
        raise TypeError("YAQA format flags must be bools.")
    if sum((dual_v2, v2b4_p64, v2b2_p32)) > 1:
        raise ValueError("YAQA Dual-V2, V2B4-P64, and V2B2-P32 are mutually exclusive.")
    if factorization is not None and (
        not isinstance(factorization, tuple)
        or len(factorization) != 2
        or any(not isinstance(factor, BlockLDLFactorization) for factor in factorization)
    ):
        raise TypeError("YAQA factorization must contain input and output BlockLDLFactorization objects.")
    segmented_v2 = v2b4_p64 or v2b2_p32
    if dual_v2 and bank_codebooks is not None:
        raise ValueError("YAQA Dual-V2 does not use V4 bank codebooks.")
    if bank_codebooks is not None:
        expected_banks = 2 if v2b2_p32 else 4
        expected_vector_size = 2 if segmented_v2 else 4
        if len(bank_codebooks) != expected_banks or any(
            tuple(bank.shape) != (1 << 16, expected_vector_size) for bank in bank_codebooks
        ):
            raise ValueError(
                f"YAQA banked mode requires {expected_banks} [65536, {expected_vector_size}] codebooks."
            )
        if any(bank.device != inner_weight.device or not bank.is_floating_point() for bank in bank_codebooks):
            raise ValueError("YAQA bank codebooks must share the weight device and use floating point.")
        codebook = bank_codebooks[0]
        if segmented_v2 and bank_codebook_stack is not None:
            raise ValueError("YAQA segmented-V2 uses the native tile quantizer and rejects a V4 bank stack.")
        if not segmented_v2 and segmented_bank_stack is not None:
            raise ValueError("YAQA segmented bank stack requires a segmented-V2 format.")
        if segmented_bank_stack is not None:
            expected_shape = (expected_banks, 1 << 16, expected_vector_size)
            expected_strides = ((1 << 16) * expected_vector_size, expected_vector_size, 1)
            if (
                tuple(segmented_bank_stack.shape) != expected_shape
                or tuple(segmented_bank_stack.stride()) != expected_strides
                or segmented_bank_stack.device != inner_weight.device
                or segmented_bank_stack.dtype != codebook.dtype
                or not segmented_bank_stack.is_contiguous()
                or any(
                    bank.data_ptr()
                    != segmented_bank_stack.data_ptr()
                    + bank_index * expected_strides[0] * bank.element_size()
                    for bank_index, bank in enumerate(bank_codebooks)
                )
            ):
                raise ValueError("YAQA segmented bank stack must share bank-major storage with its codebooks.")
        if bank_codebook_stack is not None and tuple(bank_codebook_stack.shape) != (4, 1 << 16, 4):
            raise ValueError("YAQA bank codebook stack must have shape [4, 65536, 4].")
        if bank_codebook_stack is not None and (
            bank_codebook_stack.device != inner_weight.device
            or bank_codebook_stack.dtype != codebook.dtype
            or not bank_codebook_stack.is_contiguous()
        ):
            raise ValueError("YAQA bank codebook stack must be contiguous and share the bank device and dtype.")
        if bank_codebook_stack is not None:
            expected_strides = (1 << 16) * 4, 4, 1
            if tuple(bank_codebook_stack.stride()) != expected_strides or any(
                not bank.is_contiguous()
                or tuple(bank.stride()) != (4, 1)
                or (
                bank.data_ptr()
                != bank_codebook_stack.data_ptr() + bank_index * expected_strides[0] * bank.element_size()
                )
                for bank_index, bank in enumerate(bank_codebooks)
            ):
                raise ValueError("YAQA bank codebook stack must share storage with bank_codebooks in bank-major order.")
    elif bank_codebook_stack is not None or segmented_bank_stack is not None:
        raise ValueError("YAQA bank codebook stacks require bank_codebooks.")
    if codebook.ndim != 2 or codebook.shape[0] < 1 or codebook.shape[1] < 1:
        raise ValueError("YAQA codebook must be a non-empty matrix.")
    if not codebook.is_floating_point():
        raise TypeError("YAQA codebook must use a floating-point dtype.")
    if bank_codebooks is not None and not _trusted_inputs:
        if any(not torch.isfinite(bank).all() for bank in bank_codebooks):
            raise ValueError("YAQA bank codebooks must contain only finite values.")
    elif bank_codebooks is None and not _trusted_inputs and not torch.isfinite(codebook).all():
        raise ValueError("YAQA codebook must contain only finite values.")
    if not _trusted_inputs:
        for name, tensor in (
            ("YAQA inner weight", inner_weight),
            ("YAQA input Hessian", input_hessian),
            ("YAQA output Hessian", output_hessian),
            ("YAQA codebook", codebook),
        ):
            _validate_fp32_representable(tensor, name=name)
        if bank_codebooks is not None:
            for bank_index, bank in enumerate(bank_codebooks[1:], start=1):
                _validate_fp32_representable(bank, name=f"YAQA bank codebook {bank_index}")
    if (
        inner_weight.device != input_hessian.device
        or inner_weight.device != output_hessian.device
        or inner_weight.device != codebook.device
    ):
        raise ValueError("YAQA weight, Hessians, and codebook must share one device.")

    in_features, out_features = inner_weight.shape
    if in_features < 1 or out_features < 1:
        raise ValueError("YAQA inner-weight dimensions must be positive.")
    if in_features % tile_rows or out_features % tile_cols:
        raise ValueError("YAQA inner-weight dimensions must be divisible by the tile dimensions.")
    if tuple(input_hessian.shape) != (in_features, in_features):
        raise ValueError("YAQA input Hessian must match the inner-weight input dimension.")
    if tuple(output_hessian.shape) != (out_features, out_features):
        raise ValueError("YAQA output Hessian must match the inner-weight output dimension.")
    if not input_hessian.is_floating_point() or not output_hessian.is_floating_point():
        raise TypeError("YAQA Hessians must use floating-point dtypes.")
    if not _trusted_inputs and (
        not torch.isfinite(input_hessian).all() or not torch.isfinite(output_hessian).all()
    ):
        raise ValueError("YAQA Hessians must contain only finite values.")
    if tile_rows * tile_cols % codebook.shape[1]:
        raise ValueError("YAQA tile size must be divisible by the codebook vector size.")
    if dual_v2 and tuple(codebook.shape) != (1 << 16, 2):
        raise ValueError("YAQA Dual-V2 requires the canonical [65536, 2] codebook.")
    if isinstance(trellis_batch_size, bool) or not isinstance(trellis_batch_size, int) or trellis_batch_size < 1:
        raise ValueError("YAQA trellis batch size must be a positive integer.")
    if (
        isinstance(tail_biting_candidates, bool)
        or not isinstance(tail_biting_candidates, int)
        or tail_biting_candidates < 1
    ):
        raise ValueError("YAQA tail-biting candidate count must be a positive integer.")

    input_blocks = in_features // tile_rows
    output_blocks = out_features // tile_cols
    if telemetry is not None:
        telemetry.count("yaqa_calls")
        telemetry.count("yaqa_input_features", in_features)
        telemetry.count("yaqa_output_features", out_features)
        telemetry.count("yaqa_tiles", input_blocks * output_blocks)
        telemetry.count("yaqa_anti_diagonals", input_blocks + output_blocks - 1)
        telemetry.count("yaqa_candidate_banks", 1 if bank_codebooks is None else len(bank_codebooks))

    bank0_reference = None
    bank0_reference_states = None
    if bank_codebooks is not None:
        # Tilewise YAQA scores only see diagonal Hessian blocks. Keep an exact
        # canonical-bank run so cross-tile Kronecker terms cannot make the
        # mixed-bank result worse overall.
        if _bank0_oracle is None and not _defer_segmented_cuda_checks:
            bank0_reference, bank0_reference_states = yaqa_inner(
                inner_weight,
                input_hessian,
                output_hessian,
                bank_codebooks[0],
                bits=bits,
                tile_rows=tile_rows,
                tile_cols=tile_cols,
                trellis_batch_size=trellis_batch_size,
                tail_biting_candidates=tail_biting_candidates,
                bank_codebooks=None,
                factorization=factorization,
                _rounding_bias=_rounding_bias,
                _incremental_cuda_feedback=_incremental_cuda_feedback,
                _incremental_cuda_factored_feedback=_incremental_cuda_factored_feedback,
                _incremental_cpu_factored_feedback=_incremental_cpu_factored_feedback,
            )
        elif _bank0_oracle is not None:
            bank0_reference, bank0_reference_states = _bank0_oracle

    if factorization is None:
        input_L, _ = block_ldl_factor(input_hessian.to(torch.float32), block_size=tile_rows)
        output_L, _ = block_ldl_factor(output_hessian.to(torch.float32), block_size=tile_cols)
    else:
        input_factor, output_factor = factorization
        for name, factor, hessian, block_size in (
            ("input", input_factor, input_hessian, tile_rows),
            ("output", output_factor, output_hessian, tile_cols),
        ):
            if (
                factor.hessian is not hessian
                or factor.block_size != block_size
                or factor.hessian_version != _safe_tensor_version(hessian)
                or factor.L.device != hessian.device
                or factor.L.dtype != torch.float32
                or tuple(factor.L.shape) != tuple(hessian.shape)
                or tuple(factor.D.shape) != tuple(hessian.shape)
            ):
                raise ValueError(f"YAQA prepared {name} factor does not match its originating Hessian and block size.")
        input_L = input_factor.L
        output_L = output_factor.L
    # Canonical, B2-family, and B4 candidates share these immutable factors.
    # MPSGraph specializes every shrinking suffix GEMM below.  Real projection
    # shapes consequently compile thousands of one-use graphs, while the M4
    # CPU executes the same small/skinny FP32 products directly through
    # Accelerate.  Keep the feedback recurrence in host memory on Apple and
    # transfer only compact anti-diagonal trellis batches to native Metal.
    quantization_device = inner_weight.device
    apple_host_feedback = quantization_device.type == "mps"
    feedback_device = torch.device("cpu") if apple_host_feedback else quantization_device
    input_feedback = input_L.to(device=feedback_device, copy=True).contiguous()
    output_feedback = output_L.to(device=feedback_device, copy=True).contiguous()
    input_feedback.diagonal().sub_(1)
    output_feedback.diagonal().sub_(1)
    source = inner_weight.to(device=feedback_device, dtype=torch.float32)
    rounding_bias = (
        None if _rounding_bias is None else _rounding_bias.to(device=feedback_device, dtype=torch.float32)
    )
    # Bank scoring and the full rollback proxy use one stable accumulation
    # dtype. Calibration may provide FP64, BF16, or FP16 Hessians; mixing
    # those factors directly with FP32 errors otherwise rejects einsum.
    input_hessian_fp32 = input_hessian.to(device=feedback_device, dtype=torch.float32)
    output_hessian_fp32 = output_hessian.to(device=feedback_device, dtype=torch.float32)
    input_hessian_blocks = torch.stack(
        [input_hessian_fp32[index : index + tile_rows, index : index + tile_rows]
         for index in range(0, in_features, tile_rows)]
    )
    output_hessian_blocks = torch.stack(
        [output_hessian_fp32[index : index + tile_cols, index : index + tile_cols]
         for index in range(0, out_features, tile_cols)]
    )
    quantized = torch.zeros_like(source)
    quantized_blocks = quantized.view(input_blocks, tile_rows, output_blocks, tile_cols).permute(0, 2, 1, 3)
    source_blocks = source.view(input_blocks, tile_rows, output_blocks, tile_cols).permute(0, 2, 1, 3)
    steps_per_tile = tile_rows * tile_cols // codebook.shape[1]
    tile_states = torch.empty(
        (input_blocks, output_blocks, steps_per_tile),
        dtype=torch.long,
        device=feedback_device,
    )
    segments_per_tile = (
        QVQ_V2B2_P32_SEGMENTS_PER_TILE
        if v2b2_p32
        else QVQ_V2B4_P64_SEGMENTS_PER_TILE
        if v2b4_p64
        else 1
    )
    bank_ids = torch.zeros(
        input_blocks * output_blocks * segments_per_tile,
        dtype=torch.uint8,
        device=feedback_device,
    )
    error = source.clone()
    error_blocks = error.view(input_blocks, tile_rows, output_blocks, tile_cols).permute(0, 2, 1, 3)
    incremental_cuda_feedback = _incremental_cuda_feedback and feedback_device.type == "cuda"
    incremental_cuda_factored_feedback = (
        _incremental_cuda_factored_feedback
        and feedback_device.type == "cuda"
        and not incremental_cuda_feedback
    )
    incremental_cpu_factored_feedback = (
        _incremental_cpu_factored_feedback
        and feedback_device.type == "cpu"
        and qvq_cpu_supported()
    )
    transformed_error = None
    feedback_temp = None
    left_transformed_error = None
    right_transformed_error = None
    if incremental_cuda_feedback:
        feedback_temp = torch.empty_like(source)
        transformed_error = torch.empty_like(source)
        torch.mm(input_L.transpose(0, 1), error, out=feedback_temp)
        torch.mm(feedback_temp, output_L, out=transformed_error)
    elif incremental_cuda_factored_feedback or incremental_cpu_factored_feedback:
        if telemetry is not None:
            telemetry.count("yaqa_factored_feedback_calls")
        left_transformed_error = torch.empty_like(source)
        right_transformed_error = torch.empty_like(source)
        torch.mm(input_feedback.transpose(0, 1), error, out=left_transformed_error)
        torch.mm(error, output_feedback, out=right_transformed_error)
    active_banks = bank_codebooks if bank_codebooks is not None else (codebook,)
    if segmented_v2 and segmented_bank_stack is None:
        segmented_bank_stack = torch.stack(active_banks).contiguous()
    host_segmented_bank_stack = None
    mlx_segmented_bank_stack = None
    if segmented_v2 and apple_host_feedback:
        from ..utils.qvq_mlx import qvq_mlx_prepare_v2_banked_codebooks_from_torch

        # Bind the prepared MLX norms to the same immutable host bank stack
        # used to reconstruct compact traceback results.  This incurs one
        # setup copy per family and lets every tile reuse both the staged bank
        # values and their norms without revalidating a distinct MPS alias.
        host_segmented_bank_stack = segmented_bank_stack.detach().to("cpu").contiguous()
        mlx_segmented_bank_stack = qvq_mlx_prepare_v2_banked_codebooks_from_torch(host_segmented_bank_stack)
    host_canonical_codebook_stack = None
    mlx_canonical_codebook_stack = None
    if (
        apple_host_feedback
        and not segmented_v2
        and bank_codebooks is None
        and not dual_v2
        and codebook.shape[1] == 2
        and tail_biting_candidates == 1
    ):
        from ..utils.qvq_mlx import qvq_mlx_prepare_v2_banked_codebooks_from_torch

        host_canonical_codebook_stack = codebook.detach().to("cpu").unsqueeze(0).contiguous()
        mlx_canonical_codebook_stack = qvq_mlx_prepare_v2_banked_codebooks_from_torch(
            host_canonical_codebook_stack
        )
    banked_codebooks = (
        bank_codebook_stack if bank_codebook_stack is not None else torch.stack(active_banks).contiguous()
        if (
            bank_codebooks is not None
            and not segmented_v2
            and source.device.type == "cuda"
            and tail_biting_candidates == 1
        )
        else None
    )
    yaqa_cuda_invalid = None
    yaqa_cuda_safe_bound = None
    if source.device.type == "cuda":
        # YAQA's corrected targets are produced on-device. Accumulate a
        # device-side failure flag across anti-diagonals and synchronize once
        # after the recurrence instead of stalling before every tail pass.
        yaqa_cuda_invalid = torch.zeros((), dtype=torch.bool, device=source.device)
        yaqa_cuda_safe_bound = math.sqrt(torch.finfo(torch.float32).max / steps_per_tile) / (
            2.0 * math.sqrt(codebook.shape[1])
        )
        validation_codebooks = segmented_bank_stack if segmented_v2 else codebook
        assert validation_codebooks is not None
        yaqa_cuda_invalid.logical_or_(
            validation_codebooks.detach().abs().amax() > yaqa_cuda_safe_bound
        )
    # Apple feedback stays on the CPU, but segmented trellis search still uses
    # the exact MLX Metal recurrence.  Only one compact anti-diagonal batch and
    # its traceback cross the shared-memory runtime boundary at a time.

    anti_diagonal_schedule = _yaqa_anti_diagonal_schedule(
        feedback_device,
        input_blocks,
        output_blocks,
        tile_rows,
        tile_cols,
    )
    for coordinates, input_indices, output_indices, flat_tile_indices, input_rows, output_rows in anti_diagonal_schedule:
        with _qvq_phase(telemetry, "yaqa_feedback", source.device):
            if incremental_cuda_feedback:
                assert transformed_error is not None
                transformed_blocks = transformed_error.view(
                    input_blocks,
                    tile_rows,
                    output_blocks,
                    tile_cols,
                ).permute(0, 2, 1, 3)
                corrected_tile_stack = transformed_blocks[input_indices, output_indices]
                if rounding_bias is not None:
                    bias_blocks = rounding_bias.view(
                        input_blocks,
                        tile_rows,
                        output_blocks,
                        tile_cols,
                    ).permute(0, 2, 1, 3)
                    corrected_tile_stack = corrected_tile_stack + bias_blocks[input_indices, output_indices]
            elif incremental_cuda_factored_feedback:
                assert left_transformed_error is not None and right_transformed_error is not None
                from ..utils.qvq_cuda import _qvq_cuda_yaqa_feedback_op

                corrected_tile_stack = _qvq_cuda_yaqa_feedback_op()(
                    source,
                    left_transformed_error,
                    right_transformed_error,
                    output_feedback,
                    coordinates[0][0],
                    coordinates[0][1],
                    len(coordinates),
                    rounding_bias,
                )
            elif incremental_cpu_factored_feedback:
                assert left_transformed_error is not None and right_transformed_error is not None
                from ..utils.qvq_cpu import qvq_cpu_yaqa_feedback

                corrected_tile_stack = qvq_cpu_yaqa_feedback(
                    source,
                    left_transformed_error,
                    right_transformed_error,
                    output_feedback,
                    coordinates[0][0],
                    coordinates[0][1],
                    len(coordinates),
                    rounding_bias,
                )
            else:
                corrected_tiles = []
                for input_block, output_block in coordinates:
                    input_start = input_block * tile_rows
                    input_stop = input_start + tile_rows
                    output_start = output_block * tile_cols
                    output_stop = output_start + tile_cols
                    left_feedback = input_feedback[input_start:, input_start:input_stop].transpose(0, 1)
                    right_feedback = output_feedback[output_start:, output_start:output_stop]
                    # The left-projected full suffix is already required by the
                    # two-sided term. Reuse its leading tile columns for the
                    # one-sided input term instead of issuing an identical
                    # reduction through a second, skinny GEMM.
                    left_projected_error = left_feedback @ error[input_start:, output_start:]
                    corrected_tiles.append(
                        source[input_start:input_stop, output_start:output_stop]
                        + (
                            0.0
                            if rounding_bias is None
                            else rounding_bias[input_start:input_stop, output_start:output_stop]
                        )
                        + left_projected_error @ right_feedback
                        + left_projected_error[:, :tile_cols]
                        + error[input_start:input_stop, output_start:] @ right_feedback
                    )
                corrected_tile_stack = torch.stack(corrected_tiles)
        sequences = corrected_tile_stack.reshape(len(coordinates), steps_per_tile, codebook.shape[1])
        logical_batch_size = sequences.shape[0]
        cuda_values_prevalidated = sequences.device.type == "cuda"
        if cuda_values_prevalidated:
            assert yaqa_cuda_invalid is not None and yaqa_cuda_safe_bound is not None
            yaqa_cuda_invalid.logical_or_(
                torch.logical_or(
                    ~torch.isfinite(sequences).all(),
                    sequences.detach().abs().amax() > yaqa_cuda_safe_bound,
                )
            )
        if dual_v2:
            sequences = _dual_v2_split(sequences)
        candidate_values, candidate_states = [], []
        segmented_selectors = None
        if segmented_v2:
            assert segmented_bank_stack is not None
            values, states_for_chunks, selectors = [], [], []
            segment_steps = (
                QVQ_V2B2_P32_STEPS_PER_SEGMENT if v2b2_p32 else QVQ_V2B4_P64_STEPS_PER_SEGMENT
            )
            # The native MLX recurrence assigns one independent threadgroup to
            # each tile.  Large low-rate suffix workspaces reduce Metal
            # occupancy before the unified-memory limit is reached: measured
            # W1--W2 throughput peaks around 32 tiles on M4 Max, whereas W2.5
            # and W3.5 benefit from the full 128-tile coalescing window.  Keep
            # explicit larger caller batches intact and retain the caller's
            # exact policy on every other backend.
            # Two bank-grid CTAs are launched per sequence. The historical
            # CUDA default of 16 therefore submits only 32 CTAs to 100+ SM
            # datacenter GPUs, leaving most of the device idle. Coalesce
            # independent tiles from one anti-diagonal into a one-wave launch;
            # explicit non-default caller batch policies remain authoritative.
            segmented_batch_size = _yaqa_segmented_batch_size(
                logical_batch_size,
                trellis_batch_size,
                bits,
                apple_host_feedback=apple_host_feedback,
                cuda_feedback=source.device.type == "cuda",
            )
            with _qvq_phase(telemetry, "yaqa_segmented_viterbi", source.device):
                for chunk in sequences.split(segmented_batch_size):
                    if telemetry is not None:
                        telemetry.count("yaqa_segmented_v2_chunks")
                    if apple_host_feedback:
                        from ..utils.qvq_mlx import (
                            qvq_mlx_tail_biting_v2_banked_from_torch_cpu,
                        )

                        assert host_segmented_bank_stack is not None and mlx_segmented_bank_stack is not None
                        states, segment_ids, squared_error = qvq_mlx_tail_biting_v2_banked_from_torch_cpu(
                            chunk.contiguous(),
                            host_segmented_bank_stack,
                            bits,
                            segment_steps=segment_steps,
                            mlx_codebooks=mlx_segmented_bank_stack,
                        )
                        path_banks = segment_ids.repeat_interleave(segment_steps, dim=1).to(torch.long)
                        result = BankedTrellisQuantizationResult(
                            states=states,
                            values=host_segmented_bank_stack[path_banks, states],
                            squared_error=squared_error,
                            segment_bank_ids=segment_ids,
                        )
                    else:
                        result = _tail_biting_v2_banked_quantize(
                            chunk,
                            segmented_bank_stack,
                            bits=bits,
                            segment_steps=segment_steps,
                            candidate_count=tail_biting_candidates,
                            viterbi_pruning=viterbi_pruning,
                            _cuda_values_prevalidated=cuda_values_prevalidated,
                        )
                    values.append(result.values.to(source.device))
                    states_for_chunks.append(result.states.to(source.device))
                    selectors.append(result.segment_bank_ids.to(source.device))
            reconstructed = torch.cat(values).reshape(len(coordinates), tile_rows, tile_cols)
            states = torch.cat(states_for_chunks)
            segmented_selectors = torch.cat(selectors)
            winners = torch.zeros(len(coordinates), dtype=torch.long, device=source.device)
        elif bank_codebooks is not None and sequences.device.type == "cuda" and tail_biting_candidates == 1:
            # Banked V4 YAQA has independent banks for each tile on an
            # anti-diagonal. Batch those banks into two native launches
            # (rotated provisional path plus constrained final path) instead
            # of four serial Python-side Viterbi schedules.
            from ..utils.qvq_cuda import qvq_cuda_viterbi_banked

            transition_bits = qvq_transition_bits(bits, vector_size=4)
            if transition_bits == 16:
                # Every V4 state is a complete transition at W4, so the
                # memoryless kernel already evaluates the tail-biting result.
                # There is no overlap boundary to estimate or constrain.
                constrained_states, _ = qvq_cuda_viterbi_banked(sequences, banked_codebooks, bits)
            else:
                midpoint = sequences.shape[1] // 2
                rotated = torch.roll(sequences, shifts=midpoint, dims=1).contiguous()
                provisional_states, _ = qvq_cuda_viterbi_banked(rotated, banked_codebooks, bits)
                overlap_bits = 16 - transition_bits
                overlap_mask = (1 << overlap_bits) - 1
                overlaps = (provisional_states[:, :, midpoint - 1] & overlap_mask).contiguous()
                constrained_states, _ = qvq_cuda_viterbi_banked(
                    sequences,
                    banked_codebooks,
                    bits,
                    overlap=overlaps,
                )
            candidate_states = [constrained_states[bank] for bank in range(len(active_banks))]
            candidate_values = [
                active_banks[bank][candidate_states[bank]]
                .reshape(len(coordinates), tile_rows, tile_cols)
                for bank in range(len(active_banks))
            ]
        else:
            for candidate_codebook in active_banks:
                values, states_for_bank = [], []
                with _qvq_phase(telemetry, "yaqa_viterbi", quantization_device):
                    if host_canonical_codebook_stack is not None:
                        from ..utils.qvq_mlx import (
                            qvq_mlx_tail_biting_v2_banked_from_torch_cpu,
                        )

                        assert mlx_canonical_codebook_stack is not None
                        apple_auto_batch = 32 if bits <= 2 or bits == 3 else 128
                        canonical_batch_size = max(
                            trellis_batch_size,
                            min(logical_batch_size, apple_auto_batch),
                        )
                        for chunk in sequences.split(canonical_batch_size):
                            states, _, _ = qvq_mlx_tail_biting_v2_banked_from_torch_cpu(
                                chunk.contiguous(),
                                host_canonical_codebook_stack,
                                bits,
                                segment_steps=steps_per_tile // 2,
                                mlx_codebooks=mlx_canonical_codebook_stack,
                            )
                            values.append(host_canonical_codebook_stack[0][states])
                            states_for_bank.append(states)
                    else:
                        canonical_batch_size = _yaqa_viterbi_batch_size(
                            sequences.shape[0],
                            trellis_batch_size,
                            bits,
                            cuda_feedback=sequences.device.type == "cuda",
                        )
                        for chunk in sequences.split(canonical_batch_size):
                            search_chunk = chunk.to(quantization_device) if apple_host_feedback else chunk
                            result = tail_biting_viterbi_quantize(
                                search_chunk,
                                candidate_codebook,
                                bits=bits,
                                candidate_count=tail_biting_candidates,
                                _cuda_values_prevalidated=cuda_values_prevalidated,
                            )
                            values.append(result.values.to(feedback_device))
                            states_for_bank.append(result.states.to(feedback_device))
                candidate_value = torch.cat(values)
                candidate_state = torch.cat(states_for_bank)
                if dual_v2:
                    candidate_value = _dual_v2_merge(candidate_value, logical_batch_size)
                    candidate_state = _dual_v2_merge(candidate_state, logical_batch_size)
                candidate_values.append(candidate_value.reshape(len(coordinates), tile_rows, tile_cols))
                candidate_states.append(candidate_state)
        if segmented_v2:
            pass
        elif bank_codebooks is None:
            reconstructed = candidate_values[0]
            states = candidate_states[0]
            winners = torch.zeros(len(coordinates), dtype=torch.long, device=source.device)
        else:
            candidates = torch.stack(candidate_values).to(torch.float32)
            tile_errors = corrected_tile_stack.unsqueeze(0) - candidates
            losses = []
            for candidate_index in range(candidates.shape[0]):
                candidate_losses = []
                for index, (input_block, output_block) in enumerate(coordinates):
                    # Score against the exact corrected tile supplied to the
                    # YAQA recurrence, not the untouched source tile. The
                    # distinction matters after an earlier anti-diagonal has
                    # committed quantized error.
                    tile_error = tile_errors[candidate_index, index]
                    input_block_hessian = input_hessian_blocks[input_block]
                    output_block_hessian = output_hessian_blocks[output_block]
                    candidate_losses.append(torch.einsum(
                        "ij,ik,kl,lj->",
                        tile_error,
                        input_block_hessian,
                        tile_error,
                        output_block_hessian,
                    ))
                losses.append(torch.stack(candidate_losses))
            winners = torch.stack(losses).argmin(dim=0)
            reconstructed = candidates[winners, torch.arange(len(coordinates), device=source.device)]
            states = torch.stack(candidate_states)[winners, torch.arange(len(coordinates), device=source.device)]
        # Native traceback gathers in the codebook storage dtype. YAQA's
        # feedback/error recurrence is FP32, and advanced indexed writes
        # require an exact dtype match rather than promoting implicitly.
        reconstructed = reconstructed.to(source.dtype)
        with _qvq_phase(telemetry, "yaqa_commit", source.device):
            quantized_blocks[input_indices, output_indices] = reconstructed
            error_blocks[input_indices, output_indices] = (
                source_blocks[input_indices, output_indices] - reconstructed
            )
            tile_states[input_indices, output_indices] = states
            if segmented_v2:
                bank_ids.view(input_blocks * output_blocks, segments_per_tile)[flat_tile_indices] = segmented_selectors
            elif bank_codebooks is not None:
                bank_ids[flat_tile_indices] = winners.to(torch.uint8)
        if incremental_cuda_feedback:
            assert feedback_temp is not None and transformed_error is not None
            with _qvq_phase(telemetry, "yaqa_feedback_update", source.device):
                # The anti-diagonal consists of independent 16x16 tiles. Form
                # their right projections as one batch, then concatenate the
                # matching input factors into one low-rank update. This avoids
                # materializing a mostly-zero committed matrix while retaining
                # the exact selected trellis artifact in the reference gate.
                diagonal_blocks = len(coordinates)
                left_factor = input_L[input_rows].transpose(0, 1)
                right_factor = torch.bmm(
                    reconstructed,
                    output_L[output_rows],
                ).reshape(diagonal_blocks * tile_rows, out_features)
                torch.addmm(
                    transformed_error,
                    left_factor,
                    right_factor,
                    beta=1,
                    alpha=-1,
                    out=transformed_error,
                )
        elif incremental_cuda_factored_feedback or incremental_cpu_factored_feedback:
            assert left_transformed_error is not None and right_transformed_error is not None
            with _qvq_phase(telemetry, "yaqa_feedback_update", source.device):
                # Maintain P = L_I'.T @ E and R = E @ L_O'. Distinct input
                # and output blocks make every destination disjoint within an
                # anti-diagonal. One native call therefore submits both
                # batched rank-16 updates directly into their strided cache
                # tiles, avoiding temporary bmm outputs and indexed scatters.
                if incremental_cpu_factored_feedback:
                    from ..utils.qvq_cpu import (
                        qvq_cpu_yaqa_feedback_update as _yaqa_feedback_update,
                    )
                else:
                    from ..utils.qvq_cuda import _qvq_cuda_yaqa_feedback_update_op

                    _yaqa_feedback_update = _qvq_cuda_yaqa_feedback_update_op()

                _yaqa_feedback_update(
                    left_transformed_error,
                    right_transformed_error,
                    input_feedback,
                    output_feedback,
                    reconstructed.contiguous(),
                    coordinates[0][0],
                    coordinates[0][1],
                    len(coordinates),
                )

    if (
        yaqa_cuda_invalid is not None
        and not _defer_segmented_cuda_checks
        and bool(yaqa_cuda_invalid)
    ):
        raise ValueError(
            "YAQA corrected segmented-V2 tiles exceeded finite FP32 squared-distance range."
        )

    if bank_codebooks is not None and not _defer_segmented_cuda_checks:
        with _qvq_phase(telemetry, "yaqa_full_proxy", source.device):
            mixed_error = quantized.to(torch.float32) - source
            bank0_error = bank0_reference.to(device=feedback_device, dtype=torch.float32) - source
            mixed_loss = torch.einsum(
                "ij,ik,kl,lj->", mixed_error, input_hessian_fp32, mixed_error, output_hessian_fp32
            )
            bank0_loss = torch.einsum(
                "ij,ik,kl,lj->", bank0_error, input_hessian_fp32, bank0_error, output_hessian_fp32
            )
        fallback_to_bank0 = not torch.isfinite(mixed_loss) or mixed_loss >= bank0_loss
        if _diagnostics is not None:
            bank0_states = bank0_reference_states.to(feedback_device).reshape_as(tile_states)
            _diagnostics["mixed_loss_before_fallback"] = float(mixed_loss.item())
            _diagnostics["bank0_loss"] = float(bank0_loss.item())
            _diagnostics["pre_fallback_selector_churn"] = float(
                (bank_ids != 0).to(torch.float32).mean().item()
            )
            _diagnostics["pre_fallback_state_churn"] = float(
                (tile_states != bank0_states).to(torch.float32).mean().item()
            )
        if fallback_to_bank0:
            quantized = bank0_reference.to(device=feedback_device, dtype=inner_weight.dtype)
            tile_states = bank0_reference_states.to(feedback_device).reshape(
                input_blocks, output_blocks, steps_per_tile
            )
            bank_ids.zero_()
        if telemetry is not None:
            telemetry.count("yaqa_bank0_fallback", int(fallback_to_bank0))
        if _diagnostics is not None:
            _diagnostics["fallback_to_bank0"] = bool(fallback_to_bank0)
    elif _diagnostics is not None and yaqa_cuda_invalid is not None:
        _diagnostics["_segmented_cuda_invalid"] = yaqa_cuda_invalid
    result = (
        quantized.to(device=quantization_device, dtype=inner_weight.dtype),
        tile_states.reshape(-1, steps_per_tile).to(quantization_device),
    )
    return (*result, bank_ids.to(quantization_device)) if bank_codebooks is not None else result


def yaqa_inner_v2b4_p64(
    inner_weight: torch.Tensor,
    input_hessian: torch.Tensor,
    output_hessian: torch.Tensor,
    codebooks: tuple[torch.Tensor, ...],
    segmented_bank_stack: torch.Tensor | None = None,
    block_input_hessian: torch.Tensor | None = None,
    diagnostics: dict[str, object] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run YAQA corrected-target rounding with four P64 V2 banks."""

    if len(codebooks) != 4:
        raise ValueError("YAQA V2B4-P64 requires four codebooks.")
    _, _, block_selectors = block_ldlq_inner_v2b4_p64(
        inner_weight,
        input_hessian if block_input_hessian is None else block_input_hessian,
        codebooks,
        bits=kwargs["bits"],
        tile_rows=kwargs.get("tile_rows", 16),
        tile_cols=kwargs.get("tile_cols", 16),
        trellis_batch_size=kwargs.get("trellis_batch_size", 16),
        viterbi_objective="euclidean",
        tail_biting_candidates=kwargs.get("tail_biting_candidates", 1),
        viterbi_pruning=kwargs.get("viterbi_pruning"),
    )
    local_diagnostics: dict[str, object] = {}
    weight, states, selectors = yaqa_inner(
        inner_weight,
        input_hessian,
        output_hessian,
        codebooks[0],
        bank_codebooks=codebooks,
        segmented_bank_stack=segmented_bank_stack,
        v2b4_p64=True,
        _diagnostics=local_diagnostics,
        **kwargs,
    )
    if diagnostics is not None:
        diagnostics["fallback_to_v2"] = bool(local_diagnostics.get("fallback_to_bank0", False))
        diagnostics["selector_churn"] = float((selectors != block_selectors).to(torch.float32).mean().item())
    return weight, states, selectors


def _yaqa_inner_v2b2_family_batch_cuda(
    inner_weight: torch.Tensor,
    input_hessian: torch.Tensor,
    output_hessian: torch.Tensor,
    family_stacks: torch.Tensor,
    *,
    bits: float,
    factorization: tuple[BlockLDLFactorization, BlockLDLFactorization] | None,
    rounding_bias: torch.Tensor | None,
    telemetry: QVQQuantizationTelemetry | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run independent B2 YAQA histories through one candidate-batched CUDA schedule."""

    families, in_features, out_features = family_stacks.shape[0], *inner_weight.shape
    tile_rows = tile_cols = 16
    input_blocks, output_blocks = in_features // tile_rows, out_features // tile_cols
    steps = tile_rows * tile_cols // 2
    if factorization is None:
        input_L, _ = block_ldl_factor(input_hessian.to(torch.float32), block_size=tile_rows)
        output_L, _ = block_ldl_factor(output_hessian.to(torch.float32), block_size=tile_cols)
    else:
        input_L, output_L = factorization[0].L, factorization[1].L
    input_feedback = input_L.clone().contiguous()
    output_feedback = output_L.clone().contiguous()
    input_feedback.diagonal().sub_(1)
    output_feedback.diagonal().sub_(1)
    source = inner_weight.to(torch.float32)
    bias = None if rounding_bias is None else rounding_bias.to(torch.float32)

    # Every candidate starts from the same E=W. Copy one mathematically
    # identical product so candidate batching cannot alter the initial GEMM.
    left_base = input_feedback.transpose(0, 1) @ source
    right_base = source @ output_feedback
    left = left_base.unsqueeze(0).expand(families, -1, -1).clone()
    right = right_base.unsqueeze(0).expand(families, -1, -1).clone()
    quantized = torch.zeros((families, in_features, out_features), device=source.device, dtype=torch.float32)
    quantized_blocks = quantized.view(families, input_blocks, tile_rows, output_blocks, tile_cols).permute(0, 1, 3, 2, 4)
    tile_states = torch.empty(
        (families, input_blocks, output_blocks, steps), device=source.device, dtype=torch.long
    )
    selectors = torch.empty(
        (families, input_blocks * output_blocks, QVQ_V2B2_P32_SEGMENTS_PER_TILE),
        device=source.device,
        dtype=torch.uint8,
    )
    invalid = torch.zeros((families,), device=source.device, dtype=torch.bool)
    safe_bound = math.sqrt(torch.finfo(torch.float32).max / steps) / (2.0 * math.sqrt(2.0))
    invalid.logical_or_(family_stacks.detach().abs().amax(dim=(1, 2, 3)) > safe_bound)
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    midpoint = steps // 2
    overlap_mask = (1 << (16 - transition_bits)) - 1

    from ..utils.qvq_cuda import (
        _qvq_cuda_viterbi_v2_segment_family_grid_trusted_op,
        _qvq_cuda_yaqa_feedback_op,
        _qvq_cuda_yaqa_feedback_update_op,
    )

    family_viterbi = _qvq_cuda_viterbi_v2_segment_family_grid_trusted_op()
    schedule = _yaqa_anti_diagonal_schedule(source.device, input_blocks, output_blocks)
    family_indices = torch.arange(families, device=source.device).view(families, 1, 1)
    for coordinates, input_indices, output_indices, flat_indices, _, _ in schedule:
        count = len(coordinates)
        with _qvq_phase(telemetry, "yaqa_feedback", source.device):
            corrected = _qvq_cuda_yaqa_feedback_op()(
                source, left, right, output_feedback, coordinates[0][0], coordinates[0][1], count, bias
            )
        sequences = corrected.reshape(families, count, steps, 2)
        invalid.logical_or_(
            torch.logical_or(~torch.isfinite(sequences).all(dim=(1, 2, 3)), sequences.abs().amax(dim=(1, 2, 3)) > safe_bound)
        )
        with _qvq_phase(telemetry, "yaqa_segmented_viterbi", source.device):
            provisional, _, _ = family_viterbi(
                torch.roll(sequences, shifts=midpoint, dims=2).contiguous(),
                family_stacks,
                transition_bits,
                QVQ_V2B2_P32_STEPS_PER_SEGMENT,
                None,
                None,
            )
            overlaps = (provisional[:, :, midpoint - 1] & overlap_mask).contiguous()
            if telemetry is not None:
                family_sequences = families * count
                telemetry.count("viterbi_family_grid_calls", 2)
                telemetry.count("viterbi_logical_solve_ids", 2)
                telemetry.count("viterbi_unique_logical_solve_ids", 2)
                telemetry.count("viterbi_family_grid_sequences", family_sequences * 2)
                telemetry.count("viterbi_family_state_steps", family_sequences * steps * 2)
                telemetry.count("viterbi_provisional_states_produced", family_sequences * steps)
                telemetry.count("viterbi_provisional_states_consumed", family_sequences)
                telemetry.count("viterbi_provisional_losses_discarded", family_sequences)
                telemetry.count("viterbi_provisional_selectors_discarded", family_sequences * 8)
                # Anti-diagonal coordinates are disjoint and the feedback
                # tensors mutate after every commit; exact reuse is impossible.
                telemetry.count("viterbi_exact_reuse_candidates", 0)
                telemetry.count("viterbi_reselection_revisits", 0)
            states, _, segment_ids = family_viterbi(
                sequences,
                family_stacks,
                transition_bits,
                QVQ_V2B2_P32_STEPS_PER_SEGMENT,
                overlaps,
                None,
            )
        path_banks = segment_ids.to(torch.long).repeat_interleave(QVQ_V2B2_P32_STEPS_PER_SEGMENT, dim=2)
        reconstructed = family_stacks[family_indices, path_banks, states].reshape(families, count, 16, 16).to(torch.float32)
        with _qvq_phase(telemetry, "yaqa_commit", source.device):
            quantized_blocks[:, input_indices, output_indices] = reconstructed
            tile_states[:, input_indices, output_indices] = states
            selectors[:, flat_indices] = segment_ids
        with _qvq_phase(telemetry, "yaqa_feedback_update", source.device):
            _qvq_cuda_yaqa_feedback_update_op()(
                left,
                right,
                input_feedback,
                output_feedback,
                reconstructed.contiguous(),
                coordinates[0][0],
                coordinates[0][1],
                count,
            )
    return (
        quantized.to(inner_weight.dtype),
        tile_states.reshape(families, -1, steps),
        selectors.reshape(families, -1),
        invalid,
    )


def yaqa_inner_v2b2_p32(
    inner_weight: torch.Tensor,
    input_hessian: torch.Tensor,
    output_hessian: torch.Tensor,
    codebook_library: tuple[torch.Tensor, ...],
    bank_codebook_pair_stacks: tuple[torch.Tensor, ...] | None = None,
    family_mode: str = "reselect",
    sample_strategy: str = "full",
    block_input_hessian: torch.Tensor | None = None,
    block_family_id: int | None = None,
    diagnostics: dict[str, object] | None = None,
    _parallel_candidates: bool = True,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select one complementary V2 family per module under YAQA's full proxy."""

    telemetry = kwargs.get("telemetry")
    # Dense transformed-error updates win decisively on attention-sized
    # matrices, but their cubic GEMMs only break even on very wide MLP
    # projections and retain substantially more workspace. Keep the measured
    # 2048-dimension envelope explicit until a sparse update kernel extends it.
    kwargs.setdefault(
        "_incremental_cuda_feedback",
        inner_weight.device.type == "cuda" and max(inner_weight.shape) <= 2048,
    )
    kwargs.setdefault(
        "_incremental_cpu_factored_feedback",
        inner_weight.device.type in ("cpu", "mps") and qvq_cpu_supported(),
    )

    if len(codebook_library) != 4:
        raise ValueError("YAQA V2B2-P32 requires canonical V2 plus three complementary candidates.")
    if bank_codebook_pair_stacks is not None and (
        len(bank_codebook_pair_stacks) != 3
        or any(tuple(stack.shape) != (2, 1 << 16, 2) for stack in bank_codebook_pair_stacks)
    ):
        raise ValueError("YAQA V2B2-P32 pair stacks must contain three [2, 65536, 2] tensors.")
    if family_mode not in {"fixed_block_ldlq", "reselect"}:
        raise ValueError("YAQA V2B2-P32 family mode must be `fixed_block_ldlq` or `reselect`.")
    if sample_strategy not in QVQ_YAQA_SAMPLE_TILE_COUNTS:
        raise ValueError(
            "YAQA sample strategy must be `full`, `32_16x16`, `64_16x16`, `96_16x16`, `128_16x16`, "
            "or `256_16x16`."
        )
    if family_mode != "reselect" and sample_strategy != "full":
        raise ValueError("YAQA sampled family selection requires `family_mode=reselect`.")
    if telemetry is not None:
        telemetry.count("yaqa_v2b2_modules")
        telemetry.count("yaqa_v2b2_reselect_modules", int(family_mode == "reselect"))
    if sample_strategy != "full":
        input_blocks = inner_weight.shape[0] // 16
        output_blocks = inner_weight.shape[1] // 16
        tile_count = input_blocks * output_blocks
        sample_indices_cpu = _yaqa_sample_tile_indices(tile_count, sample_strategy)
        sample_count = sample_indices_cpu.numel()
        sample_indices = sample_indices_cpu.to(inner_weight.device)
        sample_input_blocks = torch.div(sample_indices, output_blocks, rounding_mode="floor")
        sample_output_blocks = sample_indices.remainder(output_blocks)
        source_tiles_device = (
            inner_weight.view(input_blocks, 16, output_blocks, 16)
            .permute(0, 2, 1, 3)[sample_input_blocks, sample_output_blocks]
            .to(torch.float32)
        ).contiguous()
        source_tiles = source_tiles_device.to("cpu")
        input_blocks_h = torch.stack(
            [
                input_hessian[index * 16 : (index + 1) * 16, index * 16 : (index + 1) * 16]
                for index in torch.div(sample_indices_cpu, output_blocks, rounding_mode="floor").tolist()
            ]
        ).to("cpu", torch.float32)
        output_blocks_h = torch.stack(
            [
                output_hessian[index * 16 : (index + 1) * 16, index * 16 : (index + 1) * 16]
                for index in sample_indices_cpu.remainder(output_blocks).tolist()
            ]
        ).to("cpu", torch.float32)
        family_losses = []
        with _qvq_phase(telemetry, "yaqa_v2b2_sampled_family_selection", inner_weight.device):
            pair_stacks = tuple(
                (
                    torch.stack((codebook_library[0], codebook_library[alt_id])).contiguous()
                    if bank_codebook_pair_stacks is None
                    else bank_codebook_pair_stacks[alt_id - 1]
                )
                for alt_id in (1, 2, 3)
            )
            if inner_weight.device.type == "cuda" and kwargs.get("tail_biting_candidates", 1) == 1:
                from ..utils.qvq_cuda import (
                    _qvq_cuda_viterbi_v2_segment_family_grid_trusted_op,
                )

                family_codebooks = torch.stack(pair_stacks).contiguous()
                family_sequences = (
                    source_tiles_device.reshape(sample_count, 128, 2)
                    .unsqueeze(0)
                    .expand(3, -1, -1, -1)
                    .contiguous()
                )
                transition_bits = qvq_transition_bits(kwargs["bits"], vector_size=2)
                midpoint = family_sequences.shape[2] // 2
                provisional_states, _, _ = _qvq_cuda_viterbi_v2_segment_family_grid_trusted_op()(
                    torch.roll(family_sequences, shifts=midpoint, dims=2).contiguous(),
                    family_codebooks,
                    transition_bits,
                    QVQ_V2B2_P32_STEPS_PER_SEGMENT,
                    None,
                    None,
                )
                overlap_mask = (1 << (16 - transition_bits)) - 1
                overlaps = (provisional_states[:, :, midpoint - 1] & overlap_mask).contiguous()
                if telemetry is not None:
                    family_sequences_count = 3 * sample_count
                    telemetry.count("viterbi_family_grid_calls", 2)
                    telemetry.count("viterbi_logical_solve_ids", 2)
                    telemetry.count("viterbi_unique_logical_solve_ids", 2)
                    telemetry.count("viterbi_family_grid_sequences", family_sequences_count * 2)
                    telemetry.count("viterbi_family_state_steps", family_sequences_count * 128 * 2)
                    telemetry.count("viterbi_provisional_states_produced", family_sequences_count * 128)
                    telemetry.count("viterbi_provisional_states_consumed", family_sequences_count)
                    telemetry.count("viterbi_provisional_losses_discarded", family_sequences_count)
                    telemetry.count("viterbi_provisional_selectors_discarded", family_sequences_count * 8)
                    telemetry.count("viterbi_exact_reuse_candidates", 0)
                    telemetry.count("viterbi_reselection_revisits", 0)
                family_states, _, family_selectors = _qvq_cuda_viterbi_v2_segment_family_grid_trusted_op()(
                    family_sequences,
                    family_codebooks,
                    transition_bits,
                    QVQ_V2B2_P32_STEPS_PER_SEGMENT,
                    overlaps,
                    None,
                )
                path_banks = family_selectors.to(torch.long).repeat_interleave(
                    QVQ_V2B2_P32_STEPS_PER_SEGMENT,
                    dim=2,
                )
                family_indices = torch.arange(3, device=inner_weight.device).view(3, 1, 1)
                family_values = family_codebooks[family_indices, path_banks, family_states]
                for family_index in range(3):
                    error = (
                        family_values[family_index]
                        .to("cpu", torch.float32)
                        .reshape(sample_count, 16, 16)
                        - source_tiles
                    )
                    family_losses.append(
                        torch.einsum("bij,bik,bkl,blj->", error, input_blocks_h, error, output_blocks_h)
                    )
            else:
                for pair_stack in pair_stacks:
                    result = _tail_biting_v2_banked_quantize(
                        source_tiles_device.reshape(sample_count, 128, 2),
                        pair_stack,
                        bits=kwargs["bits"],
                        segment_steps=QVQ_V2B2_P32_STEPS_PER_SEGMENT,
                        candidate_count=kwargs.get("tail_biting_candidates", 1),
                        viterbi_pruning=kwargs.get("viterbi_pruning"),
                    )
                    error = result.values.to("cpu", torch.float32).reshape(sample_count, 16, 16) - source_tiles
                    family_losses.append(
                        torch.einsum("bij,bik,bkl,blj->", error, input_blocks_h, error, output_blocks_h)
                    )
        block_alt_id = int(torch.stack(family_losses).argmin().item()) + 1
        block_selectors = None
        if telemetry is not None:
            telemetry.count("yaqa_v2b2_sampled_family_tiles", sample_count)
            telemetry.count(f"yaqa_v2b2_sample_strategy_{sample_strategy}")
            telemetry.count(f"yaqa_v2b2_sampled_family_{block_alt_id}")
    elif block_family_id is None:
        with _qvq_phase(telemetry, "yaqa_v2b2_block_family_selection", inner_weight.device):
            _, _, block_selectors, block_alt_id_tensor = block_ldlq_inner_v2b2_p32(
                inner_weight,
                input_hessian if block_input_hessian is None else block_input_hessian,
                codebook_library,
                bits=kwargs["bits"],
                tile_rows=kwargs.get("tile_rows", 16),
                tile_cols=kwargs.get("tile_cols", 16),
                trellis_batch_size=kwargs.get("trellis_batch_size", 16),
                viterbi_objective="euclidean",
                tail_biting_candidates=kwargs.get("tail_biting_candidates", 1),
                bank_codebook_pair_stacks=bank_codebook_pair_stacks,
                viterbi_pruning=kwargs.get("viterbi_pruning"),
            )
        block_alt_id = int(block_alt_id_tensor.item())
    else:
        if isinstance(block_family_id, bool) or not isinstance(block_family_id, int) or block_family_id not in (1, 2, 3):
            raise ValueError("YAQA V2B2-P32 cached Block-LDLQ family ID must be 1, 2, or 3.")
        block_alt_id = block_family_id
        block_selectors = None
    alternative_ids = (block_alt_id,) if family_mode != "reselect" or sample_strategy != "full" else (1, 2, 3)
    parallel_families = (
        _parallel_candidates
        and inner_weight.device.type == "cuda"
        and len(alternative_ids) >= 1
        and kwargs.get("_incremental_cuda_factored_feedback", False)
    )
    current_stream = None
    canonical_completion = None
    canonical_diagnostics: dict[str, object] = {}
    family_streams: tuple[torch.cuda.Stream, ...] = ()
    if parallel_families:
        current_stream = torch.cuda.current_stream(inner_weight.device)
        _yaqa_anti_diagonal_schedule(
            inner_weight.device,
            inner_weight.shape[0] // kwargs.get("tile_rows", 16),
            inner_weight.shape[1] // kwargs.get("tile_cols", 16),
            kwargs.get("tile_rows", 16),
            kwargs.get("tile_cols", 16),
        )
        candidate_streams = _yaqa_family_streams(inner_weight.device, len(alternative_ids) + 1)
        canonical_stream, family_streams = candidate_streams[0], candidate_streams[1:]
        canonical_stream.wait_stream(current_stream)
        with torch.cuda.stream(canonical_stream):
            with _qvq_phase(telemetry, "yaqa_v2b2_canonical", inner_weight.device):
                canonical_weight, canonical_states = yaqa_inner(
                    inner_weight,
                    input_hessian,
                    output_hessian,
                    codebook_library[0],
                    _diagnostics=canonical_diagnostics,
                    _defer_segmented_cuda_checks=True,
                    **kwargs,
                )
            canonical_completion = torch.cuda.Event(enable_timing=False, blocking=False)
            canonical_completion.record(canonical_stream)
    else:
        with _qvq_phase(telemetry, "yaqa_v2b2_canonical", inner_weight.device):
            canonical_weight, canonical_states = yaqa_inner(
                inner_weight,
                input_hessian,
                output_hessian,
                codebook_library[0],
                **kwargs,
            )
    source = inner_weight.to(torch.float32)
    input_hessian_fp32 = input_hessian.to(torch.float32)
    output_hessian_fp32 = output_hessian.to(torch.float32)

    def full_loss(candidate: torch.Tensor) -> torch.Tensor:
        error = candidate.to(torch.float32) - source
        return torch.einsum("ij,ik,kl,lj->", error, input_hessian_fp32, error, output_hessian_fp32)

    best_weight = canonical_weight
    best_states = canonical_states
    best_selectors = torch.zeros(
        canonical_states.shape[0] * QVQ_V2B2_P32_SEGMENTS_PER_TILE,
        dtype=torch.uint8,
        device=inner_weight.device,
    )
    # The family ID remains the Block-LDLQ choice when YAQA falls back to
    # canonical V2. This makes fixed-family mode a strict experiment over one
    # unchanged codec candidate space; selectors alone become all zero.
    best_alt_id = block_alt_id
    best_loss = None
    if not parallel_families:
        with _qvq_phase(telemetry, "yaqa_v2b2_full_proxy", inner_weight.device):
            best_loss = full_loss(canonical_weight)
    selected_banked_candidate = False
    oracle = canonical_weight, canonical_states
    family_diagnostics: dict[str, dict[str, object]] = {}
    if parallel_families:
        assert current_stream is not None and canonical_completion is not None
        family_stream = family_streams[0]
        pair_stacks = torch.stack(
            tuple(
                torch.stack((codebook_library[0], codebook_library[alt_id])).contiguous()
                if bank_codebook_pair_stacks is None
                else bank_codebook_pair_stacks[alt_id - 1]
                for alt_id in alternative_ids
            )
        ).contiguous()
        if telemetry is not None:
            telemetry.count("yaqa_v2b2_family_candidates", len(alternative_ids))
            telemetry.count("yaqa_v2b2_candidate_batches")
        family_stream.wait_stream(current_stream)
        with torch.cuda.stream(family_stream):
            with _qvq_phase(telemetry, "yaqa_v2b2_family_candidate", inner_weight.device):
                candidate_weights, candidate_states, candidate_selectors, invalid = (
                    _yaqa_inner_v2b2_family_batch_cuda(
                        inner_weight,
                        input_hessian,
                        output_hessian,
                        pair_stacks,
                        bits=kwargs["bits"],
                        factorization=kwargs.get("factorization"),
                        rounding_bias=kwargs.get("_rounding_bias"),
                        telemetry=telemetry,
                    )
                )
            candidate_losses = []
            with _qvq_phase(telemetry, "yaqa_v2b2_full_proxy", inner_weight.device):
                for family_index in range(len(alternative_ids)):
                    candidate_losses.append(full_loss(candidate_weights[family_index]))
            completion = torch.cuda.Event(enable_timing=False, blocking=False)
            completion.record(family_stream)
        candidate_records = [
            (
                alt_id,
                candidate_weights[index],
                candidate_states[index],
                candidate_selectors[index],
                candidate_losses[index],
                {"_segmented_cuda_invalid": invalid[index]},
                completion,
            )
            for index, alt_id in enumerate(alternative_ids)
        ]

        current_stream.wait_event(canonical_completion)
        for *_, completion in candidate_records:
            current_stream.wait_event(completion)
        canonical_weight.record_stream(current_stream)
        canonical_states.record_stream(current_stream)
        with _qvq_phase(telemetry, "yaqa_v2b2_full_proxy", inner_weight.device):
            best_loss = full_loss(canonical_weight)
        invalid_flags = [
            canonical_diagnostics["_segmented_cuda_invalid"],
            *(record[5]["_segmented_cuda_invalid"] for record in candidate_records),
        ]
        if bool(torch.stack(invalid_flags).any()):
            raise ValueError(
                "YAQA corrected segmented-V2 tiles exceeded finite FP32 squared-distance range."
            )
        candidate_losses = torch.stack([record[4] for record in candidate_records])
        finite_losses = torch.where(torch.isfinite(candidate_losses), candidate_losses, torch.inf)
        assert best_loss is not None
        all_losses = torch.cat((best_loss.reshape(1), finite_losses))
        winner_index = int(all_losses.argmin().item())
        canonical_loss_value = float(best_loss.item())
        canonical_states_view = canonical_states.reshape_as(candidate_records[0][2])
        for record_index, (
            alt_id,
            candidate_weight,
            candidate_states,
            candidate_selectors,
            candidate_loss,
            _,
            _,
        ) in enumerate(candidate_records, start=1):
            for tensor in (candidate_weight, candidate_states, candidate_selectors, candidate_loss):
                tensor.record_stream(current_stream)
            candidate_loss_value = float(candidate_loss.item())
            fallback = not math.isfinite(candidate_loss_value) or candidate_loss_value >= canonical_loss_value
            family_diagnostics[str(alt_id)] = {
                "fallback_to_bank0": fallback,
                "mixed_loss_before_fallback": candidate_loss_value,
                "bank0_loss": canonical_loss_value,
                "pre_fallback_selector_churn": float(
                    (candidate_selectors != 0).to(torch.float32).mean().item()
                ),
                "pre_fallback_state_churn": float(
                    (candidate_states != canonical_states_view).to(torch.float32).mean().item()
                ),
            }
            if telemetry is not None:
                telemetry.count("yaqa_bank0_fallback", int(fallback))
            if record_index == winner_index:
                best_weight = candidate_weight
                best_states = candidate_states
                best_selectors = candidate_selectors
                best_alt_id = alt_id
                best_loss = candidate_loss
                selected_banked_candidate = True
    else:
        assert best_loss is not None
        for alt_id in alternative_ids:
            if telemetry is not None:
                telemetry.count("yaqa_v2b2_family_candidates")
            pair_stack = (
                torch.stack((codebook_library[0], codebook_library[alt_id])).contiguous()
                if bank_codebook_pair_stacks is None
                else bank_codebook_pair_stacks[alt_id - 1]
            )
            pair_codebooks = tuple(pair_stack[bank] for bank in range(2))
            inner_diagnostics: dict[str, object] = {}
            with _qvq_phase(telemetry, "yaqa_v2b2_family_candidate", inner_weight.device):
                candidate_weight, candidate_states, candidate_selectors = yaqa_inner(
                    inner_weight,
                    input_hessian,
                    output_hessian,
                    pair_codebooks[0],
                    bank_codebooks=pair_codebooks,
                    segmented_bank_stack=pair_stack,
                    v2b2_p32=True,
                    _bank0_oracle=oracle,
                    _diagnostics=inner_diagnostics,
                    **kwargs,
                )
            with _qvq_phase(telemetry, "yaqa_v2b2_full_proxy", inner_weight.device):
                candidate_loss = full_loss(candidate_weight)
            family_diagnostics[str(alt_id)] = {
                "fallback_to_bank0": bool(inner_diagnostics.get("fallback_to_bank0", False)),
                "mixed_loss_before_fallback": inner_diagnostics.get("mixed_loss_before_fallback"),
                "bank0_loss": inner_diagnostics.get("bank0_loss"),
                "pre_fallback_selector_churn": inner_diagnostics.get("pre_fallback_selector_churn"),
                "pre_fallback_state_churn": inner_diagnostics.get("pre_fallback_state_churn"),
            }
            if torch.isfinite(candidate_loss) and candidate_loss < best_loss:
                best_weight = candidate_weight
                best_states = candidate_states
                best_selectors = candidate_selectors
                best_alt_id = alt_id
                best_loss = candidate_loss
                selected_banked_candidate = True
    if diagnostics is not None:
        diagnostics["fallback_to_v2"] = not selected_banked_candidate
        if block_selectors is not None:
            diagnostics["selector_churn"] = float(
                (best_selectors != block_selectors).to(torch.float32).mean().item()
            )
        diagnostics["family_changed"] = bool(selected_banked_candidate and best_alt_id != block_alt_id)
        diagnostics["block_family_id"] = block_alt_id
        diagnostics["family_candidates"] = family_diagnostics
    return (
        best_weight,
        best_states,
        best_selectors,
        torch.tensor([best_alt_id], dtype=torch.uint8, device=inner_weight.device),
    )


def yaqa_output_spectral_refine_v2b2_p32(
    inner_weight: torch.Tensor,
    input_hessian: torch.Tensor,
    output_hessian: torch.Tensor,
    codebook_library: tuple[torch.Tensor, ...],
    baseline: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    ranks: tuple[int, ...],
    lambdas: tuple[float, ...],
    family_mode: str = "reselect",
    block_input_hessian: torch.Tensor | None = None,
    block_family_id: int | None = None,
    factorization: tuple[BlockLDLFactorization, BlockLDLFactorization] | None = None,
    diagnostics: dict[str, object] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Refine V2B2-P32 using output modes of its post-YAQA residual.

    The boosted factors generate candidates only. Every candidate is rescored
    with the original, unmodified YAQA Kronecker objective, and ``baseline``
    remains the exact rollback artifact. No spectral tensor is serialized.
    """

    if not ranks or any(isinstance(rank, bool) or not isinstance(rank, int) or rank < 1 for rank in ranks):
        raise ValueError("YAQA spectral ranks must be positive integers.")
    if not lambdas or any(
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(float(strength))
        or float(strength) <= 0
        for strength in lambdas
    ):
        raise ValueError("YAQA spectral lambdas must be finite and positive.")
    baseline_weight, _baseline_states, baseline_selectors, baseline_alt_id = baseline
    source = inner_weight.to(torch.float32)
    original_input = input_hessian.to(torch.float32)
    original_output = output_hessian.to(torch.float32)
    residual = source - baseline_weight.to(torch.float32)

    # For inner orientation E[in, out], YAQA's quadratic is
    # ||C_I.T @ E @ C_O||_F^2 when H_I=C_I C_I.T and H_O=C_O C_O.T.
    input_root = torch.linalg.cholesky(original_input)
    output_root = torch.linalg.cholesky(original_output)
    whitened_residual = input_root.transpose(0, 1) @ residual @ output_root
    residual_energy = whitened_residual.square().sum()
    maximum_rank = min(max(ranks), min(whitened_residual.shape))
    from ..eora.eora import _eora_compute_svd

    # MPS QR/SVD is pathological for the tall rank-focused work matrices used
    # here (minutes for 2048-wide projections versus milliseconds on the P
    # cores). The spectrum is quantization-time-only, so move this temporary
    # FP32 matrix to CPU and return only the compact factors to MPS.
    spectral_device = torch.device("cpu") if whitened_residual.device.type == "mps" else whitened_residual.device
    svd_source = whitened_residual.to(device=spectral_device)
    left_vectors, singular_values, right_vectors_h = _eora_compute_svd(
        svd_source,
        maximum_rank,
        algo="lowrank",
    )
    del left_vectors
    singular_values = singular_values.to(device=whitened_residual.device)
    right_vectors_h = right_vectors_h.to(device=whitened_residual.device)
    usable_ranks = tuple(sorted({min(rank, singular_values.numel()) for rank in ranks}))
    eps = torch.finfo(torch.float32).eps
    output_trace = original_output.diagonal().sum().clamp_min(eps)

    def original_loss(candidate: torch.Tensor) -> torch.Tensor:
        error = candidate.to(torch.float32) - source
        return torch.einsum("ij,ik,kl,lj->", error, original_input, error, original_output)

    best = baseline
    baseline_loss = original_loss(baseline_weight)
    best_loss = baseline_loss
    best_rank = None
    best_lambda = None
    best_candidate_diagnostics: dict[str, object] | None = None
    concentrations = {}
    top_energies = {}
    for rank in usable_ranks:
        top_energy = singular_values[:rank].square().sum()
        top_energies[rank] = top_energy
        concentrations[str(rank)] = float((top_energy / residual_energy.clamp_min(eps)).item())
        output_modes = right_vectors_h[:rank].transpose(0, 1)
        spectral_output = output_root @ output_modes
        spectral_output = spectral_output @ spectral_output.transpose(0, 1)
        spectral_trace = spectral_output.diagonal().sum()
        if not torch.isfinite(spectral_trace) or spectral_trace <= eps:
            continue
        spectral_output = spectral_output * (output_trace / spectral_trace)
        for strength in lambdas:
            boosted_output = original_output + float(strength) * spectral_output
            candidate_factorization = None
            if factorization is not None:
                retry_damping = torch.maximum(
                    boosted_output.diagonal().abs().mean() * YAQA_PAPER_REGULARIZATION,
                    torch.tensor(eps, device=boosted_output.device),
                )
                candidate_factorization = (
                    factorization[0],
                    stabilized_block_ldl_factor(
                        boosted_output,
                        block_size=factorization[1].block_size,
                        retry_damping=retry_damping,
                    ),
                )
            candidate_diagnostics: dict[str, object] = {}
            candidate = yaqa_inner_v2b2_p32(
                inner_weight,
                input_hessian,
                boosted_output,
                codebook_library,
                family_mode=family_mode,
                block_input_hessian=block_input_hessian,
                block_family_id=block_family_id,
                diagnostics=candidate_diagnostics,
                factorization=candidate_factorization,
                **kwargs,
            )
            candidate_loss = original_loss(candidate[0])
            if torch.isfinite(candidate_loss) and candidate_loss < best_loss:
                best = candidate
                best_loss = candidate_loss
                best_rank = rank
                best_lambda = float(strength)
                best_candidate_diagnostics = candidate_diagnostics

    if diagnostics is not None:
        diagnostics["spectral_method"] = "output_factor"
        diagnostics["spectral_alpha"] = None
        diagnostics["spectral_svd_device"] = spectral_device.type
        diagnostics["spectral_concentration"] = concentrations
        diagnostics["spectral_selected"] = best_rank is not None
        diagnostics["spectral_rank"] = best_rank
        diagnostics["spectral_lambda"] = best_lambda
        diagnostics["spectral_original_loss"] = float(baseline_loss.item())
        diagnostics["spectral_selected_loss"] = float(best_loss.item())
        if best_rank is not None:
            removable = top_energies[best_rank].clamp_min(eps)
            diagnostics["spectral_absorption_efficiency"] = float(
                ((baseline_loss - best_loss) / removable).item()
            )
            diagnostics["spectral_selector_churn"] = float(
                (best[2] != baseline_selectors).to(torch.float32).mean().item()
            )
            diagnostics["spectral_family_changed"] = bool(
                int(best[3].item()) != int(baseline_alt_id.item())
            )
            if best_candidate_diagnostics is not None:
                diagnostics.update(best_candidate_diagnostics)
        else:
            diagnostics["spectral_absorption_efficiency"] = 0.0
            diagnostics["spectral_selector_churn"] = 0.0
            diagnostics["spectral_family_changed"] = False
    return best


def yaqa_spectral_push_v2b2_p32(
    inner_weight: torch.Tensor,
    input_hessian: torch.Tensor,
    output_hessian: torch.Tensor,
    codebook_library: tuple[torch.Tensor, ...],
    baseline: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    ranks: tuple[int, ...],
    alphas: tuple[float, ...],
    family_mode: str = "fixed_block_ldlq",
    block_input_hessian: torch.Tensor | None = None,
    block_family_id: int | None = None,
    factorization: tuple[BlockLDLFactorization, BlockLDLFactorization] | None = None,
    diagnostics: dict[str, object] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Push YAQA rounding targets toward the dominant post-quant residual modes.

    The low-rank correction is a proposal only: feedback remains relative to
    ``inner_weight``, candidates are scored with the original Hessians, and the
    exact serialized ``baseline`` is retained unless a finite strict
    improvement is found. No low-rank tensor is serialized.
    """

    if not ranks or any(isinstance(rank, bool) or not isinstance(rank, int) or rank < 1 for rank in ranks):
        raise ValueError("YAQA spectral push ranks must be positive integers.")
    if not alphas or any(
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
        or float(alpha) <= 0
        for alpha in alphas
    ):
        raise ValueError("YAQA spectral push alphas must be finite and positive.")

    baseline_weight, _baseline_states, baseline_selectors, baseline_alt_id = baseline
    source = inner_weight.to(torch.float32)
    original_input = input_hessian.to(torch.float32)
    original_output = output_hessian.to(torch.float32)
    residual = source - baseline_weight.to(torch.float32)
    input_root = torch.linalg.cholesky(original_input)
    output_root = torch.linalg.cholesky(original_output)
    whitened_residual = input_root.transpose(0, 1) @ residual @ output_root
    residual_energy = whitened_residual.square().sum()
    maximum_rank = min(max(ranks), min(whitened_residual.shape))

    from ..eora.eora import _eora_compute_svd

    spectral_device = torch.device("cpu") if whitened_residual.device.type == "mps" else whitened_residual.device
    left_vectors, singular_values, right_vectors_h = _eora_compute_svd(
        whitened_residual.to(device=spectral_device),
        maximum_rank,
        algo="lowrank",
    )
    left_vectors = left_vectors.to(device=whitened_residual.device)
    singular_values = singular_values.to(device=whitened_residual.device)
    right_vectors_h = right_vectors_h.to(device=whitened_residual.device)
    usable_ranks = tuple(sorted({min(rank, singular_values.numel()) for rank in ranks}))
    eps = torch.finfo(torch.float32).eps

    def original_loss(candidate: torch.Tensor) -> torch.Tensor:
        error = candidate.to(torch.float32) - source
        return torch.einsum("ij,ik,kl,lj->", error, original_input, error, original_output)

    best = baseline
    baseline_loss = original_loss(baseline_weight)
    best_loss = baseline_loss
    best_rank = None
    best_alpha = None
    best_candidate_diagnostics: dict[str, object] | None = None
    concentrations = {}
    oracle_losses = {}
    candidate_records: dict[str, dict[str, object]] = {}
    top_energies = {}
    for rank in usable_ranks:
        top_energy = singular_values[:rank].square().sum()
        top_energies[rank] = top_energy
        concentrations[str(rank)] = float((top_energy / residual_energy.clamp_min(eps)).item())
        whitened_correction = (
            left_vectors[:, :rank] * singular_values[:rank].unsqueeze(0)
        ) @ right_vectors_h[:rank]
        correction = torch.linalg.solve_triangular(
            input_root.transpose(0, 1),
            whitened_correction,
            upper=True,
        )
        correction = torch.linalg.solve_triangular(
            output_root.transpose(0, 1),
            correction.transpose(0, 1),
            upper=True,
        ).transpose(0, 1)
        oracle_loss = original_loss(baseline_weight.to(torch.float32) + correction)
        oracle_losses[str(rank)] = float(oracle_loss.item())

        for alpha in alphas:
            candidate_diagnostics: dict[str, object] = {}
            candidate = yaqa_inner_v2b2_p32(
                inner_weight,
                input_hessian,
                output_hessian,
                codebook_library,
                family_mode=family_mode,
                block_input_hessian=block_input_hessian,
                block_family_id=block_family_id,
                diagnostics=candidate_diagnostics,
                factorization=factorization,
                _rounding_bias=correction * float(alpha),
                **kwargs,
            )
            candidate_loss = original_loss(candidate[0])
            delta = candidate[0].to(torch.float32) - baseline_weight.to(torch.float32)
            whitened_delta = input_root.transpose(0, 1) @ delta @ output_root
            alignment_denominator = whitened_delta.norm() * whitened_correction.norm()
            alignment = (
                0.0
                if float(alignment_denominator.item()) <= eps
                else float((whitened_delta * whitened_correction).sum().div(alignment_denominator).item())
            )
            candidate_key = f"r{rank}_a{float(alpha):g}"
            candidate_is_finite = bool(torch.isfinite(candidate_loss).item())
            candidate_loss_value = float(candidate_loss.item()) if candidate_is_finite else None
            relative_improvement = (
                float(((baseline_loss - candidate_loss) / baseline_loss.abs().clamp_min(eps)).item())
                if candidate_is_finite
                else None
            )
            candidate_records[candidate_key] = {
                "rank": int(rank),
                "alpha": float(alpha),
                "finite": candidate_is_finite,
                "loss": candidate_loss_value,
                "relative_improvement": relative_improvement,
                "state_churn": float((candidate[1] != baseline[1]).to(torch.float32).mean().item()),
                "selector_churn": float((candidate[2] != baseline_selectors).to(torch.float32).mean().item()),
                "family_changed": bool(int(candidate[3].item()) != int(baseline_alt_id.item())),
                "spectral_alignment": alignment,
                "fallback_to_v2": bool(candidate_diagnostics.get("fallback_to_v2", False)),
                "family_candidates": candidate_diagnostics.get("family_candidates"),
                "selected": False,
            }
            if candidate_is_finite and candidate_loss < best_loss:
                best = candidate
                best_loss = candidate_loss
                best_rank = rank
                best_alpha = float(alpha)
                best_candidate_diagnostics = candidate_diagnostics

    if best_rank is not None and best_alpha is not None:
        candidate_records[f"r{best_rank}_a{best_alpha:g}"]["selected"] = True

    if diagnostics is not None:
        diagnostics["spectral_method"] = "push"
        diagnostics["spectral_svd_device"] = spectral_device.type
        diagnostics["spectral_concentration"] = concentrations
        diagnostics["spectral_oracle_losses"] = oracle_losses
        diagnostics["spectral_candidates"] = candidate_records
        diagnostics["spectral_selected"] = best_rank is not None
        diagnostics["spectral_rank"] = best_rank
        diagnostics["spectral_lambda"] = None
        diagnostics["spectral_alpha"] = best_alpha
        diagnostics["spectral_original_loss"] = float(baseline_loss.item())
        diagnostics["spectral_selected_loss"] = float(best_loss.item())
        if best_rank is not None:
            removable = top_energies[best_rank].clamp_min(eps)
            diagnostics["spectral_absorption_efficiency"] = float(
                ((baseline_loss - best_loss) / removable).item()
            )
            diagnostics["spectral_selector_churn"] = float(
                (best[2] != baseline_selectors).to(torch.float32).mean().item()
            )
            diagnostics["spectral_family_changed"] = bool(
                int(best[3].item()) != int(baseline_alt_id.item())
            )
            if best_candidate_diagnostics is not None:
                diagnostics.update(best_candidate_diagnostics)
        else:
            diagnostics["spectral_absorption_efficiency"] = 0.0
            diagnostics["spectral_selector_churn"] = 0.0
            diagnostics["spectral_family_changed"] = False
    return best


def yaqa_localized_spectral_refine_v2b2_p32(
    inner_weight: torch.Tensor,
    input_hessian: torch.Tensor,
    output_hessian: torch.Tensor,
    codebook_library: tuple[torch.Tensor, ...],
    baseline: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    ranks: tuple[int, ...],
    alphas: tuple[float, ...],
    max_segments: int,
    max_changes: int = 1,
    replay_candidates: int = 0,
    direct_replay_candidates: int = 0,
    candidate_score: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], float
    ]
    | None = None,
    replay_gradient: torch.Tensor | None = None,
    replay_gradient_error: bool = False,
    search_inputs: torch.Tensor | None = None,
    search_target: torch.Tensor | None = None,
    diagnostics: dict[str, object] | None = None,
    bits: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Propose fixed-boundary P32 replacements without global path churn.

    Candidate generation is restricted to one 32-weight segment. Its entry
    predecessor and final V2 state are fixed to the accepted YAQA artifact, so
    every untouched segment remains bit-identical. Candidates are ranked by
    the exact first-order YAQA decrease. Without search rows, acceptance uses
    the complete original Kronecker quadratic; with search rows, it uses exact
    held-out module-output loss before an independent propagation callback.
    """

    if not ranks or any(isinstance(rank, bool) or not isinstance(rank, int) or rank < 1 for rank in ranks):
        raise ValueError("YAQA localized spectral ranks must be positive integers.")
    if not alphas or any(
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
        or float(alpha) <= 0
        for alpha in alphas
    ):
        raise ValueError("YAQA localized spectral alphas must be finite and positive.")
    if isinstance(max_segments, bool) or not isinstance(max_segments, int) or max_segments < 1:
        raise ValueError("YAQA localized spectral max_segments must be a positive integer.")
    if (
        isinstance(max_changes, bool)
        or not isinstance(max_changes, int)
        or max_changes < 1
        or max_changes > max_segments
    ):
        raise ValueError("YAQA localized spectral max_changes must be between 1 and max_segments.")
    if (
        isinstance(replay_candidates, bool)
        or not isinstance(replay_candidates, int)
        or replay_candidates < 0
    ):
        raise ValueError("YAQA localized spectral replay_candidates must be a nonnegative integer.")
    if (candidate_score is None) != (replay_candidates == 0):
        raise ValueError("YAQA localized full-horizon scoring requires both a scorer and a positive shortlist.")
    if (
        isinstance(direct_replay_candidates, bool)
        or not isinstance(direct_replay_candidates, int)
        or direct_replay_candidates < 0
        or direct_replay_candidates > replay_candidates
    ):
        raise ValueError("YAQA localized direct replay candidates must be between zero and replay_candidates.")
    if not isinstance(replay_gradient_error, bool):
        raise TypeError("YAQA localized replay-gradient error state must be boolean.")
    if replay_gradient is not None and candidate_score is None:
        raise ValueError("YAQA localized replay-gradient ranking requires full-horizon candidate scoring.")
    if len(codebook_library) != 4:
        raise ValueError("YAQA localized V2B2-P32 requires canonical V2 plus three complementary families.")

    baseline_weight, baseline_states, baseline_selectors, baseline_alt_id = baseline
    source = inner_weight.to(torch.float32)
    accepted = baseline_weight.to(torch.float32)
    if source.ndim != 2 or source.shape != accepted.shape or source.shape[0] % 16 or source.shape[1] % 16:
        raise ValueError("YAQA localized V2B2-P32 requires matching 16-aligned weight matrices.")
    if replay_gradient is not None:
        if replay_gradient.shape != source.shape or not replay_gradient.is_floating_point():
            raise ValueError("YAQA localized replay gradient must match the inner-weight geometry.")
        if replay_gradient.device != source.device or not torch.isfinite(replay_gradient).all():
            raise ValueError("YAQA localized replay gradient must be finite and share the source device.")
        replay_gradient = replay_gradient.to(torch.float32)
    input_blocks = source.shape[0] // 16
    output_blocks = source.shape[1] // 16
    tile_count = input_blocks * output_blocks
    expected_state_shape = (tile_count, 128)
    expected_selector_count = tile_count * QVQ_V2B2_P32_SEGMENTS_PER_TILE
    if tuple(baseline_states.shape) != expected_state_shape:
        raise ValueError("YAQA localized V2B2-P32 baseline states have incompatible geometry.")
    if baseline_selectors.numel() != expected_selector_count:
        raise ValueError("YAQA localized V2B2-P32 baseline selectors have incompatible geometry.")
    alt_id = int(baseline_alt_id.item())
    if baseline_alt_id.numel() != 1 or alt_id not in (1, 2, 3):
        raise ValueError("YAQA localized V2B2-P32 baseline family ID must be 1, 2, or 3.")

    original_input = input_hessian.to(torch.float32)
    original_output = output_hessian.to(torch.float32)
    if tuple(original_input.shape) != (source.shape[0], source.shape[0]):
        raise ValueError("YAQA localized input Hessian does not match the weight geometry.")
    if tuple(original_output.shape) != (source.shape[1], source.shape[1]):
        raise ValueError("YAQA localized output Hessian does not match the weight geometry.")
    residual = source - accepted
    input_root = torch.linalg.cholesky(original_input)
    output_root = torch.linalg.cholesky(original_output)
    whitened_residual = input_root.transpose(0, 1) @ residual @ output_root
    maximum_rank = min(max(ranks), min(whitened_residual.shape))
    from ..eora.eora import _eora_compute_svd

    spectral_device = torch.device("cpu") if whitened_residual.device.type == "mps" else whitened_residual.device
    left_vectors, singular_values, right_vectors_h = _eora_compute_svd(
        whitened_residual.to(device=spectral_device),
        maximum_rank,
        algo="lowrank",
    )
    left_vectors = left_vectors.to(device=source.device)
    singular_values = singular_values.to(device=source.device)
    right_vectors_h = right_vectors_h.to(device=source.device)
    usable_ranks = tuple(sorted({min(rank, singular_values.numel()) for rank in ranks}))
    error = accepted - source
    gradient = 2 * (original_input @ error @ original_output)
    baseline_loss = torch.einsum("ij,ik,kl,lj->", error, original_input, error, original_output)
    if (search_inputs is None) != (search_target is None):
        raise ValueError("YAQA localized spectral search requires both inputs and targets.")
    if search_inputs is not None:
        if (
            search_inputs.ndim != 2
            or search_target.ndim != 2
            or search_inputs.shape[0] != search_target.shape[0]
            or search_inputs.shape[1] != source.shape[0]
            or search_target.shape[1] != source.shape[1]
            or search_inputs.device != source.device
            or search_target.device != source.device
            or not search_inputs.is_floating_point()
            or not search_target.is_floating_point()
            or not torch.isfinite(search_inputs).all()
            or not torch.isfinite(search_target).all()
        ):
            raise ValueError("YAQA localized spectral search tensors must be finite, aligned rank-2 matrices.")
        search_inputs_fp32 = search_inputs.to(torch.float32)
        search_target_fp32 = search_target.to(torch.float32)
        search_residual = search_target_fp32 - search_inputs_fp32 @ accepted
        baseline_search_loss = search_residual.square().sum()
    else:
        search_inputs_fp32 = None
        search_residual = None
        baseline_search_loss = None
    pair_codebooks = torch.stack((codebook_library[0], codebook_library[alt_id])).contiguous()

    def segment_view(matrix: torch.Tensor) -> torch.Tensor:
        return (
            matrix.reshape(input_blocks, 16, output_blocks, 16)
            .permute(0, 2, 1, 3)
            .reshape(tile_count, QVQ_V2B2_P32_SEGMENTS_PER_TILE, 2, 16)
        )

    source_tiles = segment_view(source)
    baseline_tiles = segment_view(accepted)
    gradient_tiles = segment_view(gradient)
    state_tiles = baseline_states.reshape(tile_count, 128)
    candidate_records: dict[str, dict[str, object]] = {}
    candidate_payloads: dict[
        str,
        tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ] = {}
    eps = torch.finfo(torch.float32).eps

    for rank in usable_ranks:
        whitened_correction = (
            left_vectors[:, :rank] * singular_values[:rank].unsqueeze(0)
        ) @ right_vectors_h[:rank]
        correction = torch.linalg.solve_triangular(
            input_root.transpose(0, 1),
            whitened_correction,
            upper=True,
        )
        correction = torch.linalg.solve_triangular(
            output_root.transpose(0, 1),
            correction.transpose(0, 1),
            upper=True,
        ).transpose(0, 1)
        correction_tiles = segment_view(correction)
        first_order_scores = -(gradient_tiles * correction_tiles).sum(dim=(2, 3)).reshape(-1)
        candidate_count = min(max_segments, first_order_scores.numel())
        ranked_scores, ranked_segments = first_order_scores.topk(candidate_count)
        positive = ranked_scores > 0
        ranked_scores = ranked_scores[positive]
        ranked_segments = ranked_segments[positive]
        if ranked_segments.numel() == 0:
            continue
        tile_indices = ranked_segments // QVQ_V2B2_P32_SEGMENTS_PER_TILE
        segment_indices = ranked_segments % QVQ_V2B2_P32_SEGMENTS_PER_TILE
        starts = segment_indices * QVQ_V2B2_P32_STEPS_PER_SEGMENT
        exits = starts + QVQ_V2B2_P32_STEPS_PER_SEGMENT - 1
        entry_indices = torch.where(starts == 0, torch.full_like(starts, 127), starts - 1)
        entries = state_tiles[tile_indices, entry_indices]
        required_exits = state_tiles[tile_indices, exits]

        for alpha in alphas:
            targets = (
                baseline_tiles[tile_indices, segment_indices]
                + float(alpha) * correction_tiles[tile_indices, segment_indices]
            ).reshape(-1, QVQ_V2B2_P32_STEPS_PER_SEGMENT, 2)
            localized = fixed_boundary_v2b2_p32_segment_quantize(
                targets,
                pair_codebooks,
                bits=bits,
                entry_states=entries,
                exit_states=required_exits,
            )
            localized_blocks = localized.values.reshape(-1, 2, 16).to(torch.float32)
            deltas = localized_blocks - baseline_tiles[tile_indices, segment_indices]
            for candidate_index in range(ranked_segments.numel()):
                tile_index = int(tile_indices[candidate_index].item())
                segment_index = int(segment_indices[candidate_index].item())
                input_block = tile_index // output_blocks
                output_block = tile_index % output_blocks
                input_start = input_block * 16 + segment_index * 2
                output_start = output_block * 16
                delta = deltas[candidate_index]
                linear = (gradient[input_start : input_start + 2, output_start : output_start + 16] * delta).sum()
                quadratic = torch.einsum(
                    "ij,ik,kl,lj->",
                    delta,
                    original_input[input_start : input_start + 2, input_start : input_start + 2],
                    delta,
                    original_output[output_start : output_start + 16, output_start : output_start + 16],
                )
                candidate_loss = baseline_loss + linear + quadratic
                if search_inputs_fp32 is None:
                    candidate_search_loss = None
                    selection_loss = candidate_loss
                else:
                    projected = search_inputs_fp32[:, input_start : input_start + 2] @ delta
                    residual_block = search_residual[:, output_start : output_start + 16]
                    candidate_search_loss = baseline_search_loss + (
                        (residual_block - projected).square() - residual_block.square()
                    ).sum()
                    selection_loss = candidate_search_loss
                candidate_key = f"r{rank}_a{float(alpha):g}_t{tile_index}_s{segment_index}"
                relative_improvement = float(
                    ((baseline_loss - candidate_loss) / baseline_loss.abs().clamp_min(eps)).item()
                )
                candidate_records[candidate_key] = {
                    "rank": int(rank),
                    "alpha": float(alpha),
                    "tile": tile_index,
                    "segment": segment_index,
                    "predicted_first_order": float(ranked_scores[candidate_index].item()),
                    "loss": float(candidate_loss.item()),
                    "relative_improvement": relative_improvement,
                    "search_loss": (
                        None if candidate_search_loss is None else float(candidate_search_loss.item())
                    ),
                    "search_relative_improvement": (
                        None
                        if candidate_search_loss is None
                        else float(
                            (
                                (baseline_search_loss - candidate_search_loss)
                                / baseline_search_loss.abs().clamp_min(eps)
                            ).item()
                        )
                    ),
                    "selected": False,
                }
                if torch.isfinite(selection_loss) and torch.count_nonzero(delta):
                    candidate_payloads[candidate_key] = (
                        tile_index,
                        segment_index,
                        localized.states[candidate_index],
                        localized.values[candidate_index],
                        localized.segment_bank_ids[candidate_index],
                        delta,
                    )

    # Spectral proposals can fill a small replay budget with near-duplicate
    # locally favorable directions. Optionally add exact fixed-boundary
    # re-encodings toward the dense weight and reserve replay capacity for
    # them. Their local loss is only a breadth-ordering heuristic; the
    # full-horizon scorer remains the acceptance authority.
    if direct_replay_candidates:
        residual_priorities = (source - accepted).square()
        residual_priorities = segment_view(residual_priorities).sum(dim=(2, 3)).reshape(-1)
        direct_count = min(max_segments, residual_priorities.numel())
        direct_segments = residual_priorities.topk(direct_count).indices
        direct_tiles = direct_segments // QVQ_V2B2_P32_SEGMENTS_PER_TILE
        direct_segment_ids = direct_segments % QVQ_V2B2_P32_SEGMENTS_PER_TILE
        direct_starts = direct_segment_ids * QVQ_V2B2_P32_STEPS_PER_SEGMENT
        direct_exits = direct_starts + QVQ_V2B2_P32_STEPS_PER_SEGMENT - 1
        direct_entries = state_tiles[
            direct_tiles,
            torch.where(direct_starts == 0, torch.full_like(direct_starts, 127), direct_starts - 1),
        ]
        direct_required_exits = state_tiles[direct_tiles, direct_exits]
        direct = fixed_boundary_v2b2_p32_segment_quantize(
            source_tiles[direct_tiles, direct_segment_ids].reshape(
                -1, QVQ_V2B2_P32_STEPS_PER_SEGMENT, 2
            ),
            pair_codebooks,
            bits=bits,
            entry_states=direct_entries,
            exit_states=direct_required_exits,
        )
        direct_values = direct.values.reshape(-1, 2, 16).to(torch.float32)
        direct_deltas = direct_values - baseline_tiles[direct_tiles, direct_segment_ids]
        payloads_by_segment: dict[tuple[int, int], list[tuple[object, ...]]] = defaultdict(list)
        for payload in candidate_payloads.values():
            payloads_by_segment[(payload[0], payload[1])].append(payload)
        for direct_index in range(direct_segments.numel()):
            tile_index = int(direct_tiles[direct_index].item())
            segment_index = int(direct_segment_ids[direct_index].item())
            delta = direct_deltas[direct_index]
            if not torch.count_nonzero(delta):
                continue
            duplicate = any(
                payload[0] == tile_index
                and payload[1] == segment_index
                and torch.equal(payload[2], direct.states[direct_index])
                and torch.equal(payload[4], direct.segment_bank_ids[direct_index])
                for payload in payloads_by_segment.get((tile_index, segment_index), ())
            )
            if duplicate:
                continue
            candidate_key = f"direct_t{tile_index}_s{segment_index}"
            candidate_records[candidate_key] = {
                "generator": "direct_dense_reencode",
                "tile": tile_index,
                "segment": segment_index,
                "selected": False,
            }
            candidate_payloads[candidate_key] = (
                tile_index,
                segment_index,
                direct.states[direct_index],
                direct.values[direct_index],
                direct.segment_bank_ids[direct_index],
                delta,
            )

    refined_weight = baseline_weight.clone()
    refined_states = baseline_states.clone()
    refined_selectors = baseline_selectors.clone()
    refined_selector_view = refined_selectors.reshape(tile_count, QVQ_V2B2_P32_SEGMENTS_PER_TILE)
    selected_candidate_keys: list[str] = []
    selected_segments: set[tuple[int, int]] = set()
    current_search_residual = None if search_residual is None else search_residual.clone()
    current_loss = baseline_search_loss if baseline_search_loss is not None else baseline_loss
    current_replay_score = None
    baseline_replay_score = None
    replay_callback_error = replay_gradient_error
    if candidate_score is not None and not replay_gradient_error:
        try:
            current_replay_score = float(
                candidate_score(refined_weight, refined_states, refined_selectors, baseline_alt_id)
            )
            baseline_replay_score = current_replay_score
        except Exception:  # noqa: BLE001 - an external full-model scorer must fail closed
            current_replay_score = math.inf
            replay_callback_error = True
        if not math.isfinite(current_replay_score):
            replay_callback_error = True
    if replay_callback_error:
        candidate_payloads.clear()

    # Every proposal preserves the accepted entry/exit state, so replacements
    # for distinct P32 segments compose exactly. Greedy scoring is conditional:
    # after each accepted delta, all remaining gains are recomputed against the
    # current live-output residual (or complete Kronecker proxy). This avoids
    # treating individually favorable candidates as additively independent.
    for _ in range(max_changes):
        locally_ranked: list[tuple[float, str, float]] = []
        for candidate_key, payload in candidate_payloads.items():
            tile_index, segment_index, _segment_states, segment_values, _segment_bank, _delta = payload
            if (tile_index, segment_index) in selected_segments:
                continue
            input_block = tile_index // output_blocks
            output_block = tile_index % output_blocks
            input_start = input_block * 16 + segment_index * 2
            output_start = output_block * 16
            candidate_delta = segment_values.reshape(2, 16).to(torch.float32) - refined_weight[
                input_start : input_start + 2, output_start : output_start + 16
            ].to(torch.float32)
            if search_inputs_fp32 is not None:
                projected = search_inputs_fp32[:, input_start : input_start + 2] @ candidate_delta
                residual_block = current_search_residual[:, output_start : output_start + 16]
                step_loss = current_loss + (
                    (residual_block - projected).square() - residual_block.square()
                ).sum()
            else:
                candidate_weight = refined_weight.clone()
                candidate_weight[input_start : input_start + 2, output_start : output_start + 16] = (
                    segment_values.reshape(2, 16)
                )
                candidate_error = candidate_weight.to(torch.float32) - source
                step_loss = torch.einsum(
                    "ij,ik,kl,lj->", candidate_error, original_input, candidate_error, original_output
                )
            # With a full-horizon scorer, local loss orders the bounded
            # shortlist unless an exact baseline teacher-KL gradient is
            # available. The gradient then ranks by the downstream first-order
            # term <G,D>; replay remains the nonlinear selection authority.
            # Without that scorer, retain the strict local improvement gate.
            if torch.isfinite(step_loss) and (candidate_score is not None or step_loss < current_loss):
                local_loss = float(step_loss.item())
                ranking_score = local_loss
                if replay_gradient is not None:
                    propagated_first_order = float(
                        (
                            replay_gradient[input_start : input_start + 2, output_start : output_start + 16]
                            * candidate_delta
                        ).sum().item()
                    )
                    candidate_records[candidate_key]["propagated_first_order"] = propagated_first_order
                    ranking_score = propagated_first_order
                locally_ranked.append((ranking_score, candidate_key, local_loss))

        locally_ranked.sort(key=lambda item: (item[0], item[1]))
        if candidate_score is None:
            shortlisted = locally_ranked[:1]
        else:
            direct_ranked = [item for item in locally_ranked if item[1].startswith("direct_")]
            shortlisted = direct_ranked[:direct_replay_candidates]
            shortlisted_keys = {item[1] for item in shortlisted}
            shortlisted.extend(
                item
                for item in locally_ranked
                if item[1] not in shortlisted_keys
            )
            shortlisted = shortlisted[:replay_candidates]

        best_key = None
        best_step_loss = current_loss
        best_replay_score = current_replay_score
        for _ranking_score, candidate_key, local_loss in shortlisted:
            if candidate_score is None:
                best_key = candidate_key
                best_step_loss = torch.as_tensor(local_loss, device=current_loss.device)
                break
            payload = candidate_payloads[candidate_key]
            tile_index, segment_index, segment_states, segment_values, segment_bank, _delta = payload
            input_block = tile_index // output_blocks
            output_block = tile_index % output_blocks
            input_start = input_block * 16 + segment_index * 2
            output_start = output_block * 16
            trial_weight = refined_weight.clone()
            trial_states = refined_states.clone()
            trial_selectors = refined_selectors.clone()
            trial_weight[
                input_start : input_start + 2, output_start : output_start + 16
            ] = segment_values.reshape(2, 16)
            state_start = segment_index * QVQ_V2B2_P32_STEPS_PER_SEGMENT
            trial_states[tile_index, state_start : state_start + QVQ_V2B2_P32_STEPS_PER_SEGMENT] = segment_states
            trial_selectors.reshape(tile_count, QVQ_V2B2_P32_SEGMENTS_PER_TILE)[
                tile_index, segment_index
            ] = segment_bank
            try:
                replay_score = float(
                    candidate_score(trial_weight, trial_states, trial_selectors, baseline_alt_id)
                )
            except Exception:  # noqa: BLE001 - an external full-model scorer must fail closed
                replay_callback_error = True
                continue
            candidate_records[candidate_key]["replay_score"] = replay_score
            if math.isfinite(replay_score) and replay_score < best_replay_score:
                best_key = candidate_key
                best_step_loss = torch.as_tensor(local_loss, device=current_loss.device)
                best_replay_score = replay_score

        if best_key is None:
            break
        tile_index, segment_index, segment_states, segment_values, segment_bank, _delta = candidate_payloads[best_key]
        input_block = tile_index // output_blocks
        output_block = tile_index % output_blocks
        input_start = input_block * 16 + segment_index * 2
        output_start = output_block * 16
        if current_search_residual is not None:
            candidate_delta = segment_values.reshape(2, 16).to(torch.float32) - refined_weight[
                input_start : input_start + 2, output_start : output_start + 16
            ].to(torch.float32)
            current_search_residual[:, output_start : output_start + 16].sub_(
                search_inputs_fp32[:, input_start : input_start + 2] @ candidate_delta
            )
        refined_weight[input_start : input_start + 2, output_start : output_start + 16] = segment_values.reshape(2, 16)
        state_start = segment_index * QVQ_V2B2_P32_STEPS_PER_SEGMENT
        refined_states[tile_index, state_start : state_start + QVQ_V2B2_P32_STEPS_PER_SEGMENT] = segment_states
        refined_selector_view[tile_index, segment_index] = segment_bank
        selected_segments.add((tile_index, segment_index))
        selected_candidate_keys.append(best_key)
        candidate_records[best_key]["selected"] = True
        candidate_records[best_key]["conditional_selection_loss"] = float(best_step_loss.item())
        current_loss = best_step_loss
        current_replay_score = best_replay_score

    if selected_candidate_keys:
        exact_error = refined_weight.to(torch.float32) - source
        exact_proxy_loss = torch.einsum(
            "ij,ik,kl,lj->", exact_error, original_input, exact_error, original_output
        )
        exact_selection_loss = exact_proxy_loss if search_inputs_fp32 is None else current_search_residual.square().sum()
        if candidate_score is None:
            accept_exact = torch.isfinite(exact_selection_loss) and exact_selection_loss < (
                baseline_search_loss if baseline_search_loss is not None else baseline_loss
            )
        else:
            accept_exact = (
                baseline_replay_score is not None
                and current_replay_score is not None
                and math.isfinite(current_replay_score)
                and current_replay_score < baseline_replay_score
            )
        if accept_exact:
            result = refined_weight, refined_states, refined_selectors, baseline_alt_id
            best_loss = exact_selection_loss
        else:
            for candidate_key in selected_candidate_keys:
                candidate_records[candidate_key]["selected"] = False
            selected_candidate_keys = []
            result = baseline
            best_loss = baseline_search_loss if baseline_search_loss is not None else baseline_loss
    else:
        result = baseline
        best_loss = baseline_search_loss if baseline_search_loss is not None else baseline_loss

    if diagnostics is not None:
        diagnostics["spectral_method"] = "localized_p32"
        diagnostics["spectral_svd_device"] = spectral_device.type
        diagnostics["spectral_candidates"] = candidate_records
        diagnostics["spectral_selected"] = bool(selected_candidate_keys)
        diagnostics["localized_selected_changes"] = len(selected_candidate_keys)
        diagnostics["localized_selected_candidates"] = selected_candidate_keys
        diagnostics["localized_replay_candidates"] = replay_candidates
        diagnostics["localized_direct_replay_candidates"] = direct_replay_candidates
        diagnostics["localized_replay_gradient_ranked"] = replay_gradient is not None
        diagnostics["localized_replay_gradient_callback_error"] = replay_gradient_error
        diagnostics["localized_replay_score"] = current_replay_score
        diagnostics["localized_replay_callback_error"] = replay_callback_error
        diagnostics["spectral_original_loss"] = float(baseline_loss.item())
        selected_error = result[0].to(torch.float32) - source
        diagnostics["spectral_selected_loss"] = float(
            torch.einsum("ij,ik,kl,lj->", selected_error, original_input, selected_error, original_output).item()
        )
        diagnostics["localized_search_original_loss"] = (
            None if baseline_search_loss is None else float(baseline_search_loss.item())
        )
        diagnostics["localized_search_selected_loss"] = (
            None if baseline_search_loss is None else float(best_loss.item())
        )
        diagnostics["spectral_selector_churn"] = float(
            (result[2] != baseline_selectors).to(torch.float32).mean().item()
        )
        diagnostics["spectral_family_changed"] = False
        diagnostics["localized_boundary_preserved"] = True
    return result


def rht_preprocess_weight(
    weight: torch.Tensor,
    SU: torch.Tensor,
    SV: torch.Tensor,
) -> torch.Tensor:
    """Map dense ``[out, in]`` weights into QVQ's ``[in, out]`` RHT basis."""

    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("QVQ weight must be a floating-point matrix.")
    out_features, in_features = weight.shape
    if tuple(SU.shape) != (in_features,) or tuple(SV.shape) != (out_features,):
        raise ValueError("QVQ RHT sign shapes must match the weight dimensions.")
    if weight.device != SU.device or weight.device != SV.device:
        raise ValueError("QVQ weight and RHT signs must share one device.")
    if not torch.isfinite(weight).all() or not torch.isfinite(SU).all() or not torch.isfinite(SV).all():
        raise ValueError("QVQ weight and RHT signs must contain only finite values.")
    _validate_fp32_representable(weight, name="QVQ weight")
    _validate_fp32_representable(SU, name="QVQ input signs")
    _validate_fp32_representable(SV, name="QVQ output signs")

    work = weight.transpose(0, 1).to(torch.float32) * SU.to(torch.float32).unsqueeze(1)
    work = matmul_hadU(work.transpose(0, 1)).transpose(0, 1)
    work = work * SV.to(torch.float32).unsqueeze(0)
    return matmul_hadU(work, transpose=True).contiguous()


def rht_preprocess_hessian(H: torch.Tensor, SU: torch.Tensor) -> torch.Tensor:
    """Transform an input Hessian into the runtime's randomized Hadamard basis."""

    if H.ndim != 2 or H.shape[0] != H.shape[1] or not H.is_floating_point():
        raise ValueError("QVQ Hessian must be a floating-point square matrix.")
    if tuple(SU.shape) != (H.shape[0],) or H.device != SU.device:
        raise ValueError("QVQ input signs must match the Hessian width and device.")
    if not torch.isfinite(H).all() or not torch.isfinite(SU).all():
        raise ValueError("QVQ Hessian and input signs must contain only finite values.")
    _validate_fp32_representable(H, name="QVQ Hessian")
    _validate_fp32_representable(SU, name="QVQ input signs")

    signs = SU.to(torch.float32)
    transformed = H.to(torch.float32) * signs.unsqueeze(0) * signs.unsqueeze(1)
    transformed = matmul_hadU(transformed)
    return matmul_hadU(transformed.transpose(0, 1)).transpose(0, 1).contiguous()


def rht_reconstruct_weight(
    inner_weight: torch.Tensor,
    SU: torch.Tensor,
    SV: torch.Tensor,
) -> torch.Tensor:
    """Invert :func:`rht_preprocess_weight` through the inference dataflow."""

    if inner_weight.ndim != 2 or not inner_weight.is_floating_point():
        raise ValueError("QVQ inner weight must be a floating-point matrix.")
    in_features, out_features = inner_weight.shape
    if tuple(SU.shape) != (in_features,) or tuple(SV.shape) != (out_features,):
        raise ValueError("QVQ RHT scale shapes must match the inner weight dimensions.")
    if inner_weight.device != SU.device or inner_weight.device != SV.device:
        raise ValueError("QVQ inner weight and RHT scales must share one device.")
    if not torch.isfinite(inner_weight).all() or not torch.isfinite(SU).all() or not torch.isfinite(SV).all():
        raise ValueError("QVQ inner weight and RHT scales must contain only finite values.")
    _validate_fp32_representable(inner_weight, name="QVQ inner weight")
    _validate_fp32_representable(SU, name="QVQ input signs")
    _validate_fp32_representable(SV, name="QVQ output signs")

    work = matmul_hadU(inner_weight.transpose(0, 1), transpose=True).transpose(0, 1)
    work = work * SU.to(work.dtype).unsqueeze(1)
    work = matmul_hadU(work) * SV.to(work.dtype).unsqueeze(0)
    return work.transpose(0, 1).contiguous()


def rht_reconstruct_weight_adjoint(
    weight_gradient: torch.Tensor,
    SU: torch.Tensor,
    SV: torch.Tensor,
) -> torch.Tensor:
    """Map a dense-weight gradient into the exact QVQ inner-weight basis.

    ``rht_reconstruct_weight`` is linear in its inner weight.  Its adjoint is
    therefore the unique map satisfying ``<G, R(D)> = <R*(G), D>``.  Build it
    through autograd so the gradient remains exactly coupled to the production
    RHT implementation instead of duplicating its transpose/sign convention.
    """

    if weight_gradient.ndim != 2 or not weight_gradient.is_floating_point():
        raise ValueError("QVQ reconstruction gradient must be a floating-point matrix.")
    out_features, in_features = weight_gradient.shape
    if tuple(SU.shape) != (in_features,) or tuple(SV.shape) != (out_features,):
        raise ValueError("QVQ reconstruction gradient and RHT scales have incompatible shapes.")
    if weight_gradient.device != SU.device or weight_gradient.device != SV.device:
        raise ValueError("QVQ reconstruction gradient and RHT scales must share one device.")
    if not torch.isfinite(weight_gradient).all():
        raise ValueError("QVQ reconstruction gradient must contain only finite values.")
    _validate_fp32_representable(weight_gradient, name="QVQ reconstruction gradient")
    with torch.enable_grad():
        inner = torch.zeros(
            (in_features, out_features),
            device=weight_gradient.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        reconstructed = rht_reconstruct_weight(inner, SU, SV)
        (inner_gradient,) = torch.autograd.grad(
            reconstructed,
            inner,
            grad_outputs=weight_gradient.to(torch.float32),
            create_graph=False,
        )
    return inner_gradient.detach().contiguous()


def qvq_proxy_loss(weight: torch.Tensor, reconstructed_weight: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Return QVQ's FP32 input-Hessian reconstruction objective."""

    if weight.ndim != 2 or reconstructed_weight.shape != weight.shape:
        raise ValueError("QVQ proxy-loss weights must be matching rank-2 tensors.")
    if tuple(H.shape) != (weight.shape[1], weight.shape[1]):
        raise ValueError("QVQ proxy-loss Hessian must match the weight input width.")
    if weight.device != reconstructed_weight.device or weight.device != H.device:
        raise ValueError("QVQ proxy-loss tensors must share one device.")
    if not weight.is_floating_point() or not reconstructed_weight.is_floating_point() or not H.is_floating_point():
        raise TypeError("QVQ proxy-loss tensors must use floating-point dtypes.")
    if (
        not torch.isfinite(weight).all()
        or not torch.isfinite(reconstructed_weight).all()
        or not torch.isfinite(H).all()
    ):
        raise ValueError("QVQ proxy-loss tensors must contain only finite values.")
    _validate_fp32_representable(weight, name="QVQ weight")
    _validate_fp32_representable(reconstructed_weight, name="QVQ reconstructed weight")
    _validate_fp32_representable(H, name="QVQ Hessian")

    return _qvq_proxy_loss_unchecked(weight, reconstructed_weight, H)


def yaqa_proxy_loss(
    weight: torch.Tensor,
    reconstructed_weight: torch.Tensor,
    input_hessian: torch.Tensor,
    output_hessian: torch.Tensor,
) -> torch.Tensor:
    """Return YAQA's Kronecker-factored full-model proxy in FP32.

    For dense-orientation weights ``[out, in]`` this evaluates
    ``trace(E @ H_I @ E.T @ H_O)`` where ``E = reconstructed - weight``.
    """

    if weight.ndim != 2 or reconstructed_weight.shape != weight.shape:
        raise ValueError("YAQA proxy-loss weights must be matching rank-2 tensors.")
    out_features, in_features = weight.shape
    if tuple(input_hessian.shape) != (in_features, in_features):
        raise ValueError("YAQA proxy-loss input Hessian must match the weight input width.")
    if tuple(output_hessian.shape) != (out_features, out_features):
        raise ValueError("YAQA proxy-loss output Hessian must match the weight output width.")
    tensors = (weight, reconstructed_weight, input_hessian, output_hessian)
    if any(tensor.device != weight.device for tensor in tensors[1:]):
        raise ValueError("YAQA proxy-loss tensors must share one device.")
    if any(not tensor.is_floating_point() for tensor in tensors):
        raise TypeError("YAQA proxy-loss tensors must use floating-point dtypes.")
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("YAQA proxy-loss tensors must contain only finite values.")
    for name, tensor in zip(
        ("YAQA weight", "YAQA reconstructed weight", "YAQA input Hessian", "YAQA output Hessian"),
        tensors,
        strict=True,
    ):
        _validate_fp32_representable(tensor, name=name)

    error = reconstructed_weight.to(torch.float32) - weight.to(torch.float32)
    input_fp32 = input_hessian.to(torch.float32)
    output_fp32 = output_hessian.to(torch.float32)
    loss = (error @ input_fp32 @ error.transpose(0, 1) * output_fp32.transpose(0, 1)).sum()
    if not torch.isfinite(loss):
        raise ValueError("YAQA proxy-loss arithmetic overflowed FP32")
    return loss


def _qvq_proxy_loss_unchecked(
    weight: torch.Tensor,
    reconstructed_weight: torch.Tensor,
    H: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the proxy after the caller has established the tensor contract."""

    error = reconstructed_weight.to(torch.float32) - weight.to(torch.float32)
    H_fp32 = H.to(torch.float32)
    loss = (error @ H_fp32 * error).sum()
    if not torch.isfinite(loss):
        raise ValueError("QVQ proxy-loss arithmetic overflowed FP32")
    return loss


def optimize_qvq_output_channel_scales(
    weight: torch.Tensor,
    inner_weight: torch.Tensor,
    H: torch.Tensor,
    SU: torch.Tensor,
    SV: torch.Tensor,
    *,
    denominator_epsilon: float | None = None,
    optimization_H: torch.Tensor | None = None,
    correction_strength: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Optimize fixed-trellis output scales and reject every non-improving row.

    ``SV`` already contains the current signed scale.  The closed form below
    therefore solves for a positive multiplicative correction, not an
    absolute replacement scale. ``correction_strength`` shrinks that optimum
    toward the exact disabled value one before checkpoint-dtype rounding and
    original-Hessian acceptance. Trellis states and the PGC16 decoder remain
    unchanged.
    """

    current = rht_reconstruct_weight(inner_weight, SU, SV)
    if current.shape != weight.shape:
        raise ValueError("QVQ output-scale reconstruction must match the source weight shape.")
    if tuple(H.shape) != (weight.shape[1], weight.shape[1]):
        raise ValueError("QVQ output-scale Hessian must match the weight input width.")
    if weight.device != current.device or weight.device != H.device:
        raise ValueError("QVQ output-scale tensors must share one device.")
    if not weight.is_floating_point() or not H.is_floating_point():
        raise TypeError("QVQ output-scale weight and Hessian must use floating-point dtypes.")
    if not torch.isfinite(weight).all() or not torch.isfinite(H).all():
        raise ValueError("QVQ output-scale weight and Hessian must contain only finite values.")
    if optimization_H is None:
        optimization_H = H
    if tuple(optimization_H.shape) != tuple(H.shape):
        raise ValueError("QVQ output-scale optimization Hessian must match the acceptance Hessian shape.")
    if optimization_H.device != H.device:
        raise ValueError("QVQ output-scale optimization Hessian must share the acceptance Hessian device.")
    if not optimization_H.is_floating_point():
        raise TypeError("QVQ output-scale optimization Hessian must use a floating-point dtype.")
    if not torch.isfinite(optimization_H).all():
        raise ValueError("QVQ output-scale optimization Hessian must contain only finite values.")

    if denominator_epsilon is None:
        denominator_epsilon = torch.finfo(torch.float32).eps
    if not isinstance(denominator_epsilon, (int, float)) or isinstance(denominator_epsilon, bool):
        raise TypeError("QVQ output-scale denominator epsilon must be a real scalar.")
    denominator_epsilon = float(denominator_epsilon)
    if not math.isfinite(denominator_epsilon) or denominator_epsilon < 0:
        raise ValueError("QVQ output-scale denominator epsilon must be finite and nonnegative.")
    if not isinstance(correction_strength, (int, float)) or isinstance(correction_strength, bool):
        raise TypeError("QVQ output-scale correction strength must be a real scalar.")
    correction_strength = float(correction_strength)
    if not math.isfinite(correction_strength) or not 0 < correction_strength <= 1:
        raise ValueError("QVQ output-scale correction strength must be finite and in (0, 1].")

    target_fp32 = weight.to(torch.float32)
    current_fp32 = current.to(torch.float32)
    H_fp32 = H.to(torch.float32)
    optimization_H_fp32 = optimization_H.to(torch.float32)
    current_hessian_product = current_fp32 @ optimization_H_fp32
    numerator = (current_hessian_product * target_fp32).sum(dim=1)
    denominator = (current_hessian_product * current_fp32).sum(dim=1)
    valid = torch.isfinite(numerator) & torch.isfinite(denominator) & (denominator > denominator_epsilon)
    correction = torch.where(valid, numerator / denominator, torch.ones_like(denominator))
    valid &= torch.isfinite(correction) & (correction > 0)
    correction = torch.where(valid, correction, torch.ones_like(correction))
    correction = 1 + correction_strength * (correction - 1)

    # Evaluate the exact scale values that will be retained by the checkpoint.
    # A lower-precision SV can round a mathematically improving FP32 candidate
    # back to its current value, which must remain a no-op rather than being
    # reported as an optimized channel.
    candidate_SV = (SV.to(torch.float32) * correction).to(SV.dtype)
    candidate = rht_reconstruct_weight(inner_weight, SU, candidate_SV)
    current_error = current_fp32 - target_fp32
    candidate_error = candidate.to(torch.float32) - target_fp32
    current_row_loss = (current_error @ H_fp32 * current_error).sum(dim=1)
    candidate_row_loss = (candidate_error @ H_fp32 * candidate_error).sum(dim=1)
    accepted = valid & torch.isfinite(candidate_row_loss) & (candidate_row_loss < current_row_loss)

    optimized_SV = torch.where(accepted, candidate_SV, SV)
    optimized_weight = rht_reconstruct_weight(inner_weight, SU, optimized_SV)
    optimized_loss = _qvq_proxy_loss_unchecked(weight, optimized_weight, H)
    current_loss = current_row_loss.sum()
    if not torch.isfinite(optimized_loss) or optimized_loss > current_loss:
        return SV, current, current_loss, 0
    return optimized_SV, optimized_weight, optimized_loss, int(accepted.sum().item())


def optimize_qvq_module_scale(
    weight: torch.Tensor,
    inner_weight: torch.Tensor,
    H: torch.Tensor,
    SU: torch.Tensor,
    SV: torch.Tensor,
    *,
    denominator_epsilon: float | None = None,
    optimization_H: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, bool]:
    """Optimize one fixed-trellis module scale already represented by ``SV``.

    The scalar correction is the full-Hessian least-squares optimum for the
    current decoded matrix.  The exact checkpoint-dtype ``SV`` is reconstructed
    and scored under the original Hessian before acceptance.  This is also a
    cheap proposal for one scale-aware re-encoding pass in
    :func:`quantize_qvq_linear`; neither path changes the inference format.
    """

    current = rht_reconstruct_weight(inner_weight, SU, SV)
    if current.shape != weight.shape:
        raise ValueError("QVQ module-scale reconstruction must match the source weight shape.")
    if tuple(H.shape) != (weight.shape[1], weight.shape[1]):
        raise ValueError("QVQ module-scale Hessian must match the weight input width.")
    if weight.device != current.device or weight.device != H.device:
        raise ValueError("QVQ module-scale tensors must share one device.")
    if not weight.is_floating_point() or not H.is_floating_point():
        raise TypeError("QVQ module-scale weight and Hessian must use floating-point dtypes.")
    if not torch.isfinite(weight).all() or not torch.isfinite(H).all():
        raise ValueError("QVQ module-scale weight and Hessian must contain only finite values.")
    if optimization_H is None:
        optimization_H = H
    if tuple(optimization_H.shape) != tuple(H.shape):
        raise ValueError("QVQ module-scale optimization Hessian must match the acceptance Hessian shape.")
    if optimization_H.device != H.device:
        raise ValueError("QVQ module-scale optimization Hessian must share the acceptance Hessian device.")
    if not optimization_H.is_floating_point():
        raise TypeError("QVQ module-scale optimization Hessian must use a floating-point dtype.")
    if not torch.isfinite(optimization_H).all():
        raise ValueError("QVQ module-scale optimization Hessian must contain only finite values.")

    if denominator_epsilon is None:
        denominator_epsilon = torch.finfo(torch.float32).eps
    if not isinstance(denominator_epsilon, (int, float)) or isinstance(denominator_epsilon, bool):
        raise TypeError("QVQ module-scale denominator epsilon must be a real scalar.")
    denominator_epsilon = float(denominator_epsilon)
    if not math.isfinite(denominator_epsilon) or denominator_epsilon < 0:
        raise ValueError("QVQ module-scale denominator epsilon must be finite and nonnegative.")

    target_fp32 = weight.to(torch.float32)
    current_fp32 = current.to(torch.float32)
    optimization_H_fp32 = optimization_H.to(torch.float32)
    current_hessian_product = current_fp32 @ optimization_H_fp32
    numerator = (current_hessian_product * target_fp32).sum()
    denominator = (current_hessian_product * current_fp32).sum()
    valid = bool(
        torch.isfinite(numerator)
        and torch.isfinite(denominator)
        and denominator > denominator_epsilon
    )
    correction = numerator / denominator if valid else torch.ones_like(denominator)
    valid = bool(valid and torch.isfinite(correction) and correction > 0)
    correction = correction if valid else torch.ones_like(correction)

    candidate_SV = (SV.to(torch.float32) * correction).to(SV.dtype)
    candidate = rht_reconstruct_weight(inner_weight, SU, candidate_SV)
    current_loss = _qvq_proxy_loss_unchecked(weight, current, H)
    candidate_loss = _qvq_proxy_loss_unchecked(weight, candidate, H)
    accepted = bool(valid and torch.isfinite(candidate_loss) and candidate_loss < current_loss)
    if not accepted:
        return SV, current, current_loss, 1.0, False

    exact_multiplier = float((candidate_SV.abs().mean() / SV.abs().mean()).item())
    if not math.isfinite(exact_multiplier) or exact_multiplier <= 0:
        return SV, current, current_loss, 1.0, False
    return candidate_SV, candidate, candidate_loss, exact_multiplier, True


def _quantize_lr32_sequences(
    sequences: torch.Tensor,
    codebooks: tuple[torch.Tensor, ...],
    *,
    bits: float,
    trellis_batch_size: int,
    tail_biting_candidates: int,
    step_weights: torch.Tensor | None = None,
    telemetry: QVQQuantizationTelemetry | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize independent LR32 rings and choose one bank per ring.

    ``sequences`` is ring-major ``[ring, 16, 2]`` data.  Each ring owns its
    tail-biting history, so no recurrence or bank decision may cross a ring
    boundary.  Canonical bank zero wins exact ties.
    """

    if sequences.ndim != 3 or tuple(sequences.shape[1:]) != (
        QVQ_V2B2_P32_LR_RING_STEPS,
        2,
    ):
        raise ValueError("QVQ LR32 sequences must have shape [rings, 16, 2].")
    if len(codebooks) not in (1, 2):
        raise ValueError("QVQ LR32 ring quantization requires one or two codebooks.")
    if any(
        tuple(codebook.shape) != (1 << 16, 2)
        or codebook.device != sequences.device
        or not codebook.is_floating_point()
        for codebook in codebooks
    ):
        raise ValueError("QVQ LR32 codebooks must be floating-point [65536, 2] tensors on the ring device.")
    if step_weights is not None and tuple(step_weights.shape) != tuple(sequences.shape[:2]):
        raise ValueError("QVQ LR32 step weights must have shape [rings, 16].")

    bank_values = []
    bank_states = []
    bank_losses = []
    for codebook in codebooks:
        values = []
        states = []
        losses = []
        sequence_chunks = sequences.split(trellis_batch_size)
        weight_chunks = (
            (None,) * len(sequence_chunks)
            if step_weights is None
            else step_weights.split(trellis_batch_size)
        )
        with _qvq_phase(telemetry, "lr32_viterbi", sequences.device):
            for chunk, weight_chunk in zip(sequence_chunks, weight_chunks, strict=True):
                result = tail_biting_viterbi_quantize(
                    chunk.contiguous(),
                    codebook,
                    bits=bits,
                    step_weights=None if weight_chunk is None else weight_chunk.contiguous(),
                    candidate_count=tail_biting_candidates,
                )
                values.append(result.values)
                states.append(result.states)
                losses.append(result.squared_error)
        bank_values.append(torch.cat(values))
        bank_states.append(torch.cat(states))
        bank_losses.append(torch.cat(losses))

    if len(codebooks) == 1:
        selectors = torch.zeros(sequences.shape[0], dtype=torch.uint8, device=sequences.device)
        return bank_values[0], bank_states[0], selectors

    losses = torch.stack(bank_losses)
    selectors_long = losses.argmin(dim=0)
    ring_indices = torch.arange(sequences.shape[0], device=sequences.device)
    values = torch.stack(bank_values)[selectors_long, ring_indices]
    states = torch.stack(bank_states)[selectors_long, ring_indices]
    return values, states, selectors_long.to(torch.uint8)


def _block_ldlq_inner_v2b2_p32_lr_candidate(
    inner_weight: torch.Tensor,
    H: torch.Tensor,
    codebooks: tuple[torch.Tensor, ...],
    *,
    bits: float,
    trellis_batch_size: int,
    viterbi_objective: str,
    tail_biting_candidates: int,
    telemetry: QVQQuantizationTelemetry | None,
    factorization: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode one complete LR32 BlockLDLQ artifact for a fixed bank family."""

    in_features, out_features = inner_weight.shape
    tile_rows = QVQ_V2B2_P32_LR_TILE_ROWS
    tile_cols = QVQ_V2B2_P32_LR_TILE_COLS
    input_blocks = in_features // tile_rows
    output_blocks = out_features // tile_cols
    L, D = factorization
    feedback = L.clone()
    feedback.diagonal().sub_(1)
    source = inner_weight.to(device=L.device, dtype=torch.float32)
    error = source.clone()
    quantized = torch.zeros_like(source)
    tile_states = torch.empty(
        (input_blocks, output_blocks, QVQ_V2B2_P32_LR_RINGS_PER_TILE, QVQ_V2B2_P32_LR_RING_STEPS),
        dtype=torch.long,
        device=source.device,
    )
    selectors = torch.empty(
        (input_blocks, output_blocks, QVQ_V2B2_P32_LR_RINGS_PER_TILE),
        dtype=torch.uint8,
        device=source.device,
    )

    for block in range(input_blocks - 1, -1, -1):
        start = block * tile_rows
        stop = start + tile_rows
        with _qvq_phase(telemetry, "block_ldl_feedback", source.device):
            corrected = source[start:stop] + feedback[start:, start:stop].transpose(0, 1) @ error[start:]
            sequences = (
                corrected.reshape(tile_rows, output_blocks, tile_cols)
                .permute(1, 2, 0)
                .reshape(output_blocks * QVQ_V2B2_P32_LR_RINGS_PER_TILE, QVQ_V2B2_P32_LR_RING_STEPS, 2)
            )
        step_weights = None
        if viterbi_objective == "hessian_diagonal":
            diagonal_weights = D[start:stop, start:stop].diagonal().clamp_min(0)
            diagonal_mean = diagonal_weights.mean()
            if not torch.isfinite(diagonal_mean) or diagonal_mean <= torch.finfo(diagonal_weights.dtype).eps:
                raise RuntimeError("QVQ conditioned Hessian diagonal must have positive finite mean.")
            ring_weights = (diagonal_weights / diagonal_mean).reshape(-1, 2).mean(dim=1)
            step_weights = ring_weights.unsqueeze(0).expand(sequences.shape[0], -1)
        values, states, ring_selectors = _quantize_lr32_sequences(
            sequences,
            codebooks,
            bits=bits,
            trellis_batch_size=trellis_batch_size,
            tail_biting_candidates=tail_biting_candidates,
            step_weights=step_weights,
            telemetry=telemetry,
        )
        reconstructed = (
            values.reshape(output_blocks, QVQ_V2B2_P32_LR_RINGS_PER_TILE, tile_rows)
            .permute(2, 0, 1)
            .reshape(tile_rows, out_features)
            .to(torch.float32)
        )
        quantized[start:stop] = reconstructed
        error[start:stop] = source[start:stop] - reconstructed
        tile_states[block] = states.reshape(
            output_blocks,
            QVQ_V2B2_P32_LR_RINGS_PER_TILE,
            QVQ_V2B2_P32_LR_RING_STEPS,
        )
        selectors[block] = ring_selectors.reshape(output_blocks, QVQ_V2B2_P32_LR_RINGS_PER_TILE)
    return quantized.to(inner_weight.dtype), tile_states.reshape(-1, 8, 16), selectors.reshape(-1)


def block_ldlq_inner_v2b2_p32_lr(
    inner_weight: torch.Tensor,
    H: torch.Tensor,
    codebook_library: tuple[torch.Tensor, ...],
    *,
    bits: float,
    trellis_batch_size: int = 16,
    viterbi_objective: str = "euclidean",
    tail_biting_candidates: int = 1,
    telemetry: QVQQuantizationTelemetry | None = None,
    factorization: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose one module-wide alternative family for LR32 BlockLDLQ."""

    if len(codebook_library) != 4:
        raise ValueError("QVQ V2B2-P32-LR requires canonical V2 plus three complementary candidates.")
    if factorization is None:
        factorization = block_ldl_factor(H.to(torch.float32), block_size=QVQ_V2B2_P32_LR_TILE_ROWS)
    canonical_weight, canonical_states, canonical_selectors = _block_ldlq_inner_v2b2_p32_lr_candidate(
        inner_weight,
        H,
        (codebook_library[0],),
        bits=bits,
        trellis_batch_size=trellis_batch_size,
        viterbi_objective=viterbi_objective,
        tail_biting_candidates=tail_biting_candidates,
        telemetry=telemetry,
        factorization=factorization,
    )
    source = inner_weight.to(torch.float32)
    hessian = H.to(torch.float32)

    def full_loss(candidate: torch.Tensor) -> torch.Tensor:
        candidate_error = candidate.to(torch.float32) - source
        return torch.sum((hessian @ candidate_error) * candidate_error)

    best_weight = canonical_weight
    best_states = canonical_states
    best_selectors = canonical_selectors
    best_alt_id = 1
    best_loss = full_loss(canonical_weight)
    for alt_id in range(1, 4):
        candidate_weight, candidate_states, candidate_selectors = _block_ldlq_inner_v2b2_p32_lr_candidate(
            inner_weight,
            H,
            (codebook_library[0], codebook_library[alt_id]),
            bits=bits,
            trellis_batch_size=trellis_batch_size,
            viterbi_objective=viterbi_objective,
            tail_biting_candidates=tail_biting_candidates,
            telemetry=telemetry,
            factorization=factorization,
        )
        candidate_loss = full_loss(candidate_weight)
        if torch.isfinite(candidate_loss) and candidate_loss < best_loss:
            best_weight = candidate_weight
            best_states = candidate_states
            best_selectors = candidate_selectors
            best_alt_id = alt_id
            best_loss = candidate_loss
    return (
        best_weight,
        best_states,
        best_selectors,
        torch.tensor([best_alt_id], dtype=torch.uint8, device=inner_weight.device),
    )


def _yaqa_inner_v2b2_p32_lr_candidate(
    inner_weight: torch.Tensor,
    input_hessian: torch.Tensor,
    output_hessian: torch.Tensor,
    codebooks: tuple[torch.Tensor, ...],
    *,
    bits: float,
    trellis_batch_size: int,
    tail_biting_candidates: int,
    telemetry: QVQQuantizationTelemetry | None,
    factorization: tuple[BlockLDLFactorization, BlockLDLFactorization],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode one complete K32-by-N8 YAQA history for an LR32 bank family."""

    tile_rows = QVQ_V2B2_P32_LR_TILE_ROWS
    tile_cols = QVQ_V2B2_P32_LR_TILE_COLS
    in_features, out_features = inner_weight.shape
    input_blocks = in_features // tile_rows
    output_blocks = out_features // tile_cols
    input_L = factorization[0].L
    output_L = factorization[1].L
    source = inner_weight.to(torch.float32)
    quantized = torch.zeros_like(source)
    quantized_blocks = quantized.view(input_blocks, tile_rows, output_blocks, tile_cols).permute(0, 2, 1, 3)
    tile_states = torch.empty(
        (input_blocks, output_blocks, QVQ_V2B2_P32_LR_RINGS_PER_TILE, QVQ_V2B2_P32_LR_RING_STEPS),
        dtype=torch.long,
        device=source.device,
    )
    selectors = torch.empty(
        (input_blocks, output_blocks, QVQ_V2B2_P32_LR_RINGS_PER_TILE),
        dtype=torch.uint8,
        device=source.device,
    )
    transformed_error = input_L.transpose(0, 1) @ source @ output_L
    schedule = _yaqa_anti_diagonal_schedule(
        source.device,
        input_blocks,
        output_blocks,
        tile_rows,
        tile_cols,
    )
    for _, input_indices, output_indices, _, input_rows, output_rows in schedule:
        with _qvq_phase(telemetry, "yaqa_feedback", source.device):
            transformed_blocks = transformed_error.view(
                input_blocks,
                tile_rows,
                output_blocks,
                tile_cols,
            ).permute(0, 2, 1, 3)
            corrected_tiles = transformed_blocks[input_indices, output_indices]
            sequences = (
                corrected_tiles.permute(0, 2, 1)
                .reshape(-1, QVQ_V2B2_P32_LR_RING_STEPS, 2)
                .contiguous()
            )
        values, states, ring_selectors = _quantize_lr32_sequences(
            sequences,
            codebooks,
            bits=bits,
            trellis_batch_size=trellis_batch_size,
            tail_biting_candidates=tail_biting_candidates,
            telemetry=telemetry,
        )
        diagonal_blocks = input_indices.numel()
        reconstructed = (
            values.reshape(diagonal_blocks, QVQ_V2B2_P32_LR_RINGS_PER_TILE, tile_rows)
            .permute(0, 2, 1)
            .to(torch.float32)
        )
        with _qvq_phase(telemetry, "yaqa_commit", source.device):
            quantized_blocks[input_indices, output_indices] = reconstructed
            tile_states[input_indices, output_indices] = states.reshape(diagonal_blocks, 8, 16)
            selectors[input_indices, output_indices] = ring_selectors.reshape(diagonal_blocks, 8)
        with _qvq_phase(telemetry, "yaqa_feedback_update", source.device):
            left_factor = input_L[input_rows].transpose(0, 1)
            right_factor = torch.bmm(reconstructed, output_L[output_rows]).reshape(
                diagonal_blocks * tile_rows,
                out_features,
            )
            transformed_error.addmm_(left_factor, right_factor, alpha=-1)
    return quantized.to(inner_weight.dtype), tile_states.reshape(-1, 8, 16), selectors.reshape(-1)


def yaqa_inner_v2b2_p32_lr(
    inner_weight: torch.Tensor,
    input_hessian: torch.Tensor,
    output_hessian: torch.Tensor,
    codebook_library: tuple[torch.Tensor, ...],
    *,
    bits: float,
    trellis_batch_size: int = 16,
    tail_biting_candidates: int = 1,
    family_mode: str = "reselect",
    block_family_id: int | None = None,
    block_input_hessian: torch.Tensor | None = None,
    diagnostics: dict[str, object] | None = None,
    factorization: tuple[BlockLDLFactorization, BlockLDLFactorization] | None = None,
    telemetry: QVQQuantizationTelemetry | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose one module-wide alternative family for the LR32 YAQA codec."""

    if len(codebook_library) != 4:
        raise ValueError("YAQA V2B2-P32-LR requires canonical V2 plus three complementary candidates.")
    if family_mode not in {"fixed_block_ldlq", "reselect"}:
        raise ValueError("YAQA V2B2-P32-LR family mode must be `fixed_block_ldlq` or `reselect`.")
    if factorization is None:
        input_factor = stabilized_block_ldl_factor(
            input_hessian,
            block_size=QVQ_V2B2_P32_LR_TILE_ROWS,
            retry_damping=torch.tensor(torch.finfo(torch.float32).eps, device=input_hessian.device),
        )
        output_factor = stabilized_block_ldl_factor(
            output_hessian,
            block_size=QVQ_V2B2_P32_LR_TILE_COLS,
            retry_damping=torch.tensor(torch.finfo(torch.float32).eps, device=output_hessian.device),
        )
        factorization = input_factor, output_factor

    canonical_weight, canonical_states, canonical_selectors = _yaqa_inner_v2b2_p32_lr_candidate(
        inner_weight,
        input_hessian,
        output_hessian,
        (codebook_library[0],),
        bits=bits,
        trellis_batch_size=trellis_batch_size,
        tail_biting_candidates=tail_biting_candidates,
        telemetry=telemetry,
        factorization=factorization,
    )
    if block_family_id is None and family_mode == "fixed_block_ldlq":
        _, _, _, block_alt_id = block_ldlq_inner_v2b2_p32_lr(
            inner_weight,
            input_hessian if block_input_hessian is None else block_input_hessian,
            codebook_library,
            bits=bits,
            trellis_batch_size=trellis_batch_size,
            tail_biting_candidates=tail_biting_candidates,
        )
        block_family_id = int(block_alt_id.item())
    if block_family_id is not None and block_family_id not in range(4):
        raise ValueError("YAQA V2B2-P32-LR fixed family ID must be 0, 1, 2, or 3.")
    alternative_ids = (
        (1, 2, 3)
        if family_mode == "reselect" and block_family_id is None
        else ()
        if block_family_id == 0
        else (block_family_id,)
    )
    source = inner_weight.to(torch.float32)
    input_fp32 = input_hessian.to(torch.float32)
    output_fp32 = output_hessian.to(torch.float32)

    def full_loss(candidate: torch.Tensor) -> torch.Tensor:
        candidate_error = candidate.to(torch.float32) - source
        return torch.einsum("ij,ik,kl,lj->", candidate_error, input_fp32, candidate_error, output_fp32)

    best_weight = canonical_weight
    best_states = canonical_states
    best_selectors = canonical_selectors
    best_alt_id = 1 if block_family_id in (None, 0) else block_family_id
    best_loss = full_loss(canonical_weight)
    for alt_id in alternative_ids:
        candidate_weight, candidate_states, candidate_selectors = _yaqa_inner_v2b2_p32_lr_candidate(
            inner_weight,
            input_hessian,
            output_hessian,
            (codebook_library[0], codebook_library[alt_id]),
            bits=bits,
            trellis_batch_size=trellis_batch_size,
            tail_biting_candidates=tail_biting_candidates,
            telemetry=telemetry,
            factorization=factorization,
        )
        candidate_loss = full_loss(candidate_weight)
        if torch.isfinite(candidate_loss) and candidate_loss < best_loss:
            best_weight = candidate_weight
            best_states = candidate_states
            best_selectors = candidate_selectors
            best_alt_id = alt_id
            best_loss = candidate_loss
    if diagnostics is not None:
        diagnostics["fallback_to_v2"] = not bool(torch.count_nonzero(best_selectors))
        diagnostics["block_family_id"] = int(best_alt_id if block_family_id is None else block_family_id)
    return (
        best_weight,
        best_states,
        best_selectors,
        torch.tensor([best_alt_id], dtype=torch.uint8, device=inner_weight.device),
    )


def quantize_qvq_linear(
    weight: torch.Tensor,
    H: torch.Tensor,
    *,
    bits: float,
    output_hessian: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    seed: int = 0,
    damp_percent: float | None = None,
    trellis_batch_size: int | None = None,
    codebook_version: str = PGC16_CODEBOOK_VERSION,
    output_channel_scale_optimization: bool = False,
    module_scale_search: bool = False,
    viterbi_objective: str = "euclidean",
    tail_biting_candidates: int = 1,
    rounding: str = "block_ldlq",
    yaqa_v2b2_family_mode: str = "reselect",
    yaqa_v2b2_fixed_family_id: int | None = None,
    yaqa_sample_strategy: str = "full",
    yaqa_spectral_refinement: bool = False,
    yaqa_spectral_ranks: tuple[int, ...] = (8, 16, 32),
    yaqa_spectral_lambdas: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0),
    yaqa_spectral_push: bool = False,
    yaqa_spectral_push_alphas: tuple[float, ...] = (0.25, 0.5, 1.0),
    yaqa_spectral_localized: bool = False,
    yaqa_spectral_localized_alphas: tuple[float, ...] = (0.25, 0.5, 1.0),
    yaqa_spectral_localized_max_segments: int = 8,
    yaqa_spectral_localized_max_changes: int = 1,
    yaqa_spectral_localized_replay_candidates: int = 0,
    yaqa_spectral_localized_direct_replay_candidates: int = 0,
    viterbi_minimum_proxy_improvement: float = 0.0,
    vector_size: int = 2,
    trellis_window: int = 16,
    dual_v2: bool = False,
    v2b4_p64: bool = False,
    v2b2_p32: bool = False,
    v2b2_p32_lr: bool = False,
    experimental_codebook: torch.Tensor | None = None,
    telemetry: QVQQuantizationTelemetry | None = None,
    bank_count: int = 1,
    propagated_inputs: torch.Tensor | None = None,
    propagated_target_output: torch.Tensor | None = None,
    propagated_acceptance: Callable[[torch.Tensor, torch.Tensor], bool] | None = None,
    propagated_candidate_score: Callable[[torch.Tensor], float] | None = None,
    propagated_candidate_gradient: Callable[[torch.Tensor], torch.Tensor] | None = None,
    input_hessian_preparation: QVQInputHessianPreparation | None = None,
    # Exact Viterbi survivor-pruning policy (`QVQConfig.viterbi_pruning`).
    # `None` resolves to `auto`, which reproduces today's automatic behavior.
    viterbi_pruning: object | None = None,
) -> QVQLinearQuantizationResult:
    """Run RHT, BlockLDLQ/YAQA, PGC16 TCQ, and planar packing for a linear.

    ``experimental_codebook`` exists only for offline candidate screening.
    The returned dense reconstruction may be evaluated, but its trellis must
    not be serialized because QVQ checkpoints contain no custom-codebook
    metadata and the runtime decoder always uses the canonical mapping.
    """

    bits = normalize_qvq_rate(bits)
    if not isinstance(dual_v2, bool):
        raise TypeError("QVQ `dual_v2` must be a bool.")
    if not isinstance(v2b4_p64, bool):
        raise TypeError("QVQ `v2b4_p64` must be a bool.")
    if not isinstance(v2b2_p32, bool):
        raise TypeError("QVQ `v2b2_p32` must be a bool.")
    if not isinstance(v2b2_p32_lr, bool):
        raise TypeError("QVQ `v2b2_p32_lr` must be a bool.")
    if sum((dual_v2, v2b4_p64, v2b2_p32, v2b2_p32_lr)) > 1:
        raise ValueError("QVQ Dual-V2, V2B4-P64, V2B2-P32, and V2B2-P32-LR are mutually exclusive.")
    if vector_size not in (2, 4) or (vector_size == 4 and bits > 4):
        raise ValueError("QVQ `vector_size` must be 2, or 4 for rates W1-W4.")
    if trellis_window not in (16, 18):
        raise ValueError("QVQ `trellis_window` must be 16 or the experimental L18/V4 value 18.")
    if trellis_window == 18 and vector_size != 4:
        raise ValueError("QVQ L18 requires `vector_size=4`.")
    if trellis_window == 18 and bits > 2.5:
        raise ValueError("QVQ L18 supports only rates W1 through W2.5.")
    if trellis_window == 18 and bank_count != 1:
        raise ValueError("QVQ L18 uses implicit history-selected banks and requires `bank_count=1`.")
    if dual_v2 and (vector_size != 2 or trellis_window != 16 or bank_count != 1):
        raise ValueError("QVQ Dual-V2 requires vector_size=2, trellis_window=16, and bank_count=1.")
    if v2b4_p64 and (vector_size != 2 or trellis_window != 16 or bank_count != 4 or bits > 3.5):
        raise ValueError("QVQ V2B4-P64 requires vector_size=2, trellis_window=16, bank_count=4, and W1-W3.5.")
    if v2b2_p32 and (vector_size != 2 or trellis_window != 16 or bank_count != 2 or bits > 3.5):
        raise ValueError("QVQ V2B2-P32 requires vector_size=2, trellis_window=16, bank_count=2, and W1-W3.5.")
    if v2b2_p32_lr and (vector_size != 2 or trellis_window != 16 or bank_count != 2 or bits > 3.5):
        raise ValueError(
            "QVQ V2B2-P32-LR requires vector_size=2, trellis_window=16, bank_count=2, and W1-W3.5."
        )
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("QVQ weight must be a floating-point matrix.")
    out_features, in_features = weight.shape
    if in_features < 1 or out_features < 1:
        raise ValueError("QVQ linear dimensions must be positive.")
    if v2b2_p32_lr:
        if in_features % QVQ_V2B2_P32_LR_TILE_ROWS or out_features % QVQ_V2B2_P32_LR_TILE_COLS:
            raise ValueError("QVQ V2B2-P32-LR dimensions must be divisible by K32 and N8.")
    elif in_features % 16 or out_features % 16:
        raise ValueError("QVQ linear dimensions must be divisible by 16.")
    if tuple(H.shape) != (in_features, in_features):
        raise ValueError("QVQ Hessian shape must match the linear input dimension.")
    if output_hessian is not None and tuple(output_hessian.shape) != (
        out_features,
        out_features,
    ):
        raise ValueError("YAQA output Hessian shape must match the linear output dimension.")
    if bias is not None and tuple(bias.shape) != (out_features,):
        raise ValueError("QVQ bias shape must match the linear output dimension.")
    if not H.is_floating_point():
        raise TypeError("QVQ Hessian must use a floating-point dtype.")
    if output_hessian is not None and not output_hessian.is_floating_point():
        raise TypeError("YAQA output Hessian must use a floating-point dtype.")
    if not torch.isfinite(weight).all() or not torch.isfinite(H).all():
        raise ValueError("QVQ weight and Hessian must contain only finite values.")
    if output_hessian is not None and not torch.isfinite(output_hessian).all():
        raise ValueError("YAQA output Hessian must contain only finite values.")
    _validate_fp32_representable(weight, name="QVQ weight")
    _validate_fp32_representable(H, name="QVQ Hessian")
    if output_hessian is not None:
        _validate_fp32_representable(output_hessian, name="YAQA output Hessian")
    if bias is not None:
        if not bias.is_floating_point():
            raise TypeError("QVQ bias must use a floating-point dtype.")
        if not torch.isfinite(bias).all():
            raise ValueError("QVQ bias must contain only finite values.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("QVQ seed must be an integer.")
    if not isinstance(output_channel_scale_optimization, bool):
        raise TypeError("QVQ output-channel scale optimization must be boolean.")
    if not isinstance(module_scale_search, bool):
        raise TypeError("QVQ module-scale search must be boolean.")
    if isinstance(bank_count, bool) or not isinstance(bank_count, int) or bank_count not in (1, 2, 4):
        raise ValueError("QVQ `bank_count` must be 1, 2, or 4.")
    if not isinstance(rounding, str):
        raise TypeError("QVQ rounding must be a string.")
    if input_hessian_preparation is not None and not isinstance(
        input_hessian_preparation, QVQInputHessianPreparation
    ):
        raise TypeError("QVQ shared input-Hessian preparation has an invalid type.")
    rounding = rounding.strip().lower()
    if rounding not in {"block_ldlq", "yaqa"}:
        raise ValueError("QVQ rounding must be `block_ldlq` or `yaqa`.")
    if not isinstance(yaqa_v2b2_family_mode, str):
        raise TypeError("QVQ YAQA V2B2 family mode must be a string.")
    yaqa_v2b2_family_mode = yaqa_v2b2_family_mode.strip().lower()
    if yaqa_v2b2_family_mode not in {"fixed_block_ldlq", "reselect"}:
        raise ValueError("QVQ YAQA V2B2 family mode must be `fixed_block_ldlq` or `reselect`.")
    if yaqa_v2b2_fixed_family_id is not None and (
        isinstance(yaqa_v2b2_fixed_family_id, bool)
        or not isinstance(yaqa_v2b2_fixed_family_id, int)
        or yaqa_v2b2_fixed_family_id not in range(4)
    ):
        raise ValueError("QVQ YAQA V2B2 fixed family ID must be 0, 1, 2, or 3.")
    if not isinstance(yaqa_sample_strategy, str):
        raise TypeError("QVQ YAQA sample strategy must be a string.")
    yaqa_sample_strategy = yaqa_sample_strategy.strip().lower()
    if yaqa_sample_strategy not in QVQ_YAQA_SAMPLE_TILE_COUNTS:
        raise ValueError(
            "QVQ YAQA sample strategy must be `full`, `32_16x16`, `64_16x16`, `96_16x16`, `128_16x16`, "
            "or `256_16x16`."
        )
    if yaqa_v2b2_family_mode != "reselect" and yaqa_sample_strategy != "full":
        raise ValueError("QVQ YAQA sampled family selection requires `yaqa_v2b2_family_mode=reselect`.")
    if not isinstance(yaqa_spectral_refinement, bool):
        raise TypeError("QVQ YAQA spectral refinement must be boolean.")
    if not isinstance(yaqa_spectral_push, bool):
        raise TypeError("QVQ YAQA spectral push must be boolean.")
    if not isinstance(yaqa_spectral_localized, bool):
        raise TypeError("QVQ YAQA localized spectral refinement must be boolean.")
    if sum((yaqa_spectral_refinement, yaqa_spectral_push, yaqa_spectral_localized)) > 1:
        raise ValueError("QVQ YAQA spectral refinement experiments are mutually exclusive.")
    if not isinstance(yaqa_spectral_ranks, (tuple, list)) or not yaqa_spectral_ranks or any(
        isinstance(rank, bool) or not isinstance(rank, int) or rank < 1 for rank in yaqa_spectral_ranks
    ):
        raise ValueError("QVQ YAQA spectral ranks must be a non-empty sequence of positive integers.")
    yaqa_spectral_ranks = tuple(dict.fromkeys(yaqa_spectral_ranks))
    if not isinstance(yaqa_spectral_lambdas, (tuple, list)) or not yaqa_spectral_lambdas or any(
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(float(strength))
        or float(strength) <= 0
        for strength in yaqa_spectral_lambdas
    ):
        raise ValueError("QVQ YAQA spectral lambdas must be a non-empty sequence of finite positive values.")
    yaqa_spectral_lambdas = tuple(dict.fromkeys(float(strength) for strength in yaqa_spectral_lambdas))
    if not isinstance(yaqa_spectral_push_alphas, (tuple, list)) or not yaqa_spectral_push_alphas or any(
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
        or float(alpha) <= 0
        for alpha in yaqa_spectral_push_alphas
    ):
        raise ValueError("QVQ YAQA spectral push alphas must be a non-empty sequence of finite positive values.")
    yaqa_spectral_push_alphas = tuple(dict.fromkeys(float(alpha) for alpha in yaqa_spectral_push_alphas))
    if (
        not isinstance(yaqa_spectral_localized_alphas, (tuple, list))
        or not yaqa_spectral_localized_alphas
        or any(
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(float(alpha))
            or float(alpha) <= 0
            for alpha in yaqa_spectral_localized_alphas
        )
    ):
        raise ValueError("QVQ YAQA localized spectral alphas must be a non-empty sequence of finite positive values.")
    yaqa_spectral_localized_alphas = tuple(
        dict.fromkeys(float(alpha) for alpha in yaqa_spectral_localized_alphas)
    )
    if (
        isinstance(yaqa_spectral_localized_max_segments, bool)
        or not isinstance(yaqa_spectral_localized_max_segments, int)
        or yaqa_spectral_localized_max_segments < 1
    ):
        raise ValueError("QVQ YAQA localized spectral max_segments must be a positive integer.")
    if (
        isinstance(yaqa_spectral_localized_max_changes, bool)
        or not isinstance(yaqa_spectral_localized_max_changes, int)
        or yaqa_spectral_localized_max_changes < 1
        or yaqa_spectral_localized_max_changes > yaqa_spectral_localized_max_segments
    ):
        raise ValueError("QVQ YAQA localized spectral max_changes must be between 1 and max_segments.")
    if (
        isinstance(yaqa_spectral_localized_replay_candidates, bool)
        or not isinstance(yaqa_spectral_localized_replay_candidates, int)
        or yaqa_spectral_localized_replay_candidates < 0
    ):
        raise ValueError("QVQ YAQA localized replay candidate count must be a nonnegative integer.")
    if (
        isinstance(yaqa_spectral_localized_direct_replay_candidates, bool)
        or not isinstance(yaqa_spectral_localized_direct_replay_candidates, int)
        or yaqa_spectral_localized_direct_replay_candidates < 0
        or yaqa_spectral_localized_direct_replay_candidates
        > yaqa_spectral_localized_replay_candidates
    ):
        raise ValueError(
            "QVQ YAQA localized direct replay candidate count must be between zero and the replay shortlist."
        )
    if damp_percent is None:
        damp_percent = YAQA_DEFAULT_REGULARIZATION if rounding == "yaqa" else 0.01
    if isinstance(damp_percent, bool) or not isinstance(damp_percent, (int, float)):
        raise TypeError("QVQ damping percent must be a real scalar.")
    damp_percent = float(damp_percent)
    if not math.isfinite(damp_percent) or damp_percent < 0:
        raise ValueError("QVQ damping percent must be finite and nonnegative.")
    if (
        isinstance(tail_biting_candidates, bool)
        or not isinstance(tail_biting_candidates, int)
        or tail_biting_candidates < 1
    ):
        raise ValueError("QVQ tail-biting candidate count must be a positive integer.")
    if viterbi_objective not in {"euclidean", "hessian_diagonal"}:
        raise ValueError("QVQ Viterbi objective must be `euclidean` or `hessian_diagonal`.")
    if isinstance(viterbi_minimum_proxy_improvement, bool) or not isinstance(
        viterbi_minimum_proxy_improvement, (int, float)
    ):
        raise TypeError("QVQ Viterbi minimum proxy improvement must be a real scalar.")
    viterbi_minimum_proxy_improvement = float(viterbi_minimum_proxy_improvement)
    if not math.isfinite(viterbi_minimum_proxy_improvement) or viterbi_minimum_proxy_improvement < 0:
        raise ValueError("QVQ Viterbi minimum proxy improvement must be finite and nonnegative.")
    if viterbi_minimum_proxy_improvement > 0 and viterbi_objective != "hessian_diagonal":
        raise ValueError("QVQ Viterbi minimum proxy improvement requires `hessian_diagonal` objective.")
    if rounding == "yaqa":
        if output_hessian is None:
            raise ValueError("YAQA rounding requires an output Hessian from full-model calibration.")
        if output_channel_scale_optimization:
            raise ValueError("YAQA does not support the independently optimized output-channel scale control.")
        if module_scale_search:
            raise ValueError("YAQA does not support input-Hessian-only module-scale search.")
        if viterbi_objective != "euclidean":
            raise ValueError("YAQA requires the Euclidean PGC16 tile objective.")
        if input_hessian_preparation is not None:
            raise ValueError("YAQA input/output factors are module-specific and cannot use shared input preparation.")
    if yaqa_v2b2_fixed_family_id is not None and (
        rounding != "yaqa"
        or not (v2b2_p32 or v2b2_p32_lr)
        or yaqa_v2b2_family_mode != "fixed_block_ldlq"
        or yaqa_sample_strategy != "full"
        or yaqa_spectral_refinement
        or yaqa_spectral_push
        or yaqa_spectral_localized
        or propagated_inputs is not None
    ):
        raise ValueError(
            "QVQ YAQA V2B2 fixed-family encoding requires V2B2-P32 or V2B2-P32-LR YAQA, "
            "`fixed_block_ldlq` mode, full family scoring, and no subsequent candidate refinement."
        )
    if (yaqa_spectral_refinement or yaqa_spectral_push or yaqa_spectral_localized) and (
        rounding != "yaqa" or not v2b2_p32
    ):
        raise ValueError("YAQA spectral experiment requires V2B2-P32 with YAQA rounding.")
    if bank_count == 4 and vector_size != 4 and not v2b4_p64:
        raise ValueError("QVQ four-bank selection requires V4 or V2B4-P64.")
    if bank_count == 4 and (module_scale_search or output_channel_scale_optimization):
        raise ValueError("QVQ four-bank selection currently excludes scale-search controls.")
    if bank_count == 4 and experimental_codebook is not None:
        raise ValueError("QVQ four-bank selection requires the canonical rate-keyed PGC16 codebooks.")
    if bank_count == 2 and not (v2b2_p32 or v2b2_p32_lr):
        raise ValueError("QVQ two-bank selection requires V2B2-P32 or V2B2-P32-LR.")
    if bank_count == 2 and (module_scale_search or output_channel_scale_optimization or experimental_codebook is not None):
        raise ValueError("QVQ V2B2-P32 formats require canonical codebooks and exclude scale-search controls.")
    if v2b2_p32_lr and input_hessian_preparation is not None:
        raise ValueError("QVQ V2B2-P32-LR does not accept shared K16 input-Hessian preparation.")
    if v2b2_p32_lr and yaqa_sample_strategy != "full":
        raise ValueError("QVQ V2B2-P32-LR YAQA currently requires full module family scoring.")
    if v2b2_p32_lr and (yaqa_spectral_refinement or yaqa_spectral_push or yaqa_spectral_localized):
        raise ValueError("QVQ V2B2-P32-LR does not support P32-specific YAQA spectral experiments.")
    if v2b2_p32_lr and propagated_inputs is not None:
        raise ValueError("QVQ V2B2-P32-LR propagated replay is not implemented.")
    if (propagated_inputs is None) != (propagated_target_output is None):
        raise ValueError("QVQ propagated bank selection requires both held-out inputs and target outputs.")
    if propagated_inputs is not None:
        localized_v2b2_propagation = v2b2_p32 and rounding == "yaqa" and yaqa_spectral_localized
        if v2b4_p64:
            raise ValueError("QVQ banked V2 propagation replay is not enabled in the initial reference slice.")
        if not localized_v2b2_propagation and (
            bank_count != 4 or vector_size != 4 or rounding != "block_ldlq"
        ):
            raise ValueError("QVQ propagated bank selection requires bank_count=4, vector_size=4, and block_ldlq.")
        if localized_v2b2_propagation and (bank_count != 2 or vector_size != 2):
            raise ValueError("QVQ localized V2B2 propagation requires bank_count=2 and vector_size=2.")
        if propagated_inputs.ndim != 2 or propagated_target_output.ndim != 2:
            raise ValueError("QVQ propagated gate tensors must be rank-2.")
        if propagated_inputs.shape[0] != propagated_target_output.shape[0] or propagated_inputs.shape[1] != in_features:
            raise ValueError("QVQ propagated inputs must match the linear input width and target rows.")
        if propagated_target_output.shape[1] != out_features:
            raise ValueError("QVQ propagated targets must match the linear output width.")
        if not torch.isfinite(propagated_inputs).all() or not torch.isfinite(propagated_target_output).all():
            raise ValueError("QVQ propagated gate tensors must be finite.")
        propagated_inputs = propagated_inputs.to(device=weight.device, dtype=torch.float32)
        propagated_target_output = propagated_target_output.to(device=weight.device, dtype=torch.float32)
    if propagated_acceptance is not None and not callable(propagated_acceptance):
        raise TypeError("QVQ propagated acceptance gate must be callable.")
    if propagated_candidate_score is not None and not callable(propagated_candidate_score):
        raise TypeError("QVQ propagated candidate scorer must be callable.")
    if propagated_candidate_gradient is not None and not callable(propagated_candidate_gradient):
        raise TypeError("QVQ propagated candidate gradient must be callable.")
    if propagated_inputs is not None and propagated_acceptance is None:
        raise ValueError("QVQ propagated bank selection requires an explicit acceptance gate.")
    if (propagated_candidate_score is None) != (yaqa_spectral_localized_replay_candidates == 0):
        raise ValueError(
            "QVQ localized full-horizon scoring requires both a candidate scorer and a positive shortlist."
        )
    if propagated_candidate_gradient is not None and propagated_candidate_score is None:
        raise ValueError("QVQ propagated candidate gradient requires full-horizon candidate scoring.")

    device = weight.device
    if telemetry is not None:
        telemetry.count("weight_elements", weight.numel())
        telemetry.count("input_features", in_features)
        telemetry.count("output_features", out_features)
        pruning_policy = resolve_viterbi_pruning_policy(viterbi_pruning)
        telemetry.count("viterbi_pruning_configured")
        telemetry.count(f"viterbi_pruning_mode_{pruning_policy.mode}")
        telemetry.count(f"viterbi_pruning_strategy_{pruning_policy.strategy}")
        telemetry.count("viterbi_pruning_exact", int(pruning_policy.exact))
        telemetry.count(f"viterbi_pruning_fallback_{pruning_policy.fallback}")
    if trellis_batch_size is None:
        trellis_batch_size = default_qvq_trellis_batch_size(
            bits,
            device,
            trellis_window=trellis_window,
        )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    SU = torch.randint(0, 2, (in_features,), generator=generator, dtype=torch.int8).mul_(2).sub_(1)
    SV_sign = torch.randint(0, 2, (out_features,), generator=generator, dtype=torch.int8).mul_(2).sub_(1)
    SU = SU.to(device=device, dtype=torch.float32)
    SV_sign = SV_sign.to(device=device, dtype=torch.float32)

    with _qvq_phase(telemetry, "rht_weight", device):
        transformed_weight = rht_preprocess_weight(weight, SU, SV_sign)
    with _qvq_phase(telemetry, "rht_hessian", device):
        block_ldlq_control_H = None
        if input_hessian_preparation is not None:
            preparation = input_hessian_preparation
            if (
                preparation.source_hessian is not H
                or preparation.source_data_ptr != H.data_ptr()
                or preparation.source_version != _safe_tensor_version(H)
                or preparation.transformed_version != _safe_tensor_version(preparation.hessian)
                or preparation.factor_versions
                != (
                    _safe_tensor_version(preparation.factorization[0]),
                    _safe_tensor_version(preparation.factorization[1]),
                )
                or preparation.block_size != 16
                or preparation.seed != seed
                or preparation.damp_percent != damp_percent
                or preparation.hessian.device != device
                or tuple(preparation.hessian.shape) != tuple(H.shape)
                or any(factor.device != device for factor in preparation.factorization)
                or any(tuple(factor.shape) != tuple(H.shape) for factor in preparation.factorization)
            ):
                raise ValueError("QVQ shared input-Hessian preparation does not match the source geometry.")
            transformed_H = preparation.hessian
            damping = preparation.damping
        else:
            transformed_H = rht_preprocess_hessian(H.to(device=device), SU)
            transformed_H = (transformed_H + transformed_H.transpose(0, 1)) * 0.5
            mean_diagonal = transformed_H.diagonal().abs().mean()
            if rounding == "yaqa" and (v2b4_p64 or v2b2_p32 or v2b2_p32_lr):
                block_ldlq_control_H = transformed_H.clone()
                block_control_damping = torch.maximum(
                    mean_diagonal * 0.01,
                    torch.tensor(torch.finfo(torch.float32).eps, device=device),
                )
                block_ldlq_control_H.diagonal().add_(block_control_damping)
            damping = torch.maximum(
                mean_diagonal * damp_percent,
                torch.tensor(torch.finfo(torch.float32).eps, device=device),
            )
            transformed_H.diagonal().add_(damping)
    transformed_output_hessian = None
    if output_hessian is not None:
        with _qvq_phase(telemetry, "rht_output_hessian", device):
            transformed_output_hessian = rht_preprocess_hessian(output_hessian.to(device=device), SV_sign)
            transformed_output_hessian = (
                transformed_output_hessian + transformed_output_hessian.transpose(0, 1)
            ) * 0.5
            output_mean_diagonal = transformed_output_hessian.diagonal().abs().mean()
            output_damping = torch.maximum(
                output_mean_diagonal * damp_percent,
                torch.tensor(torch.finfo(torch.float32).eps, device=device),
            )
            transformed_output_hessian.diagonal().add_(output_damping)

    prepared_yaqa_factorization = None
    if rounding == "yaqa":
        assert transformed_output_hessian is not None
        with _qvq_phase(telemetry, "yaqa_factorization", device):
            input_factor = stabilized_block_ldl_factor(
                transformed_H,
                block_size=QVQ_V2B2_P32_LR_TILE_ROWS if v2b2_p32_lr else 16,
                retry_damping=damping,
            )
            output_factor = stabilized_block_ldl_factor(
                transformed_output_hessian,
                block_size=QVQ_V2B2_P32_LR_TILE_COLS if v2b2_p32_lr else 16,
                retry_damping=output_damping,
            )
            transformed_H = input_factor.hessian
            transformed_output_hessian = output_factor.hessian
            prepared_yaqa_factorization = input_factor, output_factor
            if telemetry is not None:
                telemetry.count("yaqa_input_damping_retries", input_factor.retry_count)
                telemetry.count("yaqa_output_damping_retries", output_factor.retry_count)

    with _qvq_phase(telemetry, "codebook", device):
        if experimental_codebook is None:
            codebook_dtype = torch.float16 if device.type == "cuda" else torch.float32
            codebook = _canonical_qvq_codebook(
                device=device,
                vector_size=vector_size,
                trellis_window=trellis_window,
                bits=bits,
                codebook_version=codebook_version,
                dtype=codebook_dtype,
            )
        else:
            expected_shape = (1 << trellis_window, vector_size)
            if experimental_codebook.shape != expected_shape:
                raise ValueError(f"QVQ experimental codebook must have shape {expected_shape}.")
            if not experimental_codebook.is_floating_point():
                raise TypeError("QVQ experimental codebook must use a floating-point dtype.")
            if experimental_codebook.device != device:
                raise ValueError("QVQ experimental codebook must be on the weight device.")
            if not torch.isfinite(experimental_codebook).all():
                raise ValueError("QVQ experimental codebook must contain only finite values.")
            codebook = experimental_codebook.to(dtype=torch.float32)
    bank_codebooks = None
    bank_codebook_stack = None
    segmented_bank_stack = None
    bank_codebook_pair_stacks = None
    if bank_count in (2, 4):
        bank_codebooks = (
            _canonical_qvq_v2b4_banks(
                device=device,
                bits=bits,
                codebook_version=codebook_version,
                dtype=codebook.dtype,
            )
            if v2b4_p64 or v2b2_p32 or v2b2_p32_lr
            else _canonical_qvq_v4_banks(
                device=device,
                bits=bits,
                codebook_version=codebook_version,
                dtype=codebook.dtype,
            )
        )
        if v2b4_p64:
            segmented_bank_stack = _canonical_qvq_v2b4_bank_stack(
                device=device,
                bits=bits,
                codebook_version=codebook_version,
                dtype=codebook.dtype,
            )
        elif v2b2_p32 or v2b2_p32_lr:
            bank_codebook_pair_stacks = _canonical_qvq_v2b2_pair_stacks(
                device=device,
                bits=bits,
                codebook_version=codebook_version,
                dtype=codebook.dtype,
            )
        elif (
            not v2b4_p64
            and not v2b2_p32
            and not v2b2_p32_lr
            and rounding in {"yaqa", "block_ldlq"}
            and device.type == "cuda"
            and tail_biting_candidates == 1
        ):
            bank_codebook_stack = _canonical_qvq_v4_bank_stack(
                device=device,
                bits=bits,
                codebook_version=codebook_version,
                dtype=codebook.dtype,
            )
    source_rms = transformed_weight.square().mean().sqrt()
    scale = (source_rms / PGC16_NORMALIZATION_RMS * pgc16_scale_factor(bits)).clamp_min(torch.finfo(torch.float32).eps)
    prepared_block_factors = None
    if rounding == "block_ldlq":
        with _qvq_phase(telemetry, "block_ldl_factor", device):
            prepared_block_factors = (
                input_hessian_preparation.factorization
                if input_hessian_preparation is not None
                else block_ldl_factor(
                    transformed_H.to(torch.float32),
                    block_size=QVQ_V2B2_P32_LR_TILE_ROWS if v2b2_p32_lr else 16,
                )
            )
    needs_bank0_oracle = (
        bank_codebooks is not None
        and rounding == "block_ldlq"
        and not v2b4_p64
        and not v2b2_p32
        and not v2b2_p32_lr
    )
    yaqa_bank_diagnostics: dict[str, object] = {}
    def encode_at_scale(
        candidate_scale: torch.Tensor,
        *,
        objective: str = "euclidean",
        include_bank0_oracle: bool = False,
    ):
        normalized_weight = transformed_weight / candidate_scale
        if rounding == "yaqa":
            assert transformed_output_hessian is not None
            # The exact incremental recurrence avoids hundreds of shrinking
            # suffix products on attention-sized CUDA projections.  MLP
            # Small matrices use the exact dense transformed-error recurrence.
            # Large MLP matrices use two factored transformed-error caches and
            # one grouped FP32 GEMM per anti-diagonal. This avoids thousands
            # of overlapping suffix materializations while bounding corrected-
            # tile drift against the direct FP32 expression to 1e-6.
            incremental_cuda_feedback = device.type == "cuda" and max(normalized_weight.shape) <= 2048
            incremental_cuda_factored_feedback = device.type == "cuda" and max(normalized_weight.shape) > 2048
            incremental_cpu_factored_feedback = (
                device.type in ("cpu", "mps") and qvq_cpu_supported()
            )
            if v2b4_p64:
                assert bank_codebooks is not None
                return yaqa_inner_v2b4_p64(
                    normalized_weight,
                    transformed_H,
                    transformed_output_hessian,
                    bank_codebooks,
                    block_input_hessian=block_ldlq_control_H,
                    bits=bits,
                    trellis_batch_size=trellis_batch_size,
                    tail_biting_candidates=tail_biting_candidates,
                    viterbi_pruning=viterbi_pruning,
                    diagnostics=yaqa_bank_diagnostics,
                    factorization=prepared_yaqa_factorization,
                    segmented_bank_stack=segmented_bank_stack,
                    telemetry=telemetry,
                    _incremental_cuda_feedback=incremental_cuda_feedback,
                    _incremental_cuda_factored_feedback=incremental_cuda_factored_feedback,
                    _incremental_cpu_factored_feedback=incremental_cpu_factored_feedback,
                    _trusted_inputs=True,
                )
            if v2b2_p32_lr:
                assert bank_codebooks is not None
                return yaqa_inner_v2b2_p32_lr(
                    normalized_weight,
                    transformed_H,
                    transformed_output_hessian,
                    bank_codebooks,
                    bits=bits,
                    trellis_batch_size=trellis_batch_size,
                    tail_biting_candidates=tail_biting_candidates,
                    family_mode=yaqa_v2b2_family_mode,
                    block_family_id=yaqa_v2b2_fixed_family_id,
                    block_input_hessian=block_ldlq_control_H,
                    diagnostics=yaqa_bank_diagnostics,
                    factorization=prepared_yaqa_factorization,
                    telemetry=telemetry,
                )
            if v2b2_p32:
                assert bank_codebooks is not None
                if yaqa_v2b2_fixed_family_id == 0:
                    canonical_weight, canonical_states = yaqa_inner(
                        normalized_weight,
                        transformed_H,
                        transformed_output_hessian,
                        codebook,
                        bits=bits,
                        trellis_batch_size=trellis_batch_size,
                        tail_biting_candidates=tail_biting_candidates,
                        viterbi_pruning=viterbi_pruning,
                        factorization=prepared_yaqa_factorization,
                        telemetry=telemetry,
                        _incremental_cuda_feedback=incremental_cuda_feedback,
                        _incremental_cuda_factored_feedback=incremental_cuda_factored_feedback,
                        _incremental_cpu_factored_feedback=incremental_cpu_factored_feedback,
                        _trusted_inputs=True,
                    )
                    yaqa_bank_diagnostics["fallback_to_v2"] = True
                    yaqa_bank_diagnostics["block_family_id"] = 0
                    return (
                        canonical_weight,
                        canonical_states,
                        torch.zeros(
                            canonical_states.shape[0] * 8,
                            dtype=torch.uint8,
                            device=canonical_states.device,
                        ),
                        torch.ones((1,), dtype=torch.uint8, device=canonical_states.device),
                    )
                return yaqa_inner_v2b2_p32(
                    normalized_weight,
                    transformed_H,
                    transformed_output_hessian,
                    bank_codebooks,
                    block_input_hessian=block_ldlq_control_H,
                    bits=bits,
                    trellis_batch_size=trellis_batch_size,
                    tail_biting_candidates=tail_biting_candidates,
                    viterbi_pruning=viterbi_pruning,
                    family_mode=yaqa_v2b2_family_mode,
                    sample_strategy=yaqa_sample_strategy,
                    block_family_id=yaqa_v2b2_fixed_family_id,
                    diagnostics=yaqa_bank_diagnostics,
                    factorization=prepared_yaqa_factorization,
                    bank_codebook_pair_stacks=bank_codebook_pair_stacks,
                    telemetry=telemetry,
                    _incremental_cuda_factored_feedback=incremental_cuda_factored_feedback,
                    _incremental_cpu_factored_feedback=incremental_cpu_factored_feedback,
                    _trusted_inputs=True,
                )
            return yaqa_inner(
                normalized_weight,
                transformed_H,
                transformed_output_hessian,
                codebook,
                bits=bits,
                trellis_batch_size=trellis_batch_size,
                tail_biting_candidates=tail_biting_candidates,
                viterbi_pruning=viterbi_pruning,
                bank_codebooks=bank_codebooks,
                bank_codebook_stack=bank_codebook_stack,
                dual_v2=dual_v2,
                factorization=prepared_yaqa_factorization,
                telemetry=telemetry,
                _incremental_cuda_feedback=incremental_cuda_feedback,
                _incremental_cuda_factored_feedback=incremental_cuda_factored_feedback,
                _incremental_cpu_factored_feedback=incremental_cpu_factored_feedback,
                _trusted_inputs=True,
            )
        if v2b4_p64:
            assert bank_codebooks is not None
            return block_ldlq_inner_v2b4_p64(
                normalized_weight,
                transformed_H,
                bank_codebooks,
                bits=bits,
                tile_rows=16,
                tile_cols=16,
                trellis_batch_size=trellis_batch_size,
                viterbi_objective=objective,
                tail_biting_candidates=tail_biting_candidates,
                viterbi_pruning=viterbi_pruning,
                telemetry=telemetry,
                factorization=prepared_block_factors,
            )
        if v2b2_p32:
            assert bank_codebooks is not None
            return block_ldlq_inner_v2b2_p32(
                normalized_weight,
                transformed_H,
                bank_codebooks,
                bits=bits,
                tile_rows=16,
                tile_cols=16,
                trellis_batch_size=trellis_batch_size,
                viterbi_objective=objective,
                tail_biting_candidates=tail_biting_candidates,
                viterbi_pruning=viterbi_pruning,
                telemetry=telemetry,
                factorization=prepared_block_factors,
                bank_codebook_pair_stacks=bank_codebook_pair_stacks,
            )
        if v2b2_p32_lr:
            assert bank_codebooks is not None and prepared_block_factors is not None
            return block_ldlq_inner_v2b2_p32_lr(
                normalized_weight,
                transformed_H,
                bank_codebooks,
                bits=bits,
                trellis_batch_size=trellis_batch_size,
                viterbi_objective=objective,
                tail_biting_candidates=tail_biting_candidates,
                telemetry=telemetry,
                factorization=prepared_block_factors,
            )
        if bank_codebooks is not None:
            if propagated_inputs is not None and rounding == "block_ldlq":
                # Propagation evaluates the complete held-out objective.  A
                # mixed local-bank search here is redundant and, more
                # importantly, is not the canonical slot-zero history that
                # propagation must compare against.
                canonical_weight, canonical_states = block_ldlq_inner(
                    normalized_weight,
                    transformed_H,
                    bank_codebooks[0],
                    bits=bits,
                    tile_rows=16,
                    tile_cols=16,
                    trellis_batch_size=trellis_batch_size,
                    viterbi_objective=objective,
                    tail_biting_candidates=tail_biting_candidates,
                    telemetry=telemetry,
                    factorization=prepared_block_factors,
                )
                if include_bank0_oracle:
                    return (
                        canonical_weight,
                        canonical_states,
                        torch.zeros(
                            (canonical_states.shape[0],),
                            dtype=torch.uint8,
                            device=canonical_states.device,
                        ),
                        canonical_weight,
                        canonical_states,
                    )
                return (
                    canonical_weight,
                    canonical_states,
                    torch.zeros(
                        (canonical_states.shape[0],),
                        dtype=torch.uint8,
                        device=canonical_states.device,
                    ),
                )
            return block_ldlq_inner_banked(
                normalized_weight,
                transformed_H,
                bank_codebooks,
                bits=bits,
                bank_codebook_stack=bank_codebook_stack,
                tile_rows=16,
                tile_cols=16,
                trellis_batch_size=trellis_batch_size,
                viterbi_objective=objective,
                tail_biting_candidates=tail_biting_candidates,
                telemetry=telemetry,
                factorization=prepared_block_factors,
                return_bank0_oracle=include_bank0_oracle,
            )
        return block_ldlq_inner(
            normalized_weight,
            transformed_H,
            codebook,
            bits=bits,
            trellis_batch_size=trellis_batch_size,
            viterbi_objective=objective,
            tail_biting_candidates=tail_biting_candidates,
            telemetry=telemetry,
            factorization=prepared_block_factors,
            dual_v2=dual_v2,
        )

    with _qvq_phase(telemetry, "baseline_encode", device):
        baseline_encoded = encode_at_scale(scale, include_bank0_oracle=needs_bank0_oracle)
    localized_rollback_encoded = None
    if yaqa_spectral_refinement:
        assert v2b2_p32 and bank_codebooks is not None and transformed_output_hessian is not None
        with _qvq_phase(telemetry, "yaqa_output_spectral_refinement", device):
            baseline_encoded = yaqa_output_spectral_refine_v2b2_p32(
                transformed_weight / scale,
                transformed_H,
                transformed_output_hessian,
                bank_codebooks,
                baseline_encoded,
                ranks=yaqa_spectral_ranks,
                lambdas=yaqa_spectral_lambdas,
                family_mode=yaqa_v2b2_family_mode,
                block_input_hessian=block_ldlq_control_H,
                block_family_id=int(yaqa_bank_diagnostics["block_family_id"]),
                factorization=prepared_yaqa_factorization,
                diagnostics=yaqa_bank_diagnostics,
                bits=bits,
                trellis_batch_size=trellis_batch_size,
                tail_biting_candidates=tail_biting_candidates,
                bank_codebook_pair_stacks=bank_codebook_pair_stacks,
                telemetry=telemetry,
            )
    elif yaqa_spectral_push:
        assert v2b2_p32 and bank_codebooks is not None and transformed_output_hessian is not None
        with _qvq_phase(telemetry, "yaqa_spectral_push", device):
            baseline_encoded = yaqa_spectral_push_v2b2_p32(
                transformed_weight / scale,
                transformed_H,
                transformed_output_hessian,
                bank_codebooks,
                baseline_encoded,
                ranks=yaqa_spectral_ranks,
                alphas=yaqa_spectral_push_alphas,
                family_mode=yaqa_v2b2_family_mode,
                block_input_hessian=block_ldlq_control_H,
                block_family_id=int(yaqa_bank_diagnostics["block_family_id"]),
                factorization=prepared_yaqa_factorization,
                diagnostics=yaqa_bank_diagnostics,
                bits=bits,
                trellis_batch_size=trellis_batch_size,
                tail_biting_candidates=tail_biting_candidates,
                bank_codebook_pair_stacks=bank_codebook_pair_stacks,
                telemetry=telemetry,
            )
    elif yaqa_spectral_localized:
        assert v2b2_p32 and bank_codebooks is not None and transformed_output_hessian is not None
        localized_rollback_encoded = baseline_encoded
        localized_search_inputs = None
        localized_search_target = None
        if propagated_inputs is not None:
            heldout_target = propagated_target_output
            if bias is not None:
                heldout_target = heldout_target - bias.to(device=weight.device, dtype=heldout_target.dtype)
            localized_search_inputs = matmul_hadU(propagated_inputs * SU.to(torch.float32))
            localized_search_target = matmul_hadU(
                heldout_target / (SV_sign.to(torch.float32) * scale), transpose=True
            )

        def localized_serialized_weight(
            candidate_inner: torch.Tensor,
            candidate_states: torch.Tensor,
            candidate_bank_ids: torch.Tensor,
            candidate_alt_id: torch.Tensor,
        ) -> torch.Tensor:
            candidate_trellis = pack_trellis_states(
                candidate_states,
                bits=bits,
                vector_size=2,
                trellis_window=16,
            )
            serialized_inner = reconstruct_qvq_inner_weight(
                candidate_trellis,
                bits=bits,
                vector_size=2,
                trellis_window=16,
                in_features=in_features,
                out_features=out_features,
                codebook_version=codebook_version,
                bank_ids=candidate_bank_ids,
                v2b2_p32=True,
                bank_alt_id=candidate_alt_id,
            )
            if not torch.equal(serialized_inner.to(dtype=candidate_inner.dtype), candidate_inner):
                raise ValueError("QVQ localized replay candidate failed exact serialized reconstruction.")
            return rht_reconstruct_weight(
                serialized_inner.to(dtype=candidate_inner.dtype),
                SU,
                SV_sign * scale,
            )

        localized_replay_gradient = None
        localized_gradient_callback_error = False
        if propagated_candidate_gradient is not None:
            try:
                baseline_candidate_weight = localized_serialized_weight(*baseline_encoded)
                full_gradient = propagated_candidate_gradient(baseline_candidate_weight)
                if not isinstance(full_gradient, torch.Tensor):
                    raise TypeError("QVQ propagated candidate gradient callback must return a tensor.")
                if full_gradient.shape != baseline_candidate_weight.shape or not full_gradient.is_floating_point():
                    raise ValueError("QVQ propagated candidate gradient must match the dense weight geometry.")
                if full_gradient.device != device or not torch.isfinite(full_gradient).all():
                    raise ValueError("QVQ propagated candidate gradient must be finite and share the quantization device.")
                localized_replay_gradient = rht_reconstruct_weight_adjoint(
                    full_gradient.to(torch.float32),
                    SU,
                    SV_sign * scale,
                )
            except Exception:  # noqa: BLE001 - an external gradient callback must fail closed
                localized_gradient_callback_error = True

        localized_candidate_score = None
        if propagated_candidate_score is not None:
            def localized_candidate_score(
                candidate_inner: torch.Tensor,
                candidate_states: torch.Tensor,
                candidate_bank_ids: torch.Tensor,
                candidate_alt_id: torch.Tensor,
            ) -> float:
                candidate_weight = localized_serialized_weight(
                    candidate_inner,
                    candidate_states,
                    candidate_bank_ids,
                    candidate_alt_id,
                )
                score = float(propagated_candidate_score(candidate_weight))
                return score if math.isfinite(score) else math.inf
        with _qvq_phase(telemetry, "yaqa_localized_spectral_refinement", device):
            baseline_encoded = yaqa_localized_spectral_refine_v2b2_p32(
                transformed_weight / scale,
                transformed_H,
                transformed_output_hessian,
                bank_codebooks,
                baseline_encoded,
                ranks=yaqa_spectral_ranks,
                alphas=yaqa_spectral_localized_alphas,
                max_segments=yaqa_spectral_localized_max_segments,
                max_changes=yaqa_spectral_localized_max_changes,
                replay_candidates=yaqa_spectral_localized_replay_candidates,
                direct_replay_candidates=yaqa_spectral_localized_direct_replay_candidates,
                candidate_score=localized_candidate_score,
                replay_gradient=localized_replay_gradient,
                replay_gradient_error=localized_gradient_callback_error,
                search_inputs=localized_search_inputs,
                search_target=localized_search_target,
                diagnostics=yaqa_bank_diagnostics,
                bits=bits,
            )
    baseline_bank_alt_id = None
    if bank_codebooks is None:
        baseline_inner, baseline_states = baseline_encoded
        baseline_bank_ids = None
    elif v2b2_p32 or v2b2_p32_lr:
        baseline_inner, baseline_states, baseline_bank_ids, baseline_bank_alt_id = baseline_encoded
    elif needs_bank0_oracle:
        baseline_inner, baseline_states, baseline_bank_ids, bank0_inner, bank0_states = baseline_encoded
    else:
        baseline_inner, baseline_states, baseline_bank_ids = baseline_encoded
    module_SV = SV_sign * scale
    source_H = H.to(device=device)
    scale_optimization_H = None
    if module_scale_search or output_channel_scale_optimization:
        # The same isotropic damping used by BlockLDLQ is invariant under the
        # orthonormal Hadamard transform. It regularizes rank-poor calibration
        # Hessians without weakening acceptance under the original objective.
        scale_optimization_H = source_H.to(torch.float32).clone()
        scale_optimization_H.diagonal().add_(damping)

    def finish_candidate(
        candidate_inner: torch.Tensor,
        candidate_SV: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        with _qvq_phase(telemetry, "candidate_reconstruct_proxy", device):
            candidate_weight = rht_reconstruct_weight(candidate_inner, SU, candidate_SV)
            unoptimized_loss = _qvq_proxy_loss_unchecked(weight, candidate_weight, source_H)
            candidate_loss = unoptimized_loss
            optimized_channels = 0
            if output_channel_scale_optimization:
                candidate_SV, candidate_weight, candidate_loss, optimized_channels = (
                    optimize_qvq_output_channel_scales(
                        weight,
                        candidate_inner,
                        source_H,
                        SU,
                        candidate_SV,
                        optimization_H=scale_optimization_H,
                        correction_strength=_QVQ_LOW_RATE_OUTPUT_SCALE_STRENGTH if bits <= 1.5 else 1.0,
                    )
                )
        return (
            candidate_SV,
            candidate_weight,
            candidate_loss,
            optimized_channels,
            unoptimized_loss,
        )

    quantized_inner = baseline_inner
    selected_bank_ids = baseline_bank_ids
    selected_bank_alt_id = baseline_bank_alt_id
    states = baseline_states
    selected_encoding_scale = scale
    SV, reconstructed_weight, proxy_loss, optimized_channels, baseline_proxy_loss = finish_candidate(
        baseline_inner,
        module_SV,
    )
    module_scale_search_selected = False
    module_scale_multiplier = 1.0
    module_scale_reencoded = False
    if module_scale_search:
        proposed_SV, _, _, proposed_multiplier, proposed = optimize_qvq_module_scale(
            weight,
            baseline_inner,
            source_H,
            SU,
            module_SV,
            optimization_H=scale_optimization_H,
        )
        if proposed:
            fixed_SV, fixed_weight, fixed_loss, fixed_channels, _ = finish_candidate(
                baseline_inner,
                proposed_SV,
            )
            if torch.isfinite(fixed_loss) and fixed_loss < proxy_loss:
                SV = fixed_SV
                reconstructed_weight = fixed_weight
                proxy_loss = fixed_loss
                optimized_channels = fixed_channels
                selected_encoding_scale = scale * proposed_multiplier
                module_scale_search_selected = True
                module_scale_multiplier = proposed_multiplier

            candidate_scale = scale * proposed_multiplier
            reencoded_encoded = encode_at_scale(candidate_scale, include_bank0_oracle=needs_bank0_oracle)
            if bank_codebooks is None:
                reencoded_inner, reencoded_states = reencoded_encoded
                reencoded_bank_ids = None
            elif needs_bank0_oracle:
                reencoded_inner, reencoded_states, reencoded_bank_ids, reencoded_bank0_inner, reencoded_bank0_states = (
                    reencoded_encoded
                )
            else:
                reencoded_inner, reencoded_states, reencoded_bank_ids = reencoded_encoded
            reencoded_SV, reencoded_weight, reencoded_loss, reencoded_channels, _ = finish_candidate(
                reencoded_inner,
                SV_sign * candidate_scale,
            )
            if torch.isfinite(reencoded_loss) and reencoded_loss < proxy_loss:
                quantized_inner = reencoded_inner
                states = reencoded_states
                selected_bank_ids = reencoded_bank_ids if bank_codebooks is not None else None
                SV = reencoded_SV
                reconstructed_weight = reencoded_weight
                proxy_loss = reencoded_loss
                optimized_channels = reencoded_channels
                selected_encoding_scale = candidate_scale
                module_scale_search_selected = True
                module_scale_multiplier = proposed_multiplier
                module_scale_reencoded = True
                if needs_bank0_oracle:
                    bank0_inner = reencoded_bank0_inner
                    bank0_states = reencoded_bank0_states
    hessian_viterbi_selected = False
    hessian_viterbi_candidate_relative_improvement = None
    if rounding == "block_ldlq" and viterbi_objective == "hessian_diagonal":
        hessian_encoded = encode_at_scale(
            selected_encoding_scale,
            objective="hessian_diagonal",
            include_bank0_oracle=needs_bank0_oracle,
        )
        hessian_bank_alt_id = None
        if bank_codebooks is None:
            hessian_inner, hessian_states = hessian_encoded
            hessian_bank_ids = None
        elif v2b2_p32 or v2b2_p32_lr:
            hessian_inner, hessian_states, hessian_bank_ids, hessian_bank_alt_id = hessian_encoded
        elif needs_bank0_oracle:
            hessian_inner, hessian_states, hessian_bank_ids, hessian_bank0_inner, hessian_bank0_states = hessian_encoded
        else:
            hessian_inner, hessian_states, hessian_bank_ids = hessian_encoded
        hessian_SV, hessian_weight, hessian_loss, hessian_optimized_channels, _ = finish_candidate(
            hessian_inner,
            SV_sign * selected_encoding_scale,
        )
        proxy_denominator = proxy_loss.abs().clamp_min(torch.finfo(torch.float32).eps)
        relative_improvement = (proxy_loss - hessian_loss) / proxy_denominator
        hessian_viterbi_candidate_relative_improvement = float(relative_improvement.item())
        if (
            torch.isfinite(hessian_loss)
            and hessian_loss < proxy_loss
            and relative_improvement >= viterbi_minimum_proxy_improvement
        ):
            quantized_inner = hessian_inner
            states = hessian_states
            selected_bank_ids = hessian_bank_ids if bank_codebooks is not None else None
            if v2b2_p32 or v2b2_p32_lr:
                selected_bank_alt_id = hessian_bank_alt_id
            SV = hessian_SV
            reconstructed_weight = hessian_weight
            proxy_loss = hessian_loss
            optimized_channels = hessian_optimized_channels
            hessian_viterbi_selected = True
            if needs_bank0_oracle:
                bank0_inner = hessian_bank0_inner
                bank0_states = hessian_bank0_states

    if propagated_inputs is not None and yaqa_spectral_localized:
        assert localized_rollback_encoded is not None and v2b2_p32
        rollback_inner, rollback_states, rollback_bank_ids, rollback_alt_id = localized_rollback_encoded
        rollback_SV, rollback_weight, rollback_proxy_loss, rollback_optimized_channels, _ = finish_candidate(
            rollback_inner,
            SV_sign * selected_encoding_scale,
        )
        preprop_inner = rollback_inner
        preprop_states = rollback_states
        preprop_bank_ids = rollback_bank_ids
        preprop_weight = rollback_weight
        proposal_changed = (
            not torch.equal(states, rollback_states)
            or not torch.equal(selected_bank_ids, rollback_bank_ids)
            or selected_bank_alt_id != rollback_alt_id
        )
        accepted = bool(torch.isfinite(reconstructed_weight).all()) and proposal_changed
        if accepted:
            proposal_trellis = pack_trellis_states(
                states,
                bits=bits,
                vector_size=vector_size,
                trellis_window=trellis_window,
            )
            serialized_inner = reconstruct_qvq_inner_weight(
                proposal_trellis,
                bits=bits,
                vector_size=2,
                trellis_window=16,
                in_features=in_features,
                out_features=out_features,
                codebook_version=codebook_version,
                bank_ids=selected_bank_ids,
                v2b2_p32=True,
                bank_alt_id=selected_bank_alt_id,
            )
            accepted = torch.equal(serialized_inner.to(dtype=quantized_inner.dtype), quantized_inner)
        if accepted:
            quantized_inner = serialized_inner.to(dtype=quantized_inner.dtype)
            reconstructed_weight = rht_reconstruct_weight(quantized_inner, SU, SV_sign * selected_encoding_scale)
            try:
                accepted = bool(propagated_acceptance(reconstructed_weight, rollback_weight))
            except Exception:  # noqa: BLE001 - an external confirmation gate must fail closed
                accepted = False
                if telemetry is not None:
                    telemetry.count("localized_propagation_callback_error")
        if not accepted:
            quantized_inner = rollback_inner
            states = rollback_states
            selected_bank_ids = rollback_bank_ids
            selected_bank_alt_id = rollback_alt_id
            SV = rollback_SV
            reconstructed_weight = rollback_weight
            proxy_loss = rollback_proxy_loss
            optimized_channels = rollback_optimized_channels
            yaqa_bank_diagnostics["spectral_selected"] = False
            yaqa_bank_diagnostics["spectral_selected_loss"] = yaqa_bank_diagnostics["spectral_original_loss"]
        yaqa_bank_diagnostics["localized_propagation_proposed"] = proposal_changed
        yaqa_bank_diagnostics["localized_propagation_accepted"] = accepted
        if telemetry is not None:
            telemetry.count("localized_propagation_proposal_changed", int(proposal_changed))
            telemetry.count("localized_propagation_accepted", int(accepted))
    elif propagated_inputs is not None:
        # Keep the accepted pre-propagation artifact separate from the
        # canonical bank-0 candidate.  The callback and fail-closed rollback
        # must use the former, even when the candidate search starts from the
        # latter.
        preprop_inner = quantized_inner
        preprop_states = states
        preprop_bank_ids = None if selected_bank_ids is None else selected_bank_ids.clone()
        preprop_weight = reconstructed_weight
        candidate_weights, candidate_states = block_ldlq_inner_banked_candidates(
            (transformed_weight / selected_encoding_scale),
            transformed_H,
            bank_codebooks,
            bank_codebook_stack=bank_codebook_stack,
            bits=bits,
            tile_rows=16,
            tile_cols=16,
            trellis_batch_size=trellis_batch_size,
            viterbi_objective=viterbi_objective,
            tail_biting_candidates=tail_biting_candidates,
            telemetry=telemetry,
            factorization=prepared_block_factors,
            _trusted_inputs=True,
            # The propagation candidate slot zero is the independent
            # canonical bank-0 history, never the locally mixed baseline.
            baseline_weight=bank0_inner if needs_bank0_oracle else None,
            baseline_states=bank0_states if needs_bank0_oracle else None,
        )
        heldout_target = propagated_target_output
        if bias is not None:
            heldout_target = heldout_target - bias.to(device=weight.device, dtype=heldout_target.dtype)
        # Score in the same inner space used by QVQ inference. Hadamard is
        # orthogonal, so this is exactly equivalent to output-space MSE while
        # avoiding a dense reconstruction for every bank/tile proposal.
        transformed_inputs = matmul_hadU(propagated_inputs * SU.to(torch.float32))
        transformed_target = matmul_hadU(
            heldout_target / (SV_sign.to(torch.float32) * selected_encoding_scale), transpose=True
        )
        propagation_baseline_inner = preprop_inner
        propagation_baseline_states = preprop_states
        baseline_output = transformed_inputs @ propagation_baseline_inner
        baseline_output_error = transformed_target - baseline_output
        baseline_weight = preprop_weight
        baseline_loss = baseline_output_error.to(torch.float32).square().sum()
        baseline_dense_loss = (
            heldout_target - propagated_inputs @ baseline_weight.to(torch.float32).transpose(0, 1)
        ).square().sum()
        working_inner = propagation_baseline_inner.clone()
        output_tiles = out_features // 16
        tile_count = (in_features // 16) * output_tiles
        baseline_bank_ids = (
            torch.zeros(tile_count, dtype=torch.uint8, device=weight.device)
            if preprop_bank_ids is None
            else preprop_bank_ids.to(device=weight.device, dtype=torch.uint8).clone()
        )
        baseline_states = propagation_baseline_states.reshape(tile_count, -1)
        proposed_bank_ids = baseline_bank_ids.clone()
        proposed_states = baseline_states.clone()
        candidate_states_view = candidate_states.reshape(4, tile_count, -1)
        # Input blocks are sequential because their changes affect the shared
        # residual. Output blocks are disjoint and are evaluated together.
        # Keep held-out work bounded for wide projections: the score is additive
        # over rows, so chunking rows does not change the selected bank.
        propagation_row_chunk = 256
        sample_count = transformed_inputs.shape[0]
        candidate_delta_bytes = 4 * sample_count * output_tiles * 16 * 4  # four banks, FP32 deltas
        cache_candidate_deltas = candidate_delta_bytes <= _QVQ_PROPAGATION_DELTA_CACHE_BYTES
        output_indices = torch.arange(output_tiles, device=weight.device)
        for input_block in range(in_features // 16):
            row = input_block * 16
            rows = slice(row, row + 16)
            current_tiles = working_inner[rows].reshape(output_tiles, 16, 16)
            bank_tiles = candidate_weights[:, rows].reshape(4, output_tiles, 16, 16)
            delta_tiles = bank_tiles - current_tiles.unsqueeze(0)
            flat_delta_tiles = delta_tiles.reshape(4 * output_tiles, 16, 16)

            def project_candidate_deltas(sample_inputs: torch.Tensor) -> torch.Tensor:
                sample_count_chunk = sample_inputs.shape[0]
                expanded_inputs = sample_inputs.unsqueeze(0).unsqueeze(2).expand(
                    4, sample_count_chunk, output_tiles, 16
                )
                projected = torch.bmm(
                    expanded_inputs.permute(0, 2, 1, 3).reshape(4 * output_tiles, sample_count_chunk, 16),
                    flat_delta_tiles,
                )
                return projected.reshape(4, output_tiles, sample_count_chunk, 16).permute(0, 2, 1, 3)

            losses = torch.zeros((4, output_tiles), device=weight.device, dtype=torch.float32)
            # Retain candidate deltas only when the complete cache fits the
            # budget. Large projections use the exact recomputation fallback
            # below, avoiding an unbounded O(batch * output) allocation.
            candidate_deltas = [] if cache_candidate_deltas else None
            for start in range(0, sample_count, propagation_row_chunk):
                stop = min(start + propagation_row_chunk, sample_count)
                delta = project_candidate_deltas(transformed_inputs[start:stop, rows])
                if candidate_deltas is not None:
                    candidate_deltas.append(delta)
                residual_blocks = baseline_output_error[start:stop].reshape(-1, output_tiles, 16)
                # Score every candidate against the accepted pre-propagation
                # residual.  Candidate zero is canonical bank 0, not an
                # implicit no-op when the accepted baseline is mixed-bank.
                losses[:4] += (
                    residual_blocks.unsqueeze(0) - delta
                ).square().sum(dim=(1, 3))
            winners = losses.argmin(dim=0)
            tile_indices = input_block * output_tiles + output_indices
            bank_winners = winners.clamp_max(3)
            chosen_tiles = bank_tiles[bank_winners, output_indices]
            current_tiles = chosen_tiles
            working_inner[rows] = current_tiles.permute(1, 0, 2).reshape(16, -1)
            proposed_bank_ids[tile_indices] = bank_winners.to(torch.uint8)
            chosen_states = candidate_states_view[bank_winners, tile_indices]
            proposed_states[tile_indices] = chosen_states
            for chunk_index, start in enumerate(range(0, sample_count, propagation_row_chunk)):
                stop = min(start + propagation_row_chunk, sample_count)
                delta = (
                    candidate_deltas[chunk_index]
                    if candidate_deltas is not None
                    else project_candidate_deltas(transformed_inputs[start:stop, rows])
                )
                selected_delta = delta.permute(1, 2, 0, 3).gather(
                    2,
                    bank_winners.view(1, output_tiles, 1, 1).expand(
                        delta.shape[1], output_tiles, 1, delta.shape[3]
                    ),
                ).squeeze(2)
                residual_blocks = baseline_output_error[start:stop].reshape(-1, output_tiles, 16)
                baseline_output_error[start:stop] = (residual_blocks - selected_delta).reshape(
                    stop - start, out_features
                )
        selected_loss = baseline_output_error.square().sum()
        accepted = torch.isfinite(selected_loss) and selected_loss < baseline_loss
        if selected_bank_ids is not None:
            # The staged trellis/selectors are the serialization authority.
            # Re-decode once before the acceptance callback so the proposed
            # dense weight cannot diverge from what reload will reconstruct.
            staged_inner = working_inner
            decoded_tiles = pgc16_decode_states_v4_banked(
                proposed_states,
                proposed_bank_ids,
                bits=bits,
                levels=pgc16_levels_for_version(codebook_version).to(device=states.device),
            )
            decoded_inner = (
                decoded_tiles.reshape(in_features // 16, output_tiles, 16, 16)
                .permute(0, 2, 1, 3)
                .reshape(in_features, out_features)
                .contiguous()
            )
            if not torch.equal(decoded_inner.to(dtype=staged_inner.dtype), staged_inner):
                # Keep the serialized representation authoritative below; the
                # post-pack gate will reject any remaining packing mismatch.
                if telemetry is not None:
                    telemetry.count("propagation_staged_state_mismatch")
                # Keep the quantizer's inner-weight dtype stable.  The decoder
                # follows the serialized tensor dtype (often FP16/BF16), while
                # transformed_inputs is normally FP32 for the propagation
                # objective.  Normalizing here avoids a mixed-dtype matmul in
                # the authoritative loss recomputation and preserves the
                # dtype contract of the returned quantized tensors.
                working_inner = decoded_inner.to(dtype=staged_inner.dtype)
            # The authoritative decoded proposal may differ from the values
            # used during tile scoring. Recompute its exact held-out loss before
            # accepting it; never reuse a stale improvement from the discarded
            # staged matrix.
            selected_loss = (
                transformed_target - transformed_inputs @ working_inner.to(dtype=transformed_inputs.dtype)
            ).square().sum()
            accepted = torch.isfinite(selected_loss) and selected_loss < baseline_loss
        proposed_weight = rht_reconstruct_weight(working_inner, SU, SV_sign * selected_encoding_scale)
        proposed_dense_loss = (
            heldout_target - propagated_inputs @ proposed_weight.to(torch.float32).transpose(0, 1)
        ).square().sum()
        # The module-output loss is diagnostic/screening information only.  A
        # locally worse direction can still improve the downstream model by
        # cancelling an error introduced by later quantized modules.  The
        # required acceptance callback is the authority for this second-stage
        # decision and receives the exact serialized proposal below.
        local_dense_improved = bool(
            torch.isfinite(proposed_dense_loss) and proposed_dense_loss < baseline_dense_loss
        )
        if telemetry is not None and not local_dense_improved:
            telemetry.count("propagation_local_loss_not_improved")
        accepted = bool(torch.isfinite(proposed_dense_loss))
        proposal_changed = True
        if selected_bank_ids is not None:
            # A no-op proposal cannot improve the serialized artifact.  Avoid
            # invoking an expensive downstream/full-model callback for it.
            proposal_changed = not torch.equal(proposed_states, states) or not torch.equal(
                proposed_bank_ids, selected_bank_ids
            )
        if accepted and proposal_changed and propagated_acceptance is not None:
            accepted = bool(propagated_acceptance(proposed_weight, preprop_weight))
        if accepted:
            quantized_inner = working_inner
            states = proposed_states
            selected_bank_ids = proposed_bank_ids
            reconstructed_weight = proposed_weight
            proxy_loss = _qvq_proxy_loss_unchecked(weight, reconstructed_weight, source_H)

    with _qvq_phase(telemetry, "pack_trellis", device):
        # States are already on the quantization device. Planar packing is
        # expressed entirely with device-native tensor operations and is
        # bit-exact with the CPU implementation. Keeping it local avoids a
        # full state-stream D2H copy, CPU pack, and packed-word H2D copy.
        if v2b2_p32_lr:
            trellis = pack_local_ring_states(states, bits=bits)
        elif dual_v2:
            trellis = pack_dual_v2_states(states, bits=bits)
        else:
            trellis = pack_trellis_states(
                states,
                bits=bits,
                vector_size=vector_size,
                trellis_window=trellis_window,
            )
    if propagated_inputs is not None and selected_bank_ids is not None:
        roundtrip_inner = reconstruct_qvq_inner_weight(
            trellis,
            bits=bits,
            vector_size=vector_size,
            trellis_window=trellis_window,
            in_features=in_features,
            out_features=out_features,
            codebook_version=codebook_version,
            bank_ids=selected_bank_ids,
            v2b2_p32=v2b2_p32,
            bank_alt_id=selected_bank_alt_id,
        )
        if not torch.equal(roundtrip_inner.to(dtype=quantized_inner.dtype), quantized_inner):
            # Never emit a proposal whose serialized representation differs
            # from the selected dense inner weight. Fall back atomically to the
            # exact pre-propagation state until the staging mismatch is fixed.
            quantized_inner = preprop_inner
            states = preprop_states
            selected_bank_ids = preprop_bank_ids
            reconstructed_weight = preprop_weight
            proxy_loss = _qvq_proxy_loss_unchecked(weight, reconstructed_weight, source_H)
            trellis = pack_trellis_states(
                states,
                bits=bits,
                vector_size=vector_size,
                trellis_window=trellis_window,
            )
    if v2b4_p64:
        if selected_bank_ids is None:
            raise RuntimeError("QVQ V2B4-P64 quantization did not produce segment selectors.")
        roundtrip_inner = reconstruct_qvq_inner_weight(
            trellis,
            bits=bits,
            vector_size=2,
            trellis_window=16,
            in_features=in_features,
            out_features=out_features,
            codebook_version=codebook_version,
            bank_ids=selected_bank_ids,
            v2b4_p64=True,
        )
        if not torch.equal(roundtrip_inner.to(dtype=quantized_inner.dtype), quantized_inner):
            raise RuntimeError("QVQ V2B4-P64 packed trellis/selectors do not reproduce the selected inner weight.")
        if telemetry is not None:
            telemetry.count("packed_roundtrip_verifications")
    if v2b2_p32:
        if selected_bank_ids is None or selected_bank_alt_id is None:
            raise RuntimeError("QVQ V2B2-P32 quantization did not produce selectors and an alternative bank ID.")
        roundtrip_inner = reconstruct_qvq_inner_weight(
            trellis,
            bits=bits,
            vector_size=2,
            trellis_window=16,
            in_features=in_features,
            out_features=out_features,
            codebook_version=codebook_version,
            bank_ids=selected_bank_ids,
            v2b2_p32=True,
            bank_alt_id=selected_bank_alt_id,
        )
        if not torch.equal(roundtrip_inner.to(dtype=quantized_inner.dtype), quantized_inner):
            raise RuntimeError("QVQ V2B2-P32 packed trellis/selectors do not reproduce the selected inner weight.")
        if telemetry is not None:
            telemetry.count("packed_roundtrip_verifications")
    if v2b2_p32_lr:
        if selected_bank_ids is None or selected_bank_alt_id is None:
            raise RuntimeError("QVQ V2B2-P32-LR quantization did not produce selectors and an alternative bank ID.")
        roundtrip_inner = reconstruct_local_ring_inner_weight(
            trellis,
            bits=bits,
            in_features=in_features,
            out_features=out_features,
            codebook_version=codebook_version,
            bank_ids=selected_bank_ids,
            bank_alt_id=selected_bank_alt_id,
        )
        if not torch.equal(roundtrip_inner.to(dtype=quantized_inner.dtype), quantized_inner):
            raise RuntimeError("QVQ V2B2-P32-LR packed trellis/selectors do not reproduce the selected inner weight.")
        if telemetry is not None:
            telemetry.count("packed_roundtrip_verifications")
    kronecker_proxy_loss = None
    if output_hessian is not None:
        kronecker_proxy_loss = yaqa_proxy_loss(
            weight,
            reconstructed_weight,
            source_H,
            output_hessian.to(device=device),
        )

    telemetry_result = None if telemetry is None else telemetry.finalize()
    return QVQLinearQuantizationResult(
        trellis=trellis,
        SU=SU,
        SV=SV,
        bias=None if bias is None else bias.detach().to(device=device).clone(),
        inner_weight=quantized_inner,
        weight=reconstructed_weight.to(dtype=weight.dtype),
        proxy_loss=proxy_loss,
        baseline_proxy_loss=baseline_proxy_loss,
        output_scale_optimized_channels=optimized_channels,
        hessian_viterbi_selected=hessian_viterbi_selected,
        hessian_viterbi_candidate_relative_improvement=hessian_viterbi_candidate_relative_improvement,
        rounding=rounding,
        kronecker_proxy_loss=kronecker_proxy_loss,
        module_scale_search_selected=module_scale_search_selected,
        module_scale_multiplier=module_scale_multiplier,
        module_scale_reencoded=module_scale_reencoded,
        telemetry=telemetry_result,
        serialization_allowed=experimental_codebook is None,
        bank_ids=None if selected_bank_ids is None else selected_bank_ids.detach().clone(),
        bank_selector_bits=1 if v2b2_p32 or v2b2_p32_lr else 2,
        bank_alt_id=None if selected_bank_alt_id is None else selected_bank_alt_id.detach().clone(),
        yaqa_bank_fallback_to_v2=(
            None if "fallback_to_v2" not in yaqa_bank_diagnostics else bool(yaqa_bank_diagnostics["fallback_to_v2"])
        ),
        yaqa_selector_churn=(
            None if "selector_churn" not in yaqa_bank_diagnostics else float(yaqa_bank_diagnostics["selector_churn"])
        ),
        yaqa_family_changed=(
            None if "family_changed" not in yaqa_bank_diagnostics else bool(yaqa_bank_diagnostics["family_changed"])
        ),
        yaqa_block_family_id=(
            None if "block_family_id" not in yaqa_bank_diagnostics else int(yaqa_bank_diagnostics["block_family_id"])
        ),
        yaqa_spectral_selected=(
            None if "spectral_selected" not in yaqa_bank_diagnostics
            else bool(yaqa_bank_diagnostics["spectral_selected"])
        ),
        yaqa_spectral_method=(
            None if yaqa_bank_diagnostics.get("spectral_method") is None
            else str(yaqa_bank_diagnostics["spectral_method"])
        ),
        yaqa_spectral_rank=(
            None if yaqa_bank_diagnostics.get("spectral_rank") is None
            else int(yaqa_bank_diagnostics["spectral_rank"])
        ),
        yaqa_spectral_lambda=(
            None if yaqa_bank_diagnostics.get("spectral_lambda") is None
            else float(yaqa_bank_diagnostics["spectral_lambda"])
        ),
        yaqa_spectral_alpha=(
            None if yaqa_bank_diagnostics.get("spectral_alpha") is None
            else float(yaqa_bank_diagnostics["spectral_alpha"])
        ),
        yaqa_spectral_svd_device=(
            None if "spectral_svd_device" not in yaqa_bank_diagnostics
            else str(yaqa_bank_diagnostics["spectral_svd_device"])
        ),
        yaqa_spectral_concentration=(
            None if "spectral_concentration" not in yaqa_bank_diagnostics
            else {
                str(rank): float(value)
                for rank, value in dict(yaqa_bank_diagnostics["spectral_concentration"]).items()
            }
        ),
        yaqa_spectral_oracle_losses=(
            None if "spectral_oracle_losses" not in yaqa_bank_diagnostics
            else {
                str(rank): float(value)
                for rank, value in dict(yaqa_bank_diagnostics["spectral_oracle_losses"]).items()
            }
        ),
        yaqa_spectral_candidates=(
            None if "spectral_candidates" not in yaqa_bank_diagnostics
            else {
                str(candidate): dict(values)
                for candidate, values in dict(yaqa_bank_diagnostics["spectral_candidates"]).items()
            }
        ),
        yaqa_spectral_absorption_efficiency=(
            None if "spectral_absorption_efficiency" not in yaqa_bank_diagnostics
            else float(yaqa_bank_diagnostics["spectral_absorption_efficiency"])
        ),
        yaqa_spectral_selector_churn=(
            None if "spectral_selector_churn" not in yaqa_bank_diagnostics
            else float(yaqa_bank_diagnostics["spectral_selector_churn"])
        ),
        yaqa_spectral_family_changed=(
            None if "spectral_family_changed" not in yaqa_bank_diagnostics
            else bool(yaqa_bank_diagnostics["spectral_family_changed"])
        ),
    )


def _validate_trellis_shape(*, bits: float, vector_size: int, trellis_window: int) -> int:
    if isinstance(vector_size, bool) or not isinstance(vector_size, int) or vector_size < 1:
        raise ValueError("QVQ `vector_size` must be a positive integer.")
    if isinstance(trellis_window, bool) or not isinstance(trellis_window, int) or trellis_window < 1:
        raise ValueError("QVQ `trellis_window` must be a positive integer.")
    shift = qvq_transition_bits(bits, vector_size=vector_size)
    if shift > trellis_window:
        raise ValueError("QVQ requires `rate * vector_size <= trellis_window`.")
    return shift
