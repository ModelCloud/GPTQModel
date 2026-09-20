# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Exact continuous-window P32 WMMA kernel for A100-class sm_80 GPUs."""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import torch

from ..quantization.qvq_rates import qvq_transition_bits
from .cpp import (
    TorchOpsJitExtension,
    default_jit_cflags,
    default_jit_cuda_cflags,
    default_torch_ops_build_root,
    is_nvcc_compatible,
)

_QVQ_AMPERE_NAME = "gptqmodel_qvq_ampere_ops"
_QVQ_AMPERE_NAMESPACE = "gptqmodel_qvq_ampere"
_SM80_FLAGS = (
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_80,code=compute_80",
    # Flash-Next's 640-wide expert projections cannot use the Hopper
    # TMA/WGMMA kernel's N/K=256 contract.  Compile the same exact WMMA
    # implementation natively for the measured narrow SM90 fallback instead
    # of relying on driver JIT of compute_80 PTX.
    "-gencode=arch=compute_90,code=sm_90",
)
_TORCH_NVCC_UNDEFINES = (
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
)
_SM_COUNT_CACHE: dict[tuple[str, int], int] = {}
_AutotuneCacheKey = tuple[int, torch.dtype, torch.Size, int, int, int]
_AUTOTUNE_CACHE: dict[_AutotuneCacheKey, int] = {}
_AUTOTUNE_CACHE_LOCK = threading.RLock()
_P32_TRANSITION_BITS = {2: 4, 2.5: 5, 3: 6, 3.5: 7}
_P32_WINDOW_OP: object | None = None
_P32_RANK8_PROJECT_OP: object | None = None
_P32_WINDOW_GROUPED_OP: object | None = None
_P32_WINDOW_GROUPED_FUSED_OP: object | None = None


@dataclass(frozen=True)
class QVQAmpereP32SegmentPlan:
    """The independently resolved launch plan for one grouped P32 child."""

    output_tile_start: int
    output_tile_count: int
    out_features: int
    bank_alt_id: int
    split_count: int


@dataclass(frozen=True)
class QVQAmpereGroupedP32Plan:
    """A segmented SM80 plan that never derives policy from synthetic total N."""

    in_features: int
    transition_bits: int
    segments: tuple[QVQAmpereP32SegmentPlan, ...]

    @property
    def out_features(self) -> int:
        return sum(segment.out_features for segment in self.segments)


@dataclass(frozen=True)
class QVQAmpereGroupedP32Payload:
    """Cached K-major grouped continuous-window payload for one launch plan."""

    trellis: torch.Tensor
    bank_ids: torch.Tensor
    plan: QVQAmpereGroupedP32Plan
    child_trellises: tuple[torch.Tensor, ...]
    child_bank_ids: tuple[torch.Tensor, ...]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source() -> list[str]:
    return [str(_project_root() / "gptqmodel_ext" / "qvq" / "qvq_ampere_cuda.cu")]


def _cuda_flags() -> list[str]:
    flags = [
        *_TORCH_NVCC_UNDEFINES,
        *default_jit_cuda_cflags(
            enable_bf16=False,
            include_lineinfo=True,
            include_nvcc_threads=True,
            nvcc_threads="2",
            include_split_compile=True,
            include_ptxas_optimizations=True,
            include_ptxas_verbosity=False,
            include_fatbin_compression=True,
            include_diag_suppress=True,
        ),
        *_SM80_FLAGS,
    ]
    if is_nvcc_compatible():
        flags.insert(0, "-static-global-template-stub=false")
    return flags


_QVQ_AMPERE_EXTENSION = TorchOpsJitExtension(
    name=_QVQ_AMPERE_NAME,
    namespace=_QVQ_AMPERE_NAMESPACE,
    required_ops=(
        "p32_window",
        "rank8_project",
        "p32_window_grouped",
        "p32_window_grouped_fused",
    ),
    sources=_source,
    build_root_env="GPTQMODEL_QVQ_AMPERE_BUILD_ROOT",
    default_build_root=lambda: default_torch_ops_build_root("qvq_ampere"),
    display_name="QVQ exact-P32 Ampere WMMA kernel",
    extra_cflags=lambda: default_jit_cflags(enable_bf16=False),
    extra_cuda_cflags=_cuda_flags,
    force_rebuild_env="GPTQMODEL_QVQ_AMPERE_FORCE_REBUILD",
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=True,
    merge_visible_cuda_arch_override=False,
)


def _auto_split_count(
    *, in_features: int, out_features: int, k_tiles: int, sm_count: int
) -> int:
    """Use measured Qwen3.8 splits, then a live-SM-derived fallback."""

    tuned_split = {
        (5120, 1024): 8,
        (5120, 6144): 8,
        (5120, 10240): 6,
        (5120, 12288): 5,
        (5120, 17408): 8,
        (6144, 5120): 8,
        (17408, 5120): 8,
    }.get((in_features, out_features))
    if tuned_split is not None:
        return min(tuned_split, k_tiles)
    n64_blocks = (out_features + 63) // 64
    target_blocks = max(1, sm_count * 2)
    split_count = max(1, (target_blocks + n64_blocks - 1) // n64_blocks)
    return min(split_count, 8, k_tiles)


def _device_sm_count(device: torch.device) -> int:
    """Cache the immutable SM count used by the live-device fallback."""

    key = (device.type, -1 if device.index is None else int(device.index))
    sm_count = _SM_COUNT_CACHE.get(key)
    if sm_count is None:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "QVQ Ampere SM-count cache must be prepared before CUDA Graph capture"
            )
        sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
        _SM_COUNT_CACHE[key] = sm_count
    return sm_count


def _autotune_enabled() -> bool:
    """Return whether first-use launch-plan autotuning is enabled."""

    return os.environ.get("QVQ_AMPERE_AUTOTUNE", "1").lower() not in {
        "0",
        "false",
        "off",
        "no",
        "",
    }


_AUTOTUNE_ENABLED = _autotune_enabled()


def _autotune_cache_key(
    input: torch.Tensor,
    *,
    transition_bits: int,
    out_features: int,
    bank_alt_id: int,
) -> _AutotuneCacheKey:
    # Plans never leave this process, and CUDA device indices remain stable for
    # its lifetime. Use the raw integer index rather than hashing a
    # ``torch.device`` or querying hardware properties on every hot cache hit.
    return (
        input.get_device(),
        input.dtype,
        input.shape,
        out_features,
        transition_bits,
        bank_alt_id,
    )


def _p32_window_op() -> object:
    """Resolve the registered operator once for hot cached launches."""

    global _P32_WINDOW_OP
    op = _P32_WINDOW_OP
    if op is None:
        op = _QVQ_AMPERE_EXTENSION.op("p32_window")
        _P32_WINDOW_OP = op
    return op


def _p32_rank8_project_op() -> object:
    """Resolve the batched rank-8 producer once for grouped Q/K/V calls."""

    global _P32_RANK8_PROJECT_OP
    op = _P32_RANK8_PROJECT_OP
    if op is None:
        op = _QVQ_AMPERE_EXTENSION.op("rank8_project")
        _P32_RANK8_PROJECT_OP = op
    return op


def prewarm_qvq_ampere() -> None:
    """Load the SM80 window operator before CUDA Graph capture.

    This performs only registration/JIT preparation; candidate timing and
    launch-plan selection remain separate preparation-time operations.
    """
    _p32_window_op()
    _p32_rank8_project_op()


def prewarm_qvq_ampere_grouped() -> None:
    """Load both grouped SM80 operators before CUDA Graph capture.

    Grouped callers select and pack their child payloads during preparation;
    replay must never trigger lazy TorchOp registration for either the
    segmented fallback or the fused grouped dispatcher.
    """

    _p32_rank8_project_op()
    _p32_window_grouped_op()
    _p32_window_grouped_fused_op()


def _require_warm_operator_for_capture(operator: object | None, name: str) -> None:
    """Reject lazy TorchOp registration from inside a CUDA graph capture."""

    if (
        operator is None
        and torch.cuda.is_available()
        and torch.cuda.is_current_stream_capturing()
    ):
        raise RuntimeError(
            f"QVQ Ampere {name} must be loaded before CUDA Graph capture"
        )


def _p32_window_grouped_op() -> object:
    """Resolve the native segmented dispatcher once."""

    global _P32_WINDOW_GROUPED_OP
    op = _P32_WINDOW_GROUPED_OP
    if op is None:
        op = _QVQ_AMPERE_EXTENSION.op("p32_window_grouped")
        _P32_WINDOW_GROUPED_OP = op
    return op


def _p32_window_grouped_fused_op() -> object:
    """Resolve the one-main-kernel segmented operator once."""

    global _P32_WINDOW_GROUPED_FUSED_OP
    op = _P32_WINDOW_GROUPED_FUSED_OP
    if op is None:
        op = _QVQ_AMPERE_EXTENSION.op("p32_window_grouped_fused")
        _P32_WINDOW_GROUPED_FUSED_OP = op
    return op


def _autotune_candidates(
    *, fallback: int, k_tiles: int, max_candidates: int = 12
) -> list[int]:
    """Return a small, bounded split wave around the static policy."""

    if k_tiles <= 0:
        return [1]
    max_split = min(k_tiles, 128)
    probes = (
        fallback,
        max(1, fallback // 2),
        fallback * 2,
        8,
        16,
        24,
        32,
        40,
        48,
        64,
        96,
        128,
    )
    candidates: list[int] = []
    for candidate in probes:
        candidate = min(max(1, int(candidate)), max_split)
        if candidate not in candidates:
            candidates.append(candidate)
        if len(candidates) >= max_candidates:
            break
    return candidates


def clear_qvq_ampere_autotune_cache() -> None:
    """Clear in-process plans and refresh the process-local autotune setting."""

    global _AUTOTUNE_ENABLED
    with _AUTOTUNE_CACHE_LOCK:
        _AUTOTUNE_CACHE.clear()
        _AUTOTUNE_ENABLED = _autotune_enabled()


def _autotune_split_count(
    input: torch.Tensor,
    trellis: torch.Tensor,
    levels: torch.Tensor,
    bank_ids: torch.Tensor,
    *,
    transition_bits: int,
    out_features: int,
    bank_alt_id: int,
    fallback: int,
    cache_key: _AutotuneCacheKey | None = None,
) -> int:
    """Benchmark a bounded set of split waves once and memoize the winner."""

    key = cache_key
    if key is None:
        key = _autotune_cache_key(
            input,
            transition_bits=transition_bits,
            out_features=out_features,
            bank_alt_id=bank_alt_id,
        )
    with _AUTOTUNE_CACHE_LOCK:
        cached = _AUTOTUNE_CACHE.get(key)
        if cached is not None:
            return min(cached, int(input.shape[1]) // 16)

        # CUDA event timing and host synchronization are illegal during graph
        # capture. Use the measured fallback for a cold capture without
        # memoizing it, so a later eager call can still tune this shape. A
        # shape tuned before capture takes the cached fast path above.
        if torch.cuda.is_current_stream_capturing():
            return min(int(fallback), int(input.shape[1]) // 16, 128)

        k_tiles = int(input.shape[1]) // 16
        max_candidates = max(
            2, int(os.environ.get("QVQ_AMPERE_AUTOTUNE_CANDIDATES", "12"))
        )
        candidates = _autotune_candidates(
            fallback=fallback,
            k_tiles=k_tiles,
            max_candidates=max_candidates,
        )
        warmup = max(0, int(os.environ.get("QVQ_AMPERE_AUTOTUNE_WARMUP", "5")))
        iterations = max(1, int(os.environ.get("QVQ_AMPERE_AUTOTUNE_ITERATIONS", "20")))
        repeats = max(1, int(os.environ.get("QVQ_AMPERE_AUTOTUNE_REPEATS", "3")))
        stream = torch.cuda.current_stream(input.device)
        ampere_op = _p32_window_op()
        best_split = fallback
        best_ms = float("inf")
        for candidate in candidates:
            try:
                for _ in range(warmup):
                    output = ampere_op(
                        input,
                        trellis,
                        levels,
                        bank_ids,
                        transition_bits,
                        out_features,
                        bank_alt_id,
                        candidate,
                    )
                    del output
                samples = []
                for _ in range(repeats):
                    starts = [
                        torch.cuda.Event(enable_timing=True) for _ in range(iterations)
                    ]
                    ends = [
                        torch.cuda.Event(enable_timing=True) for _ in range(iterations)
                    ]
                    for iteration in range(iterations):
                        starts[iteration].record(stream)
                        output = ampere_op(
                            input,
                            trellis,
                            levels,
                            bank_ids,
                            transition_bits,
                            out_features,
                            bank_alt_id,
                            candidate,
                        )
                        del output
                        ends[iteration].record(stream)
                    ends[-1].synchronize()
                    samples.extend(
                        start.elapsed_time(end)
                        for start, end in zip(starts, ends, strict=True)
                    )
                elapsed_ms = median(samples)
                if elapsed_ms < best_ms:
                    best_ms = elapsed_ms
                    best_split = candidate
            except (RuntimeError, ValueError):
                # Invalid candidate waves or transient launch failures should
                # not make a new model unusable; retain the known-good policy.
                continue
        _AUTOTUNE_CACHE[key] = int(best_split)
        return int(best_split)


def _resolve_transition_bits(bits: float) -> int:
    try:
        transition_bits = None if isinstance(bits, bool) else _P32_TRANSITION_BITS[bits]
    except (KeyError, TypeError):
        transition_bits = None
    if transition_bits is None:
        transition_bits = qvq_transition_bits(bits, vector_size=2)
    if transition_bits not in (4, 5, 6, 7):
        raise ValueError("QVQ P32 Ampere WMMA supports W2 through W3.5")
    return transition_bits


def _static_split_count(*, m: int, k: int, n: int, transition_bits: int) -> int:
    """Return the measured shape policy before optional timing autotune.

    This is deliberately a pure shape function.  It is shared by the runtime
    dispatcher and the public candidate enumerator so external tuners (including
    ZML) see the same Ampere baseline without importing a CUDA tensor or doing
    work during graph capture.
    """

    if m > 16:
        return 1
    if k == 640 and n == 2560:
        return {
            1: 24,
            2: 40,
            4: 40,
            8: 16,
            16: 16,
        }.get(m, 0)
    # Flash-Next direct gate/up=(K=2560, N=640) is a short scalar/WMMA
    # projection.  The generic fallback under-fills M8/M16 with eight
    # reduction waves; a thirty-two-way wave keeps the 124-SM A100 busy while
    # avoiding the extra reduction cost of the forty-way M1-M4 plan.
    if transition_bits == 6 and k == 2560 and n == 640:
        return {
            1: 40,
            2: 40,
            4: 40,
            8: 32,
            16: 32,
        }.get(m, 0)
    # Flash-Next attention-out=(K=12288, N=2560) is tuned for twelve K
    # reduction waves on SM80. Keep this rate-specific so other P32 layouts
    # continue to use the generic policy and external split_count remains
    # authoritative when supplied by ZML or another bridge.
    if transition_bits == 6 and k == 12288 and n == 2560:
        # M1-M4 use the scalar long-K stage-4 route with 32 reduction waves;
        # M8/M16 retain the WMMA route tuned for twelve waves.
        return {1: 32, 2: 32, 4: 32, 8: 12, 16: 12}.get(m, 0)
    # Flash-Next direct Q=(K=2560, N=12288) uses forty scalar waves for
    # M1-M4 and the wide WMMA sixteen-wave plan for M8/M16. This is separate
    # from grouped QKV, whose per-child policy is resolved independently.
    if transition_bits == 6 and k == 2560 and n == 12288:
        return {1: 40, 2: 40, 4: 40, 8: 16, 16: 16}.get(m, 0)
    if m == 1 and k == 5120 and n in (1024, 12288):
        return 56 if n == 1024 else 40
    if m in (2, 4) and k == 5120 and n == 1024:
        return 64
    if n == 1024 and (m, k) == (8, 5120):
        return 48
    if n == 17408 and (m, k) in ((8, 5120), (16, 5120)):
        return 10
    if n == 12288 and (m, k) == (16, 5120):
        return 9
    if n == 10240 and (m, k) == (16, 5120):
        return 10
    if n == 1024 and (m, k) == (16, 5120):
        return 32
    if n == 5120 and (m, k) == (8, 17408):
        return 40
    if n == 5120 and (m, k) == (8, 6144):
        return 24
    if n == 6144 and (m, k) == (8, 5120):
        return 40
    if n == 10240 and (m, k) == (8, 5120):
        return 16
    if n == 12288 and (m, k) == (8, 5120):
        return 14
    if n == 5120 and (m, k) == (16, 17408):
        return 24
    if n == 5120 and (m, k) == (16, 6144):
        return 12
    if n == 6144 and (m, k) == (16, 5120):
        return 10
    if (m, k) == (1, 6144) and n == 5120:
        return 48
    if (m, k) == (2, 6144) and n == 5120:
        return 64 if transition_bits == 7 else 48
    if (m, k) == (4, 5120) and n == 6144:
        return 40
    return 0


def _flash_next_group_split_counts(
    *, m: int, k: int, widths: Sequence[int], transition_bits: int
) -> tuple[int, ...] | None:
    """Return the measured grouped wave policy for Flash-Next QKV decode."""

    # The Qwen3.8-Flash-Next attention group is QKV=(12288, 512, 512) from
    # K=2560.  M1-M8 use sixteen split waves after the rectangular empty-CTA
    # tail is removed. M16 is better with four larger waves; the grouped
    # WMMA dispatcher has a matching static four-way specialization.
    if (
        transition_bits == 6
        and k == 2560
        and tuple(widths) == (12288, 512, 512)
    ):
        if m == 16:
            return (4, 4, 4)
        if m in (1, 2, 4, 8):
            return (16, 16, 16)
    # Flash-Next gate/up=(640, 640) is a short scalar grouped launch.  Forty
    # split waves keep enough K work resident for every decode batch we tune;
    # the generic M=8/16 policy otherwise falls back to eight and leaves the
    # small-N CTAs under-filled.
    if (
        transition_bits == 6
        and m in (1, 2, 4, 8, 16)
        and k == 2560
        and tuple(widths) == (640, 640)
    ):
        return (40, 40)
    return None


def qvq_h100_flash_next_expert_group_split_counts(
    *,
    device_name: str,
    compute_capability: tuple[int, int],
    m: int,
    k: int,
    widths: Sequence[int],
    transition_bits: int,
) -> tuple[int, ...] | None:
    """Return the validated narrow-expert schedule for physical H100.

    Split 32 intentionally bypasses the SM80 W3 split-40 compact
    specialization.  That specialization changes the second child's payload
    ordering when compiled for SM90; the generic segmented kernel is exact,
    graph safe, and faster than the planar fallback at every accepted rate and
    row count.
    """

    if (
        device_name == "NVIDIA H100"
        and compute_capability == (9, 0)
        and m in (1, 2, 4, 8, 16)
        and k == 2560
        and tuple(widths) == (640, 640)
        and transition_bits in (4, 5, 6, 7)
    ):
        return (32, 32)
    return None


def qvq_p32_window_ampere_kernel_candidates(
    input_shape: Sequence[int],
    *,
    out_features: int,
    bits: float,
    sm_count: int = 108,
    max_candidates: int = 12,
) -> tuple[int, ...]:
    """Enumerate legal Ampere split waves for one exact ``(M, K, N, rate)``.

    The result contains the measured shape policy first, followed by a bounded
    probe set suitable for a kernel-level autotuner.  Enumeration performs no
    CUDA allocation, event timing, or synchronization and is therefore safe to
    call while constructing a graph plan.  Benchmark and cache the selected
    ``split_count`` before capture, then pass it explicitly to
    :func:`qvq_p32_window_ampere` during replay.
    """

    shape = tuple(int(value) for value in input_shape)
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("Ampere candidate shape must be a positive (M,K) pair")
    if shape[1] % 16:
        raise ValueError("Ampere P32 input width must be divisible by 16")
    if type(out_features) is not int or out_features <= 0 or out_features % 16:
        raise ValueError("Ampere P32 output width must be a positive multiple of 16")
    if type(sm_count) is not int or sm_count <= 0:
        raise ValueError("Ampere SM count must be positive")
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    transition_bits = _resolve_transition_bits(bits)
    k_tiles = shape[1] // 16
    fallback = _static_split_count(
        m=shape[0], k=shape[1], n=out_features, transition_bits=transition_bits
    )
    if fallback == 0:
        fallback = _auto_split_count(
            in_features=shape[1],
            out_features=out_features,
            k_tiles=k_tiles,
            sm_count=sm_count,
        )
    fallback = min(max(1, fallback), k_tiles)
    return tuple(
        _autotune_candidates(
            fallback=fallback,
            k_tiles=k_tiles,
            max_candidates=max_candidates,
        )
    )


def qvq_p32_window_ampere_grouped_kernel_candidates(
    input_shape: Sequence[int],
    *,
    out_features: Sequence[int],
    bits: float,
    sm_count: int = 108,
    max_candidates: int = 12,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate per-child split waves for one grouped SM80 projection.

    Grouped QKV and gate/up launches contain children whose optimal split wave
    can differ substantially with ``N``.  A synthetic sum of the child widths
    is therefore not a valid tuning key.  This API publishes a bounded,
    deterministic set of *tuples* in child order.  It performs only shape and
    rate arithmetic, so callers may enumerate candidates while building a
    graph plan; benchmark and select one tuple before capture, then pass it as
    ``split_counts`` to :func:`qvq_p32_window_ampere_grouped`.
    """

    shape = tuple(int(value) for value in input_shape)
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError("Ampere grouped candidate shape must be a positive (M,K) pair")
    if shape[1] % 16:
        raise ValueError("Ampere grouped P32 input width must be divisible by 16")
    widths = tuple(int(value) for value in out_features)
    if not widths or len(widths) > 3:
        raise ValueError("Ampere grouped candidates support one through three children")
    if any(width <= 0 or width % 16 for width in widths):
        raise ValueError(
            "Ampere grouped P32 output widths must be positive multiples of 16"
        )
    if type(sm_count) is not int or sm_count <= 0:
        raise ValueError("Ampere SM count must be positive")
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("max_candidates must be positive")

    # Resolve the rate once up front so invalid rates fail before any candidate
    # is returned, matching the single-child public enumerator's contract.
    transition_bits = _resolve_transition_bits(bits)
    child_candidates = tuple(
        qvq_p32_window_ampere_kernel_candidates(
            shape,
            out_features=width,
            bits=bits,
            sm_count=sm_count,
            max_candidates=max_candidates,
        )
        for width in widths
    )
    grouped_policy = _flash_next_group_split_counts(
        m=shape[0],
        k=shape[1],
        widths=widths,
        transition_bits=transition_bits,
    )
    baseline = (
        grouped_policy
        if grouped_policy is not None
        else tuple(candidates[0] for candidates in child_candidates)
    )
    candidates: list[tuple[int, ...]] = [baseline]
    seen = {baseline}
    if max_candidates == 1:
        return tuple(candidates)

    # Walk the per-child alternatives in rounds.  This keeps the common tuning
    # set small and makes it possible to isolate which grouped child benefits
    # from a different wave.  The Cartesian product is added only while budget
    # remains, in deterministic lexicographic order.
    max_child_options = max(len(options) for options in child_candidates)
    for option_index in range(1, max_child_options):
        for child_index, child_options in enumerate(child_candidates):
            if option_index >= len(child_options):
                continue
            candidate = list(baseline)
            candidate[child_index] = child_options[option_index]
            value = tuple(candidate)
            if value not in seen:
                seen.add(value)
                candidates.append(value)
            if len(candidates) >= max_candidates:
                return tuple(candidates)

    if len(candidates) < max_candidates and len(child_candidates) > 1:
        import itertools

        for value in itertools.product(*(options[1:] for options in child_candidates)):
            candidate = tuple(value)
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)
            if len(candidates) >= max_candidates:
                break
    return tuple(candidates)


def _resolve_split_count(
    input: torch.Tensor,
    trellis: torch.Tensor,
    levels: torch.Tensor,
    bank_ids: torch.Tensor,
    *,
    transition_bits: int,
    out_features: int,
    bank_alt_id: int,
    split_count: int,
) -> int:
    """Resolve exactly the split policy used by an independent child launch."""

    if split_count < 0:
        raise ValueError("QVQ P32 Ampere split_count must be non-negative")
    if split_count:
        return int(split_count)
    static = _static_split_count(
        m=int(input.shape[0]),
        k=int(input.shape[1]),
        n=int(out_features),
        transition_bits=transition_bits,
    )
    if static:
        return min(static, int(input.shape[1]) // 16)

    # Environment configuration is process-level. Reading ``os.environ`` on
    # every cached launch costs more than the cache lookup itself, so refresh
    # it only when the process-local plan cache is cleared.
    autotune = _AUTOTUNE_ENABLED
    autotune_key = None
    if autotune:
        autotune_key = _autotune_cache_key(
            input,
            transition_bits=transition_bits,
            out_features=out_features,
            bank_alt_id=bank_alt_id,
        )
        cached = _AUTOTUNE_CACHE.get(autotune_key)
        if cached is not None:
            return min(int(cached), int(input.shape[1]) // 16)

    if not input.is_cuda:
        raise ValueError("QVQ P32 Ampere input must be CUDA")

    split_count = _auto_split_count(
        in_features=int(input.shape[1]),
        out_features=int(out_features),
        k_tiles=int(input.shape[1]) // 16,
        sm_count=_device_sm_count(input.device),
    )
    # The scalar M<=4 kernel groups sixteen N16 tiles per CTA. Wide
    # projections therefore need a fuller split wave than the WMMA shape
    # table to keep the Ampere SMs resident during the short decode.
    if input.shape[0] <= 4 and input.shape[1] <= 6144:
        shape = (int(input.shape[1]), int(out_features))
        small_m_split = (
            24
            if shape == (6144, 5120)
            else 32
            if shape == (5120, 1024)
            else 40
            if input.shape[0] == 4
            else 32
        )
        split_count = min(small_m_split, int(input.shape[1]) // 16)
    elif input.shape[0] == 8 and input.shape[1] <= 6144:
        m8_split = {
            1024: 48,
            5120: 32,
            6144: 32,
            10240: 16,
            12288: 16,
            17408: 16,
        }.get(int(out_features))
        if m8_split is not None:
            split_count = min(m8_split, int(input.shape[1]) // 16)
    elif input.shape[0] == 16 and input.shape[1] <= 6144:
        m16_split = {
            1024: 32,
            5120: 12,
            6144: 16,
            10240: 16,
            17408: 10,
        }.get(int(out_features))
        if m16_split is not None:
            split_count = min(m16_split, int(input.shape[1]) // 16)
    elif (int(input.shape[1]), int(out_features)) == (17408, 5120):
        long_k_split = (
            128 if input.shape[0] == 1 else 96 if input.shape[0] == 2 else 32
        )
        split_count = min(long_k_split, int(input.shape[1]) // 16)
    if autotune and autotune_key is not None:
        split_count = _autotune_split_count(
            input,
            trellis,
            levels,
            bank_ids,
            transition_bits=transition_bits,
            out_features=int(out_features),
            bank_alt_id=int(bank_alt_id),
            fallback=int(split_count),
            cache_key=autotune_key,
        )
    return int(split_count)


def qvq_p32_rank8_project(
    input: torch.Tensor,
    rank8_a: torch.Tensor,
) -> torch.Tensor:
    """Project several rank-8 factors in one SM80 FP32-output GEMM."""

    if input.ndim != 2 or rank8_a.ndim != 2:
        raise ValueError("rank8 projection inputs must be rank 2")
    if input.dtype != torch.float16 or rank8_a.dtype != torch.float16:
        raise ValueError("rank8 projection inputs must be float16")
    if tuple(rank8_a.shape[:1]) != (input.shape[1],) or rank8_a.shape[1] % 8:
        raise ValueError("rank8_a must have shape [K, 8 * segments]")
    if (
        not input.is_contiguous()
        or not rank8_a.is_contiguous()
        or input.device != rank8_a.device
    ):
        raise ValueError("rank8 projection inputs must be contiguous on one device")
    _require_warm_operator_for_capture(_P32_RANK8_PROJECT_OP, "rank8 project operator")
    return _p32_rank8_project_op()(input, rank8_a)


def qvq_p32_window_ampere(
    input: torch.Tensor,
    trellis: torch.Tensor,
    levels: torch.Tensor,
    bank_ids: torch.Tensor,
    bits: float,
    *,
    out_features: int,
    bank_alt_id: int = 3,
    split_count: int = 0,
    rank8_a: torch.Tensor | None = None,
    rank8_b: torch.Tensor | None = None,
    rank8_scale: float = 1.0,
    rank8_down: torch.Tensor | None = None,
    return_ordered_partials: bool = False,
) -> torch.Tensor:
    """Run exact continuous-window P32 with an optional additive rank-8 update.

    The rank-8 epilogue computes ``Y += rank8_scale * (input @ rank8_a) @
    rank8_b`` with FP32 accumulation. ``rank8_a`` is contiguous FP16 ``[K,
    8]`` and ``rank8_b`` is contiguous FP16 or FP32 ``[8, N]``.
    """

    try:
        transition_bits = None if isinstance(bits, bool) else _P32_TRANSITION_BITS[bits]
    except (KeyError, TypeError):
        transition_bits = None
    if transition_bits is None:
        transition_bits = qvq_transition_bits(bits, vector_size=2)
    if transition_bits not in (4, 5, 6, 7):
        raise ValueError("QVQ P32 Ampere WMMA supports W2 through W3.5")
    _require_warm_operator_for_capture(_P32_WINDOW_OP, "window operator")
    if (
        split_count != 0
        and input.is_cuda
        and not torch.cuda.is_current_stream_capturing()
    ):
        # An explicit-split warm-up can be followed by a cold auto-tuned graph
        # capture. Resolve this immutable device property while queries are
        # legal so the capture path stays allocation- and query-free.
        _device_sm_count(input.device)
    if rank8_down is not None and rank8_a is not None:
        raise ValueError("rank8_a and rank8_down are mutually exclusive")
    if return_ordered_partials and (
        rank8_a is not None or rank8_b is not None or rank8_down is not None
    ):
        raise ValueError("ordered partial output does not support rank-8 correction")
    if rank8_down is not None and rank8_b is None:
        raise ValueError("rank8_down requires rank8_b")
    if rank8_down is None and (rank8_a is None) != (rank8_b is None):
        raise ValueError("rank8_a and rank8_b must be provided together")
    if not math.isfinite(float(rank8_scale)):
        raise ValueError("rank8_scale must be finite")
    if rank8_a is not None or rank8_down is not None:
        if input.ndim != 2:
            raise ValueError("QVQ P32 Ampere input must be rank 2")
        if rank8_a is not None and (
            rank8_a.ndim != 2 or tuple(rank8_a.shape) != (input.shape[1], 8)
        ):
            raise ValueError("rank8_a must have shape [K, 8]")
        if rank8_down is not None and (
            rank8_down.ndim != 2 or tuple(rank8_down.shape) != (input.shape[0], 8)
            or rank8_down.dtype != torch.float32
            or rank8_down.stride(1) != 1
            or rank8_down.stride(0) < 8
        ):
            raise ValueError("rank8_down must be FP32 [M, 8] with unit column stride")
        if (
            rank8_b is None
            or rank8_b.ndim != 2
            or tuple(rank8_b.shape) != (8, out_features)
        ):
            raise ValueError("rank8_b must have shape [8, N]")
        if rank8_a is not None and rank8_a.dtype != torch.float16:
            raise ValueError("rank8_a must be float16")
        if rank8_b.dtype not in (
            torch.float16,
            torch.float32,
        ):
            raise ValueError(
                "rank8_a must be float16 and rank8_b must be float16 or float32"
            )
        if rank8_a is not None and not rank8_a.is_contiguous():
            raise ValueError("rank8_a must be contiguous")
        if not rank8_b.is_contiguous():
            raise ValueError("rank8_a and rank8_b must be contiguous")
        if (
            rank8_b.device != input.device
            or (rank8_a is not None and rank8_a.device != input.device)
            or (rank8_down is not None and rank8_down.device != input.device)
        ):
            raise ValueError("rank8_a and rank8_b must share the input device")
        if split_count == 0:
            split_count = _static_split_count(
                m=int(input.shape[0]),
                k=int(input.shape[1]),
                n=int(out_features),
                transition_bits=transition_bits,
            )
            if split_count == 0:
                split_count = _auto_split_count(
                    in_features=int(input.shape[1]),
                    out_features=int(out_features),
                    k_tiles=int(input.shape[1]) // 16,
                    sm_count=_device_sm_count(input.device),
                )
            if input.shape[0] > 16:
                split_count = 1
        return _p32_window_op()(
            input,
            trellis,
            levels,
            bank_ids,
            transition_bits,
            out_features,
            bank_alt_id,
            split_count,
            rank8_a,
            rank8_b,
            float(rank8_scale),
            rank8_down,
        )
    if (
        split_count == 0
        and input.shape[0] == 1
        and input.shape[1] == 5120
        and out_features in (1024, 12288)
    ):
        return _p32_window_op()(
            input,
            trellis,
            levels,
            bank_ids,
            transition_bits,
            out_features,
            bank_alt_id,
            56 if out_features == 1024 else 40,
        )
    if (
        split_count == 0
        and input.shape[0] in (2, 4)
        and input.shape[1] == 5120
        and out_features == 1024
    ):
        return _p32_window_op()(
            input,
            trellis,
            levels,
            bank_ids,
            transition_bits,
            out_features,
            bank_alt_id,
            64,
        )
    if split_count == 0 and out_features == 1024 and input.shape == (8, 5120):
        return _p32_window_op()(
            input,
            trellis,
            levels,
            bank_ids,
            transition_bits,
            out_features,
            bank_alt_id,
            48,
        )
    if split_count == 0 and out_features == 17408 and input.shape == (8, 5120):
        return _p32_window_op()(
            input,
            trellis,
            levels,
            bank_ids,
            transition_bits,
            out_features,
            bank_alt_id,
            10,
        )
    if split_count == 0 and out_features == 17408 and input.shape == (16, 5120):
        return _p32_window_op()(
            input,
            trellis,
            levels,
            bank_ids,
            transition_bits,
            out_features,
            bank_alt_id,
            10,
        )
    if split_count == 0 and out_features == 12288 and input.shape == (16, 5120):
        return _p32_window_op()(
            input,
            trellis,
            levels,
            bank_ids,
            transition_bits,
            out_features,
            bank_alt_id,
            9,
        )
    if (
        split_count == 0
        and transition_bits in (4, 5, 6, 7)
        and out_features == 10240
        and input.shape == (16, 5120)
    ):
        return _p32_window_op()(
            input,
            trellis,
            levels,
            bank_ids,
            transition_bits,
            out_features,
            bank_alt_id,
            10,
        )
    if split_count == 0 and out_features == 1024 and input.shape == (16, 5120):
        return _p32_window_op()(
            input,
            trellis,
            levels,
            bank_ids,
            transition_bits,
            out_features,
            bank_alt_id,
            32,
        )
    if split_count == 0 and out_features == 5120 and input.shape == (8, 17408):
        return _p32_window_op()(
            input,
            trellis,
            levels,
            bank_ids,
            transition_bits,
            out_features,
            bank_alt_id,
            40,
        )
    if split_count == 0 and out_features == 5120 and input.shape == (8, 6144):
        return _p32_window_op()(
            input,
            trellis,
            levels,
            bank_ids,
            transition_bits,
            out_features,
            bank_alt_id,
            24,
        )
    if split_count == 0 and out_features == 6144 and input.shape == (8, 5120):
        return _p32_window_op()(
            input,
            trellis,
            levels,
            bank_ids,
            transition_bits,
            out_features,
            bank_alt_id,
            40,
        )
    if split_count == 0 and out_features == 10240 and input.shape == (8, 5120):
        return _p32_window_op()(
            input,
            trellis,
            levels,
            bank_ids,
            transition_bits,
            out_features,
            bank_alt_id,
            16,
        )
    if split_count == 0 and out_features == 12288 and input.shape == (8, 5120):
        return _p32_window_op()(
            input,
            trellis,
            levels,
            bank_ids,
            transition_bits,
            out_features,
            bank_alt_id,
            14,
        )
    if split_count == 0 and out_features == 5120 and input.shape == (16, 17408):
        return _p32_window_op()(
            input,
            trellis,
            levels,
            bank_ids,
            transition_bits,
            out_features,
            bank_alt_id,
            24,
        )
    if split_count == 0 and out_features == 5120 and input.shape == (16, 6144):
        return _p32_window_op()(
            input,
            trellis,
            levels,
            bank_ids,
            transition_bits,
            out_features,
            bank_alt_id,
            12,
        )
    if split_count == 0 and out_features == 6144 and input.shape == (16, 5120):
        return _p32_window_op()(
            input,
            trellis,
            levels,
            bank_ids,
            transition_bits,
            out_features,
            bank_alt_id,
            10,
        )
    if split_count == 0 and input.shape[0] == 1:
        shape = (int(input.shape[1]), int(out_features))
        measured_m1_projection_split = None
        if shape == (6144, 5120):
            measured_m1_projection_split = 48
        if measured_m1_projection_split is not None:
            return _p32_window_op()(
                input,
                trellis,
                levels,
                bank_ids,
                transition_bits,
                out_features,
                bank_alt_id,
                measured_m1_projection_split,
            )
    if (
        split_count == 0
        and input.shape == (2, 6144)
        and out_features == 5120
    ):
        return _p32_window_op()(
            input,
            trellis,
            levels,
            bank_ids,
            transition_bits,
            out_features,
            bank_alt_id,
            64 if transition_bits == 7 else 48,
        )
    if (
        split_count == 0
        and input.shape == (4, 5120)
        and out_features == 6144
    ):
        return _p32_window_op()(
            input,
            trellis,
            levels,
            bank_ids,
            transition_bits,
            out_features,
            bank_alt_id,
            40,
        )
    if split_count == 0 and input.shape[1] == 640 and out_features == 2560:
        down_split = _static_split_count(
            m=int(input.shape[0]),
            k=640,
            n=2560,
            transition_bits=transition_bits,
        )
        if down_split:
            return _p32_window_op()(
                input,
                trellis,
                levels,
                bank_ids,
                transition_bits,
                out_features,
                bank_alt_id,
                down_split,
                None,
                None,
                1.0,
                None,
                return_ordered_partials,
            )
    if (
        split_count == 0
        and transition_bits == 6
        and input.shape[1] == 12288
        and out_features == 2560
    ):
        o_split = _static_split_count(
            m=int(input.shape[0]),
            k=12288,
            n=2560,
            transition_bits=transition_bits,
        )
        if o_split:
            return _p32_window_op()(
                input,
                trellis,
                levels,
                bank_ids,
                transition_bits,
                out_features,
                bank_alt_id,
                o_split,
            )
    if (
        split_count == 0
        and transition_bits == 6
        and input.shape[1] == 2560
        and out_features == 12288
    ):
        q_split = _static_split_count(
            m=int(input.shape[0]),
            k=2560,
            n=12288,
            transition_bits=transition_bits,
        )
        if q_split:
            return _p32_window_op()(
                input,
                trellis,
                levels,
                bank_ids,
                transition_bits,
                out_features,
                bank_alt_id,
                q_split,
            )
    if (
        split_count == 0
        and transition_bits == 6
        and input.shape[1] == 2560
        and out_features == 640
    ):
        gate_up_split = _static_split_count(
            m=int(input.shape[0]),
            k=2560,
            n=640,
            transition_bits=transition_bits,
        )
        if gate_up_split:
            return _p32_window_op()(
                input,
                trellis,
                levels,
                bank_ids,
                transition_bits,
                out_features,
                bank_alt_id,
                gate_up_split,
            )
    if split_count == 0:
        # Environment configuration is process-level. Reading ``os.environ``
        # on every cached launch costs more than the cache lookup itself, so
        # refresh it only when the process-local plan cache is cleared.
        autotune = _AUTOTUNE_ENABLED and input.shape[0] <= 16
        autotune_key = None
        if autotune:
            autotune_key = _autotune_cache_key(
                input,
                transition_bits=transition_bits,
                out_features=out_features,
                bank_alt_id=bank_alt_id,
            )
            # A cache hit only reads one process-local dictionary entry. Keep
            # that overwhelmingly common path out of the tuning lock; the
            # cold helper acquires the lock and rechecks before benchmarking,
            # so concurrent misses still tune exactly once.
            cached = _AUTOTUNE_CACHE.get(autotune_key)
            if cached is not None:
                return _p32_window_op()(
                    input,
                    trellis,
                    levels,
                    bank_ids,
                    transition_bits,
                    out_features,
                    bank_alt_id,
                    cached,
                )

        if not input.is_cuda:
            raise ValueError("QVQ P32 Ampere input must be CUDA")

        if split_count == 0:
            split_count = _auto_split_count(
                in_features=int(input.shape[1]),
                out_features=int(out_features),
                k_tiles=int(input.shape[1]) // 16,
                sm_count=_device_sm_count(input.device),
            )
        if input.shape[0] > 16:
            # Row-blocked prefill has ample independent CTAs; keep one K wave
            # so the partial tensor does not scale with the large M dimension.
            split_count = 1
        # The scalar M<=4 kernel groups sixteen N16 tiles per CTA.  Wide
        # projections therefore need a fuller split wave than the WMMA
        # shape table (which was tuned for four-warp/N64 CTAs) to keep all
        # 124 Ampere SMs resident during the short decode.  The same policy
        # applies to M=1, whose scalar route was tuned first.
        if input.shape[0] <= 4 and input.shape[1] <= 6144:
            # Attention-out has enough N64 CTAs that 24 slices beat the
            # reduction overhead of a 32-way wave for the scalar M1-M4
            # route. Other short-K projections retain the fuller 32-way wave.
            shape = (int(input.shape[1]), int(out_features))
            small_m_split = (
                24
                if shape == (6144, 5120)
                else 32
                if shape == (5120, 1024)
                else 40
                if input.shape[0] == 4
                else 32
            )
            split_count = min(small_m_split, int(input.shape[1]) // 16)
        elif input.shape[0] == 8 and input.shape[1] <= 6144:
            # The WMMA partial-row path uses four N16 tiles per CTA.  The
            # original shape table was tuned for M16 and leaves short-M8
            # projections under-filled, especially the small-N KV and
            # attention projections. Keep the measured split choices local
            # to M8; M5-M7 retain the conservative generic table.
            m8_split = {
                1024: 48,
                5120: 32,
                6144: 32,
                10240: 16,
                12288: 16,
                17408: 16,
            }.get(int(out_features))
            if m8_split is not None:
                split_count = min(m8_split, int(input.shape[1]) // 16)
        elif input.shape[0] == 16 and input.shape[1] <= 6144:
            # The full-row WMMA path also benefits from a fuller wave on
            # small-N projections. These choices are deliberately shape
            # specific: the wide Q and MLP projections already have enough
            # CTAs at their original measured splits.
            m16_split = {
                1024: 32,
                5120: 12,
                6144: 16,
                10240: 16,
                17408: 10,
            }.get(int(out_features))
            if m16_split is not None:
                split_count = min(m16_split, int(input.shape[1]) // 16)
        elif (int(input.shape[1]), int(out_features)) == (17408, 5120):
            # MLP-down is the only measured long-K Qwen shape. Its original
            # eight-way wave leaves too few CTAs per SM on the 124-SM A100;
            # a wider wave reduces the per-CTA K span and the split reducer
            # remains cheaper than the additional idle time. M1-M2 use the
            # scalar route and continue to benefit from a wide wave (128 for
            # M1, 96 for M2); M4+ retain the measured 32-way WMMA wave.
            long_k_split = (
                128 if input.shape[0] == 1 else 96 if input.shape[0] == 2 else 32
            )
            split_count = min(long_k_split, int(input.shape[1]) // 16)
        if autotune and autotune_key is not None:
            split_count = _autotune_split_count(
                input,
                trellis,
                levels,
                bank_ids,
                transition_bits=transition_bits,
                out_features=int(out_features),
                bank_alt_id=int(bank_alt_id),
                fallback=int(split_count),
                cache_key=autotune_key,
            )
    return _p32_window_op()(
        input,
        trellis,
        levels,
        bank_ids,
        transition_bits,
        out_features,
        bank_alt_id,
        split_count,
        None,
        None,
        1.0,
        None,
        return_ordered_partials,
    )


def qvq_p32_window_ampere_group_plan(
    input: torch.Tensor,
    trellises: Sequence[torch.Tensor],
    levels: torch.Tensor,
    bank_ids: Sequence[torch.Tensor],
    bits: float,
    *,
    out_features: Sequence[int],
    bank_alt_ids: Sequence[int],
    split_counts: Sequence[int] | None = None,
) -> QVQAmpereGroupedP32Plan:
    """Resolve every child exactly as an independent P32 launch.

    Every segment resolves the same launch policy it would receive as an
    independent projection.  In particular, no policy is selected from the
    synthetic sum of the output widths.
    """

    trellises = tuple(trellises)
    bank_ids = tuple(bank_ids)
    out_features = tuple(int(value) for value in out_features)
    bank_alt_ids = tuple(int(value) for value in bank_alt_ids)
    if not trellises:
        raise ValueError("QVQ P32 Ampere grouped execution requires at least one segment")
    segment_count = len(trellises)
    if not (
        len(bank_ids) == segment_count
        and len(out_features) == segment_count
        and len(bank_alt_ids) == segment_count
    ):
        raise ValueError("QVQ P32 Ampere grouped segment metadata lengths must match")
    transition_bits = _resolve_transition_bits(bits)
    if split_counts is None:
        tuned_splits = _flash_next_group_split_counts(
            m=int(input.shape[0]),
            k=int(input.shape[1]),
            widths=out_features,
            transition_bits=transition_bits,
        )
        requested_splits = (
            tuned_splits if tuned_splits is not None else (0,) * segment_count
        )
    else:
        requested_splits = tuple(int(value) for value in split_counts)
        if len(requested_splits) != segment_count:
            raise ValueError("QVQ P32 Ampere grouped split_counts length must match")
    if input.ndim != 2 or input.shape[1] <= 0 or input.shape[1] % 16:
        raise ValueError("QVQ P32 Ampere grouped input must be a 2D K16 matrix")
    for width, alt_id in zip(out_features, bank_alt_ids, strict=True):
        if width <= 0 or width % 16:
            raise ValueError(
                "QVQ P32 Ampere grouped output widths must be positive multiples of 16"
            )
        if alt_id < 0 or alt_id > 3:
            raise ValueError("QVQ P32 Ampere grouped bank IDs must be in [0, 3]")

    resolved_splits = tuple(
        _resolve_split_count(
            input,
            trellis,
            levels,
            selectors,
            transition_bits=transition_bits,
            out_features=width,
            bank_alt_id=alt_id,
            split_count=requested_split,
        )
        for trellis, selectors, width, alt_id, requested_split in zip(
            trellises,
            bank_ids,
            out_features,
            bank_alt_ids,
            requested_splits,
            strict=True,
        )
    )
    output_tile_start = 0
    segments = []
    for width, alt_id, split_count in zip(
        out_features, bank_alt_ids, resolved_splits, strict=True
    ):
        output_tile_count = width // 16
        segments.append(
            QVQAmpereP32SegmentPlan(
                output_tile_start=output_tile_start,
                output_tile_count=output_tile_count,
                out_features=width,
                bank_alt_id=alt_id,
                split_count=split_count,
            )
        )
        output_tile_start += output_tile_count
    return QVQAmpereGroupedP32Plan(
        in_features=int(input.shape[1]),
        transition_bits=transition_bits,
        segments=tuple(segments),
    )


def qvq_pack_p32_window_ampere_group(
    trellises: Sequence[torch.Tensor],
    bank_ids: Sequence[torch.Tensor],
    plan: QVQAmpereGroupedP32Plan,
) -> QVQAmpereGroupedP32Payload:
    """Losslessly pack child window payloads along the N16 tile dimension.

    The returned tensors are intended to be cached with the quantized module
    group.  Repacking on every inference call would erase the launch savings.
    """

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "QVQ Ampere grouped payload must be packed before CUDA Graph capture"
        )

    trellises = tuple(trellises)
    bank_ids = tuple(bank_ids)
    if len(trellises) != len(plan.segments) or len(bank_ids) != len(plan.segments):
        raise ValueError("QVQ P32 Ampere grouped payload lengths must match the plan")
    k_tiles = plan.in_features // 16
    words_per_tile = 4 * plan.transition_bits
    trellis_parts = []
    bank_parts = []
    child_trellises = []
    child_bank_ids = []
    for trellis, selectors, segment in zip(
        trellises, bank_ids, plan.segments, strict=True
    ):
        expected_tiles = k_tiles * segment.output_tile_count
        if trellis.dtype != torch.int32 or trellis.numel() != expected_tiles * words_per_tile:
            raise ValueError("QVQ P32 Ampere grouped trellis has the wrong dtype or word count")
        if selectors.dtype != torch.uint8 or selectors.numel() != expected_tiles:
            raise ValueError("QVQ P32 Ampere grouped bank ids have the wrong dtype or length")
        if trellis.device != trellises[0].device or selectors.device != trellises[0].device:
            raise ValueError("QVQ P32 Ampere grouped payloads must share one device")
        child_trellis = trellis.reshape(
            k_tiles, segment.output_tile_count, words_per_tile
        ).contiguous()
        child_banks = selectors.reshape(k_tiles, segment.output_tile_count).contiguous()
        child_trellises.append(child_trellis.reshape(-1, words_per_tile))
        child_bank_ids.append(child_banks.reshape(-1))
        trellis_parts.append(child_trellis)
        bank_parts.append(child_banks)
    grouped_trellis = torch.cat(trellis_parts, dim=1).reshape(
        -1, words_per_tile
    ).contiguous()
    grouped_bank_ids = torch.cat(bank_parts, dim=1).reshape(-1).contiguous()
    return QVQAmpereGroupedP32Payload(
        trellis=grouped_trellis,
        bank_ids=grouped_bank_ids,
        plan=plan,
        child_trellises=tuple(child_trellises),
        child_bank_ids=tuple(child_bank_ids),
    )


def qvq_p32_window_ampere_grouped_packed(
    input: torch.Tensor,
    payload: QVQAmpereGroupedP32Payload,
    levels: torch.Tensor,
    *,
    rank8_as: Sequence[torch.Tensor] | None = None,
    rank8_bs: Sequence[torch.Tensor] | None = None,
    rank8_scales: Sequence[float] | None = None,
    rank8_packed_a: torch.Tensor | None = None,
    rank8_packed_b: torch.Tensor | None = None,
    return_ordered_partials: bool = False,
) -> tuple[torch.Tensor, ...] | torch.Tensor:
    """Execute a cached grouped payload with one segmented main launch.

    ``rank8_as``/``rank8_bs`` are the child-local factors.  The A factors are
    concatenated for one shared producer, while each B remains child-local so
    no per-call transpose or slice copy is needed in the reducers.
    """

    plan = payload.plan
    if int(input.shape[1]) != plan.in_features:
        raise ValueError("QVQ P32 Ampere grouped input K does not match the plan")
    widths = [segment.out_features for segment in plan.segments]
    alt_ids = [segment.bank_alt_id for segment in plan.segments]
    split_counts = [segment.split_count for segment in plan.segments]
    if return_ordered_partials and (rank8_as is not None or rank8_bs is not None):
        raise ValueError("ordered grouped partial output does not support rank-8 correction")
    # The plain kernel uses a warp-parallel reducer for these two KV cases.
    # Its tree order cannot be represented by the generic segmented reducer,
    # so fail closed to the exact child dispatcher rather than changing bits.
    uses_warp_reducer = any(
        segment.out_features == 1024
        and (
            (input.shape[0] <= 4 and segment.split_count == 64)
            or (input.shape[0] == 8 and segment.split_count == 48)
        )
        for segment in plan.segments
    )
    if len(plan.segments) > 3 or plan.in_features > 6144 or uses_warp_reducer:
        if return_ordered_partials:
            raise ValueError("ordered grouped partial output requires the fused operator")
        if rank8_as is not None or rank8_bs is not None:
            raise ValueError(
                "grouped Ampere rank8 requires supported child window shapes"
            )
        _require_warm_operator_for_capture(_P32_WINDOW_GROUPED_OP, "grouped operator")
        k_tiles = plan.in_features // 16
        words_per_tile = 4 * plan.transition_bits
        total_n_tiles = plan.out_features // 16
        grouped_trellis = payload.trellis.reshape(k_tiles, total_n_tiles, words_per_tile)
        grouped_bank_ids = payload.bank_ids.reshape(k_tiles, total_n_tiles)
        child_trellises = []
        child_bank_ids = []
        for segment in plan.segments:
            start = segment.output_tile_start
            stop = start + segment.output_tile_count
            child_trellises.append(
                grouped_trellis[:, start:stop].reshape(-1, words_per_tile).contiguous()
            )
            child_bank_ids.append(
                grouped_bank_ids[:, start:stop].reshape(-1).contiguous()
            )
        return tuple(
            _p32_window_grouped_op()(
                input,
                child_trellises,
                levels,
                child_bank_ids,
                plan.transition_bits,
                widths,
                alt_ids,
                split_counts,
            )
        )
    if (rank8_as is None) != (rank8_bs is None):
        raise ValueError("rank8_as and rank8_bs must be provided together")
    if rank8_as is not None:
        if len(rank8_as) != len(plan.segments) or len(rank8_bs) != len(plan.segments):
            raise ValueError("grouped Ampere rank8 factor lists must match segments")
        if rank8_scales is None:
            scales = [1.0] * len(plan.segments)
        else:
            scales = [float(scale) for scale in rank8_scales]
            if len(scales) != len(plan.segments):
                raise ValueError("grouped Ampere rank8 scales must match segment count")
        transition_rate = {
            4: 2.0,
            5: 2.5,
            6: 3.0,
            7: 3.5,
        }[plan.transition_bits]
        if (
            len(plan.segments) in (2, 3)
            and input.shape[0] <= 16
            and plan.in_features <= 6144
        ):
            packed_a = (
                torch.cat(tuple(rank8_as), dim=1).contiguous()
                if rank8_packed_a is None
                else rank8_packed_a
            )
            packed_b = (
                torch.cat(tuple(rank8_bs), dim=1).contiguous()
                if rank8_packed_b is None
                else rank8_packed_b
            )
            _require_warm_operator_for_capture(
                _P32_WINDOW_GROUPED_FUSED_OP, "grouped fused operator"
            )
            grouped_output = _p32_window_grouped_fused_op()(
                input,
                payload.trellis,
                levels,
                payload.bank_ids,
                plan.transition_bits,
                widths,
                alt_ids,
                split_counts,
                packed_a,
                packed_b,
                scales,
            )
            row_count = int(input.shape[0])
            return tuple(
                child.reshape(row_count, width)
                for child, width in zip(
                    torch.split(
                        grouped_output,
                        [row_count * width for width in widths],
                    ),
                    widths,
                    strict=True,
                )
            )
        packed_a = (
            torch.cat(tuple(rank8_as), dim=1).contiguous()
            if rank8_packed_a is None
            else rank8_packed_a
        )
        rank8_down = qvq_p32_rank8_project(input, packed_a)
        outputs = []
        for segment, (rank8_b, scale) in enumerate(zip(rank8_bs, scales, strict=True)):
            outputs.append(
                qvq_p32_window_ampere(
                    input,
                    payload.child_trellises[segment],
                    levels,
                    payload.child_bank_ids[segment],
                    transition_rate,
                    out_features=plan.segments[segment].out_features,
                    bank_alt_id=plan.segments[segment].bank_alt_id,
                    split_count=plan.segments[segment].split_count,
                    rank8_b=rank8_b,
                    rank8_scale=scale,
                    rank8_down=rank8_down[:, segment * 8:(segment + 1) * 8],
                )
            )
        return tuple(outputs)
    _require_warm_operator_for_capture(
        _P32_WINDOW_GROUPED_FUSED_OP, "grouped fused operator"
    )
    grouped_output = _p32_window_grouped_fused_op()(
        input,
        payload.trellis,
        levels,
        payload.bank_ids,
        plan.transition_bits,
        widths,
        alt_ids,
        split_counts,
        None,
        None,
        [],
        return_ordered_partials,
    )
    if return_ordered_partials:
        return grouped_output
    row_count = int(input.shape[0])
    return tuple(
        child.reshape(row_count, width)
        for child, width in zip(
            torch.split(grouped_output, [row_count * width for width in widths]),
            widths,
            strict=True,
        )
    )


def qvq_p32_window_ampere_grouped(
    input: torch.Tensor,
    trellises: Sequence[torch.Tensor],
    levels: torch.Tensor,
    bank_ids: Sequence[torch.Tensor],
    bits: float,
    *,
    out_features: Sequence[int],
    bank_alt_ids: Sequence[int],
    split_counts: Sequence[int] | None = None,
    rank8_as: Sequence[torch.Tensor] | None = None,
    rank8_bs: Sequence[torch.Tensor] | None = None,
    rank8_scales: Sequence[float] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Run legal P32 siblings from one shared transformed activation.

    The fused route preserves every child K partition and its left-to-right
    FP32 split reduction. Unsupported exact-reduction cases fail closed to the
    ordinary native child dispatcher.
    """

    trellises = tuple(trellises)
    bank_ids = tuple(bank_ids)
    plan = qvq_p32_window_ampere_group_plan(
        input,
        trellises,
        levels,
        bank_ids,
        bits,
        out_features=out_features,
        bank_alt_ids=bank_alt_ids,
        split_counts=split_counts,
    )
    payload = qvq_pack_p32_window_ampere_group(trellises, bank_ids, plan)
    if (rank8_as is None) != (rank8_bs is None):
        raise ValueError("rank8_as and rank8_bs must be provided together")
    if rank8_as is None:
        return qvq_p32_window_ampere_grouped_packed(input, payload, levels)
    if len(rank8_as) != len(plan.segments) or len(rank8_bs) != len(plan.segments):
        raise ValueError("grouped Ampere rank8 factor lists must match segments")
    return qvq_p32_window_ampere_grouped_packed(
        input,
        payload,
        levels,
        rank8_as=tuple(rank8_as),
        rank8_bs=tuple(rank8_bs),
        rank8_scales=rank8_scales,
    )


__all__ = [
    "QVQAmpereGroupedP32Payload",
    "QVQAmpereGroupedP32Plan",
    "QVQAmpereP32SegmentPlan",
    "clear_qvq_ampere_autotune_cache",
    "prewarm_qvq_ampere",
    "prewarm_qvq_ampere_grouped",
    "qvq_h100_flash_next_expert_group_split_counts",
    "qvq_p32_rank8_project",
    "qvq_p32_window_ampere",
    "qvq_p32_window_ampere_group_plan",
    "qvq_p32_window_ampere_grouped",
    "qvq_p32_window_ampere_grouped_kernel_candidates",
    "qvq_p32_window_ampere_grouped_packed",
    "qvq_p32_window_ampere_kernel_candidates",
    "qvq_pack_p32_window_ampere_group",
]
