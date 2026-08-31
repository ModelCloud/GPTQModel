# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Exact continuous-window P32 WMMA kernel for A100-class sm_80 GPUs."""

from __future__ import annotations

import os
import threading
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
)
_TORCH_NVCC_UNDEFINES = (
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
)
_SM_COUNT_CACHE: dict[tuple[str, int], int] = {}
_AutotuneCacheKey = tuple[int, torch.dtype, int, int, int, int, int]
_AUTOTUNE_CACHE: dict[_AutotuneCacheKey, int] = {}
_AUTOTUNE_CACHE_LOCK = threading.RLock()


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
    required_ops=("p32_window",),
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
        int(input.shape[0]),
        int(input.shape[1]),
        int(out_features),
        int(transition_bits),
        int(bank_alt_id),
    )


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
    """Clear the in-process Ampere launch-plan entries."""

    with _AUTOTUNE_CACHE_LOCK:
        _AUTOTUNE_CACHE.clear()


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
        best_split = fallback
        best_ms = float("inf")
        for candidate in candidates:
            try:
                for _ in range(warmup):
                    output = _QVQ_AMPERE_EXTENSION.op("p32_window")(
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
                        output = _QVQ_AMPERE_EXTENSION.op("p32_window")(
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
) -> torch.Tensor:
    """Run exact continuous-window P32 with FP16 WMMA and FP32 accumulation."""

    transition_bits = qvq_transition_bits(bits, vector_size=2)
    if transition_bits not in (4, 5, 6, 7):
        raise ValueError("QVQ P32 Ampere WMMA supports W2 through W3.5")
    if split_count == 0:
        if not input.is_cuda:
            raise ValueError("QVQ P32 Ampere input must be CUDA")
        autotune = _autotune_enabled()
        autotune_key = None
        if autotune:
            autotune_key = _autotune_cache_key(
                input,
                transition_bits=transition_bits,
                out_features=int(out_features),
                bank_alt_id=int(bank_alt_id),
            )
            with _AUTOTUNE_CACHE_LOCK:
                cached = _AUTOTUNE_CACHE.get(autotune_key)
            if cached is not None:
                return _QVQ_AMPERE_EXTENSION.op("p32_window")(
                    input,
                    trellis,
                    levels,
                    bank_ids,
                    transition_bits,
                    out_features,
                    bank_alt_id,
                    min(cached, int(input.shape[1]) // 16),
                )

        if split_count == 0:
            split_count = _auto_split_count(
                in_features=int(input.shape[1]),
                out_features=int(out_features),
                k_tiles=int(input.shape[1]) // 16,
                sm_count=_device_sm_count(input.device),
            )
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
    return _QVQ_AMPERE_EXTENSION.op("p32_window")(
        input,
        trellis,
        levels,
        bank_ids,
        transition_bits,
        out_features,
        bank_alt_id,
        split_count,
    )


__all__ = [
    "clear_qvq_ampere_autotune_cache",
    "qvq_p32_window_ampere",
]
