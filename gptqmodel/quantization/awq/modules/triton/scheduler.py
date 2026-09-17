# SPDX-License-Identifier: Apache-2.0
"""Small, side-effect free AWQ Triton dispatch policy.

This module intentionally does not import Triton.  The policy can therefore be
tested on CPU and, importantly, selecting a plan never compiles or benchmarks a
kernel.  A small exact-shape table is enabled only for a measured device
profile; applications and benchmarks may install additional measured rules.
Everything else falls back to the historical dispatch.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import os
from threading import RLock
from typing import Any

import torch


# Keep these sets aligned with the packing and kernel layouts we have tested.
SUPPORTED_GROUP_SIZES = (32, 64, 128)
SUPPORTED_SPLITS = (1, 2, 4, 8, 16, 32)
MAX_PLAN_CACHE = 256
DEFAULT_SCHEDULE_MODE = os.getenv("GPTQMODEL_AWQ_TRITON_SCHEDULE", "auto").lower()


@dataclass(frozen=True)
class AwqTritonPlan:
    """A complete immutable dispatch decision."""

    path: str
    block_size_m: int = 32
    block_size_n: int = 32
    block_size_k: int = 32
    split_k_iters: int = 8
    num_warps: int | None = None
    # Triton's implicit launch default for this kernel is three stages.
    num_stages: int | None = None

    @property
    def fused(self) -> bool:
        return self.path == "fused"

    def as_kwargs(self) -> dict[str, int]:
        result = {
            "block_size_m": self.block_size_m,
            "block_size_n": self.block_size_n,
            "block_size_k": self.block_size_k,
            "split_k_iters": self.split_k_iters,
        }
        if self.num_warps is not None:
            result["num_warps"] = self.num_warps
        if self.num_stages is not None:
            result["num_stages"] = self.num_stages
        return result


def legacy_plan(M: int, K: int | None = None, N: int | None = None, group_size: int | None = None) -> AwqTritonPlan:
    """Historical policy (the boundary is deliberately strict)."""
    if M > 128:
        return AwqTritonPlan(path="dense")
    # Keep the public legacy/training contract byte-for-byte: fixed 32 tiles
    # and split-K=8. Candidate generation separately rejects zero-work splits.
    return AwqTritonPlan(path="fused")


def resolve_group_size(group_size: int, K: int) -> int:
    """Resolve the public ``-1`` spelling before candidate generation."""
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    actual = K if group_size == -1 else int(group_size)
    if actual <= 0:
        raise ValueError(f"group_size must be -1 or positive, got {group_size}")
    return actual


def validate_fused_config(
    *, M: int, N: int, K: int, group_size: int, block_size_m: int,
    block_size_n: int, block_size_k: int, split_k_iters: int,
    num_warps: int | None = None, num_stages: int | None = None,
    allow_empty_splits: bool = False,
) -> tuple[bool, str | None]:
    """Validate constraints imposed by the current AWQ packing/kernel."""
    try:
        G = resolve_group_size(group_size, K)
    except ValueError as exc:
        return False, str(exc)
    check_warps = 4 if num_warps is None else num_warps
    check_stages = 3 if num_stages is None else num_stages
    values = (M, N, K, block_size_m, block_size_n, block_size_k, check_warps, check_stages)
    if any(int(v) <= 0 for v in values):
        return False, "shape, tile, warps and stages must be positive"
    if K % G:
        return False, "K must be divisible by group_size"
    if G not in SUPPORTED_GROUP_SIZES and G != K:
        return False, f"group_size {G} is not supported by the AWQ Triton kernel"
    # The unpacked dot layout is in groups of eight output channels.  Keep the
    # candidate tile set explicit: other constexpr shapes have not been
    # validated against the current tl.interleave/dot layout.
    if N % 8:
        return False, "N must be divisible by 8 for 4-bit packing"
    if block_size_m not in (16, 32, 64):
        return False, "unsupported BLOCK_SIZE_M"
    if block_size_n not in (32, 64, 128):
        return False, "unsupported BLOCK_SIZE_N"
    if block_size_k not in (32, 64):
        return False, "unsupported BLOCK_SIZE_K"
    if block_size_n < 8 or block_size_n % 8:
        return False, "BLOCK_SIZE_N must be a positive multiple of 8"
    if block_size_k > G or G % block_size_k:
        return False, "BLOCK_SIZE_K must be <= G and divide G"
    if split_k_iters not in SUPPORTED_SPLITS:
        return False, "split_k_iters must be a power of two in [1, 32]"
    # Every split starts at pid_z * BLOCK_SIZE_K.  This rejects CTAs whose
    # first K tile is entirely outside K (a silent zero-work split).
    if not allow_empty_splits and split_k_iters > (K + block_size_k - 1) // block_size_k:
        return False, "split_k_iters creates an empty K split"
    if check_warps not in (1, 2, 4, 8) or check_stages not in (1, 2, 3, 4, 5):
        return False, "unsupported Triton launch parameters"
    return True, None


def valid_fused_config(**kwargs: Any) -> bool:
    return validate_fused_config(**kwargs)[0]


def candidate_plans(M: int, N: int, K: int, group_size: int) -> tuple[list[AwqTritonPlan], dict[str, int]]:
    """Return the finite offline-search space and aggregate rejection reasons."""
    G = resolve_group_size(group_size, K)
    # Keep both complete paths in the offline joint search.  Dense cost is
    # measured by the benchmark as dequantization plus matmul.
    candidates: list[AwqTritonPlan] = [AwqTritonPlan("dense")]
    baseline = AwqTritonPlan("fused", 32, 32, 32, 8, None, None)
    if validate_fused_config(
        M=M, N=N, K=K, group_size=G, block_size_m=32, block_size_n=32,
        block_size_k=32, split_k_iters=8,
    )[0]:
        candidates.append(baseline)
    rejected: dict[str, int] = {}
    for bm in (16, 32, 64):
        for bn in (32, 64, 128):
            for bk in (32, 64):
                for split in (1, 2, 4, 8):
                    for warps in (4, 8):
                        for stages in (2, 4):
                            if bm > max(32, 2 * M):
                                rejected["tile has no useful M work"] = rejected.get("tile has no useful M work", 0) + 1
                                continue
                            if warps == 8 and bm * bn < 4096:
                                rejected["8 warps underfilled tile"] = rejected.get("8 warps underfilled tile", 0) + 1
                                continue
                            ok, reason = validate_fused_config(
                                M=M, N=N, K=K, group_size=G,
                                block_size_m=bm, block_size_n=bn, block_size_k=bk,
                                split_k_iters=split, num_warps=warps, num_stages=stages,
                            )
                            if ok:
                                candidates.append(AwqTritonPlan("fused", bm, bn, bk, split, warps, stages))
                            else:
                                rejected[reason or "invalid"] = rejected.get(reason or "invalid", 0) + 1
    return candidates, rejected


@lru_cache(maxsize=16)
def _device_identity_cached(device_type: str, index: int | None) -> tuple[Any, ...]:
    name: str | None = None
    capability: tuple[int, int] | None = None
    sm_count: int | None = None
    total_memory: int | None = None
    if device_type == "cuda" and torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(index)
            capability = tuple(torch.cuda.get_device_capability(index))
            props = torch.cuda.get_device_properties(index)
            sm_count = int(getattr(props, "multi_processor_count", 0))
            total_memory = int(getattr(props, "total_memory", 0))
        except Exception:
            # A mocked or unavailable runtime is unverified by design.
            pass
    return (device_type, index, name, capability, sm_count, total_memory)


def device_identity(device: torch.device | str | None) -> tuple[Any, ...]:
    """A bounded, cache-safe identity that cannot cross CUDA devices."""
    d = torch.device(device) if device is not None else torch.device("cuda")
    index = d.index
    if d.type == "cuda" and index is None and torch.cuda.is_available():
        index = torch.cuda.current_device()
    return _device_identity_cached(d.type, index)


def _dtype_key(dtype: torch.dtype | None) -> str | None:
    return str(dtype) if dtype is not None else None


_PLAN_CACHE: OrderedDict[tuple[Any, ...], AwqTritonPlan] = OrderedDict()
# Offline measurements on the exact device profile below.  These are exact
# shapes on purpose: neighboring M values and dense crossovers were not stable
# enough to justify buckets.  Unlisted devices and shapes retain legacy policy.
_VERIFIED_4090_PROFILE = (
    "NVIDIA GeForce RTX 4090", (8, 9), 128, 50_950_569_984,
)
_VERIFIED_4090_FP16_RULES: dict[tuple[int, int, int, int], AwqTritonPlan] = {
    # (M, N, K, G): joint fused/dense plan
    (32, 8192, 2048, 128): AwqTritonPlan("fused", 16, 64, 64, 1, 4, 4),
    (33, 8192, 2048, 128): AwqTritonPlan("fused", 32, 64, 64, 1, 4, 2),
    (33, 4096, 4096, 128): AwqTritonPlan("fused", 64, 32, 64, 4, 4, 2),
    (64, 8192, 2048, 128): AwqTritonPlan("fused", 64, 64, 64, 1, 4, 2),
    (64, 4096, 4096, 128): AwqTritonPlan("fused", 64, 64, 64, 4, 8, 2),
    (64, 2048, 8192, 128): AwqTritonPlan("fused", 64, 64, 64, 8, 8, 2),
    (127, 2048, 2048, 128): AwqTritonPlan("fused", 32, 32, 64, 1, 4, 4),
    (127, 8192, 2048, 128): AwqTritonPlan("fused", 64, 64, 32, 1, 4, 2),
    (127, 4104, 4096, 32): AwqTritonPlan("dense"),
    (127, 2048, 8192, 128): AwqTritonPlan("fused", 64, 64, 64, 4, 8, 2),
    (128, 2048, 2048, 128): AwqTritonPlan("fused", 32, 32, 64, 1, 4, 4),
    (128, 8192, 2048, 128): AwqTritonPlan("fused", 64, 64, 32, 1, 4, 2),
    (128, 4096, 4096, 128): AwqTritonPlan("dense"),
    (128, 4104, 4096, 32): AwqTritonPlan("fused", 64, 64, 32, 2, 4, 2),
}
# Applications and the offline benchmark may additionally install exact rules.
_MEASURED_RULES: dict[tuple[Any, ...], AwqTritonPlan] = {}
_WARMED_PLANS: OrderedDict[tuple[Any, ...], AwqTritonPlan] = OrderedDict()
_CACHE_LOCK = RLock()


def _plan_key(
    *, M: int, N: int, K: int, G: int, device: torch.device | str | None,
    input_dtype: torch.dtype | None, output_dtype: torch.dtype | None,
    compute_dtype: torch.dtype | None, fp32_accum: bool,
) -> tuple[Any, ...]:
    return (
        device_identity(device), M, N, K, G, _dtype_key(input_dtype),
        _dtype_key(compute_dtype), _dtype_key(output_dtype), bool(fp32_accum),
    )


def _warmup_key(
    *, M: int, N: int, K: int, G: int, device: torch.device | str | None,
    input_dtype: torch.dtype | None, compute_dtype: torch.dtype | None,
    output_dtype: torch.dtype | None, fp32_accum: bool,
    request: tuple[Any, ...],
) -> tuple[Any, ...]:
    # This deliberately avoids device-property queries so capture lookup is a
    # pure in-memory operation. CUDA tensors carry a resolved device index.
    d = torch.device(device) if device is not None else torch.device("cuda")
    index = d.index
    if d.type == "cuda" and index is None and torch.cuda.is_available():
        index = torch.cuda.current_device()
    return (
        d.type, index, M, N, K, G, _dtype_key(input_dtype),
        _dtype_key(compute_dtype), _dtype_key(output_dtype), bool(fp32_accum), request,
    )


def _request_fingerprint(
    *, mode: str, explicit: AwqTritonPlan | dict[str, Any] | None,
    training: bool,
) -> tuple[Any, ...]:
    # Cache identity includes the caller's mode so explicit and legacy requests
    # cannot accidentally reuse an auto-selected plan.
    if training:
        return ("training",)
    if explicit is not None:
        plan = AwqTritonPlan(**explicit) if isinstance(explicit, dict) else explicit
        return ("explicit", plan)
    return (mode,)


def _builtin_plan(
    identity: tuple[Any, ...], *, M: int, N: int, K: int, G: int,
    input_dtype: torch.dtype | None, compute_dtype: torch.dtype | None,
    output_dtype: torch.dtype | None, fp32_accum: bool,
) -> AwqTritonPlan | None:
    profile = identity[2:] if len(identity) >= 6 else ()
    if (
        profile != _VERIFIED_4090_PROFILE
        or input_dtype != torch.float16
        or compute_dtype != torch.float16
        or output_dtype != torch.float16
        or not fp32_accum
    ):
        return None
    return _VERIFIED_4090_FP16_RULES.get((M, N, K, G))


def clear_awq_triton_plan_cache() -> None:
    with _CACHE_LOCK:
        _PLAN_CACHE.clear()
        _device_identity_cached.cache_clear()


def clear_awq_triton_warmup_cache() -> None:
    with _CACHE_LOCK:
        _WARMED_PLANS.clear()


def clear_awq_triton_rules() -> None:
    """Clear runtime-registered rules and caches; verified built-ins remain."""
    with _CACHE_LOCK:
        _MEASURED_RULES.clear()
        _PLAN_CACHE.clear()
        _WARMED_PLANS.clear()
        _device_identity_cached.cache_clear()


def register_awq_triton_rule(
    device: torch.device | str, plan: AwqTritonPlan, *, M: int, N: int,
    K: int, group_size: int, input_dtype: torch.dtype | None,
    output_dtype: torch.dtype | None, compute_dtype: torch.dtype | None = None,
    fp32_accum: bool = True,
) -> None:
    """Install a measured exact-shape/device rule.

    Exact keys prevent one benchmark's M bucket, dtype, or group size from
    leaking into another dispatch decision.
    """
    G = resolve_group_size(group_size, K)
    if plan.path not in ("dense", "fused"):
        raise ValueError(f"unknown AWQ Triton path: {plan.path}")
    if plan.fused:
        ok, reason = validate_fused_config(
            M=M, N=N, K=K, group_size=G,
            block_size_m=plan.block_size_m, block_size_n=plan.block_size_n,
            block_size_k=plan.block_size_k, split_k_iters=plan.split_k_iters,
            num_warps=plan.num_warps, num_stages=plan.num_stages,
        )
        if not ok:
            raise ValueError(f"invalid measured AWQ Triton rule: {reason}")
    key = _plan_key(
        M=M, N=N, K=K, G=G, device=device, input_dtype=input_dtype,
        compute_dtype=compute_dtype, output_dtype=output_dtype, fp32_accum=fp32_accum,
    )
    with _CACHE_LOCK:
        _MEASURED_RULES[key] = plan
        _PLAN_CACHE.clear()
        _WARMED_PLANS.clear()


def mark_awq_triton_plan_warmed(
    plan: AwqTritonPlan, *, M: int, N: int, K: int, group_size: int,
    device: torch.device | str | None, input_dtype: torch.dtype | None,
    compute_dtype: torch.dtype | None, output_dtype: torch.dtype | None,
    fp32_accum: bool, mode: str | None = None,
    explicit: AwqTritonPlan | dict[str, Any] | None = None,
    training: bool = False,
) -> None:
    """Record a plan only after its complete eager path launched successfully."""
    G = resolve_group_size(group_size, K)
    resolved_mode = DEFAULT_SCHEDULE_MODE if mode is None else mode.lower()
    key = _warmup_key(
        M=M, N=N, K=K, G=G, device=device, input_dtype=input_dtype,
        compute_dtype=compute_dtype, output_dtype=output_dtype,
        fp32_accum=fp32_accum,
        request=_request_fingerprint(
            mode=resolved_mode, explicit=explicit, training=training,
        ),
    )
    if _WARMED_PLANS.get(key) == plan:
        return
    with _CACHE_LOCK:
        if _WARMED_PLANS.get(key) == plan:
            return
        if len(_WARMED_PLANS) >= MAX_PLAN_CACHE and key not in _WARMED_PLANS:
            _WARMED_PLANS.popitem(last=False)
        _WARMED_PLANS[key] = plan
        _WARMED_PLANS.move_to_end(key)


def _explicit_plan(plan: AwqTritonPlan | dict[str, Any], M: int, N: int, K: int, G: int) -> AwqTritonPlan:
    if isinstance(plan, dict):
        plan = AwqTritonPlan(**plan)
    if not isinstance(plan, AwqTritonPlan):
        raise TypeError("AWQ Triton schedule must be AwqTritonPlan or a mapping")
    if plan.path == "dense":
        return plan
    if plan.path != "fused":
        raise ValueError(f"unknown AWQ Triton path: {plan.path}")
    ok, reason = validate_fused_config(
        M=M, N=N, K=K, group_size=G, block_size_m=plan.block_size_m,
        block_size_n=plan.block_size_n, block_size_k=plan.block_size_k,
        split_k_iters=plan.split_k_iters, num_warps=plan.num_warps,
        num_stages=plan.num_stages,
    )
    if not ok:
        raise ValueError(f"invalid AWQ Triton explicit schedule: {reason}")
    return plan


def select_awq_triton_plan(
    *, M: int, N: int, K: int, group_size: int,
    device: torch.device | str | None = None,
    input_dtype: torch.dtype | None = None,
    compute_dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
    fp32_accum: bool = True,
    mode: str | None = None,
    explicit: AwqTritonPlan | dict[str, Any] | None = None,
    training: bool = False,
    cuda_graph: bool = False,
) -> AwqTritonPlan:
    """Select a plan without synchronization, compilation, or benchmarking."""
    if M <= 0:
        raise ValueError("plan selection is not defined for empty input")
    G = resolve_group_size(group_size, K)
    mode = DEFAULT_SCHEDULE_MODE if mode is None else mode.lower()
    if mode not in ("auto", "legacy"):
        raise ValueError("AWQ Triton schedule mode must be 'auto' or 'legacy'")
    if cuda_graph:
        # Capture must be a lookup only: discovery, tuning, and compilation are
        # performed during the explicit warmup phase before graph capture.
        if explicit is not None and not training:
            _explicit_plan(explicit, M, N, K, G)
        key = _warmup_key(
            M=M, N=N, K=K, G=G, device=device, input_dtype=input_dtype,
            compute_dtype=compute_dtype, output_dtype=output_dtype,
            fp32_accum=fp32_accum,
            request=_request_fingerprint(
                mode=mode, explicit=explicit, training=training,
            ),
        )
        with _CACHE_LOCK:
            plan = _WARMED_PLANS.get(key)
        if plan is None:
            raise RuntimeError(
                "AWQ Triton CUDA Graph capture requires an eager prewarm for the exact plan"
            )
        return plan
    # Training deliberately keeps known backward-compatible scheduling, even
    # if an inference plan was supplied by a caller.
    if training:
        return legacy_plan(M, K, N, G)
    if explicit is not None:
        return _explicit_plan(explicit, M, N, K, G)
    if mode == "legacy":
        return legacy_plan(M, K, N, G)

    d = device if isinstance(device, torch.device) else torch.device(device or "cuda")
    index = d.index
    if d.type == "cuda" and index is None and torch.cuda.is_available():
        index = torch.cuda.current_device()
    cache_key = (
        d.type, index, M, N, K, G, input_dtype, compute_dtype,
        output_dtype, bool(fp32_accum),
    )
    # CPython protects individual mapping reads in free-threaded builds.  Keep
    # the overwhelmingly common hit path read-only; misses and all mutations
    # remain serialized below.  FIFO eviction is sufficient for this tiny,
    # bounded shape cache and avoids mutating the OrderedDict on every token.
    cached = _PLAN_CACHE.get(cache_key)
    if cached is not None:
        return cached
    with _CACHE_LOCK:
        cached = _PLAN_CACHE.get(cache_key)
        if cached is not None:
            return cached
        # Exact device rules are the only source of auto optimization. Unknown
        # devices and shapes retain the old M boundary and tile configuration.
        rule_key = _plan_key(
            M=M, N=N, K=K, G=G, device=device, input_dtype=input_dtype,
            compute_dtype=compute_dtype, output_dtype=output_dtype,
            fp32_accum=fp32_accum,
        )
        plan = _MEASURED_RULES.get(rule_key)
        if plan is None:
            plan = _builtin_plan(
                rule_key[0], M=M, N=N, K=K, G=G,
                input_dtype=input_dtype, compute_dtype=compute_dtype,
                output_dtype=output_dtype, fp32_accum=fp32_accum,
            )
        if plan is None:
            plan = legacy_plan(M, K, N, G)
        if plan.fused:
            ok, _ = validate_fused_config(
                M=M, N=N, K=K, group_size=G, block_size_m=plan.block_size_m,
                block_size_n=plan.block_size_n, block_size_k=plan.block_size_k,
                split_k_iters=plan.split_k_iters, num_warps=plan.num_warps,
                num_stages=plan.num_stages,
            )
            if not ok:
                plan = legacy_plan(M, K, N, G)
        if len(_PLAN_CACHE) >= MAX_PLAN_CACHE:
            _PLAN_CACHE.popitem(last=False)
        _PLAN_CACHE[cache_key] = plan
        return plan


__all__ = [
    "AwqTritonPlan", "SUPPORTED_GROUP_SIZES", "SUPPORTED_SPLITS",
    "legacy_plan", "resolve_group_size", "validate_fused_config",
    "valid_fused_config", "candidate_plans", "select_awq_triton_plan",
    "register_awq_triton_rule", "clear_awq_triton_plan_cache",
    "clear_awq_triton_rules", "clear_awq_triton_warmup_cache",
    "mark_awq_triton_plan_warmed", "device_identity",
]
