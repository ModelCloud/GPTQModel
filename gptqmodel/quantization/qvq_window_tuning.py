# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Correctness-gated window tuning shared by native and external compilers.

The caller owns device exclusivity and supplies a latency sampler. External
compilers receive the exact candidate controls, not an opaque auto policy.
Tuning inputs are for kernel validation, never rank8 fitting or quality selection.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import statistics
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from .qvq_rank8 import (
    P32WindowConfig,
    _base,
    _digest,
    _metadata,
    prepare_rank8,
    validate_rank8_state,
    window_kernel_candidates,
    window_tuning_key,
)


@dataclass(frozen=True)
class WindowTuningResult:
    config: P32WindowConfig
    report: dict
    cache_hit: bool


def measure_rank8_overhead(
    layer,
    inputs,
    *,
    benchmark,
    config=None,
):
    """Measure matched correction-off/on latency for one prepared policy.

    The benchmark callback is invoked outside CUDA graph capture with the
    same module and input for both states.  This is deliberately separate from
    candidate selection: a faster arithmetic path cannot use this measurement
    to bypass the quality/arithmetic-signature gates.  The original policy is
    restored even when timing or preparation fails.

    ``benchmark(callable, input)`` returns positive latency samples in
    microseconds.  The returned record is suitable for a scorecard and keeps
    the selected policy in both rows so external ZML tuners can compare the
    same geometry without reconstructing Python state.
    """
    if not callable(benchmark):
        raise TypeError("benchmark must be callable")
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    if not inputs or any(not isinstance(x, torch.Tensor) for x in inputs):
        raise ValueError("overhead measurement requires activation tensors")
    first = inputs[0]
    if (
        first.ndim < 2
        or first.shape[-1] != layer.in_features
        or first.numel() == 0
        or first.device != layer.runtime_device()
        or any(
            x.shape != first.shape or x.dtype != first.dtype or x.device != first.device
            for x in inputs
        )
    ):
        raise ValueError(
            "overhead cases must share a nonempty module input shape, dtype and device"
        )
    original = getattr(layer, "_p32_window_config", None)
    if original is None:
        raise ValueError("prepare the requested window policy before measuring overhead")
    if (
        getattr(layer, "rank8_metadata", None) is None
        or layer.rank8_A is None
        or layer.rank8_B is None
    ):
        raise ValueError("rank8 overhead requires validated recovery tensors")
    if config is None:
        config = original
    if not isinstance(config, P32WindowConfig):
        raise TypeError("config must be P32WindowConfig")
    if config.recovery_mode == "off":
        raise ValueError("overhead measurement requires an on-capable policy")
    # Preparation performs capture/training/device and quality checks before
    # timing.  It is intentionally called once before state changes so a cold
    # operator or stale payload fails before the benchmark starts.
    prepare_rank8(layer, config)
    m = first.numel() // layer.in_features
    base = replace(config, recovery_mode="off", min_m=m, max_m=m)
    on = replace(config, recovery_mode="on", min_m=m, max_m=m)
    try:
        rows = []
        for state, state_config in (("off", base), ("on", on)):
            prepare_rank8(layer, state_config)
            samples = [float(value) for value in benchmark(layer, first)]
            if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
                raise ValueError("benchmark must return finite positive latency samples")
            rows.append(
                {
                    "recovery": state,
                    "config": state_config.to_backend_config(),
                    "samples_us": samples,
                    "median_us": statistics.median(samples),
                }
            )
        off_median = rows[0]["median_us"]
        on_median = rows[1]["median_us"]
        return {
            "version": 1,
            "m": m,
            "input_shape": list(first.shape),
            "input_dtype": str(first.dtype),
            "off": rows[0],
            "on": rows[1],
            "overhead_us": on_median - off_median,
            "overhead_percent": (on_median / off_median - 1.0) * 100.0,
        }
    finally:
        prepare_rank8(layer, original)


def _errors(actual, reference):
    if (
        not isinstance(actual, torch.Tensor)
        or actual.shape != reference.shape
        or actual.dtype != reference.dtype
        or actual.device != reference.device
    ):
        raise ValueError(
            "candidate output shape, dtype or device differs from the reference"
        )
    error = (actual.double() - reference.double()).abs()
    finite = bool(torch.isfinite(actual).all())
    mae, maximum = error.mean().item(), error.max().item()
    norm = reference.double().norm().item()
    return {
        "finite": finite,
        "mae": mae if finite else None,
        "max": maximum if finite else None,
        "relative_l2": error.norm().item() / norm if finite and norm else None,
        "accepted": finite and mae <= 2e-3 and maximum <= 0.046875,
    }


def _cuda_driver_version(device):
    if device.type != "cuda" or torch.version.hip is not None:
        return None
    driver = ctypes.CDLL("nvcuda.dll" if os.name == "nt" else "libcuda.so.1")
    version = ctypes.c_int()
    status = driver.cuDriverGetVersion(ctypes.byref(version))
    if status:
        raise RuntimeError(f"cannot identify CUDA driver for tuning cache: {status}")
    return version.value


def tune_window_kernel(
    layer,
    inputs,
    *,
    benchmark,
    build_id,
    cache_dir=None,
    compile_candidate=None,
    candidates=None,
    apply=True,
    measure_recovery=False,
    tp_world_size=1,
    tp_rank=0,
):
    """Select the fastest locally correct implementation of a prepared policy.

    ``benchmark(callable, input)`` returns positive latency samples in us.
    ``compile_candidate(backend_config)`` optionally returns an external
    executable accepting one activation tensor. That very executable is
    validated and timed, enabling direct compiler/ZML tile selection.

    Set ``measure_recovery=True`` to append a matched correction-off/on
    marginal-latency record for the selected candidate.  This is a scorecard
    measurement only; it never changes quality eligibility.

    All input cases must have one shape/dtype/device. Cache entries bind exact
    validation inputs, deployment state, candidate set, hardware and compiler
    build. A hit revalidates the selected executable before applying it.
    Exceptions restore the previous module policy. This function is invoked
    outside capture/request execution, not on a captured graph's hot path.
    """
    if not isinstance(build_id, str) or not build_id:
        raise ValueError("a nonempty compiler/kernel build identity is required")
    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    if not inputs or any(not isinstance(x, torch.Tensor) for x in inputs):
        raise ValueError("tuning requires activation tensors")
    first = inputs[0]
    if (
        first.ndim < 2
        or first.shape[-1] != layer.in_features
        or first.numel() == 0
        or first.device != layer.runtime_device()
        or any(
            x.shape != first.shape or x.dtype != first.dtype or x.device != first.device
            for x in inputs
        )
    ):
        raise ValueError(
            "tuning cases must share a nonempty module input shape, dtype and device"
        )
    original = getattr(layer, "_p32_window_config", None)
    if original is None:
        raise ValueError("prepare the requested quality policy before tuning")
    # Also rejects graph capture, training and an active grouped sibling cycle.
    prepare_rank8(layer, original)
    enabled = bool(layer._p32_rank8_enabled)
    m = first.numel() // layer.in_features
    eligible = window_kernel_candidates(layer, m=m)
    choices = eligible if candidates is None else tuple(candidates)
    if not choices or any(c not in eligible for c in choices):
        raise ValueError(
            "candidates must belong to the prepared quality policy and shape"
        )
    choices = tuple(dict.fromkeys(choices))
    if enabled and original.quality_mode in ("balanced", "quality") and any(
        c.arithmetic_signature != "reference_fp32_v1" for c in choices
    ):
        raise ValueError(
            "unverified rank8 arithmetic cannot be selected for balanced/quality tuning"
        )
    baseline = replace(eligible[0], algorithm="production_window")
    tensors, metadata = _base(layer)
    state_hash = _digest(tensors, metadata)
    identity = {
        "key": window_tuning_key(
            layer,
            m=m,
            quality_mode=original.quality_mode,
            tp_world_size=tp_world_size,
            tp_rank=tp_rank,
            build_id=build_id,
        ),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cuda_driver": _cuda_driver_version(first.device),
        "input_shape": list(first.shape),
        "input_dtype": str(first.dtype),
        "input_strides": [list(x.stride()) for x in inputs],
        "input_hash": _digest({str(i): x for i, x in enumerate(inputs)}, {}),
        "state_hash": state_hash,
        "factors_hash": _metadata(layer)["factors_hash"] if enabled else None,
        "candidates": [c.to_backend_config() for c in choices],
        "gate": {"mae": 2e-3, "max": 0.046875, "version": 1},
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir) / (key + ".json") if cache_dir is not None else None
    report = {"identity": identity, "rows": [], "selected": None}
    selected = None
    cache_hit = False
    try:
        with torch.inference_mode():
            prepare_rank8(layer, baseline)
            references = [layer(x).clone() for x in inputs]
            if any(not bool(torch.isfinite(y).all()) for y in references):
                raise ValueError(
                    "production reference is non-finite; tuning cannot proceed"
                )

            def executable(config):
                prepare_rank8(layer, config)
                if bool(layer._p32_rank8_enabled) != enabled:
                    raise ValueError(
                        "a kernel candidate changed correction eligibility"
                    )
                return (
                    layer
                    if compile_candidate is None
                    else compile_candidate(config.to_backend_config())
                )

            if cache is not None and cache.exists():
                cached = json.loads(cache.read_text())
                if cached.get("identity") != json.loads(json.dumps(identity)):
                    raise ValueError("tuning cache identity mismatch")
                choice = P32WindowConfig.from_backend_config(cached["selected"])
                if choice not in choices:
                    raise ValueError("cached kernel is not an eligible candidate")
                fn = executable(choice)
                checks = [
                    _errors(fn(x), y) for x, y in zip(inputs, references, strict=True)
                ]
                if all(c["accepted"] for c in checks):
                    selected, report, cache_hit = choice, cached, True
                    report["cache_revalidation"] = checks
            if selected is None:
                for config in choices:
                    fn = executable(config)
                    checks = [
                        _errors(fn(x), y)
                        for x, y in zip(inputs, references, strict=True)
                    ]
                    accepted = all(c["accepted"] for c in checks)
                    row = {
                        "config": config.to_backend_config(),
                        "checks": checks,
                        "accepted": accepted,
                    }
                    # Failed candidates remain visible; timing them permits
                    # scoped human review of unusually large potential gains.
                    samples = [float(v) for v in benchmark(fn, first)]
                    if not samples or any(
                        not math.isfinite(v) or v <= 0 for v in samples
                    ):
                        raise ValueError(
                            "benchmark must return finite positive latency samples"
                        )
                    row.update(samples_us=samples, median_us=statistics.median(samples))
                    report["rows"].append(row)
                passing = [r for r in report["rows"] if r["accepted"]]
                if not passing:
                    raise ValueError(
                        "no kernel candidate passed all local correctness cases"
                    )
                winner = min(passing, key=lambda r: r["median_us"])
                selected = P32WindowConfig.from_backend_config(winner["config"])
                report["selected"] = selected.to_backend_config()
                for row in report["rows"]:
                    row["exception_review_required"] = (
                        not row["accepted"]
                        and winner["median_us"] / row["median_us"] > 1.25
                    )
            if enabled:
                validate_rank8_state(layer)
            current_tensors, current_metadata = _base(layer)
            if _digest(current_tensors, current_metadata) != state_hash:
                raise RuntimeError("window state changed during tuning")
            if cache is not None and not cache_hit:
                cache.parent.mkdir(parents=True, exist_ok=True)
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w", dir=cache.parent, delete=False
                    ) as file:
                        temporary = file.name
                        json.dump(report, file, indent=2, allow_nan=False)
                        file.write("\n")
                    os.replace(temporary, cache)
                finally:
                    if temporary is not None and os.path.exists(temporary):
                        os.unlink(temporary)
    finally:
        prepare_rank8(layer, original)
    if measure_recovery and enabled:
        report["recovery_overhead"] = measure_rank8_overhead(
            layer,
            first,
            benchmark=benchmark,
            config=selected,
        )
    if apply:
        prepare_rank8(layer, selected)
        # Keep the selected executable policy with the unified deployment
        # package.  This is metadata only: it is never part of the payload
        # hash and is not applied automatically on a different device or
        # shape.  A loader may use it as a candidate hint, then revalidate and
        # retune under its own graph/device contract.
        layer._p32_window_tuning = {
            "version": 1,
            "selected": selected.to_backend_config(),
            "quality_mode": original.quality_mode,
            "rank8_enabled": enabled,
            "identity": {
                "key": report["identity"]["key"],
                "input_shape": report["identity"]["input_shape"],
                "input_dtype": report["identity"]["input_dtype"],
                "input_strides": report["identity"]["input_strides"],
                "state_hash": report["identity"]["state_hash"],
                "factors_hash": report["identity"]["factors_hash"],
                "candidates": report["identity"]["candidates"],
                "gate": report["identity"]["gate"],
            },
            "cache_hit": cache_hit,
        }
    return WindowTuningResult(selected, report, cache_hit)
