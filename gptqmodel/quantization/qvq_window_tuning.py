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
    grouped_window_kernel_candidates,
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


@dataclass(frozen=True)
class GroupedWindowTuningResult:
    """Selection result for one complete grouped child-policy tuple."""

    configs: tuple[P32WindowConfig, ...]
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


def _write_tuning_cache(cache, report):
    """Atomically publish a complete tuning report after optional measurements."""
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
    measure_recovery_candidates=False,
    max_recovery_overhead_percent=None,
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

    Set ``measure_recovery_candidates=True`` to collect that same paired
    measurement for every candidate row.  This is useful when BM/BN winners
    differ between correction states: the report retains both medians for each
    geometry while selection still uses only the validated arithmetic and
    output-correctness gates.

    ``max_recovery_overhead_percent`` is an explicit promotion gate.  When
    supplied for an enabled rank8 policy, per-candidate measurements are
    required and candidates above the limit are excluded from selection.  It
    is deliberately opt-in because the target is not met by every module or
    shape yet.

    All input cases must have one shape/dtype/device. Cache entries bind exact
    validation inputs, deployment state, candidate set, hardware and compiler
    build. A hit revalidates the selected executable before applying it.
    Exceptions restore the previous module policy. This function is invoked
    outside capture/request execution, not on a captured graph's hot path.
    """
    if not isinstance(build_id, str) or not build_id:
        raise ValueError("a nonempty compiler/kernel build identity is required")
    if not isinstance(measure_recovery, bool) or not isinstance(
        measure_recovery_candidates, bool
    ):
        raise TypeError("recovery measurement flags must be bool")
    if max_recovery_overhead_percent is not None and (
        not isinstance(max_recovery_overhead_percent, (int, float))
        or isinstance(max_recovery_overhead_percent, bool)
        or not math.isfinite(float(max_recovery_overhead_percent))
        or float(max_recovery_overhead_percent) < 0
    ):
        raise ValueError("max_recovery_overhead_percent must be finite and nonnegative")
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
    if enabled and max_recovery_overhead_percent is not None and not measure_recovery_candidates:
        raise ValueError(
            "an overhead promotion gate requires measure_recovery_candidates=True"
        )
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
        "measure_recovery_candidates": measure_recovery_candidates,
        "max_recovery_overhead_percent": (
            None
            if max_recovery_overhead_percent is None
            else float(max_recovery_overhead_percent)
        ),
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

            def executable(config, *, expected_enabled=enabled):
                prepare_rank8(layer, config)
                if bool(layer._p32_rank8_enabled) != expected_enabled:
                    raise ValueError(
                        "a kernel candidate changed correction eligibility"
                    )
                return (
                    layer
                    if compile_candidate is None
                    else compile_candidate(config.to_backend_config())
                )

            def benchmark_samples(fn):
                samples = [float(v) for v in benchmark(fn, first)]
                if not samples or any(
                    not math.isfinite(v) or v <= 0 for v in samples
                ):
                    raise ValueError(
                        "benchmark must return finite positive latency samples"
                    )
                return samples

            def recovery_pair(config):
                if not enabled:
                    raise RuntimeError(
                        "recovery candidate measurement requires enabled factors"
                    )
                off_config = replace(config, recovery_mode="off")
                on_config = replace(config, recovery_mode="on")
                off_fn = executable(off_config, expected_enabled=False)
                off_samples = benchmark_samples(off_fn)
                on_fn = executable(on_config, expected_enabled=True)
                on_samples = benchmark_samples(on_fn)
                off_median = statistics.median(off_samples)
                on_median = statistics.median(on_samples)
                return {
                    "off": {
                        "config": off_config.to_backend_config(),
                        "samples_us": off_samples,
                        "median_us": off_median,
                    },
                    "on": {
                        "config": on_config.to_backend_config(),
                        "samples_us": on_samples,
                        "median_us": on_median,
                    },
                    "overhead_us": on_median - off_median,
                    "overhead_percent": (on_median / off_median - 1.0) * 100.0,
                }

            def overhead_is_eligible(overhead_percent):
                if max_recovery_overhead_percent is None:
                    return True
                # Timing medians are floating-point values; allow one ulp-scale
                # tolerance at an exact boundary such as a 5% promotion cap.
                limit = float(max_recovery_overhead_percent)
                tolerance = max(1e-9, abs(limit) * 1e-9)
                return overhead_percent <= limit + tolerance

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
                cached_row = next(
                    (
                        row
                        for row in cached.get("rows", [])
                        if row.get("config") == choice.to_backend_config()
                    ),
                    None,
                )
                cached_overhead_ok = (
                    not enabled
                    or max_recovery_overhead_percent is None
                    or bool(cached_row and cached_row.get("recovery_overhead_eligible"))
                )
                if all(c["accepted"] for c in checks) and cached_overhead_ok:
                    selected, report, cache_hit = choice, cached, True
                    report["cache_revalidation"] = checks
                elif max_recovery_overhead_percent is not None and enabled:
                    # A cache made without a passing marginal-cost record must
                    # not silently bypass the promotion gate. Rebuild a clean
                    # report so newly measured rows are authoritative.
                    report = {"identity": identity, "rows": [], "selected": None}
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
                    samples = benchmark_samples(fn)
                    row.update(samples_us=samples, median_us=statistics.median(samples))
                    if measure_recovery_candidates and enabled:
                        row["recovery_overhead"] = recovery_pair(config)
                        row["recovery_overhead_eligible"] = overhead_is_eligible(
                            row["recovery_overhead"]["overhead_percent"]
                        )
                    report["rows"].append(row)
                passing = [
                    r
                    for r in report["rows"]
                    if r["accepted"]
                    and (
                        not enabled
                        or max_recovery_overhead_percent is None
                        or r.get("recovery_overhead_eligible", False)
                    )
                ]
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
    finally:
        prepare_rank8(layer, original)
    if measure_recovery and enabled:
        report["recovery_overhead"] = measure_rank8_overhead(
            layer,
            first,
            benchmark=benchmark,
            config=selected,
        )
    if cache is not None and (not cache_hit or measure_recovery):
        _write_tuning_cache(cache, report)
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
            "max_recovery_overhead_percent": (
                None
                if max_recovery_overhead_percent is None
                else float(max_recovery_overhead_percent)
            ),
            # Keep matched correction-state timing with the selected policy so
            # an external ZML consumer can make the same shape-specific cost
            # visible without rerunning Python.  These measurements never
            # establish quality eligibility.
            "recovery_overhead": report.get("recovery_overhead"),
            "candidate_recovery_overhead": [
                {
                    "config": row["config"],
                    "recovery_overhead": row["recovery_overhead"],
                    **(
                        {"recovery_overhead_eligible": row["recovery_overhead_eligible"]}
                        if "recovery_overhead_eligible" in row
                        else {}
                    ),
                }
                for row in report.get("rows", [])
                if "recovery_overhead" in row
            ],
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


def tune_grouped_window_kernel(
    layers,
    inputs,
    *,
    benchmark,
    build_id,
    cache_dir=None,
    compile_candidate=None,
    candidates=None,
    apply=True,
    measure_recovery_candidates=False,
    max_recovery_overhead_percent=None,
    tp_world_size=1,
    tp_rank=0,
):
    """Tune one complete grouped P32 policy outside graph capture.

    ``candidates`` and the returned ``configs`` are tuples in child order.
    ``compile_candidate`` receives a tuple of backend-config dictionaries and
    must return a callable accepting one activation tensor.  Without it, the
    callable prepares each child and invokes the grouped wrappers in order.
    Correctness is checked for every child and input document before timing;
    an optional recovery budget requires matched off/on timings for every
    candidate.  The grouped identity and selected tuple are cached together,
    so one child's split winner cannot be reused with another grouped shape.
    """

    children = tuple(layers)
    if len(children) not in (2, 3):
        raise ValueError("grouped tuning requires two or three children")
    if not isinstance(build_id, str) or not build_id:
        raise ValueError("a nonempty compiler/kernel build identity is required")
    if not callable(benchmark):
        raise TypeError("benchmark must be callable")
    if not isinstance(measure_recovery_candidates, bool):
        raise TypeError("measure_recovery_candidates must be bool")
    if max_recovery_overhead_percent is not None and (
        not isinstance(max_recovery_overhead_percent, (int, float))
        or isinstance(max_recovery_overhead_percent, bool)
        or not math.isfinite(float(max_recovery_overhead_percent))
        or float(max_recovery_overhead_percent) < 0
    ):
        raise ValueError("max_recovery_overhead_percent must be finite and nonnegative")

    inputs = (inputs,) if isinstance(inputs, torch.Tensor) else tuple(inputs)
    if not inputs or any(not isinstance(value, torch.Tensor) for value in inputs):
        raise ValueError("grouped tuning requires activation tensors")
    first = inputs[0]
    if first.ndim < 2 or first.numel() == 0:
        raise ValueError("grouped tuning requires nonempty rank-2-or-higher inputs")
    reference = children[0]
    if first.shape[-1] != reference.in_features:
        raise ValueError("grouped input width does not match the first child")
    for child in children:
        if child.in_features != reference.in_features:
            raise ValueError("grouped children must have the same input width")
        if child.runtime_device() != reference.runtime_device():
            raise ValueError("grouped children must share one runtime device")
    if any(
        value.shape != first.shape
        or value.dtype != first.dtype
        or value.device != first.device
        for value in inputs
    ):
        raise ValueError("grouped tuning inputs must share shape, dtype and device")
    if first.device != reference.runtime_device():
        raise ValueError("grouped tuning inputs must use the runtime device")

    originals = tuple(getattr(child, "_p32_window_config", None) for child in children)
    if any(config is None for config in originals):
        raise ValueError("prepare every grouped child policy before tuning")
    originals = tuple(originals)
    for child, config in zip(children, originals, strict=True):
        prepare_rank8(child, config)
    enabled = tuple(bool(child._p32_rank8_enabled) for child in children)
    state_hashes = tuple(_digest(*_base(child)) for child in children)
    if max_recovery_overhead_percent is not None and any(enabled) and not measure_recovery_candidates:
        raise ValueError(
            "an overhead promotion gate requires measure_recovery_candidates=True"
        )

    m = first.numel() // reference.in_features
    eligible = grouped_window_kernel_candidates(children, m=m)
    choices = eligible if candidates is None else tuple(candidates)
    if not choices:
        raise ValueError("grouped candidate enumeration returned no policies")
    eligible_set = set(eligible)
    normalized_choices = []
    for choice in choices:
        choice = tuple(choice)
        if len(choice) != len(children) or any(
            not isinstance(config, P32WindowConfig) for config in choice
        ):
            raise TypeError("grouped candidates must be P32WindowConfig tuples")
        if choice not in eligible_set:
            raise ValueError("grouped candidates must come from the prepared shape policy")
        if choice not in normalized_choices:
            normalized_choices.append(choice)
    choices = tuple(normalized_choices)
    if not choices:
        raise ValueError("grouped candidates must not be empty")
    if any(
        enabled[index]
        and originals[index].quality_mode in ("balanced", "quality")
        and config.arithmetic_signature != "reference_fp32_v1"
        for choice in choices
        for index, config in enumerate(choice)
    ):
        raise ValueError("unverified rank8 arithmetic cannot be selected for balanced/quality tuning")

    def backend_choice(choice):
        return tuple(config.to_backend_config() for config in choice)

    identity = {
        "version": 1,
        "build_id": build_id,
        "m": m,
        "input_shape": list(first.shape),
        "input_dtype": str(first.dtype),
        "input_strides": [list(value.stride()) for value in inputs],
        "input_hash": _digest({str(i): value for i, value in enumerate(inputs)}, {}),
        "children": [
            window_tuning_key(
                child,
                m=m,
                quality_mode=config.quality_mode,
                tp_world_size=tp_world_size,
                tp_rank=tp_rank,
                build_id=build_id,
            )
            for child, config in zip(children, originals, strict=True)
        ],
        "enabled": list(enabled),
        "candidates": [backend_choice(choice) for choice in choices],
        "gate": {"mae": 2e-3, "max": 0.046875, "version": 1},
        "measure_recovery_candidates": measure_recovery_candidates,
        "max_recovery_overhead_percent": (
            None
            if max_recovery_overhead_percent is None
            else float(max_recovery_overhead_percent)
        ),
    }
    cache_key = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=list).encode()
    ).hexdigest()
    cache = Path(cache_dir) / ("grouped-" + cache_key + ".json") if cache_dir is not None else None
    report = {"identity": identity, "rows": [], "selected": None}
    selected = None
    cache_hit = False

    def prepare_choice(choice):
        for child, config in zip(children, choice, strict=True):
            prepare_rank8(child, config)

    def executable_choice(choice):
        prepare_choice(choice)
        if compile_candidate is not None:
            function = compile_candidate(backend_choice(choice))
            if not callable(function):
                raise TypeError("compile_candidate must return a callable")
            return function
        return lambda value: tuple(child(value) for child in children)

    def validate_outputs(actual, expected):
        actual = tuple(actual) if isinstance(actual, (tuple, list)) else None
        if actual is None or len(actual) != len(children):
            raise ValueError("grouped candidate must return one output per child")
        checks = []
        for got, want in zip(actual, expected, strict=True):
            checks.append(_errors(got, want))
        return checks

    def samples_for(function):
        samples = [float(value) for value in benchmark(function, first)]
        if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
            raise ValueError("benchmark must return finite positive latency samples")
        return samples

    def overhead_ok(value):
        if max_recovery_overhead_percent is None:
            return True
        tolerance = max(1e-9, abs(float(max_recovery_overhead_percent)) * 1e-9)
        return value <= float(max_recovery_overhead_percent) + tolerance

    try:
        with torch.inference_mode():
            baseline = tuple(
                replace(
                    config,
                    algorithm="production_window",
                    recovery_mode="off",
                    recovery_kernel="separate_reference",
                    recovery_projection="separate_reference",
                    arithmetic_signature="reference_fp32_v1",
                    split_k=1,
                    block_m=0,
                    block_n=0,
                    warp_groups=0,
                    chunk_m=0,
                    min_m=1,
                    max_m=8192,
                )
                for config in originals
            )
            baseline_fn = executable_choice(baseline)
            references = [tuple(baseline_fn(value)) for value in inputs]

            if cache is not None and cache.exists():
                cached = json.loads(cache.read_text())
                if cached.get("identity") != json.loads(json.dumps(identity, default=list)):
                    raise ValueError("grouped tuning cache identity mismatch")
                cached_choice = tuple(
                    P32WindowConfig.from_backend_config(config)
                    for config in cached.get("selected", [])
                )
                cached_row = next(
                    (
                        row
                        for row in cached.get("rows", [])
                        if row.get("config") == list(backend_choice(cached_choice))
                    ),
                    None,
                )
                if cached_choice in choices:
                    cached_fn = executable_choice(cached_choice)
                    checks = [
                        validate_outputs(cached_fn(value), expected)
                        for value, expected in zip(inputs, references, strict=True)
                    ]
                    cached_overhead_ok = not any(enabled) or max_recovery_overhead_percent is None or bool(
                        cached_row and cached_row.get("recovery_overhead_eligible")
                    )
                    if all(check[child_index]["accepted"] for check in checks for child_index in range(len(children))) and cached_overhead_ok:
                        selected, report, cache_hit = cached_choice, cached, True
                        report["cache_revalidation"] = checks

            if selected is None:
                for choice in choices:
                    candidate_fn = executable_choice(choice)
                    checks = [
                        validate_outputs(candidate_fn(value), expected)
                        for value, expected in zip(inputs, references, strict=True)
                    ]
                    row = {
                        "config": backend_choice(choice),
                        "checks": checks,
                        "accepted": all(
                            check[child_index]["accepted"]
                            for check in checks
                            for child_index in range(len(children))
                        ),
                    }
                    row["samples_us"] = samples_for(candidate_fn)
                    row["median_us"] = statistics.median(row["samples_us"])
                    if measure_recovery_candidates and any(enabled):
                        off = tuple(replace(config, recovery_mode="off") for config in choice)
                        on = tuple(
                            replace(config, recovery_mode="on" if enabled[index] else "off")
                            for index, config in enumerate(choice)
                        )
                        off_samples = samples_for(
                            executable_choice(off)
                        )
                        on_samples = samples_for(
                            executable_choice(on)
                        )
                        off_median = statistics.median(off_samples)
                        on_median = statistics.median(on_samples)
                        overhead = (on_median / off_median - 1.0) * 100.0
                        row["recovery_overhead"] = {
                            "off": {"config": backend_choice(off), "samples_us": off_samples, "median_us": off_median},
                            "on": {"config": backend_choice(on), "samples_us": on_samples, "median_us": on_median},
                            "overhead_us": on_median - off_median,
                            "overhead_percent": overhead,
                        }
                        row["recovery_overhead_eligible"] = overhead_ok(overhead)
                    report["rows"].append(row)
                passing = [
                    row
                    for row in report["rows"]
                    if row["accepted"]
                    and (
                        not any(enabled)
                        or max_recovery_overhead_percent is None
                        or row.get("recovery_overhead_eligible", False)
                    )
                ]
                if not passing:
                    raise ValueError("no grouped kernel candidate passed correctness and overhead gates")
                winner = min(passing, key=lambda row: row["median_us"])
                selected = tuple(
                    P32WindowConfig.from_backend_config(config)
                    for config in winner["config"]
                )
                report["selected"] = list(backend_choice(selected))
            if any(
                _digest(*_base(child)) != expected_hash
                for child, expected_hash in zip(children, state_hashes, strict=True)
            ):
                raise RuntimeError("grouped window state changed during tuning")
    finally:
        for child, config in zip(children, originals, strict=True):
            prepare_rank8(child, config)

    if cache is not None and (not cache_hit):
        _write_tuning_cache(cache, report)
    if apply:
        prepare_choice(selected)
        metadata = {
            "version": 1,
            "selected": [config.to_backend_config() for config in selected],
            "identity": {"key": cache_key, "candidates": identity["candidates"], "gate": identity["gate"]},
            "cache_hit": cache_hit,
        }
        for child in children:
            child._p32_grouped_window_tuning = metadata
    return GroupedWindowTuningResult(selected, report, cache_hit)
