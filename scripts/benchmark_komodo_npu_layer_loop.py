#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch

from benchmark_komodo_npu_ab import (
    QWEN3_6_27B_AWQ_CASES,
    QWEN3_6_27B_GPTQ_CASES,
    QWEN3_6_35B_A3B_AWQ_CASES,
    QWEN3_6_35B_A3B_GPTQ_CASES,
    _drift,
    _dtype,
    _make_awq_pair,
    _make_gptq_pair,
    _sync,
)
from gptqmodel.utils.torch import HAS_NPU


MODES = (
    "fallback",
    "native",
    "native_drop",
    "lookahead",
    "lookahead_drop",
    "prefetch_all",
    "prefetch_all_drop",
)
PROJECTION_NAMES = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj")


def _cases(model: str, method: str, dtype: str):
    if model == "qwen3_6_27b" and method == "gptq":
        cases = QWEN3_6_27B_GPTQ_CASES
    elif model == "qwen3_6_27b" and method == "awq":
        cases = QWEN3_6_27B_AWQ_CASES
    elif model == "qwen3_6_35b_a3b" and method == "gptq":
        cases = QWEN3_6_35B_A3B_GPTQ_CASES
    elif model == "qwen3_6_35b_a3b" and method == "awq":
        cases = QWEN3_6_35B_A3B_AWQ_CASES
    else:
        raise ValueError(f"Unsupported model/method pair: {model}/{method}")
    return [replace(case, dtype=dtype) for case in cases]


def _make_pair(case, *, dtype: torch.dtype, device: torch.device, seed: int):
    if case.method == "gptq":
        return _make_gptq_pair(case, dtype=dtype, device=device, seed=seed, cache_dequantized=False)
    if case.method == "awq":
        return _make_awq_pair(case, dtype=dtype, device=device, seed=seed, cache_dequantized=False)
    raise ValueError(f"Unsupported method `{case.method}`.")


def _flat_modules(stack: list[dict[str, torch.nn.Module]]) -> list[torch.nn.Module]:
    modules = []
    for layer in stack:
        modules.extend([layer[name] for name in PROJECTION_NAMES])
    return modules


def _projection_name(case_name: str) -> str:
    for name in PROJECTION_NAMES:
        if case_name.endswith(name):
            return name
    raise ValueError(f"Cannot infer projection name from `{case_name}`.")


def _link_lookahead(stack: list[dict[str, torch.nn.Module]]) -> None:
    modules = _flat_modules(stack)
    for index, module in enumerate(modules):
        next_module = modules[index + 1] if index + 1 < len(modules) else None
        module.enable_lookahead(True).set_lookahead_next(next_module)


def _forward_layer_loop(
    stack: list[dict[str, torch.nn.Module]],
    hidden: torch.Tensor,
    *,
    collect: bool,
    stabilize_scale: float,
) -> torch.Tensor:
    outputs = []
    for layer in stack:
        q_out = layer["q_proj"](hidden)
        k_out = layer["k_proj"](hidden)
        v_out = layer["v_proj"](hidden)
        gate = layer["gate_proj"](hidden)
        up = layer["up_proj"](hidden)
        gate_scaled = gate * stabilize_scale
        up_scaled = up * stabilize_scale
        hidden = layer["down_proj"](torch.nn.functional.silu(gate_scaled) * up_scaled) * stabilize_scale
        if collect:
            outputs.extend([q_out, k_out, v_out, gate_scaled, up_scaled, hidden])
    if collect:
        return torch.cat([output.reshape(-1) for output in outputs])
    return hidden


def _measure(fn, *, warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - start) * 1000.0 / max(1, iters)


def _npu_memory(device: torch.device) -> dict[str, int]:
    if device.type != "npu":
        return {}
    memory = {}
    for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
        fn = getattr(torch.npu, name, None)
        if fn is None:
            continue
        try:
            memory[name] = int(fn(device))
        except TypeError:
            memory[name] = int(fn())
    return memory


def _run(args) -> dict:
    if args.mode == "fallback":
        os.environ["GPTQMODEL_KOMODO_NATIVE_INT4"] = "0"
    else:
        os.environ["GPTQMODEL_KOMODO_NATIVE_INT4"] = "1"
    os.environ["GPTQMODEL_KOMODO_DROP_SOURCE_WEIGHTS"] = "1" if args.mode.endswith("_drop") else "0"
    os.environ["GPTQMODEL_KOMODO_CACHE_WEIGHTS"] = "0"

    dtype = _dtype(args.dtype)
    cases = _cases(args.model, args.method, args.dtype)
    torch.manual_seed(args.seed)
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")

    baseline_stack = []
    candidate_stack = []
    for layer_index in range(args.layers):
        baseline_layer = {}
        candidate_layer = {}
        for case_index, case in enumerate(cases):
            baseline, candidate = _make_pair(
                case,
                dtype=dtype,
                device=device,
                seed=args.seed + layer_index * 1000 + case_index,
            )
            name = _projection_name(case.name)
            baseline_layer[name] = baseline
            candidate_layer[name] = candidate
            if args.mode.endswith("_drop"):
                candidate.enable_source_weight_drop(True)
        baseline_stack.append(baseline_layer)
        candidate_stack.append(candidate_layer)

    if args.mode.startswith("lookahead"):
        _link_lookahead(candidate_stack)

    hidden = torch.randn(args.tokens, cases[0].in_features, dtype=dtype, device=device)
    candidate_modules = _flat_modules(candidate_stack)

    prepack_ms = 0.0
    prefetched = 0
    if args.mode.startswith("prefetch_all"):
        start = time.perf_counter()
        for module in candidate_modules:
            prefetched += int(bool(module.prefetch_native_plan(device=device, dtype=dtype)))
        _sync(device)
        prepack_ms = (time.perf_counter() - start) * 1000.0

    memory_before = _npu_memory(device)
    with torch.inference_mode():
        expected = _forward_layer_loop(baseline_stack, hidden, collect=True, stabilize_scale=args.stabilize_scale)
        _sync(device)

        first_start = time.perf_counter()
        actual = _forward_layer_loop(candidate_stack, hidden, collect=True, stabilize_scale=args.stabilize_scale)
        _sync(device)
        first_ms = (time.perf_counter() - first_start) * 1000.0

        repeat_start = time.perf_counter()
        repeat = _forward_layer_loop(candidate_stack, hidden, collect=True, stabilize_scale=args.stabilize_scale)
        _sync(device)
        repeat_ms = (time.perf_counter() - repeat_start) * 1000.0

    drift = _drift(expected, actual)
    repeat_drift = _drift(expected, repeat)

    with torch.inference_mode():
        baseline_ms = _measure(
            lambda: _forward_layer_loop(
                baseline_stack,
                hidden,
                collect=False,
                stabilize_scale=args.stabilize_scale,
            ),
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )
        candidate_ms = _measure(
            lambda: _forward_layer_loop(
                candidate_stack,
                hidden,
                collect=False,
                stabilize_scale=args.stabilize_scale,
            ),
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )

    source_dropped = sum(int(bool(getattr(module, "_native_source_dropped", False))) for module in candidate_modules)
    native_plans = sum(len(getattr(module, "_native_plan_cache", {})) for module in candidate_modules)
    memory_after = _npu_memory(device)

    del expected, actual, repeat, hidden, baseline_stack, candidate_stack
    gc.collect()

    return {
        "model": args.model,
        "method": args.method,
        "mode": args.mode,
        "device": str(device),
        "dtype": args.dtype,
        "tokens": args.tokens,
        "layers": args.layers,
        "modules": args.layers * len(cases),
        "stabilize_scale": args.stabilize_scale,
        "warmup": args.warmup,
        "iters": args.iters,
        "prefetched": prefetched,
        "prepack_ms": prepack_ms,
        "first_ms": first_ms,
        "repeat_ms": repeat_ms,
        "baseline_ms": baseline_ms,
        "komodo_ms": candidate_ms,
        "speedup": baseline_ms / candidate_ms if candidate_ms > 0 else float("inf"),
        "drift": drift,
        "repeat_drift": repeat_drift,
        "source_dropped": source_dropped,
        "native_plans": native_plans,
        "memory_before": memory_before,
        "memory_after": memory_after,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Loop Qwen-shaped Komodo projections to simulate decode-layer inference.")
    parser.add_argument("--model", choices=("qwen3_6_27b", "qwen3_6_35b_a3b"), default="qwen3_6_35b_a3b")
    parser.add_argument("--method", choices=("gptq", "awq"), default="gptq")
    parser.add_argument("--mode", choices=MODES, default="native")
    parser.add_argument("--device", type=int, default=0, help="PCI-ordered NPU device index. Use 0-6.")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--dtype", choices=("fp16",), default="fp16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=19000)
    parser.add_argument("--stabilize-scale", type=float, default=0.01)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    if not HAS_NPU:
        raise RuntimeError("Ascend NPU is required for Komodo benchmarking.")
    if not (0 <= args.device <= 6):
        raise ValueError("--device must be in the PCI-ordered first seven device ids: 0-6.")
    if args.layers < 1:
        raise ValueError("--layers must be >= 1.")
    if args.tokens < 1:
        raise ValueError("--tokens must be >= 1.")

    result = _run(args)
    print(
        "{model} {method} {device} mode={mode} layers={layers} modules={modules} "
        "baseline={baseline_ms:.4f}ms komodo={komodo_ms:.4f}ms speedup={speedup:.3f}x "
        "first={first_ms:.4f}ms repeat={repeat_ms:.4f}ms prepack={prepack_ms:.4f}ms "
        "plans={native_plans} dropped={source_dropped} max_abs={max_abs:.6g} max_rel={max_rel:.6g}".format(
            **result,
            max_abs=result["drift"]["max_abs"],
            max_rel=result["drift"]["max_rel"],
        )
    )
    print(json.dumps(result, indent=2))

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
