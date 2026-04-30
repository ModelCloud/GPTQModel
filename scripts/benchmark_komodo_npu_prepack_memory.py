#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch

from benchmark_komodo_npu_ab import (
    BenchCase,
    _dtype,
    _override_dtype,
    _pack_awq_tensor,
    _pack_gptq_qweight,
    _pack_gptq_qzeros,
    _select_cases,
)
from gptqmodel.nn_modules.qlinear.komodo import AwqKomodoLinear, KomodoLinear
from gptqmodel.utils.torch import HAS_NPU


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _source_buffer_bytes(module: torch.nn.Module) -> int:
    total = 0
    for name in getattr(module, "_native_source_buffer_names", ()):
        tensor = getattr(module, name, None)
        if isinstance(tensor, torch.Tensor):
            total += _tensor_bytes(tensor)
    return total


def _source_numels(module: torch.nn.Module) -> dict[str, int]:
    numels = {}
    for name in getattr(module, "_native_source_buffer_names", ()):
        tensor = getattr(module, name, None)
        if isinstance(tensor, torch.Tensor):
            numels[name] = int(tensor.numel())
    return numels


def _native_plan_bytes(module: torch.nn.Module) -> int:
    total = 0
    for plan in getattr(module, "_native_plan_cache", {}).values():
        for item in plan:
            if isinstance(item, torch.Tensor):
                total += _tensor_bytes(item)
    return total


def _npu_memory(device: torch.device) -> dict[str, int]:
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


def _reset_peak_memory(device: torch.device) -> None:
    torch.npu.synchronize(device)
    torch.npu.empty_cache()
    torch.npu.synchronize(device)
    reset_fn = getattr(torch.npu, "reset_peak_memory_stats", None)
    if reset_fn is not None:
        try:
            reset_fn(device)
        except TypeError:
            reset_fn()


def _make_gptq_candidate(case: BenchCase, dtype: torch.dtype, seed: int, *, drop_source: bool) -> KomodoLinear:
    torch.manual_seed(seed)
    bits = 4
    groups = math.ceil(case.in_features / case.group_size)
    int_weight = torch.randint(0, 2**bits, size=(case.in_features, case.out_features), dtype=torch.int32)
    zero_points = torch.randint(0, 2**bits, size=(groups, case.out_features), dtype=torch.int32)
    scales = (torch.rand(groups, case.out_features, dtype=torch.float32) * 0.04 + 0.01).to(dtype)
    bias = torch.randn(case.out_features, dtype=dtype) * 0.03

    candidate = KomodoLinear(
        bits=bits,
        group_size=case.group_size,
        sym=False,
        desc_act=False,
        in_features=case.in_features,
        out_features=case.out_features,
        bias=True,
        pack_dtype=torch.int32,
        register_buffers=True,
    )
    candidate.qweight.copy_(_pack_gptq_qweight(int_weight, bits))
    candidate.qzeros.copy_(_pack_gptq_qzeros(zero_points, bits))
    candidate.scales.copy_(scales.to(candidate.scales.dtype))
    candidate.g_idx.copy_(torch.arange(case.in_features, dtype=torch.int32) // case.group_size)
    candidate.bias.copy_(bias.to(candidate.bias.dtype))
    candidate.optimized = True
    candidate.post_init()
    candidate.eval()
    candidate.enable_weight_cache(False)
    candidate.enable_source_weight_drop(drop_source)
    return candidate


def _make_awq_candidate(case: BenchCase, dtype: torch.dtype, seed: int, *, drop_source: bool) -> AwqKomodoLinear:
    torch.manual_seed(seed)
    bits = 4
    groups = math.ceil(case.in_features / case.group_size)
    int_weight = torch.randint(0, 2**bits, size=(case.in_features, case.out_features), dtype=torch.int32)
    zero_points = torch.randint(0, 2**bits, size=(groups, case.out_features), dtype=torch.int32)
    scales = ((torch.rand(groups, case.out_features, dtype=torch.float32) * 2.0) + 0.25).to(dtype)
    bias = torch.randn(case.out_features, dtype=dtype)

    candidate = AwqKomodoLinear(
        bits=bits,
        group_size=case.group_size,
        sym=True,
        desc_act=False,
        in_features=case.in_features,
        out_features=case.out_features,
        bias=True,
        dtype=dtype,
        register_buffers=True,
    )
    candidate.qweight.copy_(_pack_awq_tensor(int_weight, bits))
    candidate.qzeros.copy_(_pack_awq_tensor(zero_points, bits))
    candidate.scales.copy_(scales.to(candidate.scales.dtype))
    candidate.bias.copy_(bias.to(candidate.bias.dtype))
    candidate.post_init()
    candidate.eval()
    candidate.enable_weight_cache(False)
    candidate.enable_source_weight_drop(drop_source)
    return candidate


def _make_candidate(case: BenchCase, dtype: torch.dtype, seed: int, *, drop_source: bool) -> torch.nn.Module:
    if case.method == "gptq":
        return _make_gptq_candidate(case, dtype=dtype, seed=seed, drop_source=drop_source)
    if case.method == "awq":
        return _make_awq_candidate(case, dtype=dtype, seed=seed, drop_source=drop_source)
    raise ValueError(f"Unsupported method `{case.method}`.")


def _run_case(
    case: BenchCase,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    drop_source: bool,
    tile_n: int,
) -> dict:
    candidate = _make_candidate(case, dtype=dtype, seed=seed, drop_source=drop_source)
    original_source_bytes = _source_buffer_bytes(candidate)

    gc.collect()
    _reset_peak_memory(device)
    before = _npu_memory(device)

    start = time.perf_counter()
    candidate = candidate.to(device=device, dtype=dtype)
    torch.npu.synchronize(device)
    prepack_ms = (time.perf_counter() - start) * 1000.0
    after = _npu_memory(device)

    prepacked = bool(candidate.native_plan_prepacked(device=device, dtype=dtype))
    source_numels = _source_numels(candidate)
    result = {
        **asdict(case),
        "device": str(device),
        "tile_n": tile_n,
        "drop_source": drop_source,
        "prepack_ms": prepack_ms,
        "prepacked": prepacked,
        "source_dropped": bool(getattr(candidate, "_native_source_dropped", False)),
        "source_numels": source_numels,
        "source_empty_all": bool(source_numels) and all(numel == 0 for numel in source_numels.values()),
        "original_source_bytes": original_source_bytes,
        "live_source_bytes": _source_buffer_bytes(candidate),
        "native_plan_bytes": _native_plan_bytes(candidate),
        "memory_before": before,
        "memory_after": after,
    }
    result["peak_allocated_delta_bytes"] = after.get("max_memory_allocated", 0) - before.get("memory_allocated", 0)
    result["peak_reserved_delta_bytes"] = after.get("max_memory_reserved", 0) - before.get("memory_reserved", 0)
    result["live_allocated_delta_bytes"] = after.get("memory_allocated", 0) - before.get("memory_allocated", 0)
    result["live_reserved_delta_bytes"] = after.get("memory_reserved", 0) - before.get("memory_reserved", 0)

    del candidate
    gc.collect()
    torch.npu.empty_cache()
    return result


def _case_choices() -> tuple[str, ...]:
    return (
        "quick",
        "default",
        "qwen3_6_27b_gptq",
        "qwen3_6_27b_awq",
        "qwen3_6_27b_all",
        "qwen3_6_35b_a3b_gptq",
        "qwen3_6_35b_a3b_awq",
        "qwen3_6_35b_a3b_all",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure cold Komodo native int4 load/prepack NPU peak memory.")
    parser.add_argument("--device", type=int, default=0, help="PCI-ordered NPU device index.")
    parser.add_argument("--cases", choices=_case_choices(), default="qwen3_6_27b_all")
    parser.add_argument("--dtype", choices=("fp16",), default="fp16")
    parser.add_argument("--tile-n", type=int, default=1024, help="GPTQMODEL_KOMODO_PREPACK_TILE_N for this run.")
    parser.add_argument("--drop-source", dest="drop_source", action="store_true", default=True)
    parser.add_argument("--no-drop-source", dest="drop_source", action="store_false")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=41000)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    if not HAS_NPU:
        raise RuntimeError("Ascend NPU is required for Komodo memory benchmarking.")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1.")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, --num-shards).")
    device_count = torch.npu.device_count()
    if not (0 <= args.device < device_count):
        raise ValueError(f"--device must be in [0, {device_count}); got {args.device}.")

    os.environ["GPTQMODEL_KOMODO_NATIVE_INT4"] = "1"
    os.environ["GPTQMODEL_KOMODO_EAGER_PREPACK"] = "1"
    os.environ["GPTQMODEL_KOMODO_DROP_SOURCE_WEIGHTS"] = "1" if args.drop_source else "0"
    os.environ["GPTQMODEL_KOMODO_CACHE_WEIGHTS"] = "0"
    os.environ["GPTQMODEL_KOMODO_PREPACK_TILE_N"] = str(args.tile_n)

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    dtype = _dtype(args.dtype)
    selected_cases = _override_dtype(_select_cases(args.cases), args.dtype)
    cases = [
        case
        for index, case in enumerate(selected_cases)
        if index % args.num_shards == args.shard_index
    ]
    if not cases:
        print(json.dumps({"device": str(device), "cases": 0}))
        return

    results = [
        _run_case(
            case,
            device=device,
            dtype=dtype,
            seed=args.seed + index,
            drop_source=args.drop_source,
            tile_n=args.tile_n,
        )
        for index, case in enumerate(cases)
    ]
    summary = {
        "max_peak_allocated_delta_bytes": max(result["peak_allocated_delta_bytes"] for result in results),
        "max_peak_reserved_delta_bytes": max(result["peak_reserved_delta_bytes"] for result in results),
        "max_live_allocated_delta_bytes": max(result["live_allocated_delta_bytes"] for result in results),
        "max_live_reserved_delta_bytes": max(result["live_reserved_delta_bytes"] for result in results),
        "source_dropped_all": all(result["source_dropped"] for result in results) if args.drop_source else False,
        "source_empty_all": all(result["source_empty_all"] for result in results) if args.drop_source else False,
        "prepacked_all": all(result["prepacked"] for result in results),
    }

    for result in results:
        print(
            "{name} {device} tile={tile_n} drop={drop_source} peak_alloc={peak_alloc:.3f}GB "
            "live_alloc={live_alloc:.3f}GB plan={plan:.3f}GB source_live={source:.3f}GB "
            "prepack={prepack_ms:.3f}ms dropped={source_dropped} prepacked={prepacked}".format(
                **result,
                peak_alloc=result["peak_allocated_delta_bytes"] / 1024**3,
                live_alloc=result["live_allocated_delta_bytes"] / 1024**3,
                plan=result["native_plan_bytes"] / 1024**3,
                source=result["live_source_bytes"] / 1024**3,
            )
        )

    print(
        "TOTAL cases={count} tile={tile} drop={drop} max_peak_alloc={peak:.3f}GB "
        "max_live_alloc={live:.3f}GB source_dropped_all={dropped} source_empty_all={empty}".format(
            count=len(results),
            tile=args.tile_n,
            drop=args.drop_source,
            peak=summary["max_peak_allocated_delta_bytes"] / 1024**3,
            live=summary["max_live_allocated_delta_bytes"] / 1024**3,
            dropped=summary["source_dropped_all"],
            empty=summary["source_empty_all"],
        )
    )

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "device": str(device),
            "cases": [result["name"] for result in results],
            "tile_n": args.tile_n,
            "drop_source": args.drop_source,
            "summary": summary,
            "results": results,
        }
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
