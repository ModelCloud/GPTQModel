#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pcre
import torch
from tabulate import tabulate

from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear


os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


@dataclass(frozen=True)
class BenchCase:
    case_id: str
    m: int
    in_features: int
    out_features: int
    group_size: int = 128
    desc_act: bool = False


def _build_default_cases() -> list[BenchCase]:
    cases: list[BenchCase] = []
    for m in (64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192):
        cases.append(BenchCase(case_id=f"mlp_up_m{m}", m=m, in_features=4096, out_features=11008))
    for m in (64, 80, 96, 112, 128, 160, 192):
        cases.append(BenchCase(case_id=f"mlp_down_m{m}", m=m, in_features=11008, out_features=4096))
    for m in (64, 96, 128, 192):
        cases.append(BenchCase(case_id=f"attn_m{m}", m=m, in_features=4096, out_features=4096))
    return cases


DEFAULT_CASES = _build_default_cases()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark synthetic Marlin GEMMs on A100-class PCI-ordered GPUs."
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device string to benchmark on.")
    parser.add_argument("--dtype", default="fp16", choices=("fp16", "bf16"), help="Marlin compute dtype.")
    parser.add_argument("--warmup", type=int, default=30, help="Warmup iterations per case.")
    parser.add_argument("--iters", type=int, default=80, help="Measured iterations per case.")
    parser.add_argument("--seed", type=int, default=1234, help="Base RNG seed.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of case shards.")
    parser.add_argument(
        "--desc-act",
        default="off",
        choices=("off", "on", "both"),
        help="Whether to benchmark non-act-order kernels, act-order kernels, or both.",
    )
    parser.add_argument(
        "--case-pattern",
        default=None,
        help="Optional regex filter applied to benchmark case ids.",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _subset_cases(cases: list[BenchCase], shard_index: int, num_shards: int) -> list[BenchCase]:
    if num_shards <= 0:
        raise ValueError("`num_shards` must be positive.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"`shard_index` must be in [0, {num_shards - 1}].")
    return [case for index, case in enumerate(cases) if index % num_shards == shard_index]


def _expand_desc_act_cases(cases: list[BenchCase], mode: str) -> list[BenchCase]:
    expanded: list[BenchCase] = []
    desc_modes: tuple[bool, ...]
    if mode == "off":
        desc_modes = (False,)
    elif mode == "on":
        desc_modes = (True,)
    elif mode == "both":
        desc_modes = (False, True)
    else:
        raise ValueError(f"Unsupported desc_act mode: {mode}")

    for case in cases:
        for desc_act in desc_modes:
            case_id = case.case_id if not desc_act else f"{case.case_id}_act"
            expanded.append(
                BenchCase(
                    case_id=case_id,
                    m=case.m,
                    in_features=case.in_features,
                    out_features=case.out_features,
                    group_size=case.group_size,
                    desc_act=desc_act,
                )
            )
    return expanded


def _filter_cases(cases: list[BenchCase], pattern: str | None) -> list[BenchCase]:
    if pattern is None:
        return cases
    regex = pcre.compile(pattern)
    return [case for case in cases if regex.search(case.case_id)]


def _build_module(
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    in_features: int,
    out_features: int,
    group_size: int,
    desc_act: bool,
) -> MarlinLinear:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    module = MarlinLinear(
        bits=4,
        group_size=group_size,
        desc_act=desc_act,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        dtype=dtype,
    ).to(device)
    with torch.no_grad():
        module.qweight.copy_(
            torch.randint(
                -(2**31),
                2**31 - 1,
                module.qweight.shape,
                dtype=torch.int32,
                device=device,
                generator=generator,
            )
        )
        module.scales.copy_(
            torch.rand(
                module.scales.shape,
                dtype=dtype,
                device=device,
                generator=generator,
            ) * 0.5 + 0.5
        )
        module.qzeros.zero_()
        module.g_idx.zero_()
    module.eval()
    module.post_init()
    return module


def _benchmark_case(
    *,
    module: MarlinLinear,
    case: BenchCase,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x = torch.rand(
        (case.m, case.in_features),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    with torch.inference_mode():
        for _ in range(warmup):
            module(x)
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iters):
            y = module(x)
        torch.cuda.synchronize(device)
        elapsed_s = time.perf_counter() - start
    mean_ms = elapsed_s * 1e3 / iters
    tflops = (2.0 * case.m * case.in_features * case.out_features) / (mean_ms * 1e9)
    return {
        "case_id": case.case_id,
        "m": case.m,
        "in_features": case.in_features,
        "out_features": case.out_features,
        "shape": list(y.shape),
        "mean_ms": mean_ms,
        "tflops": tflops,
    }


def main() -> None:
    args = _parse_args()
    dtype = _resolve_dtype(args.dtype)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This benchmark only supports CUDA devices.")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cases = _expand_desc_act_cases(DEFAULT_CASES, args.desc_act)
    cases = _filter_cases(cases, args.case_pattern)
    cases = _subset_cases(cases, shard_index=args.shard_index, num_shards=args.num_shards)
    if not cases:
        raise ValueError("Shard selection produced no benchmark cases.")

    module_cache: dict[tuple[int, int, int, bool], MarlinLinear] = {}
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        cache_key = (case.in_features, case.out_features, case.group_size, case.desc_act)
        module = module_cache.get(cache_key)
        if module is None:
            module = _build_module(
                device=device,
                dtype=dtype,
                seed=args.seed + index,
                in_features=case.in_features,
                out_features=case.out_features,
                group_size=case.group_size,
                desc_act=case.desc_act,
            )
            module_cache[cache_key] = module
        rows.append(
            _benchmark_case(
                module=module,
                case=case,
                dtype=dtype,
                device=device,
                seed=args.seed + 1000 + index,
                warmup=args.warmup,
                iters=args.iters,
            )
        )

    table = [
        [row["case_id"], row["m"], row["in_features"], row["out_features"], f'{row["mean_ms"]:.6f}', f'{row["tflops"]:.2f}']
        for row in rows
    ]
    print(
        tabulate(
            table,
            headers=("case", "m", "k", "n", "mean_ms", "tflops"),
            tablefmt="github",
        )
    )

    payload = {
        "dtype": args.dtype,
        "device": args.device,
        "cuda_visible_devices": visible,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "cases": [asdict(case) for case in cases],
        "results": rows,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\njson_out={args.json_out}")


if __name__ == "__main__":
    main()
