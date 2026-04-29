#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear.komodo import AwqKomodoLinear, KomodoLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.utils.torch import HAS_NPU


@dataclass(frozen=True)
class BenchCase:
    name: str
    method: str
    dtype: str
    tokens: int
    in_features: int
    out_features: int
    group_size: int


QUICK_CASES = [
    BenchCase("gptq_fp16_t1_1024", "gptq", "fp16", 1, 1024, 1024, 128),
    BenchCase("awq_fp16_t1_1024", "awq", "fp16", 1, 1024, 1024, 128),
]

DEFAULT_CASES = [
    BenchCase("gptq_fp16_t1_1024", "gptq", "fp16", 1, 1024, 1024, 128),
    BenchCase("awq_fp16_t1_1024", "awq", "fp16", 1, 1024, 1024, 128),
    BenchCase("gptq_bf16_t8_2048", "gptq", "bf16", 8, 2048, 2048, 128),
    BenchCase("awq_bf16_t8_2048", "awq", "bf16", 8, 2048, 2048, 128),
    BenchCase("gptq_fp16_t32_2048", "gptq", "fp16", 32, 2048, 2048, 128),
    BenchCase("awq_fp16_t32_2048", "awq", "fp16", 32, 2048, 2048, 128),
    BenchCase("gptq_bf16_t16_4096", "gptq", "bf16", 16, 4096, 4096, 128),
    BenchCase("awq_bf16_t16_4096", "awq", "bf16", 16, 4096, 4096, 128),
]

QWEN3_6_27B_GPTQ_CASES = [
    BenchCase("qwen3_6_27b_gptq_q_proj", "gptq", "bf16", 1, 5120, 6144, 32),
    BenchCase("qwen3_6_27b_gptq_k_proj", "gptq", "bf16", 1, 5120, 1024, 32),
    BenchCase("qwen3_6_27b_gptq_v_proj", "gptq", "bf16", 1, 5120, 1024, 32),
    BenchCase("qwen3_6_27b_gptq_gate_proj", "gptq", "bf16", 1, 5120, 17408, 32),
    BenchCase("qwen3_6_27b_gptq_up_proj", "gptq", "bf16", 1, 5120, 17408, 32),
    BenchCase("qwen3_6_27b_gptq_down_proj", "gptq", "bf16", 1, 17408, 5120, 32),
]

QWEN3_6_27B_AWQ_CASES = [
    BenchCase("qwen3_6_27b_awq_q_proj", "awq", "bf16", 1, 5120, 6144, 32),
    BenchCase("qwen3_6_27b_awq_k_proj", "awq", "bf16", 1, 5120, 1024, 32),
    BenchCase("qwen3_6_27b_awq_v_proj", "awq", "bf16", 1, 5120, 1024, 32),
    BenchCase("qwen3_6_27b_awq_gate_proj", "awq", "bf16", 1, 5120, 17408, 32),
    BenchCase("qwen3_6_27b_awq_up_proj", "awq", "bf16", 1, 5120, 17408, 32),
    BenchCase("qwen3_6_27b_awq_down_proj", "awq", "bf16", 1, 17408, 5120, 32),
]

QWEN3_6_35B_A3B_GPTQ_CASES = [
    BenchCase("qwen3_6_35b_a3b_gptq_q_proj", "gptq", "bf16", 1, 2048, 4096, 128),
    BenchCase("qwen3_6_35b_a3b_gptq_k_proj", "gptq", "bf16", 1, 2048, 512, 128),
    BenchCase("qwen3_6_35b_a3b_gptq_v_proj", "gptq", "bf16", 1, 2048, 512, 128),
    BenchCase("qwen3_6_35b_a3b_gptq_gate_proj", "gptq", "bf16", 1, 2048, 512, 128),
    BenchCase("qwen3_6_35b_a3b_gptq_up_proj", "gptq", "bf16", 1, 2048, 512, 128),
    BenchCase("qwen3_6_35b_a3b_gptq_down_proj", "gptq", "bf16", 1, 512, 2048, 128),
]

QWEN3_6_35B_A3B_AWQ_CASES = [
    BenchCase("qwen3_6_35b_a3b_awq_q_proj", "awq", "bf16", 1, 2048, 4096, 128),
    BenchCase("qwen3_6_35b_a3b_awq_k_proj", "awq", "bf16", 1, 2048, 512, 128),
    BenchCase("qwen3_6_35b_a3b_awq_v_proj", "awq", "bf16", 1, 2048, 512, 128),
    BenchCase("qwen3_6_35b_a3b_awq_gate_proj", "awq", "bf16", 1, 2048, 512, 128),
    BenchCase("qwen3_6_35b_a3b_awq_up_proj", "awq", "bf16", 1, 2048, 512, 128),
    BenchCase("qwen3_6_35b_a3b_awq_down_proj", "awq", "bf16", 1, 512, 2048, 128),
]


def _dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype `{name}`.")


def _sync(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _pack_awq_tensor(unpacked: torch.Tensor, bits: int) -> torch.Tensor:
    pack_factor = 32 // bits
    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
    packed = torch.zeros((unpacked.shape[0], unpacked.shape[1] // pack_factor), dtype=torch.int32)
    for col in range(unpacked.shape[1] // pack_factor):
        for lane, order in enumerate(order_map):
            packed[:, col] |= unpacked[:, col * pack_factor + order].to(torch.int32) << (lane * bits)
    return packed


def _pack_gptq_qweight(unpacked: torch.Tensor, bits: int) -> torch.Tensor:
    pack_factor = 32 // bits
    rows = unpacked.view(unpacked.shape[0] // pack_factor, pack_factor, unpacked.shape[1])
    packed = torch.zeros((rows.shape[0], rows.shape[2]), dtype=torch.int32)
    for lane in range(pack_factor):
        packed |= rows[:, lane, :].to(torch.int32) << (lane * bits)
    return packed


def _pack_gptq_qzeros(unpacked: torch.Tensor, bits: int) -> torch.Tensor:
    pack_factor = 32 // bits
    cols = unpacked.view(unpacked.shape[0], unpacked.shape[1] // pack_factor, pack_factor)
    packed = torch.zeros((cols.shape[0], cols.shape[1]), dtype=torch.int32)
    for lane in range(pack_factor):
        packed |= cols[:, :, lane].to(torch.int32) << (lane * bits)
    return packed


def _copy_named_buffers(dst: nn.Module, src: nn.Module) -> None:
    src_buffers = dict(src.named_buffers())
    with torch.no_grad():
        for name, dst_tensor in dst.named_buffers():
            src_tensor = src_buffers.get(name)
            if src_tensor is None or dst_tensor.shape != src_tensor.shape:
                continue
            dst_tensor.copy_(src_tensor.to(device=dst_tensor.device, dtype=dst_tensor.dtype))


def _make_gptq_pair(
    case: BenchCase,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    *,
    cache_dequantized: bool,
):
    torch.manual_seed(seed)
    bits = 4
    groups = math.ceil(case.in_features / case.group_size)
    int_weight = torch.randint(0, 2**bits, size=(case.in_features, case.out_features), dtype=torch.int32)
    zero_points = torch.randint(0, 2**bits, size=(groups, case.out_features), dtype=torch.int32)
    scales = (torch.rand(groups, case.out_features, dtype=torch.float32) * 0.04 + 0.01).to(dtype)
    bias = torch.randn(case.out_features, dtype=dtype) * 0.03

    baseline = TorchLinear(
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
    baseline.qweight.copy_(_pack_gptq_qweight(int_weight, bits))
    baseline.qzeros.copy_(_pack_gptq_qzeros(zero_points, bits))
    baseline.scales.copy_(scales.to(baseline.scales.dtype))
    baseline.g_idx.copy_(torch.arange(case.in_features, dtype=torch.int32) // case.group_size)
    baseline.bias.copy_(bias.to(baseline.bias.dtype))
    baseline.optimized = True
    baseline.post_init()
    baseline.enable_weight_cache(False)
    baseline.eval()

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
    _copy_named_buffers(candidate, baseline)
    candidate.optimized = True
    candidate.post_init()
    candidate.eval()
    candidate.enable_weight_cache(cache_dequantized)

    return baseline.to(device=device, dtype=dtype), candidate.to(device=device, dtype=dtype)


def _make_awq_pair(
    case: BenchCase,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    *,
    cache_dequantized: bool,
):
    torch.manual_seed(seed)
    bits = 4
    groups = math.ceil(case.in_features / case.group_size)
    int_weight = torch.randint(0, 2**bits, size=(case.in_features, case.out_features), dtype=torch.int32)
    zero_points = torch.randint(0, 2**bits, size=(groups, case.out_features), dtype=torch.int32)
    scales = ((torch.rand(groups, case.out_features, dtype=torch.float32) * 2.0) + 0.25).to(dtype)
    bias = torch.randn(case.out_features, dtype=dtype)

    baseline = AwqTorchLinear(
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

    qweight = _pack_awq_tensor(int_weight, bits)
    qzeros = _pack_awq_tensor(zero_points, bits)
    for module in (baseline, candidate):
        module.qweight.copy_(qweight)
        module.qzeros.copy_(qzeros)
        module.scales.copy_(scales.to(module.scales.dtype))
        module.bias.copy_(bias.to(module.bias.dtype))
        module.post_init()
        module.eval()

    candidate.enable_weight_cache(cache_dequantized)
    return baseline.to(device=device, dtype=dtype), candidate.to(device=device, dtype=dtype)


def _measure(fn, *, warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    _sync(device)

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - start) * 1000.0 / max(1, iters)


def _drift(expected: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    expected_f = expected.detach().to(device="cpu", dtype=torch.float32)
    actual_f = actual.detach().to(device="cpu", dtype=torch.float32)
    diff = actual_f - expected_f
    denom = expected_f.abs().clamp_min(1e-6)
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "max_rel": float((diff.abs() / denom).max().item()),
        "cosine": float(torch.nn.functional.cosine_similarity(expected_f.flatten(), actual_f.flatten(), dim=0).item()),
    }


def _run_case(
    case: BenchCase,
    *,
    device: torch.device,
    warmup: int,
    iters: int,
    seed: int,
    cache_dequantized: bool,
    native_int4: bool,
    prefetch_native_plan: bool,
) -> dict:
    dtype = _dtype(case.dtype)
    if case.method == "gptq":
        baseline, candidate = _make_gptq_pair(
            case,
            dtype=dtype,
            device=device,
            seed=seed,
            cache_dequantized=cache_dequantized,
        )
    elif case.method == "awq":
        baseline, candidate = _make_awq_pair(
            case,
            dtype=dtype,
            device=device,
            seed=seed,
            cache_dequantized=cache_dequantized,
        )
    else:
        raise ValueError(f"Unsupported method `{case.method}`.")

    x = torch.randn(case.tokens, case.in_features, dtype=dtype, device=device)
    with torch.inference_mode():
        candidate.clear_weight_cache()
        prefetched = False
        if native_int4 and prefetch_native_plan:
            prefetched = bool(candidate.prefetch_native_plan(device=device, dtype=dtype))
        expected = baseline(x)
        actual = candidate(x)
        repeat = candidate(x)
        _sync(device)

    drift = _drift(expected, actual)
    repeat_drift = _drift(expected, repeat)

    with torch.inference_mode():
        baseline_ms = _measure(lambda: baseline(x), warmup=warmup, iters=iters, device=device)
        candidate_ms = _measure(lambda: candidate(x), warmup=warmup, iters=iters, device=device)

    return {
        **asdict(case),
        "device": str(device),
        "baseline": baseline.__class__.__name__,
        "candidate": candidate.__class__.__name__,
        "komodo_dequant_cache": cache_dequantized,
        "komodo_native_int4": native_int4,
        "komodo_prefetch_native_plan": prefetch_native_plan,
        "komodo_prefetched": prefetched,
        "komodo_path": "native_int4_prepack" if getattr(candidate, "_native_plan_cache", None) else (
            "dequant_cache" if cache_dequantized else "no_dequant_cache"
        ),
        "baseline_ms": baseline_ms,
        "komodo_ms": candidate_ms,
        "speedup": baseline_ms / candidate_ms if candidate_ms > 0 else float("inf"),
        "drift": drift,
        "repeat_drift": repeat_drift,
    }


def _select_cases(name: str) -> list[BenchCase]:
    if name == "quick":
        return QUICK_CASES
    if name == "default":
        return DEFAULT_CASES
    if name == "qwen3_6_27b_gptq":
        return QWEN3_6_27B_GPTQ_CASES
    if name == "qwen3_6_27b_awq":
        return QWEN3_6_27B_AWQ_CASES
    if name == "qwen3_6_27b_all":
        return QWEN3_6_27B_GPTQ_CASES + QWEN3_6_27B_AWQ_CASES
    if name == "qwen3_6_35b_a3b_gptq":
        return QWEN3_6_35B_A3B_GPTQ_CASES
    if name == "qwen3_6_35b_a3b_awq":
        return QWEN3_6_35B_A3B_AWQ_CASES
    if name == "qwen3_6_35b_a3b_all":
        return QWEN3_6_35B_A3B_GPTQ_CASES + QWEN3_6_35B_A3B_AWQ_CASES
    raise ValueError(f"Unsupported case set `{name}`.")


def _override_dtype(cases: list[BenchCase], dtype: str | None) -> list[BenchCase]:
    if dtype is None:
        return cases
    return [replace(case, dtype=dtype, name=f"{case.name}_{dtype}") for case in cases]


def _mode_name(*, native_int4: bool, cache_dequantized: bool) -> str:
    if native_int4:
        return "native_int4_prepack"
    if cache_dequantized:
        return "dequant_cache"
    return "no_dequant_cache"


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B benchmark Komodo NPU kernels against torch baselines.")
    parser.add_argument("--device", type=int, default=0, help="NPU device index for this process.")
    parser.add_argument(
        "--cases",
        choices=(
            "quick",
            "default",
            "qwen3_6_27b_gptq",
            "qwen3_6_27b_awq",
            "qwen3_6_27b_all",
            "qwen3_6_35b_a3b_gptq",
            "qwen3_6_35b_a3b_awq",
            "qwen3_6_35b_a3b_all",
        ),
        default="default",
    )
    parser.add_argument("--komodo-native-int4", action="store_true", help="Benchmark the opt-in native NPU int4 path.")
    parser.add_argument(
        "--komodo-prefetch-native-plan",
        action="store_true",
        help="When native int4 is active, prebuild the packed NPU plan on a side stream before first forward.",
    )
    parser.add_argument(
        "--komodo-cache-dequantized",
        action="store_true",
        help="Opt into the dense dequantized-weight cache for comparison only.",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), help="Override the dtype for the selected cases.")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    if not HAS_NPU:
        raise RuntimeError("Ascend NPU is required for Komodo benchmarking.")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1.")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, --num-shards).")

    if args.komodo_native_int4:
        os.environ["GPTQMODEL_KOMODO_NATIVE_INT4"] = "1"
    else:
        os.environ.pop("GPTQMODEL_KOMODO_NATIVE_INT4", None)
    os.environ["GPTQMODEL_KOMODO_CACHE_WEIGHTS"] = "1" if args.komodo_cache_dequantized else "0"

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
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
            warmup=args.warmup,
            iters=args.iters,
            seed=args.seed + index,
            cache_dequantized=args.komodo_cache_dequantized,
            native_int4=args.komodo_native_int4,
            prefetch_native_plan=args.komodo_prefetch_native_plan,
        )
        for index, case in enumerate(cases)
    ]
    mode = _mode_name(native_int4=args.komodo_native_int4, cache_dequantized=args.komodo_cache_dequantized)

    for result in results:
        print(
            "{name} {device} mode={mode} baseline={baseline_ms:.4f}ms komodo={komodo_ms:.4f}ms "
            "speedup={speedup:.3f}x path={komodo_path} max_abs={max_abs:.6g} max_rel={max_rel:.6g}".format(
                **result,
                mode=mode,
                max_abs=result["drift"]["max_abs"],
                max_rel=result["drift"]["max_rel"],
            )
        )

    total_baseline_ms = sum(result["baseline_ms"] for result in results)
    total_komodo_ms = sum(result["komodo_ms"] for result in results)
    total_speedup = total_baseline_ms / total_komodo_ms if total_komodo_ms > 0 else float("inf")
    max_abs = max(result["drift"]["max_abs"] for result in results)
    max_rel = max(result["drift"]["max_rel"] for result in results)
    measured_loop_seconds = (total_baseline_ms + total_komodo_ms) * max(1, args.iters) / 1000.0
    print(
        "TOTAL cases={count} mode={mode} baseline_sum={baseline:.4f}ms "
        "komodo_sum={komodo:.4f}ms speedup={speedup:.3f}x "
        "max_abs={max_abs:.6g} max_rel={max_rel:.6g} prefetch={prefetch} measured_loop={loop:.4f}s".format(
            count=len(results),
            mode=mode,
            baseline=total_baseline_ms,
            komodo=total_komodo_ms,
            speedup=total_speedup,
            max_abs=max_abs,
            max_rel=max_rel,
            prefetch=bool(args.komodo_prefetch_native_plan),
            loop=measured_loop_seconds,
        )
    )

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "device": str(device),
            "komodo_native_int4": bool(args.komodo_native_int4),
            "komodo_prefetch_native_plan": bool(args.komodo_prefetch_native_plan),
            "komodo_dequant_cache": bool(args.komodo_cache_dequantized),
            "mode": mode,
            "dtype_override": args.dtype,
            "cases": [result["name"] for result in results],
            "summary": {
                "baseline_sum_ms": total_baseline_ms,
                "komodo_sum_ms": total_komodo_ms,
                "speedup": total_speedup,
                "max_abs": max_abs,
                "max_rel": max_rel,
                "measured_loop_seconds": measured_loop_seconds,
            },
            "results": results,
        }
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
