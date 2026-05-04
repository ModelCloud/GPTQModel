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

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear.komodo import (
    AwqKomodoLinear,
    KomodoLinear,
    _drop_source_weights_enabled,
    _native_int4_enabled,
)
from gptqmodel.nn_modules.qlinear.cannoe import (
    AwqCannoeLinear,
    CannoeLinear,
    _cannoe_prefetch_enabled,
    cannoe_plan_asdict,
)
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
    desc_act: bool = False


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

GPTQ_GROUP_SIZE_CASES = [
    BenchCase("gptq_gs16", "gptq", "fp16", 8, 1024, 1024, 16),
    BenchCase("gptq_gs32", "gptq", "fp16", 8, 1024, 1024, 32),
    BenchCase("gptq_gs64", "gptq", "fp16", 8, 1024, 1024, 64),
    BenchCase("gptq_gs128", "gptq", "fp16", 8, 1024, 1024, 128),
    BenchCase("gptq_full", "gptq", "fp16", 8, 1024, 1024, 1024),
    BenchCase("gptq_act_order_gs16", "gptq", "fp16", 8, 1024, 1024, 16, True),
    BenchCase("gptq_act_order_gs32", "gptq", "fp16", 8, 1024, 1024, 32, True),
    BenchCase("gptq_act_order_gs128", "gptq", "fp16", 8, 1024, 1024, 128, True),
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


def _set_supported_act_order_g_idx(module: TorchLinear) -> None:
    group_size = module.requested_group_size
    if group_size <= 0 or module.in_features % group_size != 0:
        raise ValueError("Synthetic Komodo act-order cases require a positive divisor group_size.")
    groups = module.in_features // group_size
    natural = torch.arange(module.in_features, dtype=torch.int32) // group_size
    act_order = torch.arange(module.in_features).reshape(groups, group_size).t().reshape(-1)
    with torch.no_grad():
        module.g_idx.copy_(natural[act_order].to(dtype=module.g_idx.dtype, device=module.g_idx.device))
    module.desc_act = True
    module._stream_reset_cache()


def _make_gptq_pair(
    case: BenchCase,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    *,
    cache_dequantized: bool,
    candidate_cls=KomodoLinear,
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
        desc_act=case.desc_act,
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
    if case.desc_act:
        _set_supported_act_order_g_idx(baseline)
    baseline.optimized = True
    baseline.post_init()
    baseline.enable_weight_cache(False)
    baseline.eval()

    candidate = candidate_cls(
        bits=bits,
        group_size=case.group_size,
        sym=False,
        desc_act=case.desc_act,
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
    candidate_cls=AwqKomodoLinear,
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
    candidate = candidate_cls(
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
    cannoe: bool,
    drop_source_weights: bool,
) -> dict:
    dtype = _dtype(case.dtype)
    gptq_candidate_cls = CannoeLinear if cannoe else KomodoLinear
    awq_candidate_cls = AwqCannoeLinear if cannoe else AwqKomodoLinear
    if case.method == "gptq":
        baseline, candidate = _make_gptq_pair(
            case,
            dtype=dtype,
            device=device,
            seed=seed,
            cache_dequantized=cache_dequantized,
            candidate_cls=gptq_candidate_cls,
        )
    elif case.method == "awq":
        baseline, candidate = _make_awq_pair(
            case,
            dtype=dtype,
            device=device,
            seed=seed,
            cache_dequantized=cache_dequantized,
            candidate_cls=awq_candidate_cls,
        )
    else:
        raise ValueError(f"Unsupported method `{case.method}`.")

    if native_int4 and drop_source_weights:
        candidate.enable_source_weight_drop(True)

    x = torch.randn(case.tokens, case.in_features, dtype=dtype, device=device)
    with torch.inference_mode():
        candidate.clear_weight_cache()
        prefetched = False
        prepack_ms = 0.0
        if native_int4 and prefetch_native_plan:
            prepack_start = time.perf_counter()
            prefetched = bool(candidate.prefetch_native_plan(device=device, dtype=dtype))
            _sync(device)
            prepack_ms = (time.perf_counter() - prepack_start) * 1000.0
        expected = baseline(x)
        _sync(device)
        first_start = time.perf_counter()
        actual = candidate(x)
        _sync(device)
        first_ms = (time.perf_counter() - first_start) * 1000.0
        repeat_start = time.perf_counter()
        repeat = candidate(x)
        _sync(device)
        repeat_ms = (time.perf_counter() - repeat_start) * 1000.0

    drift = _drift(expected, actual)
    repeat_drift = _drift(expected, repeat)

    with torch.inference_mode():
        baseline_ms = _measure(lambda: baseline(x), warmup=warmup, iters=iters, device=device)
        candidate_ms = _measure(lambda: candidate(x), warmup=warmup, iters=iters, device=device)
    native_cache = bool(getattr(candidate, "_native_plan_cache", None))
    native_group16_cache = bool(getattr(candidate, "_native_group16_plan_cache", None))
    dense_cache = bool(getattr(candidate, "_cached_weights", None))
    if native_cache:
        komodo_path = "native_int4_prepack"
    elif native_group16_cache:
        group16_path = getattr(candidate, "_native_group16_last_path", None)
        komodo_path = f"native_int4_group16_{group16_path}" if group16_path else "native_int4_group16"
    elif dense_cache:
        komodo_path = "exact_fallback_cache"
    else:
        komodo_path = "no_dequant_cache"

    return {
        **asdict(case),
        "device": str(device),
        "baseline": baseline.__class__.__name__,
        "candidate": candidate.__class__.__name__,
        "komodo_kernel": "cannoe" if cannoe else "komodo",
        "cannoe_plan": cannoe_plan_asdict(getattr(candidate, "_last_cann_plan", None)),
        "cannoe_path": getattr(candidate, "_last_cann_path", None),
        "komodo_dequant_cache": cache_dequantized,
        "komodo_native_int4": native_int4,
        "komodo_prefetch_native_plan": prefetch_native_plan,
        "cannoe_prefetch": bool(_cannoe_prefetch_enabled()) if cannoe else False,
        "komodo_drop_source_weights": drop_source_weights,
        "komodo_source_dropped": bool(getattr(candidate, "_native_source_dropped", False)),
        "komodo_prefetched": prefetched,
        "komodo_prepack_ms": prepack_ms,
        "komodo_first_ms": first_ms,
        "komodo_repeat_ms": repeat_ms,
        "komodo_path": komodo_path,
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
    if name == "gptq_group_sizes":
        return GPTQ_GROUP_SIZE_CASES
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
    parser.add_argument("--device", type=int, default=0, help="PCI-ordered NPU device index for this process.")
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
            "gptq_group_sizes",
        ),
        default="default",
    )
    parser.add_argument(
        "--komodo-native-int4",
        dest="komodo_native_int4",
        action="store_true",
        default=None,
        help="Force the native NPU int4 path.",
    )
    parser.add_argument(
        "--no-komodo-native-int4",
        dest="komodo_native_int4",
        action="store_false",
        help="Disable native NPU int4 and benchmark the exact torch-style fallback.",
    )
    parser.add_argument(
        "--komodo-prefetch-native-plan",
        action="store_true",
        help="When native int4 is active, prebuild the packed NPU plan on a side stream before first forward.",
    )
    parser.add_argument(
        "--cannoe",
        dest="cannoe",
        action="store_true",
        help="Use the separate Cannoe CANN kernel class instead of the plain Komodo kernel.",
    )
    parser.add_argument(
        "--cannoe-prefetch",
        dest="cannoe_prefetch",
        action="store_true",
        help="Enable Cannoe host-issued npu_prefetch probes. Only applies with --cannoe.",
    )
    parser.add_argument(
        "--cannoe-prefetch-max-bytes",
        dest="cannoe_prefetch_max_bytes",
        type=int,
        help="Override GPTQMODEL_CANNOE_PREFETCH_MAX_BYTES for the Cannoe kernel.",
    )
    parser.add_argument(
        "--cannoe-prefetch-min-bytes",
        dest="cannoe_prefetch_min_bytes",
        type=int,
        help="Override GPTQMODEL_CANNOE_PREFETCH_MIN_BYTES for the Cannoe kernel.",
    )
    parser.add_argument(
        "--komodo-drop-source-weights",
        dest="komodo_drop_source_weights",
        action="store_true",
        default=None,
        help="After native int4 pack, drop source GPTQ/AWQ buffers and keep only the native plan.",
    )
    parser.add_argument(
        "--no-komodo-drop-source-weights",
        dest="komodo_drop_source_weights",
        action="store_false",
        help="Keep source GPTQ/AWQ buffers after native int4 pack.",
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
    device_count = torch.npu.device_count()
    if not (0 <= args.device < device_count):
        raise ValueError(f"--device must be in [0, {device_count}); got {args.device}.")

    if args.komodo_native_int4 is True:
        os.environ["GPTQMODEL_KOMODO_NATIVE_INT4"] = "1"
    elif args.komodo_native_int4 is False:
        os.environ["GPTQMODEL_KOMODO_NATIVE_INT4"] = "0"
    if args.komodo_drop_source_weights is True:
        os.environ["GPTQMODEL_KOMODO_DROP_SOURCE_WEIGHTS"] = "1"
    elif args.komodo_drop_source_weights is False:
        os.environ["GPTQMODEL_KOMODO_DROP_SOURCE_WEIGHTS"] = "0"
    if args.cannoe_prefetch:
        os.environ["GPTQMODEL_CANNOE_PREFETCH"] = "1"
    if args.cannoe_prefetch_max_bytes is not None:
        os.environ["GPTQMODEL_CANNOE_PREFETCH_MAX_BYTES"] = str(args.cannoe_prefetch_max_bytes)
    if args.cannoe_prefetch_min_bytes is not None:
        os.environ["GPTQMODEL_CANNOE_PREFETCH_MIN_BYTES"] = str(args.cannoe_prefetch_min_bytes)
    os.environ["GPTQ_CACHE_DEQUANTIZED_WEIGHTS"] = "1" if args.komodo_cache_dequantized else "0"
    native_int4 = _native_int4_enabled()
    cannoe = bool(args.cannoe)
    cannoe_prefetch = bool(_cannoe_prefetch_enabled()) if cannoe else False
    drop_source_weights = _drop_source_weights_enabled()

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
            native_int4=native_int4,
            prefetch_native_plan=args.komodo_prefetch_native_plan,
            cannoe=cannoe,
            drop_source_weights=drop_source_weights,
        )
        for index, case in enumerate(cases)
    ]
    mode = _mode_name(native_int4=native_int4, cache_dequantized=args.komodo_cache_dequantized)

    for result in results:
        print(
            "{name} {device} mode={mode} baseline={baseline_ms:.4f}ms komodo={komodo_ms:.4f}ms "
            "speedup={speedup:.3f}x first={first_ms:.4f}ms repeat={repeat_ms:.4f}ms "
            "prepack={prepack_ms:.4f}ms kernel={kernel} cann_prefetch={cann_prefetch} drop_source={drop_source} "
            "path={komodo_path} max_abs={max_abs:.6g} max_rel={max_rel:.6g}".format(
                **result,
                mode=mode,
                first_ms=result["komodo_first_ms"],
                repeat_ms=result["komodo_repeat_ms"],
                prepack_ms=result["komodo_prepack_ms"],
                kernel=result["komodo_kernel"],
                cann_prefetch=result["cannoe_prefetch"],
                drop_source=result["komodo_source_dropped"],
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
            "komodo_kernel": "cannoe" if cannoe else "komodo",
            "cannoe_prefetch": bool(cannoe_prefetch),
            "komodo_native_int4": bool(native_int4),
            "komodo_prefetch_native_plan": bool(args.komodo_prefetch_native_plan),
            "cannoe_prefetch_max_bytes": args.cannoe_prefetch_max_bytes,
            "cannoe_prefetch_min_bytes": args.cannoe_prefetch_min_bytes,
            "komodo_drop_source_weights": bool(drop_source_weights),
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
