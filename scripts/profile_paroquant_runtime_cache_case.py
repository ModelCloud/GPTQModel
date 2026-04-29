#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import atexit
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class BenchCase:
    case_id: str
    batch: int
    seq: int
    in_features: int
    out_features: int
    group_size: int = 128
    krot: int = 8


CASES = {
    "decode_q_proj": BenchCase("decode_q_proj", batch=1, seq=1, in_features=2048, out_features=2048),
    "prefill_q_proj": BenchCase("prefill_q_proj", batch=1, seq=128, in_features=2048, out_features=2048),
    "batched_down_proj": BenchCase("batched_down_proj", batch=4, seq=128, in_features=8192, out_features=2048),
}

_TEMP_BUILD_DIRS: list[tempfile.TemporaryDirectory[str]] = []


def _register_temp_build_dir(prefix: str) -> Path:
    temp_dir = tempfile.TemporaryDirectory(prefix=prefix)
    _TEMP_BUILD_DIRS.append(temp_dir)
    return Path(temp_dir.name)


atexit.register(lambda: [temp_dir.cleanup() for temp_dir in reversed(_TEMP_BUILD_DIRS)])


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _pack_awq_tensor(unpacked: torch.Tensor, bits: int) -> torch.Tensor:
    pack_factor = 32 // bits
    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
    packed = torch.zeros((unpacked.shape[0], unpacked.shape[1] // pack_factor), dtype=torch.int32)
    for col in range(unpacked.shape[1] // pack_factor):
        for i, order in enumerate(order_map):
            value = unpacked[:, col * pack_factor + order].to(torch.int32)
            packed[:, col] |= value << (i * bits)
    return packed


def _make_quant_buffers(case: BenchCase, dtype: torch.dtype, bits: int = 4) -> dict[str, torch.Tensor]:
    from gptqmodel.utils.paroquant import build_identity_rotation_buffers

    groups = case.in_features // case.group_size
    int_weight = torch.randint(0, 2**bits, size=(case.in_features, case.out_features), dtype=torch.int32)
    zero_points = torch.randint(0, 2**bits, size=(groups, case.out_features), dtype=torch.int32)
    scales = (torch.rand(groups, case.out_features, dtype=torch.float32) * 0.5) + 0.75
    scales = scales.to(dtype=dtype)
    bias = torch.randn(case.out_features, dtype=torch.float32).to(dtype=dtype)

    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=case.in_features,
        group_size=case.group_size,
        krot=case.krot,
        dtype=dtype,
    )
    theta.uniform_(-0.2, 0.2)
    channel_scales.uniform_(0.75, 1.25)

    return {
        "qweight": _pack_awq_tensor(int_weight, bits),
        "qzeros": _pack_awq_tensor(zero_points, bits),
        "scales": scales,
        "bias": bias,
        "pairs": pairs,
        "theta": theta,
        "channel_scales": channel_scales,
    }


def _configure_runtime(args: argparse.Namespace, device: torch.device) -> None:
    if args.force_rebuild_awq:
        awq_build_root = _register_temp_build_dir(
            f"awq_jit_profile_{args.case_id}_{args.dtype}_"
            f"rt{int(args.cache_runtime_dtype)}_rot{int(args.cache_rotation_dtype)}_"
        )
        os.environ["GPTQMODEL_AWQ_BUILD_ROOT"] = str(awq_build_root)
        os.environ["GPTQMODEL_AWQ_FORCE_REBUILD"] = "1"
    elif "GPTQMODEL_AWQ_BUILD_ROOT" not in os.environ:
        os.environ.pop("GPTQMODEL_AWQ_BUILD_ROOT", None)
        os.environ.pop("GPTQMODEL_AWQ_FORCE_REBUILD", None)

    if args.force_rebuild_paroquant:
        paro_build_root = _register_temp_build_dir(
            f"paroquant_ext_profile_{args.case_id}_{args.dtype}_"
            f"rt{int(args.cache_runtime_dtype)}_rot{int(args.cache_rotation_dtype)}_"
        )
        os.environ["GPTQMODEL_PAROQUANT_BUILD_ROOT"] = str(paro_build_root)
        os.environ["GPTQMODEL_PAROQUANT_FORCE_REBUILD"] = "1"
    elif "GPTQMODEL_PAROQUANT_BUILD_ROOT" not in os.environ:
        os.environ.pop("GPTQMODEL_PAROQUANT_BUILD_ROOT", None)
        os.environ.pop("GPTQMODEL_PAROQUANT_FORCE_REBUILD", None)

    from gptqmodel.utils.awq import clear_awq_extension_cache, prewarm_awq_extension
    from gptqmodel.utils.paroquant import clear_paroquant_rotation_extension_cache, prewarm_paroquant_rotation_extension

    if args.force_rebuild_awq:
        clear_awq_extension_cache()
    if args.force_rebuild_paroquant:
        clear_paroquant_rotation_extension_cache()

    if not prewarm_awq_extension():
        raise RuntimeError("Failed to build/load AWQ runtime.")
    if not prewarm_paroquant_rotation_extension(
        fused_rotation=True,
        group_size=128,
        krot=8,
        device=device,
    ):
        raise RuntimeError("Failed to build/load ParoQuant runtime.")


def _make_module(
    case: BenchCase,
    dtype: torch.dtype,
    device: torch.device,
    cache_runtime_dtype: bool,
    cache_rotation_dtype: bool,
):
    from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear

    buffers = _make_quant_buffers(case, dtype=dtype)
    module = ParoLinear(
        bits=4,
        group_size=case.group_size,
        sym=True,
        desc_act=False,
        in_features=case.in_features,
        out_features=case.out_features,
        bias=True,
        register_buffers=True,
        krot=case.krot,
        cache_runtime_dtype=cache_runtime_dtype,
        auto_cache_bf16_runtime_dtype=cache_runtime_dtype,
        cache_rotation_dtype=cache_rotation_dtype,
        auto_cache_bf16_rotation_dtype=cache_rotation_dtype,
    ).to(device)
    module.qweight.copy_(buffers["qweight"].to(device))
    module.qzeros.copy_(buffers["qzeros"].to(device))
    module.scales.copy_(buffers["scales"].to(device))
    module.bias.copy_(buffers["bias"].to(device))
    module.pairs.copy_(buffers["pairs"].to(device))
    module.theta.copy_(buffers["theta"].to(device))
    module.channel_scales.copy_(buffers["channel_scales"].to(device))
    module.post_init()
    module.eval()
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile one ParoQuant runtime-cache case.")
    parser.add_argument("--case-id", choices=tuple(CASES.keys()), required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--cache-runtime-dtype", action="store_true")
    parser.add_argument("--cache-rotation-dtype", action="store_true")
    parser.add_argument("--force-rebuild-awq", action="store_true")
    parser.add_argument("--force-rebuild-paroquant", action="store_true")
    parser.add_argument("--torch-profiler-json-out", type=Path, default=None)
    parser.add_argument("--torch-profiler-top-n", type=int, default=12)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device(f"cuda:{args.device}")
    case = CASES[args.case_id]
    dtype = _resolve_dtype(args.dtype)
    _configure_runtime(args, device)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    module = _make_module(
        case,
        dtype=dtype,
        device=device,
        cache_runtime_dtype=args.cache_runtime_dtype,
        cache_rotation_dtype=args.cache_rotation_dtype,
    )
    x = torch.randn((case.batch, case.seq, case.in_features), device=device, dtype=dtype)

    with torch.inference_mode():
        for _ in range(args.warmup):
            module(x)
        torch.cuda.synchronize(device)

        if args.torch_profiler_json_out is not None:
            from torch.profiler import ProfilerActivity, profile

            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
                for _ in range(args.iters):
                    module(x)
                torch.cuda.synchronize(device)

            events = []
            for evt in prof.key_averages():
                cuda_time_total = getattr(evt, "cuda_time_total", getattr(evt, "device_time_total", 0.0))
                self_cuda_time_total = getattr(evt, "self_cuda_time_total", getattr(evt, "self_device_time_total", 0.0))
                events.append(
                    {
                        "key": evt.key,
                        "count": evt.count,
                        "cpu_time_total_us": evt.cpu_time_total,
                        "self_cpu_time_total_us": evt.self_cpu_time_total,
                        "cuda_time_total_us": cuda_time_total,
                        "self_cuda_time_total_us": self_cuda_time_total,
                    }
                )
            events.sort(key=lambda row: row["cuda_time_total_us"], reverse=True)
            args.torch_profiler_json_out.parent.mkdir(parents=True, exist_ok=True)
            args.torch_profiler_json_out.write_text(
                json.dumps(
                    {
                        "case_id": args.case_id,
                        "dtype": args.dtype,
                        "cache_runtime_dtype": args.cache_runtime_dtype,
                        "cache_rotation_dtype": args.cache_rotation_dtype,
                        "top_events": events[: args.torch_profiler_top_n],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        else:
            for _ in range(args.iters):
                module(x)
            torch.cuda.synchronize(device)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
