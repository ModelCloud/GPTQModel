# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import dataclass
from typing import Any

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch

from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
from gptqmodel.nn_modules.qlinear.marlin_awq import AwqMarlinLinear
from gptqmodel.nn_modules.qlinear.mentaray import MentaRayLinear
from gptqmodel.nn_modules.qlinear.mentaray_awq import AwqMentaRayLinear


@dataclass(frozen=True)
class ShapeCase:
    m: int
    k: int
    n: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A/B microbenchmark current Marlin against MentaRay.")
    parser.add_argument("--quant", choices=("gptq", "awq"), required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--shapes",
        default="1x4096x11008,16x4096x11008,64x4096x11008",
        help="Comma-separated MxKxN list.",
    )
    parser.add_argument("--output-json", action="store_true")
    return parser.parse_args()


def parse_shapes(raw: str) -> list[ShapeCase]:
    shapes: list[ShapeCase] = []
    for chunk in raw.split(","):
        text = chunk.strip().lower()
        if not text:
            continue
        parts = text.split("x")
        if len(parts) != 3:
            raise ValueError(f"Invalid shape `{chunk}`; expected MxKxN.")
        m, k, n = (int(part) for part in parts)
        shapes.append(ShapeCase(m=m, k=k, n=n))
    if not shapes:
        raise ValueError("At least one shape is required.")
    return shapes


def quant_dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "fp16" else torch.bfloat16


def make_rand_int(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.randint(0, 2**31 - 1, shape, device=device, dtype=torch.int32)


def make_rand_scale(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.rand(shape, device=device, dtype=dtype).add_(0.01)


def make_gptq_state(
    shape: ShapeCase,
    group_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    scale_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float16
    return {
        "qweight": make_rand_int((shape.k // 8, shape.n), device),
        "g_idx": torch.arange(shape.k, device=device, dtype=torch.int32),
        "scales": make_rand_scale((shape.k // group_size, shape.n), scale_dtype, device),
        "qzeros": torch.zeros((shape.k // group_size, shape.n // 8), device=device, dtype=torch.int32),
        "bias": torch.rand((shape.n,), device=device, dtype=scale_dtype),
    }


def make_awq_state(
    shape: ShapeCase,
    group_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    scale_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float16
    return {
        "qweight": make_rand_int((shape.k, shape.n // 8), device),
        "qzeros": make_rand_int((shape.k // group_size, shape.n // 8), device),
        "scales": make_rand_scale((shape.k // group_size, shape.n), scale_dtype, device),
        "bias": torch.rand((shape.n,), device=device, dtype=scale_dtype),
    }


def build_gptq_module(
    cls,
    shape: ShapeCase,
    bits: int,
    group_size: int,
    dtype: torch.dtype,
    device: torch.device,
    state: dict[str, torch.Tensor],
):
    module = cls(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
        in_features=shape.k,
        out_features=shape.n,
        bias=True,
        pack_dtype=torch.int32,
        adapter=None,
        dtype=dtype,
    ).to(device)
    module.qweight.data.copy_(state["qweight"])
    module.g_idx.data.copy_(state["g_idx"])
    module.scales.data.copy_(state["scales"])
    module.qzeros.data.copy_(state["qzeros"])
    if module.bias is not None:
        module.bias.data.copy_(state["bias"])
    module.eval()
    module.post_init()
    return module


def build_awq_module(
    cls,
    shape: ShapeCase,
    bits: int,
    group_size: int,
    dtype: torch.dtype,
    device: torch.device,
    state: dict[str, torch.Tensor],
):
    module = cls(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=False,
        in_features=shape.k,
        out_features=shape.n,
        bias=True,
        pack_dtype=torch.int32,
        adapter=None,
        register_buffers=True,
        dtype=dtype,
    ).to(device)
    module.qweight.data.copy_(state["qweight"])
    module.qzeros.data.copy_(state["qzeros"])
    module.scales.data.copy_(state["scales"])
    if module.bias is not None:
        module.bias.data.copy_(state["bias"])
    module.eval()
    module.post_init()
    return module


def build_modules(
    quant: str,
    shape: ShapeCase,
    bits: int,
    group_size: int,
    dtype: torch.dtype,
    device: torch.device,
    state: dict[str, torch.Tensor],
):
    if quant == "gptq":
        marlin = build_gptq_module(MarlinLinear, shape, bits, group_size, dtype, device, state)
        mentaray = build_gptq_module(MentaRayLinear, shape, bits, group_size, dtype, device, state)
    else:
        marlin = build_awq_module(AwqMarlinLinear, shape, bits, group_size, dtype, device, state)
        mentaray = build_awq_module(AwqMentaRayLinear, shape, bits, group_size, dtype, device, state)
    return marlin, mentaray


def benchmark_module(module: torch.nn.Module, x: torch.Tensor, warmup: int, iters: int) -> tuple[list[float], torch.Tensor]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        for _ in range(warmup):
            module(x)
        torch.cuda.synchronize(x.device)

        times_ms: list[float] = []
        out = module(x)
        torch.cuda.synchronize(x.device)
        for _ in range(iters):
            start.record()
            out = module(x)
            end.record()
            end.synchronize()
            times_ms.append(start.elapsed_time(end))
    return times_ms, out


def summarize_times(times_ms: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(times_ms),
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }


def main() -> None:
    args = parse_args()
    dtype = quant_dtype(args.dtype)
    shapes = parse_shapes(args.shapes)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the MentaRay benchmark.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    props = torch.cuda.get_device_properties(device)
    results: list[dict[str, Any]] = []

    for shape in shapes:
        state = (
            make_gptq_state(shape, args.group_size, dtype, device)
            if args.quant == "gptq"
            else make_awq_state(shape, args.group_size, dtype, device)
        )
        marlin_mod, mentaray_mod = build_modules(
            args.quant,
            shape,
            bits=args.bits,
            group_size=args.group_size,
            dtype=dtype,
            device=device,
            state=state,
        )
        x = torch.randn((shape.m, shape.k), device=device, dtype=dtype)
        marlin_times, marlin_out = benchmark_module(marlin_mod, x, args.warmup, args.iters)
        mentaray_times, mentaray_out = benchmark_module(mentaray_mod, x, args.warmup, args.iters)

        diff = (marlin_out.float() - mentaray_out.float()).abs()
        marlin_stats = summarize_times(marlin_times)
        mentaray_stats = summarize_times(mentaray_times)
        speedup = marlin_stats["median_ms"] / mentaray_stats["median_ms"]

        results.append(
            {
                "shape": {"m": shape.m, "k": shape.k, "n": shape.n},
                "marlin": marlin_stats,
                "mentaray": mentaray_stats,
                "speedup_vs_marlin": speedup,
                "max_abs_diff": diff.max().item(),
                "mean_abs_diff": diff.mean().item(),
            }
        )

        del marlin_mod, mentaray_mod, x, marlin_out, mentaray_out
        torch.cuda.empty_cache()

    payload = {
        "gpu": {
            "name": props.name,
            "sms": props.multi_processor_count,
            "capability": f"{props.major}.{props.minor}",
            "memory_gib": round(props.total_memory / 2**30, 2),
        },
        "quant": args.quant,
        "dtype": args.dtype,
        "bits": args.bits,
        "group_size": args.group_size,
        "results": results,
    }

    if args.output_json:
        print(json.dumps(payload, indent=2))
        return

    print(json.dumps(payload["gpu"]))
    for item in results:
        shape = item["shape"]
        print(
            "shape={m}x{k}x{n} marlin_med={marlin:.3f}ms mentaray_med={mentaray:.3f}ms "
            "speedup={speedup:.3f}x max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}".format(
                m=shape["m"],
                k=shape["k"],
                n=shape["n"],
                marlin=item["marlin"]["median_ms"],
                mentaray=item["mentaray"]["median_ms"],
                speedup=item["speedup_vs_marlin"],
                max_abs=item["max_abs_diff"],
                mean_abs=item["mean_abs_diff"],
            )
        )


if __name__ == "__main__":
    main()
