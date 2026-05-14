#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
#
# Ascend 910B Cannoe reference, physical NPU7, 2026-05-14:
# ASCEND_RT_VISIBLE_DEVICES=7 python scripts/benchmark_qwen3_27b_gptq_fp16.py \
#   --path cannoe --device npu:0 --tokens 1 --warmup 10 --iters 50
# Steady-state fp16 GPTQ group-32 mean_ms:
# q=0.0775, k=0.0655, v=0.0654, gate=0.2477, up=0.2440, down=0.3002,
# total=1.0001. First-run total=258.2754ms, dominated by native packing.

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("ASCEND_GLOBAL_LOG_LEVEL", "3")
os.environ.setdefault("ASCEND_SLOG_PRINT_TO_STDOUT", "0")
os.environ.setdefault("GPTQ_CACHE_DEQUANTIZED_WEIGHTS", "0")

import torch

_RESULT_FD: int | None = None


@dataclass(frozen=True)
class LayerCase:
    name: str
    in_features: int
    out_features: int


QWEN3_27B_LAYERS: dict[str, LayerCase] = {
    "q": LayerCase("q_proj", 5120, 6144),
    "k": LayerCase("k_proj", 5120, 1024),
    "v": LayerCase("v_proj", 5120, 1024),
    "gate": LayerCase("gate_proj", 5120, 17408),
    "up": LayerCase("up_proj", 5120, 17408),
    "down": LayerCase("down_proj", 17408, 5120),
}


def _parse_csv_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def _parse_layers(raw: str) -> list[str]:
    layers = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = [item for item in layers if item not in QWEN3_27B_LAYERS]
    if unknown:
        valid = ",".join(QWEN3_27B_LAYERS)
        raise argparse.ArgumentTypeError(f"unknown layer(s) {unknown}; valid values: {valid}")
    return layers


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic Qwen3 27B GPTQ fp16 projection benchmark for CUDA Marlin and Ascend Cannoe."
    )
    parser.add_argument("--path", choices=("cuda", "cannoe"), default="cannoe", help="Kernel path to benchmark.")
    parser.add_argument(
        "--cuda-kernel",
        choices=("marlin", "machete", "torch"),
        default="marlin",
        help="CUDA GPTQ kernel used when --path cuda. A100 should normally use marlin.",
    )
    parser.add_argument("--device", default=None, help="Torch device. Defaults to cuda:0 or npu:0 from --path.")
    parser.add_argument("--tokens", type=_parse_csv_ints, default=[1], help="Comma-separated M sizes, e.g. 1,8,32.")
    parser.add_argument("--layers", type=_parse_layers, default=list(QWEN3_27B_LAYERS), help="Comma-separated layer keys.")
    parser.add_argument("--group-size", type=int, default=32, help="GPTQ group size. Qwen3 27B probes use 32.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--bias", action="store_true", help="Include a synthetic fp16 bias.")
    parser.add_argument("--desc-act", action="store_true", help="Use a supported synthetic act-order g_idx.")
    parser.add_argument("--sym", action=argparse.BooleanOptionalAction, default=True, help="Use symmetric GPTQ.")
    parser.add_argument(
        "--marlin-fp32-reduce",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set GPTQMODEL_MARLIN_USE_FP32 for CUDA Marlin.",
    )
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def _quiet_cann_logs_enabled() -> bool:
    raw = os.getenv("GPTQMODEL_BENCHMARK_QUIET_CANN_LOGS")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _enable_quiet_cann_logs(path: str) -> None:
    global _RESULT_FD
    if path != "cannoe" or not _quiet_cann_logs_enabled() or _RESULT_FD is not None:
        return
    _RESULT_FD = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
    finally:
        os.close(devnull_fd)


def _emit(text: str) -> None:
    if _RESULT_FD is None:
        print(text, flush=True)
    else:
        os.write(_RESULT_FD, (text + "\n").encode("utf-8"))


def _jsonable_args(args: argparse.Namespace) -> dict:
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def _dtype() -> torch.dtype:
    return torch.float16


def _default_device(path: str) -> torch.device:
    if path == "cuda":
        return torch.device("cuda:0")
    return torch.device("npu:0")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu":
        torch.npu.synchronize(device)


def _reset_peak(device: torch.device) -> None:
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        elif device.type == "npu" and hasattr(torch.npu, "reset_peak_memory_stats"):
            torch.npu.reset_peak_memory_stats(device)
    except Exception:
        return


def _peak_mb(device: torch.device) -> float | None:
    try:
        if device.type == "cuda":
            return float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
        if device.type == "npu" and hasattr(torch.npu, "max_memory_allocated"):
            return float(torch.npu.max_memory_allocated(device)) / (1024.0 * 1024.0)
    except Exception:
        return None
    return None


def _resolve_linear_cls(path: str, cuda_kernel: str):
    if path == "cannoe":
        os.environ.setdefault("GPTQMODEL_KOMODO_NATIVE_INT4", "1")
        os.environ.setdefault("GPTQMODEL_KOMODO_DROP_SOURCE_WEIGHTS", "1")
        from gptqmodel.nn_modules.qlinear.cannoe import CannoeLinear

        return CannoeLinear

    if cuda_kernel == "marlin":
        from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear

        return MarlinLinear
    if cuda_kernel == "machete":
        from gptqmodel.nn_modules.qlinear.machete import MacheteLinear

        return MacheteLinear
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear

    return TorchLinear


def _set_supported_act_order_g_idx(module, group_size: int) -> None:
    if group_size <= 0 or module.in_features % group_size != 0:
        raise ValueError("Synthetic act-order requires group_size to divide in_features.")
    groups = module.in_features // group_size
    natural = torch.arange(module.in_features, dtype=torch.int32) // group_size
    act_order = torch.arange(module.in_features).reshape(groups, group_size).t().reshape(-1)
    module.g_idx.copy_(natural[act_order].to(dtype=module.g_idx.dtype, device=module.g_idx.device))


def _fill_synthetic_gptq(module, *, device: torch.device, dtype: torch.dtype, seed: int, desc_act: bool) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        module.qweight.copy_(
            torch.randint(
                -(2**31),
                2**31 - 1,
                module.qweight.shape,
                dtype=torch.int32,
                device=device,
            )
        )
        if getattr(module, "qzeros", None) is not None and module.qzeros.numel() > 0:
            module.qzeros.zero_()
        module.scales.copy_(
            (torch.rand(module.scales.shape, dtype=dtype, device=device) * 0.04 + 0.01).to(module.scales.dtype)
        )
        module.g_idx.copy_(
            (torch.arange(module.in_features, dtype=torch.int32, device=device) // module.group_size).to(
                module.g_idx.dtype
            )
        )
        if desc_act:
            _set_supported_act_order_g_idx(module, module.group_size)
        if getattr(module, "bias", None) is not None:
            module.bias.copy_((torch.randn(module.bias.shape, dtype=dtype, device=device) * 0.03).to(module.bias.dtype))


def _build_module(cls, *, args: argparse.Namespace, case: LayerCase, device: torch.device, seed: int):
    dtype = _dtype()
    if args.path == "cuda" and args.cuda_kernel == "marlin" and not args.sym:
        raise ValueError("CUDA Marlin supports symmetric GPTQ only; rerun with --sym or use --cuda-kernel torch.")
    if args.path == "cuda" and args.cuda_kernel == "machete" and args.group_size == 32:
        raise ValueError("Machete in this repo does not support group_size=32; use --cuda-kernel marlin on A100.")

    module = cls(
        bits=4,
        group_size=args.group_size,
        sym=bool(args.sym),
        desc_act=bool(args.desc_act),
        in_features=case.in_features,
        out_features=case.out_features,
        bias=bool(args.bias),
        pack_dtype=torch.int32,
        register_buffers=True,
        dtype=dtype,
    ).to(device=device, dtype=dtype)
    _fill_synthetic_gptq(module, device=device, dtype=dtype, seed=seed, desc_act=bool(args.desc_act))
    module.optimized = True
    module.post_init()
    module.eval()
    if hasattr(module, "enable_weight_cache"):
        module.enable_weight_cache(False)
    if hasattr(module, "enable_source_weight_drop"):
        module.enable_source_weight_drop(True)
    return module


def _measure(fn, *, warmup: int, iters: int, device: torch.device) -> tuple[float, float]:
    with torch.inference_mode():
        first_start = time.perf_counter()
        fn()
        _sync(device)
        first_ms = (time.perf_counter() - first_start) * 1000.0

        for _ in range(warmup):
            fn()
        _sync(device)

        start = time.perf_counter()
        for _ in range(iters):
            fn()
        _sync(device)
    return first_ms, (time.perf_counter() - start) * 1000.0 / max(1, iters)


def _run_case(
    *,
    cls,
    args: argparse.Namespace,
    layer_key: str,
    tokens: int,
    device: torch.device,
    seed: int,
) -> dict:
    dtype = _dtype()
    case = QWEN3_27B_LAYERS[layer_key]
    _reset_peak(device)
    module = _build_module(cls, args=args, case=case, device=device, seed=seed)
    x = torch.randn(tokens, case.in_features, dtype=dtype, device=device)

    def forward():
        return module(x)

    first_ms, mean_ms = _measure(forward, warmup=args.warmup, iters=args.iters, device=device)
    y = forward()
    _sync(device)
    flops = 2.0 * tokens * case.in_features * case.out_features
    return {
        **asdict(case),
        "layer_key": layer_key,
        "tokens": tokens,
        "path": args.path,
        "cuda_kernel": args.cuda_kernel if args.path == "cuda" else None,
        "device": str(device),
        "dtype": "fp16",
        "group_size": args.group_size,
        "sym": bool(args.sym),
        "desc_act": bool(args.desc_act),
        "bias": bool(args.bias),
        "first_ms": first_ms,
        "mean_ms": mean_ms,
        "tflops": flops / (mean_ms * 1e9) if mean_ms > 0 else float("inf"),
        "output_shape": list(y.shape),
        "peak_allocated_mb": _peak_mb(device),
    }


def _format_table(rows: list[dict]) -> str:
    headers = ("layer", "M", "K", "N", "first_ms", "mean_ms", "TFLOP/s", "peak_MB")
    table = []
    for row in rows:
        peak = row["peak_allocated_mb"]
        table.append(
            (
                row["name"],
                str(row["tokens"]),
                str(row["in_features"]),
                str(row["out_features"]),
                f'{row["first_ms"]:.4f}',
                f'{row["mean_ms"]:.4f}',
                f'{row["tflops"]:.2f}',
                "" if peak is None else f"{peak:.1f}",
            )
        )
    widths = [len(item) for item in headers]
    for row in table:
        widths = [max(width, len(item)) for width, item in zip(widths, row)]
    lines = [
        "  ".join(item.ljust(width) for item, width in zip(headers, widths)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend("  ".join(item.ljust(width) for item, width in zip(row, widths)) for row in table)
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    _enable_quiet_cann_logs(args.path)
    if args.path == "cuda":
        os.environ["GPTQMODEL_MARLIN_USE_FP32"] = "1" if args.marlin_fp32_reduce else "0"

    device = torch.device(args.device) if args.device is not None else _default_device(args.path)
    if args.path == "cuda" and device.type != "cuda":
        raise ValueError("--path cuda requires a CUDA device, e.g. --device cuda:0")
    if args.path == "cannoe" and device.type != "npu":
        raise ValueError("--path cannoe requires an NPU device, e.g. --device npu:0")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if device.type == "npu":
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("Ascend NPU is not available.")
        torch.npu.set_device(device)
    elif device.type == "cuda":
        torch.cuda.set_device(device)

    cls = _resolve_linear_cls(args.path, args.cuda_kernel)
    rows = []
    for token_index, tokens in enumerate(args.tokens):
        for layer_index, layer_key in enumerate(args.layers):
            rows.append(
                _run_case(
                    cls=cls,
                    args=args,
                    layer_key=layer_key,
                    tokens=tokens,
                    device=device,
                    seed=args.seed + token_index * 100 + layer_index,
                )
            )

    _emit(
        f"path={args.path}"
        f" cuda_kernel={args.cuda_kernel if args.path == 'cuda' else '-'}"
        f" device={device} dtype=fp16 group_size={args.group_size}"
        f" sym={args.sym} desc_act={args.desc_act} bias={args.bias}"
    )
    _emit(_format_table(rows))

    by_tokens: dict[int, dict[str, float]] = {}
    for row in rows:
        bucket = by_tokens.setdefault(row["tokens"], {"mean_ms": 0.0, "first_ms": 0.0})
        bucket["mean_ms"] += row["mean_ms"]
        bucket["first_ms"] += row["first_ms"]
    for tokens, totals in by_tokens.items():
        _emit(f"TOTAL M={tokens}: first_sum={totals['first_ms']:.4f}ms mean_sum={totals['mean_ms']:.4f}ms")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps({"args": _jsonable_args(args), "rows": rows, "totals_by_tokens": by_tokens}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
