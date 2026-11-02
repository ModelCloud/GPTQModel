#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Stage-by-stage quantization benchmark for RDNA4 (gfx1200/gfx1201/gfx1250).

Measures, per quantization method (GPTQ / AWQ / QQQ / ParoQuant):

* load time        - ``GPTQModel.load`` with the quantization config attached
* calibration time - tokenizer encode of the built-in calibration text
* quantize time    - ``model.quantize()`` (the stage the RDNA4 tuning targets)
* save time        - ``model.save()``
* peak VRAM        - ``torch.cuda.max_memory_allocated`` after every stage
* BLAS backend     - whether the quantization window actually runs on rocBLAS

The BLAS probe enters ``gptqmodel.utils.rocm.quant_blas_backend()`` and reads
``torch.backends.cuda.preferred_blas_library()`` inside it, so it reports the
backend that a real quantize would use, not just the user's torch default.
hipBLASLt ships gfx1200/gfx1201 Tensile libraries but is *not* tuned for RDNA4;
rocBLAS is the default and hipBLASLt is only an experimental opt-in via
``GPTQMODEL_ROCM_QUANT_BLAS=hipblaslt``.

This script never enables ``GPTQ_TORCH_CACHE_WEIGHTS``: quantization benchmarks
measure the packing path, and the dense-weight cache is forbidden for inference.

No GPU is required for ``--dry-run``: the plan (arguments, stages, arch probe,
BLAS probe, environment) is printed and the process exits 0.  Without a visible
GPU the same plan is printed with ``"gpu": "unavailable"`` and exit 0 as well.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


DEFAULT_MODEL = "/monster/data/model/Llama-3.2-1B-Instruct"
METHOD_CHOICES = ("gptq", "awq", "qqq", "paroquant")

# Deterministic calibration text: this benchmark measures wall time and VRAM,
# not quantization quality, so a network/dataset dependency would only add
# noise.  Swap in a real corpus when quality matters.
DEFAULT_CALIBRATION_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Quantization maps a dense weight matrix onto a small codebook and a set of "
    "per-group scales, so the kernel must decode every code before it can be "
    "multiplied by an activation. Numeric stability depends on the damping term, "
    "and the Hessian inverse is updated once per column block. "
) * 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage timings, peak VRAM and BLAS backend for RDNA4 quantization."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Local model path or hub id.")
    parser.add_argument("--method", default="gptq", choices=METHOD_CHOICES)
    parser.add_argument("--bits", type=int, default=4, choices=(2, 4, 8))
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--calibration-rows", type=int, default=64)
    parser.add_argument("--calibration-seq-len", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/gptqmodel-rdna4-quant"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit 0.")
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _device_report(device: torch.device) -> dict[str, Any]:
    """Architecture metadata, using gcnArchName (never get_device_capability)."""
    report: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "is_rocm": torch.version.hip is not None,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "name": None,
        "arch": None,
        "arch_class": None,
        "is_rdna4": None,
        "fused_arch": None,
        "wave_size": None,
        "gate_limits": None,
        "error": None,
    }
    if not report["cuda_available"]:
        return report

    try:
        report["name"] = torch.cuda.get_device_name(device)
    except Exception as exc:  # pragma: no cover - runtime dependent
        report["error"] = f"get_device_name failed: {type(exc).__name__}: {exc}"
        return report

    try:
        from gptqmodel.utils import arch

        report["arch"] = arch.rocm_arch(device)
        report["arch_class"] = arch.arch_class(device)
        report["is_rdna4"] = arch.is_rdna4(device)
        report["fused_arch"] = arch.is_rocm_fused_arch(device)
        report["wave_size"] = arch.wave_size(device)
        report["gate_limits"] = list(arch.gate_limits(device))
    except Exception as exc:  # pragma: no cover - runtime dependent
        report["error"] = f"arch probe failed: {type(exc).__name__}: {exc}"
    return report


def _blas_report() -> dict[str, Any]:
    """Report the BLAS backend that a quantize window would actually select."""
    try:
        from gptqmodel.utils import rocm
    except Exception as exc:  # pragma: no cover - import failure is informative
        return {"is_rocm": None, "error": f"import gptqmodel.utils.rocm failed: {exc}"}

    requested = os.getenv(rocm.QUANT_BLAS_BACKEND_ENV, rocm.QUANT_BLAS_BACKEND_DEFAULT)
    default_backend = rocm.preferred_blas_library()
    effective = default_backend
    error = None
    if rocm.IS_ROCM:
        try:
            with rocm.quant_blas_backend():
                effective = rocm.preferred_blas_library()
        except Exception as exc:  # pragma: no cover - runtime dependent
            error = f"{type(exc).__name__}: {exc}"
    return {
        "is_rocm": rocm.IS_ROCM,
        "requested": requested,
        "torch_default": None if default_backend is None else str(default_backend),
        "effective_during_quantize": None if effective is None else str(effective),
        "error": error,
    }


def _quant_config(method: str, bits: int, group_size: int):
    """Build the method-specific quantize config through the public factory."""
    from gptqmodel.quantization.config import FORMAT, METHOD, QuantizeConfig

    method_format = {
        "gptq": (METHOD.GPTQ, FORMAT.GPTQ),
        "awq": (METHOD.AWQ, FORMAT.GEMM),
        "qqq": (METHOD.QQQ, FORMAT.QQQ),
        "paroquant": (METHOD.PARO, FORMAT.PAROQUANT),
    }[method]
    kwargs: dict[str, Any] = {
        "method": method_format[0],
        "format": method_format[1],
        "bits": bits,
        "group_size": group_size,
    }
    if method == "paroquant":
        kwargs["krot"] = 8
    return QuantizeConfig(**kwargs)


def _load_calibration(tokenizer, rows: int, seq_len: int) -> list[list[int]]:
    ids = tokenizer(DEFAULT_CALIBRATION_TEXT, return_tensors="pt", add_special_tokens=False).input_ids[0]
    needed = rows * seq_len
    if ids.numel() < needed:
        ids = ids.repeat((needed + ids.numel() - 1) // ids.numel())
    return [ids[start : start + seq_len].tolist() for start in range(0, needed, seq_len)]


def _plan(args: argparse.Namespace, reason: str) -> dict[str, Any]:
    device = torch.device(args.device)
    return {
        "script": "benchmark_rdna4_quant.py",
        "mode": "dry-run",
        "reason": reason,
        "model": args.model,
        "method": args.method,
        "bits": args.bits,
        "group_size": args.group_size,
        "dtype": args.dtype,
        "device": str(device),
        "calibration_rows": args.calibration_rows,
        "calibration_seq_len": args.calibration_seq_len,
        "output_dir": str(args.output_dir),
        "output_json": None if args.output_json is None else str(args.output_json),
        "stages": [
            "tokenizer load",
            "calibration tokenize",
            "model load (GPTQModel.load with quantize config)",
            "quantize (model.quantize)",
            "save (model.save)",
        ],
        "recorded": ["seconds per stage", "peak VRAM MiB per stage", "BLAS backend per stage"],
        "env": {
            "GPTQMODEL_ROCM_QUANT_BLAS": os.getenv("GPTQMODEL_ROCM_QUANT_BLAS", "<unset -> rocblas>"),
            "GPTQ_TORCH_CACHE_WEIGHTS": os.getenv("GPTQ_TORCH_CACHE_WEIGHTS", "0"),
        },
        "device_info": _device_report(device),
        "blas": _blas_report(),
        "commands": {
            "rocm": (
                "/root/miniconda3/envs/gp_314t_rocm/bin/python scripts/benchmark_rdna4_quant.py "
                f"--model {args.model} --method {args.method} --bits {args.bits} "
                f"--group-size {args.group_size} --device {args.device}"
            ),
            "cuda": (
                "/root/miniconda3/envs/gp_314t/bin/python scripts/benchmark_rdna4_quant.py "
                f"--model {args.model} --method {args.method} --bits {args.bits} "
                f"--group-size {args.group_size} --device {args.device}"
            ),
        },
    }


def _print_plan(plan: dict[str, Any]) -> None:
    print(json.dumps(plan, indent=2, sort_keys=True))
    device_info = plan["device_info"]
    print(
        "plan: "
        f"method={plan['method']} bits={plan['bits']} group_size={plan['group_size']} "
        f"device={plan['device']} arch={device_info.get('arch')} class={device_info.get('arch_class')} "
        f"fused_arch={device_info.get('fused_arch')} "
        f"blas_effective={plan['blas'].get('effective_during_quantize')}",
        flush=True,
    )


def _sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _timed(device: torch.device, fn) -> tuple[float, Any]:
    _sync(device)
    started = time.perf_counter()
    result = fn()
    _sync(device)
    return time.perf_counter() - started, result


def run(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    from gptqmodel import GPTQModel
    from gptqmodel.utils.backend import BACKEND

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = _dtype(args.dtype)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    config = _quant_config(args.method, args.bits, args.group_size)
    rows: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(device)

    def record(stage: str, seconds: float) -> None:
        rows.append(
            {
                "stage": stage,
                "seconds": round(seconds, 4),
                "allocated_mib": round(torch.cuda.memory_allocated(device) / 2**20, 2),
                "peak_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 2),
            }
        )
        print(json.dumps(rows[-1], sort_keys=True), flush=True)

    seconds, tokenizer = _timed(device, lambda: AutoTokenizer.from_pretrained(args.model))
    record("tokenizer_load", seconds)

    seconds, calibration = _timed(
        device, lambda: _load_calibration(tokenizer, args.calibration_rows, args.calibration_seq_len)
    )
    record("calibration_tokenize", seconds)

    blas_before = _blas_report()
    seconds, model = _timed(
        device,
        lambda: GPTQModel.load(
            args.model,
            quantize_config=config,
            dtype=dtype,
            device_map={"": str(device)},
        ),
    )
    record("model_load", seconds)

    seconds, _ = _timed(device, lambda: model.quantize(calibration, batch_size=1, backend=BACKEND.AUTO))
    record("quantize", seconds)
    blas_during = _blas_report()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seconds, _ = _timed(device, lambda: model.save(args.output_dir))
    record("save", seconds)

    payload = {
        "mode": "measured",
        "model": args.model,
        "method": args.method,
        "bits": args.bits,
        "group_size": args.group_size,
        "dtype": args.dtype,
        "device": str(device),
        "device_info": _device_report(device),
        "blas_before_quantize": blas_before,
        "blas_after_quantize": blas_during,
        "output_dir": str(args.output_dir),
        "rows": rows,
        "total_seconds": round(sum(row["seconds"] for row in rows), 4),
        "peak_vram_mib": max(row["peak_mib"] for row in rows),
        "dense_weight_cache_enabled": os.getenv("GPTQ_TORCH_CACHE_WEIGHTS", "0") == "1",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def main() -> int:
    args = parse_args()
    if args.dry_run:
        _print_plan(_plan(args, "requested via --dry-run"))
        return 0
    if not torch.cuda.is_available():
        _print_plan(_plan(args, "no CUDA/HIP device visible; nothing measured"))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
