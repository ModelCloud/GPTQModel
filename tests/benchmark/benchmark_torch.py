#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import json
import os
import statistics
import subprocess
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


RESULT_PREFIX = "RESULT_JSON\t"


@dataclass(frozen=True)
class BenchCase:
    case_id: str
    pin: str
    cuda_visible: str | None
    cuda_order: str
    triton: str
    reps: str
    runs: int
    iters: int


FULL_CASES: list[BenchCase] = [
    BenchCase("cpu-single", "CPU", None, "N/A", "N/A", "1", 1, 200),
    BenchCase("gpu0-default-on-single", "GPU0", "0", "default", "on", "1", 1, 500),
    BenchCase("gpu0-default-off-single", "GPU0", "0", "default", "off", "1", 1, 500),
    BenchCase("gpu6-pci-on-single-a", "GPU6", "6", "PCI_BUS_ID", "on", "1", 1, 500),
    BenchCase("gpu6-pci-off-single-a", "GPU6", "6", "PCI_BUS_ID", "off", "1", 1, 500),
    BenchCase("gpu0-pci-on-single", "GPU0", "0", "PCI_BUS_ID", "on", "1", 1, 500),
    BenchCase("gpu0-pci-off-single", "GPU0", "0", "PCI_BUS_ID", "off", "1", 1, 500),
    BenchCase("gpu6-pci-on-single-b", "GPU6", "6", "PCI_BUS_ID", "on", "1", 1, 500),
    BenchCase("gpu6-pci-off-single-b", "GPU6", "6", "PCI_BUS_ID", "off", "1", 1, 500),
    BenchCase("gpu0-pci-on-median", "GPU0", "0", "PCI_BUS_ID", "on", "5-med", 5, 800),
    BenchCase("gpu0-pci-off-median", "GPU0", "0", "PCI_BUS_ID", "off", "5-med", 5, 800),
    BenchCase("gpu6-pci-on-median", "GPU6", "6", "PCI_BUS_ID", "on", "5-med", 5, 800),
    BenchCase("gpu6-pci-off-median", "GPU6", "6", "PCI_BUS_ID", "off", "5-med", 5, 800),
    BenchCase("cpu-median", "CPU", None, "N/A", "N/A", "5-med", 5, 200),
]


def _stub_pkg(name: str, path: Path) -> types.ModuleType:
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(path)]
    pkg.__spec__ = spec
    sys.modules[name] = pkg
    return pkg


def _load_baseline_and_current(repo_root: Path):
    pkg_root = repo_root / "gptqmodel"

    pkg = _stub_pkg("gptqmodel", pkg_root)
    pkg.DEBUG_ON = False
    _stub_pkg("gptqmodel.models", pkg_root / "models")

    importlib.import_module("gptqmodel.nn_modules.qlinear")
    current_mod = importlib.import_module("gptqmodel.nn_modules.qlinear.torch")
    current_cls = current_mod.TorchLinear

    baseline_src = subprocess.check_output(
        ["git", "show", "HEAD:gptqmodel/nn_modules/qlinear/torch.py"],
        cwd=str(repo_root),
        text=True,
    )
    baseline_name = f"gptqmodel.nn_modules.qlinear._torch_baseline_{os.getpid()}_{time.time_ns()}"
    baseline_mod = types.ModuleType(baseline_name)
    baseline_mod.__file__ = "<baseline_torch.py>"
    baseline_mod.__package__ = "gptqmodel.nn_modules.qlinear"
    sys.modules[baseline_name] = baseline_mod
    exec(compile(baseline_src, baseline_mod.__file__, "exec"), baseline_mod.__dict__)
    baseline_cls = baseline_mod.TorchLinear

    return baseline_cls, current_cls


def _mock_gptq_linear(
    bits: int,
    group_size: int,
    in_features: int,
    out_features: int,
    device: str,
):
    import torch
    import torch.nn as nn

    maxq = (1 << (bits - 1)) - 1
    weight = torch.randn((in_features, out_features), dtype=torch.float32, device=device)
    reshaped = weight.view(in_features // group_size, group_size, out_features)
    w_g = reshaped.permute(1, 0, 2).reshape(group_size, -1)
    scales = torch.maximum(
        w_g.abs().max(dim=0, keepdim=True).values,
        torch.full((1, w_g.shape[1]), 1e-6, device=device),
    )
    scales = scales / maxq
    q = torch.round(w_g / scales).clamp_(-maxq, maxq)
    ref = (q * scales).to(dtype=torch.float16)
    ref = ref.reshape(group_size, -1, out_features).permute(1, 0, 2).reshape(in_features, out_features)

    linear = nn.Linear(in_features, out_features, bias=False, device=device, dtype=torch.float16)
    linear.weight.data = ref.t().contiguous()

    scales_cpu = scales.reshape(-1, out_features).contiguous().cpu()
    zeros = torch.zeros_like(scales_cpu, dtype=torch.int32)
    g_idx = (torch.arange(in_features, dtype=torch.int32, device=device) // group_size).cpu()
    return linear.cpu(), scales_cpu, zeros, g_idx


def _build_module(
    cls,
    linear,
    scales,
    zeros,
    g_idx,
    bits: int,
    group_size: int,
    in_features: int,
    out_features: int,
    device: str,
):
    import torch

    mod = cls(
        bits=bits,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        pack_dtype=torch.int32,
        bias=False,
    )
    # Keep benchmark deterministic and avoid compile overhead noise.
    mod.optimize = lambda *args, **kwargs: None
    mod.pack_block(linear, scales.T, zeros.T, g_idx=g_idx)
    mod.post_init()
    mod.eval()
    return mod.to(device=device)


def _bench_ms(fn, iters: int, device: str):
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iters):
            fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e3 / iters


def _worker(case: dict[str, Any]) -> dict[str, Any]:
    import torch

    repo_root = Path(__file__).resolve().parent
    result: dict[str, Any] = {
        "case_id": case["case_id"],
        "pin": case["pin"],
        "cuda_order": case["cuda_order"],
        "triton": case["triton"],
        "reps": case["reps"],
        "base_ms": None,
        "new_ms": None,
        "delta_pct": None,
        "max_abs_diff": None,
        "torch_dev": "N/A",
        "error": None,
    }

    try:
        if case["pin"] == "CPU":
            device = "cpu"
            torch_dev = "CPU"
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            device = "cuda:0"
            torch_dev = torch.cuda.get_device_name(0)

        baseline_cls, current_cls = _load_baseline_and_current(repo_root)

        bits = 4
        group_size = 128
        in_features = 4096
        out_features = 4096

        torch.manual_seed(123)
        linear, scales, zeros, g_idx = _mock_gptq_linear(
            bits=bits,
            group_size=group_size,
            in_features=in_features,
            out_features=out_features,
            device=device,
        )

        base_mod = _build_module(
            baseline_cls,
            linear,
            scales,
            zeros,
            g_idx,
            bits=bits,
            group_size=group_size,
            in_features=in_features,
            out_features=out_features,
            device=device,
        )
        new_mod = _build_module(
            current_cls,
            linear,
            scales,
            zeros,
            g_idx,
            bits=bits,
            group_size=group_size,
            in_features=in_features,
            out_features=out_features,
            device=device,
        )

        with torch.inference_mode():
            w0 = base_mod.dequantize_weight(num_itr=1)
            w1 = new_mod.dequantize_weight(num_itr=1)
        max_abs_diff = float((w0 - w1).abs().max().item())

        warmup = 60 if case["runs"] > 1 else 40
        with torch.inference_mode():
            for _ in range(warmup):
                base_mod.dequantize_weight(num_itr=1)
                new_mod.dequantize_weight(num_itr=1)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        base_samples = [_bench_ms(lambda: base_mod.dequantize_weight(num_itr=1), case["iters"], device) for _ in range(case["runs"])]
        new_samples = [_bench_ms(lambda: new_mod.dequantize_weight(num_itr=1), case["iters"], device) for _ in range(case["runs"])]

        if case["runs"] > 1:
            base_ms = float(statistics.median(base_samples))
            new_ms = float(statistics.median(new_samples))
        else:
            base_ms = float(base_samples[0])
            new_ms = float(new_samples[0])

        delta_pct = float((base_ms - new_ms) / base_ms * 100.0) if base_ms != 0 else 0.0
        result.update(
            {
                "base_ms": base_ms,
                "new_ms": new_ms,
                "delta_pct": delta_pct,
                "max_abs_diff": max_abs_diff,
                "torch_dev": torch_dev,
                "base_samples": base_samples,
                "new_samples": new_samples,
            }
        )
    except Exception as exc:  # pragma: no cover - benchmark runtime errors are environment dependent
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def _run_case_subprocess(case: BenchCase) -> dict[str, Any]:
    env = os.environ.copy()

    if case.pin == "CPU":
        env["CUDA_VISIBLE_DEVICES"] = ""
        env.pop("CUDA_DEVICE_ORDER", None)
        env.pop("GPTQ_TORCH_TRITON_DEQUANT", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = case.cuda_visible or ""
        if case.cuda_order == "default":
            env.pop("CUDA_DEVICE_ORDER", None)
        else:
            env["CUDA_DEVICE_ORDER"] = case.cuda_order

        if case.triton == "off":
            env["GPTQ_TORCH_TRITON_DEQUANT"] = "0"
        elif case.triton == "on":
            env.pop("GPTQ_TORCH_TRITON_DEQUANT", None)

    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(asdict(case))]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")

    parsed: dict[str, Any] | None = None
    for line in reversed(combined.splitlines()):
        if line.startswith(RESULT_PREFIX):
            parsed = json.loads(line[len(RESULT_PREFIX):])
            break

    if parsed is None:
        parsed = {
            "case_id": case.case_id,
            "pin": case.pin,
            "cuda_order": case.cuda_order,
            "triton": case.triton,
            "reps": case.reps,
            "base_ms": None,
            "new_ms": None,
            "delta_pct": None,
            "max_abs_diff": None,
            "torch_dev": "N/A",
            "error": f"Worker parse failure (rc={proc.returncode})",
        }

    if proc.returncode != 0 and not parsed.get("error"):
        parsed["error"] = f"Worker exited with code {proc.returncode}"

    return parsed


def _fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def _fmt_delta(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):+0.2f}"


def _ascii_table(rows: list[dict[str, Any]]) -> str:
    headers = ["#", "Pin", "Torch Dev", "CUDA_ORDER", "Triton", "Reps", "Base ms", "New ms", "Delta %", "MaxAbsDiff"]
    table_rows: list[list[str]] = []

    for idx, row in enumerate(rows, start=1):
        if row.get("error"):
            base_s = "ERR"
            new_s = "ERR"
            delta_s = "ERR"
            diff_s = "ERR"
            torch_dev = f"{row.get('torch_dev', 'N/A')} ({row['error']})"
        else:
            base_s = _fmt_float(row.get("base_ms"))
            new_s = _fmt_float(row.get("new_ms"))
            delta_s = _fmt_delta(row.get("delta_pct"))
            diff_s = _fmt_float(row.get("max_abs_diff"), digits=1)
            torch_dev = row.get("torch_dev", "N/A")

        table_rows.append(
            [
                str(idx),
                row.get("pin", "N/A"),
                torch_dev,
                row.get("cuda_order", "N/A"),
                row.get("triton", "N/A"),
                row.get("reps", "N/A"),
                base_s,
                new_s,
                delta_s,
                diff_s,
            ]
        )

    widths = [len(h) for h in headers]
    for r in table_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def mk_border() -> str:
        return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def mk_row(cols: list[str]) -> str:
        return "| " + " | ".join(cols[i].ljust(widths[i]) for i in range(len(cols))) + " |"

    lines = [mk_border(), mk_row(headers), mk_border()]
    lines.extend(mk_row(r) for r in table_rows)
    lines.append(mk_border())
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A/B benchmark for TorchLinear dequant path.")
    p.add_argument("--worker", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated case ids or pin names (CPU,GPU0,GPU6).",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer iterations/runs for a quicker smoke benchmark.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Also print raw JSON results after the ASCII table.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if args.worker is not None:
        case = json.loads(args.worker)
        result = _worker(case)
        print(RESULT_PREFIX + json.dumps(result, sort_keys=True))
        return 0 if result.get("error") is None else 1

    cases = FULL_CASES[:]
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        cases = [c for c in cases if c.case_id in wanted or c.pin in wanted]
        if not cases:
            print("No cases selected. Check --only values.", file=sys.stderr)
            return 2

    if args.quick:
        quick_cases = []
        for c in cases:
            runs = min(c.runs, 2)
            iters = max(50, c.iters // 4)
            reps = c.reps if runs == c.runs else f"{runs}-quick"
            quick_cases.append(
                BenchCase(
                    case_id=c.case_id,
                    pin=c.pin,
                    cuda_visible=c.cuda_visible,
                    cuda_order=c.cuda_order,
                    triton=c.triton,
                    reps=reps,
                    runs=runs,
                    iters=iters,
                )
            )
        cases = quick_cases

    print("Running cases:")
    for c in cases:
        print(f"  - {c.case_id}")

    rows = []
    for c in cases:
        rows.append(_run_case_subprocess(c))

    print()
    print("baseline = HEAD:gptqmodel/nn_modules/qlinear/torch.py")
    print("new = current working tree gptqmodel/nn_modules/qlinear/torch.py")
    print()
    print(_ascii_table(rows))

    if args.json:
        print()
        print(json.dumps(rows, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
