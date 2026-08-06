#!/usr/bin/env python3
"""Synthetic per-layer memory/speed baseline for adaptive damping + clipping.

Example:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<GPU_INDEX or UUID> \
        PYTHONPATH=<path-to-GPT-QModel-Ultra-repo> \
        python scripts/profile_adaptive_memory.py
"""

import argparse
import gc
import json
import os
import subprocess
import time


def _preflight_gpu(samples: int = 3, interval: float = 1.0):
    """Require 0% compute and low baseline memory on CUDA_VISIBLE_DEVICES for the run."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    visible_tokens = [t.strip() for t in visible.split(",") if t.strip()]

    def _is_visible(idx: str, uuid: str) -> bool:
        for token in visible_tokens:
            if token.startswith("GPU-") or token.startswith("MIG-"):
                if token == uuid:
                    return True
            elif token == idx:
                return True
        return False

    for i in range(samples):
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,pci.bus_id,uuid,name,utilization.gpu,memory.used",
                "--format=csv,noheader",
            ],
            text=True,
        )
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        matched = False
        for line in lines:
            idx, pci, uuid, name, util, mem = (x.strip() for x in line.split(",", 5))
            if not _is_visible(idx, uuid):
                continue
            matched = True
            util_i = int(util.replace("%", "").strip())
            mem_i = int(mem.replace("MiB", "").strip())
            if util_i != 0 or mem_i > 500:
                raise RuntimeError(
                    f"GPU {idx} ({name} {uuid}) not idle: util={util}, mem={mem_i}MiB"
                )
        if not matched:
            raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible} did not match an nvidia-smi device")
        if i + 1 < samples:
            time.sleep(interval)
    print(f"[preflight] CUDA_VISIBLE_DEVICES={visible} idle OK")


def _build_case(rows: int, cols: int, batch: int):
    import torch
    import torch.nn as nn

    from gptqmodel.quantization.config import AdaptiveClippingConfig, AdaptiveDampingConfig, QuantizeConfig
    from gptqmodel.quantization.gptq import GPTQ

    device = torch.device("cuda:0")  # visible device mapped by CUDA_VISIBLE_DEVICES
    layer = nn.Linear(cols, rows, bias=False, dtype=torch.bfloat16, device=device).eval()
    qcfg = QuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False,
        act_group_aware=True,
        # These algorithms are intentionally opt-in. The benchmark must request
        # them explicitly or it measures the static GPTQ path instead of PR 211.
        adaptive_damping=AdaptiveDampingConfig(),
        adaptive_clipping=AdaptiveClippingConfig(),
    )
    gptq = GPTQ(layer, qcfg=qcfg)
    gptq.quantizer.configure(perchannel=True)

    inp = torch.randn(batch, cols, device=device, dtype=torch.bfloat16)
    gptq.add_batch(inp, None)
    gptq.finalize_hessian()
    return device, gptq, inp


def _discarded_warmup(rows: int, cols: int, batch: int):
    import torch

    device, gptq, inp = _build_case(rows, cols, batch)
    gptq.quantize()
    torch.cuda.synchronize(device)
    del inp, gptq
    gc.collect()
    torch.cuda.empty_cache()


def _run(rows: int, cols: int, label: str, batch: int = 256):
    import torch

    # Use a separate GPTQ instance because quantize consumes its Hessian.
    _discarded_warmup(rows, cols, batch)
    device, gptq, inp = _build_case(rows, cols, batch)

    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    qweight, scale, zero, g_idx, *_ = gptq.quantize()
    end.record()
    end.synchronize()
    duration_ms = start.elapsed_time(end)
    peak = torch.cuda.max_memory_allocated(device)
    result = {
        "label": label,
        "rows": rows,
        "cols": cols,
        "dtype": "bfloat16",
        "calibration_batch": 1,
        "calibration_rows": batch,
        "group_size": 128,
        "latency_ms": duration_ms,
        "throughput_mweights_s": (rows * cols) / (duration_ms * 1000.0),
        "baseline_gib": baseline / 1024**3,
        "peak_gib": peak / 1024**3,
        "incremental_peak_gib": (peak - baseline) / 1024**3,
        "qweight_shape": list(qweight.shape),
    }
    del inp, gptq, qweight, scale, zero, g_idx
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _load_baseline(path):
    if path is None:
        return {}, None
    with open(path) as handle:
        payload = json.load(handle)
    return {row["label"]: row for row in payload["results"]}, payload.get("sm")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-json", help="Results JSON recorded from the comparison revision.")
    parser.add_argument(
        "--max-peak-regression-pct",
        type=float,
        default=None,
        help="Fail when peak allocated VRAM exceeds a matching baseline by this percentage.",
    )
    parser.add_argument("--output-json", help="Write machine-readable results for a later comparison.")
    args = parser.parse_args()

    _preflight_gpu()
    import torch

    from gptqmodel.utils.logger import render_table

    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    baseline, baseline_sm = _load_baseline(args.baseline_json)
    if baseline_sm is not None:
        actual_sm = ".".join(str(value) for value in torch.cuda.get_device_capability(0))
        if actual_sm != str(baseline_sm):
            raise RuntimeError(f"VRAM baseline requires sm_{baseline_sm}, but the visible GPU is sm_{actual_sm}.")
    # Qwen3-8B-like MLP shapes
    results = [
        _run(4096, 12288, "down_proj"),  # mlp.down_proj
        _run(12288, 4096, "gate_proj"),  # mlp.gate/up_proj
    ]

    rows = []
    regressions = []
    for result in results:
        reference = baseline.get(result["label"])
        reference_ms = reference.get("latency_ms") if reference is not None else None
        speedup = reference_ms / result["latency_ms"] if reference_ms is not None else None
        rows.append([
            result["label"],
            f"{result['rows']}x{result['cols']}",
            result["dtype"],
            f"{result['calibration_batch']}x{result['calibration_rows']}",
            result["group_size"],
            f"{result['latency_ms']:.2f}",
            f"{result['throughput_mweights_s']:.2f}",
            f"{result['baseline_gib']:.3f}",
            f"{result['peak_gib']:.3f}",
            f"{result['incremental_peak_gib']:.3f}",
            f"{reference_ms:.2f}" if reference_ms is not None else "n/a",
            f"{speedup:.3f}x" if speedup is not None else "n/a",
        ])
        if reference is not None and args.max_peak_regression_pct is not None:
            allowed_peak = reference["peak_gib"] * (1.0 + args.max_peak_regression_pct / 100.0)
            if result["peak_gib"] > allowed_peak:
                regressions.append(
                    f"{result['label']} peak VRAM regressed: {result['peak_gib']:.3f} GiB > "
                    f"{allowed_peak:.3f} GiB ({args.max_peak_regression_pct:.1f}% gate)."
                )
    print(render_table(
        rows,
        headers=[
            "module",
            "shape",
            "dtype",
            "calib BxT",
            "group",
            "latency ms",
            "Mweights/s",
            "base GiB",
            "peak GiB",
            "delta GiB",
            "ref ms",
            "speedup",
        ],
    ))

    if regressions:
        raise RuntimeError(" ".join(regressions))

    if args.output_json:
        with open(args.output_json, "w") as handle:
            capability = torch.cuda.get_device_capability(0)
            payload = {
                "device": torch.cuda.get_device_name(0),
                "sm": ".".join(str(value) for value in capability),
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "results": results,
            }
            json.dump(payload, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
