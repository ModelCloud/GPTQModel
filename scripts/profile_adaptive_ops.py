#!/usr/bin/env python3
"""High-level per-operation GPU timing for adaptive damping + clipping vs main.

Run with an explicitly selected idle GPU index or UUID in
``CUDA_VISIBLE_DEVICES``.

Uses CUDA events and a call-stack accumulator to report exclusive GPU time per
named operation.  Only the final synchronize adds host/device sync overhead.
"""

import argparse
import functools
import gc
import json
import os
import subprocess
import time


def _preflight_gpu(samples: int = 3, interval: float = 1.0):
    """Require stable 0% compute and low memory before collecting timing."""

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    visible_tokens = [token.strip() for token in visible.split(",") if token.strip()]

    def _is_visible(index: str, uuid: str) -> bool:
        return any(token == uuid if token.startswith(("GPU-", "MIG-")) else token == index for token in visible_tokens)

    for sample in range(samples):
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,pci.bus_id,uuid,name,utilization.gpu,memory.used",
                "--format=csv,noheader",
            ],
            text=True,
        )
        matched = False
        for line in out.splitlines():
            if not line.strip():
                continue
            index, pci, uuid, name, utilization, memory = (value.strip() for value in line.split(",", 5))
            if not _is_visible(index, uuid):
                continue
            matched = True
            utilization_value = int(utilization.replace("%", "").strip())
            memory_mib = int(memory.replace("MiB", "").strip())
            if utilization_value != 0 or memory_mib > 500:
                raise RuntimeError(
                    f"GPU {index} ({name} {uuid}) not idle: util={utilization}, mem={memory_mib}MiB"
                )
        if not matched:
            raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible} did not match an nvidia-smi device")
        if sample + 1 < samples:
            time.sleep(interval)
    print(f"[preflight] CUDA_VISIBLE_DEVICES={visible} idle OK")


class _CudaTimer:
    """Exclusive CUDA timer with nested-call child subtraction."""

    def __init__(self):
        self.frames = []
        self.stack = []
        self.inclusive = {}
        self.exclusive = {}
        self.counts = {}

    def wrap(self, fn, name):
        import torch

        timer = self

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            parent = timer.stack[-1] if timer.stack else None
            frame = {"name": name, "start": start, "end": end, "parent": parent, "children_ms": 0.0}
            timer.frames.append(frame)
            timer.stack.append(frame)
            try:
                return fn(*args, **kwargs)
            finally:
                end.record()
                timer.stack.pop()

        return wrapper

    def finalize(self):
        import torch

        torch.cuda.synchronize()
        # Process in reverse order so children are subtracted from parents first.
        for frame in reversed(self.frames):
            elapsed = frame["start"].elapsed_time(frame["end"])
            name = frame["name"]
            self.inclusive[name] = self.inclusive.get(name, 0.0) + elapsed
            self.counts[name] = self.counts.get(name, 0) + 1
            exclusive = elapsed - frame["children_ms"]
            self.exclusive[name] = self.exclusive.get(name, 0.0) + exclusive
            parent = frame["parent"]
            if parent is not None:
                parent["children_ms"] += elapsed
        return self

    def results(self, total_ms):
        results = []
        names = sorted(self.exclusive.keys(), key=lambda n: self.exclusive[n], reverse=True)
        for name in names:
            results.append({
                "operation": name,
                "calls": self.counts[name],
                "inclusive_ms": self.inclusive[name],
                "exclusive_ms": self.exclusive[name],
            })
        measured = sum(self.exclusive.values())
        results.append({
            "operation": "measured_total",
            "calls": "",
            "inclusive_ms": measured,
            "exclusive_ms": measured,
        })
        results.append({
            "operation": "unmeasured/rest",
            "calls": "",
            "inclusive_ms": total_ms - measured,
            "exclusive_ms": total_ms - measured,
        })
        return results


def _build_case(rows: int, cols: int, batch: int):
    import torch
    import torch.nn as nn

    from gptqmodel.quantization.config import AdaptiveClippingConfig, AdaptiveDampingConfig, QuantizeConfig
    from gptqmodel.quantization.gptq import GPTQ

    device = torch.device("cuda:0")  # selected by CUDA_VISIBLE_DEVICES
    layer = nn.Linear(cols, rows, bias=False, dtype=torch.bfloat16, device=device).eval()
    qcfg = QuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False,
        act_group_aware=True,
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

    # Use a disposable instance because quantization consumes the Hessian.
    _discarded_warmup(rows, cols, batch)
    device, gptq, inp = _build_case(rows, cols, batch)

    timer = _CudaTimer()
    # Wrap the functions we want broken out.  Note: assigning a plain function to
    # an instance attribute replaces the bound method; the wrapper receives the
    # same positional arguments the original bound method was called with.
    gptq.hessian_inverse = timer.wrap(gptq.hessian_inverse, "hessian_inverse")
    gptq.quantizer.find_params = timer.wrap(gptq.quantizer.find_params, "find_params")
    gptq.quantizer.find_params_batched = timer.wrap(gptq.quantizer.find_params_batched, "find_params_batched")
    gptq.quantizer.adaptive_clip_search = timer.wrap(gptq.quantizer.adaptive_clip_search, "adaptive_clip_search")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    qweight, *_ = gptq.quantize()
    end.record()
    end.synchronize()
    total_ms = start.elapsed_time(end)

    result = {
        "label": label,
        "rows": rows,
        "cols": cols,
        "dtype": "bfloat16",
        "calibration_batch": 1,
        "calibration_rows": batch,
        "group_size": 128,
        "total_ms": total_ms,
        "throughput_mweights_s": (rows * cols) / (total_ms * 1000.0),
        "operations": timer.finalize().results(total_ms),
        "qweight_shape": list(qweight.shape),
    }
    del inp, gptq, qweight
    return result


def _load_baseline(path):
    if path is None:
        return {}
    with open(path) as handle:
        payload = json.load(handle)
    return {row["label"]: row for row in payload["results"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-json", help="Operation results recorded from the comparison revision.")
    parser.add_argument("--output-json", help="Write machine-readable results for a later comparison.")
    args = parser.parse_args()

    _preflight_gpu()
    import torch

    from gptqmodel.utils.logger import render_table

    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    baseline = _load_baseline(args.baseline_json)
    results = [
        _run(4096, 12288, "down_proj"),
        _run(12288, 4096, "gate_proj"),
    ]

    rows = []
    for result in results:
        reference = baseline.get(result["label"], {})
        reference_operations = {row["operation"]: row for row in reference.get("operations", [])}
        for operation in result["operations"]:
            reference_operation = reference_operations.get(operation["operation"])
            reference_ms = reference_operation.get("exclusive_ms") if reference_operation is not None else None
            speedup = (
                reference_ms / operation["exclusive_ms"]
                if reference_ms is not None and operation["exclusive_ms"] > 0
                else None
            )
            rows.append([
                result["label"],
                f"{result['rows']}x{result['cols']}",
                result["dtype"],
                f"{result['calibration_batch']}x{result['calibration_rows']}",
                result["group_size"],
                operation["operation"],
                operation["calls"],
                f"{operation['inclusive_ms']:.2f}",
                f"{operation['exclusive_ms']:.2f}",
                f"{100 * operation['exclusive_ms'] / result['total_ms']:.1f}%",
                f"{reference_ms:.2f}" if reference_ms is not None else "n/a",
                f"{speedup:.3f}x" if speedup is not None else "n/a",
                f"{result['throughput_mweights_s']:.2f}",
            ])
    print(render_table(
        rows,
        headers=[
            "module",
            "shape",
            "dtype",
            "calib BxT",
            "group",
            "operation",
            "calls",
            "inclusive ms",
            "exclusive ms",
            "% total",
            "ref ms",
            "speedup",
            "Mweights/s",
        ],
    ))

    if args.output_json:
        with open(args.output_json, "w") as handle:
            payload = {"torch": str(torch.__version__), "cuda": torch.version.cuda, "results": results}
            json.dump(payload, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
