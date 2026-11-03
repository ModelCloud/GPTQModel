# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Measure independent Metal GPU and Apple Neural Engine execution.

This intentionally uses a small, explicitly forced Core ML compute plan rather
than trying to route the differentiable RCO graph through the ANE. Core ML has
no autograd. The probe answers two narrower questions relevant to scheduling:

* Does the graph select the ANE for every eligible operation?
* Does concurrent execution beat running the ANE and GPU work one after the
  other on this host?

The GPU leg is an MLX Metal recurrent MLP. The ANE leg is a Core ML MIL program
with four fused linear/GELU/residual blocks mapped to all available ANE cores.
Both read and write the same unified-memory pool, with no HBM transfer.

    python scripts/gsq_paper/bench_gpu_ane.py --json-out /tmp/gpu-ane.json
"""

import argparse
import json
import platform
import statistics
import sys
import threading
import time
from importlib import metadata
from pathlib import Path


DEFAULT_TOKENS = 256
DEFAULT_WIDTH = 1024
DEFAULT_ANE_BLOCKS = 4
DEFAULT_GPU_REPEATS = 40


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _program(tokens, width, blocks):
    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil import Builder as mb

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, tokens, width), dtype=ct.converters.mil.mil.types.fp16)],
        opset_version=ct.target.macOS13,
    )
    def program(x):
        weight = mb.const(
            val=np.ones((width, width), dtype=np.float16) / width,
            name="weight",
        )
        value = x
        for index in range(blocks):
            a = mb.linear(x=value, weight=weight, name=f"linear_a_{index}")
            b = mb.gelu(x=a, name=f"gelu_{index}")
            c = mb.linear(x=b, weight=weight, name=f"linear_b_{index}")
            value = mb.add(x=value, y=c, name=f"residual_{index}")
        return value

    return ct, program


def _plan_rows(plan):
    rows = []
    ops = plan.model_structure.program.functions["main"].block.operations
    for op in ops:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if not usage:
            continue
        preferred = usage.preferred_compute_device
        rows.append(
            {
                "op": op.operator_name,
                "preferred": type(preferred).__name__,
                "cores": getattr(preferred, "total_core_count", None),
                "supported": sorted(type(device).__name__ for device in usage.supported_compute_devices),
            }
        )
    return rows


def _median_milliseconds(fn, repeats):
    for _ in range(3):
        fn()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), samples


def _threaded(first, second, repeats=9):
    """Median wall time for two concurrent calls, with errors captured."""
    for _ in range(3):
        errors = []

        def guarded(fn):
            try:
                fn()
            except BaseException as error:  # pragma: no cover - host probe
                errors.append(repr(error))

        workers = [
            threading.Thread(target=guarded, args=(first,)),
            threading.Thread(target=guarded, args=(second,)),
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        if errors:
            raise RuntimeError(f"Concurrent engine invocation failed: {errors}")

    samples = []
    for _ in range(repeats):
        errors = []

        def guarded(fn):
            try:
                fn()
            except BaseException as error:  # pragma: no cover - host probe
                errors.append(repr(error))

        started = time.perf_counter()
        workers = [
            threading.Thread(target=guarded, args=(first,)),
            threading.Thread(target=guarded, args=(second,)),
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        elapsed = time.perf_counter() - started
        if errors:
            raise RuntimeError(f"Concurrent engine invocation failed: {errors}")
        samples.append(elapsed)
    return statistics.median(samples), samples


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--ane-blocks", type=int, default=DEFAULT_ANE_BLOCKS)
    parser.add_argument(
        "--gpu-repeats",
        type=int,
        default=DEFAULT_GPU_REPEATS,
        help="balance ANE and GPU durations for the overlap measurement",
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()
    if min(args.tokens, args.width, args.ane_blocks, args.gpu_repeats, args.repeats) < 1:
        parser.error("all workload dimensions and repeat counts must be positive")
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise SystemExit("GPU/ANE benchmark requires macOS arm64")
    try:
        import coremltools as ct
        import mlx.core as mx
        import numpy as np
        import torch
        from coremltools.models.compute_plan import MLComputePlan
        from coremltools.models.ml_program.experimental.compute_plan_utils import apply_compute_plan
    except ImportError as error:
        raise SystemExit(
            "install gptqmodel[mlx] and coremltools to run the GPU/ANE benchmark"
        ) from error

    ct, program = _program(args.tokens, args.width, args.ane_blocks)
    source = ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
    )
    initial_plan = MLComputePlan.load_from_path(
        source.get_compiled_model_path(),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    forced = apply_compute_plan(
        source,
        initial_plan,
        lambda _op, _usage: ["ane"],
    )
    forced_plan = MLComputePlan.load_from_path(
        forced.get_compiled_model_path(),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    input_value = np.random.default_rng(7).standard_normal(
        (1, args.tokens, args.width), dtype=np.float32
    ).astype(np.float16)

    gpu_weight = mx.full((args.width, args.width), 1 / args.width, dtype=mx.float16)
    gpu_input = mx.array(input_value)

    def run_ane():
        forced.predict({"x": input_value})

    def run_gpu():
        value = gpu_input
        for _ in range(args.gpu_repeats):
            a = value @ gpu_weight
            b = mx.maximum(a * 0.5, a) + mx.minimum(a, 0) * 0.0
            c = b @ gpu_weight
            value = value + c
        mx.eval(value)
        return value

    # Warm both backends before timing; Core ML resolves its compiled plan lazily.
    for _ in range(3):
        run_ane()
        run_gpu()

    ane_median, ane_samples = _median_milliseconds(run_ane, args.repeats)
    gpu_median, gpu_samples = _median_milliseconds(run_gpu, args.repeats)
    serial_median, serial_samples = _median_milliseconds(
        lambda: (run_ane(), run_gpu()), args.repeats
    )
    overlap_median, overlap_samples = _threaded(run_ane, run_gpu, repeats=args.repeats)

    forced_rows = _plan_rows(forced_plan)
    all_core_ops = all(
        row["preferred"] == "MLNeuralEngineComputeDevice" for row in forced_rows
    )
    if not all_core_ops:
        raise RuntimeError(f"forced compute plan left non-ANE operations: {forced_rows}")
    if any("MLGPUComputeDevice" not in row["supported"] for row in forced_rows):
        raise RuntimeError("forced ANE plan does not report GPU eligibility for every operation")

    result = {
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mlx": _version("mlx"),
            "coremltools": ct.__version__,
        },
        "workload": {
            "tokens": args.tokens,
            "width": args.width,
            "ane_blocks": args.ane_blocks,
            "gpu_repeat_blocks": args.gpu_repeats,
            "storage": "unified_memory",
            "ane_forced": all_core_ops,
            "ane_core_count": max((row["cores"] or 0 for row in forced_rows), default=0),
            "ane_autograd": False,
        },
        "compute_plan": {
            "initial": _plan_rows(initial_plan),
            "forced_ane": forced_rows,
        },
        "timings_seconds": {
            "ane_median": ane_median,
            "gpu_median": gpu_median,
            "serial_median": serial_median,
            "threaded_median": overlap_median,
            "threaded_speedup_vs_serial": serial_median / overlap_median,
            "samples": {
                "ane": ane_samples,
                "gpu": gpu_samples,
                "serial": serial_samples,
                "threaded": overlap_samples,
            },
        },
        "recommendation": (
            "integrate_concurrent_dispatch"
            if overlap_median < serial_median
            else "keep_single_allocator_on_gpu"
        ),
    }
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
