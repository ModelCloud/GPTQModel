# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Qualify equal-work Qwen3 inference on Apple ANE and MLX Metal.

The benchmark reads dense search weights directly from a Qwen3 checkpoint.
Those weights and the generated activations are offline benchmark scratch; no
packed inference checkpoint is loaded or changed.

Each task runs the same FP16 inputs and weights through a forced-ANE Core ML
graph and MLX Metal. A separate FP32 Torch calculation, using the checkpoint's
source weights, is the accuracy oracle. The scheduler section runs the ANE
teacher-shaped MLP concurrently with an independent differentiable MLX
student/backward workload to test whether overlap can help RCO.

    python scripts/gsq_paper/bench_qwen3_ane.py \
        --model /path/to/Qwen3-0.6B \
        --shapes 1x16,1x128,2x128,4x128 \
        --json-out /tmp/qwen3-ane.json
"""

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from pathlib import Path

import numpy as np
import torch


PROJECTION_NAMES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
MLP_NAMES = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
TASK_ALIASES = {
    **{name.rsplit(".", 1)[-1]: name for name in PROJECTION_NAMES},
    "mlp": "mlp",
}
ACCURACY_RTOL = 2e-3
ACCURACY_ATOL = 2e-3
OVERLAP_SPEEDUP_THRESHOLD = 1.05


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _parse_shapes(value):
    shapes = []
    for item in value.split(","):
        parts = item.strip().lower().split("x")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("shapes must use BATCHxTOKENS")
        try:
            batch, tokens = (int(part) for part in parts)
        except ValueError as error:
            raise argparse.ArgumentTypeError("shapes must use integer BATCHxTOKENS") from error
        if batch < 1 or tokens < 1:
            raise argparse.ArgumentTypeError("shape dimensions must be positive")
        shapes.append((batch, tokens))
    if not shapes:
        raise argparse.ArgumentTypeError("at least one shape is required")
    return tuple(dict.fromkeys(shapes))


def _parse_positive_ints(value):
    try:
        values = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from error
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return tuple(dict.fromkeys(values))


def _weight_file(model_path):
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        payload = json.loads(index_path.read_text())
        files = sorted(set(payload["weight_map"].values()))
        return [model_path / name for name in files]
    files = sorted(model_path.glob("*.safetensors"))
    if not files:
        raise SystemExit(f"No safetensors checkpoint found under {model_path}")
    return files


def _load_weight(model_path, tensor_name):
    from safetensors import safe_open

    for filename in _weight_file(model_path):
        with safe_open(filename, framework="pt", device="cpu") as handle:
            if tensor_name in handle.keys():
                return handle.get_tensor(tensor_name)
    raise KeyError(f"{tensor_name} is absent from {model_path}")


def _layer_weights(model_path, layer):
    return {name: _load_weight(model_path, f"model.layers.{layer}.{name}.weight") for name in PROJECTION_NAMES}


def _capture_activations(model_path, corpus, sequence_index, layer):
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM

    dataset = load_from_disk(str(corpus / "validation"))
    if not 0 <= sequence_index < len(dataset):
        raise SystemExit(f"--calibration-sequence must be in [0, {len(dataset)})")
    input_ids = torch.tensor(dataset[sequence_index]["input_ids"], dtype=torch.long).unsqueeze(0)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()
    captured = {}

    def capture(name):
        def hook(_module, args, kwargs):
            value = args[0] if args else kwargs["hidden_states"]
            captured[name] = value.detach().cpu()

        return hook

    decoder = model.model.layers[layer]
    handles = [
        decoder.self_attn.q_proj.register_forward_pre_hook(capture("self_attn.q_proj"), with_kwargs=True),
        decoder.self_attn.k_proj.register_forward_pre_hook(capture("self_attn.k_proj"), with_kwargs=True),
        decoder.self_attn.v_proj.register_forward_pre_hook(capture("self_attn.v_proj"), with_kwargs=True),
        decoder.self_attn.o_proj.register_forward_pre_hook(capture("self_attn.o_proj"), with_kwargs=True),
        decoder.mlp.register_forward_pre_hook(capture("mlp"), with_kwargs=True),
        decoder.mlp.down_proj.register_forward_pre_hook(capture("mlp.down_proj"), with_kwargs=True),
    ]
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    missing = {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp",
        "mlp.down_proj",
    } - set(captured)
    if missing:
        raise RuntimeError(f"failed to capture Qwen3 activations: {sorted(missing)}")
    captured["mlp.gate_proj"] = captured["mlp"]
    captured["mlp.up_proj"] = captured["mlp"]
    return captured


def _numpy_weight(weight, dtype):
    return weight.float().numpy().astype(dtype)


def _torch_projection(x, weight):
    return x @ weight.T


def _torch_mlp(x, weights):
    gate = _torch_projection(x, weights["mlp.gate_proj"])
    up = _torch_projection(x, weights["mlp.up_proj"])
    value = torch.nn.functional.silu(gate) * up
    return x + _torch_projection(value, weights["mlp.down_proj"])


def _coreml_program(ct, mb, task, weights, shape):
    input_width = _input_width(task, weights)
    input_specs = [
        mb.TensorSpec(
            shape=(shape[0], shape[1], input_width),
            dtype=ct.converters.mil.mil.types.fp16,
        )
    ]

    @mb.program(input_specs=input_specs, opset_version=ct.target.macOS15)
    def program(x):
        if task != "mlp":
            return mb.linear(
                x=x,
                weight=_numpy_weight(weights[task], np.float16),
                name="output",
            )
        gate = mb.linear(
            x=x,
            weight=_numpy_weight(weights["mlp.gate_proj"], np.float16),
            name="gate",
        )
        up = mb.linear(
            x=x,
            weight=_numpy_weight(weights["mlp.up_proj"], np.float16),
            name="up",
        )
        hidden = mb.mul(x=mb.silu(x=gate, name="silu"), y=up, name="gated")
        residual = mb.linear(
            x=hidden,
            weight=_numpy_weight(weights["mlp.down_proj"], np.float16),
            name="down",
        )
        return mb.add(x=x, y=residual, name="output")

    return program


def _input_width(task, weights):
    return weights["mlp.gate_proj" if task == "mlp" else task].shape[1]


def _plan_rows(plan):
    rows = []
    operations = plan.model_structure.program.functions["main"].block.operations
    for operation in operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
        if not usage:
            continue
        preferred = usage.preferred_compute_device
        rows.append(
            {
                "op": operation.operator_name,
                "preferred": type(preferred).__name__,
                "cores": getattr(preferred, "total_core_count", None),
                "supported": sorted(type(device).__name__ for device in usage.supported_compute_devices),
            }
        )
    return rows


def _compile_ane(ct, mb, task, weights, shape):
    from coremltools.models.compute_plan import MLComputePlan
    from coremltools.models.ml_program.experimental.compute_plan_utils import apply_compute_plan

    started = time.perf_counter()
    source = ct.convert(
        _coreml_program(ct, mb, task, weights, shape),
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
        lambda _operation, _usage: ["ane"],
    )
    forced_plan = MLComputePlan.load_from_path(
        forced.get_compiled_model_path(),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    elapsed = time.perf_counter() - started
    rows = _plan_rows(forced_plan)
    all_ane = bool(rows) and all(row["preferred"] == "MLNeuralEngineComputeDevice" for row in rows)
    if not all_ane:
        raise RuntimeError(f"Core ML forced plan left non-ANE operations for {task}: {rows}")
    return forced, rows, elapsed


def _metal_forward(mx, nn, task, weights, input_value):
    x = mx.array(input_value)
    if task != "mlp":
        value = x @ mx.array(_numpy_weight(weights[task], np.float16)).T
        mx.eval(value)
        return value
    gate = x @ mx.array(_numpy_weight(weights["mlp.gate_proj"], np.float16)).T
    up = x @ mx.array(_numpy_weight(weights["mlp.up_proj"], np.float16)).T
    value = nn.silu(gate) * up
    value = x + value @ mx.array(_numpy_weight(weights["mlp.down_proj"], np.float16)).T
    mx.eval(value)
    return value


def _metal_student_step(mx, nn, weights, input_value, repeats):
    x = mx.array(input_value.astype(np.float32))
    gate_weight = mx.array(_numpy_weight(weights["mlp.gate_proj"], np.float32))
    up_weight = mx.array(_numpy_weight(weights["mlp.up_proj"], np.float32))
    down_weight = mx.array(_numpy_weight(weights["mlp.down_proj"], np.float32))

    def loss(gate_value, up_value, down_value):
        value = x
        for _ in range(repeats):
            gate = value @ gate_value.T
            up = value @ up_value.T
            hidden = nn.silu(gate) * up
            value = value + hidden @ down_value.T
        return mx.mean(value * value)

    value_and_grad = mx.value_and_grad(loss, argnums=(0, 1, 2))
    loss_value, gradients = value_and_grad(gate_weight, up_weight, down_weight)
    mx.eval(loss_value, *gradients)
    return loss_value, gradients


def _synchronize_mlx(mx):
    mx.synchronize()


def _timed(function, *, warmup, repeats, synchronize):
    for _ in range(warmup):
        function()
        synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        synchronize()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), samples


def _accuracy(actual, oracle):
    actual = np.asarray(actual, dtype=np.float32)
    oracle = np.asarray(oracle, dtype=np.float32)
    if actual.shape != oracle.shape:
        raise ValueError(f"shape mismatch: actual={actual.shape}, oracle={oracle.shape}")
    if not np.isfinite(actual).all():
        raise ValueError("accelerator output contains non-finite values")
    delta = np.abs(actual - oracle)
    close = np.isclose(actual, oracle, rtol=ACCURACY_RTOL, atol=ACCURACY_ATOL)
    denominator = float(np.linalg.norm(oracle))
    return {
        "max_abs_error": float(delta.max()),
        "mean_abs_error": float(delta.mean()),
        "normalized_l2_error": float(np.linalg.norm(delta) / max(denominator, 1e-12)),
        "allclose": bool(close.all()),
        "mismatches": int((~close).sum()),
        "elements": int(close.size),
        "rtol": ACCURACY_RTOL,
        "atol": ACCURACY_ATOL,
    }


def _passes_accuracy_gate(accuracy):
    return accuracy["allclose"] and accuracy["normalized_l2_error"] <= ACCURACY_ATOL


def _results_pass_accuracy_gate(results, accuracies):
    return bool(results) and all(
        _passes_accuracy_gate(result["accuracy"][accuracy]) for result in results for accuracy in accuracies
    )


def _benchmark_task(mx, nn, ct, mb, task, weights, shape, *, activations, warmup, repeats):
    input_width = _input_width(task, weights)
    if activations is None:
        input_value = (
            np.random.default_rng(7)
            .standard_normal(
                (shape[0], shape[1], input_width),
                dtype=np.float32,
            )
            .astype(np.float16)
        )
        input_source = "deterministic-random"
    else:
        base = activations[task]
        if shape[1] > base.shape[1]:
            raise SystemExit(f"shape {shape[0]}x{shape[1]} exceeds the {base.shape[1]}-token calibration capture")
        input_value = base[:, : shape[1]].repeat(shape[0], 1, 1).numpy().astype(np.float16)
        input_source = "captured-qwen3"
    oracle_input = torch.from_numpy(input_value.astype(np.float32))
    oracle_weights = {name: torch.from_numpy(weight.float().numpy()) for name, weight in weights.items()}
    with torch.inference_mode():
        oracle = (
            _torch_mlp(oracle_input, oracle_weights)
            if task == "mlp"
            else _torch_projection(oracle_input, oracle_weights[task])
        )
        oracle_visible = oracle.to(torch.float16).float()

    ane_model, plan_rows, compile_seconds = _compile_ane(ct, mb, task, weights, shape)

    def run_ane():
        return next(iter(ane_model.predict({"x": input_value}).values()))

    def run_metal():
        return _metal_forward(mx, nn, task, weights, input_value)

    ane_output = run_ane()
    metal_output = np.asarray(run_metal())
    ane_accuracy = _accuracy(ane_output, oracle.numpy())
    ane_visible_accuracy = _accuracy(ane_output, oracle_visible.numpy())
    metal_accuracy = _accuracy(metal_output, oracle.numpy())
    metal_visible_accuracy = _accuracy(metal_output, oracle_visible.numpy())
    cross_accuracy = _accuracy(ane_output, metal_output)
    ane_median, ane_samples = _timed(
        run_ane,
        warmup=warmup,
        repeats=repeats,
        synchronize=lambda: None,
    )
    metal_median, metal_samples = _timed(
        run_metal,
        warmup=warmup,
        repeats=repeats,
        synchronize=lambda: _synchronize_mlx(mx),
    )
    return {
        "task": task,
        "batch": shape[0],
        "tokens": shape[1],
        "input_width": input_width,
        "input_source": input_source,
        "output_width": int(np.asarray(metal_output).shape[-1]),
        "source_dtype": str(next(iter(weights.values())).dtype),
        "accelerated_dtype": "float16",
        "compile_seconds": compile_seconds,
        "compute_plan": plan_rows,
        "accuracy": {
            "ane_vs_torch_fp32": ane_accuracy,
            "ane_vs_torch_fp16_rounded": ane_visible_accuracy,
            "metal_vs_torch_fp32": metal_accuracy,
            "metal_vs_torch_fp16_rounded": metal_visible_accuracy,
            "ane_vs_metal": cross_accuracy,
        },
        "timings_seconds": {
            "ane_median": ane_median,
            "metal_median": metal_median,
            "metal_over_ane": metal_median / ane_median,
            "ane_samples": ane_samples,
            "metal_samples": metal_samples,
        },
    }


def _benchmark_scheduler(
    mx,
    nn,
    ct,
    mb,
    weights,
    shape,
    *,
    activations,
    warmup,
    repeats,
    teacher_counts,
    student_repeats,
):
    ane_model, plan_rows, compile_seconds = _compile_ane(ct, mb, "mlp", weights, shape)
    if activations is None:
        input_value = (
            np.random.default_rng(17)
            .standard_normal(
                (shape[0], shape[1], weights["mlp.gate_proj"].shape[1]),
                dtype=np.float32,
            )
            .astype(np.float16)
        )
        input_source = "deterministic-random"
    else:
        base = activations["mlp"]
        if shape[1] > base.shape[1]:
            raise SystemExit(
                f"scheduler shape {shape[0]}x{shape[1]} exceeds the {base.shape[1]}-token calibration capture"
            )
        input_value = base[:, : shape[1]].repeat(shape[0], 1, 1).numpy().astype(np.float16)
        input_source = "captured-qwen3"
    teacher_output = next(iter(ane_model.predict({"x": input_value}).values()))
    if not np.isfinite(teacher_output).all():
        raise RuntimeError("ANE teacher output contains non-finite values")

    def run_teacher(calls):
        result = None
        for _ in range(calls):
            result = ane_model.predict({"x": input_value})
        return result

    def run_student():
        return _metal_student_step(mx, nn, weights, input_value, student_repeats)

    for _ in range(warmup):
        run_teacher(1)
        run_student()
        _synchronize_mlx(mx)

    scenarios = []
    for teacher_calls in teacher_counts:
        teacher_median, teacher_samples = _timed(
            lambda calls=teacher_calls: run_teacher(calls),
            warmup=0,
            repeats=repeats,
            synchronize=lambda: None,
        )
        student_median, student_samples = _timed(
            run_student,
            warmup=0,
            repeats=repeats,
            synchronize=lambda: _synchronize_mlx(mx),
        )

        def run_serial(calls=teacher_calls):
            run_teacher(calls)
            run_student()

        serial_median, serial_samples = _timed(
            run_serial,
            warmup=0,
            repeats=repeats,
            synchronize=lambda: _synchronize_mlx(mx),
        )

        with ThreadPoolExecutor(max_workers=1) as executor:

            def run_threaded(calls=teacher_calls):
                pending = executor.submit(run_teacher, calls)
                try:
                    run_student()
                finally:
                    pending.result()
                _synchronize_mlx(mx)

            threaded_median, threaded_samples = _timed(
                run_threaded,
                warmup=0,
                repeats=repeats,
                synchronize=lambda: None,
            )

        scenarios.append(
            {
                "teacher_mlp_calls": teacher_calls,
                "student_repeat_blocks": student_repeats,
                "timings_seconds": {
                    "teacher_ane_median": teacher_median,
                    "student_metal_median": student_median,
                    "serial_median": serial_median,
                    "threaded_median": threaded_median,
                    "threaded_speedup_vs_serial": serial_median / threaded_median,
                    "teacher_ane_samples": teacher_samples,
                    "student_metal_samples": student_samples,
                    "serial_samples": serial_samples,
                    "threaded_samples": threaded_samples,
                },
            }
        )

    return {
        "shape": {"batch": shape[0], "tokens": shape[1]},
        "input_source": input_source,
        "compile_seconds": compile_seconds,
        "compute_plan": plan_rows,
        "scenarios": scenarios,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="local Qwen3 model directory")
    parser.add_argument("--corpus", help="optional corpus root with a validation split for captured activations")
    parser.add_argument("--calibration-sequence", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument(
        "--tasks",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,mlp",
        help="comma-separated projections plus mlp",
    )
    parser.add_argument("--shapes", type=_parse_shapes, default=_parse_shapes("1x16,1x128,2x128,4x128"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--student-repeats", type=int, default=4)
    parser.add_argument("--scheduler-teacher-calls", type=_parse_positive_ints, default=_parse_positive_ints("1,7,28"))
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()
    if args.layer < 0:
        parser.error("--layer must be nonnegative")
    if args.calibration_sequence < 0:
        parser.error("--calibration-sequence must be nonnegative")
    if min(args.warmup, args.repeats, args.student_repeats) < 0:
        parser.error("warmup and repeat counts must be nonnegative")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise SystemExit("Qwen3 ANE benchmark requires macOS arm64")
    try:
        import coremltools as ct
        import mlx.core as mx
        import mlx.nn as nn
        from coremltools.converters.mil import Builder as mb
        from transformers import AutoConfig
    except ImportError as error:
        raise SystemExit("install gptqmodel[mlx] and coremltools to run this benchmark") from error
    if not mx.metal.is_available():
        raise SystemExit("MLX Metal is unavailable")
    mx.set_default_device(mx.gpu)

    model_path = Path(args.model)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if config.model_type != "qwen3":
        raise SystemExit(f"This benchmark requires a dense Qwen3 checkpoint, got {config.model_type}")
    if not 0 <= args.layer < config.num_hidden_layers:
        raise SystemExit(f"--layer must be in [0, {config.num_hidden_layers})")
    requested_tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    if not requested_tasks or any(task not in TASK_ALIASES for task in requested_tasks):
        raise SystemExit(f"--tasks must be drawn from {', '.join(sorted(TASK_ALIASES))}")
    tasks = list(dict.fromkeys(TASK_ALIASES[task] for task in requested_tasks))

    weights = _layer_weights(model_path, args.layer)
    activations = (
        _capture_activations(model_path, Path(args.corpus), args.calibration_sequence, args.layer)
        if args.corpus
        else None
    )
    results = [
        _benchmark_task(
            mx,
            nn,
            ct,
            mb,
            task,
            weights,
            shape,
            activations=activations,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for task in tasks
        for shape in args.shapes
    ]
    scheduler_shape = args.shapes[0]
    scheduler = _benchmark_scheduler(
        mx,
        nn,
        ct,
        mb,
        weights,
        scheduler_shape,
        activations=activations,
        warmup=args.warmup,
        repeats=args.repeats,
        teacher_counts=args.scheduler_teacher_calls,
        student_repeats=args.student_repeats,
    )

    projection_results = [result for result in results if result["task"] != "mlp"]
    mlp_results = [result for result in results if result["task"] == "mlp"]
    overlap_gains = [scenario["timings_seconds"]["threaded_speedup_vs_serial"] for scenario in scheduler["scenarios"]]
    overlap_speedup = min(overlap_gains) if overlap_gains else 0.0
    result = {
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mlx": _version("mlx"),
            "mlx_lm": _version("mlx-lm"),
            "coremltools": ct.__version__,
        },
        "model": {
            "path": str(model_path),
            "model_type": config.model_type,
            "num_hidden_layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "layer": args.layer,
            "calibration_sequence": args.calibration_sequence if args.corpus else None,
        },
        "configuration": {
            "tasks": tasks,
            "shapes": [{"batch": batch, "tokens": tokens} for batch, tokens in args.shapes],
            "warmup": args.warmup,
            "repeats": args.repeats,
            "student_repeat_blocks": args.student_repeats,
            "scheduler_teacher_calls": list(args.scheduler_teacher_calls),
            "accuracy_rtol": ACCURACY_RTOL,
            "accuracy_atol": ACCURACY_ATOL,
            "accelerated_dtype": "float16",
            "oracle": "FP32 Torch using checkpoint source weights",
            "input_source": "captured Qwen3 calibration activations"
            if activations is not None
            else "deterministic random",
            "storage": "unified_memory",
        },
        "results": results,
        "scheduler": scheduler,
        "gates": {
            "ane_projection_accuracy": _results_pass_accuracy_gate(
                projection_results,
                ("ane_vs_torch_fp32", "ane_vs_torch_fp16_rounded"),
            ),
            "metal_projection_accuracy": _results_pass_accuracy_gate(
                projection_results,
                ("metal_vs_torch_fp32", "metal_vs_torch_fp16_rounded"),
            ),
            "ane_mlp_accuracy": _results_pass_accuracy_gate(
                mlp_results,
                ("ane_vs_torch_fp32", "ane_vs_torch_fp16_rounded"),
            ),
            "metal_mlp_accuracy": _results_pass_accuracy_gate(
                mlp_results,
                ("metal_vs_torch_fp32", "metal_vs_torch_fp16_rounded"),
            ),
            "overlap_speedup": overlap_speedup,
            "overlap_improves": bool(overlap_gains)
            and all(gain > OVERLAP_SPEEDUP_THRESHOLD for gain in overlap_gains),
            "overlap_threshold": OVERLAP_SPEEDUP_THRESHOLD,
        },
        "scope": (
            "Dense Qwen3 inference and scheduling qualification only. "
            "Scheduler teacher counts repeat the layer-0 MLP to measure workload volume, "
            "not a full 28-layer model chain. Core ML has no autograd, so student and "
            "backward remain on MLX Metal."
        ),
    }
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    gc.collect()


if __name__ == "__main__":
    main()
