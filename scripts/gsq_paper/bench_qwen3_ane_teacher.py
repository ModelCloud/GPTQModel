# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Qualify a complete frozen Qwen3 teacher forward on Apple ANE and MLX Metal.

The script converts a fixed-shape dense Qwen3 teacher to a Core ML program,
forces supported operations onto the ANE, and compares its final hidden state
and logits with an independent FP32 Torch oracle. It then compares teacher KL
and steady-state latency with the native MLX Metal model.

The dense model, converted package, hidden states, and logits are offline
benchmark scratch. No packed quantized inference checkpoint is loaded or
modified.

    python scripts/gsq_paper/bench_qwen3_ane_teacher.py \
        --model /path/to/Qwen3-0.6B \
        --corpus /path/to/corpus \
        --mlpackage /tmp/qwen3-teacher-1x16.mlpackage \
        --tokens 16 --warmup 2 --repeats 10 \
        --json-out /tmp/qwen3-ane-teacher.json
"""

import argparse
import gc
import json
import platform
import shutil
import statistics
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import torch


ACCURACY_RTOL = 2e-3
ACCURACY_ATOL = 2e-3
ANE_OPERATION_SHARE_THRESHOLD = 0.99
PERFORMANCE_SPEEDUP_THRESHOLD = 1.05


def _version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _package_bytes(path):
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _load_input_ids(corpus, sequence_index, tokens):
    if corpus is None:
        return torch.arange(1, tokens + 1, dtype=torch.int32).unsqueeze(0)
    from datasets import load_from_disk

    dataset = load_from_disk(str(Path(corpus) / "validation"))
    if not 0 <= sequence_index < len(dataset):
        raise SystemExit(f"--calibration-sequence must be in [0, {len(dataset)})")
    values = dataset[sequence_index]["input_ids"]
    if len(values) < tokens:
        raise SystemExit(f"calibration sequence has {len(values)} tokens; --tokens requires {tokens}")
    return torch.tensor(values[:tokens], dtype=torch.int32).unsqueeze(0)


def _fixed_rotate_half(value, half):
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def _fixed_rotary_pos_emb(query, key, cos, sin, half):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query_rotated = (query * cos) + (_fixed_rotate_half(query, half) * sin)
    key_rotated = (key * cos) + (_fixed_rotate_half(key, half) * sin)
    return query_rotated, key_rotated


def _fixed_attention_forward(
    module,
    hidden_states,
    position_embeddings,
    mask,
    tokens,
):
    batch = 1
    head_dim = module.head_dim
    num_heads = module.config.num_attention_heads
    num_key_value_heads = module.config.num_key_value_heads
    num_key_value_groups = num_heads // num_key_value_heads
    query_shape = (batch, tokens, -1, head_dim)

    query = module.q_norm(module.q_proj(hidden_states).view(query_shape)).transpose(1, 2)
    key = module.k_norm(module.k_proj(hidden_states).view(query_shape)).transpose(1, 2)
    value = module.v_proj(hidden_states).view(query_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query, key = _fixed_rotary_pos_emb(query, key, cos, sin, head_dim // 2)

    key = key[:, :, None, :, :].expand(
        batch,
        num_key_value_heads,
        num_key_value_groups,
        tokens,
        head_dim,
    )
    key = key.reshape(
        batch,
        num_key_value_heads * num_key_value_groups,
        tokens,
        head_dim,
    )
    value = value[:, :, None, :, :].expand(
        batch,
        num_key_value_heads,
        num_key_value_groups,
        tokens,
        head_dim,
    )
    value = value.reshape(
        batch,
        num_key_value_heads * num_key_value_groups,
        tokens,
        head_dim,
    )

    weights = torch.matmul(query, key.transpose(2, 3)) * module.scaling
    weights = weights + mask
    weights = torch.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    output = torch.matmul(weights, value).transpose(1, 2)
    output = output.reshape(batch, tokens, -1).contiguous()
    return module.o_proj(output)


class TeacherWrapper(torch.nn.Module):
    """Traceable Qwen3 forward with fixed-shape attention and causal mask."""

    def __init__(self, model, position_ids, mask):
        super().__init__()
        self.model = model
        self.register_buffer("position_ids", position_ids)
        self.register_buffer("mask", mask)
        self.sequence_length = mask.shape[-1]

    def forward(self, input_ids):
        hidden = self.model.model.embed_tokens(input_ids)
        position_embeddings = self.model.model.rotary_emb(hidden, self.position_ids)
        for layer in self.model.model.layers:
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            attention = _fixed_attention_forward(
                layer.self_attn,
                normalized,
                position_embeddings,
                self.mask,
                self.sequence_length,
            )
            hidden = residual + attention
            residual = hidden
            normalized = layer.post_attention_layernorm(hidden)
            hidden = residual + layer.mlp(normalized)
        hidden = self.model.model.norm(hidden)
        return hidden, self.model.lm_head(hidden)


def _make_torch_wrapper(model_path, dtype, input_ids):
    from transformers import AutoModelForCausalLM
    from transformers.masking_utils import create_causal_mask

    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            local_files_only=True,
            attn_implementation="eager",
        )
        .eval()
        .requires_grad_(False)
    )
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device).unsqueeze(0)
    with torch.inference_mode():
        embedded = model.model.embed_tokens(input_ids.long())
        mask = create_causal_mask(
            model.config,
            embedded,
            None,
            None,
            position_ids=position_ids,
        )
    return TeacherWrapper(model, position_ids, mask).eval()


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


def _summarize_plan(rows):
    by_preferred = {}
    by_operator = {}
    for row in rows:
        by_preferred[row["preferred"]] = by_preferred.get(row["preferred"], 0) + 1
        key = f"{row['op']}:{row['preferred']}"
        by_operator[key] = by_operator.get(key, 0) + 1
    total = len(rows)
    ane_operations = by_preferred.get("MLNeuralEngineComputeDevice", 0)
    unsupported_devices = {
        name: count
        for name, count in by_preferred.items()
        if name not in {"MLNeuralEngineComputeDevice", "MLCPUComputeDevice"}
    }
    return {
        "operation_count": total,
        "by_preferred": by_preferred,
        "by_operator": by_operator,
        "ane_operation_share": ane_operations / total if total else 0.0,
        "ane_cores": sorted(
            {
                row["cores"]
                for row in rows
                if row["preferred"] == "MLNeuralEngineComputeDevice" and row["cores"] is not None
            }
        ),
        "non_ane_non_cpu_devices": unsupported_devices,
    }


def _passes_placement_gate(plan):
    return plan["ane_operation_share"] >= ANE_OPERATION_SHARE_THRESHOLD and not plan["non_ane_non_cpu_devices"]


def _accuracy(actual, oracle):
    actual = np.asarray(actual, dtype=np.float32)
    oracle = np.asarray(oracle, dtype=np.float32)
    if actual.shape != oracle.shape:
        raise ValueError(f"shape mismatch: actual={actual.shape}, oracle={oracle.shape}")
    if not np.isfinite(actual).all():
        raise ValueError("accelerator output contains non-finite values")
    delta = np.abs(actual - oracle)
    close = np.isclose(
        actual,
        oracle,
        rtol=ACCURACY_RTOL,
        atol=ACCURACY_ATOL,
    )
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


def _passes_accuracy_gate(result):
    return result["allclose"] and (result["normalized_l2_error"] <= ACCURACY_ATOL)


def _distribution_metrics(actual_logits, oracle_logits):
    actual = np.asarray(actual_logits, dtype=np.float32)
    oracle = np.asarray(oracle_logits, dtype=np.float32)
    actual_logp = actual - np.logaddexp.reduce(actual, axis=-1, keepdims=True)
    oracle_logp = oracle - np.logaddexp.reduce(oracle, axis=-1, keepdims=True)
    oracle_probability = np.exp(oracle_logp)
    actual_probability = np.exp(actual_logp)
    oracle_to_actual = np.sum(
        oracle_probability * (oracle_logp - actual_logp),
        axis=-1,
    )
    actual_to_oracle = np.sum(
        actual_probability * (actual_logp - oracle_logp),
        axis=-1,
    )
    top1_actual = actual.argmax(-1)
    top1_oracle = oracle.argmax(-1)
    top5_actual = np.argpartition(actual, -5, axis=-1)[..., -5:]
    top5_oracle = np.argpartition(oracle, -5, axis=-1)[..., -5:]
    top5_overlap = [
        len(set(actual_ids.tolist()) & set(oracle_ids.tolist()))
        for actual_ids, oracle_ids in zip(
            top5_actual.reshape(-1, 5),
            top5_oracle.reshape(-1, 5),
        )
    ]
    return {
        "kl_oracle_to_actual_mean": float(oracle_to_actual.mean()),
        "kl_oracle_to_actual_max": float(oracle_to_actual.max()),
        "kl_actual_to_oracle_mean": float(actual_to_oracle.mean()),
        "kl_actual_to_oracle_max": float(actual_to_oracle.max()),
        "top1_matches": int((top1_actual == top1_oracle).sum()),
        "tokens": int(top1_actual.size),
        "top5_overlap_mean": float(np.mean(top5_overlap)),
        "top5_overlap_min": int(np.min(top5_overlap)),
    }


def _timed(function, *, warmup, repeats):
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), samples


def _extract_teacher_outputs(
    model,
    prediction,
    hidden_size,
    vocabulary_size,
    tokens,
):
    outputs = list(model.get_spec().description.output)

    def select(shape):
        for output in outputs:
            if list(output.type.multiArrayType.shape) == shape:
                return np.asarray(prediction[output.name], dtype=np.float32)
        raise RuntimeError(f"Core ML teacher output with shape {shape} is absent")

    hidden = select([1, tokens, hidden_size])
    logits = select([1, tokens, vocabulary_size])
    return hidden, logits


def _compile_or_load_ane(
    ct,
    model_path,
    input_ids,
    mlpackage,
    *,
    reuse_mlpackage,
):
    from coremltools.models.compute_plan import MLComputePlan
    from coremltools.models.ml_program.experimental.compute_plan_utils import (
        apply_compute_plan,
    )

    timings = {
        "torch_fp16_load": None,
        "trace": None,
        "convert": None,
        "initial_plan_load": None,
        "force_plan": None,
        "package_save": None,
        "package_reload": None,
        "compute_plan_load": None,
    }
    if reuse_mlpackage:
        if not mlpackage.exists():
            raise SystemExit(f"--reuse-mlpackage requested but {mlpackage} does not exist")
        started = time.perf_counter()
        model = ct.models.MLModel(
            str(mlpackage),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        timings["package_reload"] = time.perf_counter() - started
    else:
        started = time.perf_counter()
        wrapper = _make_torch_wrapper(
            model_path,
            torch.float16,
            input_ids,
        )
        timings["torch_fp16_load"] = time.perf_counter() - started
        started = time.perf_counter()
        with torch.inference_mode():
            traced = torch.jit.trace(wrapper, (input_ids,), strict=False)
        timings["trace"] = time.perf_counter() - started

        started = time.perf_counter()
        source = ct.convert(
            traced,
            inputs=[
                ct.TensorType(
                    name="input_ids",
                    shape=tuple(input_ids.shape),
                    dtype=ct.converters.mil.mil.types.int32,
                )
            ],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
        )
        timings["convert"] = time.perf_counter() - started
        started = time.perf_counter()
        initial_plan = MLComputePlan.load_from_path(
            source.get_compiled_model_path(),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        timings["initial_plan_load"] = time.perf_counter() - started
        started = time.perf_counter()
        forced = apply_compute_plan(
            source,
            initial_plan,
            lambda _operation, _usage: ["ane"],
        )
        timings["force_plan"] = time.perf_counter() - started
        if mlpackage.exists():
            shutil.rmtree(mlpackage)
        mlpackage.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        forced.save(str(mlpackage))
        timings["package_save"] = time.perf_counter() - started
        del wrapper, traced, source, forced
        gc.collect()

        started = time.perf_counter()
        model = ct.models.MLModel(
            str(mlpackage),
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        timings["package_reload"] = time.perf_counter() - started

    started = time.perf_counter()
    plan = MLComputePlan.load_from_path(
        model.get_compiled_model_path(),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    timings["compute_plan_load"] = time.perf_counter() - started
    rows = _plan_rows(plan)
    return model, rows, timings


def _torch_oracle(model_path, input_ids):
    started = time.perf_counter()

    wrapper = _make_torch_wrapper(
        model_path,
        torch.float32,
        input_ids,
    )
    load_seconds = time.perf_counter() - started
    with torch.inference_mode():
        hidden, logits = wrapper(input_ids)
    return (
        hidden.float().numpy(),
        logits.float().numpy(),
        load_seconds,
    )


def _mlx_oracle(model_path, input_ids):
    import mlx.core as mx
    from mlx_lm import load

    if not mx.metal.is_available():
        raise SystemExit("MLX Metal is unavailable")
    mx.set_default_device(mx.gpu)
    started = time.perf_counter()

    model, _ = load(str(model_path), lazy=False)
    mx_ids = mx.array(input_ids.numpy())
    load_seconds = time.perf_counter() - started

    def forward():
        logits = model(mx_ids)
        mx.eval(logits)
        return np.asarray(logits.astype(mx.float32))

    return model, mx_ids, forward, load_seconds


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="local Qwen3 checkpoint")
    parser.add_argument("--corpus", help="optional corpus root with validation split")
    parser.add_argument("--calibration-sequence", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--mlpackage", required=True)
    parser.add_argument("--revision", help="checkpoint revision for the result metadata")
    parser.add_argument(
        "--reuse-mlpackage",
        action="store_true",
        help="reuse an already converted and forced package",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()
    if args.calibration_sequence < 0:
        parser.error("--calibration-sequence must be nonnegative")
    if args.tokens < 1:
        parser.error("--tokens must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise SystemExit("Qwen3 ANE teacher benchmark requires macOS arm64")

    try:
        import coremltools as ct
        import mlx.core as mx
        from transformers import AutoConfig
    except ImportError as error:
        raise SystemExit("install gptqmodel[mlx] and coremltools to run this benchmark") from error

    model_path = Path(args.model)
    mlpackage = Path(args.mlpackage)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if config.model_type != "qwen3":
        raise SystemExit(f"This benchmark requires a dense Qwen3 checkpoint, got {config.model_type}")
    input_ids = _load_input_ids(
        args.corpus,
        args.calibration_sequence,
        args.tokens,
    )

    ane_model, plan_rows, compile_timings = _compile_or_load_ane(
        ct,
        model_path,
        input_ids,
        mlpackage,
        reuse_mlpackage=args.reuse_mlpackage,
    )
    coreml_inputs = {"input_ids": input_ids.numpy()}
    coreml_prediction = ane_model.predict(coreml_inputs)
    ane_hidden, ane_logits = _extract_teacher_outputs(
        ane_model,
        coreml_prediction,
        config.hidden_size,
        config.vocab_size,
        args.tokens,
    )

    oracle_hidden, oracle_logits, oracle_load_seconds = _torch_oracle(
        model_path,
        input_ids,
    )
    mlx_model, mx_ids, mlx_forward, mlx_load_seconds = _mlx_oracle(
        model_path,
        input_ids,
    )
    mlx_logits = mlx_forward()

    def run_ane(model=ane_model, inputs=coreml_inputs):
        return model.predict(inputs)

    ane_median, ane_samples = _timed(
        run_ane,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    mlx_median, mlx_samples = _timed(
        mlx_forward,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    rounded_hidden = oracle_hidden.astype(np.float16).astype(np.float32)
    rounded_logits = oracle_logits.astype(np.float16).astype(np.float32)
    bfloat16_logits = torch.from_numpy(oracle_logits).to(torch.bfloat16).float().numpy()
    ane_hidden_accuracy = _accuracy(ane_hidden, oracle_hidden)
    ane_hidden_rounded_accuracy = _accuracy(ane_hidden, rounded_hidden)
    ane_logits_accuracy = _accuracy(ane_logits, oracle_logits)
    ane_logits_rounded_accuracy = _accuracy(ane_logits, rounded_logits)
    mlx_logits_accuracy = _accuracy(mlx_logits, oracle_logits)
    mlx_logits_rounded_accuracy = _accuracy(mlx_logits, bfloat16_logits)
    ane_vs_mlx = _accuracy(ane_logits, mlx_logits)
    ane_distribution = _distribution_metrics(ane_logits, oracle_logits)
    mlx_distribution = _distribution_metrics(mlx_logits, oracle_logits)
    ane_mlx_distribution = _distribution_metrics(ane_logits, mlx_logits)

    mlx_over_ane = mlx_median / ane_median
    hidden_gate = _passes_accuracy_gate(ane_hidden_accuracy) and _passes_accuracy_gate(ane_hidden_rounded_accuracy)
    logits_gate = _passes_accuracy_gate(ane_logits_accuracy) and _passes_accuracy_gate(ane_logits_rounded_accuracy)
    ane_distribution_gate = (
        ane_distribution["kl_oracle_to_actual_mean"] <= ACCURACY_ATOL
        and ane_distribution["top1_matches"] == ane_distribution["tokens"]
    )
    mlx_distribution_gate = (
        mlx_distribution["kl_oracle_to_actual_mean"] <= ACCURACY_ATOL
        and mlx_distribution["top1_matches"] == mlx_distribution["tokens"]
    )
    accuracy_gate = hidden_gate and logits_gate and ane_distribution_gate
    performance_gate = mlx_over_ane >= PERFORMANCE_SPEEDUP_THRESHOLD
    plan = _summarize_plan(plan_rows)
    placement_gate = _passes_placement_gate(plan)
    setup_total_seconds = sum(value for value in compile_timings.values() if value is not None)

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
            "revision": args.revision or getattr(config, "_commit_hash", None),
            "num_hidden_layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "vocab_size": config.vocab_size,
        },
        "input": {
            "shape": {"batch": 1, "tokens": args.tokens},
            "source": "corpus validation sequence" if args.corpus else "deterministic token IDs",
            "calibration_sequence": args.calibration_sequence if args.corpus else None,
            "token_ids": input_ids.tolist(),
        },
        "configuration": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "reuse_mlpackage": args.reuse_mlpackage,
            "accelerated_dtype": "float16",
            "oracle": "FP32 Torch using the same dense checkpoint",
            "accuracy_rtol": ACCURACY_RTOL,
            "accuracy_atol": ACCURACY_ATOL,
            "ane_operation_share_threshold": ANE_OPERATION_SHARE_THRESHOLD,
            "performance_speedup_threshold": PERFORMANCE_SPEEDUP_THRESHOLD,
            "storage": "unified_memory",
        },
        "compile_timings_seconds": compile_timings,
        "setup_total_seconds": setup_total_seconds,
        "oracle_load_seconds": {
            "torch_fp32": oracle_load_seconds,
            "mlx_lm": mlx_load_seconds,
        },
        "mlpackage_bytes": _package_bytes(mlpackage),
        "compute_plan": plan,
        "accuracy": {
            "ane_hidden_vs_torch_fp32": ane_hidden_accuracy,
            "ane_hidden_vs_torch_fp16_rounded": ane_hidden_rounded_accuracy,
            "ane_logits_vs_torch_fp32": ane_logits_accuracy,
            "ane_logits_vs_torch_fp16_rounded": ane_logits_rounded_accuracy,
            "mlx_logits_vs_torch_fp32": mlx_logits_accuracy,
            "mlx_logits_vs_torch_bfloat16_rounded": (mlx_logits_rounded_accuracy),
            "ane_logits_vs_mlx": ane_vs_mlx,
        },
        "teacher_distribution": {
            "ane_vs_torch_fp32": ane_distribution,
            "mlx_vs_torch_fp32": mlx_distribution,
            "ane_vs_mlx": ane_mlx_distribution,
        },
        "timings_seconds": {
            "ane_median": ane_median,
            "mlx_median": mlx_median,
            "mlx_over_ane": mlx_over_ane,
            "ane_samples": ane_samples,
            "mlx_samples": mlx_samples,
        },
        "gates": {
            "ane_hidden": hidden_gate,
            "ane_logits": logits_gate,
            "ane_distribution": ane_distribution_gate,
            "mlx_distribution": mlx_distribution_gate,
            "accuracy": accuracy_gate,
            "performance": performance_gate,
            "placement": placement_gate,
            "dispatch_ready": accuracy_gate and performance_gate and placement_gate,
        },
        "scope": (
            "Complete fixed-shape frozen Qwen3 teacher forward only. "
            "Core ML has no autograd, so the differentiable student and "
            "backward pass remain on MLX Metal."
        ),
    }
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

    del ane_model, mlx_model, mx_ids
    gc.collect()
    if mx.metal.is_available():
        mx.clear_cache()


if __name__ == "__main__":
    main()
