#!/usr/bin/env python3
"""Compare QVQ trellis topologies on full-width Llama attention blocks."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import time
import uuid
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


try:
    from analyze_gptq_low_bit_grid import (
        capture_calibration_hessians,
        capture_forward,
        load_nm_calibration_batches,
        load_nm_evaluation_batch,
        request_performance_qos,
        tensor_metrics,
    )
except ModuleNotFoundError:  # Package import used by unit tests.
    from scripts.analyze_gptq_low_bit_grid import (
        capture_calibration_hessians,
        capture_forward,
        load_nm_calibration_batches,
        load_nm_evaluation_batch,
        request_performance_qos,
        tensor_metrics,
    )
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from gptqmodel.quantization.qvq import (
    default_qvq_trellis_batch_size,
    quantize_qvq_linear,
)
from gptqmodel.quantization.qvq_rates import normalize_qvq_rate
from gptqmodel.quantization.qvq_yaqa import capture_yaqa_sketch_b


QKVO_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)
MLP_SUFFIXES = (
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
MODULE_SCOPE_SUFFIXES = {
    "qkvo": QKVO_SUFFIXES,
    "all-linear": (*QKVO_SUFFIXES, *MLP_SUFFIXES),
}
ARM_CONFIG = {
    "v2": {"vector_size": 2, "trellis_window": 16, "dual_v2": False},
    "v2b2-p32": {
        "vector_size": 2,
        "trellis_window": 16,
        "dual_v2": False,
        "v2b2_p32": True,
        "bank_count": 2,
    },
    "v2b4-p64": {
        "vector_size": 2,
        "trellis_window": 16,
        "dual_v2": False,
        "v2b4_p64": True,
        "bank_count": 4,
    },
    "dual-v2": {"vector_size": 2, "trellis_window": 16, "dual_v2": True},
    "v4": {"vector_size": 4, "trellis_window": 16, "dual_v2": False},
    "l18-v4": {"vector_size": 4, "trellis_window": 18, "dual_v2": False},
    "v2-yaqa": {"vector_size": 2, "trellis_window": 16, "dual_v2": False, "rounding": "yaqa"},
    "v2b2-p32-yaqa-fixed": {
        "vector_size": 2,
        "trellis_window": 16,
        "dual_v2": False,
        "v2b2_p32": True,
        "bank_count": 2,
        "rounding": "yaqa",
        "yaqa_v2b2_family_mode": "fixed_block_ldlq",
    },
    "v2b2-p32-yaqa": {
        "vector_size": 2,
        "trellis_window": 16,
        "dual_v2": False,
        "v2b2_p32": True,
        "bank_count": 2,
        "rounding": "yaqa",
        "yaqa_v2b2_family_mode": "reselect",
    },
    "v2b2-p32-yaqa-spectral": {
        "vector_size": 2,
        "trellis_window": 16,
        "dual_v2": False,
        "v2b2_p32": True,
        "bank_count": 2,
        "rounding": "yaqa",
        "yaqa_v2b2_family_mode": "reselect",
        "yaqa_spectral_refinement": True,
    },
    "v2b2-p32-yaqa-spectral-fixed": {
        "vector_size": 2,
        "trellis_window": 16,
        "dual_v2": False,
        "v2b2_p32": True,
        "bank_count": 2,
        "rounding": "yaqa",
        "yaqa_v2b2_family_mode": "fixed_block_ldlq",
        "yaqa_spectral_refinement": True,
    },
    "v2b2-p32-yaqa-spectral-push-fixed": {
        "vector_size": 2,
        "trellis_window": 16,
        "dual_v2": False,
        "v2b2_p32": True,
        "bank_count": 2,
        "rounding": "yaqa",
        "yaqa_v2b2_family_mode": "fixed_block_ldlq",
        "yaqa_spectral_push": True,
    },
    "v2b2-p32-yaqa-spectral-push": {
        "vector_size": 2,
        "trellis_window": 16,
        "dual_v2": False,
        "v2b2_p32": True,
        "bank_count": 2,
        "rounding": "yaqa",
        "yaqa_v2b2_family_mode": "reselect",
        "yaqa_spectral_push": True,
    },
}
DEFAULT_ARMS = ("v2", "v2b2-p32")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument(
        "--module-scope",
        choices=tuple(MODULE_SCOPE_SUFFIXES),
        default="qkvo",
        help="Quantize attention Q/K/V/O only, or every Q/K/V/O and gate/up/down projection.",
    )
    parser.add_argument("--rates", nargs="+", type=float, default=(1, 1.5, 2, 2.5))
    parser.add_argument("--arms", nargs="+", choices=tuple(ARM_CONFIG), default=DEFAULT_ARMS)
    parser.add_argument("--calibration-rows", type=int, default=64)
    parser.add_argument("--evaluation-rows", type=int, default=64)
    parser.add_argument("--evaluation-row-offset", type=int, default=64)
    parser.add_argument("--yaqa-rows", type=int, default=512)
    parser.add_argument("--yaqa-row-offset", type=int)
    parser.add_argument("--yaqa-batch-size", type=int, default=8)
    parser.add_argument("--yaqa-seed", type=int, default=0)
    parser.add_argument("--yaqa-spectral-ranks", nargs="+", type=int, default=(8, 16, 32))
    parser.add_argument("--yaqa-spectral-lambdas", nargs="+", type=float, default=(0.1, 0.25, 0.5, 1.0))
    parser.add_argument("--yaqa-spectral-push-alphas", nargs="+", type=float, default=(0.25, 0.5, 1.0))
    parser.add_argument(
        "--yaqa-factor-cache",
        type=Path,
        help="Optional validated CPU cache shared by matched YAQA rate workers.",
    )
    parser.add_argument(
        "--prepare-yaqa-only",
        action="store_true",
        help="Collect and cache Sketch-B factors, then exit before Hessian capture and quantization.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional per-row truncation limit; omitted means each row's full tokenized length.",
    )
    parser.add_argument("--seed", type=int, default=18240)
    parser.add_argument("--trellis-batch-size", type=int)
    return parser


def _padded_batch_chunks(
    encoded: dict[str, torch.Tensor],
    *,
    batch_size: int,
) -> list[dict[str, torch.Tensor]]:
    """Split padded rows into batches while trimming padding-only edge columns."""

    attention_mask = encoded.get("attention_mask")
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
        raise ValueError("YAQA encoding must contain a rank-2 attention mask")
    if batch_size < 1:
        raise ValueError("YAQA batch size must be positive")
    batches = []
    for start in range(0, attention_mask.shape[0], batch_size):
        stop = min(start + batch_size, attention_mask.shape[0])
        mask = attention_mask[start:stop]
        active_columns = mask.ne(0).any(dim=0).nonzero(as_tuple=False).flatten()
        if active_columns.numel() == 0:
            raise ValueError("YAQA batch contains no valid tokens")
        column_start = int(active_columns[0])
        column_stop = int(active_columns[-1]) + 1
        batch = {}
        for name, value in encoded.items():
            if value.ndim >= 2 and tuple(value.shape[:2]) == tuple(attention_mask.shape):
                batch[name] = value[start:stop, column_start:column_stop].contiguous()
            else:
                batch[name] = value[start:stop].contiguous()
        batches.append(batch)
    return batches


def _yaqa_cache_metadata(args: argparse.Namespace, module_shapes: dict[str, list[int]], row_offset: int) -> dict:
    metadata = {
        "version": 1,
        "model": str(args.model.resolve()),
        "dataset": str(args.dataset.resolve()),
        "layers": args.layers,
        "module_shapes": module_shapes,
        "rows": args.yaqa_rows,
        "row_offset": row_offset,
        "batch_size": args.yaqa_batch_size,
        "seed": args.yaqa_seed,
        "max_length": args.max_length,
    }
    # Preserve compatibility with existing QKVO caches while making expanded
    # all-linear caches self-describing and impossible to reuse accidentally.
    if args.module_scope != "qkvo":
        metadata["version"] = 2
        metadata["module_scope"] = args.module_scope
    return metadata


def _load_yaqa_factor_cache(
    path: Path,
    *,
    expected_metadata: dict,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("metadata") != expected_metadata:
        raise ValueError(f"YAQA factor cache metadata does not match this sweep: {path}")
    input_hessians = payload.get("input_hessians")
    output_hessians = payload.get("output_hessians")
    stats = payload.get("stats")
    expected_names = set(expected_metadata["module_shapes"])
    if not isinstance(input_hessians, dict) or not isinstance(output_hessians, dict) or not isinstance(stats, dict):
        raise ValueError(f"YAQA factor cache has an invalid payload: {path}")
    if set(input_hessians) != expected_names or set(output_hessians) != expected_names:
        raise ValueError(f"YAQA factor cache module names do not match this sweep: {path}")
    for name, shape in expected_metadata["module_shapes"].items():
        out_features, in_features = shape
        input_factor = input_hessians[name]
        output_factor = output_hessians[name]
        if (
            not isinstance(input_factor, torch.Tensor)
            or input_factor.device.type != "cpu"
            or input_factor.dtype != torch.float32
            or tuple(input_factor.shape) != (in_features, in_features)
            or not input_factor.is_contiguous()
        ):
            raise ValueError(f"YAQA input factor {name} has invalid storage or geometry")
        if (
            not isinstance(output_factor, torch.Tensor)
            or output_factor.device.type != "cpu"
            or output_factor.dtype != torch.float32
            or tuple(output_factor.shape) != (out_features, out_features)
            or not output_factor.is_contiguous()
        ):
            raise ValueError(f"YAQA output factor {name} has invalid storage or geometry")
    return input_hessians, output_hessians, stats


def _save_yaqa_factor_cache(
    path: Path,
    *,
    metadata: dict,
    input_hessians: dict[str, torch.Tensor],
    output_hessians: dict[str, torch.Tensor],
    stats: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(
        {
            "metadata": metadata,
            "input_hessians": input_hessians,
            "output_hessians": output_hessians,
            "stats": stats,
        },
        temporary,
    )
    temporary.replace(path)


def _quantized_linear_modules(
    model: torch.nn.Module,
    *,
    layer_count: int,
    module_scope: str,
) -> dict[str, torch.nn.Linear]:
    """Select the exact decoder projection set requested by the comparison run."""

    try:
        suffixes = MODULE_SCOPE_SUFFIXES[module_scope]
    except KeyError as error:
        raise ValueError(f"unsupported module scope: {module_scope}") from error
    modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.endswith(suffixes)
    }
    expected = layer_count * len(suffixes)
    if len(modules) != expected:
        raise ValueError(
            f"expected {expected} {module_scope} modules in {layer_count} decoder layers, found {tuple(modules)}"
        )
    return modules


def _qkvo_modules(model: torch.nn.Module, *, layer_count: int) -> dict[str, torch.nn.Linear]:
    """Retain the original QKVO-only selector for callers importing this helper."""

    return _quantized_linear_modules(model, layer_count=layer_count, module_scope="qkvo")


def _joined(outputs: dict[str, torch.Tensor], names: tuple[str, ...]) -> torch.Tensor:
    rows = {outputs[name].shape[0] for name in names}
    if len(rows) != 1:
        raise ValueError("selected module outputs do not share one held-out token geometry")
    return torch.cat([outputs[name] for name in names], dim=-1)


def _weight_metrics(dense: torch.Tensor, quantized: torch.Tensor) -> dict[str, float]:
    error = quantized.double() - dense.double()
    dense_norm = dense.double().norm().clamp_min(torch.finfo(torch.float64).eps)
    error_norm = error.norm()
    return {
        "mse": error.square().mean().item(),
        "relative_l2": (error_norm / dense_norm).item(),
        "sqnr_db": (20 * torch.log10(dense_norm / error_norm.clamp_min(torch.finfo(torch.float64).eps))).item(),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _selector_metrics(histogram: list[int]) -> dict[str, object] | None:
    total = sum(histogram)
    if total == 0:
        return None
    probabilities = [count / total for count in histogram if count]
    return {
        "count": total,
        "histogram": histogram,
        "nonzero_fraction": sum(histogram[1:]) / total,
        "entropy_bits": -sum(probability * math.log2(probability) for probability in probabilities),
        "selector_bpw": 2 / 64,
    }


def _unpadded_evaluation_rows(
    encoded: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], ...]:
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None or attention_mask.ndim != 2:
        raise ValueError("evaluation encoding must contain a rank-2 attention mask")
    rows = []
    for row_index in range(attention_mask.shape[0]):
        valid = attention_mask[row_index].ne(0).nonzero(as_tuple=False).flatten()
        if valid.numel() == 0:
            raise ValueError(f"evaluation row {row_index} contains no valid tokens")
        start = int(valid[0])
        stop = int(valid[-1]) + 1
        row = {}
        for name, value in encoded.items():
            if value.ndim >= 2 and tuple(value.shape[:2]) == tuple(attention_mask.shape):
                row[name] = value[row_index : row_index + 1, start:stop].to(device)
            else:
                row[name] = value[row_index : row_index + 1].to(device)
        if not bool(row["attention_mask"].ne(0).all()):
            raise ValueError(f"evaluation row {row_index} was not fully unpadded")
        rows.append(row)
    return tuple(rows)


@torch.inference_mode()
def _capture_forward_rows(
    model: torch.nn.Module,
    rows: tuple[dict[str, torch.Tensor], ...],
    modules: dict[str, torch.nn.Linear],
    *,
    capture_inputs: bool,
    layer_count: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    logits = []
    inputs: dict[str, list[torch.Tensor]] = defaultdict(list)
    outputs: dict[str, list[torch.Tensor]] = defaultdict(list)
    for row in rows:
        row_logits, row_inputs, row_outputs = capture_forward(
            model,
            row,
            modules,
            capture_inputs=capture_inputs,
            layer_count=layer_count,
        )
        logits.append(row_logits)
        for name, value in row_inputs.items():
            inputs[name].append(value)
        for name, value in row_outputs.items():
            outputs[name].append(value)
    return (
        torch.cat(logits, dim=0),
        {name: torch.cat(values, dim=0) for name, values in inputs.items()},
        {name: torch.cat(values, dim=0) for name, values in outputs.items()},
    )


class _WeightedMetricAccumulator:
    """Merge row metrics without retaining full-vocabulary tensors."""

    def __init__(self) -> None:
        self.weight = 0
        self.template = None
        self.totals: dict[tuple[str, ...], float] = {}
        self.maxima: dict[tuple[str, ...], float] = {}
        self.booleans: dict[tuple[str, ...], bool] = {}
        self.shape: list[int] | None = None

    def add(self, metrics: dict, *, rows: int) -> None:
        if rows < 1:
            raise ValueError("streamed metric rows must be positive")
        if self.template is None:
            self.template = metrics
        self.weight += rows

        def visit(value, path: tuple[str, ...]):
            if isinstance(value, dict):
                for name, child in value.items():
                    visit(child, (*path, name))
            elif path == ("shape",):
                if self.shape is None:
                    self.shape = list(value)
                else:
                    if self.shape[1:] != list(value)[1:]:
                        raise ValueError("streamed metric feature shapes do not match")
                    self.shape[0] += int(value[0])
            elif isinstance(value, bool):
                self.booleans[path] = self.booleans.get(path, True) and value
            elif isinstance(value, (int, float)):
                if path[-1] in {"max", "max_abs_error"}:
                    self.maxima[path] = max(self.maxima.get(path, -math.inf), float(value))
                else:
                    self.totals[path] = self.totals.get(path, 0.0) + float(value) * rows

        visit(metrics, ())

    def result(self) -> dict:
        if self.template is None or self.weight < 1:
            raise ValueError("streamed metric accumulator is empty")

        def build(value, path: tuple[str, ...]):
            if isinstance(value, dict):
                return {name: build(child, (*path, name)) for name, child in value.items()}
            if path == ("shape",):
                return self.shape
            if isinstance(value, bool):
                return self.booleans[path]
            if isinstance(value, (int, float)):
                if path[-1] in {"max", "max_abs_error"}:
                    return self.maxima[path]
                return self.totals[path] / self.weight
            return value

        result = build(self.template, ())
        result["aggregation"] = "exact token-weighted means; row-weighted auxiliary quantiles"
        result["streamed_rows"] = self.weight
        return result

    def mean(self, *path: str) -> float | None:
        key = tuple(path)
        return None if self.weight == 0 or key not in self.totals else self.totals[key] / self.weight


@torch.inference_mode()
def _streaming_compare_models(
    dense_model: torch.nn.Module,
    quantized_model: torch.nn.Module,
    rows: tuple[dict[str, torch.Tensor], ...],
    dense_modules: dict[str, torch.nn.Linear],
    quantized_modules: dict[str, torch.nn.Linear],
    reconstructions: dict[str, torch.Tensor],
    *,
    layer_count: int,
    progress_label: str,
    module_scope: str = "qkvo",
) -> dict:
    module_names = tuple(dense_modules)
    local_accumulator = _WeightedMetricAccumulator()
    live_accumulator = _WeightedMetricAccumulator()
    layer_accumulators = {index: _WeightedMetricAccumulator() for index in range(layer_count)}
    logit_accumulator = _WeightedMetricAccumulator()
    for row_index, row in enumerate(rows, start=1):
        dense_logits, dense_inputs, dense_outputs = capture_forward(
            dense_model,
            row,
            dense_modules,
            capture_inputs=True,
            layer_count=layer_count,
        )
        quantized_logits, _, live_outputs = capture_forward(
            quantized_model,
            row,
            quantized_modules,
            capture_inputs=False,
            layer_count=layer_count,
        )
        token_count = dense_logits.shape[0]
        local_outputs = {
            name: F.linear(
                dense_inputs[name],
                reconstructions[name],
                None if dense_modules[name].bias is None else dense_modules[name].bias.detach().cpu().float(),
            )
            for name in module_names
        }
        dense_joined = _joined(dense_outputs, module_names)
        local_accumulator.add(
            tensor_metrics(
                dense_joined,
                _joined(local_outputs, module_names),
                normalize_distribution=True,
            ),
            rows=token_count,
        )
        live_accumulator.add(
            tensor_metrics(
                dense_joined,
                _joined(live_outputs, module_names),
                normalize_distribution=True,
            ),
            rows=token_count,
        )
        for layer_index, accumulator in layer_accumulators.items():
            accumulator.add(
                tensor_metrics(
                    dense_outputs[f"layer.{layer_index}.hidden"],
                    live_outputs[f"layer.{layer_index}.hidden"],
                    normalize_distribution=True,
                ),
                rows=token_count,
            )
        logit_accumulator.add(
            tensor_metrics(
                dense_logits,
                quantized_logits,
                normalize_distribution=False,
                include_top10=True,
            ),
            rows=token_count,
        )
        if row_index % 16 == 0 or row_index == len(rows):
            print(
                f"{progress_label}: eval {row_index}/{len(rows)} rows, "
                f"finalKL={logit_accumulator.mean('kl_forward', 'mean'):.6f} "
                f"top1={logit_accumulator.mean('top1_agreement'):.4f} "
                f"top5={logit_accumulator.mean('top5_overlap', 'mean'):.4f} "
                f"top10={logit_accumulator.mean('top10_overlap', 'mean'):.4f}",
                flush=True,
            )
    local_metrics = local_accumulator.result()
    live_metrics = live_accumulator.result()
    result = {
        "local_modules": local_metrics,
        "live_modules": live_metrics,
        "layers": {str(index): accumulator.result() for index, accumulator in layer_accumulators.items()},
        "logits": logit_accumulator.result(),
    }
    if module_scope == "qkvo":
        result["local_qkvo"] = local_metrics
        result["live_qkvo"] = live_metrics
    return result


def main() -> None:
    args = _parser().parse_args()
    if args.layers < 1:
        raise ValueError("layer count must be positive")
    if args.prepare_yaqa_only and args.yaqa_factor_cache is None:
        raise ValueError("--prepare-yaqa-only requires --yaqa-factor-cache")
    if args.evaluation_row_offset < args.calibration_rows:
        raise ValueError("evaluation rows must be disjoint from calibration rows")
    yaqa_enabled = any(ARM_CONFIG[arm].get("rounding") == "yaqa" for arm in args.arms)
    yaqa_row_offset = (
        args.evaluation_row_offset + args.evaluation_rows if args.yaqa_row_offset is None else args.yaqa_row_offset
    )
    if yaqa_enabled and (
        args.yaqa_rows < 1
        or args.yaqa_batch_size < 1
        or yaqa_row_offset < args.evaluation_row_offset + args.evaluation_rows
    ):
        raise ValueError("YAQA rows must be positive and disjoint from calibration and evaluation rows")
    rates = tuple(normalize_qvq_rate(rate) for rate in args.rates)
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.manual_seed(args.seed)
    qos_requested = request_performance_qos()

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    source_layers = int(config.num_hidden_layers)
    config.num_hidden_layers = args.layers
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    modules = _quantized_linear_modules(model, layer_count=args.layers, module_scope=args.module_scope)
    module_names = tuple(modules)
    module_shapes = {name: list(module.weight.shape) for name, module in modules.items()}
    yaqa_input_hessians = {}
    yaqa_output_hessians = {}
    yaqa_stats = None
    if yaqa_enabled:
        cache_metadata = _yaqa_cache_metadata(args, module_shapes, yaqa_row_offset)
        if args.yaqa_factor_cache is not None and args.yaqa_factor_cache.is_file():
            yaqa_input_hessians, yaqa_output_hessians, yaqa_stats = _load_yaqa_factor_cache(
                args.yaqa_factor_cache,
                expected_metadata=cache_metadata,
            )
            yaqa_stats = dict(yaqa_stats)
            yaqa_stats["cache"] = "loaded"
            yaqa_stats["cache_path"] = str(args.yaqa_factor_cache)
            print(f"Loaded validated YAQA Sketch-B factors from {args.yaqa_factor_cache}", flush=True)
        else:
            yaqa_encoded, yaqa_data_stats = load_nm_evaluation_batch(
                tokenizer,
                dataset_path=args.dataset,
                row_offset=yaqa_row_offset,
                rows=args.yaqa_rows,
                max_length=args.max_length,
            )
            yaqa_batches = _padded_batch_chunks(yaqa_encoded, batch_size=args.yaqa_batch_size)
            print(
                f"Capturing YAQA Sketch B from rows {yaqa_row_offset}:{yaqa_row_offset + args.yaqa_rows} "
                f"in {len(yaqa_batches)} batches",
                flush=True,
            )

            def sketch_progress(stats):
                print(
                    f"Sketch-B {stats['completed_batches']}/{stats['total_batches']} batches, "
                    f"{stats['completed_sequences']}/{args.yaqa_rows} rows, {stats['valid_tokens']} valid tokens",
                    flush=True,
                )

            yaqa_started = time.perf_counter()
            yaqa_input_hessians, yaqa_output_hessians, yaqa_stats = capture_yaqa_sketch_b(
                model,
                yaqa_batches,
                modules,
                device=device,
                seed=args.yaqa_seed,
                minimum_sequences=args.yaqa_rows,
                checkpoint_modules=tuple(model.model.layers),
                progress_callback=sketch_progress,
            )
            yaqa_stats["collection_seconds"] = time.perf_counter() - yaqa_started
            yaqa_stats["data"] = yaqa_data_stats
            yaqa_stats["row_offset"] = yaqa_row_offset
            yaqa_stats["batch_size"] = args.yaqa_batch_size
            yaqa_stats["cache"] = "collected"
            if args.yaqa_factor_cache is not None:
                _save_yaqa_factor_cache(
                    args.yaqa_factor_cache,
                    metadata=cache_metadata,
                    input_hessians=yaqa_input_hessians,
                    output_hessians=yaqa_output_hessians,
                    stats=yaqa_stats,
                )
                yaqa_stats["cache_path"] = str(args.yaqa_factor_cache)
                print(f"Saved YAQA Sketch-B factors to {args.yaqa_factor_cache}", flush=True)
    if args.prepare_yaqa_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"metadata": cache_metadata, "stats": yaqa_stats}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("YAQA Sketch-B factor preparation complete", flush=True)
        return

    calibration, calibration_stats = load_nm_calibration_batches(
        tokenizer,
        config,
        dataset_path=args.dataset,
        rows=args.calibration_rows,
        concat_size=None,
        batch_size=1,
    )
    if calibration_stats["concat_size"] is not None or calibration_stats["batch_size"] != 1:
        raise RuntimeError("codec comparison requires independent, non-concatenated calibration rows at batch 1")
    if calibration_stats["prepared_sequences"] != args.calibration_rows:
        raise RuntimeError("codec comparison calibration did not preserve one sequence per source row")
    evaluation, evaluation_stats = load_nm_evaluation_batch(
        tokenizer,
        dataset_path=args.dataset,
        row_offset=args.evaluation_row_offset,
        rows=args.evaluation_rows,
        max_length=args.max_length,
    )
    evaluation_rows = _unpadded_evaluation_rows(evaluation, device)

    print(f"Capturing {args.module_scope} calibration Hessians", flush=True)
    hessians, sample_counts = capture_calibration_hessians(model, calibration, modules, device=device)
    original_weights = {name: module.weight.detach().cpu().float().clone() for name, module in modules.items()}
    print("Loading an immutable dense replay model for row-streamed metrics", flush=True)
    dense_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval().to(device)
    dense_modules = _quantized_linear_modules(
        dense_model,
        layer_count=args.layers,
        module_scope=args.module_scope,
    )

    report = {
        "settings": {
            "model": str(args.model),
            "source_layers": source_layers,
            "tested_layers": args.layers,
            "module_scope": args.module_scope,
            "modules": list(module_names),
            "module_shapes": module_shapes,
            "rates": list(rates),
            "arms": list(args.arms),
            "seed": args.seed,
            "device": str(device),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "performance_qos_requested": qos_requested,
            "calibration": calibration_stats,
            "calibration_execution": "independent full rows; batch=1; no sequence concatenation",
            "calibration_samples": sample_counts,
            "evaluation": evaluation_stats,
            "evaluation_batch_size": 1,
            "evaluation_execution": "independent full rows; batch=1; no sequence concatenation",
            "yaqa_sketch_b": yaqa_stats,
            "yaqa_spectral_ranks": list(args.yaqa_spectral_ranks),
            "yaqa_spectral_lambdas": list(args.yaqa_spectral_lambdas),
            "yaqa_spectral_push_alphas": list(args.yaqa_spectral_push_alphas),
            "serialization": "disabled; dense reconstruction comparison",
        },
        "results": {},
    }

    for rate in rates:
        report["results"][str(rate)] = {}
        for arm in args.arms:
            started = time.perf_counter()
            geometry = dict(ARM_CONFIG[arm])
            if geometry.get("v2b2_p32") and rate > 3.5:
                report["results"][str(rate)][arm] = {
                    "status": "unsupported",
                    "reason": "V2B2-P32 supports W1 through W3.5",
                }
                print(f"Skipping W{rate:g} {arm}: V2B2-P32 supports W1 through W3.5", flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                continue
            rounding = geometry.get("rounding", "block_ldlq")
            if geometry.get("yaqa_spectral_refinement", False):
                geometry["yaqa_spectral_ranks"] = tuple(args.yaqa_spectral_ranks)
                geometry["yaqa_spectral_lambdas"] = tuple(args.yaqa_spectral_lambdas)
            if geometry.get("yaqa_spectral_push", False):
                geometry["yaqa_spectral_ranks"] = tuple(args.yaqa_spectral_ranks)
                geometry["yaqa_spectral_push_alphas"] = tuple(args.yaqa_spectral_push_alphas)
            batch_size = args.trellis_batch_size or default_qvq_trellis_batch_size(
                rate,
                device,
                trellis_window=geometry["trellis_window"],
            )
            reconstructions: dict[str, torch.Tensor] = {}
            weight_metrics = {}
            selector_histogram = [0, 0, 0, 0]
            alternative_bank_histogram = [0, 0, 0, 0]
            print(f"Starting W{rate:g} {arm} with trellis batch {batch_size}", flush=True)
            for index, (name, module) in enumerate(modules.items(), start=1):
                module_started = time.perf_counter()
                result = quantize_qvq_linear(
                    original_weights[name].to(device),
                    (yaqa_input_hessians[name] if rounding == "yaqa" else hessians[name]).to(device),
                    bits=rate,
                    output_hessian=(
                        yaqa_output_hessians[name].to(device) if rounding == "yaqa" else None
                    ),
                    seed=args.seed,
                    trellis_batch_size=batch_size,
                    **geometry,
                )
                reconstruction = result.weight.detach().cpu().float()
                reconstructions[name] = reconstruction
                weight_metrics[name] = _weight_metrics(original_weights[name], reconstruction)
                weight_metrics[name]["proxy_loss"] = float(result.proxy_loss)
                weight_metrics[name]["kronecker_proxy_loss"] = (
                    None if result.kronecker_proxy_loss is None else float(result.kronecker_proxy_loss)
                )
                weight_metrics[name]["yaqa_spectral"] = {
                    "selected": result.yaqa_spectral_selected,
                    "method": result.yaqa_spectral_method,
                    "rank": result.yaqa_spectral_rank,
                    "lambda": result.yaqa_spectral_lambda,
                    "alpha": result.yaqa_spectral_alpha,
                    "svd_device": result.yaqa_spectral_svd_device,
                    "concentration": result.yaqa_spectral_concentration,
                    "oracle_losses": result.yaqa_spectral_oracle_losses,
                    "absorption_efficiency": result.yaqa_spectral_absorption_efficiency,
                    "selector_churn": result.yaqa_spectral_selector_churn,
                    "family_changed": result.yaqa_spectral_family_changed,
                }
                if result.bank_ids is not None:
                    counts = torch.bincount(result.bank_ids.to(torch.int64).cpu(), minlength=4)
                    selector_histogram = [
                        current + int(count)
                        for current, count in zip(selector_histogram, counts.tolist(), strict=True)
                    ]
                if result.bank_alt_id is not None:
                    alternative_bank_histogram[int(result.bank_alt_id.item())] += 1
                print(
                    f"W{rate:g} {arm}: {index}/{len(modules)} {name} "
                    f"in {time.perf_counter() - module_started:.2f}s",
                    flush=True,
                )

            with torch.no_grad():
                for name, module in modules.items():
                    module.weight.copy_(reconstructions[name].to(device=device, dtype=module.weight.dtype))
            streamed_metrics = _streaming_compare_models(
                dense_model,
                model,
                evaluation_rows,
                dense_modules,
                modules,
                reconstructions,
                layer_count=args.layers,
                progress_label=f"W{rate:g} {arm}",
                module_scope=args.module_scope,
            )
            arm_report = {
                "seconds": time.perf_counter() - started,
                "trellis_batch_size": batch_size,
                "rounding": rounding,
                "effective_bpw": rate + (2 / 64 if geometry.get("v2b2_p32") or geometry.get("v2b4_p64") else 0),
                "bank_selectors": _selector_metrics(selector_histogram),
                "module_alternative_bank_histogram": (
                    alternative_bank_histogram if geometry.get("v2b2_p32") else None
                ),
                "weight": {
                    "modules": weight_metrics,
                    "mean_mse": _mean([metric["mse"] for metric in weight_metrics.values()]),
                    "mean_relative_l2": _mean([metric["relative_l2"] for metric in weight_metrics.values()]),
                },
                **streamed_metrics,
            }
            report["results"][str(rate)][arm] = arm_report
            logits = arm_report["logits"]
            layer_kl = _mean(
                [metrics["kl_forward"]["mean"] for metrics in arm_report["layers"].values()]
            )
            print(
                f"W{rate:g} {arm}: relL2={arm_report['weight']['mean_relative_l2']:.6f} "
                f"localKL={arm_report['local_modules']['kl_forward']['mean']:.6f} "
                f"liveKL={arm_report['live_modules']['kl_forward']['mean']:.6f} "
                f"layerKL={layer_kl:.6f} "
                f"logitKL={logits['kl_forward']['mean']:.6f} "
                f"top1={logits['top1_agreement']:.4f} "
                f"top5={logits['top5_overlap']['mean']:.4f} "
                f"top10={logits['top10_overlap']['mean']:.4f}",
                flush=True,
            )
            with torch.no_grad():
                for name, module in modules.items():
                    module.weight.copy_(original_weights[name].to(device=device, dtype=module.weight.dtype))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            del reconstructions, streamed_metrics
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
