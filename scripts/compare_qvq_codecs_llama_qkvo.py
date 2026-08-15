#!/usr/bin/env python3
"""Compare QVQ trellis topologies on one full-width Llama attention block."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from analyze_gptq_low_bit_grid import (
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

QKVO_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)
ARM_CONFIG = {
    "v2": {"vector_size": 2, "trellis_window": 16, "dual_v2": False},
    "dual-v2": {"vector_size": 2, "trellis_window": 16, "dual_v2": True},
    "v4": {"vector_size": 4, "trellis_window": 16, "dual_v2": False},
    "l18-v4": {"vector_size": 4, "trellis_window": 18, "dual_v2": False},
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--rates", nargs="+", type=float, default=(1, 1.5, 2, 2.5))
    parser.add_argument("--arms", nargs="+", choices=tuple(ARM_CONFIG), default=tuple(ARM_CONFIG))
    parser.add_argument("--calibration-rows", type=int, default=8)
    parser.add_argument("--evaluation-rows", type=int, default=8)
    parser.add_argument("--evaluation-row-offset", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=18240)
    parser.add_argument("--trellis-batch-size", type=int)
    return parser


def _qkvo_modules(model: torch.nn.Module, *, layer_count: int) -> dict[str, torch.nn.Linear]:
    modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.endswith(QKVO_SUFFIXES)
    }
    expected = layer_count * len(QKVO_SUFFIXES)
    if len(modules) != expected:
        raise ValueError(f"expected {expected} QKVO modules in {layer_count} decoder layers, found {tuple(modules)}")
    return modules


def _joined(outputs: dict[str, torch.Tensor], names: tuple[str, ...]) -> torch.Tensor:
    rows = {outputs[name].shape[0] for name in names}
    if len(rows) != 1:
        raise ValueError("QKVO outputs do not share one held-out token geometry")
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


def main() -> None:
    args = _parser().parse_args()
    if args.layers < 1:
        raise ValueError("layer count must be positive")
    if args.evaluation_row_offset < args.calibration_rows:
        raise ValueError("evaluation rows must be disjoint from calibration rows")
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

    calibration, calibration_stats = load_nm_calibration_batches(
        tokenizer,
        config,
        dataset_path=args.dataset,
        rows=args.calibration_rows,
        concat_size=None,
        batch_size=1,
    )
    evaluation, evaluation_stats = load_nm_evaluation_batch(
        tokenizer,
        dataset_path=args.dataset,
        row_offset=args.evaluation_row_offset,
        rows=args.evaluation_rows,
        max_length=args.max_length,
    )
    evaluation_rows = _unpadded_evaluation_rows(evaluation, device)
    modules = _qkvo_modules(model, layer_count=args.layers)
    module_names = tuple(modules)

    print("Capturing QKVO calibration Hessians", flush=True)
    hessians, sample_counts = capture_calibration_hessians(model, calibration, modules, device=device)
    print("Capturing dense held-out QKVO/layer/logit outputs", flush=True)
    dense_logits, dense_inputs, dense_outputs = _capture_forward_rows(
        model,
        evaluation_rows,
        modules,
        capture_inputs=True,
        layer_count=args.layers,
    )
    dense_qkvo = _joined(dense_outputs, module_names)
    original_weights = {name: module.weight.detach().cpu().float().clone() for name, module in modules.items()}

    report = {
        "settings": {
            "model": str(args.model),
            "source_layers": source_layers,
            "tested_layers": args.layers,
            "modules": list(module_names),
            "module_shapes": {name: list(module.weight.shape) for name, module in modules.items()},
            "rates": list(rates),
            "arms": list(args.arms),
            "seed": args.seed,
            "device": str(device),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "performance_qos_requested": qos_requested,
            "calibration": calibration_stats,
            "calibration_samples": sample_counts,
            "evaluation": evaluation_stats,
            "evaluation_batch_size": 1,
            "rounding": "block_ldlq",
            "serialization": "disabled; dense reconstruction comparison",
        },
        "results": {},
    }

    for rate in rates:
        report["results"][str(rate)] = {}
        for arm in args.arms:
            started = time.perf_counter()
            geometry = ARM_CONFIG[arm]
            batch_size = args.trellis_batch_size or default_qvq_trellis_batch_size(
                rate,
                device,
                trellis_window=geometry["trellis_window"],
            )
            reconstructions: dict[str, torch.Tensor] = {}
            local_outputs: dict[str, torch.Tensor] = {}
            weight_metrics = {}
            print(f"Starting W{rate:g} {arm} with trellis batch {batch_size}", flush=True)
            for index, (name, module) in enumerate(modules.items(), start=1):
                module_started = time.perf_counter()
                result = quantize_qvq_linear(
                    original_weights[name].to(device),
                    hessians[name].to(device),
                    bits=rate,
                    seed=args.seed,
                    trellis_batch_size=batch_size,
                    **geometry,
                )
                reconstruction = result.weight.detach().cpu().float()
                reconstructions[name] = reconstruction
                weight_metrics[name] = _weight_metrics(original_weights[name], reconstruction)
                bias = None if module.bias is None else module.bias.detach().cpu().float()
                local_outputs[name] = F.linear(dense_inputs[name], reconstruction, bias)
                print(
                    f"W{rate:g} {arm}: {index}/{len(modules)} {name} "
                    f"in {time.perf_counter() - module_started:.2f}s",
                    flush=True,
                )

            with torch.no_grad():
                for name, module in modules.items():
                    module.weight.copy_(reconstructions[name].to(device=device, dtype=module.weight.dtype))
            quantized_logits, _, live_outputs = _capture_forward_rows(
                model,
                evaluation_rows,
                modules,
                capture_inputs=False,
                layer_count=args.layers,
            )
            local_qkvo = _joined(local_outputs, module_names)
            live_qkvo = _joined(live_outputs, module_names)
            arm_report = {
                "seconds": time.perf_counter() - started,
                "trellis_batch_size": batch_size,
                "weight": {
                    "modules": weight_metrics,
                    "mean_mse": _mean([metric["mse"] for metric in weight_metrics.values()]),
                    "mean_relative_l2": _mean([metric["relative_l2"] for metric in weight_metrics.values()]),
                },
                "local_qkvo": tensor_metrics(dense_qkvo, local_qkvo, normalize_distribution=True),
                "live_qkvo": tensor_metrics(dense_qkvo, live_qkvo, normalize_distribution=True),
                "layer": tensor_metrics(
                    dense_outputs[f"layer.{args.layers - 1}.hidden"],
                    live_outputs[f"layer.{args.layers - 1}.hidden"],
                    normalize_distribution=True,
                ),
                "logits": tensor_metrics(dense_logits, quantized_logits, normalize_distribution=False),
            }
            report["results"][str(rate)][arm] = arm_report
            logits = arm_report["logits"]
            print(
                f"W{rate:g} {arm}: relL2={arm_report['weight']['mean_relative_l2']:.6f} "
                f"localKL={arm_report['local_qkvo']['kl_forward']['mean']:.6f} "
                f"liveKL={arm_report['live_qkvo']['kl_forward']['mean']:.6f} "
                f"layerKL={arm_report['layer']['kl_forward']['mean']:.6f} "
                f"logitKL={logits['kl_forward']['mean']:.6f} "
                f"top1={logits['top1_agreement']:.4f} top5={logits['top5_overlap']['mean']:.4f}",
                flush=True,
            )
            with torch.no_grad():
                for name, module in modules.items():
                    module.weight.copy_(original_weights[name].to(device=device, dtype=module.weight.dtype))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            del reconstructions, local_outputs, quantized_logits, live_outputs
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
