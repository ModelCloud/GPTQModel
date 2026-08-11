#!/usr/bin/env python3
"""Measure low-bit GPTQ scale-grid error through a short decoder stack.

This diagnostic intentionally isolates grouped activation-weighted parameter
search and fake-quantized reconstruction before packing/backend effects.  It
reports both local module error (dense module inputs) and live propagated error
through a truncated model.  Intermediate KL metrics use channel distributions
standardized with dense activation statistics; final-logit KL uses raw logits.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization.config import (
    QuantizeConfig,
    ScaleSearchConfig,
)
from gptqmodel.quantization.quantizer import Quantizer

TARGET_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

CALIBRATION_TEXTS = (
    "Quantization maps continuous weights onto a small discrete grid.",
    "The quick brown fox jumps over the lazy dog while the rain begins.",
    "Explain why calibration activations can change a weighted error objective.",
    "A matrix multiplication combines input channels using learned coefficients.",
)
EVALUATION_TEXTS = (
    "Ultra-low-bit language models are difficult because",
    "When a symmetric distribution is represented by four integer codes,",
    "The difference between local reconstruction error and propagated error is",
    "A robust numerical comparison should include divergence, cosine, and",
)


def request_performance_qos() -> bool:
    """Request Darwin user-interactive QoS for the main and subsequently created workers."""

    if platform.system() != "Darwin":
        return False
    qos_class_user_interactive = 0x21
    libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    set_qos = libsystem.pthread_set_qos_class_self_np
    set_qos.argtypes = [ctypes.c_uint, ctypes.c_int]
    set_qos.restype = ctypes.c_int
    return set_qos(qos_class_user_interactive, 0) == 0


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value:
        return _first_tensor(value[0])
    raise TypeError(f"Cannot extract a tensor from {type(value)!r}")


def _summary(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().flatten()
    if values.numel() == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    quantiles = torch.quantile(values, torch.tensor([0.50, 0.95, 0.99]))
    return {
        "mean": values.mean().item(),
        "p50": quantiles[0].item(),
        "p95": quantiles[1].item(),
        "p99": quantiles[2].item(),
        "max": values.max().item(),
    }


@torch.inference_mode()
def tensor_metrics(
    dense: torch.Tensor,
    quantized: torch.Tensor,
    *,
    normalize_distribution: bool,
) -> dict[str, Any]:
    dense = dense.detach().float()
    quantized = quantized.detach().float()
    if dense.shape != quantized.shape:
        raise ValueError(f"metric shape mismatch: {tuple(dense.shape)} != {tuple(quantized.shape)}")

    error = quantized - dense
    dense_flat = dense.flatten()
    quantized_flat = quantized.flatten()
    error_flat = error.flatten()
    eps = torch.finfo(torch.float32).eps
    dense_energy = dense_flat.square().sum().clamp_min(eps)
    error_energy = error_flat.square().sum().clamp_min(eps)
    dense_centered = dense_flat - dense_flat.mean()
    quantized_centered = quantized_flat - quantized_flat.mean()

    dense_rows = dense.reshape(-1, dense.shape[-1])
    quantized_rows = quantized.reshape(-1, quantized.shape[-1])
    if normalize_distribution:
        dense_mean = dense_rows.mean(dim=-1, keepdim=True)
        dense_std = dense_rows.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        dense_logits = (dense_rows - dense_mean) / dense_std
        quantized_logits = (quantized_rows - dense_mean) / dense_std
    else:
        dense_logits = dense_rows
        quantized_logits = quantized_rows

    dense_log_prob = F.log_softmax(dense_logits, dim=-1)
    quantized_log_prob = F.log_softmax(quantized_logits, dim=-1)
    dense_prob = dense_log_prob.exp()
    quantized_prob = quantized_log_prob.exp()
    midpoint = (dense_prob + quantized_prob) * 0.5
    midpoint_log = midpoint.clamp_min(1e-30).log()
    kl_forward = (dense_prob * (dense_log_prob - quantized_log_prob)).sum(dim=-1)
    kl_reverse = (quantized_prob * (quantized_log_prob - dense_log_prob)).sum(dim=-1)
    js = 0.5 * (
        (dense_prob * (dense_log_prob - midpoint_log)).sum(dim=-1)
        + (quantized_prob * (quantized_log_prob - midpoint_log)).sum(dim=-1)
    )
    total_variation = 0.5 * (dense_prob - quantized_prob).abs().sum(dim=-1)
    hellinger = ((dense_prob.sqrt() - quantized_prob.sqrt()).square().sum(dim=-1) * 0.5).sqrt()
    row_cosine = F.cosine_similarity(dense_rows, quantized_rows, dim=-1)

    topk = min(5, dense.shape[-1])
    dense_topk = dense_logits.topk(topk, dim=-1).indices
    quantized_topk = quantized_logits.topk(topk, dim=-1).indices
    topk_overlap = (dense_topk.unsqueeze(-1) == quantized_topk.unsqueeze(-2)).any(dim=-1).float().mean(dim=-1)

    return {
        "shape": list(dense.shape),
        "finite": bool(torch.isfinite(quantized).all()),
        "mae": error_flat.abs().mean().item(),
        "rmse": error_flat.square().mean().sqrt().item(),
        "relative_l2": (error_energy / dense_energy).sqrt().item(),
        "sqnr_db": (10.0 * torch.log10(dense_energy / error_energy)).item(),
        "max_abs_error": error_flat.abs().max().item(),
        "abs_error": _summary(error_flat.abs()),
        "bias": error_flat.mean().item(),
        "error_std": error_flat.std(unbiased=False).item(),
        "cosine": F.cosine_similarity(dense_flat, quantized_flat, dim=0).item(),
        "pearson": F.cosine_similarity(dense_centered, quantized_centered, dim=0).item(),
        "row_cosine": _summary(row_cosine),
        "norm_ratio": (quantized_flat.norm() / dense_flat.norm().clamp_min(eps)).item(),
        "sign_agreement": ((dense_flat >= 0) == (quantized_flat >= 0)).float().mean().item(),
        "kl_forward": _summary(kl_forward),
        "kl_reverse": _summary(kl_reverse),
        "jensen_shannon": _summary(js),
        "total_variation": _summary(total_variation),
        "hellinger": _summary(hellinger),
        "top1_agreement": (dense_topk[:, 0] == quantized_topk[:, 0]).float().mean().item(),
        "top5_overlap": _summary(topk_overlap),
    }


def target_modules(model: nn.Module) -> dict[str, nn.Linear]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.endswith(TARGET_SUFFIXES)
    }


def decoder_layers(model: nn.Module) -> list[nn.Module]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("Expected a decoder model exposing model.layers")
    return list(layers)


@torch.inference_mode()
def capture_forward(
    model: nn.Module,
    encoded: dict[str, torch.Tensor],
    modules: dict[str, nn.Linear],
    *,
    capture_inputs: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    captured_inputs: dict[str, list[torch.Tensor]] = defaultdict(list)
    captured_outputs: dict[str, list[torch.Tensor]] = defaultdict(list)
    handles = []

    for name, module in modules.items():
        def module_hook(_module, args, output, module_name=name):
            if capture_inputs:
                captured_inputs[module_name].append(_first_tensor(args).detach().cpu().float())
            captured_outputs[module_name].append(_first_tensor(output).detach().cpu().float())

        handles.append(module.register_forward_hook(module_hook))

    for index, layer in enumerate(decoder_layers(model)):
        def layer_hook(_module, _args, output, layer_index=index):
            captured_outputs[f"layer.{layer_index}.hidden"].append(_first_tensor(output).detach().cpu().float())

        handles.append(layer.register_forward_hook(layer_hook))

    try:
        logits = model(**encoded, use_cache=False).logits.detach().cpu().float()
    finally:
        for handle in handles:
            handle.remove()

    merged_inputs = {name: torch.cat(values, dim=0) for name, values in captured_inputs.items()}
    merged_outputs = {name: torch.cat(values, dim=0) for name, values in captured_outputs.items()}
    return logits, merged_inputs, merged_outputs


@torch.inference_mode()
def quantize_module_weight(
    weight: torch.Tensor,
    calibration_input: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    sym: bool,
    adjacent_zero_search: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    rows, columns = weight.shape
    if columns % group_size:
        raise ValueError(f"{columns=} is not divisible by {group_size=}")
    groups = columns // group_size
    activation = calibration_input.reshape(-1, columns).float()
    importance = activation.square().mean(dim=0).reshape(groups, group_size).contiguous()
    grouped_weight = weight.float().reshape(rows, groups, group_size).contiguous()
    qcfg = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        sym=sym,
        mse=2.0,
        scale_search=ScaleSearchConfig.ACTIVATION,
        adaptive_clipping=None,
        offload_to_disk=False,
    )
    quantizer = Quantizer(qcfg=qcfg)
    quantizer.configure(perchannel=True)
    if not adjacent_zero_search:
        quantizer._search_adjacent_zero_points = lambda _maxq: False
        old_cpu_search = os.environ.get("GPTQMODEL_SCALE_SEARCH_CPU")
        os.environ["GPTQMODEL_SCALE_SEARCH_CPU"] = "0"
    else:
        old_cpu_search = None
    try:
        scale, zero = quantizer.find_params_batched(grouped_weight, weight=True, hessian=importance)
    finally:
        if not adjacent_zero_search:
            if old_cpu_search is None:
                os.environ.pop("GPTQMODEL_SCALE_SEARCH_CPU", None)
            else:
                os.environ["GPTQMODEL_SCALE_SEARCH_CPU"] = old_cpu_search

    q = torch.round(grouped_weight / scale.unsqueeze(-1))
    q.add_(zero.unsqueeze(-1)).clamp_(0, (1 << bits) - 1).sub_(zero.unsqueeze(-1))
    reconstructed = (q * scale.unsqueeze(-1)).reshape_as(weight)
    zero_one_fraction = (zero == 1).float().mean().item() if bits == 2 else 0.0
    return reconstructed, {
        "weight": tensor_metrics(weight.float(), reconstructed, normalize_distribution=True),
        "zero_one_fraction": zero_one_fraction,
        "zero_two_fraction": (zero == 2).float().mean().item() if bits == 2 else 0.0,
    }


def mean_metric(entries: dict[str, dict[str, Any]], path: tuple[str, ...]) -> float:
    values = []
    for entry in entries.values():
        value: Any = entry
        for key in path:
            value = value[key]
        values.append(float(value))
    return sum(values) / max(len(values), 1)


def print_arm_summary(name: str, result: dict[str, Any]) -> None:
    module_metrics = result["modules"]
    layer_metrics = result["layers"]
    logits = result["logits"]
    print(
        f"{name:22s} "
        f"w_rel={mean_metric(module_metrics, ('weight', 'relative_l2')):.5f} "
        f"local_KL={mean_metric(module_metrics, ('local', 'kl_forward', 'mean')):.6f} "
        f"live_KL={mean_metric(module_metrics, ('live', 'kl_forward', 'mean')):.6f} "
        f"layer_KL={mean_metric(layer_metrics, ('kl_forward', 'mean')):.6f} "
        f"logit_KL={logits['kl_forward']['mean']:.6f} "
        f"top1={logits['top1_agreement']:.4f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--performance-qos", action="store_true")
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(0)
    performance_qos = request_performance_qos() if args.performance_qos else False
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    os.environ.setdefault("GPTQMODEL_SCALE_SEARCH_CPU", "1")

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config.num_hidden_layers = args.layers
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    calibration = tokenizer(
        list(CALIBRATION_TEXTS),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )
    evaluation = tokenizer(
        list(EVALUATION_TEXTS),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
    )
    modules = target_modules(model)
    expected_modules = args.layers * len(TARGET_SUFFIXES)
    if len(modules) != expected_modules:
        raise ValueError(f"Expected {expected_modules} target modules, found {len(modules)}")

    _, calibration_inputs, _ = capture_forward(model, calibration, modules, capture_inputs=True)
    dense_logits, evaluation_inputs, dense_outputs = capture_forward(model, evaluation, modules, capture_inputs=True)
    original_weights = {name: module.weight.detach().cpu().clone() for name, module in modules.items()}

    arms = (
        ("w4-symmetric", 4, True, True),
        ("w3-symmetric", 3, True, True),
        ("w2-symmetric", 2, True, True),
        ("w2-asymmetric-legacy", 2, False, False),
        ("w2-asymmetric-adjacent", 2, False, True),
    )
    report: dict[str, Any] = {
        "settings": {
            "model": str(args.model),
            "layers": args.layers,
            "group_size": args.group_size,
            "max_length": args.max_length,
            "threads": args.threads,
            "performance_qos": performance_qos,
            "device": "cpu",
            "intermediate_distribution": "dense-standardized channel softmax",
            "final_distribution": "raw vocabulary logits softmax",
        },
        "arms": {},
    }

    for arm_name, bits, sym, adjacent in arms:
        started = time.perf_counter()
        module_results: dict[str, Any] = {}
        for module_name, module in modules.items():
            dense_weight = original_weights[module_name]
            reconstructed, weight_result = quantize_module_weight(
                dense_weight,
                calibration_inputs[module_name],
                bits=bits,
                group_size=args.group_size,
                sym=sym,
                adjacent_zero_search=adjacent,
            )
            local_output = F.linear(evaluation_inputs[module_name], reconstructed, module.bias)
            weight_result["local"] = tensor_metrics(
                dense_outputs[module_name], local_output, normalize_distribution=True
            )
            with torch.no_grad():
                module.weight.copy_(reconstructed)
            module_results[module_name] = weight_result

        quant_logits, _, live_outputs = capture_forward(model, evaluation, modules, capture_inputs=False)
        for module_name in modules:
            module_results[module_name]["live"] = tensor_metrics(
                dense_outputs[module_name], live_outputs[module_name], normalize_distribution=True
            )
        layer_results = {
            f"layer.{index}": tensor_metrics(
                dense_outputs[f"layer.{index}.hidden"],
                live_outputs[f"layer.{index}.hidden"],
                normalize_distribution=True,
            )
            for index in range(args.layers)
        }
        result = {
            "bits": bits,
            "sym": sym,
            "adjacent_zero_search": adjacent,
            "seconds": time.perf_counter() - started,
            "modules": module_results,
            "layers": layer_results,
            "logits": tensor_metrics(dense_logits, quant_logits, normalize_distribution=False),
        }
        report["arms"][arm_name] = result
        print_arm_summary(arm_name, result)

        for module_name, module in modules.items():
            with torch.no_grad():
                module.weight.copy_(original_weights[module_name])

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
