"""Run the matched two-layer QVQ low-rate output-scale gate.

The baseline, full closed-form correction, and shrunk correction share one
fixed trellis per module. This isolates the output-scale decision from Viterbi
variation and avoids repeating the dominant quantization work.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import time
import zlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


if __package__:
    from scripts.analyze_gptq_low_bit_grid import (
        TARGET_SUFFIXES,
        _summary,
        capture_calibration_hessians,
        capture_forward,
        decoder_layers,
        load_nm_calibration_batches,
        load_nm_evaluation_batch,
        mean_metric,
        request_performance_qos,
        target_modules,
        tensor_metrics,
    )
else:
    from analyze_gptq_low_bit_grid import (
        TARGET_SUFFIXES,
        _summary,
        capture_calibration_hessians,
        capture_forward,
        decoder_layers,
        load_nm_calibration_batches,
        load_nm_evaluation_batch,
        mean_metric,
        request_performance_qos,
        target_modules,
        tensor_metrics,
    )
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from gptqmodel.quantization.qvq import (
    default_qvq_trellis_batch_size,
    optimize_qvq_output_channel_scales,
    quantize_qvq_linear,
    rht_preprocess_hessian,
)
from gptqmodel.quantization.qvq_codecs import PGC16_CODEBOOK_VERSION
from gptqmodel.quantization.qvq_rates import normalize_qvq_rate


ARMS = ("baseline", "alpha-full", "alpha-half")


def _prompt_hash(encoded: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(encoded):
        tensor = encoded[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _distribution_metrics_chunked(
    dense: torch.Tensor,
    quantized: torch.Tensor,
    *,
    chunk_rows: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Compute raw-logit metrics without materializing several vocabulary-sized copies."""

    if dense.shape != quantized.shape or dense.ndim != 2:
        raise ValueError("chunked logit metrics require matching rank-2 tensors")
    if chunk_rows < 1:
        raise ValueError("chunked logit metric row count must be positive")

    scalar_sums = {
        "dense": 0.0,
        "quantized": 0.0,
        "dense_square": 0.0,
        "quantized_square": 0.0,
        "product": 0.0,
        "error": 0.0,
        "error_square": 0.0,
        "abs_error": 0.0,
        "sign_agreement": 0.0,
    }
    max_abs_error = 0.0
    rows: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "kl_forward",
            "kl_reverse",
            "jensen_shannon",
            "total_variation",
            "hellinger",
            "dense_entropy",
            "cross_entropy",
            "row_cosine",
            "row_rmse",
            "top1",
            "top5_overlap",
            "top5_exact",
            "dense_top1_in_quantized_top5",
            "quantized_top1_in_dense_top5",
        )
    }

    for start in range(0, dense.shape[0], chunk_rows):
        dense_chunk = dense[start : start + chunk_rows].float()
        quantized_chunk = quantized[start : start + chunk_rows].float()
        error = quantized_chunk - dense_chunk
        dense64 = dense_chunk.double()
        quantized64 = quantized_chunk.double()
        error64 = error.double()
        scalar_sums["dense"] += dense64.sum().item()
        scalar_sums["quantized"] += quantized64.sum().item()
        scalar_sums["dense_square"] += dense64.square().sum().item()
        scalar_sums["quantized_square"] += quantized64.square().sum().item()
        scalar_sums["product"] += (dense64 * quantized64).sum().item()
        scalar_sums["error"] += error64.sum().item()
        scalar_sums["error_square"] += error64.square().sum().item()
        scalar_sums["abs_error"] += error64.abs().sum().item()
        scalar_sums["sign_agreement"] += ((dense_chunk >= 0) == (quantized_chunk >= 0)).sum().item()
        max_abs_error = max(max_abs_error, error.abs().max().item())

        dense_log_prob = F.log_softmax(dense_chunk, dim=-1)
        quantized_log_prob = F.log_softmax(quantized_chunk, dim=-1)
        dense_prob = dense_log_prob.exp()
        quantized_prob = quantized_log_prob.exp()
        midpoint = (dense_prob + quantized_prob) * 0.5
        midpoint_log = midpoint.clamp_min(1e-30).log()
        rows["kl_forward"].append((dense_prob * (dense_log_prob - quantized_log_prob)).sum(dim=-1).cpu())
        rows["kl_reverse"].append(
            (quantized_prob * (quantized_log_prob - dense_log_prob)).sum(dim=-1).cpu()
        )
        rows["jensen_shannon"].append(
            (
                (dense_prob * (dense_log_prob - midpoint_log)).sum(dim=-1)
                + (quantized_prob * (quantized_log_prob - midpoint_log)).sum(dim=-1)
            ).mul(0.5).cpu()
        )
        rows["total_variation"].append((dense_prob - quantized_prob).abs().sum(dim=-1).mul(0.5).cpu())
        rows["hellinger"].append(
            (dense_prob.sqrt() - quantized_prob.sqrt()).square().sum(dim=-1).mul(0.5).sqrt().cpu()
        )
        rows["dense_entropy"].append(-(dense_prob * dense_log_prob).sum(dim=-1).cpu())
        rows["cross_entropy"].append(-(dense_prob * quantized_log_prob).sum(dim=-1).cpu())
        rows["row_cosine"].append(F.cosine_similarity(dense_chunk, quantized_chunk, dim=-1).cpu())
        rows["row_rmse"].append(error.square().mean(dim=-1).sqrt().cpu())

        dense_top5 = dense_chunk.topk(5, dim=-1).indices
        quantized_top5 = quantized_chunk.topk(5, dim=-1).indices
        rows["top1"].append((dense_top5[:, 0] == quantized_top5[:, 0]).float().cpu())
        rows["top5_overlap"].append(
            (dense_top5.unsqueeze(-1) == quantized_top5.unsqueeze(-2)).any(dim=-1).float().mean(dim=-1).cpu()
        )
        rows["top5_exact"].append(
            (dense_top5.sort(dim=-1).values == quantized_top5.sort(dim=-1).values).all(dim=-1).float().cpu()
        )
        rows["dense_top1_in_quantized_top5"].append(
            (dense_top5[:, :1] == quantized_top5).any(dim=-1).float().cpu()
        )
        rows["quantized_top1_in_dense_top5"].append(
            (quantized_top5[:, :1] == dense_top5).any(dim=-1).float().cpu()
        )

    row_values = {name: torch.cat(values).float() for name, values in rows.items()}
    element_count = dense.numel()
    dense_energy = max(scalar_sums["dense_square"], torch.finfo(torch.float64).eps)
    error_energy = max(scalar_sums["error_square"], torch.finfo(torch.float64).eps)
    dense_norm = math.sqrt(dense_energy)
    quantized_norm = math.sqrt(max(scalar_sums["quantized_square"], torch.finfo(torch.float64).eps))
    dense_mean = scalar_sums["dense"] / element_count
    quantized_mean = scalar_sums["quantized"] / element_count
    covariance = scalar_sums["product"] - element_count * dense_mean * quantized_mean
    dense_centered = max(scalar_sums["dense_square"] - element_count * dense_mean**2, 0.0)
    quantized_centered = max(scalar_sums["quantized_square"] - element_count * quantized_mean**2, 0.0)
    pearson_denominator = math.sqrt(dense_centered * quantized_centered)
    error_mean = scalar_sums["error"] / element_count
    error_variance = max(scalar_sums["error_square"] / element_count - error_mean**2, 0.0)

    metrics = {
        "shape": list(dense.shape),
        "finite": bool(torch.isfinite(quantized).all()),
        "mae": scalar_sums["abs_error"] / element_count,
        "rmse": math.sqrt(scalar_sums["error_square"] / element_count),
        "relative_l2": math.sqrt(error_energy / dense_energy),
        "sqnr_db": 10.0 * math.log10(dense_energy / error_energy),
        "max_abs_error": max_abs_error,
        "bias": error_mean,
        "error_std": math.sqrt(error_variance),
        "cosine": scalar_sums["product"] / (dense_norm * quantized_norm),
        "pearson": covariance / pearson_denominator if pearson_denominator else 0.0,
        "row_cosine": _summary(row_values["row_cosine"]),
        "row_rmse": _summary(row_values["row_rmse"]),
        "norm_ratio": quantized_norm / dense_norm,
        "sign_agreement": scalar_sums["sign_agreement"] / element_count,
        "kl_forward": _summary(row_values["kl_forward"]),
        "kl_reverse": _summary(row_values["kl_reverse"]),
        "jensen_shannon": _summary(row_values["jensen_shannon"]),
        "total_variation": _summary(row_values["total_variation"]),
        "hellinger": _summary(row_values["hellinger"]),
        "dense_entropy": _summary(row_values["dense_entropy"]),
        "dense_to_quantized_cross_entropy": _summary(row_values["cross_entropy"]),
        "top1_agreement": row_values["top1"].mean().item(),
        "top5_overlap": _summary(row_values["top5_overlap"]),
        "top5_exact_agreement": row_values["top5_exact"].mean().item(),
        "dense_top1_in_quantized_top5": row_values["dense_top1_in_quantized_top5"].mean().item(),
        "quantized_top1_in_dense_top5": row_values["quantized_top1_in_dense_top5"].mean().item(),
    }
    return metrics, row_values


def _cluster_bootstrap_delta(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    prompt_ids: torch.Tensor,
    *,
    seed: int,
    samples: int,
) -> dict[str, float | int]:
    """Bootstrap paired token-mean deltas by resampling whole prompts."""

    if baseline.shape != candidate.shape or baseline.shape != prompt_ids.shape:
        raise ValueError("paired bootstrap values and prompt IDs must have matching shapes")
    if baseline.ndim != 1:
        raise ValueError("paired bootstrap values must be rank-1 token metrics")
    if not baseline.is_floating_point() or not candidate.is_floating_point():
        raise TypeError("paired bootstrap values must use floating-point dtypes")
    if prompt_ids.dtype == torch.bool or prompt_ids.is_floating_point() or prompt_ids.is_complex():
        raise TypeError("paired bootstrap prompt IDs must use an integer dtype")
    if baseline.numel() == 0 or not torch.isfinite(baseline).all() or not torch.isfinite(candidate).all():
        raise ValueError("paired bootstrap values must be nonempty and finite")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("paired bootstrap seed must be an integer")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("paired bootstrap sample count must be a positive integer")
    if torch.any(prompt_ids < 0):
        raise ValueError("paired bootstrap prompt IDs must be nonnegative")
    prompt_count = int(prompt_ids.max().item()) + 1
    observed_prompts = torch.unique(prompt_ids)
    if observed_prompts.numel() != prompt_count:
        raise ValueError("paired bootstrap prompt IDs must form a contiguous zero-based range")
    delta = candidate.double() - baseline.double()
    cluster_sum = torch.zeros(prompt_count, dtype=torch.float64).scatter_add_(0, prompt_ids, delta)
    cluster_count = torch.zeros(prompt_count, dtype=torch.float64).scatter_add_(
        0, prompt_ids, torch.ones_like(delta)
    )
    generator = torch.Generator().manual_seed(seed)
    bootstrap = []
    batch = 512
    for start in range(0, samples, batch):
        draw_count = min(batch, samples - start)
        indices = torch.randint(0, prompt_count, (draw_count, prompt_count), generator=generator)
        sampled_sum = cluster_sum[indices].sum(dim=1)
        sampled_count = cluster_count[indices].sum(dim=1)
        bootstrap.append(sampled_sum / sampled_count)
    distribution = torch.cat(bootstrap)
    lower, upper = torch.quantile(distribution, torch.tensor([0.025, 0.975], dtype=torch.float64))
    prompt_mean = cluster_sum / cluster_count
    return {
        "delta": delta.mean().item(),
        "ci95_low": lower.item(),
        "ci95_high": upper.item(),
        "prompt_wins": int((prompt_mean < 0).sum().item()),
        "prompt_ties": int((prompt_mean == 0).sum().item()),
        "prompt_count": prompt_count,
        "bootstrap_samples": samples,
    }


def _scale_variants(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    bits: float,
    module_name: str,
    device: torch.device,
    trellis_batch_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, float | int]], dict[str, int | float]]:
    seed = zlib.crc32(module_name.encode("utf-8")) & 0x7FFFFFFF
    result = quantize_qvq_linear(
        weight.to(device),
        hessian.to(device),
        bits=bits,
        seed=seed,
        trellis_batch_size=trellis_batch_size,
        codebook_version=PGC16_CODEBOOK_VERSION,
    )
    source_hessian = hessian.to(device=device, dtype=torch.float32)
    transformed_hessian = rht_preprocess_hessian(source_hessian, result.SU)
    transformed_hessian = (transformed_hessian + transformed_hessian.mT) * 0.5
    damping = torch.maximum(
        transformed_hessian.diagonal().abs().mean() * 0.01,
        torch.tensor(torch.finfo(torch.float32).eps, device=device),
    )
    optimization_hessian = source_hessian.clone()
    optimization_hessian.diagonal().add_(damping)

    variants = {"baseline": result.weight.detach().cpu()}
    diagnostics: dict[str, dict[str, float | int]] = {
        "baseline": {
            "proxy_loss": result.proxy_loss.item(),
            "optimized_channels": 0,
            "correction_strength": 0.0,
        }
    }
    for arm, strength in (("alpha-full", 1.0), ("alpha-half", 0.5)):
        _, reconstructed, loss, optimized_channels = optimize_qvq_output_channel_scales(
            weight.to(device),
            result.inner_weight,
            source_hessian,
            result.SU,
            result.SV,
            optimization_H=optimization_hessian,
            correction_strength=strength,
        )
        variants[arm] = reconstructed.detach().cpu()
        diagnostics[arm] = {
            "proxy_loss": loss.item(),
            "optimized_channels": optimized_channels,
            "correction_strength": strength,
        }

    payload_bytes = result.trellis.numel() * result.trellis.element_size()
    auxiliary_bytes = sum(tensor.numel() * tensor.element_size() for tensor in (result.SU, result.SV))
    storage = {
        "weight_numel": weight.numel(),
        "payload_bytes": payload_bytes,
        "auxiliary_bytes": auxiliary_bytes,
        "stored_bytes": payload_bytes + auxiliary_bytes,
        "payload_bits_per_weight": payload_bytes * 8 / weight.numel(),
        "effective_bits_per_weight": (payload_bytes + auxiliary_bytes) * 8 / weight.numel(),
        "trellis_sha256": hashlib.sha256(result.trellis.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
    }
    del result, source_hessian, transformed_hessian, optimization_hessian
    if device.type == "mps":
        torch.mps.empty_cache()
    return variants, diagnostics, storage


def _summary_row(
    rate: str,
    arm: str,
    result: dict[str, Any],
    paired: dict[str, dict[str, dict[str, float | int]]],
) -> dict[str, Any]:
    logits = result["logits"]
    row = {
        "rate": rate,
        "arm": arm,
        "mean_weight_relative_l2": mean_metric(result["modules"], ("weight", "relative_l2")),
        "mean_local_kl": mean_metric(result["modules"], ("local", "kl_forward", "mean")),
        "mean_local_relative_l2": mean_metric(result["modules"], ("local", "relative_l2")),
        "mean_local_rmse": mean_metric(result["modules"], ("local", "rmse")),
        "mean_live_kl": mean_metric(result["modules"], ("live", "kl_forward", "mean")),
        "mean_live_relative_l2": mean_metric(result["modules"], ("live", "relative_l2")),
        "mean_live_rmse": mean_metric(result["modules"], ("live", "rmse")),
        "mean_layer_kl": mean_metric(result["layers"], ("kl_forward", "mean")),
        "mean_layer_relative_l2": mean_metric(result["layers"], ("relative_l2",)),
        "mean_layer_rmse": mean_metric(result["layers"], ("rmse",)),
        "final_logit_kl_mean": logits["kl_forward"]["mean"],
        "final_logit_kl_p95": logits["kl_forward"]["p95"],
        "final_logit_js_mean": logits["jensen_shannon"]["mean"],
        "final_logit_relative_l2": logits["relative_l2"],
        "final_logit_sqnr_db": logits["sqnr_db"],
        "final_logit_top1_agreement": logits["top1_agreement"],
        "final_logit_top5_overlap": logits["top5_overlap"]["mean"],
        "optimized_channels": sum(
            int(module["scale_diagnostics"]["optimized_channels"]) for module in result["modules"].values()
        ),
    }
    comparison = paired.get(arm)
    for metric in ("kl_forward", "jensen_shannon", "top1", "top5_overlap"):
        values = None if comparison is None else comparison[metric]
        row[f"{metric}_delta_vs_baseline"] = None if values is None else values["delta"]
        row[f"{metric}_ci95_low"] = None if values is None else values["ci95_low"]
        row[f"{metric}_ci95_high"] = None if values is None else values["ci95_high"]
        row[f"{metric}_prompt_wins"] = None if values is None else values["prompt_wins"]
    return row


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration-dataset", type=Path, required=True)
    parser.add_argument("--bits", type=float, nargs="+", default=[1, 1.5])
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="mps")
    parser.add_argument("--calibration-rows", type=int, default=128)
    parser.add_argument("--evaluation-rows", type=int, default=128)
    parser.add_argument("--evaluation-row-offset", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--metric-chunk-rows", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    return parser


def _device_unavailable_reason(device: str) -> str | None:
    if device == "mps" and not torch.backends.mps.is_available():
        return "MPS is unavailable"
    if device == "cuda" and not torch.cuda.is_available():
        return "CUDA is unavailable"
    return None


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    rates = [normalize_qvq_rate(rate) for rate in args.bits]
    if args.layers != 2:
        parser.error("this acceptance gate requires exactly two decoder layers")
    if any(rate not in (1, 1.5) for rate in rates):
        parser.error("this focused acceptance gate supports W1 and W1.5")
    if args.evaluation_row_offset < args.calibration_rows:
        parser.error("held-out evaluation rows must not overlap calibration rows")
    unavailable_reason = _device_unavailable_reason(args.device)
    if unavailable_reason is not None:
        parser.error(unavailable_reason)
    if platform.system() == "Darwin" and not request_performance_qos():
        raise RuntimeError("failed to request performance-core QoS")

    os.environ.setdefault("GPTQMODEL_SCALE_SEARCH_CPU", "1")
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    source_layers = int(config.num_hidden_layers)
    config.num_hidden_layers = args.layers
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    calibration_batches, calibration_stats = load_nm_calibration_batches(
        tokenizer,
        config,
        dataset_path=args.calibration_dataset,
        rows=args.calibration_rows,
        concat_size=2048,
        batch_size=1,
    )
    evaluation, evaluation_stats = load_nm_evaluation_batch(
        tokenizer,
        dataset_path=args.calibration_dataset,
        row_offset=args.evaluation_row_offset,
        rows=args.evaluation_rows,
        max_length=args.max_length,
    )
    evaluation_prompt_hash = _prompt_hash(evaluation)
    attention_mask = evaluation["attention_mask"]
    prompt_ids = torch.arange(attention_mask.shape[0]).unsqueeze(1).expand_as(attention_mask)[attention_mask.bool()]
    evaluation_device = {name: value.to(device) for name, value in evaluation.items()}
    modules = target_modules(model, layer_count=args.layers)
    expected_modules = args.layers * len(TARGET_SUFFIXES)
    if len(modules) != expected_modules:
        raise ValueError(f"expected {expected_modules} target modules, found {len(modules)}")
    if len(decoder_layers(model)) != args.layers:
        raise ValueError("the gate model was not truncated to exactly two layers")

    print("Capturing calibration Hessians", flush=True)
    hessians, sample_counts = capture_calibration_hessians(
        model,
        calibration_batches,
        modules,
        device=device,
    )
    if set(sample_counts.values()) != {calibration_stats["valid_tokens"]}:
        raise AssertionError("calibration Hessian sample counts do not match valid tokens")
    print("Capturing dense held-out reference", flush=True)
    dense_logits, evaluation_inputs, dense_outputs = capture_forward(
        model,
        evaluation_device,
        modules,
        capture_inputs=True,
        layer_count=args.layers,
    )
    original_weights = {name: module.weight.detach().cpu().clone() for name, module in modules.items()}

    report: dict[str, Any] = {
        "settings": {
            "model": str(args.model),
            "model_revision": args.model.name,
            "source_model_layers": source_layers,
            "target_model_layers": args.layers,
            "target_modules": len(modules),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "compute_capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
            "model_dtype": str(next(model.parameters()).dtype),
            "threads": args.threads,
            "host": platform.platform(),
            "torch": torch.__version__,
            "calibration": calibration_stats,
            "evaluation": evaluation_stats,
            "evaluation_prompt_hash": evaluation_prompt_hash,
            "codebook": PGC16_CODEBOOK_VERSION,
            "trellis_reuse": "one fixed baseline trellis shared by baseline, alpha-full, and alpha-half",
            "bootstrap_unit": "held-out prompt",
            "bootstrap_samples": args.bootstrap_samples,
        },
        "rates": {},
    }

    for bits in rates:
        rate_name = f"w{bits}"
        started = time.perf_counter()
        trellis_batch_size = default_qvq_trellis_batch_size(bits, device)
        reconstructions = {arm: {} for arm in ARMS}
        module_results = {arm: {} for arm in ARMS}
        print(f"{rate_name}: quantizing {len(modules)} modules with trellis batch {trellis_batch_size}", flush=True)
        for module_index, (module_name, module) in enumerate(modules.items(), start=1):
            module_started = time.perf_counter()
            print(f"{rate_name}: module {module_index}/{len(modules)} {module_name}", flush=True)
            variants, scale_diagnostics, storage = _scale_variants(
                original_weights[module_name],
                hessians[module_name],
                bits=bits,
                module_name=module_name,
                device=device,
                trellis_batch_size=trellis_batch_size,
            )
            bias = None if module.bias is None else module.bias.detach().cpu()
            for arm in ARMS:
                reconstruction = variants[arm]
                reconstructions[arm][module_name] = reconstruction
                local_output = F.linear(evaluation_inputs[module_name], reconstruction, bias)
                module_results[arm][module_name] = {
                    "weight": tensor_metrics(
                        original_weights[module_name].float(), reconstruction.float(), normalize_distribution=True
                    ),
                    "local": tensor_metrics(
                        dense_outputs[module_name], local_output, normalize_distribution=True
                    ),
                    "scale_diagnostics": scale_diagnostics[arm],
                    **storage,
                }
                del local_output
            print(
                f"{rate_name}: module {module_index}/{len(modules)} done in {time.perf_counter() - module_started:.2f}s "
                f"channels full={scale_diagnostics['alpha-full']['optimized_channels']} "
                f"half={scale_diagnostics['alpha-half']['optimized_channels']}",
                flush=True,
            )

        arm_rows: dict[str, dict[str, torch.Tensor]] = {}
        rate_report: dict[str, Any] = {"trellis_batch_size": trellis_batch_size, "arms": {}, "paired": {}}
        for arm in ARMS:
            print(f"{rate_name}: replaying {arm}", flush=True)
            for module_name, module in modules.items():
                with torch.no_grad():
                    module.weight.copy_(reconstructions[arm][module_name].to(device))
            quantized_logits, _, live_outputs = capture_forward(
                model,
                evaluation_device,
                modules,
                capture_inputs=False,
                layer_count=args.layers,
            )
            for module_name in modules:
                module_results[arm][module_name]["live"] = tensor_metrics(
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
            logit_metrics, arm_rows[arm] = _distribution_metrics_chunked(
                dense_logits,
                quantized_logits,
                chunk_rows=args.metric_chunk_rows,
            )
            rate_report["arms"][arm] = {
                "modules": module_results[arm],
                "layers": layer_results,
                "logits": logit_metrics,
            }
            print(
                f"{rate_name} {arm}: KLD={logit_metrics['kl_forward']['mean']:.6f} "
                f"JSD={logit_metrics['jensen_shannon']['mean']:.6f} "
                f"top1={logit_metrics['top1_agreement']:.4f} "
                f"top5={logit_metrics['top5_overlap']['mean']:.4f}",
                flush=True,
            )
            del quantized_logits, live_outputs
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()

        for arm in ("alpha-full", "alpha-half"):
            comparisons = {}
            for metric, higher_is_better in (
                ("kl_forward", False),
                ("jensen_shannon", False),
                ("top1", True),
                ("top5_overlap", True),
            ):
                comparison = _cluster_bootstrap_delta(
                    arm_rows["baseline"][metric],
                    arm_rows[arm][metric],
                    prompt_ids,
                    seed=20260812 + int(bits * 10) + len(metric) + len(arm),
                    samples=args.bootstrap_samples,
                )
                if higher_is_better:
                    comparison["prompt_wins"] = (
                        comparison["prompt_count"] - comparison["prompt_wins"] - comparison["prompt_ties"]
                    )
                comparisons[metric] = comparison
            rate_report["paired"][arm] = comparisons

        rate_report["seconds"] = time.perf_counter() - started
        report["rates"][rate_name] = rate_report
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{rate_name}: wrote incremental report to {args.json_out}", flush=True)
        del reconstructions, module_results, arm_rows
        gc.collect()

    rows = [
        _summary_row(rate, arm, result, rate_result["paired"])
        for rate, rate_result in report["rates"].items()
        for arm, result in rate_result["arms"].items()
    ]
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.csv_out}", flush=True)


if __name__ == "__main__":
    main()
