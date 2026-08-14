#!/usr/bin/env python3
"""Large paired W1/W2 QVQ module-scale search accuracy benchmark."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from gptqmodel.quantization.qvq import quantize_qvq_linear


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=float, nargs="+", default=[1, 2])
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--calibration-samples", type=int, default=1024)
    parser.add_argument("--heldout-samples", type=int, default=4096)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=2026081200)
    parser.add_argument("--trellis-batch-size", type=int, default=16)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if any(bits not in (1, 2) for bits in args.bits):
        parser.error("--bits currently accepts only W1 and W2")
    if args.features < 16 or args.features % 16:
        parser.error("--features must be a positive multiple of 16")
    if args.calibration_samples < 2 or args.heldout_samples < 2:
        parser.error("sample counts must be at least two")
    if args.seeds < 2:
        parser.error("--seeds must be at least two for confidence intervals")
    if args.trellis_batch_size < 1:
        parser.error("--trellis-batch-size must be positive")
    return args


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _mean_ci95(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    margin = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - margin, mean + margin


def _summary(values: list[float]) -> dict[str, float]:
    mean, ci_low, ci_high = _mean_ci95(values)
    return {
        "mean": mean,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "median": statistics.median(values),
        "p05": _quantile(values, 0.05),
        "p95": _quantile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def _distribution_summary(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().double().flatten().cpu()
    return {
        "mean": values.mean().item(),
        "p50": torch.quantile(values, 0.50).item(),
        "p95": torch.quantile(values, 0.95).item(),
        "p99": torch.quantile(values, 0.99).item(),
        "max": values.max().item(),
    }


def _make_case(
    *,
    features: int,
    calibration_samples: int,
    heldout_samples: int,
    seed: int,
):
    generator = torch.Generator().manual_seed(seed)
    row_scale = (
        torch.exp(torch.randn((features, 1), generator=generator) * 0.55) * 0.075
    )
    weight = torch.randn((features, features), generator=generator) * row_scale
    outlier_columns = torch.arange(0, features, 17)
    weight[:, outlier_columns] *= 2.5

    activation_scale = torch.exp(torch.randn((features,), generator=generator) * 0.60)

    def sample(count: int) -> torch.Tensor:
        values = torch.randn((count, features), generator=generator) * activation_scale
        original = values.clone()
        values[:, 1:] += original[:, :-1] * 0.30
        values[:, 2:] -= original[:, :-2] * 0.10
        return values

    calibration = sample(calibration_samples)
    heldout = sample(heldout_samples)
    hessian = calibration.T @ calibration / calibration.shape[0]
    return weight, hessian, heldout


def _output_metrics(
    dense: torch.Tensor,
    candidate: torch.Tensor,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    dense = dense.float()
    candidate = candidate.float()
    dense_log_prob = F.log_softmax(dense, dim=-1)
    candidate_log_prob = F.log_softmax(candidate, dim=-1)
    dense_prob = dense_log_prob.exp()
    candidate_prob = candidate_log_prob.exp()
    midpoint = (dense_prob + candidate_prob) * 0.5
    midpoint_log = midpoint.clamp_min(1e-30).log()
    kld = (dense_prob * (dense_log_prob - candidate_log_prob)).sum(dim=-1)
    jsd = 0.5 * (
        (dense_prob * (dense_log_prob - midpoint_log)).sum(dim=-1)
        + (candidate_prob * (candidate_log_prob - midpoint_log)).sum(dim=-1)
    )
    dense_top5 = dense.topk(5, dim=-1).indices
    candidate_top5 = candidate.topk(5, dim=-1).indices
    top1 = (dense_top5[:, 0] == candidate_top5[:, 0]).float()
    top5 = (
        (dense_top5.unsqueeze(-1) == candidate_top5.unsqueeze(-2))
        .any(dim=-1)
        .float()
        .mean(dim=-1)
    )
    error = candidate - dense
    relative_l2 = error.double().norm() / dense.double().norm().clamp_min(
        torch.finfo(torch.float64).eps
    )
    return (
        {
            "kld": kld.mean().item(),
            "jsd": jsd.mean().item(),
            "rmse": error.square().mean().sqrt().item(),
            "relative_l2": relative_l2.item(),
            "top1": top1.mean().item(),
            "top5": top5.mean().item(),
        },
        {"kld": kld, "jsd": jsd, "top1": top1, "top5": top5},
    )


def _paired_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    directions = {
        "proxy_ratio": "lower",
        "weight_relative_l2_delta": "lower",
        "kld_delta": "lower",
        "jsd_delta": "lower",
        "rmse_delta": "lower",
        "relative_l2_delta": "lower",
        "top1_delta": "higher",
        "top5_delta": "higher",
        "quantization_seconds_ratio": "lower",
    }
    summary: dict[str, object] = {}
    for metric, direction in directions.items():
        values = [float(row[metric]) for row in rows]
        if metric.endswith("_ratio"):
            improved = sum(value < 1.0 for value in values)
            regressed = sum(value > 1.0 for value in values)
        elif direction == "lower":
            improved = sum(value < 0.0 for value in values)
            regressed = sum(value > 0.0 for value in values)
        else:
            improved = sum(value > 0.0 for value in values)
            regressed = sum(value < 0.0 for value in values)
        summary[metric] = {
            **_summary(values),
            "direction": direction,
            "improved_seeds": improved,
            "equal_seeds": len(values) - improved - regressed,
            "regressed_seeds": regressed,
        }
    return summary


def _print_table(summaries: dict[float, dict[str, object]]) -> None:
    columns = (
        "W",
        "Metric",
        "Mean paired delta",
        "95% CI",
        "Median",
        "P95",
        "Wins",
        "Losses",
    )
    rows = []
    selected_metrics = (
        "proxy_ratio",
        "kld_delta",
        "jsd_delta",
        "rmse_delta",
        "top1_delta",
        "top5_delta",
    )
    for bits, report in summaries.items():
        paired = report["paired"]
        for metric in selected_metrics:
            values = paired[metric]
            rows.append(
                (
                    f"{bits:g}",
                    metric,
                    f"{values['mean']:+.6f}",
                    f"[{values['ci95_low']:+.6f}, {values['ci95_high']:+.6f}]",
                    f"{values['median']:+.6f}",
                    f"{values['p95']:+.6f}",
                    str(values["improved_seeds"]),
                    str(values["regressed_seeds"]),
                )
            )
    widths = [
        max(len(columns[index]), *(len(row[index]) for row in rows))
        for index in range(len(columns))
    ]
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    print(separator)
    print(
        "| "
        + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(columns)
        )
        + " |"
    )
    print(separator)
    for row in rows:
        print(
            "| "
            + " | ".join(
                value.ljust(widths[index]) for index, value in enumerate(row)
            )
            + " |"
        )
    print(separator)


def main() -> None:
    args = _parse_args()
    all_rows: list[dict[str, object]] = []
    distributions: dict[float, dict[str, list[torch.Tensor]]] = {
        bits: {
            f"{arm}_{metric}": []
            for arm in ("baseline", "searched")
            for metric in ("kld", "jsd", "top1", "top5")
        }
        for bits in sorted(set(args.bits))
    }
    for bits in sorted(set(args.bits)):
        for seed_index in range(args.seeds):
            seed = args.seed_base + int(bits * 1000) + seed_index
            weight, hessian, heldout = _make_case(
                features=args.features,
                calibration_samples=args.calibration_samples,
                heldout_samples=args.heldout_samples,
                seed=seed,
            )
            started = time.perf_counter()
            baseline = quantize_qvq_linear(
                weight,
                hessian,
                bits=bits,
                seed=seed,
                trellis_batch_size=args.trellis_batch_size,
            )
            baseline_seconds = time.perf_counter() - started
            started = time.perf_counter()
            searched = quantize_qvq_linear(
                weight,
                hessian,
                bits=bits,
                seed=seed,
                trellis_batch_size=args.trellis_batch_size,
                module_scale_search=True,
            )
            searched_seconds = time.perf_counter() - started

            dense_output = heldout @ weight.T
            baseline_metrics, baseline_distributions = _output_metrics(
                dense_output,
                heldout @ baseline.weight.T,
            )
            searched_metrics, searched_distributions = _output_metrics(
                dense_output,
                heldout @ searched.weight.T,
            )
            for metric in baseline_distributions:
                distributions[bits][f"baseline_{metric}"].append(baseline_distributions[metric])
                distributions[bits][f"searched_{metric}"].append(searched_distributions[metric])

            weight_norm = weight.double().norm().clamp_min(
                torch.finfo(torch.float64).eps
            )
            baseline_weight_relative_l2 = (
                (baseline.weight - weight).double().norm() / weight_norm
            ).item()
            searched_weight_relative_l2 = (
                (searched.weight - weight).double().norm() / weight_norm
            ).item()
            row = {
                "bits": bits,
                "seed": seed,
                "selected": searched.module_scale_search_selected,
                "reencoded": searched.module_scale_reencoded,
                "multiplier": searched.module_scale_multiplier,
                "proxy_ratio": (searched.proxy_loss / baseline.proxy_loss).item(),
                "weight_relative_l2_delta": (
                    searched_weight_relative_l2 - baseline_weight_relative_l2
                ),
                "kld_delta": searched_metrics["kld"] - baseline_metrics["kld"],
                "jsd_delta": searched_metrics["jsd"] - baseline_metrics["jsd"],
                "rmse_delta": searched_metrics["rmse"] - baseline_metrics["rmse"],
                "relative_l2_delta": (
                    searched_metrics["relative_l2"] - baseline_metrics["relative_l2"]
                ),
                "top1_delta": searched_metrics["top1"] - baseline_metrics["top1"],
                "top5_delta": searched_metrics["top5"] - baseline_metrics["top5"],
                "quantization_seconds_ratio": searched_seconds / baseline_seconds,
                "baseline": {
                    **baseline_metrics,
                    "weight_relative_l2": baseline_weight_relative_l2,
                    "proxy_loss": baseline.proxy_loss.item(),
                    "quantization_seconds": baseline_seconds,
                },
                "searched": {
                    **searched_metrics,
                    "weight_relative_l2": searched_weight_relative_l2,
                    "proxy_loss": searched.proxy_loss.item(),
                    "quantization_seconds": searched_seconds,
                },
            }
            all_rows.append(row)
            print(
                f"W{bits:g} seed {seed_index + 1:02d}/{args.seeds}: "
                f"KLD {baseline_metrics['kld']:.6f}->{searched_metrics['kld']:.6f}, "
                f"top1 {baseline_metrics['top1']:.4f}->{searched_metrics['top1']:.4f}, "
                f"top5 {baseline_metrics['top5']:.4f}->{searched_metrics['top5']:.4f}",
                flush=True,
            )

    summaries: dict[float, dict[str, object]] = {}
    for bits in sorted(set(args.bits)):
        rows = [row for row in all_rows if row["bits"] == bits]
        summaries[bits] = {
            "paired": _paired_summary(rows),
            "selection_count": sum(bool(row["selected"]) for row in rows),
            "reencode_count": sum(bool(row["reencoded"]) for row in rows),
            "token_distributions": {
                name: _distribution_summary(torch.cat(values))
                for name, values in distributions[bits].items()
            },
        }

    report = {
        "environment": {
            "python": __import__("sys").version.split()[0],
            "torch": torch.__version__,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "settings": {
            "bits": sorted(set(args.bits)),
            "features": args.features,
            "calibration_samples_per_seed": args.calibration_samples,
            "heldout_samples_per_seed": args.heldout_samples,
            "seeds": args.seeds,
            "evaluated_logits_per_arm_per_rate": (
                args.seeds * args.heldout_samples * args.features
            ),
            "trellis_batch_size": args.trellis_batch_size,
            "calibration_and_heldout_are_independent": True,
        },
        "summaries": {str(bits): value for bits, value in summaries.items()},
        "rows": all_rows,
    }
    _print_table(summaries)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
