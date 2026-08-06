# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Compare accuracy-oriented GPTQ defaults on disjoint held-out activations.

This micro-benchmark complements model task scores with direct dense-reference
output-error telemetry. Calibration and held-out samples use independent RNG
streams. Run it on an isolated GPU for release/default decisions, for example:

    CUDA_VISIBLE_DEVICES=GPU-... python scripts/validate_gptq_default_accuracy.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from gptqmodel.looper.named_module import NamedModule  # noqa: E402
from gptqmodel.quantization import QuantizeConfig, ScaleSearchConfig  # noqa: E402
from gptqmodel.quantization.diagnostics import analyze_output_error  # noqa: E402
from gptqmodel.quantization.gptq import GPTQ  # noqa: E402


ADAPTIVE_DAMPING_V42 = {
    "enabled": True,
    "base_percdamp": 0.05,
    "min": 0.02,
    "max": 0.08,
    "method": "power_iteration",
    "eigen_iterations": 10,
    "spectral_alpha": 0.25,
    "module_prior_enabled": True,
    "group_error_enabled": True,
    "online_feedback_enabled": True,
    "group_size_prior_enabled": True,
    "group_error_use_hessian_weighting": True,
    "group_error_measure_raw_residual": True,
}
GPTQ_ERROR_CLIPPING = {
    "enabled": True,
    "metric": "gptq_error",
    "per_group": True,
}


@dataclass(frozen=True)
class Variant:
    label: str
    adaptive_damping: dict | None
    adaptive_clipping: dict | None
    scale_search: ScaleSearchConfig | None


VARIANTS = (
    Variant("A fixed-5% + activation", None, None, ScaleSearchConfig.ACTIVATION),
    Variant("B fixed-5% + GPTQ clip", None, GPTQ_ERROR_CLIPPING, None),
    Variant("C adaptive-v4.2 + activation", ADAPTIVE_DAMPING_V42, None, ScaleSearchConfig.ACTIVATION),
    Variant("D adaptive-v4.2 + GPTQ clip", ADAPTIVE_DAMPING_V42, GPTQ_ERROR_CLIPPING, None),
)
MODULE_ROLES = ("self_attn.q_proj", "mlp.gate_proj", "mlp.down_proj")
SCENARIOS = ("gaussian", "weight_outlier", "activation_outlier", "correlated", "rank_deficient", "ill_conditioned")


def _generator(seed: int, device: torch.device) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def _make_case(
    scenario: str,
    *,
    rows: int,
    columns: int,
    samples: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weight = torch.randn(rows, columns, generator=_generator(seed, device), device=device)
    calibration = torch.randn(samples, columns, generator=_generator(seed + 1, device), device=device)
    heldout = torch.randn(samples, columns, generator=_generator(seed + 2, device), device=device)

    if scenario == "weight_outlier":
        weight[:, ::17] *= 12.0
        weight[::11] *= 3.0
    elif scenario == "activation_outlier":
        scales = torch.ones(columns, device=device)
        scales[::13] = 10.0
        calibration *= scales
        heldout *= scales
    elif scenario == "correlated":
        latent_cal = torch.randn(samples, 8, generator=_generator(seed + 3, device), device=device)
        latent_eval = torch.randn(samples, 8, generator=_generator(seed + 4, device), device=device)
        mixing = torch.randn(8, columns, generator=_generator(seed + 5, device), device=device)
        calibration = latent_cal @ mixing + 0.03 * calibration
        heldout = latent_eval @ mixing + 0.03 * heldout
    elif scenario == "rank_deficient":
        calibration[:, columns // 2 :] = calibration[:, : columns // 2]
        heldout[:, columns // 2 :] = heldout[:, : columns // 2]
    elif scenario == "ill_conditioned":
        scales = torch.logspace(-3, 3, columns, device=device)
        calibration *= scales
        heldout *= scales
    elif scenario != "gaussian":
        raise ValueError(f"Unknown scenario: {scenario}")

    # Match common model quantization input while retaining FP32 Hessian math.
    return weight.to(torch.bfloat16), calibration.to(torch.bfloat16), heldout.to(torch.bfloat16)


def _quantize(
    weight: torch.Tensor,
    calibration: torch.Tensor,
    heldout: torch.Tensor,
    *,
    role: str,
    variant: Variant,
    group_size: int,
) -> dict[str, float]:
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=False, device=weight.device, dtype=weight.dtype)
    layer.weight.data.copy_(weight)
    wrapped = NamedModule(layer, role, f"model.layers.0.{role}", layer_index=0)
    role_flag = role.split(".")[-1].removesuffix("_proj")
    wrapped.state["module_tree_flags"] = frozenset({role_flag})
    qcfg = QuantizeConfig(
        bits=4,
        group_size=group_size,
        sym=True,
        adaptive_damping=variant.adaptive_damping,
        adaptive_clipping=variant.adaptive_clipping,
        scale_search=variant.scale_search,
    )
    gptq = GPTQ(wrapped, qcfg=qcfg)
    gptq.quantizer.configure(perchannel=True)
    gptq.add_batch(calibration, None)
    if weight.is_cuda:
        torch.cuda.synchronize(weight.device)
    start = time.perf_counter()
    reconstructed, _, _, _, _, _, damp_percent, _ = gptq.quantize()
    if weight.is_cuda:
        torch.cuda.synchronize(weight.device)
    elapsed = time.perf_counter() - start
    metrics = analyze_output_error(heldout, weight, reconstructed)
    if not metrics.get("available"):
        raise RuntimeError(f"Output telemetry unavailable: {metrics}")
    required = ("mean_absolute_error", "rmse", "relative_l2_error", "softmax_kld_mean", "top1_agreement")
    if not all(torch.isfinite(torch.tensor(float(metrics[key]))) for key in required):
        raise RuntimeError(f"Non-finite output telemetry: {metrics}")
    return {
        **{key: float(metrics[key]) for key in required},
        "damp_percent": float(damp_percent),
        "seconds": elapsed,
    }


def _table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def render(row: tuple[str, ...]) -> str:
        return "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"

    return "\n".join((separator, render(headers), separator, *(render(row) for row in rows), separator))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rows", type=int, default=96)
    # Two groups are intentional: they exercise the prior-group GPTQ update
    # and adaptive damping's optional online group feedback.
    parser.add_argument("--columns", type=int, default=256)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
    records: dict[str, list[dict[str, float]]] = {variant.label: [] for variant in VARIANTS}
    scenario_records: dict[tuple[str, str], list[dict[str, float]]] = {
        (scenario, variant.label): [] for scenario in SCENARIOS for variant in VARIANTS
    }
    for scenario_index, scenario in enumerate(SCENARIOS):
        weight, calibration, heldout = _make_case(
            scenario,
            rows=args.rows,
            columns=args.columns,
            samples=args.samples,
            device=device,
            seed=9100 + 101 * scenario_index,
        )
        for role in MODULE_ROLES:
            for variant in VARIANTS:
                metrics = _quantize(
                    weight,
                    calibration,
                    heldout,
                    role=role,
                    variant=variant,
                    group_size=args.group_size,
                )
                records[variant.label].append(metrics)
                scenario_records[(scenario, variant.label)].append(metrics)

    rows = []
    for variant in VARIANTS:
        values = records[variant.label]
        rows.append(
            (
                variant.label,
                f"{statistics.fmean(item['mean_absolute_error'] for item in values):.8g}",
                f"{statistics.fmean(item['rmse'] for item in values):.8g}",
                f"{statistics.fmean(item['relative_l2_error'] for item in values):.8g}",
                f"{statistics.fmean(item['softmax_kld_mean'] for item in values):.8g}",
                f"{statistics.fmean(item['top1_agreement'] for item in values):.6f}",
                f"{statistics.fmean(item['damp_percent'] for item in values):.6f}",
                f"{sum(item['seconds'] for item in values):.3f}",
            )
        )
    print(
        _table(
            ("Variant", "Mean MAE", "Mean RMSE", "Mean rel-L2", "Mean KLD", "Top-1", "Mean damp", "Time s"),
            rows,
        )
    )
    detail_rows = []
    for scenario in SCENARIOS:
        for variant in VARIANTS:
            values = scenario_records[(scenario, variant.label)]
            detail_rows.append(
                (
                    scenario,
                    variant.label,
                    f"{statistics.fmean(item['mean_absolute_error'] for item in values):.8g}",
                    f"{statistics.fmean(item['relative_l2_error'] for item in values):.8g}",
                    f"{statistics.fmean(item['softmax_kld_mean'] for item in values):.8g}",
                    f"{statistics.fmean(item['top1_agreement'] for item in values):.6f}",
                )
            )
    print(
        _table(
            ("Scenario", "Variant", "Mean MAE", "Mean rel-L2", "Mean KLD", "Top-1"),
            detail_rows,
        )
    )
    print(
        f"device={device}; dtype=bf16; bits=4; group_size={args.group_size}; "
        f"cases={len(SCENARIOS) * len(MODULE_ROLES)}; calibration/heldout={args.samples}/{args.samples}"
    )


if __name__ == "__main__":
    main()
