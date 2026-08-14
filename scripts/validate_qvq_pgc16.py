"""Validate PGC16 capacity, distortion, and exact planar payload size.

This is a codebook/trellis gate. KLD and top-k agreement belong to
``analyze_gptq_low_bit_grid.py`` because they require actual module and model
output distributions.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization.qvq import (
    pack_trellis_states,
    tail_biting_viterbi_quantize,
)
from gptqmodel.quantization.qvq_codecs import PGC16_STATE_COUNT, pgc16_codebook
from gptqmodel.quantization.qvq_codecs.hyb_reference import (
    canonical_hyb_lut,
    hyb_codebook,
)


def request_performance_qos() -> bool:
    """Request Darwin user-interactive QoS so CPU work is scheduled on P cores."""

    if platform.system() != "Darwin":
        return True
    libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    set_qos = libsystem.pthread_set_qos_class_self_np
    set_qos.argtypes = [ctypes.c_uint, ctypes.c_int]
    set_qos.restype = ctypes.c_int
    return set_qos(0x21, 0) == 0


def reconstruction_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    """Return non-distributional error metrics for Gaussian reconstruction."""

    reference = reference.float()
    actual = actual.float()
    error = actual - reference
    reference_energy = reference.square().sum().clamp_min(torch.finfo(torch.float32).eps)
    error_energy = error.square().sum().clamp_min(torch.finfo(torch.float32).eps)
    abs_error = error.abs().flatten()
    quantiles = torch.quantile(abs_error, torch.tensor([0.95, 0.99], device=abs_error.device))
    return {
        "mse": error.square().mean().item(),
        "rmse": error.square().mean().sqrt().item(),
        "mae": abs_error.mean().item(),
        "relative_l2": (error_energy / reference_energy).sqrt().item(),
        "sqnr_db": (10 * torch.log10(reference_energy / error_energy)).item(),
        "cosine": F.cosine_similarity(reference.flatten(), actual.flatten(), dim=0).item(),
        "max_abs_error": abs_error.max().item(),
        "p95_abs_error": quantiles[0].item(),
        "p99_abs_error": quantiles[1].item(),
        "bias": error.mean().item(),
    }


@torch.inference_mode()
def best_reconstruction(
    sequences: torch.Tensor,
    codebook: torch.Tensor,
    *,
    bits: int,
    scale_factors: torch.Tensor,
) -> tuple[dict[str, float], float, torch.Tensor]:
    """Select the lowest-MSE scale from the declared shared grid."""

    base_scale = sequences.square().mean().sqrt() / codebook.square().mean().sqrt()
    best_metrics: dict[str, float] | None = None
    best_factor = math.nan
    best_states: torch.Tensor | None = None
    for factor in scale_factors:
        scale = base_scale * factor
        result = tail_biting_viterbi_quantize(sequences / scale, codebook, bits=bits)
        reconstructed = result.values * scale
        metrics = reconstruction_metrics(sequences, reconstructed)
        if best_metrics is None or metrics["mse"] < best_metrics["mse"]:
            best_metrics = metrics
            best_factor = float(factor.item())
            best_states = result.states.detach().cpu()
    assert best_metrics is not None and best_states is not None
    return best_metrics, best_factor, best_states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--scale-min", type=float, default=0.8)
    parser.add_argument("--scale-max", type=float, default=1.6)
    parser.add_argument("--scale-steps", type=int, default=9)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if args.batches < 1 or args.steps < 2 or args.scale_steps < 1:
        parser.error("batches/scale-steps must be positive and steps must be at least two")
    if args.scale_min <= 0 or args.scale_max < args.scale_min:
        parser.error("scale bounds must be positive and ordered")
    if not request_performance_qos():
        raise RuntimeError("failed to request performance-core QoS")

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS was requested but is unavailable")

    generator = torch.Generator().manual_seed(args.seed)
    sequences = torch.randn((args.batches, args.steps, 2), generator=generator).to(device)
    codebooks = {
        "hyb-reference": hyb_codebook(canonical_hyb_lut()).to(device),
        "pgc16-v1": pgc16_codebook(device=device),
    }
    scale_factors = torch.linspace(args.scale_min, args.scale_max, args.scale_steps, device=device)
    report: dict[str, Any] = {
        "settings": {
            "device": str(device),
            "batches": args.batches,
            "steps": args.steps,
            "seed": args.seed,
            "scale_factors": scale_factors.cpu().tolist(),
            "performance_qos": "user-interactive",
        },
        "unique_pgc16_vectors": int(torch.unique(pgc16_codebook(dtype=torch.float16), dim=0).shape[0]),
        "results": {},
        "payload_bytes_per_weight": {},
    }
    best_states: dict[int, torch.Tensor] = {}
    for bits in range(2, 9):
        for name, codebook in codebooks.items():
            metrics, scale_factor, states = best_reconstruction(
                sequences,
                codebook,
                bits=bits,
                scale_factors=scale_factors,
            )
            report["results"].setdefault(str(bits), {})[name] = {
                **metrics,
                "scale_factor": scale_factor,
            }
            if name == "pgc16-v1":
                best_states[bits] = states

        packed = pack_trellis_states(best_states[bits], bits=bits)
        weight_count = best_states[bits].numel() * 2
        report["payload_bytes_per_weight"][str(bits)] = packed.numel() * packed.element_size() / weight_count

    pgc_mse = {bits: report["results"][str(bits)]["pgc16-v1"]["mse"] for bits in range(2, 9)}
    gates = {
        "unique_65536": report["unique_pgc16_vectors"] == PGC16_STATE_COUNT,
        "w2_no_regression": pgc_mse[2] <= report["results"]["2"]["hyb-reference"]["mse"],
        "w5_w8_no_plateau": all(pgc_mse[bits + 1] < pgc_mse[bits] * 0.95 for bits in range(5, 8)),
        "exact_payload": all(
            report["payload_bytes_per_weight"][str(bits)] == bits / 8 for bits in range(2, 9)
        ),
    }
    report["gates"] = gates
    report["passed"] = all(gates.values())

    print(" bits | HYB MSE    | PGC16 MSE  | PGC/HYB | PGC rel-L2 | PGC SQNR | bytes/w")
    print("------+------------+------------+---------+------------+----------+--------")
    for bits in range(2, 9):
        hyb = report["results"][str(bits)]["hyb-reference"]
        pgc = report["results"][str(bits)]["pgc16-v1"]
        print(
            f" {bits:>4} | {hyb['mse']:.8f} | {pgc['mse']:.8f} | "
            f"{pgc['mse'] / hyb['mse']:.4f}  | {pgc['relative_l2']:.6f}   | "
            f"{pgc['sqnr_db']:7.2f}  | {report['payload_bytes_per_weight'][str(bits)]:.3f}"
        )
    print("gates:", json.dumps(gates, sort_keys=True))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
