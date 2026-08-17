#!/usr/bin/env python3
"""Combine matched YAQA Sketch-B caches across independent Monte Carlo seeds."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import torch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    return parser


def combine_yaqa_factor_payloads(payloads: list[dict]) -> dict:
    """Average matched factor estimators, weighted by their Monte Carlo sample counts."""

    if len(payloads) < 2:
        raise ValueError("YAQA factor ensemble requires at least two caches")
    reference_metadata = payloads[0].get("metadata")
    if not isinstance(reference_metadata, dict):
        raise TypeError("YAQA factor cache metadata must be a dictionary")
    contract = {key: value for key, value in reference_metadata.items() if key != "seed"}
    names = None
    weights = []
    seeds = []
    for payload in payloads:
        metadata = payload.get("metadata")
        stats = payload.get("stats")
        input_hessians = payload.get("input_hessians")
        output_hessians = payload.get("output_hessians")
        if not all(isinstance(value, dict) for value in (metadata, stats, input_hessians, output_hessians)):
            raise ValueError("YAQA factor cache payload is incomplete")
        if {key: value for key, value in metadata.items() if key != "seed"} != contract:
            raise ValueError("YAQA factor caches must share rows, shapes, model, dataset, and batch contract")
        current_names = set(input_hessians)
        if current_names != set(output_hessians) or (names is not None and current_names != names):
            raise ValueError("YAQA factor caches must contain identical module names")
        names = current_names
        samples = stats.get("monte_carlo_samples_per_output")
        sequences = stats.get("independent_sequences")
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise ValueError("YAQA factor cache has an invalid Monte Carlo sample count")
        if sequences != metadata.get("rows"):
            raise ValueError("YAQA factor cache sequence count does not match its row contract")
        weights.append(samples)
        seeds.append(metadata.get("seed"))

    assert names is not None
    total_weight = sum(weights)

    def combine_factor(kind: str, name: str) -> torch.Tensor:
        factors = [payload[kind][name] for payload in payloads]
        reference = factors[0]
        if (
            reference.device.type != "cpu"
            or reference.dtype != torch.float32
            or not reference.is_contiguous()
            or any(
                factor.device.type != "cpu"
                or factor.dtype != torch.float32
                or not factor.is_contiguous()
                or tuple(factor.shape) != tuple(reference.shape)
                or not torch.isfinite(factor).all()
                for factor in factors
            )
        ):
            raise ValueError(f"YAQA factor `{name}` has incompatible storage, geometry, or values")
        combined = sum(factor * (weight / total_weight) for factor, weight in zip(factors, weights, strict=True))
        return combined.contiguous()

    input_hessians = {name: combine_factor("input_hessians", name) for name in sorted(names)}
    output_hessians = {name: combine_factor("output_hessians", name) for name in sorted(names)}
    metadata = dict(reference_metadata)
    metadata["seed"] = f"ensemble:{','.join(map(str, seeds))}"
    stats = dict(payloads[0]["stats"])
    stats.update(
        {
            "method": "YAQA-v3 Sketch B Monte Carlo factor ensemble",
            "monte_carlo_samples_per_output": total_weight,
            "factor_ensemble_seeds": seeds,
            "factor_ensemble_components": len(payloads),
        }
    )
    return {
        "metadata": metadata,
        "input_hessians": input_hessians,
        "output_hessians": output_hessians,
        "stats": stats,
    }


def main() -> None:
    args = _parser().parse_args()
    payloads = [torch.load(path, map_location="cpu", weights_only=True) for path in args.input]
    if any(not isinstance(payload, dict) for payload in payloads):
        raise ValueError("YAQA factor caches must contain dictionary payloads")
    combined = combine_yaqa_factor_payloads(payloads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{uuid.uuid4().hex}.tmp")
    torch.save(combined, temporary)
    temporary.replace(args.output)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(
        json.dumps({"metadata": combined["metadata"], "stats": combined["stats"]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Saved {len(payloads)}-seed YAQA factor ensemble with "
        f"{combined['stats']['monte_carlo_samples_per_output']} samples/output to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
