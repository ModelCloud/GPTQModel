#!/usr/bin/env python3
"""Run the Wave-11 frozen first-module QVQ solver canary.

The input snapshot is produced once by the opt-in hook in ``qvq.py``.  This
script intentionally bypasses model loading and calibration: it feeds the
same serialized tensors to the imported commit's ``quantize_qvq_linear`` and
records canonical/fixed-family costs plus all discrete packed outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

import torch


def _sha256_tensor(tensor: torch.Tensor | None) -> str | None:
    if tensor is None:
        return None
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _result_record(result) -> dict[str, object]:
    serialized = result.serialized_tensors()
    return {
        "proxy_loss": float(result.proxy_loss.detach().to(device="cpu").item()),
        "baseline_proxy_loss": float(result.baseline_proxy_loss.detach().to(device="cpu").item()),
        "kronecker_proxy_loss": (
            None
            if result.kronecker_proxy_loss is None
            else float(result.kronecker_proxy_loss.detach().to(device="cpu").item())
        ),
        "bank_selected_family_id": (
            None if result.bank_alt_id is None else int(result.bank_alt_id.reshape(-1)[0].detach().cpu().item())
        ),
        "bank_ids_sha256": _sha256_tensor(result.bank_ids),
        "bank_alt_id_sha256": _sha256_tensor(result.bank_alt_id),
        "trellis_sha256": _sha256_tensor(result.trellis),
        "SU_sha256": _sha256_tensor(result.SU),
        "SV_sha256": _sha256_tensor(result.SV),
        "inner_weight_sha256": _sha256_tensor(result.inner_weight),
        "weight_sha256": _sha256_tensor(result.weight),
        "serialized_sha256": {
            name: _sha256_tensor(value) for name, value in serialized.items()
        },
        "telemetry": result.telemetry,
        "yaqa_bank_fallback_to_v2": result.yaqa_bank_fallback_to_v2,
        "yaqa_selector_churn": result.yaqa_selector_churn,
        "yaqa_family_changed": result.yaqa_family_changed,
        "yaqa_block_family_id": result.yaqa_block_family_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--module", default="model.layers.0.self_attn.q_proj")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    # The caller's run_in_worktree bootstrap has already loaded the requested
    # historical package.  This is the actual commit-specific solver under test.
    from gptqmodel.quantization.qvq import quantize_qvq_linear

    payload = torch.load(args.snapshot, map_location="cpu")
    if payload.get("schema") != "qvq.frozen-solver-input.v1":
        raise ValueError(f"Unsupported frozen snapshot schema: {payload.get('schema')!r}")
    if payload.get("module") != args.module:
        raise ValueError(f"Snapshot module {payload.get('module')!r} does not match {args.module!r}")

    device = torch.device(args.device)
    weight = payload["weight"].to(device=device, dtype=torch.float32)
    input_hessian = payload["input_hessian"].to(device=device, dtype=torch.float32)
    output_hessian = payload["output_hessian"]
    if output_hessian is not None:
        output_hessian = output_hessian.to(device=device, dtype=torch.float32)

    common = {
        "output_hessian": output_hessian,
        "bits": float(payload["bits"]),
        "seed": int(payload["seed"]),
        "damp_percent": float(payload["damp_percent"]),
        "rounding": str(payload["rounding"]),
        "codebook_version": str(payload["codebook_version"]),
        "vector_size": int(payload["vector_size"]),
        "trellis_window": int(payload["trellis_window"]),
        "dual_v2": bool(payload["dual_v2"]),
        "v2b4_p64": bool(payload["v2b4_p64"]),
        "v2b2_p32": bool(payload["v2b2_p32"]),
        "bank_count": int(payload["bank_count"]),
        "yaqa_sample_strategy": str(payload["yaqa_sample_strategy"]),
        "viterbi_pruning": None,
    }
    supported = set(inspect.signature(quantize_qvq_linear).parameters)
    common = {key: value for key, value in common.items() if key in supported}

    # Candidate zero is canonical reselect.  The four fixed-family entries
    # expose the discrete alternatives and make candidate-cost differences
    # explicit in the artifact.
    candidates: dict[str, dict[str, object]] = {}
    candidate_args = [("canonical_reselect", None)] + [
        (f"fixed_family_{family_id}", family_id) for family_id in range(4)
    ]
    for name, family_id in candidate_args:
        kwargs = dict(common)
        if family_id is None:
            kwargs["yaqa_v2b2_family_mode"] = str(payload["yaqa_v2b2_family_mode"])
        else:
            kwargs["yaqa_v2b2_family_mode"] = "fixed_block_ldlq"
            kwargs["yaqa_v2b2_fixed_family_id"] = family_id
        result = quantize_qvq_linear(weight, input_hessian, **kwargs)
        candidates[name] = _result_record(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema": "qvq.wave11-frozen-solver-result.v1",
        "module": args.module,
        "snapshot": str(args.snapshot),
        "snapshot_sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
        "device": str(device),
        "input_hashes": {
            key: _sha256_tensor(payload.get(key))
            for key in (
                "weight",
                "input_hessian",
                "output_hessian",
                "SU",
                "SV_sign",
                "transformed_weight",
                "transformed_hessian",
                "transformed_output_hessian",
            )
        },
        "candidates": candidates,
    }
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
