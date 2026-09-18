#!/usr/bin/env python3
"""Gate QVQ+YAQA P32 checkpoints with independent FP32 and FP64 objectives."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization.qvq import reconstruct_qvq_inner_weight


PAYLOAD_SUFFIXES = ("SU", "SV", "trellis", "bank_ids", "bank_alt_id")


def _tensor_file(checkpoint: Path, key: str) -> Path:
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        if key not in weight_map:
            raise KeyError(f"Checkpoint does not contain `{key}`.")
        return checkpoint / weight_map[key]
    candidates = sorted(checkpoint.glob("*.safetensors"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one safetensors file or an index in `{checkpoint}`.")
    return candidates[0]


def _load_module_payload(checkpoint: Path, module: str, device: torch.device) -> dict[str, torch.Tensor]:
    payload = {}
    for suffix in PAYLOAD_SUFFIXES:
        key = f"{module}.{suffix}"
        with safe_open(str(_tensor_file(checkpoint, key)), framework="pt", device=str(device)) as handle:
            payload[suffix] = handle.get_tensor(key)
    return payload


def _objective(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    input_hessian: torch.Tensor,
    output_hessian: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Evaluate trace(E.T H_I E H_O) without sharing YAQA's FP32 helper."""

    error = reconstructed.to(dtype) - target.to(dtype)
    h_input = input_hessian.to(dtype)
    h_output = output_hessian.to(dtype)
    return torch.einsum("ij,ik,kl,lj->", error, h_input, error, h_output)


def _checkpoint_oracle(
    payload: dict[str, torch.Tensor],
    snapshot: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor, float]:
    transformed_weight = snapshot["transformed_weight"]
    if not isinstance(transformed_weight, torch.Tensor):
        raise TypeError("Frozen snapshot transformed_weight must be a tensor.")
    in_features, out_features = transformed_weight.shape
    snapshot_su = snapshot["SU"]
    snapshot_sv_sign = snapshot["SV_sign"]
    if not isinstance(snapshot_su, torch.Tensor) or not isinstance(snapshot_sv_sign, torch.Tensor):
        raise TypeError("Frozen snapshot must contain the RHT sign vectors.")
    if not torch.equal(payload["SU"].cpu().float(), snapshot_su.float()):
        raise ValueError("Checkpoint SU does not match the frozen solver input.")
    if not torch.equal(payload["SV"].sign().cpu().float(), snapshot_sv_sign.sign().float()):
        raise ValueError("Checkpoint SV signs do not match the frozen solver input.")
    scale = payload["SV"].float().abs().mean()
    if not bool(torch.isfinite(scale)) or float(scale.item()) <= 0:
        raise ValueError("Checkpoint SV does not define a finite positive module scale.")
    reconstructed = reconstruct_qvq_inner_weight(
        payload["trellis"],
        bits=float(snapshot["bits"]),
        in_features=in_features,
        out_features=out_features,
        bank_ids=payload["bank_ids"],
        v2b2_p32=True,
        bank_alt_id=payload["bank_alt_id"],
    )
    target = transformed_weight.to(reconstructed.device) / scale
    h_input = snapshot["transformed_hessian"]
    h_output = snapshot["transformed_output_hessian"]
    if not isinstance(h_input, torch.Tensor) or not isinstance(h_output, torch.Tensor):
        raise TypeError("Frozen YAQA snapshot must contain both transformed Hessian factors.")
    fp32 = _objective(reconstructed, target, h_input.to(reconstructed.device), h_output.to(reconstructed.device), torch.float32)
    fp64 = _objective(reconstructed, target, h_input.to(reconstructed.device), h_output.to(reconstructed.device), torch.float64)
    return fp32, fp64, float(scale.item())


def _gate_delta(baseline: float, candidate: float, absolute: float, relative: float) -> dict[str, object]:
    delta = candidate - baseline
    allowance = max(absolute, abs(baseline) * relative)
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "relative_delta": delta / abs(baseline) if baseline else (0.0 if delta == 0 else math.inf),
        "allowance": allowance,
        "passed": delta <= allowance,
    }


def _relative_gap(low_precision: float, reference: float) -> float:
    delta = abs(low_precision - reference)
    return delta / abs(reference) if reference else (0.0 if delta == 0 else math.inf)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--module", help="Defaults to the module recorded by the frozen snapshot.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-absolute-regression", type=float, default=0.0)
    parser.add_argument("--max-relative-regression", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_absolute_regression < 0 or args.max_relative_regression < 0:
        parser.error("oracle regression allowances must be nonnegative")

    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    snapshot = torch.load(args.snapshot, map_location="cpu", weights_only=True)
    if snapshot.get("schema") != "qvq.frozen-solver-input.v1" or not snapshot.get("v2b2_p32"):
        raise ValueError("Oracle requires a QVQ V2B2-P32 frozen solver snapshot.")
    module = args.module or snapshot.get("module")
    if not isinstance(module, str) or not module:
        raise ValueError("Oracle module must be supplied or recorded in the snapshot.")

    baseline_payload = _load_module_payload(args.baseline, module, device)
    candidate_payload = _load_module_payload(args.candidate, module, device)
    payload_equal = {
        suffix: bool(torch.equal(baseline_payload[suffix], candidate_payload[suffix]))
        for suffix in PAYLOAD_SUFFIXES
    }
    baseline_fp32, baseline_fp64, baseline_scale = _checkpoint_oracle(baseline_payload, snapshot)
    candidate_fp32, candidate_fp64, candidate_scale = _checkpoint_oracle(candidate_payload, snapshot)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    fp32_gate = _gate_delta(
        float(baseline_fp32.item()),
        float(candidate_fp32.item()),
        args.max_absolute_regression,
        args.max_relative_regression,
    )
    fp64_gate = _gate_delta(
        float(baseline_fp64.item()),
        float(candidate_fp64.item()),
        args.max_absolute_regression,
        args.max_relative_regression,
    )
    report = {
        "schema": "qvq.yaqa-p32-dual-oracle.v1",
        "module": module,
        "bits": float(snapshot["bits"]),
        "policy": {
            "float32_matmul_precision": "highest",
            "tf32": False,
            "max_absolute_regression": args.max_absolute_regression,
            "max_relative_regression": args.max_relative_regression,
        },
        "payload_equal": payload_equal,
        "all_payload_equal": all(payload_equal.values()),
        "scale": {"baseline": baseline_scale, "candidate": candidate_scale},
        "fp32": fp32_gate,
        "fp64": fp64_gate,
        "precision_gap": {
            "baseline_relative": _relative_gap(fp32_gate["baseline"], fp64_gate["baseline"]),
            "candidate_relative": _relative_gap(fp32_gate["candidate"], fp64_gate["candidate"]),
        },
        "passed": bool(fp32_gate["passed"] and fp64_gate["passed"]),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
