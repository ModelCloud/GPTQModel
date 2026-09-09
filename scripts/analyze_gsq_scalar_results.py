"""Audit completed scalar GSQ reports and their saved packed tensors."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def analyze(directory):
    path = directory / "report.json"
    report = json.loads(path.read_text())
    if report["state"] != "complete" or not all(v["complete"] for v in report["model"].values()):
        raise ValueError(f"Incomplete run: {directory}")
    arms = report["provenance"]["arms"]
    result = {"report_sha256": digest(path), "method": report["provenance"]["method"],
              "bits": report["provenance"]["bits"], "layers": {}, "model": report["model"],
              "paired_intervals": report["paired_intervals"], "tensor_bytes": {a: 0 for a in arms}}
    result["paired_input_shift"] = {}
    bits = result["bits"]
    if bits not in (2, 4, 8):
        raise ValueError("Logical-code difference audit currently supports non-straddling 2/4/8-bit words")
    for name, layer in report["layers"].items():
        if result["method"] == "gptaq":
            inputs = torch.load(directory / f"{name}.inputs.pt", weights_only=True, map_location="cpu")
            result["paired_input_shift"][name] = {}
            for split in ("train", "heldout"):
                native = torch.cat(inputs[split]).float()
                propagated = torch.cat(inputs["propagated"][split]).float()
                if native.shape != propagated.shape or not torch.isfinite(propagated).all():
                    raise ValueError("Invalid paired activations")
                result["paired_input_shift"][name][split] = {
                    "tokens": len(native), "relative_squared_error":
                        float((native-propagated).square().sum()/native.square().sum()),
                }
            del inputs, native, propagated
        baseline = torch.load(directory / f"{name}.baseline.pt", map_location="cpu", weights_only=True)
        result["layers"][name] = {}
        for arm in arms:
            payload = directory / f"{name}.{arm}.pt"
            record = layer["arms"][arm]
            if digest(payload) != record["payload_sha256"]:
                raise ValueError(f"Payload changed: {payload}")
            if not record["reload_exact"] or not record["runtime_parity"]["pass"] or not record["packed_objective_matches"]:
                raise ValueError(f"Failed layer gate: {name}/{arm}")
            state = torch.load(payload, map_location="cpu", weights_only=True)
            if state.keys() != baseline.keys():
                raise ValueError("Changed packed schema")
            for field in ("qzeros", "g_idx"):
                if not torch.equal(state[field], baseline[field]):
                    raise ValueError(f"Unexpected changed {field}")
            tensor_bytes = sum(v.numel()*v.element_size() for v in state.values())
            result["tensor_bytes"][arm] += tensor_bytes
            xor = (state["qweight"].to(torch.int64) ^ baseline["qweight"].to(torch.int64)).unsqueeze(1)
            shifts = torch.arange(0, 32, bits).view(1, -1, 1)
            changed_codes = int(((xor >> shifts) & (2**bits-1)).count_nonzero())
            diagnostics = record["gsq"]
            result["layers"][name][arm] = {
                "tensor_bytes": tensor_bytes, "changed_codes": changed_codes,
                "changed_scales": int((state["scales"] != baseline["scales"]).count_nonzero()),
                "all_tensors_equal_baseline": all(torch.equal(v, baseline[k]) for k, v in state.items()),
                "objective_reduction_percent": None if diagnostics is None or diagnostics["before"] == 0 or
                    diagnostics["objective"] == "asymmetric_quadratic_without_constant" else
                    100*(1-diagnostics["after"]/diagnostics["before"]),
                "objective_delta": None if diagnostics is None else diagnostics["after"]-diagnostics["before"],
                "runtime_parity": record["runtime_parity"],
            }
    if len(set(result["tensor_bytes"].values())) != 1:
        raise ValueError("Changed packed tensor size")
    result["deltas"] = {}
    baseline_mean = report["model"]["baseline"]["mean"]
    for arm in arms[1:]:
        metrics = report["model"][arm]["mean"]
        result["deltas"][arm] = {
            "kld_reduction_percent": 100*(1-metrics["kl_teacher_candidate"]/baseline_mean["kl_teacher_candidate"]),
            "mse_change_percent": 100*(metrics["mse"]/baseline_mean["mse"]-1),
            "top1_change_percentage_points": 100*(metrics["top1_agreement"]-baseline_mean["top1_agreement"]),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an earlier audit")
    results = {str(directory): analyze(directory) for directory in args.runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps({k: {"deltas": v["deltas"], "tensor_bytes": v["tensor_bytes"]} for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
