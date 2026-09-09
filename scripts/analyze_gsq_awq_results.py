"""Audit saved AWQ GSQ payloads and report paired real-model effects."""

import argparse
import json
from pathlib import Path

import torch

from scripts.validate_qvq_gsq_layers import TARGETS, digest, write_json


def codes(words):
    # AWQ GEMM interleaves columns within each eight-code word.
    shifts = torch.arange(0, 32, 4, dtype=torch.int64)
    unpacked = (words.to(torch.int64).unsqueeze(-1) >> shifts) & 15
    return unpacked[..., [0, 4, 1, 5, 2, 6, 3, 7]].reshape(words.shape[0], -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an earlier audit")
    report = json.loads((args.run / "report.json").read_text())
    if report["state"] != "complete" or report["provenance"]["method"] != "awq":
        raise ValueError("Expected a complete real AWQ report")
    result = {"run": str(args.run), "report_sha256": digest(args.run / "report.json"),
              "layers": {}, "effects": {}, "scope": "saved AWQ GEMM QKV and reported F6 canonical propagation"}
    for name in TARGETS:
        states = {}
        for arm in ("baseline", "gsq_fixed", "gsq_scales"):
            entry = report["layers"][name]["arms"][arm]
            path = args.run / f"{name}.{arm}.pt"
            if digest(path) != entry["payload_sha256"] or not entry["runtime_parity"]["pass"]:
                raise ValueError("Payload integrity or runtime parity failed")
            if not entry["reload_exact"] or not entry["packed_objective_matches"]:
                raise ValueError("Export did not match fitting objective")
            states[arm] = torch.load(path, weights_only=True, map_location="cpu")
        baseline = states["baseline"]
        baseline_bytes = sum(v.numel()*v.element_size() for v in baseline.values())
        rows = {}
        for arm, state in states.items():
            if state.keys() != baseline.keys() or not torch.equal(state["qzeros"], baseline["qzeros"]):
                raise ValueError("GSQ changed checkpoint schema or zero points")
            size = sum(v.numel()*v.element_size() for v in state.values())
            if size != baseline_bytes or not torch.isfinite(state["scales"]).all() or (state["scales"] <= 0).any():
                raise ValueError("GSQ changed storage size or produced invalid scales")
            if arm == "gsq_fixed" and not torch.equal(state["scales"], baseline["scales"]):
                raise ValueError("Fixed-scale GSQ changed scales")
            rows[arm] = {"tensor_bytes": size,
                         "changed_codes": int((codes(state["qweight"]) != codes(baseline["qweight"])).sum()),
                         "changed_scales": int((state["scales"] != baseline["scales"]).sum()),
                         "exact_payload": all(torch.equal(v, baseline[k]) for k, v in state.items())}
        result["layers"][name] = rows
    baseline = report["model"]["baseline"]["mean"]
    for arm in ("gsq_fixed", "gsq_scales"):
        result["effects"][arm] = {metric: {"baseline": value, "candidate": report["model"][arm]["mean"][metric],
                                          "delta": report["model"][arm]["mean"][metric]-value,
                                          "paired_document_interval": report["paired_intervals"][arm][metric]}
                                  for metric, value in baseline.items()}
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
