"""Serialize each tested rank and account for complete standalone operator files."""

import argparse
import json
from pathlib import Path

import torch
import torchao  # noqa: F401 -- register the tensor classes in our own saved exports

ROOT = Path("/root/p32-native-recovery")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    for name in ["q_proj", "k_proj", "gate_proj", "down_proj"]:
        root = args.root / name
        # Files were created by this experiment; no third-party pickle inputs.
        bundle = torch.load(root / "export.pt", map_location="cpu", weights_only=False)
        report = json.loads((root / "report.json").read_text())
        k, n = bundle["a"].shape[0], bundle["b"].shape[1]
        records = []
        for rank in [16, 32, 64, 128]:
            r = min(rank, bundle["a"].shape[1])
            export = dict(bundle)
            # Clone both factors: contiguous row slices can retain a larger storage.
            export["a"] = bundle["a"][:, :r].clone(
                memory_format=torch.contiguous_format
            )
            export["b"] = bundle["b"][:r].clone(memory_format=torch.contiguous_format)
            export["in_features"] = k
            export["out_features"] = n
            export["actual_rank"] = r
            export["arithmetic"] = (
                "BF16 W4A16 tinygemm output cast FP32 plus (X.float @ A.float) @ B.float"
            )
            path = root / f"rank{rank}.pt"
            torch.save(export, path)
            size = path.stat().st_size
            records.append(
                {
                    "requested_rank": rank,
                    "actual_rank": r,
                    "file": str(path),
                    "serialized_operator_bytes": size,
                    "serialized_operator_bpw": 8 * size / (k * n),
                }
            )
        report["rank_exports"] = records
        report["storage_scope"] = (
            "Standalone operator serialization includes native tensor metadata, packed codes/scales, factors, dimensions and arithmetic configuration; excludes the rest of the model and runtime buffers."
        )
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            name,
            [
                (r["requested_rank"], round(r["serialized_operator_bpw"], 4))
                for r in records
            ],
            flush=True,
        )


if __name__ == "__main__":
    main()
