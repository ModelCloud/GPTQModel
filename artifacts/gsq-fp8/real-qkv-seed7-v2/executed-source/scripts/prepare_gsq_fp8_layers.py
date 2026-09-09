"""Bind real FP8 GSQ experiments to saved dense weights and disjoint documents.

Preparation only: no quantization or quality claims. All 16 calibration
documents train the fitter; 32 held-out documents remain unselected.
"""

import argparse
import json
import subprocess
from pathlib import Path

from scripts.validate_qvq_gsq_layers import ROOT, TARGETS, digest, write_json


def prepare(inputs, output):
    import torch
    from safetensors import safe_open

    if output.exists():
        raise ValueError("Use a fresh output directory")
    source = json.loads((inputs / "provenance.json").read_text())
    documents = json.loads((inputs / "inputs.json").read_text())
    if digest(inputs / "inputs.json") != source["inputs_sha256"]:
        raise ValueError("Saved token selection changed")
    if len(documents["train"]) != 16 or len(documents["heldout"]) != 32:
        raise ValueError("Expected locked 16 calibration / 32 evaluation documents")
    splits = {"train": documents["train"], "heldout": documents["heldout"]}
    token_sets = {key: {tuple(row["input_ids"]) for row in rows} for key, rows in splits.items()}
    for left, right in (("train", "heldout"),):
        if token_sets[left] & token_sets[right]:
            raise ValueError(f"Duplicate token documents across {left}/{right}")
    for path, sha in source["source_hashes"].items():
        if digest(path) != sha:
            raise ValueError(f"Calibration source changed: {path}")
    dense = Path(source["dense"]) / "model.safetensors"
    if digest(dense) != source["file_hashes"][str(dense)]:
        raise ValueError("Dense model source hash changed")
    files = [dense, inputs / "inputs.json", inputs / "provenance.json", Path(__file__).resolve()]
    shapes = {}
    with safe_open(dense, framework="pt", device="cpu") as handle:
        for name in TARGETS:
            path = inputs / f"{name}.inputs.pt"
            fixture = torch.load(path, weights_only=True)
            weight = fixture["weight"]
            if not torch.equal(weight, handle.get_tensor(name + ".weight").float()):
                raise ValueError(f"Dense weight mismatch: {name}")
            for split in ("train", "heldout"):
                for x, row in zip(fixture[split], documents[split], strict=True):
                    if x.shape != (len(row["input_ids"]), weight.shape[1]) or not torch.isfinite(x).all():
                        raise ValueError(f"Invalid saved activations: {name}/{split}")
            shapes[name] = list(weight.shape)
            files.append(path)
    files.extend(ROOT / path for path in (
        "gptqmodel/quantization/gsq_fp8.py", "gptqmodel/quantization/gsq_scalar.py",
        "gptqmodel/quantization/config.py", "gptqmodel/looper/weight_only_processor.py",
        "gptqmodel/nn_modules/qlinear/fp8.py", "scripts/validate_gsq_fp8_layers.py"))
    # Resolve all paths and hashes before creating any output.
    hashes = {str(path.resolve()): digest(path) for path in files}
    manifest = {
        "state": "prepared_not_executed", "inputs": str(inputs.resolve()),
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "files": hashes, "shapes": shapes, "seed": 7, "bits": 8, "format": "float8_e4m3fn", "scale_method": "row",
        "steps": 100, "candidates": 3, "learn_scales": False,
        "arms": ["baseline", "gsq_weight", "gsq_calibrated"],
        "scope": "real block-0 QKV FP8 fixed-scale fitting and packed reconstruction",
        "calibration_indices": {"train": list(range(16))},
        "heldout_indices": list(range(32)),
        "documents": {key: len(rows) for key, rows in splits.items()},
        "tokens": {key: sum(len(row["input_ids"]) for row in rows) for key, rows in splits.items()},
        "train_source_weights": source["train_source_weights"],
        "weighting": "multiply calibration activations by sqrt(source weight); heldout unweighted",
        "selection": "baseline is ordinary FP8 rounding; GSQ selects calibration objective only; heldout never selects",
    }
    output.mkdir(parents=True)
    write_json(output / "provenance.json", manifest)
    write_json(output / "documents.json", splits)
    print(json.dumps({key: manifest[key] for key in ("state", "shapes", "documents", "tokens")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.inputs.resolve(), args.output.resolve())
