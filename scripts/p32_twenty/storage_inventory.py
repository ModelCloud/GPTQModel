"""Inventory the read-only F6 snapshot using safetensor headers, without allocating tensors."""

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

SNAPSHOT = Path(
    "/root/qvq-results/calibration-fisher-frontier-wave14-v1/llama32-1b-f6_yaqa125x_seed7"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(SNAPSHOT):
        parser.error("Output must be outside the snapshot")
    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    tensors = {}
    shards = []
    for name in sorted(set(index.values())):
        path = SNAPSHOT / name
        with path.open("rb") as f:
            header_length = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_length))
        payload = 0
        for key, value in header.items():
            if key == "__metadata__":
                continue
            if index.get(key) != name:
                raise ValueError("Index/header mismatch: " + key)
            size = value["data_offsets"][1] - value["data_offsets"][0]
            tensors[key] = {
                "dtype": value["dtype"],
                "shape": value["shape"],
                "bytes": size,
            }
            payload += size
        shards.append(
            {
                "file": name,
                "file_bytes": path.stat().st_size,
                "tensor_bytes": payload,
                "overhead_bytes": path.stat().st_size - payload,
            }
        )
    if set(index) != set(tensors):
        raise ValueError("Missing indexed tensors")
    modules = []
    quant_keys = set()
    for key in sorted(tensors):
        if not key.endswith(".trellis"):
            continue
        prefix = key[:-8]
        params = {
            k.rsplit(".", 1)[-1]: v
            for k, v in tensors.items()
            if k.startswith(prefix + ".")
        }
        quant_keys.update(prefix + "." + k for k in params)
        weights = math.prod(params["SU"]["shape"]) * math.prod(params["SV"]["shape"])
        components = {k: v["bytes"] for k, v in params.items()}
        modules.append(
            {
                "module": prefix,
                "format": "P32" if "bank_alt_id" in params else "QVQ",
                "logical_weights": weights,
                "components_bytes": components,
                "total_bytes": sum(components.values()),
                "effective_bpw": 8 * sum(components.values()) / weights,
            }
        )
    dense = {k: v for k, v in tensors.items() if k not in quant_keys}
    qw = sum(m["logical_weights"] for m in modules)
    qb = sum(m["total_bytes"] for m in modules)
    dw = sum(math.prod(t["shape"]) for t in dense.values())
    db = sum(t["bytes"] for t in dense.values())
    meta = []
    for path in sorted(SNAPSHOT.iterdir()):
        if path.is_file() and (
            path.name
            in (
                "config.json",
                "quantize_config.json",
                "model.safetensors.index.json",
                "generation_config.json",
            )
            or path.name.startswith(("tokenizer", "special_tokens"))
        ):
            meta.append(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    stored = sum(s["file_bytes"] for s in shards) + sum(m["bytes"] for m in meta)
    report = {
        "snapshot": str(SNAPSHOT),
        "modules": modules,
        "dense_tensors": dense,
        "shards": shards,
        "metadata_files": meta,
        "quantized_logical_weights": qw,
        "quantized_tensor_bytes": qb,
        "quantized_projection_bpw": 8 * qb / qw,
        "dense_stored_elements": dw,
        "dense_tensor_bytes": db,
        "whole_model_logical_elements": qw + dw,
        "serialized_inference_files_bytes": stored,
        "whole_model_serialized_bpw": 8 * stored / (qw + dw),
        "conventions": "Shared stored dense tensors counted once (tied weights not duplicated). Whole-model denominator includes saved dense scalar/vector parameters. Metadata includes tokenizer/config/index, excludes evaluation/audit reports. Quantized BPW excludes shard header and model-global files, itemized separately. Runtime LUTs, workspaces and decoded caches are not serialized checkpoint storage; candidates must add their own inventories.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if isinstance(v, (int, float))}, indent=2
        )
    )


if __name__ == "__main__":
    main()
