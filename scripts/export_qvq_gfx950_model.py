"""Export exact checkpoint geometry coverage for QVQ's external HIP ABI."""

import argparse
import json
import struct
from pathlib import Path

from export_qvq_gfx950 import export


def checkpoint_projections(model):
    tensors = {}
    for shard in sorted(model.glob("*.safetensors")):
        with shard.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            if length > 64 * 1024 * 1024:
                raise ValueError(f"unreasonable header size: {shard}")
            header = json.loads(stream.read(length))
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ValueError(f"duplicate tensor: {name}")
            tensors[name] = (shard, 8 + length, entry)
    projections = []
    for name, (_, _, entry) in sorted(tensors.items()):
        if not name.endswith(".trellis"):
            continue
        prefix = name.removesuffix(".trellis")
        k = tensors[prefix + ".SU"][2]["shape"][0]
        n = tensors[prefix + ".SV"][2]["shape"][0]
        tiles, words = entry["shape"]
        if entry["dtype"] != "I32" or tiles != k * n // 256 or words % 4:
            raise ValueError(f"unsupported trellis layout: {name}")
        bits = words // 4
        bank = 0
        if prefix + ".bank_alt_id" in tensors:
            shard, start, alt = tensors[prefix + ".bank_alt_id"]
            if alt["dtype"] != "U8" or alt["shape"] != [1]:
                raise ValueError(f"unsupported alternate bank: {prefix}")
            with shard.open("rb") as stream:
                stream.seek(start + alt["data_offsets"][0])
                bank = stream.read(1)[0]
            if prefix + ".bank_ids" not in tensors or bits not in range(4, 8):
                raise ValueError(f"incomplete P32 projection: {prefix}")
        elif bits != 8:
            raise ValueError(f"unsupported canonical projection: {prefix}")
        projections.append(
            {
                "name": prefix,
                "k": k,
                "n": n,
                "transition_bits": bits,
                "bank_alt_id": bank,
            }
        )
    if not projections:
        raise ValueError("no QVQ projections found")
    return projections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args()
    projections = checkpoint_projections(args.model)
    shapes = sorted(
        {(p["k"], p["n"], p["transition_bits"], p["bank_alt_id"]) for p in projections}
    )
    args.output.mkdir(parents=True, exist_ok=False)
    record = {
        "model": str(args.model.resolve()),
        "rows": args.rows,
        "projections": projections,
        "unique_geometries": len(shapes),
        "artifacts": [],
        "status": "scanned",
    }
    manifest = args.output / "model_manifest.json"
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"{len(projections)} projections, {len(shapes)} geometries, rows={args.rows}",
        flush=True,
    )
    if args.scan_only:
        return
    for m in args.rows:
        for k, n, bits, bank in shapes:
            name = f"m{m}-k{k}-n{n}-t{bits}-b{bank}"
            export(args.output / name, m, k, n, bits, bank)
            record["artifacts"].append(name)
            manifest.write_text(json.dumps(record, indent=2) + "\n")
    record["status"] = "exported-not-validated"
    manifest.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
