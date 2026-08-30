#!/usr/bin/env python3
"""Hash a QVQ checkpoint and optionally locate its first module divergence.

This is deliberately independent of model loading.  It hashes the exact
serialized safetensors payloads plus the run/config manifests, which makes it
safe to use after a quantization process has exited or before an evaluator is
started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from safetensors import safe_open


SERIALIZED_FIELDS = ("SU", "SV", "bank_ids", "bank_alt_id", "trellis", "bias")
ROLE_ORDER = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def module_name(key: str) -> tuple[str | None, str | None]:
    for field in SERIALIZED_FIELDS:
        suffix = "." + field
        if key.endswith(suffix):
            return key[: -len(suffix)], field
    return None, None


def module_sort_key(name: str):
    match = re.search(r"\.layers\.(\d+)\.", name)
    layer = int(match.group(1)) if match else 10**9
    role = name.rsplit(".", 1)[-1]
    role_index = ROLE_ORDER.index(role) if role in ROLE_ORDER else len(ROLE_ORDER)
    return layer, role_index, name


def load_hashes(checkpoint: Path) -> dict:
    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing safetensors index: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map", {})
    tensors: dict[str, str] = {}
    for key, shard_name in sorted(weight_map.items()):
        shard = checkpoint / shard_name
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            tensors[key] = tensor_sha256(handle.get_tensor(key))
    modules: dict[str, dict[str, str]] = {}
    for key, digest in tensors.items():
        name, field = module_name(key)
        if name is not None and field is not None:
            modules.setdefault(name, {})[field] = digest
    result = {
        "checkpoint": str(checkpoint),
        "model_index_sha256": sha256_file(index_path),
        "quantize_config_sha256": sha256_file(checkpoint / "quantize_config.json")
        if (checkpoint / "quantize_config.json").is_file()
        else None,
        "run_manifest_sha256": sha256_file(checkpoint / "qvq_quantize_run.json")
        if (checkpoint / "qvq_quantize_run.json").is_file()
        else None,
        "tensor_count": len(tensors),
        "module_count": len(modules),
        "tensors": tensors,
        "modules": modules,
    }
    run_path = checkpoint / "qvq_quantize_run.json"
    if run_path.is_file():
        run = json.loads(run_path.read_text(encoding="utf-8"))
        result["run"] = {
            "commit": run.get("commit"),
            "model": run.get("model"),
            "device_name": run.get("device_name"),
            "datasets": run.get("datasets"),
            "quant_log_rows": run.get("quant_log_rows"),
        }
    return result


def compare(left: dict, right: dict) -> dict:
    keys = sorted(set(left["tensors"]) | set(right["tensors"]))
    differing = [key for key in keys if left["tensors"].get(key) != right["tensors"].get(key)]
    module_names = sorted(set(left["modules"]) | set(right["modules"]), key=module_sort_key)
    differing_modules = [
        name for name in module_names if left["modules"].get(name) != right["modules"].get(name)
    ]
    first_module = differing_modules[0] if differing_modules else None
    first_module_fields = []
    if first_module is not None:
        fields = set(left["modules"].get(first_module, {})) | set(right["modules"].get(first_module, {}))
        first_module_fields = sorted(
            field
            for field in fields
            if left["modules"].get(first_module, {}).get(field)
            != right["modules"].get(first_module, {}).get(field)
        )
    return {
        "index_equal": left["model_index_sha256"] == right["model_index_sha256"],
        "tensor_key_count": len(keys),
        "differing_tensor_key_count": len(differing),
        "differing_tensor_keys": differing,
        "module_count": len(module_names),
        "differing_module_count": len(differing_modules),
        "first_logical_module_difference": first_module,
        "first_logical_module_differing_fields": first_module_fields,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    left = load_hashes(args.checkpoint.expanduser().resolve())
    payload = {"left": left}
    if args.compare is not None:
        right = load_hashes(args.compare.expanduser().resolve())
        payload["right"] = right
        payload["comparison"] = compare(left, right)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.expanduser().resolve().write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
