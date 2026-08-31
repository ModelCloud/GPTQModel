#!/usr/bin/env python3
"""Measure packed-state and reconstructed-weight distance between QVQ seeds."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.quantization.qvq import (
    QVQ_V2B2_P32_LR_TILE_COLS,
    QVQ_V2B2_P32_LR_TILE_ROWS,
    rht_reconstruct_weight,
    unpack_qvq_bank_ids,
    unpack_qvq_binary_bank_ids,
    unpack_trellis_states,
)

PACKED_FIELDS = ("bank_ids", "bank_alt_id", "trellis", "SU", "SV")


def _layer_index(name: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.", name)
    return None if match is None else int(match.group(1))


def _difference(left: torch.Tensor | None, right: torch.Tensor | None) -> dict:
    if left is None or right is None:
        equal = left is None and right is None
        return {
            "compatible": equal,
            "elements": 0,
            "differing_elements": 0 if equal else None,
            "disagreement_fraction": 0.0 if equal else None,
        }
    if left.shape != right.shape:
        return {
            "compatible": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "elements": None,
            "differing_elements": None,
            "disagreement_fraction": None,
        }
    left = left.detach().to(device="cpu")
    right = right.detach().to(device="cpu")
    differing = int(torch.count_nonzero(left != right).item())
    elements = int(left.numel())
    return {
        "compatible": True,
        "elements": elements,
        "differing_elements": differing,
        "disagreement_fraction": differing / elements if elements else 0.0,
    }


def _bank_selectors(module: QVQLinear) -> torch.Tensor | None:
    if module.bank_ids is None:
        return None
    if module.v2b2_p32_lr:
        tile_count = (module.in_features // QVQ_V2B2_P32_LR_TILE_ROWS) * (
            module.out_features // QVQ_V2B2_P32_LR_TILE_COLS
        )
    else:
        tile_count = (module.in_features // 16) * (module.out_features // 16)
    if module.v2b2_p32 or module.v2b2_p32_lr:
        return unpack_qvq_binary_bank_ids(module.bank_ids, tile_count * 8)
    if module.v2b4_p64:
        return unpack_qvq_bank_ids(module.bank_ids, tile_count * 4)
    return unpack_qvq_bank_ids(module.bank_ids, tile_count)


def _trellis_states(module: QVQLinear) -> torch.Tensor:
    return unpack_trellis_states(
        module.trellis,
        bits=module.bits,
        vector_size=module.vector_size,
        trellis_window=module.trellis_window,
    )


def _reconstructed_weight(module: QVQLinear, device: torch.device) -> torch.Tensor:
    inner = module.get_inner_weight_tensor(dtype=torch.float32).to(device=device)
    return rht_reconstruct_weight(
        inner,
        module.SU.to(device=device, dtype=torch.float32),
        module.SV.to(device=device, dtype=torch.float32),
    )


def _load(path: Path, device: str):
    return GPTQModel.load(
        str(path),
        backend=BACKEND.QVQ,
        dtype=torch.float16,
        device_map={"": device},
        attn_implementation="eager",
    )


def compare(left_path: Path, right_path: Path, *, device: str) -> dict:
    left_model = _load(left_path, device)
    right_model = _load(right_path, device)
    left_modules = {
        name: module
        for name, module in left_model.model.named_modules()
        if isinstance(module, QVQLinear)
    }
    right_modules = {
        name: module
        for name, module in right_model.model.named_modules()
        if isinstance(module, QVQLinear)
    }
    names = sorted(set(left_modules) | set(right_modules))
    if set(left_modules) != set(right_modules):
        raise ValueError("checkpoints do not contain the same QVQ module names")

    modules = {}
    packed_field_count = 0
    differing_packed_field_count = 0
    packed_elements = 0
    differing_packed_elements = 0
    layer_differing_elements: dict[int, int] = {}
    device_obj = torch.device(device)
    for name in names:
        left = left_modules[name]
        right = right_modules[name]
        fields = {}
        for field in PACKED_FIELDS:
            stats = _difference(getattr(left, field, None), getattr(right, field, None))
            fields[field] = stats
            if stats["compatible"] and stats["elements"]:
                packed_field_count += 1
                packed_elements += stats["elements"]
                if stats["differing_elements"]:
                    differing_packed_field_count += 1
                    differing_packed_elements += stats["differing_elements"]
                    layer = _layer_index(name)
                    if layer is not None:
                        layer_differing_elements[layer] = (
                            layer_differing_elements.get(layer, 0)
                            + stats["differing_elements"]
                        )

        bank = _difference(_bank_selectors(left), _bank_selectors(right))
        trellis = _difference(_trellis_states(left), _trellis_states(right))
        left_weight = _reconstructed_weight(left, device_obj)
        right_weight = _reconstructed_weight(right, device_obj)
        delta = torch.linalg.vector_norm(left_weight - right_weight).item()
        denominator = torch.linalg.vector_norm(left_weight).item()
        reconstructed_relative_l2 = delta / denominator if denominator else 0.0
        modules[name] = {
            "layer": _layer_index(name),
            "packed_fields": fields,
            "bank_id_disagreement_fraction": bank["disagreement_fraction"],
            "trellis_disagreement_fraction": trellis["disagreement_fraction"],
            "reconstructed_weight_relative_l2": reconstructed_relative_l2,
        }
        del left_weight, right_weight

    differing_modules = sum(
        any(
            field["differing_elements"]
            for field in module["packed_fields"].values()
            if field["compatible"]
        )
        for module in modules.values()
    )
    return {
        "left": str(left_path.resolve()),
        "right": str(right_path.resolve()),
        "module_count": len(modules),
        "differing_module_count": differing_modules,
        "differing_module_fraction": differing_modules / len(modules) if modules else 0.0,
        "packed_field_tensor_count": packed_field_count,
        "differing_packed_field_tensor_count": differing_packed_field_count,
        "differing_packed_field_tensor_fraction": (
            differing_packed_field_count / packed_field_count if packed_field_count else 0.0
        ),
        "packed_elements": packed_elements,
        "differing_packed_elements": differing_packed_elements,
        "differing_packed_element_fraction": (
            differing_packed_elements / packed_elements if packed_elements else 0.0
        ),
        "layer_differing_element_share": {
            str(layer): count / differing_packed_elements
            for layer, count in sorted(layer_differing_elements.items())
        }
        if differing_packed_elements
        else {},
        "modules": modules,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = compare(args.left, args.right, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "modules"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
