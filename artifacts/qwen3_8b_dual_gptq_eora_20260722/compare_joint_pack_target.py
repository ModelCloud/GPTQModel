#!/usr/bin/env python3
"""Identify whether joint GPTQ+EoRA serialization packed base or adapted weights."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("GPTQ_TORCH_TRITON_DEQUANT", "0")
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from safetensors import safe_open

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear import PackableQuantLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.utils.model_dequant import unpack_cols, unpack_rows


MODULES = (
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-model", required=True, type=Path)
    parser.add_argument("--joint-model", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--expected-pci-bus", required=True)
    parser.add_argument("--expected-uuid", required=True)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def assert_gpu(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID" or torch.cuda.device_count() != 1:
        raise RuntimeError("Diagnostic requires exactly one PCI-ordered visible GPU")
    properties = torch.cuda.get_device_properties(0)
    actual_uuid = str(properties.uuid).removeprefix("GPU-").lower()
    expected_uuid = args.expected_uuid.removeprefix("GPU-").lower()
    actual_bus = int(properties.pci_bus_id)
    expected_bus = int(args.expected_pci_bus.split(":")[-2], 16)
    if actual_uuid != expected_uuid or actual_bus != expected_bus:
        raise RuntimeError(
            f"Expected physical GPU {args.physical_gpu} uuid={expected_uuid}, bus={expected_bus:#x}; "
            f"found uuid={actual_uuid}, bus={actual_bus:#x}"
        )
    return {
        "physical_index_pci_order": args.physical_gpu,
        "process_cuda_index": 0,
        "pci_bus_id": args.expected_pci_bus,
        "uuid": f"GPU-{actual_uuid}",
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "sm_count": properties.multi_processor_count,
        "memory_bytes": properties.total_memory,
    }


def adapter_key(name: str, side: str) -> str:
    return f"base_model.model.{name}.lora_{side}.weight"


def unpack_codes(module: TorchLinear) -> tuple[torch.Tensor, torch.Tensor]:
    codes = unpack_rows(module.qweight, module.bits)[: module.in_features, : module.out_features]
    zeros = unpack_cols(module.qzeros, module.bits)[:, : module.out_features]
    return codes.to(torch.int32), zeros.to(torch.int32)


def codes_for_weight(weight: torch.Tensor, module: TorchLinear, zeros: torch.Tensor) -> torch.Tensor:
    scales = module.scales.float()
    group_indices = module.g_idx.long()
    return (
        torch.round(weight.float() / scales[group_indices])
        .to(torch.int32)
        .add_(zeros[group_indices])
        .clamp_(0, module.maxq)
    )


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        left.float().reshape(1, -1),
        right.float().reshape(1, -1),
        dim=-1,
    ).item()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    gpu = assert_gpu(args)

    clean = GPTQModel.load(
        str(args.clean_model),
        backend=BACKEND.GPTQ_TORCH,
        dtype="auto",
        device_map={"": "cuda:0"},
        attn_implementation="flash_attention_2",
    )
    joint = GPTQModel.load(
        str(args.joint_model),
        backend=BACKEND.GPTQ_TORCH,
        dtype="auto",
        device_map={"": "cuda:0"},
        attn_implementation="flash_attention_2",
    )

    adapter_file = args.adapter / "adapter_model.safetensors"
    records = []
    with safe_open(adapter_file, framework="pt", device="cpu") as adapter:
        for name in MODULES:
            clean_module = clean.model.get_submodule(name)
            joint_module = joint.model.get_submodule(name)
            if not isinstance(clean_module, TorchLinear) or not isinstance(joint_module, TorchLinear):
                raise TypeError(f"Expected TorchLinear for {name}")

            clean_codes, clean_zeros = unpack_codes(clean_module)
            joint_codes, joint_zeros = unpack_codes(joint_module)
            clean_weight = PackableQuantLinear.dequantize_weight(clean_module)
            joint_weight = PackableQuantLinear.dequantize_weight(joint_module)

            a = adapter.get_tensor(adapter_key(name, "A")).to(device="cuda:0", dtype=torch.bfloat16)
            b = adapter.get_tensor(adapter_key(name, "B")).to(device="cuda:0", dtype=torch.bfloat16)
            adapted_weight = (clean_weight.T.to(torch.bfloat16) + b @ a).T

            predicted_base = codes_for_weight(clean_weight, joint_module, joint_zeros)
            predicted_adapted = codes_for_weight(adapted_weight, joint_module, joint_zeros)

            records.append(
                {
                    "name": name,
                    "clean_vs_joint_scales_equal": bool(torch.equal(clean_module.scales, joint_module.scales)),
                    "clean_vs_joint_scales_max_abs": (
                        clean_module.scales.float() - joint_module.scales.float()
                    ).abs().max().item(),
                    "clean_vs_joint_zeros_equal": bool(torch.equal(clean_zeros, joint_zeros)),
                    "clean_vs_joint_codes_agreement": clean_codes.eq(joint_codes).float().mean().item(),
                    "joint_vs_predicted_base_codes_agreement": (
                        joint_codes.eq(predicted_base).float().mean().item()
                    ),
                    "joint_vs_predicted_adapted_codes_agreement": (
                        joint_codes.eq(predicted_adapted).float().mean().item()
                    ),
                    "joint_weight_vs_clean_weight_cosine": cosine(joint_weight, clean_weight),
                    "joint_weight_vs_clean_plus_adapter_cosine": cosine(joint_weight, adapted_weight),
                    "clean_weight_vs_clean_plus_adapter_cosine": cosine(clean_weight, adapted_weight),
                }
            )

    payload = {
        "clean_model": str(args.clean_model),
        "joint_model": str(args.joint_model),
        "adapter": str(args.adapter),
        "gpu": gpu,
        "modules": records,
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
