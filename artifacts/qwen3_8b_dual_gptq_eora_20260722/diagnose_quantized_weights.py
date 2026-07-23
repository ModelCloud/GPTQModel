#!/usr/bin/env python3
"""Compare packed GPTQ weights and prompt behavior with dense BF16."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("GPTQ_TORCH_TRITON_DEQUANT", "0")
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear import PackableQuantLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.utils.model_dequant import unpack_cols, unpack_rows


BASE = Path("/monster/data/model/Qwen3-8B")
PROMPT = "The capital city of France is"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bits", required=True, type=int, choices=(2, 3, 4, 8))
    parser.add_argument("--group-size", required=True, type=int)
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


def tensor_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    candidate_f = candidate.detach().float()
    reference_f = reference.detach().float()
    delta = candidate_f - reference_f
    candidate_flat = candidate_f.reshape(-1)
    reference_flat = reference_f.reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(candidate_flat[None], reference_flat[None], dim=-1).item()
    return {
        "shape": list(candidate.shape),
        "all_finite": bool(torch.isfinite(candidate_f).all().item()),
        "mae": delta.abs().mean().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "max_abs_error": delta.abs().max().item(),
        "cosine_similarity": cosine,
        "candidate_norm": candidate_flat.norm().item(),
        "reference_norm": reference_flat.norm().item(),
    }


def tokenizer_record(quant_tokenizer: Any, base: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    direct = AutoTokenizer.from_pretrained(str(base))
    quant_encoded = quant_tokenizer(PROMPT, return_tensors="pt")
    direct_encoded = direct(PROMPT, return_tensors="pt")
    record = {
        "raw_prompt": PROMPT,
        "apply_chat_template": False,
        "quant_tokenizer_class": type(quant_tokenizer).__name__,
        "direct_tokenizer_class": type(direct).__name__,
        "quant_input_ids": quant_encoded["input_ids"][0].tolist(),
        "direct_input_ids": direct_encoded["input_ids"][0].tolist(),
        "input_ids_equal": bool(torch.equal(quant_encoded["input_ids"], direct_encoded["input_ids"])),
        "quant_attention_mask": quant_encoded["attention_mask"][0].tolist(),
        "direct_attention_mask": direct_encoded["attention_mask"][0].tolist(),
        "quant_special_tokens": quant_tokenizer.special_tokens_map,
        "direct_special_tokens": direct.special_tokens_map,
    }
    return record, quant_encoded


def compare_end_to_end(
    quant: Any,
    dense: Any,
    encoded_cpu: dict[str, torch.Tensor],
) -> dict[str, Any]:
    encoded = {key: value.to("cuda:0") for key, value in encoded_cpu.items()}
    with torch.inference_mode():
        quant_logits = quant(**encoded, use_cache=False).logits.detach().float()
        dense_logits = dense(**encoded, use_cache=False).logits.detach().float()
        generated = quant.generate(**encoded, max_new_tokens=24, do_sample=False).detach().cpu()
    continuation = quant.tokenizer.decode(
        generated[0, encoded["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    metrics = tensor_metrics(quant_logits[:, -1, :], dense_logits[:, -1, :])
    metrics.update(
        {
            "generated_continuation": continuation,
            "quant_top1_token_id": int(quant_logits[:, -1, :].argmax(dim=-1).item()),
            "dense_top1_token_id": int(dense_logits[:, -1, :].argmax(dim=-1).item()),
        }
    )
    return metrics


def compare_module(
    args: argparse.Namespace,
    name: str,
    module: TorchLinear,
    dense: Any,
) -> dict[str, Any]:
    if module.bits != args.bits or module.group_size != args.group_size:
        raise RuntimeError(
            f"Unexpected module contract for {name}: bits={module.bits}, group={module.group_size}, "
            f"qzero_format={module.qzero_format()}"
        )
    quant_values = unpack_rows(module.qweight, module.bits)[: module.in_features, : module.out_features]
    logical_zeros = unpack_cols(module.qzeros, module.bits)[:, : module.out_features]
    group_indices = module.g_idx.long()
    scales = module.scales.float()
    quant_values = quant_values.to(device=scales.device)
    logical_zeros = logical_zeros.to(device=scales.device)
    dense_weight = dense.get_submodule(name).weight.detach().T.float()

    with torch.inference_mode():
        eager = PackableQuantLinear.dequantize_weight(module).float()
    current_metrics = tensor_metrics(eager, dense_weight)
    reconstructed = scales[group_indices] * (quant_values - logical_zeros[group_indices])
    eager_equivalence = tensor_metrics(eager, reconstructed)

    direct_codes = torch.round(dense_weight / scales[group_indices]).to(torch.int32)
    direct_codes.add_(logical_zeros[group_indices]).clamp_(0, module.maxq)
    direct_with_saved_scales = scales[group_indices] * (direct_codes - logical_zeros[group_indices])
    direct_metrics = tensor_metrics(direct_with_saved_scales, dense_weight)
    code_agreement = direct_codes.eq(quant_values).float().mean().item()
    code_delta = quant_values.to(torch.int64) - direct_codes.to(torch.int64)
    code_delta_counts = torch.bincount(
        code_delta.reshape(-1) + module.maxq,
        minlength=2 * module.maxq + 1,
    ).tolist()
    boundary_repaired_codes = quant_values.clone()
    boundary_repaired_codes.masked_fill_((quant_values == 0) & (direct_codes == module.maxq), module.maxq)
    boundary_repaired_codes.masked_fill_((quant_values == module.maxq) & (direct_codes == 0), 0)
    boundary_repaired = scales[group_indices] * (boundary_repaired_codes - logical_zeros[group_indices])
    boundary_repaired_metrics = tensor_metrics(boundary_repaired, dense_weight)

    grouped_dense = dense_weight.reshape(-1, module.group_size, module.out_features)
    rtn_scales = grouped_dense.abs().amax(dim=1).mul_(2.0 / module.maxq).clamp_min_(torch.finfo(torch.float32).eps)
    symmetric_zero = (module.maxq + 1) // 2
    rtn_codes = torch.round(dense_weight / rtn_scales[group_indices]).to(torch.int32)
    rtn_codes.add_(symmetric_zero).clamp_(0, module.maxq)
    rtn_dequantized = rtn_scales[group_indices] * (rtn_codes - symmetric_zero)
    rtn_metrics = tensor_metrics(rtn_dequantized, dense_weight)

    result = {
        "name": name,
        "bits": module.bits,
        "group_size": module.group_size,
        "qzero_format": module.qzero_format(),
        "qweight_shape": list(module.qweight.shape),
        "qzeros_shape": list(module.qzeros.shape),
        "scales_shape": list(module.scales.shape),
        "scale_min": scales.min().item(),
        "scale_max": scales.max().item(),
        "scale_mean": scales.mean().item(),
        "logical_zero_value_counts": torch.bincount(
            logical_zeros.reshape(-1).to(torch.int64), minlength=module.maxq + 1
        ).tolist(),
        "packed_dequant_vs_dense": current_metrics,
        "direct_round_with_saved_scales_vs_dense": direct_metrics,
        "packed_vs_direct_round_code_agreement": code_agreement,
        "packed_minus_direct_code_delta_min": int(code_delta.min().item()),
        "packed_minus_direct_code_delta_max": int(code_delta.max().item()),
        "packed_minus_direct_code_delta_counts": {
            str(delta): count for delta, count in zip(range(-module.maxq, module.maxq + 1), code_delta_counts)
        },
        "boundary_unwrap_candidate_vs_dense": boundary_repaired_metrics,
        "independent_symmetric_rtn_vs_dense": rtn_metrics,
        "generic_eager_vs_manual_unpack": eager_equivalence,
    }
    del (
        quant_values,
        logical_zeros,
        group_indices,
        scales,
        dense_weight,
        eager,
        reconstructed,
        direct_codes,
        code_delta,
        boundary_repaired_codes,
        boundary_repaired,
        direct_with_saved_scales,
        grouped_dense,
        rtn_scales,
        rtn_codes,
        rtn_dequantized,
    )
    return result


def mean_metric(records: list[dict[str, Any]], section: str, metric: str) -> float:
    return sum(record[section][metric] for record in records) / len(records)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    gpu = assert_gpu(args)

    print(f"Loading {args.model} on {gpu['uuid']} with eager GPTQ_TORCH", flush=True)
    quant = GPTQModel.load(
        str(args.model),
        backend=BACKEND.GPTQ_TORCH,
        dtype="auto",
        device_map={"": "cuda:0"},
        attn_implementation="flash_attention_2",
    )
    quant.eval()
    modules = [(name, module) for name, module in quant.model.named_modules() if isinstance(module, TorchLinear)]
    if not modules or any(module._triton_dequant_enabled for _, module in modules):
        raise RuntimeError("Expected one or more eager-Torch quantized modules")

    layer_indices = sorted({int(name.split(".")[2]) for name, _ in modules if name.startswith("model.layers.")})
    selected_layers = tuple(dict.fromkeys([*layer_indices[:3], layer_indices[-1]]))

    print("Loading dense BF16 reference", flush=True)
    dense = AutoModelForCausalLM.from_pretrained(
        str(args.base),
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="flash_attention_2",
    )
    dense.eval()

    tokenizer, encoded = tokenizer_record(quant.tokenizer, args.base)
    end_to_end = compare_end_to_end(quant, dense, encoded)
    selected = [(name, module) for name, module in modules if int(name.split(".")[2]) in selected_layers]
    records = []
    for index, (name, module) in enumerate(selected, start=1):
        record = compare_module(args, name, module, dense)
        records.append(record)
        print(
            f"{index:02d}/{len(selected)} {name}: packed={record['packed_dequant_vs_dense']['cosine_similarity']:.6f}, "
            f"direct={record['direct_round_with_saved_scales_vs_dense']['cosine_similarity']:.6f}, "
            f"RTN={record['independent_symmetric_rtn_vs_dense']['cosine_similarity']:.6f}, "
            f"code_match={record['packed_vs_direct_round_code_agreement']:.4f}",
            flush=True,
        )

    summary = {
        "packed_dequant_cosine_mean": mean_metric(records, "packed_dequant_vs_dense", "cosine_similarity"),
        "direct_saved_scale_cosine_mean": mean_metric(
            records, "direct_round_with_saved_scales_vs_dense", "cosine_similarity"
        ),
        "independent_rtn_cosine_mean": mean_metric(
            records, "independent_symmetric_rtn_vs_dense", "cosine_similarity"
        ),
        "packed_vs_direct_code_agreement_mean": sum(
            record["packed_vs_direct_round_code_agreement"] for record in records
        ) / len(records),
    }
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu,
        "model": str(args.model),
        "base_model": str(args.base),
        "backend": "gptq_torch eager dequantization",
        "module_count": len(modules),
        "selected_layers": list(selected_layers),
        "selected_module_count": len(records),
        "tokenizer": tokenizer,
        "end_to_end": end_to_end,
        "summary": summary,
        "modules": records,
    }
    write_json(args.output, payload)
    print(json.dumps({"end_to_end": end_to_end, "summary": summary}, indent=2, sort_keys=True), flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
