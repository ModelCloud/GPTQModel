#!/usr/bin/env python3
"""Compare eager Torch, TriLin, fused TriLin+EoRA, and dense Qwen3 behavior."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from functools import wraps
from pathlib import Path
from typing import Any


os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("GPTQ_TORCH_TRITON_DEQUANT", "0")
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import transformers
from transformers import AutoModelForCausalLM

import gptqmodel
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.adapter.adapter import Lora
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.utils import trilin as trilin_utils
from gptqmodel.utils.torch import torch_empty_cache


PROMPT = "The capital city of France is"
SEED = 898


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--expected-pci-bus", required=True)
    parser.add_argument("--expected-uuid", required=True)
    args = parser.parse_args()
    if (args.adapter is None) != (args.rank is None):
        parser.error("--adapter and --rank must be supplied together")
    return args


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def assert_gpu(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID" or torch.cuda.device_count() != 1:
        raise RuntimeError("Backend comparison requires exactly one PCI-ordered visible GPU")
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
    if candidate.shape != reference.shape:
        return {
            "shape_matches": False,
            "candidate_shape": list(candidate.shape),
            "reference_shape": list(reference.shape),
        }
    candidate_f = candidate.detach().float()
    reference_f = reference.detach().float()
    delta = candidate_f - reference_f
    candidate_flat = candidate_f.reshape(1, -1)
    reference_flat = reference_f.reshape(1, -1)
    return {
        "shape_matches": True,
        "shape": list(candidate.shape),
        "all_finite": bool(torch.isfinite(candidate_f).all().item()),
        "mae": delta.abs().mean().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "max_abs_error": delta.abs().max().item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            candidate_flat, reference_flat, dim=-1
        ).item(),
    }


def adapter_for(args: argparse.Namespace) -> Lora | None:
    return Lora(rank=args.rank, path=str(args.adapter)) if args.adapter is not None else None


def requested_dtype(args: argparse.Namespace) -> str | torch.dtype:
    return "auto" if args.dtype == "auto" else getattr(torch, args.dtype)


def module_contract(model: Any) -> dict[str, Any]:
    modules = [module for module in model.model.modules() if isinstance(module, BaseQuantLinear)]
    class_counts: dict[str, int] = {}
    scale_dtype_counts: dict[str, int] = {}
    for module in modules:
        class_name = type(module).__name__
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        scale_dtype = str(module.scales.dtype)
        scale_dtype_counts[scale_dtype] = scale_dtype_counts.get(scale_dtype, 0) + 1
    return {
        "module_count": len(modules),
        "class_counts": class_counts,
        "scale_dtype_counts": scale_dtype_counts,
        "active_adapter_count": sum(getattr(module, "adapter", None) is not None for module in modules),
        "trilin_native_count": sum(bool(getattr(module, "_trilin_native_3bit", False)) for module in modules),
        "trilin_eora_workspace_count": sum(hasattr(module, "_trilin_eora_workspace") for module in modules),
    }


def run_generation(
    model: Any,
    tokenizer: Any,
    encoded: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    with torch.inference_mode():
        logits = model(**encoded, use_cache=False).logits.detach().float().cpu()
        generated = model.generate(
            **encoded,
            max_new_tokens=24,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    generated_ids = generated.sequences.detach().cpu()
    scores = torch.stack([score.detach().float().cpu() for score in generated.scores], dim=0)
    prompt_tokens = encoded["input_ids"].shape[1]
    continuation_ids = generated_ids[:, prompt_tokens:]
    summary = {
        "prefill_logits_shape": list(logits.shape),
        "decode_scores_shape": list(scores.shape),
        "generated_token_ids": continuation_ids[0].tolist(),
        "generated_continuation": tokenizer.decode(continuation_ids[0], skip_special_tokens=True),
        "prefill_top1_token_id": int(logits[:, -1, :].argmax(dim=-1).item()),
        "all_finite": bool(torch.isfinite(logits).all().item() and torch.isfinite(scores).all().item()),
    }
    return summary, {
        "prefill_logits": logits,
        "decode_scores": scores,
        "generated_ids": continuation_ids,
    }


def compare_runs(candidate: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "prefill_logits": tensor_metrics(candidate["prefill_logits"], reference["prefill_logits"]),
        "last_prefill_token": tensor_metrics(
            candidate["prefill_logits"][:, -1, :], reference["prefill_logits"][:, -1, :]
        ),
        "decode_scores": tensor_metrics(candidate["decode_scores"], reference["decode_scores"]),
        "generated_ids_equal": bool(torch.equal(candidate["generated_ids"], reference["generated_ids"])),
    }


def run_trilin_variant(
    model: Any,
    tokenizer: Any,
    encoded: dict[str, torch.Tensor],
    *,
    fused_eora: bool,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    os.environ["GPTQMODEL_TRILIN_EORA"] = "1" if fused_eora else "0"
    calls = {"trilin_matmul": 0, "trilin_matmul_eora": 0}
    originals = {name: getattr(trilin_utils, name) for name in calls}

    for name, original in originals.items():
        def counted(*call_args, __name=name, __original=original, **call_kwargs):
            calls[__name] += 1
            return __original(*call_args, **call_kwargs)

        setattr(trilin_utils, name, wraps(original)(counted))
    try:
        summary, tensors = run_generation(model, tokenizer, encoded)
    finally:
        for name, original in originals.items():
            setattr(trilin_utils, name, original)
    summary["fused_eora_requested"] = fused_eora
    summary["native_call_counts"] = calls
    return summary, tensors


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    gpu = assert_gpu(args)

    print("Loading eager GPTQ_TORCH reference", flush=True)
    torch_model = GPTQModel.load(
        str(args.model),
        backend=BACKEND.GPTQ_TORCH,
        adapter=adapter_for(args),
        dtype=requested_dtype(args),
        device_map={"": "cuda:0"},
        attn_implementation="flash_attention_2",
    )
    torch_model.eval()
    tokenizer = torch_model.tokenizer
    encoded_cpu = tokenizer(PROMPT, return_tensors="pt")
    encoded = {key: value.to("cuda:0") for key, value in encoded_cpu.items()}
    torch_summary, torch_tensors = run_generation(torch_model, tokenizer, encoded)
    torch_contract = module_contract(torch_model)
    del torch_model
    torch_empty_cache()

    print("Loading GPTQ_TRITON/TriLin candidate", flush=True)
    trilin_model = GPTQModel.load(
        str(args.model),
        backend=BACKEND.GPTQ_TRITON,
        adapter=adapter_for(args),
        dtype=requested_dtype(args),
        device_map={"": "cuda:0"},
        attn_implementation="flash_attention_2",
    )
    trilin_model.eval()
    trilin_contract = module_contract(trilin_model)
    trilin_unfused_summary, trilin_unfused_tensors = run_trilin_variant(
        trilin_model,
        trilin_model.tokenizer,
        encoded,
        fused_eora=False,
    )
    trilin_fused_summary, trilin_fused_tensors = run_trilin_variant(
        trilin_model,
        trilin_model.tokenizer,
        encoded,
        fused_eora=True,
    )
    del trilin_model
    torch_empty_cache()

    print("Loading dense BF16 reference", flush=True)
    dense = AutoModelForCausalLM.from_pretrained(
        str(args.base),
        dtype=requested_dtype(args) if args.dtype != "auto" else torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="flash_attention_2",
    )
    dense.eval()
    dense_summary, dense_tensors = run_generation(dense, tokenizer, encoded)
    del dense
    torch_empty_cache()

    payload = {
        "model": str(args.model),
        "base_model": str(args.base),
        "adapter": str(args.adapter) if args.adapter is not None else None,
        "rank": args.rank,
        "dtype": args.dtype,
        "prompt": PROMPT,
        "input_ids": encoded_cpu["input_ids"][0].tolist(),
        "gpu": gpu,
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "gptqmodel": getattr(gptqmodel, "__version__", None),
        },
        "torch": {"contract": torch_contract, "generation": torch_summary},
        "trilin": {
            "contract": trilin_contract,
            "unfused_generation": trilin_unfused_summary,
            "fused_generation": trilin_fused_summary,
        },
        "dense": dense_summary,
        "comparisons": {
            "torch_vs_dense": compare_runs(torch_tensors, dense_tensors),
            "trilin_unfused_vs_torch": compare_runs(trilin_unfused_tensors, torch_tensors),
            "trilin_fused_vs_unfused": compare_runs(trilin_fused_tensors, trilin_unfused_tensors),
            "trilin_fused_vs_dense": compare_runs(trilin_fused_tensors, dense_tensors),
        },
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
