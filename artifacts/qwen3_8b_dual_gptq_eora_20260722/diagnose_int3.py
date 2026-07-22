#!/usr/bin/env python3
"""Isolate Qwen3 INT3 eager/Triton dequantization and EoRA effects."""

# ruff: noqa: E402

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("GPTQ_TORCH_TRITON_DEQUANT", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForCausalLM

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.adapter.adapter import Lora
from gptqmodel.nn_modules.qlinear import PackableQuantLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear


MODEL = Path("/monster/data/model/Qwen3-8B-GPTQ-3bit-g64-EoRA-r128-cal512")
BASE = Path("/monster/data/model/Qwen3-8B")
ADAPTER = MODEL / "eora-rank128"
OUTPUT = MODEL / "int3_diagnostic.json"
EXPECTED_UUID = "724ea08e-67c3-c0ce-29bb-e6c48e7dde28"
PROMPT = "The capital city of France is"


def tensor_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, object]:
    candidate = candidate.detach().float().cpu()
    reference = reference.detach().float().cpu()
    delta = candidate - reference
    candidate_flat = candidate.reshape(1, -1)
    reference_flat = reference.reshape(1, -1)
    cosine = torch.nn.functional.cosine_similarity(candidate_flat, reference_flat, dim=-1).item()
    return {
        "shape": list(candidate.shape),
        "all_finite": bool(torch.isfinite(candidate).all()),
        "mae": delta.abs().mean().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "max_abs_error": delta.abs().max().item(),
        "cosine_similarity": cosine,
    }


def tensor_summary(value: torch.Tensor) -> dict[str, float]:
    value = value.detach().float().cpu()
    return {
        "min": value.min().item(),
        "max": value.max().item(),
        "mean": value.mean().item(),
        "std": value.std().item(),
    }


def run_variant(model, encoded: dict[str, torch.Tensor], *, adapters: bool) -> tuple[torch.Tensor, dict[str, object]]:
    modules = [module for module in model.model.modules() if isinstance(module, TorchLinear)]
    saved_adapters = [module.adapter for module in modules]
    if not adapters:
        for module in modules:
            module.adapter = None

    try:
        with torch.inference_mode():
            logits = model(**encoded).logits.detach().float().cpu()
            generated = model.generate(**encoded, max_new_tokens=24, do_sample=False).detach().cpu()
        continuation = model.tokenizer.decode(
            generated[0, encoded["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        return logits, {
            "adapters_enabled": adapters,
            "generated_continuation": continuation,
            "all_finite": bool(torch.isfinite(logits).all()),
        }
    finally:
        if not adapters:
            for module, adapter in zip(modules, saved_adapters):
                module.adapter = adapter


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    props = torch.cuda.get_device_properties(0)
    actual_uuid = str(props.uuid).removeprefix("GPU-").lower()
    if torch.cuda.device_count() != 1 or actual_uuid != EXPECTED_UUID:
        raise RuntimeError(f"Expected only GPU 7 UUID {EXPECTED_UUID}, found {actual_uuid}")

    model = GPTQModel.load(
        str(MODEL),
        backend=BACKEND.GPTQ_TORCH,
        adapter=Lora(rank=128, path=str(ADAPTER)),
        dtype="auto",
        device_map={"": "cuda:0"},
        attn_implementation="flash_attention_2",
    )
    model.eval()
    modules = [(name, module) for name, module in model.model.named_modules() if isinstance(module, TorchLinear)]
    if len(modules) != 252:
        raise RuntimeError(f"Expected 252 TorchLinear modules, found {len(modules)}")
    if any(module._triton_dequant_enabled for _, module in modules):
        raise RuntimeError("Diagnostic requested eager Torch dequantization but Triton remained enabled")

    tokenizer = model.tokenizer
    encoded = {key: value.to("cuda:0") for key, value in tokenizer(PROMPT, return_tensors="pt").items()}
    eager_adapter_logits, eager_adapter = run_variant(model, encoded, adapters=True)
    eager_no_adapter_logits, eager_no_adapter = run_variant(model, encoded, adapters=False)

    first_name, first_module = modules[0]
    with torch.inference_mode():
        first_eager = PackableQuantLinear.dequantize_weight(first_module)
        first_triton = first_module._dequantize_weight_triton()
    first_dequant_comparison = tensor_metrics(first_triton, first_eager)

    dense = AutoModelForCausalLM.from_pretrained(
        str(BASE),
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="flash_attention_2",
    )
    dense.eval()
    with torch.inference_mode():
        dense_logits = dense(**encoded).logits.detach().float().cpu()

    dense_first_module = dense.get_submodule(first_name).weight.detach().T
    first_adapter = first_module.adapter
    first_effective = first_eager + torch.matmul(
        first_adapter.lora_A.float(),
        first_adapter.lora_B.float(),
    ).to(dtype=first_eager.dtype)
    first_dequant_comparison["eager_vs_dense"] = tensor_metrics(first_eager, dense_first_module)
    first_dequant_comparison["eager_plus_eora_vs_dense"] = tensor_metrics(first_effective, dense_first_module)
    first_dequant_comparison["eager_summary"] = tensor_summary(first_eager)
    first_dequant_comparison["eora_delta_summary"] = tensor_summary(first_effective - first_eager)
    first_dequant_comparison["dense_summary"] = tensor_summary(dense_first_module)

    dense_last = dense_logits[:, -1, :]
    eager_adapter_last = eager_adapter_logits[:, -1, :]
    eager_no_adapter_last = eager_no_adapter_logits[:, -1, :]
    eager_adapter["last_token_vs_dense"] = tensor_metrics(eager_adapter_last, dense_last)
    eager_no_adapter["last_token_vs_dense"] = tensor_metrics(eager_no_adapter_last, dense_last)
    eager_adapter["top1_token_id"] = int(eager_adapter_last.argmax(dim=-1).item())
    eager_no_adapter["top1_token_id"] = int(eager_no_adapter_last.argmax(dim=-1).item())

    payload = {
        "gpu": {
            "uuid": f"GPU-{actual_uuid}",
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
        },
        "model": str(MODEL),
        "prompt": PROMPT,
        "module_count": len(modules),
        "first_module": first_name,
        "first_module_triton_vs_eager": first_dequant_comparison,
        "eager_with_eora": eager_adapter,
        "eager_without_eora": eager_no_adapter,
        "dense_top1_token_id": int(dense_last.argmax(dim=-1).item()),
        "prior_default_triton_runtime": json.loads((MODEL / "runtime_smoke.json").read_text()),
        "prior_default_triton_dense_comparison": json.loads((MODEL / "dense_reference.json").read_text()),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
