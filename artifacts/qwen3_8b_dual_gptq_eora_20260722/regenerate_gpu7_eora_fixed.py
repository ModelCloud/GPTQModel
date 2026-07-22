#!/usr/bin/env python3
"""Regenerate and validate the GPU 7 EoRA adapter with stable covariance inversion."""

# ruff: noqa: E402

from __future__ import annotations

import gc
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("GPTQ_TORCH_TRITON_DEQUANT", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault(
    "PYTORCH_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import datasets
import gptqmodel
import safetensors
import torch
import transformers
from datasets import load_dataset
from safetensors import safe_open
from transformers import AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.adapter.adapter import AdapterCache, Lora
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.utils.linalg_warmup import run_torch_linalg_warmup
from gptqmodel.utils.threadx import DeviceThreadPool, WarmUpCtx, WarmupTask
from gptqmodel.utils.torch import torch_empty_cache


BASE_MODEL = Path("/monster/data/model/Qwen3-8B")
MODEL = Path("/monster/data/model/Qwen3-8B-GPTQ-3bit-g64-EoRA-r128-cal512")
OLD_ADAPTER = MODEL / "eora-rank128"
FIXED_ADAPTER = MODEL / "eora-rank128-eighfix"
CALIBRATION_DATASET = Path("/monster/data/model/dataset/nm-calibration")
OUTPUT = MODEL / "eora_eighfix_validation.json"
EXPECTED_UUID = "724ea08e-67c3-c0ce-29bb-e6c48e7dde28"
EXPECTED_PCI_BUS = "00000000:E4:00.0"
PHYSICAL_GPU = 7
TARGET_MODULE = "model.layers.2.mlp.down_proj"
PROMPT = "The capital city of France is"
SEED = 898


def utc_now() -> str:
    """Return a timezone-aware timestamp for the validation artifact."""

    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    """Atomically persist the validation artifact."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def assert_gpu() -> dict[str, Any]:
    """Require the requested PCI-ordered GPU before allocating model state."""

    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID" or torch.cuda.device_count() != 1:
        raise RuntimeError("Validation requires exactly one PCI-ordered visible GPU")
    properties = torch.cuda.get_device_properties(0)
    actual_uuid = str(properties.uuid).removeprefix("GPU-").lower()
    actual_bus = int(properties.pci_bus_id)
    expected_bus = int(EXPECTED_PCI_BUS.split(":")[-2], 16)
    if actual_uuid != EXPECTED_UUID or actual_bus != expected_bus:
        raise RuntimeError(
            f"Expected GPU {PHYSICAL_GPU} uuid={EXPECTED_UUID}, bus={expected_bus:#x}; "
            f"found uuid={actual_uuid}, bus={actual_bus:#x}"
        )
    return {
        "physical_index_pci_order": PHYSICAL_GPU,
        "process_cuda_index": 0,
        "pci_bus_id": EXPECTED_PCI_BUS,
        "uuid": f"GPU-{actual_uuid}",
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "sm_count": properties.multi_processor_count,
        "memory_bytes": properties.total_memory,
    }


def configure_runtime() -> dict[str, Any]:
    """Bound CPU workers and initialize deterministic per-device linalg workers."""

    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise

    if getattr(gptqmodel, "_DEVICE_THREAD_POOL", None) is not None:
        raise RuntimeError("GPTQModel device thread pool initialized before validation setup")
    worker_counts = {
        "cuda:per": 1,
        "xpu:per": 1,
        "npu:per": 1,
        "mps": 1,
        "cpu": 1,
        "model_loader:cpu": 1,
    }
    gptqmodel._DEVICE_THREAD_POOL = DeviceThreadPool(
        inference_mode=True,
        warmups={
            "cuda": WarmupTask(run_torch_linalg_warmup, scope=WarmUpCtx.THREAD_AND_DEVICE),
            "xpu": WarmupTask(run_torch_linalg_warmup, scope=WarmUpCtx.THREAD_AND_DEVICE),
            "mps": WarmupTask(run_torch_linalg_warmup, scope=WarmUpCtx.THREAD_AND_DEVICE),
            "cpu": WarmupTask(run_torch_linalg_warmup, scope=WarmUpCtx.THREAD_AND_DEVICE),
        },
        workers=worker_counts,
        empty_cache_every_n=512,
    )
    return {
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "worker_counts": worker_counts,
    }


def load_calibration() -> Any:
    """Load the exact first 512 calibration rows used by the original checkpoint."""

    dataset = load_dataset(path=str(CALIBRATION_DATASET), name="LLM", split="train")
    if len(dataset) < 512:
        raise RuntimeError(f"Calibration dataset only has {len(dataset)} rows")
    return dataset.select(range(512))


def tensor_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """Measure bounded numerical drift between two tensors."""

    candidate_f = candidate.detach().float()
    reference_f = reference.detach().float()
    error = candidate_f - reference_f
    denominator = candidate_f.reshape(-1).norm() * reference_f.reshape(-1).norm()
    cosine = (
        (candidate_f.reshape(-1) @ reference_f.reshape(-1)) / denominator
        if denominator > 0
        else torch.tensor(float("nan"))
    )
    return {
        "shape": list(candidate.shape),
        "all_finite": bool(torch.isfinite(candidate_f).all().item()),
        "candidate_norm": candidate_f.norm().item(),
        "reference_norm": reference_f.norm().item(),
        "mae": error.abs().mean().item(),
        "rmse": error.square().mean().sqrt().item(),
        "max_abs_error": error.abs().max().item(),
        "cosine_similarity": cosine.item(),
    }


def adapter_summary(path: Path) -> dict[str, Any]:
    """Summarize adapter factors and the formerly catastrophic layer-2 correction."""

    weights_path = path / "adapter_model.safetensors"
    if not weights_path.is_file():
        raise RuntimeError(f"Missing adapter weights: {weights_path}")

    factor_records = []
    target_tensors: dict[str, torch.Tensor] = {}
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            tensor_f = tensor.float()
            factor_records.append(
                {
                    "key": key,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "all_finite": bool(torch.isfinite(tensor_f).all().item()),
                    "max_abs": tensor_f.abs().max().item(),
                    "norm": tensor_f.norm().item(),
                }
            )
            if key.endswith(f"{TARGET_MODULE}.lora_A.weight"):
                target_tensors["A"] = tensor_f
            elif key.endswith(f"{TARGET_MODULE}.lora_B.weight"):
                target_tensors["B"] = tensor_f

    if set(target_tensors) != {"A", "B"}:
        raise RuntimeError(f"Missing target adapter tensors in {weights_path}: {sorted(target_tensors)}")
    correction = target_tensors["B"] @ target_tensors["A"]
    target = {
        "module": TARGET_MODULE,
        "lora_A": next(record for record in factor_records if record["key"].endswith(f"{TARGET_MODULE}.lora_A.weight")),
        "lora_B": next(record for record in factor_records if record["key"].endswith(f"{TARGET_MODULE}.lora_B.weight")),
        "correction_all_finite": bool(torch.isfinite(correction).all().item()),
        "correction_max_abs": correction.abs().max().item(),
        "correction_norm": correction.norm().item(),
    }
    return {
        "path": str(path),
        "weights_bytes": weights_path.stat().st_size,
        "tensor_count": len(factor_records),
        "all_finite": all(record["all_finite"] for record in factor_records),
        "maximum_factor_abs": max(record["max_abs"] for record in factor_records),
        "maximum_factor_norm": max(record["norm"] for record in factor_records),
        "target": target,
    }


def model_device(model: Any) -> torch.device:
    """Resolve the materialized device for one loaded model."""

    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("Could not resolve model device")


def run_candidate(model: Any, encoded: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Capture logits, hidden states, generation, and the target adapter update."""

    modules = dict(model.named_modules())
    target = modules.get(TARGET_MODULE)
    if not isinstance(target, TorchLinear) or target.adapter is None:
        raise RuntimeError(f"Missing TorchLinear EoRA target: {TARGET_MODULE}")

    captured: dict[str, torch.Tensor] = {}

    def capture_adapter_input(module: TorchLinear, inputs: tuple[torch.Tensor, ...]) -> None:
        x = inputs[0].detach()
        lora_a, lora_b = module.adapter._forward_lora_tensors(x)
        captured["adapter_output"] = ((x.float() @ lora_a.float()) @ lora_b.float()).cpu()

    handle = target.register_forward_pre_hook(capture_adapter_input)
    with torch.inference_mode():
        output = model(
            **encoded,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        generated = model.generate(**encoded, max_new_tokens=24, do_sample=False)
    handle.remove()

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Loaded model has no tokenizer")
    continuation = tokenizer.decode(
        generated[0, encoded["input_ids"].shape[1] :].detach().cpu(),
        skip_special_tokens=True,
    )
    adapter_output = captured.get("adapter_output")
    if adapter_output is None:
        raise RuntimeError("Target adapter hook did not run")
    return {
        "logits": output.logits.detach().float().cpu(),
        "hidden_states": [hidden.detach().float().cpu() for hidden in output.hidden_states],
        "adapter_output": adapter_output,
        "continuation": continuation,
    }


def validate_runtime() -> dict[str, Any]:
    """Compare fixed EoRA, adapter-disabled INT3, and dense BF16 on one prompt."""

    AdapterCache.reset()
    fixed = Lora(rank=128, path=str(FIXED_ADAPTER))
    wrapper = GPTQModel.load(
        model_id_or_path=str(MODEL),
        backend=BACKEND.GPTQ_TORCH,
        adapter=fixed,
        device="cuda:0",
    )
    model = wrapper.model
    tokenizer = wrapper.tokenizer
    device = model_device(model)
    encoded = {key: value.to(device) for key, value in tokenizer(PROMPT, return_tensors="pt").items()}

    fixed_result = run_candidate(model, encoded)
    quant_modules = [module for module in model.modules() if isinstance(module, TorchLinear)]
    saved_adapters = [module.adapter for module in quant_modules]
    for module in quant_modules:
        module.adapter = None
    with torch.inference_mode():
        base_output = model(
            **encoded,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    base_logits = base_output.logits.detach().float().cpu()
    base_hidden = [hidden.detach().float().cpu() for hidden in base_output.hidden_states]
    for module, adapter in zip(quant_modules, saved_adapters):
        module.adapter = adapter

    attention = "flash_attention_2" if is_flash_attn_2_available() else "eager"
    dense = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL),
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation=attention,
    )
    dense.eval()
    dense_encoded = {key: value.to(model_device(dense)) for key, value in encoded.items()}
    with torch.inference_mode():
        dense_output = dense(
            **dense_encoded,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    dense_logits = dense_output.logits.detach().float().cpu()
    dense_hidden = [hidden.detach().float().cpu() for hidden in dense_output.hidden_states]

    fixed_last = fixed_result["logits"][:, -1, :]
    base_last = base_logits[:, -1, :]
    dense_last = dense_logits[:, -1, :]
    fixed_vs_dense = tensor_metrics(fixed_last, dense_last)
    fixed_vs_dense.update(
        {
            "candidate_top1_token_id": int(fixed_last.argmax(dim=-1).item()),
            "reference_top1_token_id": int(dense_last.argmax(dim=-1).item()),
        }
    )
    result = {
        "prompt": PROMPT,
        "quant_linear_count": len(quant_modules),
        "fixed_continuation": fixed_result["continuation"],
        "target_adapter_output": {
            "shape": list(fixed_result["adapter_output"].shape),
            "all_finite": bool(torch.isfinite(fixed_result["adapter_output"]).all().item()),
            "norm": fixed_result["adapter_output"].norm().item(),
            "max_abs": fixed_result["adapter_output"].abs().max().item(),
        },
        "layer2_hidden_fixed_vs_dense": tensor_metrics(fixed_result["hidden_states"][3], dense_hidden[3]),
        "layer2_hidden_base_vs_dense": tensor_metrics(base_hidden[3], dense_hidden[3]),
        "last_token_fixed_vs_dense": fixed_vs_dense,
        "last_token_base_vs_dense": tensor_metrics(base_last, dense_last),
        "last_token_fixed_vs_base": tensor_metrics(fixed_last, base_last),
    }

    del dense, dense_output, dense_logits, dense_hidden
    del wrapper, model, fixed_result, base_output, base_logits, base_hidden
    torch_empty_cache()
    return result


def main() -> None:
    """Generate the fixed adapter once and emit factor/runtime comparisons."""

    started = time.perf_counter()
    manifest: dict[str, Any] = {
        "started_at": utc_now(),
        "gpu": assert_gpu(),
        "runtime": configure_runtime(),
        "base_model": str(BASE_MODEL),
        "quantized_model": str(MODEL),
        "old_adapter": str(OLD_ADAPTER),
        "fixed_adapter": str(FIXED_ADAPTER),
        "calibration": {
            "path": str(CALIBRATION_DATASET),
            "name": "LLM",
            "rows": 512,
            "selection": "first 512 rows",
            "sort": "desc",
            "concat_size": None,
            "batch_size": 1,
        },
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "gptqmodel": getattr(gptqmodel, "__version__", None),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "safetensors": safetensors.__version__,
        },
    }

    weights_path = FIXED_ADAPTER / "adapter_model.safetensors"
    config_path = FIXED_ADAPTER / "adapter_config.json"
    if weights_path.exists() != config_path.exists():
        raise RuntimeError(f"Refusing to reuse partial fixed adapter at {FIXED_ADAPTER}")

    if not weights_path.is_file():
        calibration = load_calibration()
        generation_started = time.perf_counter()
        GPTQModel.adapter.generate(
            adapter=Lora(rank=128, path=str(FIXED_ADAPTER)),
            model_id_or_path=str(BASE_MODEL),
            quantized_model_id_or_path=str(MODEL),
            calibration_dataset=calibration,
            calibration_dataset_concat_size=None,
            calibration_dataset_sort="desc",
            batch_size=1,
            dtype="auto",
        )
        manifest["adapter_generation_wall_s"] = time.perf_counter() - generation_started
        del calibration
        gc.collect()
        torch_empty_cache()
    else:
        manifest["adapter_reused"] = True

    manifest["old_adapter_summary"] = adapter_summary(OLD_ADAPTER)
    manifest["fixed_adapter_summary"] = adapter_summary(FIXED_ADAPTER)
    manifest["runtime_validation"] = validate_runtime()
    manifest["completed_at"] = utc_now()
    manifest["total_wall_s"] = time.perf_counter() - started
    write_json(OUTPUT, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
