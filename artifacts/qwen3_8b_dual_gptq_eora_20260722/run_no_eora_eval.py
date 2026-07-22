#!/usr/bin/env python3
"""Evaluate the saved GPU 6 GPTQ checkpoint with no EoRA adapter loaded."""

# ruff: noqa: E402

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault(
    "PYTORCH_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7",
)
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import evalution
import gptqmodel
import torch
import transformers
from transformers import AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.utils.torch import torch_empty_cache
from tests.eval import evaluate, get_eval_task_results


MODEL = Path("/monster/data/model/Qwen3-8B-GPTQ-4bit-g128-activation-GAR-EoRA-r128-cal512")
BASE_MODEL = Path("/monster/data/model/Qwen3-8B")
OUTPUT_DIR = MODEL / "evalution_no_eora"
EXPECTED_UUID = "737e2423-874a-23a4-1126-dfbe3e77c294"
EXPECTED_PCI_BUS = "00000000:DE:00.0"
SEED = 898
TASKS = ("arc_challenge", "gsm8k_platinum_cot")
BATCH_SIZE = 16


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if hasattr(value, "value"):
        return jsonable(value.value)
    return repr(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def assert_gpu() -> dict[str, Any]:
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible CUDA device, found {torch.cuda.device_count()}")

    properties = torch.cuda.get_device_properties(0)
    actual_uuid = str(properties.uuid).removeprefix("GPU-").lower()
    actual_bus = int(properties.pci_bus_id)
    expected_bus = int(EXPECTED_PCI_BUS.split(":")[-2], 16)
    if actual_uuid != EXPECTED_UUID or actual_bus != expected_bus:
        raise RuntimeError(
            f"GPU mismatch: expected uuid={EXPECTED_UUID}, bus={expected_bus:#x}; "
            f"found uuid={actual_uuid}, bus={actual_bus:#x}"
        )
    return {
        "physical_index_pci_order": 6,
        "process_cuda_index": 0,
        "pci_bus_id": EXPECTED_PCI_BUS,
        "uuid": f"GPU-{actual_uuid}",
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "sm_count": properties.multi_processor_count,
        "memory_bytes": properties.total_memory,
    }


def tensor_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    candidate = candidate.detach().float().cpu()
    reference = reference.detach().float().cpu()
    delta = candidate - reference
    return {
        "shape": list(candidate.shape),
        "all_finite": bool(torch.isfinite(candidate).all()),
        "mae": delta.abs().mean().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "max_abs_error": delta.abs().max().item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(candidate, reference, dim=-1).mean().item(),
        "candidate_top1_token_id": int(candidate.argmax(dim=-1).item()),
        "reference_top1_token_id": int(reference.argmax(dim=-1).item()),
        "top1_agrees": bool(candidate.argmax(dim=-1).item() == reference.argmax(dim=-1).item()),
    }


def validate_runtime(model: Any) -> tuple[dict[str, Any], torch.Tensor, dict[str, torch.Tensor]]:
    modules = [(name, module) for name, module in model.model.named_modules() if isinstance(module, BaseQuantLinear)]
    if len(modules) != 252:
        raise RuntimeError(f"Expected 252 quantized modules, found {len(modules)}")
    active_adapters = [name for name, module in modules if getattr(module, "adapter", None) is not None]
    if active_adapters:
        raise RuntimeError(f"No-EoRA run loaded adapters unexpectedly: {active_adapters[:5]}")

    tokenizer = model.tokenizer
    prompt = "The capital city of France is"
    encoded = {key: value.to("cuda:0") for key, value in tokenizer(prompt, return_tensors="pt").items()}
    with torch.inference_mode():
        logits = model(**encoded).logits.detach().float().cpu()
        generated = model.generate(**encoded, max_new_tokens=24, do_sample=False).detach().cpu()
    if not torch.isfinite(logits).all():
        raise RuntimeError("No-EoRA runtime smoke produced non-finite logits")

    continuation = tokenizer.decode(
        generated[0, encoded["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    result = {
        "adapter_mode": "disabled/not loaded",
        "quant_linear_module_count": len(modules),
        "active_adapter_count": 0,
        "quant_linear_class_counts": {},
        "logits_shape": list(logits.shape),
        "logits_all_finite": True,
        "prompt": prompt,
        "generated_continuation": continuation,
    }
    for _, module in modules:
        class_name = type(module).__name__
        result["quant_linear_class_counts"][class_name] = (
            result["quant_linear_class_counts"].get(class_name, 0) + 1
        )
    return result, logits, encoded


def compare_dense(quant_logits: torch.Tensor, encoded: dict[str, torch.Tensor]) -> dict[str, Any]:
    attention = "flash_attention_2" if is_flash_attn_2_available() else "eager"
    dense = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL),
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation=attention,
    )
    dense.eval()
    with torch.inference_mode():
        dense_logits = dense(**encoded).logits.detach().float().cpu()
    metrics = tensor_metrics(quant_logits[:, -1, :], dense_logits[:, -1, :])
    metrics.update(
        {
            "reference": "dense BF16 Qwen3-8B",
            "candidate": "GPTQ INT4 group-128 with no EoRA adapter",
            "scope": "last-token logits for runtime smoke prompt",
        }
    )
    del dense, dense_logits
    torch_empty_cache()
    return metrics


def run_tasks(model: Any) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    for task in TASKS:
        result_path = OUTPUT_DIR / f"{task}.json"
        if result_path.is_file():
            existing = json.loads(result_path.read_text())
            summaries[task] = existing["metrics"]
            print(f"Reusing completed no-EoRA task {task}", flush=True)
            continue

        print(f"Starting full no-EoRA Evalution task={task}, batch_size={BATCH_SIZE}", flush=True)
        started = time.perf_counter()
        result = evaluate(
            model_or_id_or_path=model,
            tasks=[task],
            batch_size=BATCH_SIZE,
            backend=BACKEND.GPTQ_TORCH,
            model_args={"device": "cuda:0", "seed": SEED, "random_seed": SEED},
            apply_chat_template=False,
            gen_kwargs="do_sample=false,temperature=0.0,top_p=1.0,top_k=50,max_new_tokens=256",
        )
        wall_s = time.perf_counter() - started
        metrics = get_eval_task_results(result).get(task, {})
        if not metrics:
            raise RuntimeError(f"Evalution returned no metrics for {task}: {result}")
        if any(isinstance(value, (int, float)) and not math.isfinite(float(value)) for value in metrics.values()):
            raise RuntimeError(f"Evalution returned non-finite metrics for {task}: {metrics}")
        write_json(
            result_path,
            {
                "task": task,
                "adapter_mode": "disabled/not loaded",
                "full_dataset": True,
                "max_rows": None,
                "batch_size": BATCH_SIZE,
                "apply_chat_template": False,
                "wall_s": wall_s,
                "metrics": metrics,
                "result": result,
            },
        )
        summaries[task] = metrics
        print(f"Completed no-EoRA {task}: {metrics}", flush=True)
        torch_empty_cache()
    return summaries


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    gpu = assert_gpu()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    write_json(
        MODEL / "no_eora_eval_manifest.json",
        {
            "started_at": utc_now(),
            "model": str(MODEL),
            "base_model": str(BASE_MODEL),
            "adapter_mode": "disabled/not loaded",
            "gpu": gpu,
            "versions": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "transformers": transformers.__version__,
                "gptqmodel": getattr(gptqmodel, "__version__", None),
                "evalution": getattr(evalution, "__version__", None),
            },
            "tasks": list(TASKS),
            "batch_size": BATCH_SIZE,
            "seed": SEED,
        },
    )

    attention = "flash_attention_2" if is_flash_attn_2_available() else "eager"
    print(f"Loading {MODEL} with GPTQ_TORCH and no adapter", flush=True)
    load_started = time.perf_counter()
    model = GPTQModel.load(
        str(MODEL),
        backend=BACKEND.GPTQ_TORCH,
        dtype="auto",
        device_map={"": "cuda:0"},
        attn_implementation=attention,
    )
    model.eval()
    runtime, quant_logits, encoded = validate_runtime(model)
    runtime["reload_wall_s"] = time.perf_counter() - load_started
    write_json(MODEL / "runtime_smoke_no_eora.json", runtime)
    dense_metrics = compare_dense(quant_logits, encoded)
    write_json(MODEL / "dense_reference_no_eora.json", dense_metrics)
    del quant_logits, encoded
    torch_empty_cache()

    tasks = run_tasks(model)
    write_json(
        MODEL / "evaluation_complete_no_eora.json",
        {
            "completed_at": utc_now(),
            "backend": "gptq_torch",
            "adapter_mode": "disabled/not loaded",
            "tasks": tasks,
        },
    )
    print(f"All no-EoRA validations completed: {tasks}", flush=True)


if __name__ == "__main__":
    main()
