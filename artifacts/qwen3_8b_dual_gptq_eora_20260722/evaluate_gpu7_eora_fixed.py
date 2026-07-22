#!/usr/bin/env python3
"""Evaluate the corrected GPU 7 EoRA adapter on ARC and GSM8K Platinum."""

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
os.environ.setdefault("GPTQ_TORCH_TRITON_DEQUANT", "0")
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
from transformers.utils import is_flash_attn_2_available

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.adapter.adapter import Lora
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.utils.torch import torch_empty_cache
from tests.eval import evaluate, get_eval_task_results


MODEL = Path("/monster/data/model/Qwen3-8B-GPTQ-3bit-g64-EoRA-r128-cal512")
ADAPTER = MODEL / "eora-rank128-eighfix"
OUTPUT_DIR = MODEL / "evalution_eora_eighfix"
EXPECTED_UUID = "724ea08e-67c3-c0ce-29bb-e6c48e7dde28"
EXPECTED_PCI_BUS = "00000000:E4:00.0"
PHYSICAL_GPU = 7
SEED = 898
TASKS = ("arc_challenge", "gsm8k_platinum_cot")
BATCH_SIZE = 16
EVAL_BACKEND = BACKEND.GPTQ_TORCH


def utc_now() -> str:
    """Return a timezone-aware timestamp for evaluation metadata."""

    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    """Convert Evalution result objects into JSON-compatible values."""

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
    """Atomically persist one evaluation artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def assert_gpu() -> dict[str, Any]:
    """Require the requested PCI-ordered GPU before model loading."""

    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID" or torch.cuda.device_count() != 1:
        raise RuntimeError("Evaluation requires exactly one PCI-ordered visible GPU")
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


def validate_runtime(model: Any) -> dict[str, Any]:
    """Verify adapter coverage and a finite deterministic smoke generation."""

    modules = [(name, module) for name, module in model.model.named_modules() if isinstance(module, BaseQuantLinear)]
    if len(modules) != 252:
        raise RuntimeError(f"Expected 252 quantized modules, found {len(modules)}")
    missing = [name for name, module in modules if getattr(module, "adapter", None) is None]
    ranks = sorted(
        {
            int(module.adapter.rank)
            for _, module in modules
            if getattr(module, "adapter", None) is not None
        }
    )
    if missing or ranks != [128]:
        raise RuntimeError(f"Fixed EoRA coverage failed: missing={missing[:5]}, ranks={ranks}")

    prepared_eora_modules = [
        name
        for name, module in modules
        if getattr(module, "eora_cooperative_state", None) is not None
    ]
    cooperative_setting = os.getenv("GPTQMODEL_EORA_MARLIN_COOPERATIVE", "1").strip().lower()
    cooperative_requested = cooperative_setting not in {"0", "false", "no", "off"}
    if EVAL_BACKEND == BACKEND.MARLIN and cooperative_requested and len(prepared_eora_modules) != len(modules):
        raise RuntimeError(
            "Marlin EoRA preparation did not cover every quantized module: "
            f"prepared={len(prepared_eora_modules)}, modules={len(modules)}"
        )
    if EVAL_BACKEND == BACKEND.MARLIN and not cooperative_requested and prepared_eora_modules:
        raise RuntimeError("Cooperative Marlin EoRA state was prepared even though it was explicitly disabled")

    tokenizer = model.tokenizer
    prompt = "The capital city of France is"
    encoded = {key: value.to("cuda:0") for key, value in tokenizer(prompt, return_tensors="pt").items()}
    with torch.inference_mode():
        logits = model(**encoded).logits.detach().float().cpu()
        generated = model.generate(**encoded, max_new_tokens=24, do_sample=False).detach().cpu()
    if not torch.isfinite(logits).all():
        raise RuntimeError("Fixed EoRA runtime smoke produced non-finite logits")
    continuation = tokenizer.decode(
        generated[0, encoded["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    return {
        "adapter_mode": "fixed truncated-pseudoinverse EoRA",
        "adapter_path": str(ADAPTER),
        "backend": EVAL_BACKEND.value,
        "quant_linear_module_count": len(modules),
        "active_adapter_count": len(modules) - len(missing),
        "prepared_eora_module_count": len(prepared_eora_modules),
        "cooperative_eora_requested": cooperative_requested,
        "adapter_ranks": ranks,
        "logits_shape": list(logits.shape),
        "logits_all_finite": True,
        "prompt": prompt,
        "generated_continuation": continuation,
    }


def run_tasks(model: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """Run full deterministic Evalution tasks, reusing only complete task files."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    task_backends: dict[str, str] = {}
    for task in TASKS:
        result_path = OUTPUT_DIR / f"{task}.json"
        if result_path.is_file():
            existing = json.loads(result_path.read_text())
            summaries[task] = existing["metrics"]
            task_backends[task] = existing.get("backend", BACKEND.GPTQ_TORCH.value)
            print(
                f"Reusing completed fixed-EoRA task {task} (backend={task_backends[task]})",
                flush=True,
            )
            continue

        print(f"Starting full fixed-EoRA Evalution task={task}, batch_size={BATCH_SIZE}", flush=True)
        started = time.perf_counter()
        result = evaluate(
            model_or_id_or_path=model,
            tasks=[task],
            batch_size=BATCH_SIZE,
            backend=EVAL_BACKEND,
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
                "adapter_mode": "fixed truncated-pseudoinverse EoRA",
                "adapter_path": str(ADAPTER),
                "backend": EVAL_BACKEND.value,
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
        task_backends[task] = EVAL_BACKEND.value
        print(f"Completed fixed-EoRA {task}: {metrics}", flush=True)
        torch_empty_cache()
    return summaries, task_backends


def main() -> None:
    """Load the fixed adapter and run runtime plus full quality validation."""

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    gpu = assert_gpu()
    write_json(
        MODEL / "eora_eighfix_eval_manifest.json",
        {
            "started_at": utc_now(),
            "model": str(MODEL),
            "adapter": str(ADAPTER),
            "adapter_mode": "fixed truncated-pseudoinverse EoRA",
            "backend": EVAL_BACKEND.value,
            "eora_marlin_dispatch": {
                "cooperative": os.getenv("GPTQMODEL_EORA_MARLIN_COOPERATIVE"),
                "fused_max_rows": os.getenv("GPTQMODEL_EORA_MARLIN_FUSED_MAX_M"),
                "cuda_up_add": os.getenv("GPTQMODEL_EORA_MARLIN_CUDA_UP_ADD"),
            },
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
    print(f"Loading {MODEL} with fixed adapter {ADAPTER}", flush=True)
    model = GPTQModel.load(
        str(MODEL),
        backend=EVAL_BACKEND,
        adapter=Lora(rank=128, path=str(ADAPTER)),
        dtype="auto",
        device_map={"": "cuda:0"},
        attn_implementation=attention,
    )
    model.eval()
    runtime = validate_runtime(model)
    write_json(MODEL / "runtime_smoke_eora_eighfix.json", runtime)
    tasks, task_backends = run_tasks(model)
    write_json(
        MODEL / "evaluation_complete_eora_eighfix.json",
        {
            "completed_at": utc_now(),
            "backend": EVAL_BACKEND.value,
            "task_backends": task_backends,
            "adapter_mode": "fixed truncated-pseudoinverse EoRA",
            "adapter_path": str(ADAPTER),
            "tasks": tasks,
        },
    )
    print(f"All fixed-EoRA validations completed: {tasks}", flush=True)


if __name__ == "__main__":
    main()
