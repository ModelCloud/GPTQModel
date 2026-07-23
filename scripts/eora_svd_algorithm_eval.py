#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Generate/evaluate one EoRA SVD variant or score the base quantized model."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import gc
import hashlib
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import datasets
import evalution
import gptqmodel
import safetensors
import torch
import transformers
from datasets import load_dataset
from evalution.scorers.gsm8k import INVALID_ANSWER
from safetensors import safe_open
from transformers.utils import is_flash_attn_2_available

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.adapter.adapter import AdapterCache, EORA_SVD_ALGOS, EoRAConfig, Lora
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.utils.linalg_warmup import run_torch_linalg_warmup
from gptqmodel.utils.threadx import DeviceThreadPool, WarmUpCtx, WarmupTask
from gptqmodel.utils.torch import torch_empty_cache
from tests.eval import evaluate, get_eval_task_results


DEFAULT_BASE_MODEL = Path("/monster/data/model/Qwen3-8B")
DEFAULT_QUANTIZED_MODEL = Path(
    "/monster/data/model/Qwen3-8B-GPTQ-4bit-g128-activation-GAR-EoRA-r128-cal512"
)
DEFAULT_CALIBRATION_DATASET = Path("/monster/data/model/dataset/nm-calibration")
CALIBRATION_TEXT_SHA256 = "a89c5ed40152f435d5102657166cb72399603042596c57dd2f1095724ad8f59d"
FOUR_BIT_TASK_BACKENDS = {
    "arc_challenge": BACKEND.GPTQ_TORCH,
    "gsm8k_platinum_cot": BACKEND.MARLIN,
}
THREE_BIT_TASK_BACKENDS = {
    "arc_challenge": BACKEND.GPTQ_TORCH,
    "gsm8k_platinum_cot": BACKEND.GPTQ_TRITON,
}
TWO_BIT_TASK_BACKENDS = {
    "arc_challenge": BACKEND.GPTQ_TORCH,
    "gsm8k_platinum_cot": BACKEND.GPTQ_TORCH,
}
SEED = 898
DEFAULT_RANK = 128
BATCH_SIZE = 16
BASE_QUANT_VARIANT = "base_quant"
DENSE_MODEL_PARAMETER_COUNT = 8_190_735_360


def parse_args() -> argparse.Namespace:
    """Builds one independently runnable algorithm experiment."""

    parser = argparse.ArgumentParser(description=__doc__)
    variant = parser.add_mutually_exclusive_group(required=True)
    variant.add_argument("--algo", choices=EORA_SVD_ALGOS)
    variant.add_argument(
        "--base-quant",
        action="store_true",
        help="Evaluate the quantized snapshot without loading or generating an EoRA adapter.",
    )
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK, help="EoRA rank and experiment-group label.")
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--expected-pci-bus", required=True)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--quantized-model", type=Path, default=DEFAULT_QUANTIZED_MODEL)
    parser.add_argument("--bits", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--group-size", type=int, choices=(32, 64, 128), default=128)
    parser.add_argument("--calibration-dataset", type=Path, default=DEFAULT_CALIBRATION_DATASET)
    parser.add_argument("--stage", choices=("generate", "eval", "all"), default="all")
    args = parser.parse_args()
    if args.rank <= 0:
        parser.error("--rank must be positive")
    if args.base_quant and args.stage == "generate":
        parser.error("--base-quant has no generation stage")
    return args


def utc_now() -> str:
    """Returns a timezone-aware timestamp for result manifests."""

    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    """Converts Evalution and enum values into stable JSON values."""

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


def numeric_output_counts(result: Any) -> dict[str, int]:
    """Counts scored numeric samples and extraction failures in an Evalution result."""

    if not isinstance(result, dict) or not isinstance(result.get("tests"), list):
        raise RuntimeError("Evalution result does not expose a tests list for numeric-output validation")
    sample_count = 0
    invalid_count = 0
    for test in result["tests"]:
        if not isinstance(test, dict) or not isinstance(test.get("samples"), list):
            continue
        for sample in test["samples"]:
            sample_count += 1
            extracted = sample.get("extracted") if isinstance(sample, dict) else None
            valid = isinstance(extracted, dict) and any(
                value not in (None, "", INVALID_ANSWER) for value in extracted.values()
            )
            invalid_count += not valid
    if sample_count == 0:
        raise RuntimeError("Evalution result contains no numeric samples")
    return {"sample_count": sample_count, "invalid_output_count": invalid_count}


def write_json(path: Path, payload: Any) -> None:
    """Atomically writes a result so interrupted runs never look complete."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def experiment_variant(args: argparse.Namespace) -> str:
    """Returns the stable output label for an algorithm or the adapter-free baseline."""

    return BASE_QUANT_VARIANT if args.base_quant else str(args.algo)


def output_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    """Returns non-overlapping artifact paths for one configured experiment variant."""

    variant = experiment_variant(args)
    experiment_name = f"eora_rank{args.rank}_{variant}"
    return {
        "adapter": None if args.base_quant else args.quantized_model / f"eora-rank{args.rank}-svd-{args.algo}",
        "eval": args.quantized_model / f"evalution_{experiment_name}",
        "manifest": args.quantized_model / f"{experiment_name}_experiment.json",
    }


def task_backends(bits: int) -> dict[str, BACKEND]:
    """Returns the verified backend contract for each bit-width study."""

    if bits == 4:
        return FOUR_BIT_TASK_BACKENDS
    if bits == 3:
        return THREE_BIT_TASK_BACKENDS
    return TWO_BIT_TASK_BACKENDS


def load_quantization_contract(
    path: Path,
    expected_bits: int,
    expected_group_size: int,
) -> dict[str, Any]:
    """Loads and validates the controlled GPTQ snapshot used by the quality study."""

    config_path = path / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"Quantized snapshot is missing config.json: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise RuntimeError(f"Quantized snapshot has no quantization_config: {config_path}")
    metadata = quantization.get("meta")
    if not isinstance(metadata, dict):
        metadata = {}

    contract = {
        "bits": quantization.get("bits"),
        "group_size": quantization.get("group_size"),
        "sym": quantization.get("sym"),
        "desc_act": quantization.get("desc_act"),
        "scale_search": metadata.get("scale_search"),
        "act_group_aware": metadata.get("act_group_aware"),
        "mse": metadata.get("mse"),
    }
    expected = {
        "bits": expected_bits,
        "group_size": expected_group_size,
        "sym": True,
        "desc_act": False,
        "scale_search": "activation",
        "act_group_aware": True,
        "mse": 2.0,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": contract[key]}
        for key, expected_value in expected.items()
        if contract[key] != expected_value
    }
    if mismatches:
        raise RuntimeError(f"Quantized snapshot does not match the controlled study contract: {mismatches}")
    return contract


def assert_gpu(args: argparse.Namespace) -> dict[str, Any]:
    """Requires exactly the requested PCI-ordered device before allocating tensors."""

    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID" or torch.cuda.device_count() != 1:
        raise RuntimeError("Experiment requires exactly one PCI-ordered visible GPU")
    properties = torch.cuda.get_device_properties(0)
    actual_uuid = str(properties.uuid).removeprefix("GPU-").lower()
    expected_uuid = args.expected_uuid.removeprefix("GPU-").lower()
    actual_bus = int(properties.pci_bus_id)
    expected_bus = int(args.expected_pci_bus.split(":")[-2], 16)
    if actual_uuid != expected_uuid or actual_bus != expected_bus:
        raise RuntimeError(
            f"Expected GPU {args.physical_gpu} uuid={expected_uuid}, bus={expected_bus:#x}; "
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


def configure_runtime() -> dict[str, Any]:
    """Bounds host workers and installs deterministic per-device linalg warmup."""

    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise

    if getattr(gptqmodel, "_DEVICE_THREAD_POOL", None) is not None:
        raise RuntimeError("GPTQModel device thread pool initialized before experiment setup")
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
        "seed": SEED,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "worker_counts": worker_counts,
    }


def load_calibration(path: Path) -> tuple[Any, dict[str, Any]]:
    """Loads and fingerprints the exact first 512 calibration rows."""

    dataset = load_dataset(path=str(path), name="LLM", split="train")
    if len(dataset) < 512:
        raise RuntimeError(f"Calibration dataset only has {len(dataset)} rows")
    selected = dataset.select(range(512))

    digest = hashlib.sha256()
    text_characters = 0
    for row in selected:
        text = str(row["text"])
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        text_characters += len(text)
    actual_digest = digest.hexdigest()
    if actual_digest != CALIBRATION_TEXT_SHA256:
        raise RuntimeError(
            f"Calibration snapshot mismatch: expected {CALIBRATION_TEXT_SHA256}, found {actual_digest}"
        )
    return selected, {
        "path": str(path),
        "name": "LLM",
        "split": "train",
        "dataset_rows": len(dataset),
        "selected_rows": len(selected),
        "selection": "first 512 rows",
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "selected_text_sha256": actual_digest,
        "selected_text_characters": text_characters,
        "sort": "desc",
        "concat_size": None,
        "batch_size": 1,
    }


def summarize_adapter(path: Path, generation_config: Lora, rank: int) -> dict[str, Any]:
    """Checks every generated factor and records its persisted layout."""

    weights_path = path / "adapter_model.safetensors"
    config_path = path / "adapter_config.json"
    if not weights_path.is_file() or not config_path.is_file():
        raise RuntimeError(f"Incomplete generated adapter at {path}")

    tensor_count = 0
    total_elements = 0
    ranks: set[int] = set()
    dtypes: set[str] = set()
    all_finite = True
    maximum_abs = 0.0
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            tensor_count += 1
            total_elements += tensor.numel()
            dtypes.add(str(tensor.dtype))
            all_finite = all_finite and bool(torch.isfinite(tensor).all().item())
            maximum_abs = max(maximum_abs, float(tensor.abs().max().item()))
            if key.endswith("lora_A.weight"):
                ranks.add(int(tensor.shape[0]))
            elif key.endswith("lora_B.weight"):
                ranks.add(int(tensor.shape[1]))

    if tensor_count != 504 or ranks != {rank} or not all_finite:
        raise RuntimeError(
            f"Generated adapter validation failed: tensors={tensor_count}, ranks={sorted(ranks)}, finite={all_finite}"
        )
    return {
        "path": str(path),
        "weights_bytes": weights_path.stat().st_size,
        "tensor_count": tensor_count,
        "total_elements": total_elements,
        "dtypes": sorted(dtypes),
        "observed_ranks": sorted(ranks),
        "all_finite": all_finite,
        "maximum_factor_abs": maximum_abs,
        "generation_config": generation_config.to_dict(),
        "peft_config": json.loads(config_path.read_text(encoding="utf-8")),
    }


def summarize_storage(quantized_model: Path, adapter_path: Path | None) -> dict[str, Any]:
    """Measures the saved model and EoRA tensor footprint against the dense parameter count."""

    model_paths = sorted(quantized_model.glob("model-*.safetensors"))
    if not model_paths and (quantized_model / "model.safetensors").is_file():
        model_paths = [quantized_model / "model.safetensors"]
    if not model_paths:
        raise RuntimeError(f"Quantized snapshot has no model safetensors: {quantized_model}")
    model_bytes = sum(path.stat().st_size for path in model_paths)

    adapter_bytes = 0
    if adapter_path is not None:
        adapter_weights = adapter_path / "adapter_model.safetensors"
        if not adapter_weights.is_file():
            raise RuntimeError(f"EoRA snapshot has no adapter safetensors: {adapter_path}")
        adapter_bytes = adapter_weights.stat().st_size

    combined_bytes = model_bytes + adapter_bytes
    return {
        "dense_base_parameter_count": DENSE_MODEL_PARAMETER_COUNT,
        "quantized_model_tensor_bytes": model_bytes,
        "eora_tensor_bytes": adapter_bytes,
        "combined_tensor_bytes": combined_bytes,
        "effective_bits_per_base_parameter": 8 * combined_bytes / DENSE_MODEL_PARAMETER_COUNT,
    }


def generate_adapter(args: argparse.Namespace, paths: dict[str, Path | None], manifest: dict[str, Any]) -> None:
    """Generates one fresh configured-rank adapter from the shared calibration snapshot."""

    adapter_path = paths["adapter"]
    manifest_path = paths["manifest"]
    if adapter_path is None or manifest_path is None or args.algo is None:
        raise RuntimeError("Adapter generation requires an EoRA algorithm")
    if adapter_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing algorithm adapter: {adapter_path}")

    calibration, calibration_metadata = load_calibration(args.calibration_dataset)
    adapter = Lora(
        rank=args.rank,
        path=str(adapter_path),
        eora_config=EoRAConfig(algo=args.algo),
    )
    started = time.perf_counter()
    GPTQModel.adapter.generate(
        adapter=adapter,
        model_id_or_path=str(args.base_model),
        quantized_model_id_or_path=str(args.quantized_model),
        calibration_dataset=calibration,
        calibration_dataset_concat_size=None,
        calibration_dataset_sort="desc",
        batch_size=1,
        dtype="auto",
    )
    manifest["generation"] = {
        "completed_at": utc_now(),
        "wall_s": time.perf_counter() - started,
        "calibration": calibration_metadata,
        "adapter": summarize_adapter(adapter_path, adapter, rank=args.rank),
    }
    manifest["storage"] = summarize_storage(args.quantized_model, adapter_path)
    write_json(manifest_path, manifest)
    del calibration
    gc.collect()
    torch_empty_cache()


def validate_loaded_adapter(
    model: Any,
    adapter_path: Path | None,
    rank: int,
    bits: int,
    group_size: int,
) -> dict[str, Any]:
    """Checks adapter coverage and finite deterministic generation before scoring."""

    modules = [module for module in model.model.modules() if isinstance(module, BaseQuantLinear)]
    active = [module for module in modules if getattr(module, "adapter", None) is not None]
    ranks = sorted({int(module.adapter.rank) for module in active})
    observed_bits = sorted({int(module.bits) for module in modules})
    observed_group_sizes = sorted({int(module.group_size) for module in modules})
    expected_active = 0 if adapter_path is None else 252
    expected_ranks = [] if adapter_path is None else [rank]
    if (
        len(modules) != 252
        or len(active) != expected_active
        or ranks != expected_ranks
        or observed_bits != [bits]
        or observed_group_sizes != [group_size]
    ):
        raise RuntimeError(
            "Quantized module validation failed: "
            f"modules={len(modules)}, active={len(active)}, ranks={ranks}, "
            f"bits={observed_bits}, group_sizes={observed_group_sizes}"
        )

    tokenizer = model.tokenizer
    encoded = {key: value.to("cuda:0") for key, value in tokenizer("The capital city of France is", return_tensors="pt").items()}
    with torch.inference_mode():
        logits = model(**encoded).logits.detach().float().cpu()
        generated = model.generate(**encoded, max_new_tokens=24, do_sample=False).detach().cpu()
    if not torch.isfinite(logits).all():
        raise RuntimeError("EoRA runtime smoke produced non-finite logits")
    return {
        "adapter_path": str(adapter_path) if adapter_path is not None else None,
        "quant_linear_module_count": len(modules),
        "active_adapter_count": len(active),
        "adapter_ranks": ranks,
        "bits": observed_bits,
        "group_sizes": observed_group_sizes,
        "logits_all_finite": True,
        "continuation": tokenizer.decode(
            generated[0, encoded["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ),
    }


def evaluate_tasks(args: argparse.Namespace, paths: dict[str, Path | None], manifest: dict[str, Any]) -> None:
    """Runs complete ARC Challenge and GSM8K Platinum with matched settings."""

    eval_path = paths["eval"]
    manifest_path = paths["manifest"]
    if eval_path is None or manifest_path is None:
        raise RuntimeError("Experiment output paths are incomplete")
    eval_path.mkdir(parents=True, exist_ok=True)
    evaluation = manifest.get("evaluation")
    if evaluation is None:
        evaluation = {
            "started_at": utc_now(),
            "batch_size": BATCH_SIZE,
            "apply_chat_template": False,
            "max_rows": None,
            "seed": SEED,
            "tasks": {},
        }
        manifest["evaluation"] = evaluation
    if not isinstance(evaluation, dict):
        raise RuntimeError(f"Experiment manifest has invalid evaluation metadata: {manifest_path}")
    expected_evaluation = {
        "batch_size": BATCH_SIZE,
        "apply_chat_template": False,
        "max_rows": None,
        "seed": SEED,
    }
    mismatches = {
        key: {"expected": value, "actual": evaluation.get(key)}
        for key, value in expected_evaluation.items()
        if evaluation.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Existing evaluation contract does not match this experiment: {mismatches}")
    if not isinstance(evaluation.get("tasks"), dict):
        raise RuntimeError(f"Experiment manifest has invalid evaluation tasks: {manifest_path}")
    write_json(manifest_path, manifest)

    attention = "flash_attention_2" if is_flash_attn_2_available() else "eager"
    for task, backend in task_backends(args.bits).items():
        task_path = eval_path / f"{task}.json"
        if task_path.is_file():
            task_payload = json.loads(task_path.read_text(encoding="utf-8"))
            expected = {
                "task": task,
                "variant": experiment_variant(args),
                "algo": args.algo,
                "rank": args.rank,
                "backend": backend.value,
                "full_dataset": True,
                "max_rows": None,
            }
            mismatches = {
                key: {"expected": value, "actual": task_payload.get(key)}
                for key, value in expected.items()
                if task_payload.get(key) != value
            }
            metrics = task_payload.get("metrics")
            if mismatches or not isinstance(metrics, dict) or not metrics:
                raise RuntimeError(
                    f"Existing task result does not match this experiment: path={task_path}, "
                    f"mismatches={mismatches}, metrics={metrics}"
                )
            output_counts = (
                numeric_output_counts(task_payload.get("result"))
                if task == "gsm8k_platinum_cot"
                else {"sample_count": None, "invalid_output_count": None}
            )
            if any(task_payload.get(key) != value for key, value in output_counts.items()):
                task_payload.update(output_counts)
                write_json(task_path, task_payload)
            evaluation["tasks"][task] = {
                "backend": backend.value,
                "load_wall_s": task_payload.get("load_wall_s"),
                "eval_wall_s": task_payload.get("eval_wall_s"),
                "metrics": metrics,
                **output_counts,
            }
            write_json(manifest_path, manifest)
            print(f"Reusing completed Evalution variant={experiment_variant(args)} task={task}: {metrics}", flush=True)
            continue

        AdapterCache.reset()
        load_started = time.perf_counter()
        model = GPTQModel.load(
            str(args.quantized_model),
            backend=backend,
            adapter=None if paths["adapter"] is None else Lora(rank=args.rank, path=str(paths["adapter"])),
            dtype="auto",
            device_map={"": "cuda:0"},
            attn_implementation=attention,
        )
        model.eval()
        load_wall_s = time.perf_counter() - load_started
        runtime = validate_loaded_adapter(
            model,
            paths["adapter"],
            rank=args.rank,
            bits=args.bits,
            group_size=args.group_size,
        )

        variant = experiment_variant(args)
        print(f"Starting full Evalution variant={variant} task={task} backend={backend.value}", flush=True)
        eval_started = time.perf_counter()
        result = evaluate(
            model_or_id_or_path=model,
            tasks=[task],
            batch_size=BATCH_SIZE,
            backend=backend,
            model_args={"device": "cuda:0", "seed": SEED, "random_seed": SEED},
            apply_chat_template=False,
            gen_kwargs="do_sample=false,temperature=0.0,top_p=1.0,top_k=50,max_new_tokens=256",
        )
        eval_wall_s = time.perf_counter() - eval_started
        metrics = get_eval_task_results(result).get(task, {})
        if not metrics:
            raise RuntimeError(f"Evalution returned no metrics for {task}: {result}")
        if any(isinstance(value, (int, float)) and not math.isfinite(float(value)) for value in metrics.values()):
            raise RuntimeError(f"Evalution returned non-finite metrics for {task}: {metrics}")
        output_counts = (
            numeric_output_counts(result)
            if task == "gsm8k_platinum_cot"
            else {"sample_count": None, "invalid_output_count": None}
        )

        task_payload = {
            "task": task,
            "variant": variant,
            "algo": args.algo,
            "rank": args.rank,
            "adapter_path": str(paths["adapter"]) if paths["adapter"] is not None else None,
            "backend": backend.value,
            "full_dataset": True,
            "max_rows": None,
            "batch_size": BATCH_SIZE,
            "apply_chat_template": False,
            "load_wall_s": load_wall_s,
            "eval_wall_s": eval_wall_s,
            "runtime": runtime,
            "metrics": metrics,
            **output_counts,
            "result": result,
        }
        write_json(task_path, task_payload)
        evaluation["tasks"][task] = {
            "backend": backend.value,
            "load_wall_s": load_wall_s,
            "eval_wall_s": eval_wall_s,
            "metrics": metrics,
            **output_counts,
        }
        write_json(manifest_path, manifest)
        print(f"Completed Evalution variant={variant} task={task}: {metrics}", flush=True)

        del model, result
        gc.collect()
        torch_empty_cache()

    evaluation["completed_at"] = utc_now()
    write_json(manifest_path, manifest)


def main() -> None:
    """Runs the requested generation/evaluation stages and persists progress."""

    args = parse_args()
    paths = output_paths(args)
    gpu = assert_gpu(args)
    runtime = configure_runtime()
    variant = experiment_variant(args)
    quantization_contract = load_quantization_contract(
        args.quantized_model,
        expected_bits=args.bits,
        expected_group_size=args.group_size,
    )
    manifest = {
        "started_at": utc_now(),
        "variant": variant,
        "algo": args.algo,
        "rank": args.rank,
        "eora_config": None if args.base_quant else EoRAConfig(algo=args.algo).to_dict(),
        "base_model": str(args.base_model),
        "quantized_model": str(args.quantized_model),
        "adapter_path": str(paths["adapter"]) if paths["adapter"] is not None else None,
        "eval_path": str(paths["eval"]),
        "gpu": gpu,
        "runtime": runtime,
        "quantization_contract": quantization_contract,
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "gptqmodel": getattr(gptqmodel, "__version__", None),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "safetensors": safetensors.__version__,
            "evalution": getattr(evalution, "__version__", None),
        },
    }

    manifest_path = paths["manifest"]
    if manifest_path is None:
        raise RuntimeError("Experiment manifest path is missing")

    if args.stage in {"generate", "all"} and not args.base_quant:
        if manifest_path.exists():
            raise RuntimeError(f"Refusing to overwrite existing experiment manifest: {manifest_path}")
        write_json(manifest_path, manifest)
        generate_adapter(args, paths, manifest)
    elif not args.base_quant and (paths["adapter"] is None or not paths["adapter"].is_dir()):
        raise RuntimeError(f"Evaluation requires an existing adapter: {paths['adapter']}")
    elif args.stage == "all":
        if manifest_path.exists():
            raise RuntimeError(f"Refusing to overwrite existing experiment manifest: {manifest_path}")
        write_json(manifest_path, manifest)

    if args.stage in {"eval", "all"}:
        if args.stage == "eval":
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            else:
                write_json(manifest_path, manifest)
        manifest["storage"] = summarize_storage(args.quantized_model, paths["adapter"])
        write_json(manifest_path, manifest)
        evaluate_tasks(args, paths, manifest)

    manifest["completed_at"] = utc_now()
    write_json(manifest_path, manifest)
    print(json.dumps(jsonable(manifest), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
