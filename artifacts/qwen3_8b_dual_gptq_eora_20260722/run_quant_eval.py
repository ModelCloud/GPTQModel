#!/usr/bin/env python3
"""Quantize Qwen3-8B with GPTQ, optionally generate EoRA, and validate it with Evalution.

This is an operational run harness, not a unit test. One process owns one
CUDA-visible device and one output directory so independent arms can run
concurrently without sharing mutable model or adapter state.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# These must be set before importing torch/GPTQModel.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault(
    "PYTORCH_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7",
)
for thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_variable, "1")

# Python puts the script's artifact directory, rather than the working tree,
# first on sys.path when this file is launched directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import datasets
import evalution
import gptqmodel
import safetensors
import torch
import transformers
from datasets import load_dataset
from safetensors import safe_open
from transformers import AutoModelForCausalLM
from transformers.utils import is_flash_attn_2_available

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig, ScaleSearchConfig
from gptqmodel.adapter.adapter import EoRAConfig, Lora
from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.config import GcMode
from gptqmodel.utils.linalg_warmup import run_torch_linalg_warmup
from gptqmodel.utils.threadx import DeviceThreadPool, WarmUpCtx, WarmupTask
from gptqmodel.utils.torch import torch_empty_cache
from tests.eval import evaluate, get_eval_task_results


BASE_MODEL = Path("/monster/data/model/Qwen3-8B")
CALIBRATION_DATASET = Path("/monster/data/model/dataset/nm-calibration")
CALIBRATION_NAME = "LLM"
CALIBRATION_ROWS = 512
EORA_RANK = 128
SEED = 898
TASKS = ("arc_challenge", "gsm8k_platinum_cot")


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
    if isinstance(value, torch.dtype):
        return str(value)
    if hasattr(value, "value"):
        return jsonable(value.value)
    return repr(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run_text(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return completed.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--expected-pci-bus", required=True)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--bits", required=True, type=int, choices=(2, 3, 4))
    parser.add_argument("--group-size", required=True, type=int, choices=(32, 64, 128))
    parser.add_argument("--scale-search", required=True, choices=("activation", "disabled"))
    parser.add_argument("--gar", required=True, choices=("enabled", "disabled"))
    parser.add_argument("--eora", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument(
        "--eora-zero-correction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Diagnostic control: retain the joint EoRA lifecycle but force every generated B @ A correction to zero.",
    )
    parser.add_argument(
        "--static-groups",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Precompute group quantizers from the original weights before sequential GPTQ error feedback.",
    )
    parser.add_argument(
        "--quantization-diagnostics",
        choices=("off", "auto", "channel"),
        default="auto",
        help="Quantization-time anomaly diagnostics; channel additionally scans every scale tensor.",
    )
    parser.add_argument("--pack-impl", choices=("cpu", "gpu", "original"), default="cpu")
    parser.add_argument("--eval-backend", required=True, choices=("marlin", "torch"))
    parser.add_argument("--eval-task", action="append", choices=TASKS)
    parser.add_argument("--eval-max-rows", type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--stage", choices=("all", "quantize", "eval"), default="all")
    args = parser.parse_args()
    if args.eval_max_rows is not None and args.eval_max_rows <= 0:
        parser.error("--eval-max-rows must be positive")
    if args.eora_zero_correction and args.eora != "enabled":
        parser.error("--eora-zero-correction requires --eora enabled")
    return args


def requested_eval_tasks(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(args.eval_task) if args.eval_task else TASKS


def evaluation_result_path(output: Path, task: str, max_rows: int | None) -> Path:
    suffix = "" if max_rows is None else f".maxrows{max_rows}"
    return output / "evalution" / f"{task}{suffix}.json"


def assert_device(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Each run must see exactly one CUDA device; actual "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}, count={torch.cuda.device_count()}"
        )

    props = torch.cuda.get_device_properties(0)
    actual_uuid = str(props.uuid).removeprefix("GPU-").lower()
    expected_uuid = args.expected_uuid.removeprefix("GPU-").lower()
    if actual_uuid != expected_uuid:
        raise RuntimeError(f"Visible GPU UUID mismatch: expected {expected_uuid}, got {actual_uuid}")

    expected_bus_number = int(args.expected_pci_bus.split(":")[-2], 16)
    actual_bus_number = int(props.pci_bus_id)
    if actual_bus_number != expected_bus_number:
        raise RuntimeError(
            f"Visible GPU PCI bus mismatch: expected {expected_bus_number:#x}, got {actual_bus_number:#x}"
        )

    return {
        "requested_physical_index_pci_order": args.physical_gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "process_cuda_index": 0,
        "pci_bus_id": args.expected_pci_bus,
        "uuid": f"GPU-{actual_uuid}",
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "memory_bytes": props.total_memory,
    }


def configure_cpu_threads() -> dict[str, int]:
    """Bound nested CPU pools used by concurrent GPTQ worker threads."""
    requested = int(os.environ.get("GPTQ_RUN_TORCH_THREADS", "1"))
    torch.set_num_threads(requested)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    return {
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }


def configure_device_thread_pool() -> dict[str, int]:
    """Serialize CPU packing while retaining a dedicated CUDA worker."""
    if getattr(gptqmodel, "_DEVICE_THREAD_POOL", None) is not None:
        raise RuntimeError("GPTQModel device thread pool was initialized before run configuration")

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
    return worker_counts


def load_calibration() -> tuple[Any, dict[str, Any]]:
    dataset = load_dataset(path=str(CALIBRATION_DATASET), name=CALIBRATION_NAME, split="train")
    if len(dataset) < CALIBRATION_ROWS:
        raise RuntimeError(f"Calibration dataset only has {len(dataset)} rows")
    selected = dataset.select(range(CALIBRATION_ROWS))

    digest = hashlib.sha256()
    token_source_chars = 0
    for row in selected:
        text = str(row["text"])
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        token_source_chars += len(text)

    metadata = {
        "path": str(CALIBRATION_DATASET),
        "name": CALIBRATION_NAME,
        "split": "train",
        "dataset_rows": len(dataset),
        "selected_rows": len(selected),
        "selection": "first 512 rows",
        "columns": list(dataset.column_names),
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "selected_text_sha256": digest.hexdigest(),
        "selected_text_characters": token_source_chars,
        "calibration_sort": "desc",
        "calibration_concat_size": None,
        "calibration_data_min_length": 10,
        "quant_batch_size": 1,
    }
    return selected, metadata


def build_quant_config(args: argparse.Namespace, adapter_dir: Path) -> QuantizeConfig:
    scale_search = (
        ScaleSearchConfig.ACTIVATION if args.scale_search == "activation" else None
    )
    adapter = (
        Lora(
            rank=EORA_RANK,
            path=str(adapter_dir),
            eora_config=EoRAConfig(algo="lowrank"),
        )
        if args.eora == "enabled"
        else None
    )
    return QuantizeConfig(
        quant_method=METHOD.GPTQ,
        format=FORMAT.GPTQ,
        bits=args.bits,
        group_size=args.group_size,
        sym=True,
        desc_act=False,
        act_group_aware=args.gar == "enabled",
        scale_search=scale_search,
        mse=2.0 if scale_search is not None else 0.0,
        static_groups=args.static_groups,
        quantization_diagnostics=args.quantization_diagnostics,
        adapter=adapter,
        pack_impl=args.pack_impl,
        gc_mode=GcMode.ON_STAGE_END,
        # GPTQ and EoRA share processor/module state.  On this Python 3.14t
        # runtime, overlapping layer N finalizers with layer N+1 quantization
        # reproducibly deadlocks at the next projection group.
        wait_for_submodule_finalizers=True,
    )


def package_versions() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "gptqmodel": getattr(gptqmodel, "__version__", None),
        "evalution": getattr(evalution, "__version__", None),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "safetensors": safetensors.__version__,
    }


def artifact_complete(args: argparse.Namespace, output: Path, adapter_dir: Path) -> bool:
    model_files = sorted(output.glob("*.safetensors"))
    model_complete = (output / "config.json").is_file() and bool(model_files)
    if args.eora == "disabled":
        return model_complete
    return model_complete and (adapter_dir / "adapter_config.json").is_file() and (
        adapter_dir / "adapter_model.safetensors"
    ).is_file()


def summarize_adapter(adapter_dir: Path) -> dict[str, Any]:
    weights_path = adapter_dir / "adapter_model.safetensors"
    ranks: set[int] = set()
    tensor_count = 0
    total_elements = 0
    dtypes: set[str] = set()
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor_count += 1
            tensor_slice = handle.get_slice(key)
            shape = tuple(int(dim) for dim in tensor_slice.get_shape())
            total_elements += math.prod(shape)
            dtypes.add(str(tensor_slice.get_dtype()))
            if len(shape) == 2:
                if key.endswith("lora_A.weight"):
                    ranks.add(shape[0])
                elif key.endswith("lora_B.weight"):
                    ranks.add(shape[1])

    return {
        "path": str(adapter_dir),
        "weights_path": str(weights_path),
        "weights_bytes": weights_path.stat().st_size,
        "tensor_count": tensor_count,
        "total_elements": total_elements,
        "dtypes": sorted(dtypes),
        "observed_ranks": sorted(ranks),
        "config": json.loads((adapter_dir / "adapter_config.json").read_text()),
    }


def summarize_model_artifact(output: Path) -> dict[str, Any]:
    config = json.loads((output / "config.json").read_text())
    files = []
    total_bytes = 0
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        size = path.stat().st_size
        total_bytes += size
        files.append({"path": str(path.relative_to(output)), "bytes": size})
    return {
        "path": str(output),
        "total_bytes": total_bytes,
        "files": files,
        "saved_quantization_config": config.get("quantization_config"),
    }


def validate_saved_metadata(args: argparse.Namespace, output: Path, adapter_dir: Path) -> dict[str, Any]:
    if not artifact_complete(args, output, adapter_dir):
        raise RuntimeError(f"Saved artifact is incomplete: {output}")

    config = json.loads((output / "config.json").read_text())
    quant = config.get("quantization_config") or {}
    quant_meta = quant.get("meta") or {}
    expected = {
        "method": "gptq",
        "format": "gptq",
        "bits": args.bits,
        "group_size": args.group_size,
        "sym": True,
        "desc_act": False,
        "act_group_aware": args.gar == "enabled",
        "static_groups": args.static_groups,
        "scale_search": "activation" if args.scale_search == "activation" else None,
        "wait_for_submodule_finalizers": True,
        "gc_mode": "on_stage_end",
    }
    mismatches = {
        key: {
            "expected": value,
            "actual": quant[key] if key in quant else quant_meta.get(key),
        }
        for key, value in expected.items()
        if (quant[key] if key in quant else quant_meta.get(key)) != value
    }
    if mismatches:
        raise RuntimeError(f"Saved quantization metadata mismatch: {mismatches}")

    adapter_summary = summarize_adapter(adapter_dir) if args.eora == "enabled" else None
    if adapter_summary is not None and adapter_summary["observed_ranks"] != [EORA_RANK]:
        raise RuntimeError(f"Expected only EoRA rank {EORA_RANK}, observed {adapter_summary['observed_ranks']}")
    return {
        "expected": expected,
        "adapter": adapter_summary,
        "artifact": summarize_model_artifact(output),
    }


def install_zero_eora_correction_control() -> None:
    """Replace EoRA factor generation with shape-correct zeros for one diagnostic process."""

    from gptqmodel.looper import eora_processor as eora_processor_module

    def zero_eora_compute_lora(
        *,
        w_wq_delta: torch.Tensor,
        name: str,
        eigen_scaling_diag_matrix: torch.Tensor,
        rank: int,
        dtype: torch.dtype,
        device: torch.device,
        use_cholesky: bool,
        eora_config: EoRAConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del name, eigen_scaling_diag_matrix, use_cholesky, eora_config
        output_features, input_features = w_wq_delta.shape
        return (
            torch.zeros((rank, input_features), dtype=dtype, device=device),
            torch.zeros((output_features, rank), dtype=dtype, device=device),
        )

    eora_processor_module.eora_compute_lora = zero_eora_compute_lora


def quantize(args: argparse.Namespace, output: Path, adapter_dir: Path, manifest: dict[str, Any]) -> None:
    quant_marker = output / "quantization_complete.json"
    if quant_marker.is_file():
        print(f"[{args.label}] Quantization marker exists; reusing {output}", flush=True)
        return
    if artifact_complete(args, output, adapter_dir):
        validation = validate_saved_metadata(args, output, adapter_dir)
        write_json(
            quant_marker,
            {
                "completed_at": utc_now(),
                "recovered_complete_artifact": True,
                "validation": validation,
            },
        )
        print(f"[{args.label}] Recovered complete saved artifact", flush=True)
        return

    partial_model_files = list(output.glob("*.safetensors"))
    partial_adapter_files = list(adapter_dir.glob("*")) if args.eora == "enabled" and adapter_dir.exists() else []
    if partial_model_files or partial_adapter_files:
        raise RuntimeError(
            f"Refusing to overwrite partial artifacts in {output}; model={partial_model_files}, "
            f"adapter={partial_adapter_files}"
        )

    calibration, calibration_metadata = load_calibration()
    manifest["calibration"] = calibration_metadata
    quant_config = build_quant_config(args, adapter_dir)
    if args.eora_zero_correction:
        install_zero_eora_correction_control()
    manifest["quantization_config"] = quant_config.to_dict()
    write_json(output / "run_manifest.json", manifest)

    attention = "flash_attention_2" if is_flash_attn_2_available() else "eager"
    print(
        f"[{args.label}] Loading dense model {BASE_MODEL} on cuda:0 with attention={attention}",
        flush=True,
    )
    started = time.perf_counter()
    model = GPTQModel.load(
        str(BASE_MODEL),
        quantize_config=quant_config,
        dtype="auto",
        device_map={"": "cuda:0"},
        attn_implementation=attention,
    )
    load_wall_s = time.perf_counter() - started

    print(
        f"[{args.label}] Quantizing {args.bits}-bit/group-{args.group_size}; "
        f"scale_search={args.scale_search}, GAR={args.gar}, EoRA={args.eora}, "
        f"zero_correction={args.eora_zero_correction}, static_groups={args.static_groups}, "
        f"diagnostics={args.quantization_diagnostics}",
        flush=True,
    )
    quant_started = time.perf_counter()
    quant_result = model.quantize(
        calibration=calibration,
        calibration_concat_size=None,
        calibration_sort="desc",
        batch_size=1,
        backend=BACKEND.GPTQ_TORCH,
        calibration_data_min_length=10,
    )
    quant_wall_s = time.perf_counter() - quant_started

    print(f"[{args.label}] Saving quantized model (EoRA={args.eora}) to {output}", flush=True)
    save_started = time.perf_counter()
    model.save(str(output))
    save_wall_s = time.perf_counter() - save_started
    del model, calibration
    torch_empty_cache()

    validation = validate_saved_metadata(args, output, adapter_dir)
    result_summary = {
        "type": type(quant_result).__name__,
        "keys": sorted(str(key) for key in quant_result) if isinstance(quant_result, dict) else None,
        "length": len(quant_result) if hasattr(quant_result, "__len__") else None,
    }
    write_json(
        quant_marker,
        {
            "completed_at": utc_now(),
            "load_wall_s": load_wall_s,
            "quantize_wall_s": quant_wall_s,
            "save_wall_s": save_wall_s,
            "quantize_return": result_summary,
            "validation": validation,
        },
    )
    print(
        f"[{args.label}] Quantization saved and validated in {quant_wall_s:.1f}s "
        f"(save {save_wall_s:.1f}s)",
        flush=True,
    )


def model_device(model: Any) -> torch.device:
    inner = getattr(model, "model", model)
    for parameter in inner.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    for buffer in inner.buffers():
        if buffer.device.type != "meta":
            return buffer.device
    raise RuntimeError("Could not resolve model device")


def validate_loaded_runtime(
    args: argparse.Namespace,
    model: Any,
    output: Path,
) -> tuple[dict[str, Any], torch.Tensor, dict[str, torch.Tensor]]:
    modules = []
    for name, module in model.model.named_modules():
        if isinstance(module, BaseQuantLinear):
            modules.append((name, module))
    if not modules:
        raise RuntimeError("Reloaded model has no quantized linear modules")

    bad_bits = [(name, module.bits) for name, module in modules if module.bits != args.bits]
    bad_groups = [
        (name, module.group_size)
        for name, module in modules
        if module.group_size != args.group_size
    ]
    missing_adapters = [name for name, module in modules if getattr(module, "adapter", None) is None]
    adapter_ranks = sorted(
        {
            int(module.adapter.rank)
            for _, module in modules
            if getattr(module, "adapter", None) is not None
        }
    )
    adapter_contract_failed = (
        (args.eora == "enabled" and (missing_adapters or adapter_ranks != [EORA_RANK]))
        or (args.eora == "disabled" and len(missing_adapters) != len(modules))
    )
    if bad_bits or bad_groups or adapter_contract_failed:
        raise RuntimeError(
            "Reloaded module contract failed: "
            f"bad_bits={bad_bits[:3]}, bad_groups={bad_groups[:3]}, "
            f"missing_adapters={missing_adapters[:3]}, adapter_ranks={adapter_ranks}"
        )

    tokenizer = model.tokenizer
    prompt = "The capital city of France is"
    encoded = tokenizer(prompt, return_tensors="pt")
    device = model_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        quant_output = model(**encoded)
        quant_logits_dtype = str(quant_output.logits.dtype)
        quant_logits = quant_output.logits.detach().float().cpu()
        generated = model.generate(
            **encoded,
            max_new_tokens=24,
            do_sample=False,
        )
    if not torch.isfinite(quant_logits).all():
        raise RuntimeError("Quantized+EoRA smoke forward produced non-finite logits")

    generated_cpu = generated.detach().cpu()
    response = tokenizer.decode(
        generated_cpu[0, encoded["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )
    summary = {
        "backend": args.eval_backend,
        "quant_linear_class_counts": {},
        "quant_linear_module_count": len(modules),
        "adapter_module_count": len(modules) - len(missing_adapters),
        "adapter_ranks": adapter_ranks,
        "logits_shape": list(quant_logits.shape),
        "logits_dtype_before_fp32_copy": quant_logits_dtype,
        "logits_all_finite": True,
        "prompt": prompt,
        "generated_continuation": response,
    }
    for _, module in modules:
        name = type(module).__name__
        summary["quant_linear_class_counts"][name] = summary["quant_linear_class_counts"].get(name, 0) + 1

    write_json(output / "runtime_smoke.json", summary)
    return summary, quant_logits, encoded


def dense_reference(
    args: argparse.Namespace,
    output: Path,
    quant_logits: torch.Tensor,
    encoded: dict[str, torch.Tensor],
) -> dict[str, Any]:
    attention = "flash_attention_2" if is_flash_attn_2_available() else "eager"
    print(f"Loading dense BF16 reference for a bounded logit comparison ({attention})", flush=True)
    started = time.perf_counter()
    dense = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL),
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation=attention,
    )
    dense.eval()
    dense_device = next(dense.parameters()).device
    dense_encoded = {key: value.to(dense_device) for key, value in encoded.items()}
    with torch.inference_mode():
        dense_logits = dense(**dense_encoded).logits.detach().float().cpu()
    wall_s = time.perf_counter() - started
    if dense_logits.shape != quant_logits.shape:
        raise RuntimeError(
            f"Dense/quant logit shape mismatch: {tuple(dense_logits.shape)} vs {tuple(quant_logits.shape)}"
        )
    if not torch.isfinite(dense_logits).all():
        raise RuntimeError("Dense reference produced non-finite logits")

    dense_last = dense_logits[:, -1, :]
    quant_last = quant_logits[:, -1, :]
    error = quant_last - dense_last
    cosine = torch.nn.functional.cosine_similarity(quant_last, dense_last, dim=-1).mean()
    dense_top5 = torch.topk(dense_last, k=5, dim=-1).indices
    quant_top5 = torch.topk(quant_last, k=5, dim=-1).indices
    top5_overlap = len(set(dense_top5[0].tolist()) & set(quant_top5[0].tolist()))
    result = {
        "reference": "dense BF16 base Qwen3-8B",
        "candidate": "quantized GPTQ plus EoRA rank 128" if args.eora == "enabled" else "native GPTQ without EoRA",
        "scope": "last-token logits for the runtime smoke prompt",
        "load_and_forward_wall_s": wall_s,
        "dense_dtype": str(next(dense.parameters()).dtype),
        "shape": list(dense_last.shape),
        "mae": error.abs().mean().item(),
        "rmse": error.square().mean().sqrt().item(),
        "max_abs_error": error.abs().max().item(),
        "cosine_similarity": cosine.item(),
        "dense_top1_token_id": int(dense_top5[0, 0]),
        "quant_top1_token_id": int(quant_top5[0, 0]),
        "top1_agrees": bool(dense_top5[0, 0] == quant_top5[0, 0]),
        "top5_overlap_count": top5_overlap,
    }
    write_json(output / "dense_reference.json", result)
    del dense, dense_logits, dense_last, quant_last, error
    torch_empty_cache()
    return result


def evaluate_tasks(
    args: argparse.Namespace,
    model: Any,
    output: Path,
) -> dict[str, Any]:
    evaluation_dir = output / "evalution"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    backend = BACKEND.MARLIN if args.eval_backend == "marlin" else BACKEND.GPTQ_TORCH

    for task in requested_eval_tasks(args):
        result_path = evaluation_result_path(output, task, args.eval_max_rows)
        if result_path.is_file():
            existing = json.loads(result_path.read_text())
            summaries[task] = existing["metrics"]
            print(f"[{args.label}] Reusing completed Evalution task {task}", flush=True)
            continue

        print(
            f"[{args.label}] Evalution starting task={task}, max_rows={args.eval_max_rows}, "
            f"batch_size={args.eval_batch_size}",
            flush=True,
        )
        started = time.perf_counter()
        result = evaluate(
            model_or_id_or_path=model,
            tasks=[task],
            batch_size=args.eval_batch_size,
            backend=backend,
            model_args={"device": "cuda:0", "seed": SEED, "random_seed": SEED},
            apply_chat_template=False,
            gen_kwargs="do_sample=false,temperature=0.0,top_p=1.0,top_k=50,max_new_tokens=256",
            suite_kwargs={"max_rows": args.eval_max_rows} if args.eval_max_rows is not None else {},
        )
        wall_s = time.perf_counter() - started
        task_metrics = get_eval_task_results(result).get(task, {})
        if not task_metrics:
            raise RuntimeError(f"Evalution returned no metrics for {task}: {result}")
        for metric_name, metric_value in task_metrics.items():
            if isinstance(metric_value, (int, float)) and not math.isfinite(float(metric_value)):
                raise RuntimeError(f"Evalution returned non-finite {task}:{metric_name}={metric_value}")

        write_json(
            result_path,
            {
                "task": task,
                "full_dataset": args.eval_max_rows is None,
                "max_rows": args.eval_max_rows,
                "batch_size": args.eval_batch_size,
                "apply_chat_template": False,
                "wall_s": wall_s,
                "metrics": task_metrics,
                "result": result,
            },
        )
        summaries[task] = task_metrics
        print(f"[{args.label}] Evalution completed {task}: {task_metrics}", flush=True)
        torch_empty_cache()

    write_json(
        output / (
            "evaluation_complete.json"
            if args.eval_max_rows is None
            else f"evaluation_complete.maxrows{args.eval_max_rows}.json"
        ),
        {
            "completed_at": utc_now(),
            "backend": args.eval_backend,
            "tasks": summaries,
        },
    )
    return summaries


def run_evaluation(args: argparse.Namespace, output: Path, adapter_dir: Path) -> None:
    validate_saved_metadata(args, output, adapter_dir)
    backend = BACKEND.MARLIN if args.eval_backend == "marlin" else BACKEND.GPTQ_TORCH
    adapter = Lora(rank=EORA_RANK, path=str(adapter_dir)) if args.eora == "enabled" else None
    attention = "flash_attention_2" if is_flash_attn_2_available() else "eager"
    print(
        f"[{args.label}] Reloading saved model with backend={backend.value}, EoRA={args.eora}",
        flush=True,
    )
    load_started = time.perf_counter()
    model = GPTQModel.load(
        str(output),
        backend=backend,
        adapter=adapter,
        dtype="auto",
        device_map={"": "cuda:0"},
        attn_implementation=attention,
    )
    load_wall_s = time.perf_counter() - load_started
    model.eval()

    runtime, quant_logits, encoded = validate_loaded_runtime(args, model, output)
    runtime["reload_wall_s"] = load_wall_s
    write_json(output / "runtime_smoke.json", runtime)

    if not (output / "dense_reference.json").is_file():
        dense_reference(args, output, quant_logits, encoded)
    else:
        print(f"[{args.label}] Reusing dense-reference comparison", flush=True)
    del quant_logits, encoded
    torch_empty_cache()

    summaries = evaluate_tasks(args, model, output)
    del model
    torch_empty_cache()
    print(f"[{args.label}] All requested validations completed: {summaries}", flush=True)


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    adapter_dir = output / "eora-rank128"
    output.mkdir(parents=True, exist_ok=True)

    cpu_threads = configure_cpu_threads()
    device_pool_workers = configure_device_thread_pool()
    gpu = assert_device(args)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    manifest = {
        "label": args.label,
        "started_at": utc_now(),
        "repository": {
            "path": str(REPO_ROOT),
            "head": run_text(["git", "rev-parse", "HEAD"]),
            "status_short": run_text(["git", "status", "--short"]),
        },
        "base_model": str(BASE_MODEL),
        "base_model_config": json.loads((BASE_MODEL / "config.json").read_text()),
        "output": str(output),
        "adapter_output": str(adapter_dir) if args.eora == "enabled" else None,
        "gpu": gpu,
        "versions": package_versions(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_DEVICE_ORDER",
                "CUDA_VISIBLE_DEVICES",
                "PYTORCH_ALLOC_CONF",
                "TORCH_CUDA_ARCH_LIST",
                "MAX_JOBS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "GPTQ_RUN_TORCH_THREADS",
                "GPTQMODEL_PACK_THREADS",
            )
        },
        "cpu_threads": cpu_threads,
        "device_pool_workers": device_pool_workers,
        "nvidia_smi": run_text(
            [
                "nvidia-smi",
                "--query-gpu=index,pci.bus_id,name,uuid,memory.total,memory.used,compute_cap",
                "--format=csv,noheader",
            ]
        ),
        "arguments": vars(args),
        "evaluation": {
            "framework": "Evalution",
            "tasks": list(requested_eval_tasks(args)),
            "full_dataset": args.eval_max_rows is None,
            "max_rows": args.eval_max_rows,
            "apply_chat_template": False,
            "batch_size": args.eval_batch_size,
            "seed": SEED,
        },
    }
    manifest_path = output / "run_manifest.json"
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text())
        existing_manifest.setdefault("quantization_gpu", existing_manifest.get("gpu"))
        existing_manifest["last_invocation_at"] = utc_now()
        existing_manifest["last_invocation_arguments"] = vars(args)
        existing_manifest["last_invocation_gpu"] = manifest["gpu"]
        existing_manifest["environment"] = manifest["environment"]
        existing_manifest["cpu_threads"] = manifest["cpu_threads"]
        existing_manifest["device_pool_workers"] = manifest["device_pool_workers"]
        existing_manifest["versions"] = manifest["versions"]
        existing_manifest["last_invocation_nvidia_smi"] = manifest["nvidia_smi"]
        existing_manifest["evaluation"] = manifest["evaluation"]
        manifest = existing_manifest
    write_json(manifest_path, manifest)

    if args.stage in {"all", "quantize"}:
        quantize(args, output, adapter_dir, manifest)
    if args.stage in {"all", "eval"}:
        run_evaluation(args, output, adapter_dir)

    write_json(
        output / "run_complete.json",
        {
            "completed_at": utc_now(),
            "label": args.label,
            "stage": args.stage,
        },
    )


if __name__ == "__main__":
    main()
