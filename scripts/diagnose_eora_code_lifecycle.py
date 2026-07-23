#!/usr/bin/env python3
"""Trace sampled GPTQ codes through a bounded native or zero-correction EoRA run."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import datasets
import gptqmodel
import torch
import transformers
from datasets import load_dataset
from transformers.utils import is_flash_attn_2_available

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig, ScaleSearchConfig
from gptqmodel.adapter.adapter import EoRAConfig, Lora
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.config import GcMode
from gptqmodel.utils.linalg_warmup import run_torch_linalg_warmup
from gptqmodel.utils.threadx import DeviceThreadPool, WarmUpCtx, WarmupTask


BASE_MODEL = Path("/monster/data/model/Qwen3-8B")
CALIBRATION_DATASET = Path("/monster/data/model/dataset/nm-calibration")
CALIBRATION_NAME = "LLM"
SEED = 898


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--eora", choices=("disabled", "zero"), required=True)
    parser.add_argument("--calibration-rows", type=int, default=8)
    parser.add_argument("--layer-count", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.physical_gpu not in range(6):
        parser.error("--physical-gpu must be in the allowed range 0..5")
    if args.calibration_rows <= 0:
        parser.error("--calibration-rows must be positive")
    if args.layer_count <= 0:
        parser.error("--layer-count must be positive")
    return args


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
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "value"):
        return jsonable(value.value)
    return repr(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def configure_runtime() -> dict[str, Any]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if getattr(gptqmodel, "_DEVICE_THREAD_POOL", None) is not None:
        raise RuntimeError("GPTQModel device pool initialized before diagnostic configuration")
    workers = {
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
        workers=workers,
        empty_cache_every_n=512,
    )
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected one visible CUDA device, got {torch.cuda.device_count()} "
            f"from CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )
    properties = torch.cuda.get_device_properties(0)
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "name": properties.name,
        "uuid": str(properties.uuid),
        "pci_bus_id": properties.pci_bus_id,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "sm_count": properties.multi_processor_count,
        "memory_bytes": properties.total_memory,
        "workers": workers,
    }


def install_zero_correction() -> None:
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


def main() -> None:
    args = parse_args()
    hardware = configure_runtime()
    config = json.loads((BASE_MODEL / "config.json").read_text(encoding="utf-8"))
    model_layer_count = int(config["num_hidden_layers"])
    if args.layer_count > model_layer_count:
        raise ValueError(f"layer count {args.layer_count} exceeds model depth {model_layer_count}")

    calibration = load_dataset(
        path=str(CALIBRATION_DATASET),
        name=CALIBRATION_NAME,
        split="train",
    ).select(range(args.calibration_rows))
    dynamic = {
        f"-:^model\\.layers\\.{layer_index}\\.": {}
        for layer_index in range(args.layer_count, model_layer_count)
    }
    adapter_dir = args.output.parent / f"{args.output.stem}-adapter"
    adapter = (
        Lora(
            rank=128,
            path=str(adapter_dir),
            eora_config=EoRAConfig(algo="lowrank"),
        )
        if args.eora == "zero"
        else None
    )
    quant_config = QuantizeConfig(
        quant_method=METHOD.GPTQ,
        format=FORMAT.GPTQ,
        bits=2,
        group_size=32,
        sym=True,
        desc_act=False,
        act_group_aware=True,
        scale_search=ScaleSearchConfig.ACTIVATION,
        mse=2.0,
        static_groups=False,
        quantization_diagnostics="channel",
        adapter=adapter,
        dynamic=dynamic,
        pack_impl="gpu",
        gc_mode=GcMode.ON_STAGE_END,
        wait_for_submodule_finalizers=True,
    )
    if args.eora == "zero":
        install_zero_correction()

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    attention = "flash_attention_2" if is_flash_attn_2_available() else "eager"
    started = time.perf_counter()
    model = GPTQModel.load(
        str(BASE_MODEL),
        quantize_config=quant_config,
        dtype="auto",
        device_map={"": "cuda:0"},
        attn_implementation=attention,
    )
    load_wall_s = time.perf_counter() - started
    quant_started = time.perf_counter()
    result = model.quantize(
        calibration=calibration,
        calibration_concat_size=None,
        calibration_sort="desc",
        batch_size=1,
        backend=BACKEND.GPTQ_TORCH,
        calibration_data_min_length=10,
    )
    quant_wall_s = time.perf_counter() - quant_started
    payload = {
        "label": args.label,
        "completed_at": utc_now(),
        "arguments": vars(args),
        "base_model": str(BASE_MODEL),
        "model_layer_count": model_layer_count,
        "quantized_layer_count": args.layer_count,
        "calibration_rows": len(calibration),
        "hardware": hardware,
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "gptqmodel": getattr(gptqmodel, "__version__", None),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
        },
        "quantization_config": quant_config.to_dict(),
        "load_wall_s": load_wall_s,
        "quantize_wall_s": quant_wall_s,
        "quantize_result_keys": sorted(result) if isinstance(result, dict) else None,
        "diagnostics": getattr(model, "quantization_diagnostics", None),
    }
    write_json(args.output, payload)
    print(json.dumps(jsonable(payload["diagnostics"]["code_fingerprints"]), indent=2), flush=True)
    print(f"Saved {args.output} after {quant_wall_s:.3f}s", flush=True)


if __name__ == "__main__":
    main()
