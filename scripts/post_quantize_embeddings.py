#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Post-quantize input embeddings and an untied LM head in an existing GPTQ checkpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gptqmodel import BACKEND, GPTQModel, QuantizeEmbed, QuantizeEmbedConfig  # noqa: E402
from gptqmodel.nn_modules.qlinear import BaseQuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.torch import TorchQuantEmbeddings  # noqa: E402
from gptqmodel.quantization.config import VramStrategy  # noqa: E402
from gptqmodel.utils.paroquant_benchmark import load_nm_calibration  # noqa: E402


# Defaults reproduce the Qwen3-8B source and calibration regime recorded in the repository reports.
DEFAULT_SOURCE = Path(
    "/monster/data/model/Qwen3-8B-Base-GPTQ-4bit-g128-activation-GAR-cal512"
)
DEFAULT_OUTPUT = Path(
    "/monster/data/model/Qwen3-8B-Base-GPTQ-4bit-g128-activation-GAR-postquant-embed-lmhead-8bit-g128"
)
DEFAULT_RESULT = Path("post_quantize_embeddings_result.json")
DEFAULT_PROMPT = "Solve carefully: If a box has 17 marbles and 8 are removed, how many remain?"
SEED = 898


def _run_text(command: list[str]) -> str:
    """Run a metadata command without making the benchmark depend on its success."""

    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return result.stdout.strip()


def _set_seed(seed: int) -> None:
    """Make the calibration and diagnostic prompt path deterministic."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _hardware_metadata() -> dict[str, Any]:
    """Capture the visible accelerator contract used by the conversion."""

    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "visible_index": index,
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "sm_count": properties.multi_processor_count,
                "total_memory_bytes": properties.total_memory,
            }
        )
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": _run_text(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name,compute_cap,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "devices": devices,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "python_gil_env": os.environ.get("PYTHON_GIL"),
        "python_gil_enabled": bool(getattr(sys, "_is_gil_enabled", lambda: True)()),
    }


def _exact_dynamic_pattern(module_name: str) -> str:
    """Build an ordered exact-match dynamic-config rule for one module."""

    return f"+:^{re.escape(module_name)}$"


def _configure_embedding_overrides(model, *, bits: int, group_size: int) -> dict[str, Any]:
    """Keep decoder metadata intact while assigning the requested target-module contract."""

    input_name = model.get_input_embeddings_name()
    output_name = model.get_output_embeddings_name()
    if not input_name or not output_name:
        raise ValueError(f"Could not resolve embedding targets: input={input_name!r}, output={output_name!r}.")
    if input_name == output_name or model.config.tie_word_embeddings:
        raise NotImplementedError("This driver requires distinct, untied input and output embedding modules.")

    override = {
        "bits": bits,
        "group_size": group_size,
        "sym": True,
        "desc_act": False,
        "act_group_aware": True,
        "mse": 2.0,
        "scale_search": "activation",
    }
    input_pattern = _exact_dynamic_pattern(input_name)
    output_pattern = _exact_dynamic_pattern(output_name)
    existing = {
        pattern: value
        for pattern, value in (model.quantize_config.dynamic or {}).items()
        if pattern not in {input_pattern, output_pattern}
    }
    model.quantize_config.dynamic = {
        input_pattern: dict(override),
        output_pattern: dict(override),
        **existing,
    }
    model.quantize_config.offload_to_disk = False
    return {
        "input_name": input_name,
        "output_name": output_name,
        "override": override,
    }


def _configure_execution(model, *, require_multi_gpu: bool) -> dict[str, Any]:
    """Enable all visible CUDA work queues and reject ineffective multi-GPU runtimes."""

    device_count = torch.cuda.device_count()
    if require_multi_gpu and device_count < 2:
        raise RuntimeError(f"Expected at least two visible CUDA devices, found {device_count}.")
    gil_enabled = bool(getattr(sys, "_is_gil_enabled", lambda: True)())
    if device_count > 1 and (os.environ.get("PYTHON_GIL") != "0" or gil_enabled):
        raise RuntimeError(
            "Multi-GPU post-quantization requires starting Python with `PYTHON_GIL=0`; "
            f"env={os.environ.get('PYTHON_GIL')!r}, gil_enabled={gil_enabled}."
        )

    visible_devices = [f"cuda:{index}" for index in range(device_count)]
    if device_count > 1:
        model.quantize_config.auto_forward_data_parallel = True
        model.quantize_config.calibration_data_device = "balanced"
        # ForwardExecutor must own the topology here: EXCLUSIVE with no fixed
        # placement map permits it to replicate each active module and split
        # calibration batches across every visible device. BALANCED is module
        # placement, which deliberately makes dense subset forward serial.
        model.quantize_config.dense_vram_strategy = VramStrategy.EXCLUSIVE
        model.quantize_config.dense_vram_strategy_devices = None
    return {
        "visible_device_count": device_count,
        "visible_devices": visible_devices,
        "python_gil_env": os.environ.get("PYTHON_GIL"),
        "python_gil_enabled": gil_enabled,
        "auto_forward_data_parallel": model.quantize_config.auto_forward_data_parallel,
        "calibration_data_device": model.quantize_config.calibration_data_device,
        "dense_vram_strategy": model.quantize_config.dense_vram_strategy.value,
        "dense_vram_strategy_devices": model.quantize_config.dense_vram_strategy_devices,
    }


def _sample_rows(row_count: int, token_ids: torch.Tensor, sample_count: int = 64) -> torch.Tensor:
    """Choose deterministic boundary, prompt-token, and evenly spaced weight rows."""

    rows = {0, max(0, row_count - 1)}
    rows.update(int(value) for value in token_ids.detach().cpu().reshape(-1).tolist())
    if row_count > 1:
        rows.update(
            int(value)
            for value in torch.linspace(0, row_count - 1, steps=min(sample_count, row_count)).round().tolist()
        )
    return torch.tensor(sorted(value for value in rows if 0 <= value < row_count), dtype=torch.long)


def _dense_weight_rows(module: torch.nn.Module, rows: torch.Tensor) -> torch.Tensor:
    """Copy selected semantic `[out, in]` dense rows to CPU FP32."""

    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"Expected a dense weight tensor, found {type(weight)}.")
    return weight.detach().index_select(0, rows.to(weight.device)).float().cpu()


def _dequantized_weight_rows(module: BaseQuantLinear, rows: torch.Tensor) -> torch.Tensor:
    """Copy selected semantic `[out, in]` rows from a packed GPTQ module."""

    dequantized = module.dequantize_weight()
    if not isinstance(module, TorchQuantEmbeddings):
        dequantized = dequantized.T
    return dequantized.index_select(0, rows.to(dequantized.device)).float().cpu()


def _error_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    """Measure bounded weight error against the pre-quantized BF16 targets."""

    delta = actual - reference
    reference_flat = reference.reshape(-1)
    actual_flat = actual.reshape(-1)
    denominator = float(torch.linalg.vector_norm(reference_flat) * torch.linalg.vector_norm(actual_flat))
    cosine = float(torch.dot(reference_flat, actual_flat) / denominator) if denominator else math.nan
    return {
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "rmse": float(torch.sqrt(torch.mean(delta.square()))),
        "cosine": cosine,
        "reference_norm": float(torch.linalg.vector_norm(reference_flat)),
        "actual_norm": float(torch.linalg.vector_norm(actual_flat)),
    }


def _last_token_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Run one deterministic forward and retain only the last-token logits."""

    input_device = next(
        (tensor.device for tensor in model.get_input_embeddings().buffers() if tensor.device.type != "meta"),
        torch.device("cuda:0"),
    )
    with torch.inference_mode():
        output = model.model(input_ids=input_ids.to(input_device), use_cache=False)
    return output.logits[0, -1].float().cpu()


def _logit_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    """Compare last-token logits without retaining a full-sequence output."""

    delta = actual - reference
    denominator = float(torch.linalg.vector_norm(reference) * torch.linalg.vector_norm(actual))
    return {
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "rmse": float(torch.sqrt(torch.mean(delta.square()))),
        "cosine": float(torch.dot(reference, actual) / denominator) if denominator else math.nan,
        "reference_top1": int(reference.argmax()),
        "actual_top1": int(actual.argmax()),
        "top1_match": bool(reference.argmax() == actual.argmax()),
    }


def _qweight_sha256(module: BaseQuantLinear) -> str:
    """Fingerprint a packed decoder tensor to detect unintended requantization."""

    qweight = module.qweight.detach().contiguous().cpu()
    return hashlib.sha256(qweight.numpy().tobytes()).hexdigest()


def _decoder_fingerprints(model, excluded_names: set[str]) -> dict[str, str]:
    """Fingerprint the first and last decoder modules as lifecycle sentinels."""

    modules = [
        (name, module)
        for name, module in model.model.named_modules()
        if isinstance(module, BaseQuantLinear) and name not in excluded_names
    ]
    if not modules:
        raise RuntimeError("Loaded source contains no quantized decoder modules.")
    selected = dict((modules[0], modules[-1]))
    return {name: _qweight_sha256(module) for name, module in selected.items()}


def _module_contract(module: torch.nn.Module) -> dict[str, Any]:
    """Serialize the runtime fields that define a packed target module."""

    return {
        "class": type(module).__name__,
        "bits": getattr(module, "bits", None),
        "group_size": getattr(module, "group_size", None),
        "sym": getattr(module, "sym", None),
        "desc_act": getattr(module, "desc_act", None),
        "in_features": getattr(module, "in_features", None),
        "out_features": getattr(module, "out_features", None),
        "qweight_shape": list(module.qweight.shape) if isinstance(module, BaseQuantLinear) else None,
        "qzeros_shape": list(module.qzeros.shape) if isinstance(module, BaseQuantLinear) else None,
        "scales_shape": list(module.scales.shape) if isinstance(module, BaseQuantLinear) else None,
    }


def _decoder_contract(model, excluded_names: set[str]) -> dict[str, Any]:
    """Resolve the uniform decoder contract without assuming a source bit width."""

    decoder_modules = [
        module
        for name, module in model.model.named_modules()
        if isinstance(module, BaseQuantLinear) and name not in excluded_names
    ]
    if len(decoder_modules) != 252:
        raise AssertionError(f"Expected 252 Qwen3-8B decoder projections, found {len(decoder_modules)}.")

    first = decoder_modules[0]
    contract = {
        "bits": first.bits,
        "group_size": first.group_size,
        "sym": first.sym,
        "desc_act": first.desc_act,
    }
    expected = tuple(contract.values())
    if any(
        (module.bits, module.group_size, module.sym, module.desc_act) != expected
        for module in decoder_modules
    ):
        raise AssertionError("The source checkpoint has a non-uniform decoder GPTQ contract.")
    return {**contract, "module_count": len(decoder_modules)}


def _validate_target_contract(
    model,
    target: dict[str, Any],
    *,
    bits: int,
    group_size: int,
    decoder_contract: dict[str, Any],
) -> dict[str, Any]:
    """Require both targets and every pre-existing decoder module to retain their intended widths."""

    modules = dict(model.model.named_modules())
    input_module = modules[target["input_name"]]
    output_module = modules[target["output_name"]]
    for name, module in ((target["input_name"], input_module), (target["output_name"], output_module)):
        if not isinstance(module, BaseQuantLinear):
            raise TypeError(f"Expected `{name}` to reload as BaseQuantLinear, found {type(module)}.")
        actual = (module.bits, module.group_size, module.sym, module.desc_act)
        expected = (bits, group_size, True, False)
        if actual != expected:
            raise AssertionError(f"Unexpected target contract for `{name}`: expected={expected}, actual={actual}.")

    decoder_modules = [
        module
        for name, module in modules.items()
        if isinstance(module, BaseQuantLinear) and name not in {target["input_name"], target["output_name"]}
    ]
    if len(decoder_modules) != 252:
        raise AssertionError(f"Expected 252 Qwen3-8B decoder projections, found {len(decoder_modules)}.")
    expected_decoder = (
        decoder_contract["bits"],
        decoder_contract["group_size"],
        decoder_contract["sym"],
        decoder_contract["desc_act"],
    )
    if any(
        (module.bits, module.group_size, module.sym, module.desc_act) != expected_decoder
        for module in decoder_modules
    ):
        raise AssertionError("At least one pre-existing decoder projection changed its GPTQ contract.")

    return {
        "input": _module_contract(input_module),
        "output": _module_contract(output_module),
        "decoder_module_count": len(decoder_modules),
        "decoder_contract": {
            key: value
            for key, value in decoder_contract.items()
            if key != "module_count"
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist the conversion manifest atomically enough for a single-process experiment."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> None:
    """Execute post-quantization, save/reload validation, and numerical comparisons."""

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty checkpoint directory: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _set_seed(args.seed)

    model = GPTQModel.load(
        str(args.source),
        backend=BACKEND.GPTQ_TORCH,
        dtype="auto",
        # Keep one authoritative module tree. ModuleLooper owns multi-GPU
        # calibration cloning/placement; an Accelerate-sharded tree leaves
        # cross-device hooks behind when embed-only replay reaches model.norm.
        device_map={"": "cuda:0"},
        trust_remote_code=False,
    )
    if model.config.tie_word_embeddings:
        raise NotImplementedError("Qwen3 post-quantization requires untied input and output embeddings.")

    execution = _configure_execution(model, require_multi_gpu=args.require_multi_gpu)
    target = _configure_embedding_overrides(model, bits=args.bits, group_size=args.group_size)
    tokenizer_output = model.tokenizer(args.prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = tokenizer_output["input_ids"]
    input_rows = _sample_rows(model.config.vocab_size, input_ids)
    output_rows = input_rows.clone()
    input_reference = _dense_weight_rows(model.get_input_embeddings(), input_rows)
    output_reference = _dense_weight_rows(model.get_output_embeddings(), output_rows)
    excluded_names = {target["input_name"], target["output_name"]}
    decoder_contract = _decoder_contract(model, excluded_names)
    act_group_aware = getattr(model.quantize_config, "act_group_aware", None)
    if act_group_aware is None:
        act_group_aware = (getattr(model.quantize_config, "meta", None) or {}).get("act_group_aware", False)
    base_quantization = {
        "method": "gptq",
        **decoder_contract,
        "act_group_aware": bool(act_group_aware),
    }
    decoder_before = _decoder_fingerprints(model, excluded_names)
    logits_before = _last_token_logits(model, input_ids)

    calibration = load_nm_calibration(args.calibration_rows)
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    quant_started = time.perf_counter()
    quant_log = model.requantize(
        calibration=calibration,
        calibration_concat_size=args.calibration_concat_size,
        calibration_sort="desc",
        batch_size=args.batch_size,
        backend=BACKEND.GPTQ_TORCH,
        embed_quant_config=QuantizeEmbedConfig(
            embed_quant_mode=QuantizeEmbed.BOTH,
            embed_only=True,
        ),
    )
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)
    quant_wall_seconds = time.perf_counter() - quant_started
    quant_peak_allocated_bytes = {
        str(index): torch.cuda.max_memory_allocated(index)
        for index in range(torch.cuda.device_count())
    }
    quant_region_snapshot = model.quant_region_timer.snapshot()

    direct_contract = _validate_target_contract(
        model,
        target,
        bits=args.bits,
        group_size=args.group_size,
        decoder_contract=decoder_contract,
    )
    input_direct = _dequantized_weight_rows(model.get_input_embeddings(), input_rows)
    output_direct = _dequantized_weight_rows(model.get_output_embeddings(), output_rows)
    decoder_direct = _decoder_fingerprints(model, {target["input_name"], target["output_name"]})
    if decoder_direct != decoder_before:
        raise AssertionError("A sampled decoder qweight changed during embedding-only post-quantization.")

    save_started = time.perf_counter()
    model.save(str(args.output))
    save_wall_seconds = time.perf_counter() - save_started
    del model, calibration
    gc.collect()
    torch.cuda.empty_cache()

    reload_started = time.perf_counter()
    reloaded = GPTQModel.load(
        str(args.output),
        backend=BACKEND.GPTQ_TORCH,
        dtype="auto",
        device_map="auto",
        trust_remote_code=False,
    )
    reload_wall_seconds = time.perf_counter() - reload_started
    reload_contract = _validate_target_contract(
        reloaded,
        target,
        bits=args.bits,
        group_size=args.group_size,
        decoder_contract=decoder_contract,
    )
    input_reloaded = _dequantized_weight_rows(reloaded.get_input_embeddings(), input_rows)
    output_reloaded = _dequantized_weight_rows(reloaded.get_output_embeddings(), output_rows)
    logits_reloaded = _last_token_logits(reloaded, input_ids)
    decoder_reloaded = _decoder_fingerprints(reloaded, {target["input_name"], target["output_name"]})
    if decoder_reloaded != decoder_before:
        raise AssertionError("A sampled decoder qweight changed after saving and reloading.")

    payload = {
        "schema": "gptqmodel-post-quant-embeddings-v1",
        "source": str(args.source),
        "output": str(args.output),
        "seed": args.seed,
        "hardware": _hardware_metadata(),
        "execution": execution,
        "base_quantization": base_quantization,
        "post_quantization": {
            **target,
            "calibration_rows": args.calibration_rows,
            "calibration_concat_size": args.calibration_concat_size,
            "calibration_sort": "desc",
            "batch_size": args.batch_size,
            "backend": BACKEND.GPTQ_TORCH.value,
            "pack_impl": reloaded.quantize_config.pack_impl,
            "quant_wall_seconds": quant_wall_seconds,
            "save_wall_seconds": save_wall_seconds,
            "reload_wall_seconds": reload_wall_seconds,
            "peak_allocated_bytes": quant_peak_allocated_bytes,
        },
        "contracts": {
            "direct": direct_contract,
            "reloaded": reload_contract,
        },
        "decoder_qweight_sha256": {
            "before": decoder_before,
            "direct": decoder_direct,
            "reloaded": decoder_reloaded,
        },
        "sampled_weight_error": {
            "input_direct_vs_dense": _error_metrics(input_reference, input_direct),
            "input_reload_vs_dense": _error_metrics(input_reference, input_reloaded),
            "input_reload_vs_direct": _error_metrics(input_direct, input_reloaded),
            "output_direct_vs_dense": _error_metrics(output_reference, output_direct),
            "output_reload_vs_dense": _error_metrics(output_reference, output_reloaded),
            "output_reload_vs_direct": _error_metrics(output_direct, output_reloaded),
            "input_sample_rows": input_rows.tolist(),
            "output_sample_rows": output_rows.tolist(),
        },
        "last_token_logits": {
            "prompt": args.prompt,
            "input_ids": input_ids.tolist(),
            "reload_vs_pre": _logit_metrics(logits_before, logits_reloaded),
            "direct_post_quant_inference": "not run; the quantization lifecycle is validated after save/reload",
        },
        "quant_log": quant_log,
        "quant_region_snapshot": quant_region_snapshot,
    }
    _write_json(args.result, payload)
    print(json.dumps({key: payload[key] for key in ("source", "output", "post_quantization")}, indent=2))
    print(json.dumps(payload["sampled_weight_error"], indent=2))
    print(json.dumps(payload["last_token_logits"], indent=2))
    print(f"result_json={args.result}")


def parse_args() -> argparse.Namespace:
    """Parse the reproducible post-quantization contract."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--calibration-rows", type=int, default=512)
    parser.add_argument("--calibration-concat-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--require-multi-gpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
