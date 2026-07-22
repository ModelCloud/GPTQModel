#!/usr/bin/env python3
"""Localize the Qwen3-8B GPTQ INT3 quality collapse against dense BF16."""

# ruff: noqa: E402

from __future__ import annotations

import ast
import json
import os
import statistics
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


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

import torch
from transformers import AutoModelForCausalLM

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.adapter.adapter import Lora
from gptqmodel.nn_modules.qlinear import PackableQuantLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.utils.torch import torch_empty_cache


MODEL = Path("/monster/data/model/Qwen3-8B-GPTQ-3bit-g64-EoRA-r128-cal512")
BASE = Path("/monster/data/model/Qwen3-8B")
ADAPTER = MODEL / "eora-rank128"
OUTPUT = MODEL / "int3_layerwise_diagnostic.json"
QUANT_LOG = REPO_ROOT / "artifacts/qwen3_8b_dual_gptq_eora_20260722/gpu7.log"
EXPECTED_UUID = "724ea08e-67c3-c0ce-29bb-e6c48e7dde28"
EXPECTED_PCI_BUS = "00000000:E4:00.0"
PROMPT = "The capital city of France is"


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def assert_gpu() -> dict[str, Any]:
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID" or torch.cuda.device_count() != 1:
        raise RuntimeError("Diagnostic requires one PCI-ordered visible GPU")
    properties = torch.cuda.get_device_properties(0)
    actual_uuid = str(properties.uuid).removeprefix("GPU-").lower()
    actual_bus = int(properties.pci_bus_id)
    expected_bus = int(EXPECTED_PCI_BUS.split(":")[-2], 16)
    if actual_uuid != EXPECTED_UUID or actual_bus != expected_bus:
        raise RuntimeError(
            f"Expected GPU 7 uuid={EXPECTED_UUID}, bus={expected_bus:#x}; "
            f"found uuid={actual_uuid}, bus={actual_bus:#x}"
        )
    return {
        "physical_index_pci_order": 7,
        "process_cuda_index": 0,
        "pci_bus_id": EXPECTED_PCI_BUS,
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
    denominator = candidate_flat.norm() * reference_flat.norm()
    cosine = (candidate_flat @ reference_flat) / denominator if denominator > 0 else torch.tensor(float("nan"))
    result = {
        "shape": list(candidate.shape),
        "all_finite": bool(torch.isfinite(candidate_f).all().item()),
        "mae": delta.abs().mean().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "max_abs_error": delta.abs().max().item(),
        "cosine_similarity": cosine.item(),
        "candidate_norm": candidate_flat.norm().item(),
        "reference_norm": reference_flat.norm().item(),
    }
    del candidate_f, reference_f, delta, candidate_flat, reference_flat, denominator, cosine
    return result


def last_token_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    result = tensor_metrics(candidate[:, -1, :], reference[:, -1, :])
    candidate_last = candidate[:, -1, :]
    reference_last = reference[:, -1, :]
    result.update(
        {
            "candidate_top1_token_id": int(candidate_last.argmax(dim=-1).item()),
            "reference_top1_token_id": int(reference_last.argmax(dim=-1).item()),
            "top1_agrees": bool(candidate_last.argmax(dim=-1).item() == reference_last.argmax(dim=-1).item()),
        }
    )
    return result


@contextmanager
def adapter_mode(modules: list[tuple[str, TorchLinear]], *, enabled: bool) -> Iterator[None]:
    saved = [module.adapter for _, module in modules]
    if not enabled:
        for _, module in modules:
            module.adapter = None
    try:
        yield
    finally:
        if not enabled:
            for (_, module), adapter in zip(modules, saved):
                module.adapter = adapter


def run_with_hidden_states(model: Any, encoded: dict[str, torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    with torch.inference_mode():
        output = model(
            **encoded,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    logits = output.logits.detach().float().cpu()
    hidden_states = [hidden.detach().float().cpu() for hidden in output.hidden_states]
    return logits, hidden_states


def compare_hidden_states(candidate: list[torch.Tensor], reference: list[torch.Tensor]) -> list[dict[str, Any]]:
    if len(candidate) != len(reference):
        raise RuntimeError(f"Hidden-state count mismatch: {len(candidate)} vs {len(reference)}")
    records = []
    for index, (candidate_hidden, reference_hidden) in enumerate(zip(candidate, reference)):
        name = "embedding_output" if index == 0 else f"decoder_layer_{index - 1}_output"
        metrics = tensor_metrics(candidate_hidden, reference_hidden)
        metrics.update({"index": index, "name": name})
        records.append(metrics)
    return records


def parse_quantization_log() -> dict[str, Any]:
    entries = []
    for line in QUANT_LOG.read_text(errors="replace").replace("\r", "\n").splitlines():
        start = line.find("{'process': 'gptq'")
        if start < 0:
            continue
        end = line.find("}", start)
        if end < 0:
            continue
        try:
            payload = ast.literal_eval(line[start : end + 1])
        except (SyntaxError, ValueError):
            continue
        if payload.get("process") == "gptq":
            entries.append(payload)

    if len(entries) < 252:
        raise RuntimeError(f"Expected at least 252 GPTQ log records, found {len(entries)}")
    entries = entries[-252:]
    identities = {(int(entry["layer"]), str(entry["module"])) for entry in entries}
    if len(identities) != 252:
        raise RuntimeError(f"Final GPTQ log block has {len(identities)} unique modules, expected 252")

    normalized = []
    for entry in entries:
        normalized.append(
            {
                "layer": int(entry["layer"]),
                "module": str(entry["module"]),
                "loss": float(entry["loss"]),
                "samples": int(entry["samples"]),
                "damp": float(entry["damp"]),
                "time_s": float(entry["time"]),
                "forward_time_s": float(entry["fwd_time"]),
            }
        )
    by_layer = []
    for layer in range(36):
        values = [entry["loss"] for entry in normalized if entry["layer"] == layer]
        by_layer.append(
            {
                "layer": layer,
                "mean_loss": statistics.fmean(values),
                "max_loss": max(values),
            }
        )
    return {
        "records": normalized,
        "by_layer": by_layer,
        "worst_modules": sorted(normalized, key=lambda item: item["loss"], reverse=True)[:20],
    }


def compare_non_quantized_weights(quant_model: Any, dense_model: Any) -> list[dict[str, Any]]:
    names = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    for layer in range(36):
        names.extend(
            [
                f"model.layers.{layer}.input_layernorm.weight",
                f"model.layers.{layer}.post_attention_layernorm.weight",
            ]
        )
    records = []
    for name in names:
        candidate = quant_model.get_parameter(name)
        reference = dense_model.get_parameter(name)
        metrics = tensor_metrics(candidate, reference)
        metrics.update({"name": name, "exact": bool(torch.equal(candidate, reference))})
        records.append(metrics)
    return records


def compare_quantized_weights(
    modules: list[tuple[str, TorchLinear]],
    dense_model: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    for index, (name, module) in enumerate(modules, start=1):
        dense_weight = dense_model.get_submodule(name).weight.detach().T
        with torch.inference_mode():
            dequantized = PackableQuantLinear.dequantize_weight(module)
        base_metrics = tensor_metrics(dequantized, dense_weight)

        adapter = module.adapter
        if adapter is None or adapter.lora_A is None or adapter.lora_B is None:
            raise RuntimeError(f"Missing EoRA tensors for {name}")
        with torch.inference_mode():
            eora_delta = torch.matmul(adapter.lora_A.float(), adapter.lora_B.float())
            effective = dequantized.float() + eora_delta
        effective_metrics = tensor_metrics(effective, dense_weight)

        reference_error = dense_weight.float() - dequantized.float()
        error_flat = reference_error.reshape(-1)
        delta_flat = eora_delta.reshape(-1)
        denominator = error_flat.norm() * delta_flat.norm()
        delta_error_cosine = (
            ((error_flat @ delta_flat) / denominator).item() if denominator > 0 else float("nan")
        )
        residual_ratio = (reference_error - eora_delta).norm().div(reference_error.norm()).item()

        records.append(
            {
                "index": index,
                "name": name,
                "layer": int(name.split(".")[2]),
                "module_type": name.rsplit(".", 1)[-1],
                "bits": int(module.bits),
                "group_size": int(module.group_size),
                "shape": list(dequantized.shape),
                "qzero_format": int(module.qzero_format()),
                "scale_min": module.scales.float().min().item(),
                "scale_max": module.scales.float().max().item(),
                "scale_mean": module.scales.float().mean().item(),
                "base_vs_dense": base_metrics,
                "effective_eora_vs_dense": effective_metrics,
                "eora_delta_vs_true_error_cosine": delta_error_cosine,
                "eora_residual_norm_ratio": residual_ratio,
                "eora_improves_rmse": effective_metrics["rmse"] < base_metrics["rmse"],
            }
        )
        del dequantized, dense_weight, eora_delta, effective, reference_error, error_flat, delta_flat, denominator
        if index % 12 == 0:
            print(f"Compared {index}/{len(modules)} quantized weights", flush=True)
            torch_empty_cache()

    base_cosines = [record["base_vs_dense"]["cosine_similarity"] for record in records]
    effective_cosines = [record["effective_eora_vs_dense"]["cosine_similarity"] for record in records]
    summary = {
        "module_count": len(records),
        "base_cosine_min": min(base_cosines),
        "base_cosine_median": statistics.median(base_cosines),
        "base_cosine_mean": statistics.fmean(base_cosines),
        "effective_cosine_min": min(effective_cosines),
        "effective_cosine_median": statistics.median(effective_cosines),
        "effective_cosine_mean": statistics.fmean(effective_cosines),
        "eora_improves_rmse_count": sum(record["eora_improves_rmse"] for record in records),
        "eora_worsens_rmse_count": sum(not record["eora_improves_rmse"] for record in records),
        "eora_delta_error_cosine_median": statistics.median(
            record["eora_delta_vs_true_error_cosine"] for record in records
        ),
        "eora_residual_norm_ratio_median": statistics.median(
            record["eora_residual_norm_ratio"] for record in records
        ),
        "worst_base_cosine_modules": sorted(
            records,
            key=lambda item: item["base_vs_dense"]["cosine_similarity"],
        )[:20],
        "worst_effective_cosine_modules": sorted(
            records,
            key=lambda item: item["effective_eora_vs_dense"]["cosine_similarity"],
        )[:20],
        "worst_eora_alignment_modules": sorted(
            records,
            key=lambda item: item["eora_delta_vs_true_error_cosine"],
        )[:20],
    }
    return records, summary


def run_hybrid_localization(
    quant_model: Any,
    dense_model: Any,
    encoded: dict[str, torch.Tensor],
    dense_logits: torch.Tensor,
    modules: list[tuple[str, TorchLinear]],
) -> list[dict[str, Any]]:
    quant_layers = quant_model.model.model.layers
    dense_layers = dense_model.model.layers
    if len(quant_layers) != 36 or len(dense_layers) != 36:
        raise RuntimeError("Expected 36 decoder layers in both models")

    variants: list[tuple[str, list[int]]] = []
    for end in (6, 12, 18, 24, 30, 36):
        variants.append((f"dense_prefix_0_{end - 1}", list(range(0, end))))
    for start in (0, 6, 12, 18, 24, 30):
        variants.append((f"dense_suffix_{start}_35", list(range(start, 36))))
    for start in range(0, 36, 6):
        variants.append((f"dense_block_{start}_{start + 5}", list(range(start, start + 6))))

    records = []
    with adapter_mode(modules, enabled=False):
        for label, indices in variants:
            originals = {index: quant_layers[index] for index in indices}
            try:
                for index in indices:
                    quant_layers[index] = dense_layers[index]
                with torch.inference_mode():
                    logits = quant_model(**encoded, use_cache=False).logits.detach().float().cpu()
            finally:
                for index, original in originals.items():
                    quant_layers[index] = original
            metrics = last_token_metrics(logits, dense_logits)
            records.append(
                {
                    "variant": label,
                    "dense_layer_count": len(indices),
                    "dense_layers": indices,
                    "last_token_vs_dense": metrics,
                }
            )
            print(f"Hybrid {label}: cosine={metrics['cosine_similarity']:.6f}", flush=True)
    return records


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    gpu = assert_gpu()
    quant_log = parse_quantization_log()

    print("Loading INT3 checkpoint with EoRA using eager Torch dequantization", flush=True)
    quant = GPTQModel.load(
        str(MODEL),
        backend=BACKEND.GPTQ_TORCH,
        adapter=Lora(rank=128, path=str(ADAPTER)),
        dtype="auto",
        device_map={"": "cuda:0"},
        attn_implementation="flash_attention_2",
    )
    quant.eval()
    modules = [(name, module) for name, module in quant.model.named_modules() if isinstance(module, TorchLinear)]
    if len(modules) != 252 or any(module._triton_dequant_enabled for _, module in modules):
        raise RuntimeError("Expected 252 eager Torch INT3 modules")

    print("Loading dense BF16 reference", flush=True)
    dense = AutoModelForCausalLM.from_pretrained(
        str(BASE),
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="flash_attention_2",
    )
    dense.eval()

    tokenizer = quant.tokenizer
    encoded = {key: value.to("cuda:0") for key, value in tokenizer(PROMPT, return_tensors="pt").items()}
    dense_logits, dense_hidden = run_with_hidden_states(dense, encoded)
    with adapter_mode(modules, enabled=True):
        eora_logits, eora_hidden = run_with_hidden_states(quant, encoded)
    with adapter_mode(modules, enabled=False):
        no_eora_logits, no_eora_hidden = run_with_hidden_states(quant, encoded)

    non_quantized_weights = compare_non_quantized_weights(quant.model, dense)
    print("Comparing all 252 dequantized weights and EoRA deltas", flush=True)
    weight_records, weight_summary = compare_quantized_weights(modules, dense)
    print("Running dense/quant decoder-layer hybrids", flush=True)
    hybrids = run_hybrid_localization(quant, dense, encoded, dense_logits, modules)

    payload = {
        "gpu": gpu,
        "model": str(MODEL),
        "base_model": str(BASE),
        "adapter": str(ADAPTER),
        "backend": "gptq_torch eager dequantization",
        "prompt": PROMPT,
        "module_count": len(modules),
        "quantization_log": quant_log,
        "end_to_end": {
            "with_eora_last_token_vs_dense": last_token_metrics(eora_logits, dense_logits),
            "without_eora_last_token_vs_dense": last_token_metrics(no_eora_logits, dense_logits),
            "with_eora_hidden_state_drift": compare_hidden_states(eora_hidden, dense_hidden),
            "without_eora_hidden_state_drift": compare_hidden_states(no_eora_hidden, dense_hidden),
        },
        "non_quantized_weight_metrics": non_quantized_weights,
        "quantized_weight_summary": weight_summary,
        "quantized_weight_records": weight_records,
        "hybrid_decoder_localization": hybrids,
    }
    write_json(OUTPUT, payload)
    summary = {
        "output": str(OUTPUT),
        "with_eora_last_token_vs_dense": payload["end_to_end"]["with_eora_last_token_vs_dense"],
        "without_eora_last_token_vs_dense": payload["end_to_end"]["without_eora_last_token_vs_dense"],
        "quantized_weight_summary": weight_summary,
        "non_quantized_exact_count": sum(record["exact"] for record in non_quantized_weights),
        "non_quantized_total": len(non_quantized_weights),
        "hybrids": [
            {
                "variant": record["variant"],
                "cosine": record["last_token_vs_dense"]["cosine_similarity"],
            }
            for record in hybrids
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
