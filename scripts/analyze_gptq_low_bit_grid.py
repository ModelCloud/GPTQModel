#!/usr/bin/env python3
"""Measure low-bit GPTQ scale-grid error through a short decoder stack.

This diagnostic intentionally isolates grouped activation-weighted parameter
search and fake-quantized reconstruction before packing/backend effects.  It
reports both local module error (dense module inputs) and live propagated error
through a truncated model.  Intermediate KL metrics use channel distributions
standardized with dense activation statistics; final-logit KL uses raw logits.
"""

# ruff: noqa: E402 -- the strict GPU idle gate must run before importing Torch.

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import os
import platform
import re
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from scripts.gpu_idle_preflight import (
        add_gpu_idle_preflight_args,
        bootstrap_gpu_idle_preflight,
        recheck_gpu_exclusivity,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from gpu_idle_preflight import (
        add_gpu_idle_preflight_args,
        bootstrap_gpu_idle_preflight,
        recheck_gpu_exclusivity,
    )


GPU_IDLE_PREFLIGHT = bootstrap_gpu_idle_preflight() if __name__ == "__main__" else None

import torch
import torch.nn.functional as F
try:
    from datasets import load_dataset
    _HAVE_DATASETS = True
except ImportError:  # pragma: no cover - fallback reads nm-calibration text JSONL
    _HAVE_DATASETS = False


class _TextRows:
    """Minimal HF-Dataset-compatible view over local calibration texts.

    Mirrors only the API surface used by this harness (len/select/rows/
    column access) so the diagnostic can run without the `datasets` package.
    """

    def __init__(self, texts):
        self._texts = list(texts)

    def __len__(self):
        return len(self._texts)

    def __iter__(self):
        return ({"text": t} for t in self._texts)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "text":
                return list(self._texts)
            raise KeyError(key)
        if isinstance(key, slice):
            return [{"text": t} for t in self._texts[key]]
        return {"text": self._texts[key]}

    def select(self, indices):
        return _TextRows([self._texts[i] for i in indices])


def _load_nm_calibration(dataset_path, config="LLM", split="train"):
    """Load nm-calibration rows; prefer `datasets`, fall back to JSONL."""

    if _HAVE_DATASETS:
        try:
            return load_dataset(path=str(dataset_path), name=config, split=split)
        except Exception:
            pass
    parquet_path = None
    candidate_path = str(dataset_path)
    if candidate_path.endswith(".parquet") and os.path.isfile(candidate_path):
        parquet_path = candidate_path
    elif os.path.isdir(candidate_path):
        matches = sorted(
            os.path.join(candidate_path, name)
            for name in os.listdir(candidate_path)
            if name.endswith(".parquet")
        )
        if matches:
            parquet_path = matches[0]
    if parquet_path is not None:
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("parquet calibration input requires pyarrow") from exc
        table = parquet.read_table(parquet_path, columns=["messages"])
        texts = []
        for row in table.column("messages").to_pylist():
            if isinstance(row, list):
                text = "\n\n".join(
                    str(item.get("content", ""))
                    for item in row
                    if isinstance(item, dict) and item.get("content")
                )
            else:
                text = str(row or "")
            if text:
                texts.append(text)
        if not texts:
            raise ValueError(f"no non-empty messages found in parquet calibration input {parquet_path}")
        return _TextRows(texts)
    candidate = os.environ.get("NM_CALIBRATION_JSONL")
    if candidate is None:
        base = str(dataset_path)
        for name in ("llm.jsonl", "text.jsonl", "calibration.jsonl"):
            p = os.path.join(base, name) if os.path.isdir(base) else (base + ".jsonl" if name == "llm.jsonl" else None)
            if p and os.path.isfile(p):
                candidate = p
                break
    if candidate is None or not os.path.isfile(candidate):
        raise FileNotFoundError(
            f"no calibration JSONL found for {dataset_path}; set NM_CALIBRATION_JSONL"
        )
    import json as _json
    texts = []
    with open(candidate) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = _json.loads(line)
            texts.append(row.get("text", row.get("texts", "")))
    return _TextRows(texts)
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.quantization.config import (
    QuantizeConfig,
    ScaleSearchConfig,
)
from gptqmodel.quantization.quantizer import Quantizer
from gptqmodel.quantization.qvq import (
    default_qvq_trellis_batch_size,
    quantize_qvq_linear,
)
from gptqmodel.quantization.qvq_codecs import (
    PGC16_CODEBOOK_VERSION,
)
from gptqmodel.quantization.qvq_rates import normalize_qvq_rate
from gptqmodel.quantization.qvq_yaqa import (
    YAQA_PAPER_MINIMUM_SEQUENCES,
    YAQA_PAPER_REGULARIZATION,
    capture_yaqa_sketch_b,
)
from gptqmodel.utils.calibration import prepare_calibration_dataset
from gptqmodel.utils.diagnostic_metrics import native_tensor_metrics


TARGET_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def request_performance_qos() -> bool:
    """Request Darwin user-interactive QoS for the main and subsequently created workers."""

    if platform.system() != "Darwin":
        return False
    qos_class_user_interactive = 0x21
    libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    set_qos = libsystem.pthread_set_qos_class_self_np
    set_qos.argtypes = [ctypes.c_uint, ctypes.c_int]
    set_qos.restype = ctypes.c_int
    return set_qos(qos_class_user_interactive, 0) == 0


def resolve_qvq_device(requested: str) -> torch.device:
    """Resolve the reference quantizer device and fail closed on unavailable explicit backends."""

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--qvq-device=cuda requested but PyTorch CUDA is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--qvq-device=mps requested but PyTorch MPS is unavailable")
    return torch.device(requested)


def synchronize_benchmark_device(device: torch.device) -> None:
    """Close asynchronous CUDA work before recording a benchmark boundary."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def select_valid_token_rows(tensor: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Flatten batch/sequence dimensions while excluding every masked token."""

    if tensor.ndim < 3:
        raise ValueError(f"expected a [batch, sequence, ...] tensor, got shape {tuple(tensor.shape)}")
    if attention_mask.ndim != 2:
        raise ValueError(f"expected a rank-2 attention mask, got shape {tuple(attention_mask.shape)}")
    if tuple(tensor.shape[:2]) != tuple(attention_mask.shape):
        raise ValueError(
            f"attention mask shape {tuple(attention_mask.shape)} does not match tensor token geometry "
            f"{tuple(tensor.shape[:2])}"
        )
    keep = attention_mask.to(device=tensor.device, dtype=torch.bool)
    if not bool(keep.any()):
        raise ValueError("attention mask contains no valid tokens")
    return tensor[keep]


def load_nm_calibration_batches(
    tokenizer,
    config,
    *,
    dataset_path: Path,
    rows: int,
    concat_size: int | None,
    batch_size: int,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, int | str | None]]:
    """Load and prepare the same local nm-calibration rows used by model tests."""

    if rows < 1:
        raise ValueError("calibration rows must be positive")
    if concat_size is not None and concat_size < 0:
        raise ValueError("calibration concat size must be nonnegative")
    if batch_size < 1:
        raise ValueError("calibration batch size must be positive")
    normalized_concat_size = None if concat_size in (None, 0) else concat_size
    dataset = _load_nm_calibration(dataset_path)

    if len(dataset) < rows:
        raise ValueError(f"requested {rows} calibration rows, but the dataset contains only {len(dataset)}")
    selected = dataset.select(range(rows))
    qmodel = SimpleNamespace(
        tokenizer=tokenizer,
        support_batch_quantize=True,
        model=SimpleNamespace(config=config),
        quantize_config=None,
    )
    batches = prepare_calibration_dataset(
        qmodel,
        calibration_dataset=selected,
        calibration_dataset_concat_size=normalized_concat_size,
        calibration_dataset_sort="desc",
        batch_size=batch_size,
    )
    prepared_sequences = sum(int(batch["attention_mask"].shape[0]) for batch in batches)
    if normalized_concat_size is None and batch_size == 1 and prepared_sequences != rows:
        raise ValueError(
            "unpacked calibration must preserve one prepared sequence per source row: "
            f"selected {rows} rows but prepared {prepared_sequences} sequences"
        )
    valid_tokens = sum(int(batch["attention_mask"].ne(0).sum()) for batch in batches)
    padded_tokens = sum(int(batch["attention_mask"].eq(0).sum()) for batch in batches)
    return batches, {
        "dataset_path": str(dataset_path),
        "dataset_config": "LLM",
        "source_rows": rows,
        "concat_size": normalized_concat_size,
        "batch_size": batch_size,
        "prepared_batches": len(batches),
        "prepared_sequences": prepared_sequences,
        "valid_tokens": valid_tokens,
        "padded_tokens_excluded": padded_tokens,
    }


def load_nm_evaluation_batch(
    tokenizer,
    *,
    dataset_path: Path,
    row_offset: int,
    rows: int,
    max_length: int,
) -> tuple[dict[str, torch.Tensor], dict[str, int | str]]:
    """Tokenize a disjoint nm-calibration slice for held-out logit evaluation."""

    if row_offset < 0:
        raise ValueError("evaluation row offset must be nonnegative")
    if rows < 1 or max_length < 1:
        raise ValueError("evaluation rows and max length must be positive")
    dataset = _load_nm_calibration(dataset_path)
    row_end = row_offset + rows
    if len(dataset) < row_end:
        raise ValueError(
            f"requested evaluation rows [{row_offset}, {row_end}), but the dataset contains only {len(dataset)}"
        )
    selected = dataset.select(range(row_offset, row_end))
    texts = list(selected["text"])
    if len(texts) != rows or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("evaluation dataset slice must contain one nonempty `text` value per row")
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    if "attention_mask" not in encoded:
        raise ValueError("evaluation tokenizer output must include an attention mask")
    valid_tokens = int(encoded["attention_mask"].ne(0).sum())
    if valid_tokens < 1:
        raise ValueError("evaluation dataset slice produced no valid tokens")
    padded_tokens = int(encoded["attention_mask"].eq(0).sum())
    return dict(encoded), {
        "dataset_path": str(dataset_path),
        "dataset_config": "LLM",
        "row_start": row_offset,
        "row_end_exclusive": row_end,
        "source_rows": rows,
        "max_length": max_length,
        "valid_tokens": valid_tokens,
        "padded_tokens_excluded": padded_tokens,
    }


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value:
        return _first_tensor(value[0])
    raise TypeError(f"Cannot extract a tensor from {type(value)!r}")


def _summary(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().flatten()
    if values.numel() == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    # torch.quantile rejects tensors above 2**24 elements. Compute the same
    # linear interpolation from exact order statistics so large held-out
    # logit matrices remain fully measured rather than sampled.
    quantiles = []
    last_index = values.numel() - 1
    for probability in (0.50, 0.95, 0.99):
        position = last_index * probability
        lower_index = math.floor(position)
        upper_index = math.ceil(position)
        lower = values.kthvalue(lower_index + 1).values
        if lower_index == upper_index:
            quantiles.append(lower)
            continue
        upper = values.kthvalue(upper_index + 1).values
        quantiles.append(lower + (upper - lower) * (position - lower_index))
    return {
        "mean": values.double().mean().item(),
        "p50": quantiles[0].item(),
        "p95": quantiles[1].item(),
        "p99": quantiles[2].item(),
        "max": values.max().item(),
    }


@torch.inference_mode()
def tensor_metrics(
    dense: torch.Tensor,
    quantized: torch.Tensor,
    *,
    normalize_distribution: bool,
) -> dict[str, Any]:
    dense = dense.detach().float()
    quantized = quantized.detach().float()
    if dense.shape != quantized.shape:
        raise ValueError(f"metric shape mismatch: {tuple(dense.shape)} != {tuple(quantized.shape)}")
    native_metrics = native_tensor_metrics(
        dense,
        quantized,
        normalize_distribution=normalize_distribution,
    )
    if native_metrics is not None:
        return native_metrics

    error = quantized - dense
    dense_flat = dense.flatten()
    quantized_flat = quantized.flatten()
    error_flat = error.flatten()
    dense_flat64 = dense_flat.double()
    quantized_flat64 = quantized_flat.double()
    error_flat64 = error_flat.double()
    eps64 = torch.finfo(torch.float64).eps
    dense_energy = dense_flat64.square().sum()
    error_energy = error_flat64.square().sum()
    dense_energy_floor = dense_energy.clamp_min(eps64)
    error_energy_floor = error_energy.clamp_min(eps64)
    dense_centered = dense_flat64 - dense_flat64.mean()
    quantized_centered = quantized_flat64 - quantized_flat64.mean()

    dense_rows = dense.reshape(-1, dense.shape[-1])
    quantized_rows = quantized.reshape(-1, quantized.shape[-1])
    if normalize_distribution:
        dense_mean = dense_rows.mean(dim=-1, keepdim=True)
        dense_std = dense_rows.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        dense_logits = (dense_rows - dense_mean) / dense_std
        quantized_logits = (quantized_rows - dense_mean) / dense_std
    else:
        dense_logits = dense_rows
        quantized_logits = quantized_rows

    dense_log_prob = F.log_softmax(dense_logits, dim=-1)
    quantized_log_prob = F.log_softmax(quantized_logits, dim=-1)
    dense_prob = dense_log_prob.exp()
    quantized_prob = quantized_log_prob.exp()
    midpoint = (dense_prob + quantized_prob) * 0.5
    midpoint_log = midpoint.clamp_min(1e-30).log()
    kl_forward = (dense_prob * (dense_log_prob - quantized_log_prob)).sum(dim=-1)
    kl_reverse = (quantized_prob * (quantized_log_prob - dense_log_prob)).sum(dim=-1)
    js = 0.5 * (
        (dense_prob * (dense_log_prob - midpoint_log)).sum(dim=-1)
        + (quantized_prob * (quantized_log_prob - midpoint_log)).sum(dim=-1)
    )
    total_variation = 0.5 * (dense_prob - quantized_prob).abs().sum(dim=-1)
    hellinger = ((dense_prob.sqrt() - quantized_prob.sqrt()).square().sum(dim=-1) * 0.5).sqrt()
    dense_entropy = -(dense_prob * dense_log_prob).sum(dim=-1)
    dense_to_quantized_cross_entropy = -(dense_prob * quantized_log_prob).sum(dim=-1)
    row_cosine = F.cosine_similarity(dense_rows, quantized_rows, dim=-1)

    topk = min(5, dense.shape[-1])
    dense_topk = dense_logits.topk(topk, dim=-1).indices
    quantized_topk = quantized_logits.topk(topk, dim=-1).indices
    topk_overlap = (dense_topk.unsqueeze(-1) == quantized_topk.unsqueeze(-2)).any(dim=-1).float().mean(dim=-1)
    topk_exact = (dense_topk.sort(dim=-1).values == quantized_topk.sort(dim=-1).values).all(dim=-1)
    dense_top1_in_quantized_topk = (dense_topk[:, :1] == quantized_topk).any(dim=-1)
    quantized_top1_in_dense_topk = (quantized_topk[:, :1] == dense_topk).any(dim=-1)
    cosine = F.cosine_similarity(dense_flat64, quantized_flat64, dim=0).clamp(-1.0, 1.0)
    pearson = F.cosine_similarity(dense_centered, quantized_centered, dim=0).clamp(-1.0, 1.0)

    return {
        "shape": list(dense.shape),
        "finite": bool(torch.isfinite(quantized).all()),
        "mae": error_flat64.abs().mean().item(),
        "rmse": error_flat64.square().mean().sqrt().item(),
        "relative_l2": (error_energy / dense_energy_floor).sqrt().item(),
        "sqnr_db": (10.0 * torch.log10(dense_energy_floor / error_energy_floor)).item(),
        "max_abs_error": error_flat.abs().max().item(),
        "abs_error": _summary(error_flat.abs()),
        "bias": error_flat64.mean().item(),
        "error_std": error_flat64.std(unbiased=False).item(),
        "cosine": cosine.item(),
        "pearson": pearson.item(),
        "row_cosine": _summary(row_cosine),
        "norm_ratio": (quantized_flat64.norm() / dense_flat64.norm().clamp_min(eps64)).item(),
        "sign_agreement": ((dense_flat >= 0) == (quantized_flat >= 0)).float().mean().item(),
        "kl_forward": _summary(kl_forward),
        "kl_reverse": _summary(kl_reverse),
        "jensen_shannon": _summary(js),
        "total_variation": _summary(total_variation),
        "hellinger": _summary(hellinger),
        "dense_entropy": _summary(dense_entropy),
        "dense_to_quantized_cross_entropy": _summary(dense_to_quantized_cross_entropy),
        "top1_agreement": (dense_topk[:, 0] == quantized_topk[:, 0]).float().mean().item(),
        "top5_overlap": _summary(topk_overlap),
        "top5_exact_agreement": topk_exact.float().mean().item(),
        "dense_top1_in_quantized_top5": dense_top1_in_quantized_topk.float().mean().item(),
        "quantized_top1_in_dense_top5": quantized_top1_in_dense_topk.float().mean().item(),
    }


def target_modules(model: nn.Module, *, layer_count: int | None = None) -> dict[str, nn.Linear]:
    """Return quantizable projections, optionally restricted to early decoder layers."""

    selected_module_ids: set[int] | None = None
    if layer_count is not None:
        layers = decoder_layers(model)
        if isinstance(layer_count, bool) or not isinstance(layer_count, int) or not 1 <= layer_count <= len(layers):
            raise ValueError(f"target layer count must be in [1, {len(layers)}], got {layer_count!r}")
        selected_module_ids = {id(module) for layer in layers[:layer_count] for module in layer.modules()}

    return {
        name: module
        for name, module in model.named_modules()
        if (
            isinstance(module, nn.Linear)
            and name.endswith(TARGET_SUFFIXES)
            and (selected_module_ids is None or id(module) in selected_module_ids)
        )
    }


def decoder_layers(model: nn.Module) -> list[nn.Module]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("Expected a decoder model exposing model.layers")
    return list(layers)


@torch.inference_mode()
def capture_forward(
    model: nn.Module,
    encoded: dict[str, torch.Tensor],
    modules: dict[str, nn.Linear],
    *,
    capture_inputs: bool,
    layer_count: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        raise ValueError("attention_mask is required so padded outputs cannot enter diagnostic metrics")
    captured_inputs: dict[str, list[torch.Tensor]] = defaultdict(list)
    captured_outputs: dict[str, list[torch.Tensor]] = defaultdict(list)
    handles = []
    layers = decoder_layers(model)
    if layer_count is not None:
        if isinstance(layer_count, bool) or not isinstance(layer_count, int) or not 1 <= layer_count <= len(layers):
            raise ValueError(f"capture layer count must be in [1, {len(layers)}], got {layer_count!r}")
        layers = layers[:layer_count]

    for name, module in modules.items():

        def module_hook(_module, args, output, module_name=name):
            if capture_inputs:
                rows = select_valid_token_rows(_first_tensor(args).detach(), attention_mask)
                captured_inputs[module_name].append(rows.cpu().float())
            rows = select_valid_token_rows(_first_tensor(output).detach(), attention_mask)
            captured_outputs[module_name].append(rows.cpu().float())

        handles.append(module.register_forward_hook(module_hook))

    for index, layer in enumerate(layers):

        def layer_hook(_module, _args, output, layer_index=index):
            rows = select_valid_token_rows(_first_tensor(output).detach(), attention_mask)
            captured_outputs[f"layer.{layer_index}.hidden"].append(rows.cpu().float())

        handles.append(layer.register_forward_hook(layer_hook))

    try:
        logits = model(**encoded, use_cache=False).logits.detach()
        logits = select_valid_token_rows(logits, attention_mask).cpu().float()
    finally:
        for handle in handles:
            handle.remove()

    merged_inputs = {name: torch.cat(values, dim=0) for name, values in captured_inputs.items()}
    merged_outputs = {name: torch.cat(values, dim=0) for name, values in captured_outputs.items()}
    return logits, merged_inputs, merged_outputs


@torch.inference_mode()
def capture_calibration_hessians(
    model: nn.Module,
    batches: list[dict[str, torch.Tensor]],
    modules: dict[str, nn.Linear],
    *,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Stream calibration batches and accumulate X.T@X from valid token rows only."""

    accumulators: dict[str, torch.Tensor] = {}
    sample_counts = dict.fromkeys(modules, 0)
    active_mask: torch.Tensor | None = None
    active_token_count = 0
    handles = []
    backbone = getattr(model, "model", model)

    for name, module in modules.items():

        def module_hook(_module, args, _output, module_name=name):
            if active_mask is None:
                raise RuntimeError("calibration hook ran without an active attention mask")
            tensor = _first_tensor(args).detach()
            if tuple(tensor.shape[:2]) != tuple(active_mask.shape):
                raise ValueError(
                    f"attention mask shape {tuple(active_mask.shape)} does not match calibration activation "
                    f"geometry {tuple(tensor.shape[:2])}"
                )
            rows = torch.where(active_mask.unsqueeze(-1).bool(), tensor.float(), 0).flatten(0, 1)
            gram = rows.mT @ rows
            if module_name in accumulators:
                accumulators[module_name].add_(gram)
            else:
                accumulators[module_name] = gram
            sample_counts[module_name] += active_token_count

        handles.append(module.register_forward_hook(module_hook))

    try:
        for batch in batches:
            if "attention_mask" not in batch:
                raise ValueError("every calibration batch must include attention_mask")
            active_token_count = int(batch["attention_mask"].ne(0).sum())
            if active_token_count == 0:
                raise ValueError("calibration batch contains no valid tokens")
            encoded = {name: value.to(device) for name, value in batch.items()}
            active_mask = encoded["attention_mask"]
            backbone(**encoded, use_cache=False)
    finally:
        active_mask = None
        for handle in handles:
            handle.remove()

    hessians = {}
    for name in modules:
        count = sample_counts[name]
        if count == 0 or name not in accumulators:
            raise ValueError(f"module {name} observed no valid calibration tokens")
        hessians[name] = accumulators[name].div(count).cpu().contiguous()
    return hessians, sample_counts


@torch.inference_mode()
def quantize_module_weight(
    weight: torch.Tensor,
    calibration_hessian: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    sym: bool,
    adjacent_zero_search: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    rows, columns = weight.shape
    if columns % group_size:
        raise ValueError(f"{columns=} is not divisible by {group_size=}")
    groups = columns // group_size
    if calibration_hessian.shape != (columns, columns):
        raise ValueError(
            f"calibration Hessian shape {tuple(calibration_hessian.shape)} does not match weight columns {columns}"
        )
    importance = calibration_hessian.diagonal().reshape(groups, group_size).contiguous()
    grouped_weight = weight.float().reshape(rows, groups, group_size).contiguous()
    qcfg = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        sym=sym,
        mse=2.0,
        scale_search=ScaleSearchConfig.ACTIVATION,
        adaptive_clipping=None,
        offload_to_disk=False,
    )
    quantizer = Quantizer(qcfg=qcfg)
    quantizer.configure(perchannel=True)
    if not adjacent_zero_search:
        quantizer._search_adjacent_zero_points = lambda _maxq: False
        old_cpu_search = os.environ.get("GPTQMODEL_SCALE_SEARCH_CPU")
        os.environ["GPTQMODEL_SCALE_SEARCH_CPU"] = "0"
    else:
        old_cpu_search = None
    try:
        scale, zero = quantizer.find_params_batched(grouped_weight, weight=True, hessian=importance)
    finally:
        if not adjacent_zero_search:
            if old_cpu_search is None:
                os.environ.pop("GPTQMODEL_SCALE_SEARCH_CPU", None)
            else:
                os.environ["GPTQMODEL_SCALE_SEARCH_CPU"] = old_cpu_search

    q = torch.round(grouped_weight / scale.unsqueeze(-1))
    q.add_(zero.unsqueeze(-1)).clamp_(0, (1 << bits) - 1).sub_(zero.unsqueeze(-1))
    reconstructed = (q * scale.unsqueeze(-1)).reshape_as(weight)
    zero_one_fraction = (zero == 1).float().mean().item() if bits == 2 else 0.0
    return reconstructed, {
        "method": "gptq",
        "weight": tensor_metrics(weight.float(), reconstructed, normalize_distribution=True),
        "zero_one_fraction": zero_one_fraction,
        "zero_two_fraction": (zero == 2).float().mean().item() if bits == 2 else 0.0,
    }


@torch.inference_mode()
def quantize_module_weight_qvq(
    weight: torch.Tensor,
    H: torch.Tensor,
    *,
    bits: float,
    output_hessian: torch.Tensor | None = None,
    bias: torch.Tensor | None,
    module_name: str,
    device: torch.device,
    trellis_batch_size: int,
    tail_biting_candidates: int = 1,
    codebook_version: str = PGC16_CODEBOOK_VERSION,
    module_scale_search: bool = False,
    output_channel_scale_optimization: bool = False,
    viterbi_objective: str = "euclidean",
    viterbi_minimum_proxy_improvement: float = 0.0,
    rounding: str = "block_ldlq",
    damp_percent: float = 0.01,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run the pack-identical QVQ path and return its dense reconstruction."""

    seed = zlib.crc32(module_name.encode("utf-8")) & 0x7FFFFFFF
    result = quantize_qvq_linear(
        weight.to(device),
        H.to(device),
        bits=bits,
        output_hessian=(None if output_hessian is None else output_hessian.to(device)),
        bias=None if bias is None else bias.to(device),
        seed=seed,
        damp_percent=damp_percent,
        trellis_batch_size=trellis_batch_size,
        tail_biting_candidates=tail_biting_candidates,
        codebook_version=codebook_version,
        module_scale_search=module_scale_search,
        output_channel_scale_optimization=output_channel_scale_optimization,
        viterbi_objective=viterbi_objective,
        viterbi_minimum_proxy_improvement=viterbi_minimum_proxy_improvement,
        rounding=rounding,
    )
    reconstructed = result.weight.cpu()
    payload_bytes = result.trellis.numel() * result.trellis.element_size()
    auxiliary_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (result.SU, result.SV)
    )
    stored_bytes = payload_bytes + auxiliary_bytes
    report = {
        "method": "qvq",
        "codebook": codebook_version,
        "rounding": result.rounding,
        "damp_percent": damp_percent,
        "weight": tensor_metrics(weight.float(), reconstructed.float(), normalize_distribution=True),
        "proxy_loss": result.proxy_loss.item(),
        "baseline_proxy_loss": result.baseline_proxy_loss.item(),
        "module_scale_search_selected": getattr(result, "module_scale_search_selected", False),
        "module_scale_multiplier": getattr(result, "module_scale_multiplier", 1.0),
        "module_scale_reencoded": getattr(result, "module_scale_reencoded", False),
        "output_scale_optimized_channels": result.output_scale_optimized_channels,
        "hessian_viterbi_selected": result.hessian_viterbi_selected,
        "hessian_viterbi_candidate_relative_improvement": (result.hessian_viterbi_candidate_relative_improvement),
        "trellis_shape": list(result.trellis.shape),
        "trellis_words": result.trellis.numel(),
        "weight_numel": weight.numel(),
        "payload_bytes": payload_bytes,
        "auxiliary_bytes": auxiliary_bytes,
        "stored_bytes": stored_bytes,
        "payload_bits_per_weight": payload_bytes * 8 / weight.numel(),
        "effective_bits_per_weight": stored_bytes * 8 / weight.numel(),
        "tail_biting_candidates": tail_biting_candidates,
    }
    if result.kronecker_proxy_loss is not None:
        report["kronecker_proxy_loss"] = result.kronecker_proxy_loss.item()
    return reconstructed, report


@torch.inference_mode()
def quantize_module_weight_exl3(
    weight: torch.Tensor,
    H: torch.Tensor,
    *,
    bits: int,
    device: torch.device,
    codebook: str = "mcg",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run the production EXL3 quantizer and return its exact dense reconstruction."""

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("EXL3 comparison requires a CUDA/HIP quantization device")
    if codebook not in {"mcg", "mul1", "3inst"}:
        raise ValueError(f"unsupported EXL3 codebook {codebook!r}")
    if weight.ndim != 2 or H.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("EXL3 weight and Hessian dimensions do not match")
    if weight.shape[0] % 128 or weight.shape[1] % 128:
        raise ValueError("EXL3 comparison requires input and output widths divisible by 128")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

    quant_args: dict[str, object] = {
        "K": bits,
        "devices": [device],
        "apply_out_scales": None,
        "sigma_reg": 0.025,
        "seed": 787,
    }
    if codebook == "mcg":
        quant_args["mcg"] = True
    elif codebook == "mul1":
        quant_args["mul1"] = True

    # The comparison harness stores X^T X / N. EXL3's production entry point
    # expects an accumulated Hessian plus its count, so count=1 preserves the
    # exact matrix without fabricating calibration samples.
    h_data = {
        "H": H.to(device=device, dtype=torch.float32).clone(),
        "count": 1,
        "finalized": False,
    }
    input_weight = weight.t().to(device=device, dtype=torch.float32).contiguous()
    fork_device = device.index if device.index is not None else torch.cuda.current_device()
    with torch.random.fork_rng(devices=[fork_device]):
        weight_q, proxy_error, out_tensors = quantize_exl3(
            weight=input_weight,
            H_data=h_data,
            quant_args=quant_args,
            return_weight_q=True,
        )

    reconstructed = weight_q.t().contiguous().cpu()
    payload_tensor = out_tensors["trellis"]
    payload_bytes = payload_tensor.numel() * payload_tensor.element_size()
    stored_bytes = sum(tensor.numel() * tensor.element_size() for tensor in out_tensors.values())
    result = {
        "method": "exl3",
        "codebook": codebook,
        "weight": tensor_metrics(weight.float(), reconstructed.float(), normalize_distribution=True),
        "proxy_loss": proxy_error,
        "weight_numel": weight.numel(),
        "payload_bytes": payload_bytes,
        "auxiliary_bytes": stored_bytes - payload_bytes,
        "stored_bytes": stored_bytes,
        "payload_bits_per_weight": payload_bytes * 8 / weight.numel(),
        "effective_bits_per_weight": stored_bytes * 8 / weight.numel(),
        "trellis_shape": list(out_tensors["trellis"].shape),
    }
    del h_data, input_weight, weight_q, out_tensors
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return reconstructed, result


def mean_metric(entries: dict[str, dict[str, Any]], path: tuple[str, ...]) -> float:
    values = []
    for entry in entries.values():
        value: Any = entry
        for key in path:
            value = value[key]
        values.append(float(value))
    return sum(values) / max(len(values), 1)


def mean_optional_metric(entries: dict[str, dict[str, Any]], key: str) -> float | None:
    """Average an optional scalar emitted by every module in an arm."""

    if not entries or not all(key in entry for entry in entries.values()):
        return None
    return sum(float(entry[key]) for entry in entries.values()) / len(entries)


def aggregate_storage_rate(
    entries: dict[str, dict[str, Any]],
    byte_key: str,
    *,
    shared_auxiliary_bytes: int = 0,
) -> float | None:
    """Return whole-arm bits/weight from exact serialized tensor bytes."""

    if isinstance(shared_auxiliary_bytes, bool) or not isinstance(shared_auxiliary_bytes, int):
        raise TypeError("shared auxiliary storage must be an integer byte count")
    if shared_auxiliary_bytes < 0:
        raise ValueError("shared auxiliary storage must be nonnegative")
    if not entries or not all(byte_key in entry and "weight_numel" in entry for entry in entries.values()):
        return None
    total_weights = sum(int(entry["weight_numel"]) for entry in entries.values())
    if total_weights <= 0:
        raise ValueError("storage-rate accounting requires a positive total weight count")
    total_bytes = sum(int(entry[byte_key]) for entry in entries.values()) + shared_auxiliary_bytes
    return total_bytes * 8 / total_weights


def print_arm_summary(name: str, result: dict[str, Any]) -> None:
    module_metrics = result["modules"]
    layer_metrics = result["layers"]
    logits = result["logits"]
    payload_bpw = aggregate_storage_rate(module_metrics, "payload_bytes")
    effective_bpw = aggregate_storage_rate(
        module_metrics,
        "stored_bytes",
        shared_auxiliary_bytes=result.get("shared_auxiliary_bytes", 0),
    )
    storage_summary = (
        f"bpw={payload_bpw:.5f}/{effective_bpw:.5f} "
        if payload_bpw is not None and effective_bpw is not None
        else ""
    )
    print(
        f"{name:22s} "
        f"{storage_summary}"
        f"w_rel={mean_metric(module_metrics, ('weight', 'relative_l2')):.5f} "
        f"local_KL={mean_metric(module_metrics, ('local', 'kl_forward', 'mean')):.6f} "
        f"live_KL={mean_metric(module_metrics, ('live', 'kl_forward', 'mean')):.6f} "
        f"layer_KL={mean_metric(layer_metrics, ('kl_forward', 'mean')):.6f} "
        f"logit_KL={logits['kl_forward']['mean']:.6f} "
        f"top1={logits['top1_agreement']:.4f} "
        f"top5={logits['top5_overlap']['mean']:.4f} "
        f"quant_s={result['timing']['quantization_seconds']:.2f} "
        f"replay_s={result['timing']['post_quant_inference_seconds']:.2f} "
        f"metric_s={result['timing']['post_quant_metric_seconds']:.2f} "
        f"total_s={result['seconds']:.2f}",
        flush=True,
    )


def arm_summary_row(name: str, result: dict[str, Any]) -> dict[str, Any]:
    module_metrics = result["modules"]
    layer_metrics = result["layers"]
    logits = result["logits"]
    return {
        "arm": name,
        "method": result["method"],
        "codebook": result.get("codebook"),
        "bits": result["bits"],
        "tail_biting_candidates": result.get("tail_biting_candidates"),
        "sym": result["sym"],
        "seconds": result["seconds"],
        "quantization_seconds": result["timing"]["quantization_seconds"],
        "module_validation_seconds": result["timing"]["module_validation_seconds"],
        "post_quant_inference_seconds": result["timing"]["post_quant_inference_seconds"],
        "post_quant_metric_seconds": result["timing"]["post_quant_metric_seconds"],
        "payload_bits_per_weight": aggregate_storage_rate(module_metrics, "payload_bytes"),
        "effective_bits_per_weight": aggregate_storage_rate(
            module_metrics,
            "stored_bytes",
            shared_auxiliary_bytes=result.get("shared_auxiliary_bytes", 0),
        ),
        "mean_full_hessian_proxy_loss": mean_optional_metric(module_metrics, "proxy_loss"),
        "mean_baseline_full_hessian_proxy_loss": mean_optional_metric(module_metrics, "baseline_proxy_loss"),
        "mean_weight_relative_l2": mean_metric(module_metrics, ("weight", "relative_l2")),
        "mean_weight_mae": mean_metric(module_metrics, ("weight", "mae")),
        "mean_weight_rmse": mean_metric(module_metrics, ("weight", "rmse")),
        "mean_weight_sqnr_db": mean_metric(module_metrics, ("weight", "sqnr_db")),
        "mean_weight_cosine": mean_metric(module_metrics, ("weight", "cosine")),
        "mean_local_kl": mean_metric(module_metrics, ("local", "kl_forward", "mean")),
        "mean_local_top1_agreement": mean_metric(module_metrics, ("local", "top1_agreement")),
        "mean_local_top5_overlap": mean_metric(module_metrics, ("local", "top5_overlap", "mean")),
        "mean_live_kl": mean_metric(module_metrics, ("live", "kl_forward", "mean")),
        "mean_live_top1_agreement": mean_metric(module_metrics, ("live", "top1_agreement")),
        "mean_live_top5_overlap": mean_metric(module_metrics, ("live", "top5_overlap", "mean")),
        "mean_layer_kl": mean_metric(layer_metrics, ("kl_forward", "mean")),
        "mean_layer_top1_agreement": mean_metric(layer_metrics, ("top1_agreement",)),
        "mean_layer_top5_overlap": mean_metric(layer_metrics, ("top5_overlap", "mean")),
        "final_logit_kl_mean": logits["kl_forward"]["mean"],
        "final_logit_kl_p95": logits["kl_forward"]["p95"],
        "final_logit_kl_p99": logits["kl_forward"]["p99"],
        "final_logit_kl_max": logits["kl_forward"]["max"],
        "final_logit_reverse_kl_mean": logits["kl_reverse"]["mean"],
        "final_logit_js_mean": logits["jensen_shannon"]["mean"],
        "final_logit_dense_entropy_mean": logits["dense_entropy"]["mean"],
        "final_logit_cross_entropy_mean": logits["dense_to_quantized_cross_entropy"]["mean"],
        "final_logit_rmse": logits["rmse"],
        "final_logit_relative_l2": logits["relative_l2"],
        "final_logit_sqnr_db": logits["sqnr_db"],
        "final_logit_cosine": logits["cosine"],
        "final_logit_top1_agreement": logits["top1_agreement"],
        "final_logit_top5_overlap_mean": logits["top5_overlap"]["mean"],
        "final_logit_top5_exact_agreement": logits["top5_exact_agreement"],
        "final_logit_dense_top1_in_quantized_top5": logits["dense_top1_in_quantized_top5"],
        "final_logit_quantized_top1_in_dense_top5": logits["quantized_top1_in_dense_top5"],
    }


def normalize_requested_rates(rates: list[float], *, method: str) -> list[int | float]:
    """Canonicalize rates and reject half steps only when GPTQ is requested."""

    normalized = [normalize_qvq_rate(rate) for rate in rates]
    if method in {"gptq", "both", "all"} and any(not isinstance(rate, int) for rate in normalized):
        raise ValueError("half-step rates require QVQ and/or EXL3 without a GPTQ comparison arm")
    return normalized


def allocate_exl3_module_bits(
    module_numels: dict[str, int],
    *,
    target_bpw: float,
) -> dict[str, int]:
    """Reproduce EXL3's grouped integer-rate allocation for a target average BPW.

    EXL3 encodes each tensor at an integer rate. Fractional model rates floor
    every tensor and then spend the remaining weight-counted bit budget on
    architecture groups: attention before MLP, q/k/v together, and gate/up
    together. Groups nearer either end of a decoder stack win ties. This is the
    production ExLlamaV3 allocation policy, specialized to the projection names
    exercised by this diagnostic.
    """

    if isinstance(target_bpw, bool) or not isinstance(target_bpw, (int, float)):
        raise TypeError("EXL3 target BPW must be a real scalar")
    target_bpw = float(target_bpw)
    if not math.isfinite(target_bpw) or not 1.0 <= target_bpw <= 8.0:
        raise ValueError("EXL3 target BPW must be finite and in [1, 8]")
    if not module_numels:
        raise ValueError("EXL3 allocation requires at least one module")
    if any(isinstance(numel, bool) or not isinstance(numel, int) or numel <= 0 for numel in module_numels.values()):
        raise ValueError("EXL3 allocation module sizes must be positive integers")

    base_bpw = math.floor(target_bpw)
    targets = {name: base_bpw for name in module_numels}
    total_numel = sum(module_numels.values())
    used_bits = base_bpw * total_numel
    maximum_bits = int(target_bpw * total_numel)

    groups: dict[str, dict[str, Any]] = {}
    layer_pattern = re.compile(r"^(.*?\.layers\.)(\d+)(\..*)$")
    for index, name in enumerate(module_numels):
        match = layer_pattern.match(name)
        stack = match.group(1) if match else None
        layer = int(match.group(2)) if match else -1
        if name.endswith((".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj")):
            group_key = name.rsplit(".", 1)[0] + ".qkv"
            priority = 2
        elif name.endswith(".self_attn.o_proj"):
            group_key = name
            priority = 2
        elif name.endswith((".mlp.gate_proj", ".mlp.up_proj")):
            group_key = name.rsplit(".", 1)[0] + ".gu"
            priority = 1
        elif name.endswith(".mlp.down_proj"):
            group_key = name
            priority = 1
        else:
            group_key = name
            priority = 0
        group = groups.setdefault(
            group_key,
            {"names": [], "priority": priority, "stack": stack, "layer": layer, "index": index},
        )
        group["names"].append(name)

    stack_max: dict[str, int] = {}
    for group in groups.values():
        if group["stack"] is not None:
            stack_max[group["stack"]] = max(stack_max.get(group["stack"], -1), group["layer"])

    def order_key(group: dict[str, Any]) -> tuple[int, int, int, int]:
        stack = group["stack"]
        layer = group["layer"]
        distance = 0 if stack is None else min(layer, stack_max[stack] - layer)
        return -group["priority"], distance, layer, group["index"]

    ordered_groups = sorted(groups.values(), key=order_key)
    while used_bits < maximum_bits:
        updated = False
        for group in ordered_groups:
            names = group["names"]
            cost = sum(module_numels[name] for name in names if targets[name] < 8)
            if cost and used_bits + cost <= maximum_bits:
                for name in names:
                    targets[name] = min(8, targets[name] + 1)
                used_bits += cost
                updated = True
        if not updated:
            break
    return targets


def main() -> None:
    parser = argparse.ArgumentParser()
    add_gpu_idle_preflight_args(parser)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=48)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--calibration-dataset",
        type=Path,
        default=Path("/monster/data/model/dataset/nm-calibration"),
    )
    parser.add_argument("--calibration-rows", type=int, default=256)
    parser.add_argument(
        "--calibration-concat-size",
        type=int,
        default=0,
        help="target packed sequence length; 0 preserves one independent sequence per source row",
    )
    parser.add_argument("--calibration-batch-size", type=int, default=1)
    parser.add_argument("--evaluation-rows", type=int, default=128)
    parser.add_argument(
        "--evaluation-row-offset",
        type=int,
        help="first held-out nm-calibration row; defaults to --calibration-rows",
    )
    parser.add_argument("--bits", type=float, nargs="+", default=list(range(1, 9)))
    parser.add_argument(
        "--method",
        choices=("gptq", "qvq", "exl3", "qvq-exl3", "both", "all"),
        default="gptq",
    )
    parser.add_argument(
        "--symmetry",
        choices=("both", "symmetric", "asymmetric"),
        default="both",
    )
    parser.add_argument("--qvq-device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--qvq-trellis-batch-size",
        type=int,
        help="override the measured backend/rate default",
    )
    parser.add_argument(
        "--qvq-tail-biting-candidates",
        type=int,
        nargs="+",
        default=[1],
        help="one or more non-regressing tail-biting overlap candidate counts",
    )
    parser.add_argument("--exl3-codebook", choices=("mcg", "mul1", "3inst"), default="mcg")
    parser.add_argument(
        "--qvq-module-scale-search",
        action="store_true",
        help="search one module scale folded into the existing SV tensor",
    )
    parser.add_argument(
        "--qvq-output-channel-scales",
        action="store_true",
        help="optimize fixed-trellis per-output scales already stored in SV",
    )
    parser.add_argument(
        "--qvq-viterbi-objective",
        choices=("euclidean", "hessian_diagonal"),
        default="euclidean",
        help="offline Viterbi emission objective; full-Hessian baseline remains an acceptance candidate",
    )
    parser.add_argument(
        "--qvq-viterbi-minimum-proxy-improvement",
        type=float,
        default=0.0,
        help="minimum relative full-Hessian improvement required to select a non-Euclidean path",
    )
    parser.add_argument(
        "--qvq-rounding",
        nargs="+",
        choices=("block_ldlq", "yaqa"),
        default=["block_ldlq"],
        help="run local BlockLDLQ, full-model YAQA-v3 Sketch B, or both",
    )
    parser.add_argument("--qvq-yaqa-seed", type=int, default=0)
    parser.add_argument(
        "--qvq-yaqa-regularization",
        type=float,
        default=YAQA_PAPER_REGULARIZATION,
        help="YAQA factor diagonal regularization relative to mean diagonal (paper default: 1e-4)",
    )
    parser.add_argument(
        "--qvq-yaqa-minimum-sequences",
        type=int,
        default=YAQA_PAPER_MINIMUM_SEQUENCES,
        help="minimum independent Sketch-B sequences (paper's smallest reported successful ablation: 2000)",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--csv-out", type=Path)
    args = parser.parse_args()

    try:
        args.bits = normalize_requested_rates(args.bits, method=args.method)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if args.qvq_trellis_batch_size is not None and args.qvq_trellis_batch_size < 1:
        parser.error("--qvq-trellis-batch-size must be positive")
    if any(candidate_count < 1 for candidate_count in args.qvq_tail_biting_candidates):
        parser.error("--qvq-tail-biting-candidates must be positive")
    if args.calibration_rows < 1 or args.calibration_concat_size < 0 or args.calibration_batch_size < 1:
        parser.error("calibration rows and batch size must be positive; concat size must be nonnegative")
    evaluation_row_offset = args.calibration_rows if args.evaluation_row_offset is None else args.evaluation_row_offset
    if args.evaluation_rows < 1 or evaluation_row_offset < 0:
        parser.error("evaluation rows must be positive and row offset must be nonnegative")
    if evaluation_row_offset < args.calibration_rows:
        parser.error(
            "evaluation rows overlap calibration rows; --evaluation-row-offset must be at least --calibration-rows"
        )
    run_gptq = args.method in ("gptq", "both", "all")
    run_qvq = args.method in ("qvq", "qvq-exl3", "both", "all")
    run_exl3 = args.method in ("exl3", "qvq-exl3", "all")
    qvq_roundings = tuple(dict.fromkeys(args.qvq_rounding))
    qvq_tail_biting_candidates = tuple(dict.fromkeys(args.qvq_tail_biting_candidates))
    run_yaqa = run_qvq and "yaqa" in qvq_roundings
    if run_yaqa and args.qvq_output_channel_scales:
        parser.error("YAQA cannot be combined with --qvq-output-channel-scales")
    if run_yaqa and args.qvq_module_scale_search:
        parser.error("YAQA cannot be combined with --qvq-module-scale-search")
    if run_yaqa and args.qvq_viterbi_objective != "euclidean":
        parser.error("YAQA requires --qvq-viterbi-objective=euclidean")
    if not math.isfinite(args.qvq_yaqa_regularization) or args.qvq_yaqa_regularization < 0:
        parser.error("--qvq-yaqa-regularization must be finite and nonnegative")
    if args.qvq_yaqa_minimum_sequences < 1:
        parser.error("--qvq-yaqa-minimum-sequences must be positive")
    try:
        qvq_device = resolve_qvq_device(args.qvq_device)
    except RuntimeError as error:
        parser.error(str(error))
    if run_exl3 and qvq_device.type != "cuda":
        parser.error("EXL3 comparison requires --qvq-device=cuda")

    torch.manual_seed(0)
    performance_qos = request_performance_qos()
    if platform.system() == "Darwin" and not performance_qos:
        raise RuntimeError("failed to request performance-core QoS")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    os.environ.setdefault("GPTQMODEL_SCALE_SEARCH_CPU", "1")

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    source_model_layers = int(config.num_hidden_layers)
    if args.layers > source_model_layers:
        parser.error(f"--layers={args.layers} exceeds the source model's {source_model_layers} decoder layers")
    if not run_yaqa:
        config.num_hidden_layers = args.layers
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            config=config,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        .eval()
        .to(qvq_device)
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    calibration_batches, calibration_stats = load_nm_calibration_batches(
        tokenizer,
        config,
        dataset_path=args.calibration_dataset,
        rows=args.calibration_rows,
        concat_size=args.calibration_concat_size,
        batch_size=args.calibration_batch_size,
    )
    evaluation, evaluation_stats = load_nm_evaluation_batch(
        tokenizer,
        dataset_path=args.calibration_dataset,
        row_offset=evaluation_row_offset,
        rows=args.evaluation_rows,
        max_length=args.max_length,
    )
    evaluation = {name: value.to(qvq_device) for name, value in evaluation.items()}
    loaded_model_layers = len(decoder_layers(model))
    if run_yaqa and loaded_model_layers != source_model_layers:
        raise AssertionError(
            f"YAQA requires all {source_model_layers} source decoder layers, but loaded {loaded_model_layers}"
        )
    modules = target_modules(model, layer_count=args.layers)
    expected_modules = args.layers * len(TARGET_SUFFIXES)
    if len(modules) != expected_modules:
        raise ValueError(f"Expected {expected_modules} target modules, found {len(modules)}")

    calibration_hessians, calibration_sample_counts = capture_calibration_hessians(
        model,
        calibration_batches,
        modules,
        device=qvq_device,
    )
    if set(calibration_sample_counts.values()) != {calibration_stats["valid_tokens"]}:
        raise AssertionError(
            f"calibration sample counts {set(calibration_sample_counts.values())} do not match "
            f"the prepared valid-token count {calibration_stats['valid_tokens']}"
        )
    yaqa_input_hessians: dict[str, torch.Tensor] = {}
    yaqa_output_hessians: dict[str, torch.Tensor] = {}
    yaqa_stats: dict[str, Any] | None = None
    if run_yaqa:
        print(
            f"YAQA Sketch B: collecting full-{source_model_layers}-layer gradients for "
            f"{len(modules)} projections in the first {args.layers} layers",
            flush=True,
        )
        synchronize_benchmark_device(qvq_device)
        yaqa_started = time.perf_counter()
        yaqa_start_allocated_bytes = None
        if qvq_device.type == "cuda":
            yaqa_start_allocated_bytes = torch.cuda.memory_allocated(qvq_device)
            torch.cuda.reset_peak_memory_stats(qvq_device)
        yaqa_input_hessians, yaqa_output_hessians, yaqa_stats = capture_yaqa_sketch_b(
            model,
            calibration_batches,
            modules,
            device=qvq_device,
            seed=args.qvq_yaqa_seed,
            minimum_sequences=args.qvq_yaqa_minimum_sequences,
            checkpoint_modules=decoder_layers(model),
        )
        synchronize_benchmark_device(qvq_device)
        yaqa_stats.update(
            {
                "collection_seconds": time.perf_counter() - yaqa_started,
                "cuda_start_allocated_bytes": yaqa_start_allocated_bytes,
                "cuda_peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated(qvq_device) if qvq_device.type == "cuda" else None
                ),
                "cuda_peak_reserved_bytes": (
                    torch.cuda.max_memory_reserved(qvq_device) if qvq_device.type == "cuda" else None
                ),
                "source_model_layers": source_model_layers,
                "loaded_model_layers": loaded_model_layers,
                "target_decoder_layers": args.layers,
                "target_modules": len(modules),
            }
        )
    dense_logits, evaluation_inputs, dense_outputs = capture_forward(
        model,
        evaluation,
        modules,
        capture_inputs=True,
        layer_count=args.layers,
    )
    original_weights = {name: module.weight.detach().cpu().clone() for name, module in modules.items()}
    exl3_allocations = (
        {
            bits: allocate_exl3_module_bits(
                {name: weight.numel() for name, weight in original_weights.items()},
                target_bpw=bits,
            )
            for bits in sorted(set(args.bits))
        }
        if run_exl3
        else {}
    )
    symmetries = {
        "both": (True, False),
        "symmetric": (True,),
        "asymmetric": (False,),
    }[args.symmetry]
    gptq_arms = (
        tuple(
            (
                f"w{bits}-{'symmetric' if sym else 'asymmetric-adjacent'}",
                "gptq",
                bits,
                sym,
                True,
                None,
                None,
                None,
            )
            for bits in sorted(set(args.bits), reverse=True)
            for sym in symmetries
        )
        if run_gptq
        else ()
    )
    qvq_arms = (
        tuple(
            (
                f"w{bits}-qvq-{PGC16_CODEBOOK_VERSION}"
                + (
                    f"-candidates-{candidate_count}"
                    if len(qvq_tail_biting_candidates) > 1 or candidate_count != 1
                    else ""
                )
                + ("-modulescale" if args.qvq_module_scale_search else "")
                + ("-outscale" if args.qvq_output_channel_scales else "")
                + (f"-{rounding.replace('_', '-')}" if rounding != "block_ldlq" or len(qvq_roundings) > 1 else ""),
                "qvq",
                bits,
                True,
                False,
                PGC16_CODEBOOK_VERSION,
                rounding,
                candidate_count,
            )
            for bits in sorted(set(args.bits), reverse=True)
            for rounding in qvq_roundings
            for candidate_count in qvq_tail_biting_candidates
        )
        if run_qvq
        else ()
    )
    exl3_arms = (
        tuple(
            (
                f"w{bits}-exl3-{args.exl3_codebook}",
                "exl3",
                bits,
                True,
                False,
                args.exl3_codebook,
                None,
                None,
            )
            for bits in sorted(set(args.bits), reverse=True)
        )
        if run_exl3
        else ()
    )
    arms = gptq_arms + qvq_arms + exl3_arms
    report: dict[str, Any] = {
        "settings": {
            "model": str(args.model),
            "layers": args.layers,
            "source_model_layers": source_model_layers,
            "loaded_model_layers": loaded_model_layers,
            "full_model_loaded_for_yaqa": run_yaqa,
            "group_size": args.group_size,
            "max_length": args.max_length,
            "threads": args.threads,
            "performance_qos": performance_qos,
            "gpu_idle_preflight": GPU_IDLE_PREFLIGHT.as_dict() if GPU_IDLE_PREFLIGHT is not None else None,
            "device": str(qvq_device),
            "calibration": calibration_stats,
            "evaluation": evaluation_stats,
            "padding_policy": "attention-mask-selected valid tokens only",
            "evaluation_valid_tokens": int(evaluation["attention_mask"].ne(0).sum()),
            "evaluation_padded_tokens_excluded": int(evaluation["attention_mask"].eq(0).sum()),
            "methods": args.method,
            "qvq_device": str(qvq_device) if run_qvq else None,
            "qvq_trellis_batch_size": (
                args.qvq_trellis_batch_size if args.qvq_trellis_batch_size is not None else "auto-by-rate"
            )
            if run_qvq
            else None,
            "qvq_tail_biting_candidates": list(qvq_tail_biting_candidates) if run_qvq else None,
            "qvq_codebook": PGC16_CODEBOOK_VERSION if run_qvq else None,
            "qvq_module_scale_search": args.qvq_module_scale_search if run_qvq else None,
            "qvq_output_channel_scales": args.qvq_output_channel_scales if run_qvq else None,
            "qvq_viterbi_objective": args.qvq_viterbi_objective if run_qvq else None,
            "qvq_viterbi_minimum_proxy_improvement": (
                args.qvq_viterbi_minimum_proxy_improvement if run_qvq else None
            ),
            "qvq_rounding": list(qvq_roundings) if run_qvq else None,
            "qvq_yaqa_seed": args.qvq_yaqa_seed if run_yaqa else None,
            "qvq_yaqa_regularization": args.qvq_yaqa_regularization if run_yaqa else None,
            "qvq_yaqa_minimum_sequences": args.qvq_yaqa_minimum_sequences if run_yaqa else None,
            "exl3_codebook": args.exl3_codebook if run_exl3 else None,
            "exl3_out_scales": "auto" if run_exl3 else None,
            "exl3_sigma_reg": 0.025 if run_exl3 else None,
            "exl3_fractional_allocation": (
                "production grouped integer rates: attention before MLP; q/k/v and gate/up coupled"
                if run_exl3
                else None
            ),
            "gptq_method": "grouped activation-weighted affine parameter-search boundary",
            "qvq_method": "RHT + BlockLDLQ/YAQA-v3 + PGC16 L16 V2 two-pass tail-biting trellis",
            "exl3_method": "production ExLlamaV3 Hessian-aware trellis quantizer",
            "packing": "GPTQ fake-quantized dense; QVQ and EXL3 pack-identical reconstruction",
            "asymmetric_zero_search": "adjacent integer zero points",
            "intermediate_distribution": "dense-standardized channel softmax",
            "final_distribution": "raw vocabulary logits softmax",
        },
        "yaqa_sketch_b": yaqa_stats,
        "arms": {},
    }

    if GPU_IDLE_PREFLIGHT is not None:
        recheck_gpu_exclusivity(GPU_IDLE_PREFLIGHT)

    for (
        arm_name,
        method,
        bits,
        sym,
        adjacent,
        codebook_version,
        rounding,
        tail_biting_candidates,
    ) in arms:
        synchronize_benchmark_device(qvq_device)
        started = time.perf_counter()
        quantization_seconds = 0.0
        module_validation_seconds = 0.0
        module_results: dict[str, Any] = {}
        trellis_batch_size = None
        if method == "qvq":
            trellis_batch_size = (
                args.qvq_trellis_batch_size
                if args.qvq_trellis_batch_size is not None
                else default_qvq_trellis_batch_size(bits, qvq_device)
            )
        for module_index, (module_name, module) in enumerate(modules.items(), start=1):
            print(
                f"{arm_name}: module {module_index}/{len(modules)} {module_name}"
                + (f" trellis_batch={trellis_batch_size}" if method == "qvq" else ""),
                flush=True,
            )
            dense_weight = original_weights[module_name]
            synchronize_benchmark_device(qvq_device)
            quantization_started = time.perf_counter()
            if method == "qvq":
                input_hessian = (
                    yaqa_input_hessians[module_name] if rounding == "yaqa" else calibration_hessians[module_name]
                )
                output_hessian = yaqa_output_hessians[module_name] if rounding == "yaqa" else None
                reconstructed, weight_result = quantize_module_weight_qvq(
                    dense_weight,
                    input_hessian,
                    bits=bits,
                    output_hessian=output_hessian,
                    bias=module.bias,
                    module_name=module_name,
                    device=qvq_device,
                    trellis_batch_size=trellis_batch_size,
                    tail_biting_candidates=tail_biting_candidates,
                    codebook_version=codebook_version,
                    module_scale_search=args.qvq_module_scale_search,
                    output_channel_scale_optimization=args.qvq_output_channel_scales,
                    viterbi_objective=args.qvq_viterbi_objective,
                    viterbi_minimum_proxy_improvement=args.qvq_viterbi_minimum_proxy_improvement,
                    rounding=rounding,
                    damp_percent=(args.qvq_yaqa_regularization if rounding == "yaqa" else 0.01),
                )
                assert trellis_batch_size is not None
                weight_result["trellis_batch_size"] = trellis_batch_size
            elif method == "exl3":
                module_bits = exl3_allocations[bits][module_name]
                reconstructed, weight_result = quantize_module_weight_exl3(
                    dense_weight,
                    calibration_hessians[module_name],
                    bits=module_bits,
                    device=qvq_device,
                    codebook=codebook_version,
                )
                weight_result["allocated_bits"] = module_bits
            else:
                reconstructed, weight_result = quantize_module_weight(
                    dense_weight,
                    calibration_hessians[module_name],
                    bits=bits,
                    group_size=args.group_size,
                    sym=sym,
                    adjacent_zero_search=adjacent,
                )
            synchronize_benchmark_device(qvq_device)
            quantization_seconds += time.perf_counter() - quantization_started

            module_validation_started = time.perf_counter()
            bias = None if module.bias is None else module.bias.detach().cpu()
            local_output = F.linear(evaluation_inputs[module_name], reconstructed, bias)
            weight_result["local"] = tensor_metrics(
                dense_outputs[module_name], local_output, normalize_distribution=True
            )
            with torch.no_grad():
                module.weight.copy_(reconstructed.to(module.weight.device))
            module_results[module_name] = weight_result
            synchronize_benchmark_device(qvq_device)
            module_validation_seconds += time.perf_counter() - module_validation_started

        synchronize_benchmark_device(qvq_device)
        post_quant_inference_started = time.perf_counter()
        quant_logits, _, live_outputs = capture_forward(
            model,
            evaluation,
            modules,
            capture_inputs=False,
            layer_count=args.layers,
        )
        synchronize_benchmark_device(qvq_device)
        post_quant_inference_seconds = time.perf_counter() - post_quant_inference_started

        post_quant_metric_started = time.perf_counter()
        for module_name in modules:
            module_results[module_name]["live"] = tensor_metrics(
                dense_outputs[module_name],
                live_outputs[module_name],
                normalize_distribution=True,
            )
        layer_results = {
            f"layer.{index}": tensor_metrics(
                dense_outputs[f"layer.{index}.hidden"],
                live_outputs[f"layer.{index}.hidden"],
                normalize_distribution=True,
            )
            for index in range(args.layers)
        }
        logits_result = tensor_metrics(dense_logits, quant_logits, normalize_distribution=False)
        synchronize_benchmark_device(qvq_device)
        post_quant_metric_seconds = time.perf_counter() - post_quant_metric_started
        result = {
            "method": method,
            "codebook": codebook_version,
            "rounding": rounding,
            "tail_biting_candidates": tail_biting_candidates if method == "qvq" else None,
            "bits": bits,
            "sym": sym,
            "adjacent_zero_search": adjacent,
            "module_bit_allocation": exl3_allocations[bits] if method == "exl3" else None,
            "shared_auxiliary_bytes": 0,
            "seconds": time.perf_counter() - started,
            "timing": {
                "quantization_seconds": quantization_seconds,
                "module_validation_seconds": module_validation_seconds,
                "post_quant_inference_seconds": post_quant_inference_seconds,
                "post_quant_metric_seconds": post_quant_metric_seconds,
            },
            "modules": module_results,
            "layers": layer_results,
            "logits": logits_result,
        }
        report["arms"][arm_name] = result
        print_arm_summary(arm_name, result)

        for module_name, module in modules.items():
            with torch.no_grad():
                module.weight.copy_(original_weights[module_name].to(module.weight.device))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.json_out}", flush=True)
    if args.csv_out is not None:
        rows = [arm_summary_row(name, result) for name, result in report["arms"].items()]
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.csv_out}", flush=True)


if __name__ == "__main__":
    main()
