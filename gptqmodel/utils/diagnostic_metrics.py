# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Exact, bounded-memory CPU diagnostics for very large model outputs."""

from __future__ import annotations

import math
import os
import platform
from pathlib import Path
from typing import Any

import torch

from .cpp import TorchOpsJitExtension, default_torch_ops_build_root

_GLOBAL_METRIC_NAMES = (
    "finite",
    "mae",
    "rmse",
    "relative_l2",
    "sqnr_db",
    "max_abs_error",
    "bias",
    "error_std",
    "cosine",
    "pearson",
    "norm_ratio",
    "sign_agreement",
)
_ROW_METRIC_NAMES = (
    "row_cosine",
    "kl_forward",
    "kl_reverse",
    "jensen_shannon",
    "total_variation",
    "hellinger",
    "dense_entropy",
    "dense_to_quantized_cross_entropy",
    "top5_overlap",
)
_TOP_AGGREGATE_COLUMNS = {
    "top1_agreement": 9,
    "top5_exact_agreement": 10,
    "dense_top1_in_quantized_top5": 11,
    "quantized_top1_in_dense_top5": 12,
}


def _source_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "gptqmodel_ext"
        / "diagnostic_metrics_cpu.cpp"
    )


def _extra_cflags() -> list[str]:
    flags = ["-O3", "-fno-math-errno"]
    if platform.system() == "Linux":
        flags.append("-fopenmp")
        if platform.machine().lower() in {"x86_64", "amd64"}:
            flags.append("-march=native")
    return flags


def _extra_ldflags() -> list[str]:
    return ["-fopenmp"] if platform.system() == "Linux" else []


_DIAGNOSTIC_METRICS_CPU_EXTENSION = TorchOpsJitExtension(
    name="gptqmodel_diagnostic_metrics_cpu",
    namespace="gptqmodel_diagnostic_metrics",
    required_ops=("tensor_metrics_cpu",),
    sources=lambda: [str(_source_path())],
    build_root_env="GPTQMODEL_EXT_BUILD",
    default_build_root=lambda: default_torch_ops_build_root("diagnostic_metrics_cpu"),
    display_name="diagnostic_metrics_cpu",
    extra_cflags=_extra_cflags,
    extra_ldflags=_extra_ldflags,
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=False,
)


def _summary(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().flatten()
    if values.numel() == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
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


def _extension_enabled() -> bool:
    raw = os.getenv("GPTQMODEL_DIAGNOSTIC_METRICS_CPU", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def native_tensor_metrics(
    dense: torch.Tensor,
    quantized: torch.Tensor,
    *,
    normalize_distribution: bool,
) -> dict[str, Any] | None:
    """Return exact diagnostic metrics through the native CPU op when available.

    ``None`` means that the caller should use its reference implementation. A
    loaded native op never hides validation or execution errors behind fallback.
    """

    if (
        not _extension_enabled()
        or dense.device.type != "cpu"
        or quantized.device.type != "cpu"
    ):
        return None
    if dense.shape != quantized.shape:
        raise ValueError(
            f"metric shape mismatch: {tuple(dense.shape)} != {tuple(quantized.shape)}"
        )
    if not _DIAGNOSTIC_METRICS_CPU_EXTENSION.load():
        return None

    dense_float = dense.detach().float().contiguous()
    quantized_float = quantized.detach().float().contiguous()
    operation = _DIAGNOSTIC_METRICS_CPU_EXTENSION.op("tensor_metrics_cpu")
    global_values, absolute_error_values, row_values = operation(
        dense_float,
        quantized_float,
        normalize_distribution,
    )
    global_metrics = dict(zip(_GLOBAL_METRIC_NAMES, global_values.tolist()))
    absolute_error = absolute_error_values.tolist()
    result: dict[str, Any] = {
        "shape": list(dense.shape),
        **global_metrics,
        "finite": bool(global_metrics["finite"]),
        "abs_error": {
            "mean": absolute_error[0],
            "p50": absolute_error[1],
            "p95": absolute_error[2],
            "p99": absolute_error[3],
            "max": absolute_error[4],
        },
    }
    for column, name in enumerate(_ROW_METRIC_NAMES):
        result[name] = _summary(row_values[:, column])
    for name, column in _TOP_AGGREGATE_COLUMNS.items():
        result[name] = row_values[:, column].mean().item()
    return result


__all__ = ["native_tensor_metrics"]
