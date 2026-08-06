# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
import statistics
from enum import Enum
from typing import Any, Iterable, Mapping

import torch


QUANTIZATION_DIAGNOSTICS_ENV = "GPTQMODEL_QUANTIZATION_DIAGNOSTICS"


class QuantizationDiagnosticsMode(str, Enum):
    """Cost tiers for quantization-time anomaly diagnostics."""

    OFF = "off"
    AUTO = "auto"
    CHANNEL = "channel"


def normalize_quantization_diagnostics_mode(
    value: str | QuantizationDiagnosticsMode | None,
) -> QuantizationDiagnosticsMode:
    """Normalize the public diagnostics selector."""

    if value is None:
        return QuantizationDiagnosticsMode.AUTO
    if isinstance(value, QuantizationDiagnosticsMode):
        return value
    try:
        return QuantizationDiagnosticsMode(str(value).strip().lower())
    except ValueError as exc:
        choices = ", ".join(mode.value for mode in QuantizationDiagnosticsMode)
        raise ValueError(f"QuantizeConfig: `quantization_diagnostics` must be one of {{{choices}}}.") from exc


def resolve_quantization_diagnostics_mode(
    configured: str | QuantizationDiagnosticsMode | None,
) -> QuantizationDiagnosticsMode:
    """Apply the optional process-level override to the configured mode."""

    override = os.getenv(QUANTIZATION_DIAGNOSTICS_ENV)
    return normalize_quantization_diagnostics_mode(override if override is not None else configured)


def _parse_loss(entry: Mapping[str, Any]) -> float | None:
    try:
        value = float(entry.get("loss"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


def _percentiles(values: torch.Tensor) -> dict[str, float | None]:
    """Return bounded distribution landmarks for a one-dimensional tensor."""

    finite = values.detach().to(device="cpu", dtype=torch.float64).flatten()
    finite = finite[torch.isfinite(finite)]
    if finite.numel() == 0:
        return {
            "median": None,
            "p95": None,
            "p99": None,
            "maximum": None,
        }
    return {
        "median": float(torch.quantile(finite, 0.50).item()),
        "p95": float(torch.quantile(finite, 0.95).item()),
        "p99": float(torch.quantile(finite, 0.99).item()),
        "maximum": float(finite.max().item()),
    }


def _top_axis_errors(
    source_squared: torch.Tensor,
    error_squared: torch.Tensor,
    *,
    axis_name: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Rank bounded row/feature error records without retaining model tensors."""

    source_l2 = source_squared.clamp_min(0).sqrt()
    error_l2 = error_squared.clamp_min(0).sqrt()
    relative = error_l2 / source_l2.clamp_min(torch.finfo(torch.float64).eps)
    count = min(max(1, int(top_k)), relative.numel())
    if count == 0:
        return []
    indexes = torch.topk(relative, k=count, largest=True, sorted=True).indices
    return [
        {
            axis_name: int(index),
            "relative_rmse": float(relative[index].item()),
            "signal_l2": float(source_l2[index].item()),
            "error_l2": float(error_l2[index].item()),
        }
        for index in indexes.tolist()
    ]


def analyze_quantization_losses(
    entries: Iterable[Mapping[str, Any]],
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Summarize module loss concentration without touching model tensors."""

    records = []
    by_role: dict[str, list[float]] = {}
    skipped = 0
    for entry in entries:
        loss = _parse_loss(entry)
        if loss is None:
            skipped += 1
            continue
        role = str(entry.get("module", "unknown"))
        record = {
            "layer": entry.get("layer"),
            "module": role,
            "full_name": entry.get("full_name"),
            "loss": loss,
        }
        for name in ("bits", "group_size", "sym", "desc_act", "samples", "damp"):
            if name in entry:
                record[name] = entry.get(name)
        records.append(record)
        by_role.setdefault(role, []).append(loss)

    if not records:
        return {
            "module_count": 0,
            "skipped_loss_count": skipped,
            "mean_loss": None,
            "median_loss": None,
            "total_loss": 0.0,
            "max_to_mean_ratio": None,
            "max_total_loss_share": None,
            "severe_concentration": False,
            "top": [],
            "records": [],
        }

    losses = [record["loss"] for record in records]
    total_loss = math.fsum(losses)
    mean_loss = total_loss / len(losses)
    role_medians = {role: statistics.median(values) for role, values in by_role.items()}

    for record in records:
        role_median = role_medians[record["module"]]
        record["role_median_loss"] = role_median
        record["role_median_ratio"] = (
            record["loss"] / role_median if role_median > 0 else None
        )
        role_losses = by_role[record["module"]]
        record["role_percentile"] = 100.0 * sum(
            candidate <= record["loss"] for candidate in role_losses
        ) / len(role_losses)
        record["total_loss_share"] = record["loss"] / total_loss if total_loss > 0 else 0.0

    records.sort(key=lambda item: item["loss"], reverse=True)
    maximum = records[0]
    max_to_mean = maximum["loss"] / mean_loss if mean_loss > 0 else None
    max_share = maximum["total_loss_share"]

    # This deliberately requires both a large mean ratio and material ownership
    # of total loss. It catches a single catastrophic module without treating
    # normal depth-dependent variation as a quantization failure.
    severe = bool(
        max_to_mean is not None
        and max_to_mean >= 50.0
        and max_share >= 0.25
    )
    return {
        "module_count": len(records),
        "skipped_loss_count": skipped,
        "mean_loss": mean_loss,
        "median_loss": statistics.median(losses),
        "total_loss": total_loss,
        "max_to_mean_ratio": max_to_mean,
        "max_total_loss_share": max_share,
        "severe_concentration": severe,
        "top": records[: max(1, int(top_k))],
        "records": records,
    }


def analyze_scale_channels(scales: torch.Tensor) -> dict[str, Any]:
    """Reduce a scale tensor to output-channel anomaly statistics.

    Quantizer-returned GPTQ scale tensors are shaped
    ``[output_channels, groups]`` before packing. Saved QuantLinear scale
    tensors are transposed to ``[groups, output_channels]`` and must be handled
    separately by checkpoint analyzers. This function intentionally performs
    reductions only; it never copies the full tensor to CPU. Callers should
    keep it behind the ``channel`` diagnostics mode because each module's
    complete scale tensor must still be scanned.
    """

    raw_values = scales.detach().to(dtype=torch.float32)
    finite = torch.isfinite(raw_values)
    nonfinite_count = int((~finite).sum().item())
    nonpositive_count = int((finite & (raw_values <= 0)).sum().item())
    values = raw_values.abs()
    values = torch.where(finite, values, torch.zeros_like(values))

    if values.numel() == 0:
        return {
            "scale_shape": list(values.shape),
            "scale_element_count": 0,
            "output_channel_count": 0,
            "group_count": 0,
            "nonfinite_scale_count": nonfinite_count,
            "nonpositive_scale_count": nonpositive_count,
            "zero_scale_count": 0,
            "max_scale": None,
            "p99_channel_max_scale": None,
            "median_channel_max_scale": None,
            "max_to_median_ratio": None,
            "max_scale_output_channel": None,
            "max_scale_group": None,
            "scale_count_above_10": 0,
            "scale_percentiles": _percentiles(values),
        }

    if values.ndim == 0:
        channel_max = values.reshape(1)
    elif values.ndim == 1:
        channel_max = values
    else:
        channel_max = values.amax(dim=tuple(range(1, values.ndim)))

    max_scale, max_channel = channel_max.max(dim=0)
    median_scale = channel_max.median()
    p99_scale = torch.quantile(channel_max, 0.99)
    max_group = None
    group_count = 1
    if values.ndim == 2:
        group_count = int(values.shape[1])
        maximum_flat_index = int(values.flatten().argmax().item())
        max_group = maximum_flat_index % group_count
    elif values.ndim > 2:
        group_count = int(math.prod(values.shape[1:]))

    max_value = float(max_scale.item())
    median_value = float(median_scale.item())
    return {
        "scale_shape": list(values.shape),
        "scale_element_count": values.numel(),
        "output_channel_count": channel_max.numel(),
        "group_count": group_count,
        "nonfinite_scale_count": nonfinite_count,
        "nonpositive_scale_count": nonpositive_count,
        "zero_scale_count": int((finite & (raw_values == 0)).sum().item()),
        "max_scale": max_value,
        "p99_channel_max_scale": float(p99_scale.item()),
        "median_channel_max_scale": median_value,
        "max_to_median_ratio": max_value / median_value if median_value > 0 else None,
        "max_scale_output_channel": int(max_channel.item()),
        "max_scale_group": max_group,
        "scale_count_above_10": int((values > 10.0).sum().item()),
        "scale_percentiles": _percentiles(values),
    }


def analyze_group_index(
    g_idx: torch.Tensor | None,
    *,
    group_count: int | None = None,
) -> dict[str, Any]:
    """Summarize the exact zero-based input-feature to group mapping."""

    if g_idx is None:
        return {
            "available": False,
            "input_index_count": 0,
            "group_count": group_count,
        }
    values = g_idx.detach().to(device="cpu").flatten()
    if values.numel() == 0:
        return {
            "available": True,
            "input_index_count": 0,
            "group_count": group_count,
            "minimum_group": None,
            "maximum_group": None,
            "unique_group_count": 0,
            "negative_group_count": 0,
            "out_of_range_group_count": 0,
            "monotonic_decrease_count": 0,
            "is_monotonic_non_decreasing": True,
        }

    if values.is_floating_point():
        finite = torch.isfinite(values)
        nonfinite_count = int((~finite).sum().item())
        safe_values = torch.where(finite, values, torch.zeros_like(values)).to(dtype=torch.int64)
    else:
        nonfinite_count = 0
        safe_values = values.to(dtype=torch.int64)
    minimum = int(safe_values.min().item())
    maximum = int(safe_values.max().item())
    negative_count = int((safe_values < 0).sum().item())
    out_of_range_count = (
        int((safe_values >= int(group_count)).sum().item())
        if group_count is not None
        else 0
    )
    decreases = int((safe_values[1:] < safe_values[:-1]).sum().item())
    return {
        "available": True,
        "input_index_count": int(safe_values.numel()),
        "group_count": group_count,
        "minimum_group": minimum,
        "maximum_group": maximum,
        "unique_group_count": int(torch.unique(safe_values).numel()),
        "negative_group_count": negative_count,
        "out_of_range_group_count": out_of_range_count,
        "nonfinite_group_count": nonfinite_count,
        "monotonic_decrease_count": decreases,
        "is_monotonic_non_decreasing": decreases == 0,
    }


def analyze_output_error(
    inputs: torch.Tensor,
    source_weight: torch.Tensor,
    reconstructed_weight: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Measure dense-to-reconstructed output drift on identical captured inputs.

    KLD is computed after a FP32 softmax over the final output dimension. It is
    token-distribution KLD for an LM head and a bounded sensitivity diagnostic
    for hidden projections; callers must not present the latter as perplexity.
    """

    if inputs.ndim != 2 or source_weight.ndim != 2 or source_weight.shape != reconstructed_weight.shape:
        return {"available": False, "reason": "inputs and equal-shape two-dimensional weights are required"}
    if inputs.shape[1] != source_weight.shape[1] or inputs.numel() == 0:
        return {"available": False, "reason": "captured input width must match weight input features"}

    device = source_weight.device
    compute_dtype = source_weight.dtype
    sample = inputs.to(device=device, dtype=compute_dtype)
    source_output = torch.nn.functional.linear(sample, source_weight, bias).float()
    reconstructed_output = torch.nn.functional.linear(
        sample,
        reconstructed_weight.to(device=device, dtype=compute_dtype),
        bias,
    ).float()
    difference = reconstructed_output - source_output
    absolute = difference.abs()
    source_norm = torch.linalg.vector_norm(source_output)
    error_norm = torch.linalg.vector_norm(difference)
    eps = torch.finfo(torch.float32).eps

    source_log_prob = torch.log_softmax(source_output, dim=-1)
    reconstructed_log_prob = torch.log_softmax(reconstructed_output, dim=-1)
    kld = torch.sum(source_log_prob.exp() * (source_log_prob - reconstructed_log_prob), dim=-1)
    result = {
        "available": True,
        "sample_count": int(sample.shape[0]),
        "output_width": int(source_output.shape[-1]),
        "mean_absolute_error": float(absolute.mean().item()),
        "mean_signed_error": float(difference.mean().item()),
        "rmse": float(difference.square().mean().sqrt().item()),
        "relative_l2_error": float((error_norm / source_norm.clamp_min(eps)).item()),
        "softmax_kld_mean": float(kld.mean().item()),
        "softmax_kld_median": float(kld.median().item()),
        "softmax_kld_p95": float(torch.quantile(kld, 0.95).item()),
        "softmax_kld_max": float(kld.max().item()),
        "top1_agreement": float(
            (source_output.argmax(dim=-1) == reconstructed_output.argmax(dim=-1)).float().mean().item()
        ),
    }
    del sample, source_output, reconstructed_output, difference, absolute, kld
    return result


def analyze_reconstruction_error(
    source: torch.Tensor,
    reconstructed: torch.Tensor,
    *,
    max_chunk_values: int = 8 * 1024 * 1024,
    top_k_axes: int = 5,
) -> dict[str, Any]:
    """Compare dense ``W`` with canonical GPTQ ``Wq`` using bounded chunks.

    Axis labels follow the tensors reaching GPTQ: axis 0 is the output row and
    axis 1 is the input feature. The function retains only aggregate scalars
    and bounded top-k index records.
    """

    if source.ndim != 2 or reconstructed.ndim != 2 or source.shape != reconstructed.shape:
        return {
            "available": False,
            "reason": "source and reconstructed weights must be equal-shape two-dimensional tensors",
            "source_shape": list(source.shape),
            "reconstructed_shape": list(reconstructed.shape),
        }

    rows, columns = (int(source.shape[0]), int(source.shape[1]))
    chunk_rows = max(1, int(max_chunk_values) // max(columns, 1))
    source_squared_by_row = torch.zeros(rows, dtype=torch.float64)
    error_squared_by_row = torch.zeros(rows, dtype=torch.float64)
    source_squared_by_feature = torch.zeros(columns, dtype=torch.float64)
    error_squared_by_feature = torch.zeros(columns, dtype=torch.float64)

    source_squared_total = 0.0
    reconstructed_squared_total = 0.0
    error_squared_total = 0.0
    absolute_error_total = 0.0
    dot_total = 0.0
    finite_pair_count = 0
    source_nonfinite_count = 0
    reconstructed_nonfinite_count = 0
    maximum_absolute_error = -1.0
    maximum_error_output_row = None
    maximum_error_input_feature = None

    for row_start in range(0, rows, chunk_rows):
        row_end = min(row_start + chunk_rows, rows)
        source_chunk = source[row_start:row_end].detach().to(dtype=torch.float32)
        reconstructed_chunk = reconstructed[row_start:row_end].detach().to(dtype=torch.float32)
        source_finite = torch.isfinite(source_chunk)
        reconstructed_finite = torch.isfinite(reconstructed_chunk)
        finite = source_finite & reconstructed_finite
        source_nonfinite_count += int((~source_finite).sum().item())
        reconstructed_nonfinite_count += int((~reconstructed_finite).sum().item())
        finite_pair_count += int(finite.sum().item())

        source_chunk = torch.where(finite, source_chunk, torch.zeros_like(source_chunk))
        reconstructed_chunk = torch.where(
            finite,
            reconstructed_chunk,
            torch.zeros_like(reconstructed_chunk),
        )
        error = reconstructed_chunk - source_chunk
        source_squared = source_chunk.square()
        reconstructed_squared = reconstructed_chunk.square()
        error_squared = error.square()

        source_squared_by_row[row_start:row_end] = source_squared.sum(dim=1).to(
            device="cpu",
            dtype=torch.float64,
        )
        error_squared_by_row[row_start:row_end] = error_squared.sum(dim=1).to(
            device="cpu",
            dtype=torch.float64,
        )
        source_squared_by_feature.add_(
            source_squared.sum(dim=0).to(device="cpu", dtype=torch.float64)
        )
        error_squared_by_feature.add_(
            error_squared.sum(dim=0).to(device="cpu", dtype=torch.float64)
        )

        source_squared_total += float(source_squared.sum().item())
        reconstructed_squared_total += float(reconstructed_squared.sum().item())
        error_squared_total += float(error_squared.sum().item())
        absolute_error_total += float(error.abs().sum().item())
        dot_total += float((source_chunk * reconstructed_chunk).sum().item())

        chunk_absolute_error = error.abs()
        chunk_maximum = float(chunk_absolute_error.max().item())
        if chunk_maximum > maximum_absolute_error:
            flat_index = int(chunk_absolute_error.flatten().argmax().item())
            maximum_absolute_error = chunk_maximum
            maximum_error_output_row = row_start + flat_index // columns
            maximum_error_input_feature = flat_index % columns

        del source_chunk, reconstructed_chunk, error, source_squared, reconstructed_squared, error_squared

    source_l2 = math.sqrt(max(source_squared_total, 0.0))
    reconstructed_l2 = math.sqrt(max(reconstructed_squared_total, 0.0))
    error_l2 = math.sqrt(max(error_squared_total, 0.0))
    denominator = max(source_l2 * reconstructed_l2, torch.finfo(torch.float64).eps)
    relative_rmse = error_l2 / max(source_l2, torch.finfo(torch.float64).eps)
    row_relative = error_squared_by_row.clamp_min(0).sqrt() / source_squared_by_row.clamp_min(
        torch.finfo(torch.float64).eps
    ).sqrt()
    feature_relative = error_squared_by_feature.clamp_min(0).sqrt() / source_squared_by_feature.clamp_min(
        torch.finfo(torch.float64).eps
    ).sqrt()

    return {
        "available": True,
        "shape": [rows, columns],
        "axis_semantics": {
            "axis_0": "output_row",
            "axis_1": "input_feature",
            "index_base": 0,
        },
        "source_dtype": str(source.dtype),
        "reconstructed_dtype": str(reconstructed.dtype),
        "source_device": str(source.device),
        "reconstructed_device": str(reconstructed.device),
        "element_count": rows * columns,
        "finite_pair_count": finite_pair_count,
        "source_nonfinite_count": source_nonfinite_count,
        "reconstructed_nonfinite_count": reconstructed_nonfinite_count,
        "source_l2": source_l2,
        "reconstructed_l2": reconstructed_l2,
        "error_l2": error_l2,
        "rmse": math.sqrt(error_squared_total / max(finite_pair_count, 1)),
        "mean_absolute_error": absolute_error_total / max(finite_pair_count, 1),
        "relative_rmse": relative_rmse,
        "cosine_similarity": dot_total / denominator,
        "sqnr_db": (
            20.0 * math.log10(source_l2 / error_l2)
            if source_l2 > 0 and error_l2 > 0
            else None
        ),
        "maximum_absolute_error": max(maximum_absolute_error, 0.0),
        "maximum_error_output_row": maximum_error_output_row,
        "maximum_error_input_feature": maximum_error_input_feature,
        "output_row_relative_rmse": _percentiles(row_relative),
        "input_feature_relative_rmse": _percentiles(feature_relative),
        "top_output_rows": _top_axis_errors(
            source_squared_by_row,
            error_squared_by_row,
            axis_name="output_row",
            top_k=top_k_axes,
        ),
        "top_input_features": _top_axis_errors(
            source_squared_by_feature,
            error_squared_by_feature,
            axis_name="input_feature",
            top_k=top_k_axes,
        ),
    }


def _sample_indexes(length: int, count: int, device: torch.device) -> torch.Tensor:
    if length <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    count = min(length, max(1, int(count)))
    if count == length:
        return torch.arange(length, dtype=torch.long, device=device)
    return torch.linspace(0, length - 1, steps=count, device=device).round().to(dtype=torch.long).unique()


def sample_reconstructed_quant_codes(
    weight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    g_idx: torch.Tensor,
    *,
    bits: int,
    sample_size: int = 64,
    input_indexes: torch.Tensor | None = None,
    output_indexes: torch.Tensor | None = None,
) -> dict[str, Any] | None:
    """Sample logical codes obtained by requantizing a reconstructed weight.

    This mirrors the generic GPTQ packer's ``round((weight + zero * scale) / scale)``
    boundary. It is intentionally sampled and used only by the opt-in channel
    diagnostics tier.
    """

    if weight.ndim != 2 or scales.ndim != 2 or zeros.ndim != 2 or g_idx.ndim != 1:
        return None

    source = weight.detach()
    in_features = int(g_idx.numel())
    if source.shape[1] != in_features:
        if source.shape[0] == in_features:
            source = source.T
        else:
            return None
    out_features = source.shape[0]

    if scales.shape[0] == out_features:
        scales_by_output = scales
    elif scales.shape[1] == out_features:
        scales_by_output = scales.T
    else:
        return None
    if zeros.shape[0] == out_features:
        zeros_by_output = zeros
    elif zeros.shape[1] == out_features:
        zeros_by_output = zeros.T
    else:
        return None
    if scales_by_output.shape != zeros_by_output.shape:
        return None

    device = source.device
    if input_indexes is None:
        input_indexes = _sample_indexes(in_features, sample_size, device)
    else:
        input_indexes = input_indexes.to(device=device, dtype=torch.long)
    if output_indexes is None:
        output_indexes = _sample_indexes(out_features, sample_size, device)
    else:
        output_indexes = output_indexes.to(device=device, dtype=torch.long)

    group_indexes = g_idx.to(device=device, dtype=torch.long).index_select(0, input_indexes)
    if group_indexes.numel() and (
        int(group_indexes.min().item()) < 0
        or int(group_indexes.max().item()) >= scales_by_output.shape[1]
    ):
        return None

    selected_scales = (
        scales_by_output.to(device=device, dtype=torch.float32)
        .index_select(0, output_indexes)
        .index_select(1, group_indexes)
    )
    selected_zeros = (
        zeros_by_output.to(device=device, dtype=torch.float32)
        .index_select(0, output_indexes)
        .index_select(1, group_indexes)
    )
    selected_weight = (
        source.to(dtype=torch.float32)
        .index_select(0, output_indexes)
        .index_select(1, input_indexes)
    )
    finite = (
        torch.isfinite(selected_weight)
        & torch.isfinite(selected_scales)
        & torch.isfinite(selected_zeros)
        & (selected_scales != 0)
    )
    raw_codes = torch.where(
        finite,
        torch.round((selected_weight + selected_zeros * selected_scales) / selected_scales),
        torch.zeros_like(selected_weight),
    )
    maxq = (1 << int(bits)) - 1
    return {
        "input_indexes": input_indexes.to(device="cpu"),
        "output_indexes": output_indexes.to(device="cpu"),
        "codes": raw_codes.clamp(0, maxq).to(device="cpu", dtype=torch.int16),
        "sample_code_count": raw_codes.numel(),
        "nonfinite_count": int((~finite).sum().item()),
        "below_range_count": int((raw_codes < 0).sum().item()),
        "above_range_count": int((raw_codes > maxq).sum().item()),
    }


def sample_packed_quant_codes(
    qweight: torch.Tensor,
    *,
    bits: int,
    input_indexes: torch.Tensor,
    output_indexes: torch.Tensor,
) -> torch.Tensor | None:
    """Read sampled logical input-row/output-column codes from GPTQ qweight."""

    if qweight.ndim != 2 or qweight.element_size() * 8 != 32:
        return None
    input_indexes = input_indexes.to(device=qweight.device, dtype=torch.long)
    output_indexes = output_indexes.to(device=qweight.device, dtype=torch.long)
    words = qweight.to(dtype=torch.int64) & 0xFFFFFFFF
    maxq = (1 << int(bits)) - 1

    if bits in (2, 4, 8):
        pack_factor = 32 // bits
        packed_rows = torch.div(input_indexes, pack_factor, rounding_mode="floor")
        if packed_rows.numel() and int(packed_rows.max().item()) >= words.shape[0]:
            return None
        shifts = (input_indexes % pack_factor) * bits
        selected_words = words.index_select(0, packed_rows).index_select(1, output_indexes)
        codes = (selected_words >> shifts[:, None]) & maxq
        return codes.T.to(device="cpu", dtype=torch.int16)

    if bits != 3:
        return None
    blocks = torch.div(input_indexes, 32, rounding_mode="floor")
    if blocks.numel() and int((blocks.max() * 3 + 2).item()) >= words.shape[0]:
        return None
    offsets = input_indexes % 32
    sampled_rows = []
    for block, offset in zip(blocks.tolist(), offsets.tolist()):
        word0 = words[block * 3].index_select(0, output_indexes)
        word1 = words[block * 3 + 1].index_select(0, output_indexes)
        word2 = words[block * 3 + 2].index_select(0, output_indexes)
        if offset <= 9:
            row = (word0 >> (3 * offset)) & 0x7
        elif offset == 10:
            row = ((word0 >> 30) & 0x3) | (((word1 >> 0) << 2) & 0x4)
        elif offset <= 20:
            row = (word1 >> (1 + 3 * (offset - 11))) & 0x7
        elif offset == 21:
            row = ((word1 >> 31) & 0x1) | (((word2 >> 0) << 1) & 0x6)
        else:
            row = (word2 >> (2 + 3 * (offset - 22))) & 0x7
        sampled_rows.append(row)
    if not sampled_rows:
        return torch.empty((output_indexes.numel(), 0), dtype=torch.int16)
    return torch.stack(sampled_rows, dim=0).T.to(device="cpu", dtype=torch.int16)


def compare_quant_code_samples(reference: torch.Tensor, candidate: torch.Tensor | None) -> dict[str, Any]:
    """Compare two equally indexed sampled logical-code matrices."""

    if candidate is None or reference.shape != candidate.shape:
        return {
            "sample_code_count": reference.numel(),
            "mismatch_count": None,
            "mismatch_rate": None,
            "maximum_absolute_delta": None,
        }
    delta = (candidate.to(dtype=torch.int16) - reference.to(dtype=torch.int16)).abs()
    mismatch_count = int((delta != 0).sum().item())
    return {
        "sample_code_count": delta.numel(),
        "mismatch_count": mismatch_count,
        "mismatch_rate": mismatch_count / delta.numel() if delta.numel() else 0.0,
        "maximum_absolute_delta": int(delta.max().item()) if delta.numel() else 0,
    }


def summarize_quant_code_fingerprints(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate post-GPTQ, restored pre-pack, and packed sample comparisons."""

    rows = list(records)
    prepack_mismatches = sum(
        int(row["prepack"]["mismatch_count"] or 0)
        for row in rows
        if row["prepack"]["mismatch_count"] is not None
    )
    packed_mismatches = sum(
        int(row["packed"]["mismatch_count"] or 0)
        for row in rows
        if row["packed"]["mismatch_count"] is not None
    )
    sample_count = sum(int(row["prepack"]["sample_code_count"]) for row in rows)
    ranked = sorted(
        rows,
        key=lambda row: max(
            float(row["prepack"]["mismatch_rate"] or 0.0),
            float(row["packed"]["mismatch_rate"] or 0.0),
        ),
        reverse=True,
    )
    return {
        "module_count": len(rows),
        "sample_code_count": sample_count,
        "prepack_mismatch_count": prepack_mismatches,
        "prepack_mismatch_rate": prepack_mismatches / sample_count if sample_count else 0.0,
        "packed_mismatch_count": packed_mismatches,
        "packed_mismatch_rate": packed_mismatches / sample_count if sample_count else 0.0,
        "top": ranked[:5],
        "records": ranked,
    }


def _format_number(value: Any, *, digits: int = 6, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return str(numeric)
    return f"{numeric:.{digits}g}{suffix}"


def _format_percent(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.{digits}f}%"


def _markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> list[str]:
    escaped_rows = [
        [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        for row in rows
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in escaped_rows)
    return lines


def _display_module(record: Mapping[str, Any]) -> str:
    full_name = record.get("full_name")
    if full_name:
        return f"`{full_name}`"
    layer = record.get("layer")
    module = record.get("module", "unknown")
    if layer is None:
        return f"`{module}`"
    return f"layer {layer} / `{module}`"


def _diagnostic_findings(diagnostics: Mapping[str, Any]) -> list[str]:
    findings = []
    loss = diagnostics.get("loss") or {}
    if loss.get("severe_concentration"):
        worst = (loss.get("top") or [{}])[0]
        findings.append(
            "Observed severe loss concentration at "
            f"{_display_module(worst)}: {_format_percent(worst.get('total_loss_share'))} of total reported loss."
        )

    reconstruction = diagnostics.get("reconstruction") or {}
    records = reconstruction.get("records") or []
    nonfinite = [
        record
        for record in records
        if int(record.get("source_nonfinite_count", 0))
        or int(record.get("reconstructed_nonfinite_count", 0))
    ]
    if nonfinite:
        findings.append(
            f"Observed non-finite source or reconstructed weights in {len(nonfinite)} module(s); "
            "stop before packing and inspect the earliest module."
        )
    elif records:
        worst = records[0]
        findings.append(
            "Largest measured canonical reconstruction error: "
            f"{_display_module(worst)} at {_format_percent(worst.get('relative_rmse'))} relative RMSE. "
            "This is a localization lead, not a causal quality finding."
        )

    scales = diagnostics.get("scale_channels") or {}
    scale_records = scales.get("records") or []
    bad_scale_modules = [
        record
        for record in scale_records
        if int(record.get("nonfinite_scale_count", 0))
        or int(record.get("nonpositive_scale_count", 0))
    ]
    if bad_scale_modules:
        findings.append(
            f"Observed invalid scale values in {len(bad_scale_modules)} module(s); do not treat packing as valid."
        )

    fingerprints = diagnostics.get("code_fingerprints") or {}
    prepack_mismatch_count = int(fingerprints.get("prepack_mismatch_count", 0))
    packed_mismatch_count = int(fingerprints.get("packed_mismatch_count", 0))
    if prepack_mismatch_count:
        findings.append(
            f"Observed {prepack_mismatch_count} sampled code mismatches before packing. "
            "The canonical reconstructed weight changed after the GPTQ boundary."
        )
    if packed_mismatch_count:
        findings.append(
            f"Observed {packed_mismatch_count} sampled code mismatches after packing. "
            "Compare the packer with independent logical-code unpacking before testing optimized kernels."
        )
    if not findings:
        findings.append(
            "No severe boundary anomaly was observed in the enabled diagnostics. "
            "Continue with clean-reload, eager-dequantization, activation, and logit checks."
        )
    return findings


def _definition_group_recommendation_rows(diagnostics: Mapping[str, Any]) -> list[list[str]]:
    """Expand observed leads using the loaded GPTQModel definition's groups."""

    loss_records = (diagnostics.get("loss") or {}).get("records") or []
    reconstruction_records = (diagnostics.get("reconstruction") or {}).get("records") or []
    scale_records = (diagnostics.get("scale_channels") or {}).get("records") or []
    evidence: dict[str, list[str]] = {}
    policy: dict[str, tuple[Any, Any]] = {}

    def remember(record: Mapping[str, Any], reason: str | None = None) -> None:
        name = record.get("full_name")
        if not name:
            return
        name = str(name)
        policy.setdefault(name, (record.get("bits", "?"), record.get("group_size", "?")))
        if reason is not None:
            evidence.setdefault(name, []).append(reason)

    for rank, record in enumerate(loss_records[:12], start=1):
        remember(record, f"loss rank {rank}")
    for rank, record in enumerate(reconstruction_records[:12], start=1):
        remember(record, f"reconstruction rank {rank}")
    scale_candidates = [
        record
        for record in scale_records
        if float(record.get("max_to_median_ratio") or 0.0) >= 10.0
    ][:20]
    for record in scale_candidates:
        remember(record, f"scale max/median {_format_number(record.get('max_to_median_ratio'), digits=4)}×")

    for record in [*loss_records, *reconstruction_records, *scale_records]:
        remember(record)

    grouping = diagnostics.get("module_grouping") or {}
    module_groups = grouping.get("groups") or []
    group_source = str(grouping.get("source") or "not recorded")
    groups: dict[tuple[int, str], dict[str, Any]] = {}
    for name, reasons in evidence.items():
        match = _match_definition_group(name, module_groups)
        if match is None:
            continue
        group_index, suffixes, prefix, members = match
        key = (group_index, prefix)
        group = groups.setdefault(
            key,
            {
                "group": _definition_group_kind(suffixes),
                "group_index": group_index,
                "group_source": group_source,
                "members": members,
                "triggers": {},
                "reason": (
                    "GPTQModel declares these modules in the same supported model-definition group. "
                    "Serving-engine fusion is context, not the source of this closure."
                ),
            },
        )
        group["triggers"][name] = reasons

    rows = []
    for group in groups.values():
        existing_members = list(group["members"])
        triggers = [name for name in existing_members if name in group["triggers"]]
        companions = [name for name in existing_members if name not in group["triggers"]]
        policies = {
            f"{name.rsplit('.', 1)[-1]}={policy.get(name, ('?', '?'))[0]}b/g{policy.get(name, ('?', '?'))[1]}"
            for name in existing_members
        }
        matched = len({policy.get(name, ("?", "?")) for name in existing_members}) <= 1
        trigger_text = "; ".join(
            f"`{name}` ({', '.join(group['triggers'][name])})"
            for name in triggers
        )
        companion_text = ", ".join(f"`{name}`" for name in companions) or "none"
        recommendation = (
            "Current precision is group-compatible; if any trigger is promoted or exempted, apply the same "
            "tested policy to every companion before validation."
            if matched
            else "Align every member to the strictest tested precision before validation."
        )
        rows.append(
            [
                group["group"],
                str(group["group_index"]),
                f"`{group['group_source']}`",
                trigger_text,
                companion_text,
                ", ".join(sorted(policies)),
                f"{recommendation} {group['reason']}",
            ]
        )
    return rows


def _match_definition_group(
    module_name: str,
    module_groups: Iterable[Iterable[str]],
) -> tuple[int, list[str], str, list[str]] | None:
    for group_index, raw_group in enumerate(module_groups):
        suffixes = [str(member).split(":", 1)[0] for member in raw_group]
        suffixes = [member for member in suffixes if member]
        if len(suffixes) <= 1:
            continue
        for suffix in suffixes:
            if not module_name.endswith(suffix):
                continue
            prefix = module_name[: -len(suffix)]
            if prefix and not prefix.endswith("."):
                continue
            members = [f"{prefix}{candidate}" for candidate in suffixes]
            return group_index, suffixes, prefix, members
    return None


def _definition_group_kind(suffixes: Iterable[str]) -> str:
    leaves = {suffix.rsplit(".", 1)[-1] for suffix in suffixes}
    if leaves == {"q_proj", "k_proj", "v_proj"}:
        return "qkv_projection_group"
    if leaves == {"gate_proj", "up_proj"}:
        return "gate_up_projection_group"
    return "model_definition_group"


def render_quantization_diagnostics_markdown(diagnostics: Mapping[str, Any]) -> str:
    """Render the saved during-quantization evidence as a human-readable report."""

    mode = str(diagnostics.get("mode", "unknown"))
    loss = diagnostics.get("loss") or {}
    reconstruction = diagnostics.get("reconstruction") or {}
    output_error = diagnostics.get("output_error") or {}
    scale_channels = diagnostics.get("scale_channels") or {}
    fingerprints = diagnostics.get("code_fingerprints") or {}
    config = diagnostics.get("quantization_config") or {}
    lines = [
        "# During-quantization error analysis",
        "",
        "> Evidence status: **Observed during-quantization diagnostics**. These measurements identify the earliest "
        "suspicious numeric boundary; they do not independently prove downstream quality impact.",
        "",
        "## Executive summary",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["Item", "Value"],
            [
                ["Diagnostics mode", mode],
                ["Loss records", loss.get("module_count", 0)],
                ["Severe loss concentration", bool(loss.get("severe_concentration", False))],
                ["Reconstruction records", reconstruction.get("module_count", 0)],
                ["Post-quant output-error records", output_error.get("module_count", 0)],
                ["Post-quant sampled activation rows", output_error.get("sample_count", 0)],
                ["Mean module output MAE", _format_number(output_error.get("mean_module_absolute_error"))],
                ["Mean module softmax KLD", _format_number(output_error.get("mean_module_softmax_kld"))],
                ["Scale/channel records", scale_channels.get("module_count", 0)],
                ["Code-fingerprint records", fingerprints.get("module_count", 0)],
                ["Pre-pack sampled code mismatches", fingerprints.get("prepack_mismatch_count", 0)],
                ["Post-pack sampled code mismatches", fingerprints.get("packed_mismatch_count", 0)],
            ],
        )
    )
    lines.extend(["", "## Quantization configuration", ""])
    lines.extend(
        _markdown_table(
            ["Setting", "Value"],
            [
                ["Method / format", f"{config.get('method', 'unknown')} / {config.get('format', 'unknown')}"],
                ["Default weight precision", f"{config.get('bits', 'unknown')}-bit"],
                ["Default group size", config.get("group_size", "unknown")],
                ["Symmetric", config.get("sym", "unknown")],
                ["Activation order", config.get("desc_act", "unknown")],
                ["Pack implementation", config.get("pack_impl", "unknown")],
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Index semantics",
            "",
            "- Layer indices are emitted exactly as received from the model lifecycle; no `+1` conversion is performed.",
            "- Output rows, input features, groups, and sampled code coordinates are zero-based tensor indices.",
            "- Canonical linear weights use `[output_row, input_feature]` in the reconstruction tables.",
            "- Quantizer-returned GPTQ scales use `[output_channel, group]` before packing.",
            "",
            "## Boundary findings",
            "",
        ]
    )
    lines.extend(f"- {finding}" for finding in _diagnostic_findings(diagnostics))

    loss_records = loss.get("records") or loss.get("top") or []
    lines.extend(["", "## Quantization-loss ranking", ""])
    if loss_records:
        lines.extend(
            _markdown_table(
                [
                    "Module",
                    "Loss",
                    "Role percentile",
                    "Role median ratio",
                    "Total loss share",
                    "Bits / group",
                ],
                [
                    [
                        _display_module(record),
                        _format_number(record.get("loss")),
                        _format_number(record.get("role_percentile"), digits=4, suffix="%"),
                        _format_number(record.get("role_median_ratio"), digits=4, suffix="×"),
                        _format_percent(record.get("total_loss_share")),
                        f"{record.get('bits', '?')} / {record.get('group_size', '?')}",
                    ]
                    for record in loss_records[:40]
                ],
            )
        )
    else:
        lines.append("No valid module-loss records were available.")

    reconstruction_records = reconstruction.get("records") or []
    lines.extend(["", "## Canonical GPTQ reconstruction error", ""])
    if reconstruction_records:
        lines.extend(
            _markdown_table(
                [
                    "Module",
                    "Relative RMSE",
                    "RMSE",
                    "Cosine",
                    "SQNR dB",
                    "Max abs. error coordinate",
                    "Bits / group",
                ],
                [
                    [
                        _display_module(record),
                        _format_percent(record.get("relative_rmse")),
                        _format_number(record.get("rmse")),
                        _format_number(record.get("cosine_similarity")),
                        _format_number(record.get("sqnr_db")),
                        (
                            f"row {record.get('maximum_error_output_row')}, "
                            f"feature {record.get('maximum_error_input_feature')}: "
                            f"{_format_number(record.get('maximum_absolute_error'))}"
                        ),
                        f"{record.get('bits', '?')} / {record.get('group_size', '?')}",
                    ]
                    for record in reconstruction_records[:40]
                ],
            )
        )
    else:
        lines.append(
            "Reconstruction diagnostics were not collected. Use `quantization_diagnostics=\"channel\"` "
            "for an investigation run."
        )

    output_error_records = output_error.get("records") or []
    lines.extend(["", "## Post-quantization output error", ""])
    lines.append(
        "Metrics replay the same bounded calibration inputs through dense and reconstructed GPTQ weights. "
        "Softmax KLD is token-distribution KLD only for an LM head; for hidden projections it is a sensitivity signal."
    )
    lines.append("")
    if output_error_records:
        lines.extend(
            _markdown_table(
                ["Module", "Samples", "Mean abs. error", "Relative L2", "Mean KLD", "P95 KLD", "Top-1"],
                [
                    [
                        _display_module(record),
                        record.get("sample_count", 0),
                        _format_number(record.get("mean_absolute_error")),
                        _format_percent(record.get("relative_l2_error")),
                        _format_number(record.get("softmax_kld_mean")),
                        _format_number(record.get("softmax_kld_p95")),
                        _format_percent(record.get("top1_agreement")),
                    ]
                    for record in output_error_records[:40]
                ],
            )
        )
    else:
        lines.append(
            "Output-error telemetry was not collected. Use `quantization_diagnostics=\"channel\"` for an investigation run."
        )

    lines.extend(["", "## Localized reconstruction rows and input features", ""])
    localized_rows = []
    for record in reconstruction_records[:40]:
        worst_row = (record.get("top_output_rows") or [{}])[0]
        worst_feature = (record.get("top_input_features") or [{}])[0]
        localized_rows.append(
            [
                _display_module(record),
                (
                    f"row {worst_row.get('output_row', 'n/a')}: "
                    f"{_format_percent(worst_row.get('relative_rmse'))}, "
                    f"signal L2 {_format_number(worst_row.get('signal_l2'))}"
                ),
                (
                    f"feature {worst_feature.get('input_feature', 'n/a')}: "
                    f"{_format_percent(worst_feature.get('relative_rmse'))}, "
                    f"signal L2 {_format_number(worst_feature.get('signal_l2'))}"
                ),
            ]
        )
    if localized_rows:
        lines.extend(_markdown_table(["Module", "Worst output row", "Worst input feature"], localized_rows))
    else:
        lines.append("No row/feature reconstruction records were available.")

    scale_records = scale_channels.get("records") or scale_channels.get("top") or []
    lines.extend(["", "## Scale and group-index candidates", ""])
    if scale_records:
        lines.extend(
            _markdown_table(
                [
                    "Module",
                    "Max scale coordinate",
                    "Max scale",
                    "P99 channel max",
                    "Max/median",
                    "Invalid scales",
                    "g_idx",
                ],
                [
                    [
                        _display_module(record),
                        (
                            f"channel {record.get('max_scale_output_channel')}, "
                            f"group {record.get('max_scale_group')}"
                        ),
                        _format_number(record.get("max_scale")),
                        _format_number(record.get("p99_channel_max_scale")),
                        _format_number(record.get("max_to_median_ratio"), digits=4, suffix="×"),
                        (
                            f"nonfinite={record.get('nonfinite_scale_count', 0)}, "
                            f"nonpositive={record.get('nonpositive_scale_count', 0)}"
                        ),
                        (
                            f"groups={record.get('group_index', {}).get('unique_group_count', 'n/a')}, "
                            f"decreases={record.get('group_index', {}).get('monotonic_decrease_count', 'n/a')}"
                        ),
                    ]
                    for record in scale_records[:40]
                ],
            )
        )
    else:
        lines.append(
            "Scale/channel diagnostics were not collected. Use `quantization_diagnostics=\"channel\"` "
            "for an investigation run."
        )

    fingerprint_records = fingerprints.get("records") or fingerprints.get("top") or []
    lines.extend(["", "## Logical-code lifecycle", ""])
    if fingerprint_records:
        lines.extend(
            _markdown_table(
                [
                    "Module",
                    "Samples",
                    "GPTQ→pre-pack mismatches",
                    "GPTQ→packed mismatches",
                    "Range excursions",
                    "Packer",
                ],
                [
                    [
                        _display_module(record),
                        record.get("post_gptq", {}).get("sample_code_count", 0),
                        (
                            f"{record.get('prepack', {}).get('mismatch_count')} "
                            f"({_format_percent(record.get('prepack', {}).get('mismatch_rate'))})"
                        ),
                        (
                            f"{record.get('packed', {}).get('mismatch_count')} "
                            f"({_format_percent(record.get('packed', {}).get('mismatch_rate'))})"
                        ),
                        (
                            f"below={record.get('post_gptq', {}).get('below_range_count', 0)}, "
                            f"above={record.get('post_gptq', {}).get('above_range_count', 0)}, "
                            f"nonfinite={record.get('post_gptq', {}).get('nonfinite_count', 0)}"
                        ),
                        record.get("packer", "unknown"),
                    ]
                    for record in fingerprint_records[:40]
                ],
            )
        )
    else:
        lines.append(
            "Logical-code fingerprints were not collected. Use `quantization_diagnostics=\"channel\"` "
            "for an investigation run."
        )

    fusion_rows = _definition_group_recommendation_rows(diagnostics)
    lines.extend(["", "## GPTQModel definition-group recommendations", ""])
    lines.extend(
        [
            "A group companion is recommended because the loaded GPTQModel definition places the modules in one "
            "supported group. It is **not** evidence that the companion itself has abnormal local error. "
            "The model definition—not a vLLM/SGLang name heuristic—is the fusibility authority.",
            "",
        ]
    )
    if fusion_rows:
        lines.extend(
            _markdown_table(
                [
                    "Definition group",
                    "Index",
                    "Definition source",
                    "Observed trigger(s)",
                    "Group companion(s)",
                    "Current policy",
                    "Recommendation",
                ],
                fusion_rows,
            )
        )
    else:
        lines.append(
            "No definition-group closure was induced. Group metadata may be unavailable, or no ranked finding "
            "matched a complete supported group."
        )
    lines.extend(
        [
            "",
            "For Qwen3, `o_proj` and `down_proj` have separate model-definition group IDs, so an anomaly in either "
            "does not automatically promote Q/K/V or gate/up weights.",
        ]
    )

    lines.extend(
        [
            "",
            "## Interpretation and next experiments",
            "",
            "- If canonical reconstruction is already poor, test a matched higher-bit/group-size override before "
            "debugging packing or inference kernels.",
            "- If GPTQ→pre-pack codes differ, inspect processors, replay, restoration, smoothing, adapters, and aliasing.",
            "- If only GPTQ→packed codes differ, independently unpack the saved codes and compare pack implementations.",
            "- If these boundaries agree, continue with clean reload, eager dequantization, layerwise activations, "
            "logit KLD, paired answer flips, and optimized-backend comparisons.",
            "- Treat high relative error on a near-zero row or feature as low-signal until activation-weighted evidence "
            "shows that the coordinate matters.",
            "",
            "## Limitations",
            "",
            "- Module loss is the quantizer's calibration objective and can overfit calibration data.",
            "- Reconstruction metrics are weight-local and are not activation-, token-, route-, or logit-weighted.",
            "- Logical-code fingerprints are bounded samples, not a complete unpack of every code.",
            "- A candidate becomes causal only after an independent precision rescue or downstream reproduction.",
            "",
        ]
    )
    return "\n".join(lines)
