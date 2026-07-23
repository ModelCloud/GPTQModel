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
            "loss": loss,
        }
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

    values = scales.detach().to(dtype=torch.float32).abs()
    finite = torch.isfinite(values)
    nonfinite_count = int((~finite).sum().item())
    values = torch.where(finite, values, torch.zeros_like(values))

    if values.ndim == 0:
        channel_max = values.reshape(1)
    elif values.ndim == 1:
        channel_max = values
    else:
        channel_max = values.amax(dim=tuple(range(1, values.ndim)))

    max_scale, max_channel = channel_max.max(dim=0)
    median_scale = channel_max.median()
    p99_scale = torch.quantile(channel_max, 0.99)

    max_value = float(max_scale.item())
    median_value = float(median_scale.item())
    return {
        "scale_element_count": values.numel(),
        "output_channel_count": channel_max.numel(),
        "nonfinite_scale_count": nonfinite_count,
        "max_scale": max_value,
        "p99_channel_max_scale": float(p99_scale.item()),
        "median_channel_max_scale": median_value,
        "max_to_median_ratio": max_value / median_value if median_value > 0 else None,
        "max_scale_output_channel": int(max_channel.item()),
        "scale_count_above_10": int((values > 10.0).sum().item()),
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
    }
