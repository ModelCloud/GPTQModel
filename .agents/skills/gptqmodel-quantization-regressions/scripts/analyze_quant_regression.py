#!/usr/bin/env python3
"""Compare quantizer losses and optional saved-scale output-channel outliers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path
from typing import Any


def parse_snapshot(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--snapshot must use LABEL=/path/to/model")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser().resolve()
    if not label:
        raise argparse.ArgumentTypeError("snapshot label cannot be empty")
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"snapshot path is not a directory: {path}")
    if not (path / "quant_log.csv").is_file():
        raise argparse.ArgumentTypeError(f"snapshot has no quant_log.csv: {path}")
    return label, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        action="append",
        required=True,
        type=parse_snapshot,
        metavar="LABEL=PATH",
        help="Snapshot label and model directory; the first snapshot is the reference.",
    )
    parser.add_argument("--scan-scales", action="store_true", help="Read every saved .scales tensor.")
    parser.add_argument(
        "--scan-codes",
        action="store_true",
        help="Compare logical packed-weight codes and exact scale/zero/group metadata; requires matching bit widths.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Rows to show per outlier table.")
    parser.add_argument("--json", type=Path, help="Optional machine-readable result path.")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    labels = [label for label, _path in args.snapshot]
    if len(set(labels)) != len(labels):
        parser.error("--snapshot labels must be unique")
    return args


def load_losses(path: Path) -> list[dict[str, Any]]:
    rows = []
    with (path / "quant_log.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                loss = float(row["loss"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(loss) or loss < 0:
                continue
            try:
                layer: int | str = int(row["layer"])
            except (KeyError, TypeError, ValueError):
                layer = row.get("layer", "unknown")
            rows.append({"layer": layer, "module": row.get("module", "unknown"), "loss": loss})
    return rows


def summarize_losses(rows: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    if not rows:
        return {"module_count": 0, "mean_loss": None, "median_loss": None, "top": []}
    by_role: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_role[row["module"]].append(row["loss"])
    role_medians = {role: statistics.median(values) for role, values in by_role.items()}
    total = math.fsum(row["loss"] for row in rows)
    enriched = []
    for row in rows:
        role_median = role_medians[row["module"]]
        enriched.append(
            {
                **row,
                "role_median_loss": role_median,
                "role_median_ratio": row["loss"] / role_median if role_median > 0 else None,
                "total_loss_share": row["loss"] / total if total > 0 else 0.0,
            }
        )
    enriched.sort(key=lambda row: row["loss"], reverse=True)
    losses = [row["loss"] for row in rows]
    return {
        "module_count": len(rows),
        "mean_loss": total / len(rows),
        "median_loss": statistics.median(losses),
        "total_loss": total,
        "top": enriched[:top_k],
    }


def compare_losses(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    reference_by_key = {
        (row["layer"], row["module"]): row["loss"]
        for row in reference["loss_rows"]
    }
    matched = []
    for row in candidate["loss_rows"]:
        key = (row["layer"], row["module"])
        reference_loss = reference_by_key.get(key)
        if reference_loss is None or reference_loss <= 0:
            continue
        ratio = row["loss"] / reference_loss
        matched.append({**row, "reference_loss": reference_loss, "ratio": ratio})
    median_ratio = statistics.median(row["ratio"] for row in matched) if matched else None
    for row in matched:
        row["normalized_degradation"] = (
            row["ratio"] / median_ratio if median_ratio is not None and median_ratio > 0 else None
        )
    matched.sort(key=lambda row: row["normalized_degradation"] or 0.0, reverse=True)
    return {
        "reference": reference["label"],
        "candidate": candidate["label"],
        "matched_module_count": len(matched),
        "median_loss_ratio": median_ratio,
        "top": matched[:top_k],
    }


def saved_tensor_map(path: Path, suffix: str) -> dict[str, str]:
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        return {
            name: shard
            for name, shard in payload["weight_map"].items()
            if name.endswith(suffix)
        }
    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no Safetensors model shards found in {path}")
    from safetensors import safe_open

    result = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name.endswith(suffix):
                    result[name] = shard.name
    return result


def scale_tensor_map(path: Path) -> dict[str, str]:
    return saved_tensor_map(path, ".scales")


def load_quantization_bits(path: Path) -> int:
    quant_config_path = path / "quantize_config.json"
    if quant_config_path.is_file():
        payload = json.loads(quant_config_path.read_text(encoding="utf-8"))
    else:
        config_path = path / "config.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        payload = payload.get("quantization_config", payload)
    try:
        bits = int(payload["bits"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot resolve quantization bits for {path}") from exc
    if bits not in (2, 3, 4, 8):
        raise ValueError(f"unsupported packed-code width {bits} for {path}")
    return bits


def _open_tensor_handles(stack: ExitStack, path: Path, tensor_map: dict[str, str]) -> dict[str, Any]:
    from safetensors import safe_open

    return {
        shard: stack.enter_context(safe_open(path / shard, framework="pt", device="cpu"))
        for shard in sorted(set(tensor_map.values()))
    }


def count_logical_code_differences(reference: Any, candidate: Any, bits: int) -> dict[str, Any]:
    """Compare packed GPTQ codes without materializing a whole model's unpacked INT32 weights."""

    import torch

    if reference.shape != candidate.shape:
        raise ValueError(f"packed shape mismatch: {tuple(reference.shape)} != {tuple(candidate.shape)}")
    if reference.dtype != candidate.dtype:
        raise ValueError(f"packed dtype mismatch: {reference.dtype} != {candidate.dtype}")
    if reference.ndim != 2:
        raise ValueError(f"expected a rank-2 qweight tensor, got shape {tuple(reference.shape)}")

    packed_word_count = reference.numel()
    if torch.equal(reference, candidate):
        if bits == 3:
            if reference.shape[0] % 3:
                raise ValueError(f"3-bit packed rows must be divisible by 3, got {reference.shape[0]}")
            code_count = reference.shape[0] // 3 * 32 * reference.shape[1]
        else:
            pack_bits = reference.element_size() * 8
            if pack_bits % bits:
                raise ValueError(f"{bits}-bit codes do not divide {pack_bits}-bit packing words")
            code_count = packed_word_count * (pack_bits // bits)
        return {
            "packed_word_count": packed_word_count,
            "packed_word_match_count": packed_word_count,
            "packed_word_match_rate": 1.0,
            "code_count": code_count,
            "code_mismatch_count": 0,
            "code_mismatch_rate": 0.0,
            "mean_absolute_code_delta": 0.0,
            "maximum_absolute_code_delta": 0,
            "_absolute_code_delta_sum": 0,
        }

    packed_word_match_count = 0
    code_count = 0
    code_mismatch_count = 0
    absolute_code_delta_sum = 0
    maximum_absolute_code_delta = 0

    if bits == 3:
        if reference.shape[0] % 3:
            raise ValueError(f"3-bit packed rows must be divisible by 3, got {reference.shape[0]}")
        packed_rows_per_chunk = 96
        for row_start in range(0, reference.shape[0], packed_rows_per_chunk):
            row_end = min(row_start + packed_rows_per_chunk, reference.shape[0])
            reference_chunk = reference[row_start:row_end]
            candidate_chunk = candidate[row_start:row_end]
            packed_word_match_count += int((reference_chunk == candidate_chunk).sum().item())
            reference_codes = unpack_three_bit_rows(reference_chunk)
            candidate_codes = unpack_three_bit_rows(candidate_chunk)
            delta = (candidate_codes - reference_codes).abs()
            code_count += delta.numel()
            code_mismatch_count += int((delta != 0).sum().item())
            absolute_code_delta_sum += int(delta.sum(dtype=torch.int64).item())
            maximum_absolute_code_delta = max(maximum_absolute_code_delta, int(delta.max().item()))
    else:
        import numpy as np

        pack_bits = reference.element_size() * 8
        if pack_bits % bits:
            raise ValueError(f"{bits}-bit codes do not divide {pack_bits}-bit packing words")
        if 8 % bits:
            raise ValueError(f"{bits}-bit codes cannot use the byte-pair comparison path")
        mismatch_lut, delta_sum_lut, delta_max_lut = byte_pair_code_luts(bits)
        reference_numpy = reference.contiguous().numpy()
        candidate_numpy = candidate.contiguous().numpy()
        packed_word_match_count = int(np.count_nonzero(reference_numpy == candidate_numpy))
        reference_bytes = reference_numpy.view(np.uint8).reshape(-1)
        candidate_bytes = candidate_numpy.view(np.uint8).reshape(-1)
        code_count = reference_bytes.size * (8 // bits)
        byte_chunk_size = 16 * 1024 * 1024
        for byte_start in range(0, reference_bytes.size, byte_chunk_size):
            byte_end = min(byte_start + byte_chunk_size, reference_bytes.size)
            pair_indexes = (
                reference_bytes[byte_start:byte_end].astype(np.uint16) * 256
                + candidate_bytes[byte_start:byte_end]
            )
            code_mismatch_count += int(mismatch_lut[pair_indexes].sum(dtype=np.uint64))
            absolute_code_delta_sum += int(delta_sum_lut[pair_indexes].sum(dtype=np.uint64))
            maximum_absolute_code_delta = max(
                maximum_absolute_code_delta,
                int(delta_max_lut[pair_indexes].max(initial=0)),
            )

    return {
        "packed_word_count": packed_word_count,
        "packed_word_match_count": packed_word_match_count,
        "packed_word_match_rate": packed_word_match_count / packed_word_count if packed_word_count else None,
        "code_count": code_count,
        "code_mismatch_count": code_mismatch_count,
        "code_mismatch_rate": code_mismatch_count / code_count if code_count else None,
        "mean_absolute_code_delta": absolute_code_delta_sum / code_count if code_count else None,
        "maximum_absolute_code_delta": maximum_absolute_code_delta,
        "_absolute_code_delta_sum": absolute_code_delta_sum,
    }


@lru_cache(maxsize=3)
def byte_pair_code_luts(bits: int) -> tuple[Any, Any, Any]:
    """Return per-byte-pair mismatch, absolute-delta-sum, and maximum-delta lookup tables."""

    import numpy as np

    if bits not in (2, 4, 8):
        raise ValueError(f"byte-pair code tables do not support {bits}-bit values")
    reference = np.arange(256, dtype=np.uint16)[:, None]
    candidate = np.arange(256, dtype=np.uint16)[None, :]
    mismatch_count = np.zeros((256, 256), dtype=np.uint8)
    absolute_delta_sum = np.zeros((256, 256), dtype=np.uint16)
    maximum_absolute_delta = np.zeros((256, 256), dtype=np.uint8)
    code_mask = (1 << bits) - 1
    for shift in range(0, 8, bits):
        reference_code = (reference >> shift) & code_mask
        candidate_code = (candidate >> shift) & code_mask
        delta = np.abs(candidate_code.astype(np.int16) - reference_code.astype(np.int16)).astype(np.uint8)
        mismatch_count += delta != 0
        absolute_delta_sum += delta
        maximum_absolute_delta = np.maximum(maximum_absolute_delta, delta)
    return (
        mismatch_count.reshape(-1),
        absolute_delta_sum.reshape(-1),
        maximum_absolute_delta.reshape(-1),
    )


def unpack_three_bit_rows(packed: Any) -> Any:
    """Unpack GPTQ's three-INT32-words-per-32-row 3-bit layout."""

    import torch

    if packed.element_size() * 8 != 32:
        raise ValueError("3-bit GPTQ code comparison requires 32-bit packing words")
    rows, columns = packed.shape
    if rows % 3:
        raise ValueError(f"3-bit packed rows must be divisible by 3, got {rows}")

    blocks = rows // 3
    words = packed.to(dtype=torch.int64).reshape(blocks, 3, columns)
    word0 = words[:, 0, :]
    word1 = words[:, 1, :]
    word2 = words[:, 2, :]
    result = torch.empty((blocks, 32, columns), dtype=torch.int32)
    for index in range(10):
        result[:, index, :] = ((word0 >> (3 * index)) & 0x7).to(dtype=torch.int32)
    result[:, 10, :] = (
        ((word0 >> 30) & 0x3) | (((word1 >> 0) << 2) & 0x4)
    ).to(dtype=torch.int32)
    for index in range(10):
        result[:, 11 + index, :] = ((word1 >> (1 + 3 * index)) & 0x7).to(dtype=torch.int32)
    result[:, 21, :] = (
        ((word1 >> 31) & 0x1) | (((word2 >> 0) << 1) & 0x6)
    ).to(dtype=torch.int32)
    for index in range(10):
        result[:, 22 + index, :] = ((word2 >> (2 + 3 * index)) & 0x7).to(dtype=torch.int32)
    return result.reshape(blocks * 32, columns)


def compare_tensor_category(reference_path: Path, candidate_path: Path, suffix: str) -> dict[str, Any]:
    import torch

    reference_map = saved_tensor_map(reference_path, suffix)
    candidate_map = saved_tensor_map(candidate_path, suffix)
    common_names = sorted(reference_map.keys() & candidate_map.keys())
    exact_count = 0
    shape_mismatch_count = 0
    dtype_mismatch_count = 0
    mismatched_tensors = []
    with ExitStack() as stack:
        reference_handles = _open_tensor_handles(stack, reference_path, reference_map)
        candidate_handles = _open_tensor_handles(stack, candidate_path, candidate_map)
        for name in common_names:
            reference = reference_handles[reference_map[name]].get_tensor(name)
            candidate = candidate_handles[candidate_map[name]].get_tensor(name)
            if reference.shape != candidate.shape:
                shape_mismatch_count += 1
                mismatched_tensors.append(name)
                continue
            if reference.dtype != candidate.dtype:
                dtype_mismatch_count += 1
                mismatched_tensors.append(name)
                continue
            if torch.equal(reference, candidate):
                exact_count += 1
            else:
                mismatched_tensors.append(name)
    return {
        "suffix": suffix,
        "reference_tensor_count": len(reference_map),
        "candidate_tensor_count": len(candidate_map),
        "matched_tensor_count": len(common_names),
        "exact_tensor_count": exact_count,
        "shape_mismatch_count": shape_mismatch_count,
        "dtype_mismatch_count": dtype_mismatch_count,
        "mismatched_tensors": mismatched_tensors,
    }


def compare_packed_codes(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    reference_path = Path(reference["path"])
    candidate_path = Path(candidate["path"])
    reference_bits = load_quantization_bits(reference_path)
    candidate_bits = load_quantization_bits(candidate_path)
    result = {
        "reference": reference["label"],
        "candidate": candidate["label"],
        "reference_bits": reference_bits,
        "candidate_bits": candidate_bits,
    }
    if reference_bits != candidate_bits:
        result["skipped_reason"] = (
            f"logical-code comparison requires matching bits ({reference_bits} != {candidate_bits})"
        )
        return result

    reference_map = saved_tensor_map(reference_path, ".qweight")
    candidate_map = saved_tensor_map(candidate_path, ".qweight")
    common_names = sorted(reference_map.keys() & candidate_map.keys())
    rows = []
    totals = {
        "packed_word_count": 0,
        "packed_word_match_count": 0,
        "code_count": 0,
        "code_mismatch_count": 0,
        "absolute_code_delta_sum": 0,
        "maximum_absolute_code_delta": 0,
        "exact_packed_tensor_count": 0,
        "shape_or_dtype_mismatch_count": 0,
    }
    started = time.perf_counter()
    with ExitStack() as stack:
        reference_handles = _open_tensor_handles(stack, reference_path, reference_map)
        candidate_handles = _open_tensor_handles(stack, candidate_path, candidate_map)
        for name in common_names:
            reference_tensor = reference_handles[reference_map[name]].get_tensor(name)
            candidate_tensor = candidate_handles[candidate_map[name]].get_tensor(name)
            try:
                row = count_logical_code_differences(reference_tensor, candidate_tensor, reference_bits)
            except ValueError:
                totals["shape_or_dtype_mismatch_count"] += 1
                continue
            row["tensor"] = name
            rows.append(row)
            totals["packed_word_count"] += row["packed_word_count"]
            totals["packed_word_match_count"] += row["packed_word_match_count"]
            totals["code_count"] += row["code_count"]
            totals["code_mismatch_count"] += row["code_mismatch_count"]
            totals["absolute_code_delta_sum"] += row["_absolute_code_delta_sum"]
            totals["maximum_absolute_code_delta"] = max(
                totals["maximum_absolute_code_delta"],
                row["maximum_absolute_code_delta"],
            )
            if row["code_mismatch_count"] == 0:
                totals["exact_packed_tensor_count"] += 1

    rows.sort(key=lambda row: row["code_mismatch_rate"] or 0.0, reverse=True)
    code_count = totals["code_count"]
    packed_word_count = totals["packed_word_count"]
    result.update(
        {
            "matched_packed_tensor_count": len(common_names),
            **totals,
            "packed_word_match_rate": (
                totals["packed_word_match_count"] / packed_word_count if packed_word_count else None
            ),
            "code_mismatch_rate": (
                totals["code_mismatch_count"] / code_count if code_count else None
            ),
            "mean_absolute_code_delta": (
                totals["absolute_code_delta_sum"] / code_count if code_count else None
            ),
            "scan_wall_s": time.perf_counter() - started,
            "top": rows[:top_k],
            "metadata": {
                suffix: compare_tensor_category(reference_path, candidate_path, suffix)
                for suffix in (".scales", ".qzeros", ".g_idx")
            },
        }
    )
    return result


def scan_scales(path: Path, top_k: int) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    tensor_map = scale_tensor_map(path)
    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in tensor_map.items():
        by_shard[shard].append(name)

    rows = []
    channel_maxima = {}
    started = time.perf_counter()
    for shard, names in sorted(by_shard.items()):
        with safe_open(path / shard, framework="pt", device="cpu") as handle:
            for name in sorted(names):
                scales = handle.get_tensor(name).to(dtype=torch.float32).abs()
                finite = torch.isfinite(scales)
                nonfinite_count = int((~finite).sum().item())
                scales = torch.where(finite, scales, torch.zeros_like(scales))
                if scales.ndim <= 1:
                    channel_max = scales.reshape(-1)
                else:
                    channel_max = scales.amax(dim=tuple(range(scales.ndim - 1)))
                max_scale, channel = channel_max.max(dim=0)
                median = channel_max.median()
                maximum = float(max_scale.item())
                median_value = float(median.item())
                row = {
                    "tensor": name,
                    "max_scale": maximum,
                    "output_channel": int(channel.item()),
                    "p99_channel_max_scale": float(torch.quantile(channel_max, 0.99).item()),
                    "median_channel_max_scale": median_value,
                    "max_to_median_ratio": maximum / median_value if median_value > 0 else None,
                    "scale_count_above_10": int((scales > 10.0).sum().item()),
                    "nonfinite_scale_count": nonfinite_count,
                }
                rows.append(row)
                channel_maxima[name] = channel_max
    rows.sort(key=lambda row: row["max_scale"], reverse=True)
    return {
        "tensor_count": len(rows),
        "scan_wall_s": time.perf_counter() - started,
        "top": rows[:top_k],
        "_channel_maxima": channel_maxima,
    }


def compare_scale_channels(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    import torch

    reference_maxima = reference["scales"]["_channel_maxima"]
    candidate_maxima = candidate["scales"]["_channel_maxima"]
    records = []
    all_ratios = []
    for name in sorted(reference_maxima.keys() & candidate_maxima.keys()):
        reference_values = reference_maxima[name]
        candidate_values = candidate_maxima[name]
        if reference_values.shape != candidate_values.shape:
            continue
        valid = reference_values > 0
        ratios = torch.zeros_like(reference_values)
        ratios[valid] = candidate_values[valid] / reference_values[valid]
        all_ratios.extend(ratios[valid].tolist())
        count = min(top_k, ratios.numel())
        values, indexes = torch.topk(ratios, k=count)
        for ratio, index in zip(values.tolist(), indexes.tolist()):
            records.append(
                {
                    "tensor": name,
                    "output_channel": index,
                    "reference_scale": float(reference_values[index].item()),
                    "candidate_scale": float(candidate_values[index].item()),
                    "ratio": ratio,
                }
            )
    median_ratio = statistics.median(all_ratios) if all_ratios else None
    for row in records:
        row["normalized_degradation"] = (
            row["ratio"] / median_ratio if median_ratio is not None and median_ratio > 0 else None
        )
    records.sort(key=lambda row: row["normalized_degradation"] or 0.0, reverse=True)
    return {
        "reference": reference["label"],
        "candidate": candidate["label"],
        "matched_channel_count": len(all_ratios),
        "median_scale_ratio": median_ratio,
        "top": records[:top_k],
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_table(headers: list[str], rows: list[list[Any]]) -> None:
    rendered = [[fmt(value) for value in row] for row in rows]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rendered))
        for index in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rendered:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def printable_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: printable_payload(item)
            for key, item in value.items()
            if not key.startswith("_")
        }
    if isinstance(value, list):
        return [printable_payload(item) for item in value]
    return value


def main() -> int:
    args = parse_args()
    snapshots = []
    for label, path in args.snapshot:
        rows = load_losses(path)
        snapshot = {
            "label": label,
            "path": str(path),
            "loss_rows": rows,
            "loss_summary": summarize_losses(rows, args.top_k),
        }
        if args.scan_scales:
            snapshot["scales"] = scan_scales(path, args.top_k)
        snapshots.append(snapshot)

    print("\nSnapshot loss summary")
    print_table(
        ["snapshot", "modules", "mean loss", "median loss", "worst layer/module", "worst loss", "total share"],
        [
            [
                snapshot["label"],
                snapshot["loss_summary"]["module_count"],
                snapshot["loss_summary"]["mean_loss"],
                snapshot["loss_summary"]["median_loss"],
                (
                    f"{snapshot['loss_summary']['top'][0]['layer']}/"
                    f"{snapshot['loss_summary']['top'][0]['module']}"
                    if snapshot["loss_summary"]["top"]
                    else "-"
                ),
                snapshot["loss_summary"]["top"][0]["loss"] if snapshot["loss_summary"]["top"] else None,
                (
                    f"{100 * snapshot['loss_summary']['top'][0]['total_loss_share']:.2f}%"
                    if snapshot["loss_summary"]["top"]
                    else "-"
                ),
            ]
            for snapshot in snapshots
        ],
    )

    reference = snapshots[0]
    loss_comparisons = [
        compare_losses(reference, candidate, args.top_k)
        for candidate in snapshots[1:]
    ]
    for comparison in loss_comparisons:
        print(f"\nLoss degradation: {comparison['candidate']} vs {comparison['reference']}")
        print(f"Matched median candidate/reference ratio: {fmt(comparison['median_loss_ratio'])}x")
        print_table(
            ["layer", "module", "reference", "candidate", "ratio", "normalized"],
            [
                [
                    row["layer"],
                    row["module"],
                    row["reference_loss"],
                    row["loss"],
                    f"{row['ratio']:.3f}x",
                    f"{row['normalized_degradation']:.3f}x",
                ]
                for row in comparison["top"]
            ],
        )

    scale_comparisons = []
    if args.scan_scales:
        for snapshot in snapshots:
            print(
                f"\nScale scan: {snapshot['label']} "
                f"({snapshot['scales']['tensor_count']} tensors in {snapshot['scales']['scan_wall_s']:.3f}s)"
            )
            print_table(
                ["tensor", "channel", "max", "p99", "max/median", ">10", "nonfinite"],
                [
                    [
                        row["tensor"],
                        row["output_channel"],
                        row["max_scale"],
                        row["p99_channel_max_scale"],
                        row["max_to_median_ratio"],
                        row["scale_count_above_10"],
                        row["nonfinite_scale_count"],
                    ]
                    for row in snapshot["scales"]["top"]
                ],
            )
        scale_comparisons = [
            compare_scale_channels(reference, candidate, args.top_k)
            for candidate in snapshots[1:]
        ]
        for comparison in scale_comparisons:
            print(f"\nScale-channel degradation: {comparison['candidate']} vs {comparison['reference']}")
            print(f"Matched median candidate/reference ratio: {fmt(comparison['median_scale_ratio'])}x")
            print_table(
                ["tensor", "channel", "reference", "candidate", "ratio", "normalized"],
                [
                    [
                        row["tensor"],
                        row["output_channel"],
                        row["reference_scale"],
                        row["candidate_scale"],
                        f"{row['ratio']:.3f}x",
                        f"{row['normalized_degradation']:.3f}x",
                    ]
                    for row in comparison["top"]
                ],
            )

    code_comparisons = []
    if args.scan_codes:
        code_comparisons = [
            compare_packed_codes(reference, candidate, args.top_k)
            for candidate in snapshots[1:]
        ]
        for comparison in code_comparisons:
            print(f"\nPacked-code comparison: {comparison['candidate']} vs {comparison['reference']}")
            if comparison.get("skipped_reason"):
                print(f"Skipped: {comparison['skipped_reason']}")
                continue
            print_table(
                [
                    "bits",
                    "qweight tensors",
                    "exact tensors",
                    "logical codes",
                    "code mismatch",
                    "mean |delta|",
                    "max |delta|",
                    "packed-word match",
                    "wall",
                ],
                [
                    [
                        comparison["reference_bits"],
                        comparison["matched_packed_tensor_count"],
                        comparison["exact_packed_tensor_count"],
                        comparison["code_count"],
                        f"{100 * comparison['code_mismatch_rate']:.4f}%",
                        comparison["mean_absolute_code_delta"],
                        comparison["maximum_absolute_code_delta"],
                        f"{100 * comparison['packed_word_match_rate']:.4f}%",
                        f"{comparison['scan_wall_s']:.3f}s",
                    ]
                ],
            )
            print_table(
                ["tensor", "logical codes", "code mismatch", "mean |delta|", "max |delta|", "word match"],
                [
                    [
                        row["tensor"],
                        row["code_count"],
                        f"{100 * row['code_mismatch_rate']:.4f}%",
                        row["mean_absolute_code_delta"],
                        row["maximum_absolute_code_delta"],
                        f"{100 * row['packed_word_match_rate']:.4f}%",
                    ]
                    for row in comparison["top"]
                ],
            )
            print_table(
                ["metadata", "matched", "exact", "shape mismatch", "dtype mismatch"],
                [
                    [
                        suffix,
                        metadata["matched_tensor_count"],
                        metadata["exact_tensor_count"],
                        metadata["shape_mismatch_count"],
                        metadata["dtype_mismatch_count"],
                    ]
                    for suffix, metadata in comparison["metadata"].items()
                ],
            )

    if args.json:
        payload = printable_payload(
            {
                "snapshots": snapshots,
                "loss_comparisons": loss_comparisons,
                "scale_comparisons": scale_comparisons,
                "code_comparisons": code_comparisons,
            }
        )
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
