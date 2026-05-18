#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from tabulate import tabulate

# Let the script import the local checkout when executed directly from scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Keep adapter constants local so --gpu can set CUDA visibility before Torch/GPTQModel imports.
DEFAULT_DEFAULT_RANK = 256
HF_ADAPTER_FILE_NAME = "adapter_model.safetensors"
HF_ADAPTER_CONFIG_FILE_NAME = "adapter_config.json"


def adapter_complete(path: Path) -> bool:
    """Return whether an adapter directory has the expected config and weights."""

    return (path / HF_ADAPTER_CONFIG_FILE_NAME).is_file() and (path / HF_ADAPTER_FILE_NAME).is_file()


def adapter_file_mb(path: Path) -> float:
    """Return the adapter safetensors file size in MiB."""

    return (path / HF_ADAPTER_FILE_NAME).stat().st_size / 1024**2


def compressed_tensor_keys(target_key: str, side: str) -> tuple[str, str, str]:
    """Build grouped-LoRA tensor keys for a target and LoRA side."""

    base = f"{target_key}.{side}.weight"
    return (f"{base}.qweight", f"{base}.scales", f"{base}.shape")


def parse_args() -> argparse.Namespace:
    """Build the CLI for a single combined mixed-rank EoRA adapter evaluation."""

    parser = argparse.ArgumentParser(
        description=(
            "Build one EoRA/LoRA adapter from a conservative per-module rank map and run a full "
            "GSM8K Platinum evaluation."
        )
    )
    parser.add_argument("--quantized-model", required=True, type=Path)
    parser.add_argument("--rank-map", required=True, type=Path)
    parser.add_argument("--rank-bank-json", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--adapter-name", default="conservative_mixed")
    parser.add_argument("--default-rank", type=int, default=DEFAULT_DEFAULT_RANK)
    parser.add_argument("--backend", default="marlin")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def load_rank_bank(path: Path) -> dict[int, Path]:
    """Load rank-to-adapter directories from a sweep rank bank JSON file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    rank_bank = {int(rank): Path(adapter_path) for rank, adapter_path in payload.items()}
    missing = [rank for rank, adapter_path in rank_bank.items() if not adapter_complete(adapter_path)]
    if missing:
        raise FileNotFoundError(f"Incomplete rank adapters in bank for ranks: {missing}")
    return rank_bank


def load_rank_map(path: Path) -> tuple[dict[str, int], dict[str, Any] | None]:
    """Load a conservative dynamic-rank map and its optional baseline metadata."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    rank_map = {str(key): int(value) for key, value in payload["rank_map"].items()}
    return rank_map, payload.get("baseline")


def runtime_to_weight_key(default_weights: dict[str, Any], runtime_key: str) -> str:
    """Resolve a runtime module key to the adapter safetensors key prefix."""

    suffix = ".lora_A.weight.qweight"
    candidates = (
        f"base_model.model.{runtime_key}",
        f"base_model.{runtime_key}",
        runtime_key,
    )
    for candidate in candidates:
        if f"{candidate}{suffix}" in default_weights:
            return candidate
    raise KeyError(f"Could not resolve rank-map target in default adapter: {runtime_key}")


def build_combined_adapter(
    *,
    rank_bank: dict[int, Path],
    rank_map: dict[str, int],
    default_rank: int,
    output_dir: Path,
    skip_existing: bool,
) -> dict[str, Any]:
    """Save one adapter whose tensors and rank_pattern follow the full rank map."""

    from safetensors.torch import load_file, save_file

    if skip_existing and adapter_complete(output_dir):
        return {
            "adapter_dir": str(output_dir),
            "adapter_file_mb": adapter_file_mb(output_dir),
            "adapter_status": "reused",
            "rank_pattern": json.loads((output_dir / HF_ADAPTER_CONFIG_FILE_NAME).read_text()).get("rank_pattern", {}),
        }

    if default_rank not in rank_bank:
        raise KeyError(f"Rank bank does not include default rank {default_rank}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rank_weights = {rank: load_file(path / HF_ADAPTER_FILE_NAME) for rank, path in rank_bank.items()}
    default_weights = rank_weights[default_rank]
    combined = {key: tensor for key, tensor in default_weights.items()}
    rank_pattern: dict[str, int] = {}
    replaced: list[str] = []

    for runtime_key, rank in sorted(rank_map.items()):
        if rank == default_rank:
            continue
        if rank not in rank_weights:
            raise KeyError(f"Rank map references rank {rank}, but rank bank only has {sorted(rank_weights)}")
        weight_key = runtime_to_weight_key(default_weights, runtime_key)
        for side in ("lora_A", "lora_B"):
            for tensor_key in compressed_tensor_keys(weight_key, side):
                if tensor_key not in rank_weights[rank]:
                    raise KeyError(f"Missing {tensor_key} in rank {rank} adapter")
                combined[tensor_key] = rank_weights[rank][tensor_key]
                replaced.append(tensor_key)
        rank_pattern[runtime_key] = rank

    config_payload = json.loads((rank_bank[default_rank] / HF_ADAPTER_CONFIG_FILE_NAME).read_text(encoding="utf-8"))
    config_payload["r"] = default_rank
    config_payload["lora_alpha"] = default_rank
    config_payload["rank_pattern"] = rank_pattern
    (output_dir / HF_ADAPTER_CONFIG_FILE_NAME).write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    save_file(combined, output_dir / HF_ADAPTER_FILE_NAME, metadata={"format": "pt"})
    return {
        "adapter_dir": str(output_dir),
        "adapter_file_mb": adapter_file_mb(output_dir),
        "adapter_status": "built",
        "rank_pattern": rank_pattern,
        "replaced_tensor_count": len(replaced),
    }


def table_rows(payload: dict[str, Any]) -> list[list[Any]]:
    """Create stable fixed-column result rows for markdown and console output."""

    baseline = payload.get("baseline") or {}
    summary = ((payload.get("eval") or {}).get("summary") or {})
    correct = summary.get("correct")
    score = summary.get("acc_num") or summary.get("score")
    baseline_correct = baseline.get("correct")
    baseline_score = baseline.get("score")
    delta_correct = ""
    delta_score = ""
    if correct is not None and baseline_correct is not None:
        delta_correct = int(correct) - int(baseline_correct)
    if score is not None and baseline_score is not None:
        delta_score = float(score) - float(baseline_score)
    return [
        [
            payload["adapter_name"],
            summary.get("sample_count", ""),
            "" if score is None else f"{float(score):.6f}",
            correct if correct is not None else "",
            "" if delta_score == "" else f"{delta_score:+.6f}",
            delta_correct,
            "" if (payload.get("eval") or {}).get("rows_per_s") is None else f"{float(payload['eval']['rows_per_s']):.3f}",
            "" if (payload.get("eval") or {}).get("eval_seconds") is None else f"{float(payload['eval']['eval_seconds']):.1f}",
            "" if (payload.get("eval") or {}).get("peak_allocated_gb") is None else f"{float(payload['eval']['peak_allocated_gb']):.2f}",
            f"{float(payload['adapter']['adapter_file_mb']):.1f}",
            payload["non_default_target_count"],
        ]
    ]


def write_outputs(work_dir: Path, payload: dict[str, Any]) -> str:
    """Write JSON and markdown outputs and return the rendered summary table."""

    headers = [
        "adapter",
        "rows",
        "acc,num",
        "correct",
        "delta acc",
        "delta correct",
        "rows/s",
        "eval s",
        "peak GB",
        "adapter MB",
        "non256 targets",
    ]
    table = tabulate(table_rows(payload), headers=headers, tablefmt="github", stralign="right", numalign="right")
    payload["table"] = table
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = [
        "# Combined EoRA mixed-rank evaluation",
        "",
        f"- quantized model: `{payload['context']['quantized_model']}`",
        f"- adapter: `{payload['adapter']['adapter_dir']}`",
        f"- rank map: `{payload['context']['rank_map']}`",
        f"- selected-rank counts: `{payload['selected_rank_counts']}`",
        "",
        table,
        "",
    ]
    (work_dir / "results.md").write_text("\n".join(md), encoding="utf-8")
    return table


def main() -> int:
    """Build and evaluate one combined mixed-rank EoRA adapter."""

    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    rank_bank = load_rank_bank(args.rank_bank_json)
    rank_map, baseline = load_rank_map(args.rank_map)
    output_dir = args.work_dir / "adapters" / args.adapter_name
    start = time.perf_counter()
    adapter_payload = build_combined_adapter(
        rank_bank=rank_bank,
        rank_map=rank_map,
        default_rank=args.default_rank,
        output_dir=output_dir,
        skip_existing=args.skip_existing,
    )
    build_seconds = time.perf_counter() - start
    counts = dict(sorted(Counter(rank_map.values()).items()))

    payload: dict[str, Any] = {
        "adapter_name": args.adapter_name,
        "context": {
            "quantized_model": str(args.quantized_model),
            "rank_map": str(args.rank_map),
            "rank_bank_json": str(args.rank_bank_json),
            "default_rank": args.default_rank,
            "backend": args.backend,
            "eval_batch_size": args.eval_batch_size,
            "max_new_tokens": args.max_new_tokens,
            "max_rows": args.max_rows,
            "gpu": args.gpu,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "baseline": baseline,
        "selected_rank_counts": counts,
        "non_default_target_count": sum(1 for rank in rank_map.values() if rank != args.default_rank),
        "adapter": adapter_payload,
        "build_seconds": build_seconds,
    }

    if not args.skip_eval:
        from scripts.eora_lora_quantized_adapter_sweep import _evaluate_adapter

        print(f"Evaluating combined adapter: {output_dir}", flush=True)
        payload["eval"] = _evaluate_adapter(
            quantized_model=args.quantized_model,
            adapter_dir=output_dir,
            rank=args.default_rank,
            backend=args.backend,
            eval_batch_size=args.eval_batch_size,
            max_new_tokens=args.max_new_tokens,
            max_rows=args.max_rows,
        )

    table = write_outputs(args.work_dir, payload)
    print(table)
    print(f"\nresults: {args.work_dir / 'results.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
