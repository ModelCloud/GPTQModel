#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from safetensors.torch import load_file
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gptqmodel import GPTQModel
from gptqmodel.adapter.adapter import HF_ADAPTER_CONFIG_FILE_NAME, HF_ADAPTER_FILE_NAME, Lora
from gptqmodel.utils.torch import torch_empty_cache
from scripts.eora_lora_quantized_adapter_sweep import _dir_size_bytes, _evaluate_adapter, convert_adapter


DEFAULT_RANKS = [8, *range(16, 257, 16)]


def _parse_csv_ints(value: str) -> list[int]:
    """Parse comma-separated integer CLI values while preserving user order."""

    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_args() -> argparse.Namespace:
    """Build the CLI for an int4/group128 EoRA rank sweep."""

    parser = argparse.ArgumentParser(
        description="Generate or reuse EoRA adapters across ranks, convert them to int4/group128, and evaluate GSM8K."
    )
    parser.add_argument("--native-model", required=True, type=Path)
    parser.add_argument("--quantized-model", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument(
        "--ranks",
        default=",".join(str(rank) for rank in DEFAULT_RANKS),
        help="Comma-separated EoRA ranks. Default: 8, then 16..256 step 16.",
    )
    parser.add_argument(
        "--dense-search-root",
        action="append",
        default=[],
        type=Path,
        help="Directory containing adapters/rank<N> dense EoRA adapters to reuse. May be repeated.",
    )
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--scale-dtype", default="bfloat16")
    parser.add_argument("--dequant-mode", choices=["load", "forward"], default="load")
    parser.add_argument("--backend", default="marlin")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--calibration-rows", type=int, default=512)
    parser.add_argument("--calibration-batch-size", type=int, default=4)
    parser.add_argument("--calibration-dataset", default="allenai/c4")
    parser.add_argument("--calibration-files", default="en/c4-train.00001-of-01024.json.gz")
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument("--calibration-concat-size", type=int, default=0)
    parser.add_argument("--reference-results-json", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def _adapter_complete(path: Path) -> bool:
    """Return whether a directory has the files needed to load an adapter."""

    return (path / HF_ADAPTER_CONFIG_FILE_NAME).is_file() and (path / HF_ADAPTER_FILE_NAME).is_file()


def _find_dense_adapter(rank: int, search_roots: list[Path]) -> Path | None:
    """Find an existing dense rank adapter under known sweep roots."""

    for root in search_roots:
        candidates = [
            root / "adapters" / f"rank{rank}",
            root / f"rank{rank}",
        ]
        for candidate in candidates:
            if _adapter_complete(candidate):
                return candidate
    return None


def _load_calibration(args: argparse.Namespace) -> list[str]:
    """Load the text calibration rows used for true missing-rank EoRA generation."""

    dataset = load_dataset(
        str(args.calibration_dataset),
        data_files=str(args.calibration_files),
        split=str(args.calibration_split),
    )
    return dataset.select(range(args.calibration_rows))["text"]


def _generate_dense_adapter(
    *,
    rank: int,
    output_dir: Path,
    args: argparse.Namespace,
    calibration_dataset: list[str],
) -> dict[str, Any]:
    """Generate one dense EoRA adapter for ranks not available from previous sweeps."""

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    GPTQModel.adapter.generate(
        adapter=Lora(path=str(output_dir), rank=rank),
        model_id_or_path=str(args.native_model),
        quantized_model_id_or_path=str(args.quantized_model),
        calibration_dataset=calibration_dataset,
        calibration_dataset_concat_size=args.calibration_concat_size,
        batch_size=args.calibration_batch_size,
    )
    seconds = time.perf_counter() - start
    peak_allocated_gb = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3
    torch_empty_cache()
    return {
        "dense_status": "generated",
        "adapter_generate_seconds": seconds,
        "adapter_generate_peak_allocated_gb": peak_allocated_gb,
        "adapter_generate_peak_reserved_gb": peak_reserved_gb,
    }


def _dense_metadata(path: Path) -> dict[str, Any]:
    """Read lightweight metadata for one dense adapter directory."""

    adapter_file = path / HF_ADAPTER_FILE_NAME
    config = json.loads((path / HF_ADAPTER_CONFIG_FILE_NAME).read_text(encoding="utf-8"))
    tensors = load_file(adapter_file)
    param_count = sum(tensor.numel() for tensor in tensors.values())
    return {
        "dense_adapter_dir": str(path),
        "dense_file_mb": adapter_file.stat().st_size / 1024**2,
        "dense_param_count": int(param_count),
        "dense_param_mb": param_count * 2 / 1024**2,
        "dense_tensor_count": len(tensors),
        "dense_dtypes": sorted({str(tensor.dtype) for tensor in tensors.values()}),
        "config_rank": config.get("r"),
    }


def _load_reference(path: Path | None) -> dict[str, Any]:
    """Load qbase and bf16 references from prior full-row rank sweeps."""

    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "qbase": payload.get("qbase") or payload.get("base"),
        "bf16": payload.get("bf16"),
    }


def _summary_value(payload: dict[str, Any] | None, key: str) -> Any:
    """Read a nested summary value without assuming the reference exists."""

    if not payload:
        return None
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return None
    return summary.get(key)


def _completed_ranks(payload: dict[str, Any]) -> set[int]:
    """Return ranks that already have completed rows in a resume payload."""

    return {
        int(case["rank"])
        for case in payload.get("cases", [])
        if case.get("status") == "completed" and (case.get("eval") or payload["context"].get("skip_eval"))
    }


def _table_rows(payload: dict[str, Any]) -> list[list[Any]]:
    """Build fixed-column rows for the markdown and CLI table."""

    qbase = payload.get("reference", {}).get("qbase")
    bf16 = payload.get("reference", {}).get("bf16")
    qbase_acc = _summary_value(qbase, "acc_num")
    qbase_correct = _summary_value(qbase, "correct")
    qbase_rows = _summary_value(qbase, "sample_count")
    bf16_acc = _summary_value(bf16, "acc_num")
    bf16_rows = _summary_value(bf16, "sample_count")
    quantized_model_size = payload["context"].get("quantized_model_size")

    rows = []
    for case in payload.get("cases", []):
        eval_payload = case.get("eval", {})
        summary = eval_payload.get("summary", {})
        acc = summary.get("acc_num")
        correct = summary.get("correct")
        sample_count = summary.get("sample_count")
        qbase_comparable = qbase_rows is None or sample_count == qbase_rows
        bf16_comparable = bf16_rows is None or sample_count == bf16_rows
        file_mb = case.get("file_mb")
        pct_qmodel = None
        if quantized_model_size and file_mb is not None:
            pct_qmodel = file_mb * 1024**2 / quantized_model_size * 100.0
        rows.append(
            [
                case["rank"],
                sample_count or "",
                "" if acc is None else f"{float(acc):.4f}",
                "" if correct is None else correct,
                "" if acc is None or qbase_acc is None or not qbase_comparable else f"{float(acc) - float(qbase_acc):.4f}",
                "" if correct is None or qbase_correct is None or not qbase_comparable else int(correct) - int(qbase_correct),
                "" if acc is None or bf16_acc is None or not bf16_comparable else f"{float(acc) - float(bf16_acc):.4f}",
                "" if eval_payload.get("rows_per_s") is None else f"{float(eval_payload['rows_per_s']):.3f}",
                "" if eval_payload.get("eval_seconds") is None else f"{float(eval_payload['eval_seconds']):.1f}",
                "" if eval_payload.get("peak_allocated_gb") is None else f"{float(eval_payload['peak_allocated_gb']):.2f}",
                "" if file_mb is None else f"{float(file_mb):.1f}",
                "" if pct_qmodel is None else f"{pct_qmodel:.2f}",
                "" if case.get("dense_file_mb") is None else f"{float(case['dense_file_mb']):.1f}",
                "" if case.get("rel_l2") is None else f"{float(case['rel_l2']):.5f}",
                "" if case.get("sqnr_db") is None else f"{float(case['sqnr_db']):.2f}",
                case.get("dense_status", ""),
            ]
        )
    return sorted(rows, key=lambda item: int(item[0]))


def _write_outputs(work_dir: Path, payload: dict[str, Any]) -> None:
    """Persist the sweep payload and render the live comparison table."""

    headers = [
        "rank",
        "rows",
        "acc,num",
        "correct",
        "delta qbase",
        "delta correct",
        "gap bf16",
        "rows/s",
        "eval s",
        "peak GB",
        "int4 MB",
        "% qmodel",
        "dense MB",
        "rel L2",
        "SQNR dB",
        "dense",
    ]
    table = tabulate(_table_rows(payload), headers=headers, tablefmt="github", stralign="right", numalign="right")
    payload["table"] = table
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = [
        "# EoRA int4/group128 rank sweep",
        "",
        f"- native model: `{payload['context']['native_model']}`",
        f"- quantized model: `{payload['context']['quantized_model']}`",
        f"- ranks: `{payload['context']['ranks']}`",
        f"- calibration rows: `{payload['context']['calibration_rows']}`",
        f"- eval batch size: `{payload['context']['eval_batch_size']}`",
        f"- full rows requested: `{payload['context']['max_rows'] is None}`",
        "",
        table,
        "",
    ]
    (work_dir / "results.md").write_text("\n".join(md), encoding="utf-8")
    print(table, flush=True)
    print(f"\nresults: {work_dir / 'results.md'}", flush=True)


def main() -> int:
    """Run a true EoRA rank sweep with grouped int4 adapter compression."""

    args = _parse_args()
    ranks = _parse_csv_ints(args.ranks)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    existing = args.work_dir / "results.json"
    if args.skip_existing and existing.exists():
        payload = json.loads(existing.read_text(encoding="utf-8"))
    else:
        payload = {
            "context": {
                "native_model": str(args.native_model),
                "quantized_model": str(args.quantized_model),
                "ranks": ranks,
                "bits": args.bits,
                "group_size": args.group_size,
                "backend": args.backend,
                "scale_dtype": args.scale_dtype,
                "dequant_mode": args.dequant_mode,
                "eval_batch_size": args.eval_batch_size,
                "max_new_tokens": args.max_new_tokens,
                "max_rows": args.max_rows,
                "calibration_rows": args.calibration_rows,
                "calibration_batch_size": args.calibration_batch_size,
                "calibration_dataset": args.calibration_dataset,
                "calibration_files": args.calibration_files,
                "calibration_concat_size": args.calibration_concat_size,
                "skip_eval": args.skip_eval,
                "quantized_model_size": _dir_size_bytes(args.quantized_model),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "reference": _load_reference(args.reference_results_json),
            "cases": [],
        }

    completed = _completed_ranks(payload)
    calibration_dataset: list[str] | None = None

    for rank in ranks:
        if args.skip_existing and rank in completed:
            print(f"Skipping completed rank {rank}", flush=True)
            continue

        dense_dir = _find_dense_adapter(rank, args.dense_search_root)
        dense_status = "reused"
        generation_payload: dict[str, Any] = {
            "dense_status": dense_status,
            "adapter_generate_seconds": None,
            "adapter_generate_peak_allocated_gb": None,
            "adapter_generate_peak_reserved_gb": None,
        }
        if dense_dir is None:
            if calibration_dataset is None:
                calibration_dataset = _load_calibration(args)
            dense_dir = args.work_dir / "dense" / f"rank{rank}"
            if _adapter_complete(dense_dir):
                generation_payload["dense_status"] = "generated_existing"
            else:
                print(f"Generating dense rank {rank}: {dense_dir}", flush=True)
                generation_payload = _generate_dense_adapter(
                    rank=rank,
                    output_dir=dense_dir,
                    args=args,
                    calibration_dataset=calibration_dataset,
                )

        print(f"Converting rank {rank} to int{args.bits}/group{args.group_size}", flush=True)
        int4_dir = args.work_dir / "adapters" / f"rank{rank}_int{args.bits}_g{args.group_size}"
        case = convert_adapter(
            source_adapter=dense_dir,
            output_dir=int4_dir,
            bits=args.bits,
            group_size=args.group_size,
            scale_dtype=getattr(torch, args.scale_dtype),
            dequant_mode=args.dequant_mode,
        )
        case.update(
            {
                "rank": rank,
                "status": "converted",
                "int4_adapter_dir": str(int4_dir),
                **generation_payload,
                **_dense_metadata(dense_dir),
            }
        )

        if not args.skip_eval:
            print(f"Evaluating rank {rank}: {int4_dir}", flush=True)
            case["eval"] = _evaluate_adapter(
                quantized_model=args.quantized_model,
                adapter_dir=int4_dir,
                rank=rank,
                backend=args.backend,
                eval_batch_size=args.eval_batch_size,
                max_new_tokens=args.max_new_tokens,
                max_rows=args.max_rows,
            )
        case["status"] = "completed"
        payload["cases"].append(case)
        _write_outputs(args.work_dir, payload)

    _write_outputs(args.work_dir, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
