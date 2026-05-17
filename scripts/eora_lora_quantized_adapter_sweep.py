#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.adapter.adapter import HF_ADAPTER_CONFIG_FILE_NAME, HF_ADAPTER_FILE_NAME, Lora
from gptqmodel.adapter.peft import LoraConfig
from gptqmodel.adapter.quant import (
    compressed_weight_keys,
    dequantize_tensor_groupwise_int,
    dtype_from_name,
    lora_grouped_format_from_bits,
    quantize_tensor_groupwise_int,
)
from gptqmodel.utils.torch import torch_empty_cache
from tests.eval import evaluate, format_eval_result_table, get_eval_task_metrics


def parse_csv_ints(value: str) -> list[int]:
    """Parses comma-separated integer CLI values."""

    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    """Builds the CLI for compressed EoRA/LoRA adapter sweeps."""

    parser = argparse.ArgumentParser(
        description="Convert one BF16 EoRA/LoRA adapter to grouped int4/int6/int8 variants and evaluate GSM8K Platinum."
    )
    parser.add_argument("--source-adapter", required=True, type=Path)
    parser.add_argument("--quantized-model", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--bits", default="4,6", help="Comma-separated LoRA bit widths.")
    parser.add_argument("--groups", default="32,64,96,128", help="Comma-separated LoRA group sizes.")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--backend", default="marlin")
    parser.add_argument("--scale-dtype", default="bfloat16")
    parser.add_argument("--dequant-mode", choices=["load", "forward"], default="load")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--baseline-json", type=Path, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def _dir_size_bytes(path: Path) -> int:
    """Returns recursive file size for one directory."""

    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _metric_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Extracts the GSM8K Platinum accuracy summary used by the sweep table."""

    metrics = get_eval_task_metrics(result, "gsm8k_platinum_cot")
    acc = float(metrics["acc,num"])
    rows = 0
    for test in result.get("tests", []):
        if test.get("name") == "gsm8k_platinum_cot":
            rows = _test_sample_count(test)
            break
    correct = int(round(acc * rows)) if rows else None
    return {
        "task": "gsm8k_platinum_cot",
        "sample_count": rows,
        "acc_num": acc,
        "correct": correct,
        "metrics": metrics,
    }


def _test_sample_count(test: dict[str, Any]) -> int:
    """Returns evaluated sample count from Evalution test payload variants."""

    for key in ("sample_count", "num_samples", "total_samples"):
        value = test.get(key)
        if value:
            return int(value)

    samples = test.get("samples")
    if isinstance(samples, list):
        return len(samples)

    metadata = test.get("metadata")
    if isinstance(metadata, dict):
        for key in ("sample_count", "num_samples", "total_samples", "max_rows"):
            value = metadata.get(key)
            if value:
                return int(value)

    return 0


def _evaluate_adapter(
    *,
    quantized_model: Path,
    adapter_dir: Path,
    rank: int,
    backend: str,
    eval_batch_size: int,
    max_new_tokens: int,
    max_rows: int | None,
) -> dict[str, Any]:
    """Loads one quantized model plus adapter and runs GSM8K Platinum."""

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before_allocated = torch.cuda.memory_allocated()
    before_reserved = torch.cuda.memory_reserved()
    load_start = time.perf_counter()
    model = GPTQModel.load(
        model_id_or_path=str(quantized_model),
        backend=getattr(BACKEND, str(backend).upper(), backend),
        adapter=Lora(path=str(adapter_dir), rank=rank),
    )
    load_seconds = time.perf_counter() - load_start

    eval_start = time.perf_counter()
    result = evaluate(
        model_or_id_or_path=model,
        tasks=["gsm8k_platinum_cot"],
        batch_size=eval_batch_size,
        apply_chat_template=True,
        gen_kwargs={
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "temperature": 0.0,
        },
        suite_kwargs={"max_rows": max_rows} if max_rows is not None else {},
    )
    eval_seconds = time.perf_counter() - eval_start
    summary = _metric_summary(result)
    rows = int(summary.get("sample_count") or 0)
    payload = {
        "load_seconds": load_seconds,
        "eval_seconds": eval_seconds,
        "rows_per_s": rows / eval_seconds if rows else None,
        "before_allocated_gb": before_allocated / 1024**3,
        "before_reserved_gb": before_reserved / 1024**3,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
        "after_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "after_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        "summary": summary,
        "table": format_eval_result_table(result),
    }
    del model
    torch_empty_cache()
    return payload


def convert_adapter(
    *,
    source_adapter: Path,
    output_dir: Path,
    bits: int,
    group_size: int,
    scale_dtype: torch.dtype,
    dequant_mode: str,
) -> dict[str, Any]:
    """Converts dense LoRA safetensors into a grouped low-bit adapter directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((source_adapter / HF_ADAPTER_CONFIG_FILE_NAME).read_text(encoding="utf-8"))
    source_weights = load_file(source_adapter / HF_ADAPTER_FILE_NAME)
    weights: dict[str, torch.Tensor] = {}
    total_numel = 0
    total_scales = 0
    sq_error = 0.0
    sq_signal = 0.0
    abs_weight_sum = 0.0
    abs_error_sum = 0.0
    worst: list[dict[str, Any]] = []

    start = time.perf_counter()
    for key, tensor in source_weights.items():
        total_numel += tensor.numel()
        q_key, scales_key, shape_key = compressed_weight_keys(key)
        qweight, scales, shape = quantize_tensor_groupwise_int(
            tensor,
            bits=bits,
            group_size=group_size,
            scale_dtype=scale_dtype,
        )
        weights[q_key] = qweight
        weights[scales_key] = scales
        weights[shape_key] = shape
        total_scales += scales.numel()

        dense = dequantize_tensor_groupwise_int(
            qweight=qweight,
            scales=scales,
            shape=shape,
            bits=bits,
            group_size=group_size,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        source = tensor.float()
        error = dense - source
        tensor_sq_error = float(torch.sum(error * error).item())
        tensor_sq_signal = float(torch.sum(source * source).item())
        sq_error += tensor_sq_error
        sq_signal += tensor_sq_signal
        abs_weight_sum += float(torch.sum(source.abs()).item())
        abs_error_sum += float(torch.sum(error.abs()).item())
        worst.append(
            {
                "key": key,
                "rel_l2": math.sqrt(tensor_sq_error / tensor_sq_signal) if tensor_sq_signal > 0 else 0.0,
                "numel": int(tensor.numel()),
            }
        )

    config.update(
        {
            "gptqmodel_lora_weight_format": lora_grouped_format_from_bits(bits),
            "gptqmodel_lora_weight_bits": bits,
            "gptqmodel_lora_group_size": group_size,
            "gptqmodel_lora_scale_dtype": str(scale_dtype).removeprefix("torch."),
            "gptqmodel_lora_dequant_mode": dequant_mode,
        }
    )
    LoraConfig(**{k: v for k, v in config.items() if k in LoraConfig.__dataclass_fields__}).save_pretrained(
        str(output_dir)
    )
    save_file(weights, output_dir / HF_ADAPTER_FILE_NAME, metadata={"format": "pt"})
    convert_seconds = time.perf_counter() - start
    file_mb = (output_dir / HF_ADAPTER_FILE_NAME).stat().st_size / 1024**2
    rel_l2 = math.sqrt(sq_error / sq_signal) if sq_signal > 0 else 0.0
    return {
        "bits": bits,
        "group_size": group_size,
        "adapter_dir": str(output_dir),
        "params": int(total_numel),
        "scale_count": int(total_scales),
        "file_mb": file_mb,
        "rel_l2": rel_l2,
        "sqnr_db": 20.0 * math.log10(1.0 / rel_l2) if rel_l2 > 0 else float("inf"),
        "mean_abs_weight": abs_weight_sum / total_numel,
        "mean_abs_error": abs_error_sum / total_numel,
        "worst": sorted(worst, key=lambda item: item["rel_l2"], reverse=True)[:10],
        "convert_seconds": convert_seconds,
    }


def _baseline_rows(path: Path | None) -> list[dict[str, Any]]:
    """Loads previous int8 result rows when provided for side-by-side comparison."""

    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for case in payload.get("cases", []):
        eval_payload = case.get("eval", {})
        summary = eval_payload.get("summary", {})
        rows.append(
            {
                "bits": 8,
                "group_size": case.get("group_size"),
                "summary": summary,
                "eval": eval_payload,
                "file_mb": case.get("int8_file_mb"),
                "rel_l2": case.get("rel_l2"),
                "sqnr_db": case.get("sqnr_db"),
            }
        )
    return rows


def _table_rows(cases: list[dict[str, Any]], quantized_model_size: int | None) -> list[list[Any]]:
    """Builds fixed-column rows for CLI and markdown output."""

    rows = []
    for case in cases:
        eval_payload = case.get("eval", {})
        summary = case.get("summary") or eval_payload.get("summary", {})
        sample_rows = int(summary.get("sample_count") or 0)
        rows_per_s = eval_payload.get("rows_per_s")
        if rows_per_s is None and sample_rows and eval_payload.get("eval_seconds"):
            rows_per_s = sample_rows / float(eval_payload["eval_seconds"])
        file_mb = case.get("file_mb")
        pct_qmodel = ""
        if quantized_model_size and file_mb is not None:
            pct_qmodel = (float(file_mb) * 1024**2) / quantized_model_size * 100.0
        rows.append(
            [
                int(case["bits"]),
                int(case["group_size"]),
                sample_rows or "",
                "" if summary.get("acc_num") is None else f"{float(summary['acc_num']):.4f}",
                summary.get("correct", ""),
                "" if rows_per_s is None else f"{float(rows_per_s):.3f}",
                "" if eval_payload.get("peak_allocated_gb") is None else f"{float(eval_payload['peak_allocated_gb']):.2f}",
                "" if file_mb is None else f"{float(file_mb):.1f}",
                "" if pct_qmodel == "" else f"{pct_qmodel:.2f}",
                "" if case.get("rel_l2") is None else f"{float(case['rel_l2']):.5f}",
                "" if case.get("sqnr_db") is None else f"{float(case['sqnr_db']):.2f}",
            ]
        )
    return sorted(rows, key=lambda item: (item[0], item[1]))


def write_outputs(work_dir: Path, payload: dict[str, Any]) -> None:
    """Writes JSON and markdown sweep summaries."""

    headers = [
        "bits",
        "group",
        "rows",
        "acc,num",
        "correct",
        "rows/s",
        "peak GB",
        "file MB",
        "% qmodel",
        "rel L2",
        "SQNR dB",
    ]
    rows = _table_rows(payload["cases_with_baseline"], payload["context"].get("quantized_model_size"))
    table = tabulate(rows, headers=headers, tablefmt="github", stralign="right", numalign="right")
    payload["table"] = table
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = [
        "# EoRA/LoRA grouped low-bit adapter sweep",
        "",
        f"- source adapter: `{payload['context']['source_adapter_path']}`",
        f"- quantized model: `{payload['context']['quantized_model_path']}`",
        f"- dequant mode: `{payload['context']['dequant_mode']}`",
        "",
        table,
        "",
    ]
    (work_dir / "results.md").write_text("\n".join(md), encoding="utf-8")
    print(table)
    print(f"\nresults: {work_dir / 'results.md'}")


def main() -> int:
    """Runs the conversion/evaluation sweep and writes result tables."""

    args = parse_args()
    bits_list = parse_csv_ints(args.bits)
    groups = parse_csv_ints(args.groups)
    scale_dtype = dtype_from_name(args.scale_dtype)
    source_config = LoraConfig.from_pretrained(str(args.source_adapter), HF_ADAPTER_CONFIG_FILE_NAME)
    rank = int(args.rank or source_config.r)
    quantized_model_size = _dir_size_bytes(args.quantized_model) if args.quantized_model.exists() else None
    args.work_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    for bits in bits_list:
        for group_size in groups:
            adapter_dir = args.work_dir / "adapters" / f"int{bits}_g{group_size}"
            case = convert_adapter(
                source_adapter=args.source_adapter,
                output_dir=adapter_dir,
                bits=bits,
                group_size=group_size,
                scale_dtype=scale_dtype,
                dequant_mode=args.dequant_mode,
            )
            if not args.skip_eval:
                print(f"Evaluating int{bits} group {group_size}: {adapter_dir}", flush=True)
                case["eval"] = _evaluate_adapter(
                    quantized_model=args.quantized_model,
                    adapter_dir=adapter_dir,
                    rank=rank,
                    backend=args.backend,
                    eval_batch_size=args.eval_batch_size,
                    max_new_tokens=args.max_new_tokens,
                    max_rows=args.max_rows,
                )
            cases.append(case)
            payload = {
                "context": {
                    "source_adapter_path": str(args.source_adapter),
                    "quantized_model_path": str(args.quantized_model),
                    "bits": bits_list,
                    "groups": groups,
                    "rank": rank,
                    "backend": args.backend,
                    "eval_batch_size": args.eval_batch_size,
                    "max_new_tokens": args.max_new_tokens,
                    "max_rows": args.max_rows,
                    "dequant_mode": args.dequant_mode,
                    "quantized_model_size": quantized_model_size,
                    "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                },
                "baseline_cases": _baseline_rows(args.baseline_json),
                "cases": cases,
                "cases_with_baseline": _baseline_rows(args.baseline_json) + cases,
            }
            write_outputs(args.work_dir, payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
