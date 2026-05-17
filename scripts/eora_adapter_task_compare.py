#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.adapter.adapter import HF_ADAPTER_CONFIG_FILE_NAME, HF_ADAPTER_FILE_NAME, Lora
from gptqmodel.adapter.peft import LoraConfig
from gptqmodel.utils.torch import torch_empty_cache
from tests.eval import evaluate, get_eval_task_metrics, resolve_eval_metric_alias


@dataclass(frozen=True)
class EvalTask:
    """Task settings needed to run one Evalution suite consistently across adapter variants."""

    name: str
    metric: str
    apply_chat_template: bool
    max_rows: int | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare native and grouped-low-bit EoRA/LoRA adapters on GSM8K Platinum and MMLU STEM."
    )
    parser.add_argument("--quantized-model", required=True, type=Path)
    parser.add_argument("--native-adapter", required=True, type=Path)
    parser.add_argument("--int8-g96-adapter", required=True, type=Path)
    parser.add_argument("--int4-g128-adapter", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--backend", default="marlin")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gsm8k-max-rows", type=int, default=None)
    parser.add_argument("--mmlu-max-rows", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _dir_size_mb(path: Path) -> float:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / 1024**2


def _adapter_file_mb(path: Path) -> float | None:
    adapter_file = path / HF_ADAPTER_FILE_NAME
    if not adapter_file.exists():
        return None
    return adapter_file.stat().st_size / 1024**2


def _adapter_metadata(path: Path) -> dict[str, Any]:
    config_path = path / HF_ADAPTER_CONFIG_FILE_NAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "file_mb": _adapter_file_mb(path),
        "format": config.get("gptqmodel_lora_weight_format", "dense"),
        "bits": config.get("gptqmodel_lora_weight_bits"),
        "group_size": config.get("gptqmodel_lora_group_size"),
        "rank": config.get("r"),
    }


def _test_sample_count(test: dict[str, Any]) -> int:
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


def _result_sample_count(result: dict[str, Any], task_name: str) -> int:
    candidates = {task_name}
    for test in result.get("tests", []):
        if test.get("name") in candidates:
            return _test_sample_count(test)
    return 0


def _metric_value(metrics: dict[str, float], requested_metric: str) -> tuple[str, float]:
    metric_name = requested_metric
    if metric_name not in metrics:
        alias = resolve_eval_metric_alias(metric_name, metrics)
        if alias:
            metric_name = alias
    if metric_name not in metrics:
        raise KeyError(f"Metric `{requested_metric}` not found in metrics: {sorted(metrics)}")
    return metric_name, float(metrics[metric_name])


def _task_summary(result: dict[str, Any], task: EvalTask) -> dict[str, Any]:
    metrics = get_eval_task_metrics(result, task.name)
    metric_name, score = _metric_value(metrics, task.metric)
    rows = _result_sample_count(result, task.name)
    return {
        "task": task.name,
        "metric": metric_name,
        "score": score,
        "sample_count": rows,
        "correct": int(round(score * rows)) if rows else None,
        "metrics": metrics,
    }


def _load_rank(adapter_path: Path, explicit_rank: int | None) -> int:
    if explicit_rank is not None:
        return explicit_rank
    config = LoraConfig.from_pretrained(str(adapter_path), HF_ADAPTER_CONFIG_FILE_NAME)
    return int(config.r)


def _evaluate_task(
    *,
    model: GPTQModel,
    task: EvalTask,
    batch_size: int,
    max_new_tokens: int,
    backend: BACKEND | str,
) -> dict[str, Any]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    suite_kwargs: dict[str, Any] = {}
    if task.max_rows is not None:
        suite_kwargs["max_rows"] = task.max_rows
    result = evaluate(
        model_or_id_or_path=model,
        tasks=[task.name],
        batch_size=batch_size,
        backend=backend,
        apply_chat_template=task.apply_chat_template,
        gen_kwargs={
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "temperature": 0.0,
        },
        suite_kwargs=suite_kwargs,
    )
    eval_seconds = time.perf_counter() - start
    summary = _task_summary(result, task)
    rows = int(summary.get("sample_count") or 0)
    return {
        "summary": summary,
        "eval_seconds": eval_seconds,
        "rows_per_s": rows / eval_seconds if rows else None,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
    }


def _table_rows(payload: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for row in payload["results"]:
        summary = row["summary"]
        adapter = payload["adapters"][row["variant"]]
        rows.append(
            [
                row["variant"],
                summary["task"],
                summary["sample_count"] or "",
                summary["metric"],
                f"{summary['score']:.4f}",
                summary["correct"] if summary["correct"] is not None else "",
                f"{row['rows_per_s']:.3f}" if row.get("rows_per_s") is not None else "",
                f"{row['eval_seconds']:.1f}",
                f"{row['load_seconds']:.1f}",
                f"{row['peak_allocated_gb']:.2f}",
                f"{adapter['file_mb']:.1f}" if adapter.get("file_mb") is not None else "",
                adapter.get("format", ""),
                adapter.get("bits") or "",
                adapter.get("group_size") or "",
            ]
        )
    return rows


def _write_outputs(work_dir: Path, payload: dict[str, Any]) -> None:
    headers = [
        "variant",
        "task",
        "rows",
        "metric",
        "score",
        "correct",
        "rows/s",
        "eval s",
        "load s",
        "peak GB",
        "adapter MB",
        "format",
        "bits",
        "group",
    ]
    table = tabulate(_table_rows(payload), headers=headers, tablefmt="github", stralign="right", numalign="right")
    payload["table"] = table
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = [
        "# EoRA Adapter Task Comparison",
        "",
        f"- quantized model: `{payload['context']['quantized_model']}`",
        f"- backend: `{payload['context']['backend']}`",
        f"- eval batch size: `{payload['context']['eval_batch_size']}`",
        "",
        table,
        "",
    ]
    (work_dir / "results.md").write_text("\n".join(md), encoding="utf-8")
    print(table, flush=True)
    print(f"\nresults: {work_dir / 'results.md'}", flush=True)


def main() -> int:
    args = _parse_args()
    backend = getattr(BACKEND, str(args.backend).upper(), args.backend)
    variants = [
        ("native", args.native_adapter),
        ("int8_g96", args.int8_g96_adapter),
        ("int4_g128", args.int4_g128_adapter),
    ]
    tasks = [
        EvalTask("gsm8k_platinum_cot", "acc,num", True, args.gsm8k_max_rows),
        EvalTask("mmlu_stem", "acc", False, args.mmlu_max_rows),
    ]
    payload: dict[str, Any] = {
        "context": {
            "quantized_model": str(args.quantized_model),
            "quantized_model_mb": _dir_size_mb(args.quantized_model),
            "backend": str(args.backend),
            "eval_batch_size": args.eval_batch_size,
            "max_new_tokens": args.max_new_tokens,
            "gsm8k_max_rows": args.gsm8k_max_rows,
            "mmlu_max_rows": args.mmlu_max_rows,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "adapters": {label: _adapter_metadata(path) for label, path in variants},
        "results": [],
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)

    existing = args.work_dir / "results.json"
    if args.skip_existing and existing.exists():
        payload = json.loads(existing.read_text(encoding="utf-8"))

    completed = {(row["variant"], row["summary"]["task"]) for row in payload.get("results", [])}
    for label, adapter_path in variants:
        rank = _load_rank(adapter_path, args.rank)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        load_start = time.perf_counter()
        model = GPTQModel.load(
            model_id_or_path=str(args.quantized_model),
            backend=backend,
            adapter=Lora(path=str(adapter_path), rank=rank),
        )
        load_seconds = time.perf_counter() - load_start
        for task in tasks:
            if args.skip_existing and (label, task.name) in completed:
                print(f"Skipping existing {label} {task.name}", flush=True)
                continue
            print(f"Evaluating {label} on {task.name}", flush=True)
            task_payload = _evaluate_task(
                model=model,
                task=task,
                batch_size=args.eval_batch_size,
                max_new_tokens=args.max_new_tokens,
                backend=backend,
            )
            task_payload.update(
                {
                    "variant": label,
                    "adapter_path": str(adapter_path),
                    "rank": rank,
                    "load_seconds": load_seconds,
                }
            )
            payload["results"].append(task_payload)
            _write_outputs(args.work_dir, payload)
        del model
        torch_empty_cache()
    _write_outputs(args.work_dir, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
