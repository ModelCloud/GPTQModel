#!/usr/bin/env python
# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 gptqmodel contributors
# SPDX-License-Identifier: Apache-2.0
"""CLI helper to run Evalution-backed tasks against a GPT-QModel checkpoint."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import gptqmodel
from tabulate import tabulate
from gptqmodel import GPTQModel
from gptqmodel.models.base import BaseQModel
from tests.eval import (
    evaluate,
    get_eval_task_results,
    list_supported_tasks,
    normalize_eval_task_name,
)


if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault(
    "PYTORCH_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7",
)

DEFAULT_RESULTS_PATH = Path("evalution_results.json")
DEFAULT_TASKS = ("arc_challenge",)


def _available_backends() -> Dict[str, gptqmodel.BACKEND]:
    return {member.name.lower(): member for member in gptqmodel.BACKEND}


def _parse_backend(value: str) -> gptqmodel.BACKEND:
    lookup = _available_backends()
    key = value.strip().lower()
    if key not in lookup:
        expected = ", ".join(sorted(lookup.keys()))
        raise argparse.ArgumentTypeError(f"Unknown backend '{value}'. Expected one of: {expected}")
    return lookup[key]


def _parse_batch_size(value: str) -> str | int:
    normalized = value.strip()
    if normalized.lower() == "auto":
        return "auto"
    try:
        return int(normalized, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Batch size must be 'auto' or an integer") from exc


def _coerce_value(text: str):
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in text:
            return float(text)
        return int(text, 10)
    except ValueError:
        return text


def _parse_key_value_pairs(pairs: Iterable[str]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for item in pairs:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Argument '{item}' must be in key=value format")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise argparse.ArgumentTypeError(f"Argument '{item}' is missing a key")
        value = _coerce_value(raw_value.strip())
        result[key] = value
    return result


def _resolve_task(name: str) -> str:
    normalized = normalize_eval_task_name(name)
    if normalized in list_supported_tasks():
        return normalized
    available = ", ".join(list_supported_tasks())
    raise argparse.ArgumentTypeError(f"Unknown Evalution task '{name}'. Expected one of: {available}")


def _list_tasks() -> None:
    rows = [(task_name, task_name) for task_name in list_supported_tasks()]
    print(tabulate(rows, headers=["Name", "Identifier"]))


def _extract_metrics(results: Dict) -> Dict[str, Dict[str, float]]:
    aggregated = get_eval_task_results(results)
    return {
        task_name: {
            metric: value
            for metric, value in metrics.items()
            if metric != "alias" and "stderr" not in metric
        }
        for task_name, metrics in aggregated.items()
    }


def _print_metrics_table(metrics: Dict[str, Dict[str, float]], table_format: str) -> None:
    rows: List[Tuple[str, str, object]] = []
    for task_name in sorted(metrics):
        for metric_name in sorted(metrics[task_name]):
            rows.append((task_name, metric_name, metrics[task_name][metric_name]))
    if not rows:
        print("No metrics to display.")
        return
    print(
        tabulate(
            rows,
            headers=["Task", "Metric", "Value"],
            tablefmt=table_format,
            floatfmt=".4f",
        )
    )


def _split_tasks(arg_value: str | None) -> List[str]:
    if not arg_value:
        return []
    return [item.strip() for item in arg_value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Evalution tasks against a quantized model loaded via gptqmodel."
    )
    parser.add_argument("--model", help="Model path or Hugging Face repo id.")
    parser.add_argument(
        "--backend",
        default="auto",
        type=_parse_backend,
        help="Inference backend to use when loading with gptqmodel.load.",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated Evalution task identifiers (see --list-tasks).",
    )
    parser.add_argument(
        "--chat-template-tasks",
        default=None,
        help="Comma-separated tasks that should apply the model's chat template during evaluation.",
    )
    parser.add_argument(
        "--batch-size",
        default="auto",
        type=_parse_batch_size,
        help="Evaluation batch size passed to Evalution (integer or 'auto').",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="dtype override forwarded to gptqmodel.load (default: auto).",
    )
    parser.add_argument(
        "--gen-kwargs",
        default=None,
        help="Generation kwargs forwarded to Evalution, e.g. 'temperature=0.0,top_k=50'.",
    )
    parser.add_argument(
        "--model-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra model_args forwarded to Evalution (repeatable).",
    )
    parser.add_argument(
        "--load-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional keyword arguments passed to gptqmodel.load (repeatable).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow loading models that require remote code execution.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_RESULTS_PATH),
        help="JSON file to store aggregated metrics (use '-' to skip saving).",
    )
    parser.add_argument(
        "--table-format",
        default="github",
        help="Tabulate table format (defaults to 'github').",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List supported Evalution task identifiers and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_tasks:
        _list_tasks()
        return
    if not args.model:
        raise ValueError("--model is required unless --list-tasks is used.")

    tasks = [_resolve_task(name) for name in _split_tasks(args.tasks)]
    if not tasks:
        raise ValueError("No Evalution tasks specified.")
    chat_template_tasks = {_resolve_task(name) for name in _split_tasks(args.chat_template_tasks)}
    backend: gptqmodel.BACKEND = args.backend

    load_kwargs = _parse_key_value_pairs(args.load_arg)
    model = GPTQModel.load(
        args.model,
        backend=backend,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        **load_kwargs,
    )

    if not isinstance(model, BaseQModel):
        raise RuntimeError("Failed to load GPT-QModel; received unexpected object type.")

    model_args = _parse_key_value_pairs(args.model_arg)

    aggregated_metrics: Dict[str, Dict[str, float]] = {}

    grouped_tasks: Dict[bool, List[str]] = {}
    for task in tasks:
        apply_chat = task in chat_template_tasks
        grouped_tasks.setdefault(apply_chat, []).append(task)

    for apply_chat_template, grouped in grouped_tasks.items():
        if not grouped:
            continue

        result = evaluate(
            model_or_id_or_path=model,
            tasks=grouped,
            batch_size=args.batch_size,
            trust_remote_code=args.trust_remote_code,
            output_path=None,
            backend=backend,
            model_args=model_args.copy(),
            gen_kwargs=args.gen_kwargs,
            apply_chat_template=apply_chat_template,
        )

        group_metrics = _extract_metrics(result)
        aggregated_metrics.update(group_metrics)

    _print_metrics_table(aggregated_metrics, args.table_format)

    if args.output and args.output != "-":
        output_path = Path(args.output).expanduser()
        output_path.write_text(json.dumps(aggregated_metrics, indent=2))
        print(f"Saved aggregated metrics to {output_path}")


if __name__ == "__main__":
    main()
