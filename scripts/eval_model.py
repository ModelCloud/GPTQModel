#!/usr/bin/env python
# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 gptqmodel contributors
# SPDX-License-Identifier: Apache-2.0
"""CLI helper to run lm-eval tasks against a GPTQModel checkpoint."""

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
from gptqmodel.utils.eval import EVAL


if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault(
    "PYTORCH_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7",
)

DEFAULT_RESULTS_PATH = Path("lm_eval_results.json")
DEFAULT_TASKS = (EVAL.LM_EVAL.ARC_CHALLENGE,)
DEFAULT_TASK_MANAGER_PATH = Path(__file__).resolve().parent.parent / "tests" / "tasks"


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


def _resolve_task(name: str) -> EVAL.LM_EVAL:
    normalized = name.strip()
    for task in EVAL.LM_EVAL:
        if normalized.lower() in {task.value.lower(), task.name.lower()}:
            return task
    available = ", ".join(task.value for task in EVAL.LM_EVAL)
    raise argparse.ArgumentTypeError(f"Unknown lm-eval task '{name}'. Expected one of: {available}")


def _list_tasks() -> None:
    rows = [(task.name, task.value) for task in EVAL.LM_EVAL]
    print(tabulate(rows, headers=["Name", "Identifier"]))


def _extract_metrics(results: Dict) -> Dict[str, Dict[str, float]]:
    aggregated: Dict[str, Dict[str, float]] = {}
    task_results = results.get("results", {})
    for task_name, metrics in task_results.items():
        filtered = {
            metric: value
            for metric, value in metrics.items()
            if metric != "alias" and "stderr" not in metric
        }
        aggregated[task_name] = filtered
    return aggregated


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
        description="Run lm-eval tasks against a quantized model loaded via gptqmodel."
    )
    parser.add_argument("--model", required=True, help="Model path or Hugging Face repo id.")
    parser.add_argument(
        "--backend",
        default="auto",
        type=_parse_backend,
        help="Inference backend to use when loading with gptqmodel.load.",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(task.value for task in DEFAULT_TASKS),
        help="Comma-separated lm-eval task identifiers (see --list-tasks).",
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
        help="Evaluation batch size passed to lm-eval (integer or 'auto').",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="dtype override forwarded to gptqmodel.load (default: auto).",
    )
    parser.add_argument(
        "--gen-kwargs",
        default=None,
        help="Generation kwargs forwarded to lm-eval, e.g. 'temperature=0.0,top_k=50'.",
    )
    parser.add_argument(
        "--model-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra model_args forwarded to GPTQModel.eval (repeatable).",
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
        "--use-vllm",
        action="store_true",
        help="Run evaluation with the vLLM backend instead of the default gptqmodel harness.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional max_model_len passed to vLLM model args.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=898,
        help="Seed propagated to lm-eval for reproducibility.",
    )
    parser.add_argument(
        "--task-manager-path",
        type=str,
        default=str(DEFAULT_TASK_MANAGER_PATH) if DEFAULT_TASK_MANAGER_PATH.exists() else None,
        help="Optional path containing custom lm-eval tasks.",
    )
    parser.add_argument(
        "--include-default-tasks",
        action="store_true",
        help="Include lm-eval's builtin task registry alongside the custom task directory.",
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
        help="List supported lm-eval task identifiers and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_tasks:
        _list_tasks()
        return

    tasks = [_resolve_task(name) for name in _split_tasks(args.tasks)]
    if not tasks:
        raise ValueError("No lm-eval tasks specified.")
    chat_template_tasks = {_resolve_task(name).value for name in _split_tasks(args.chat_template_tasks)}

    llm_backend = "vllm" if args.use_vllm else "gptqmodel"
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
        raise RuntimeError("Failed to load GPTQModel; received unexpected object type.")

    model_args = _parse_key_value_pairs(args.model_arg)
    if args.max_model_len is not None:
        model_args.setdefault("max_model_len", args.max_model_len)

    if args.use_vllm:
        model_args.setdefault("dtype", "auto")
        model_args.setdefault("tensor_parallel_size", 1)
        model_args.setdefault("gpu_memory_utilization", 0.8)

    task_manager = None
    if args.task_manager_path:
        task_manager_path = Path(args.task_manager_path).expanduser().resolve()
        if not task_manager_path.exists():
            raise FileNotFoundError(f"Task manager path does not exist: {task_manager_path}")
        from lm_eval.tasks import TaskManager

        task_manager = TaskManager(
            include_path=str(task_manager_path),
            include_defaults=args.include_default_tasks,
        )

    aggregated_metrics: Dict[str, Dict[str, float]] = {}

    grouped_tasks: Dict[bool, List[EVAL.LM_EVAL]] = {}
    for task in tasks:
        apply_chat = task.value in chat_template_tasks
        grouped_tasks.setdefault(apply_chat, []).append(task)

    for apply_chat_template, grouped in grouped_tasks.items():
        if not grouped:
            continue

        result = gptqmodel.GPTQModel.eval(
            model_or_id_or_path=model,
            tasks=grouped,
            framework=EVAL.LM_EVAL,
            batch_size=args.batch_size,
            trust_remote_code=args.trust_remote_code,
            output_path=None,
            llm_backend=llm_backend,
            backend=backend,
            random_seed=args.random_seed,
            model_args=model_args.copy(),
            gen_kwargs=args.gen_kwargs,
            apply_chat_template=apply_chat_template,
            task_manager=task_manager,
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
