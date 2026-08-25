#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Run the standard post-quant Evalution tasks on one saved quantized checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.utils.backend import BACKEND  # noqa: E402
from tests.eval import evaluate, format_eval_result_table, get_eval_task_results  # noqa: E402


MMLU_HISTORY_SUBSETS = (
    "humanities.high_school_european_history",
    "humanities.high_school_us_history",
    "humanities.high_school_world_history",
    "humanities.prehistory",
)

# QVQ's CUDA transforms and decoded GEMM are FP16-native. Loading the dense
# shell in BF16 would round every QVQLinear output back to BF16 between layers.
QVQ_INFERENCE_DTYPE = "float16"
SUPPORTED_BACKENDS = (BACKEND.QVQ, BACKEND.EXL3_EXLLAMA_V3)

TASKS = (
    ("arc_challenge", "arc_challenge", True, {}),
    ("gsm8k_platinum_cot", "gsm8k_platinum_cot", True, {}),
    ("mmlu_stem", "mmlu_stem", False, {}),
    ("mmlu_humanities", "mmlu", False, {"subsets": "humanities"}),
    ("mmlu_history", "mmlu", False, {"subsets": MMLU_HISTORY_SUBSETS}),
)


def _select_tasks(labels: list[str] | None):
    """Select independent full-row task gates for one-GPU-per-task execution."""

    if labels is None:
        return TASKS
    if len(labels) != len(set(labels)):
        raise ValueError("Quantized evaluation task labels must be unique.")
    requested = set(labels)
    selected = tuple(task for task in TASKS if task[0] in requested)
    missing = requested - {task[0] for task in selected}
    if missing:
        raise ValueError(f"Unknown quantized evaluation task labels: {sorted(missing)}")
    return selected


def _parse_backend(value: str) -> BACKEND:
    """Restrict checkpoint evaluation to the two native low-bit backends in this sweep."""

    backend = BACKEND(value)
    if backend not in SUPPORTED_BACKENDS:
        expected = ", ".join(item.value for item in SUPPORTED_BACKENDS)
        raise argparse.ArgumentTypeError(f"Unsupported checkpoint backend {value!r}; expected one of: {expected}.")
    return backend


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--backend",
        type=_parse_backend,
        default=BACKEND.QVQ,
        help="Native checkpoint backend (default: qvq).",
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=tuple(task[0] for task in TASKS),
        help="Run only this task label; repeat to select multiple tasks.",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Quantized checkpoint does not exist: {checkpoint}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    payload = {
        "checkpoint": str(checkpoint),
        "backend": args.backend.value,
        "dtype": QVQ_INFERENCE_DTYPE,
        "batch_size": args.batch_size,
        "tasks": {},
    }
    for label, task, apply_chat_template, suite_kwargs in _select_tasks(args.task):
        print(
            f"[eval] task={task} checkpoint={checkpoint} "
            f"chat_template={apply_chat_template} batch_size={args.batch_size}",
            flush=True,
        )
        started = time.perf_counter()
        result = evaluate(
            model_or_id_or_path=str(checkpoint),
            tasks=[task],
            backend=args.backend,
            model_args={"dtype": QVQ_INFERENCE_DTYPE},
            batch_size=args.batch_size,
            apply_chat_template=apply_chat_template,
            gen_kwargs="do_sample=false,temperature=0.0,top_p=1.0,top_k=50",
            suite_kwargs=suite_kwargs,
            trust_remote_code=False,
        )
        print(format_eval_result_table(result), flush=True)
        task_results = get_eval_task_results(result)
        payload["tasks"][label] = {
            "evalution_task": task,
            "suite_kwargs": suite_kwargs,
            "seconds": time.perf_counter() - started,
            "metrics": next(iter(task_results.values())) if task_results else {},
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
