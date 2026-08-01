#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Dense BF16 baseline evaluation for Llama-3.2-1B-Instruct."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gptqmodel import GPTQModel  # noqa: E402
from gptqmodel.utils.backend import BACKEND  # noqa: E402
from tests.eval import evaluate, get_eval_task_results  # noqa: E402

MMLU_HISTORY_SUBSETS = [
    "humanities.high_school_european_history",
    "humanities.high_school_us_history",
    "humanities.high_school_world_history",
    "humanities.prehistory",
]


def _run_task(model, task: str, apply_chat_template: bool, suite_kwargs: dict | None = None, batch_size: int = 64) -> dict:
    print(f"\n[eval] {task} (chat_template={apply_chat_template}, batch_size={batch_size}) ...")
    result = evaluate(
        model_or_id_or_path=model,
        tasks=[task],
        backend=BACKEND.AUTO,
        batch_size=batch_size,
        apply_chat_template=apply_chat_template,
        gen_kwargs="do_sample=false,temperature=0.0,top_p=1.0,top_k=50",
        suite_kwargs=suite_kwargs or {},
        trust_remote_code=False,
    )
    task_results = get_eval_task_results(result)
    metrics = next(iter(task_results.values())) if task_results else {}
    print(metrics)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/monster/data/model/Llama-3.2-1B-Instruct")
    parser.add_argument("--mmlu-batch-size", type=int, default=8)
    parser.add_argument("--output", default="/tmp/llama32_dense_baseline/results.json")
    args = parser.parse_args()

    print(f"Loading dense model from {args.model} ...")
    model = GPTQModel.load(args.model, backend=BACKEND.AUTO, trust_remote_code=False)
    print(f"Loaded {type(model).__name__}")

    scores = {"label": "dense_bf16", "model": args.model}
    scores["gsm8k_platinum_cot"] = _run_task(model, "gsm8k_platinum_cot", apply_chat_template=True)
    scores["arc_challenge"] = _run_task(model, "arc_challenge", apply_chat_template=True)
    scores["mmlu_stem"] = _run_task(
        model, "mmlu_stem", apply_chat_template=False, batch_size=args.mmlu_batch_size
    )
    scores["mmlu_history"] = _run_task(
        model,
        "mmlu",
        apply_chat_template=False,
        batch_size=args.mmlu_batch_size,
        suite_kwargs={"subsets": MMLU_HISTORY_SUBSETS},
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scores, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== dense baseline results ===")
    print(json.dumps(scores, indent=2, sort_keys=True))
    print(f"\nWrote results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
