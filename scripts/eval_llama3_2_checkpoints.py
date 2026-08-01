#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Evaluate saved Llama-3.2-1B GPTQ checkpoints with MMLU-STEM and an MMLU aggregate.

Runs the four standard tasks (gsm8k_platinum_cot, arc_challenge, mmlu_stem,
and a configurable MMLU subset aggregate) against each checkpoint and writes a
JSON comparison table.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gptqmodel.utils.backend import BACKEND  # noqa: E402
from tests.eval import evaluate, format_eval_result_table, get_eval_task_results  # noqa: E402

CHECKPOINTS = [
    ("/tmp/llama3_2_gptq_saved_ckpt", "nm_512"),
    ("/tmp/llama3_2_imatrix_gptq_saved_ckpt", "imatrix_22"),
    ("/tmp/llama3_2_mixed_gptq_saved_ckpt", "mixed_22_512"),
]

MMLU_CHEMISTRY_SUBSETS = ["stem.college_chemistry", "stem.high_school_chemistry"]


def _run_task(
    checkpoint: str,
    task: str,
    apply_chat_template: bool,
    suite_kwargs: dict | None = None,
    batch_size: int = 64,
) -> dict:
    print(f"\n[eval] {task} on {checkpoint} (chat_template={apply_chat_template}, batch_size={batch_size}) ...")
    result = evaluate(
        model_or_id_or_path=checkpoint,
        tasks=[task],
        backend=BACKEND.MARLIN,
        batch_size=batch_size,
        apply_chat_template=apply_chat_template,
        gen_kwargs="do_sample=false,temperature=0.0,top_p=1.0,top_k=50",
        suite_kwargs=suite_kwargs or {},
        trust_remote_code=False,
    )
    print(format_eval_result_table(result))
    task_results = get_eval_task_results(result)
    # Flatten to the first (only) task's metrics.
    return next(iter(task_results.values())) if task_results else {}


def _eval_checkpoint(
    checkpoint: str,
    label: str,
    mmlu_subsets: list[str] | None = None,
    mmlu_label: str = "chemistry",
    mmlu_batch_size: int = 16,
) -> dict:
    scores = {"label": label, "checkpoint": checkpoint}

    # Chat-template tasks
    scores["gsm8k_platinum_cot"] = _run_task(
        checkpoint, "gsm8k_platinum_cot", apply_chat_template=True
    )
    scores["arc_challenge"] = _run_task(
        checkpoint, "arc_challenge", apply_chat_template=True
    )

    # MMLU-STEM (loglikelihood, no chat template)
    scores["mmlu_stem"] = _run_task(
        checkpoint, "mmlu_stem", apply_chat_template=False, batch_size=mmlu_batch_size
    )

    # MMLU aggregate (default: Chemistry; override with --mmlu-subsets/--mmlu-label)
    mmlu_result = _run_task(
        checkpoint,
        "mmlu",
        apply_chat_template=False,
        batch_size=mmlu_batch_size,
        suite_kwargs={"subsets": mmlu_subsets or MMLU_CHEMISTRY_SUBSETS},
    )
    scores[f"mmlu_{mmlu_label}"] = mmlu_result

    return scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help="space-separated checkpoint paths (default: the three calibration variants)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="labels for each checkpoint",
    )
    parser.add_argument(
        "--mmlu-subsets",
        nargs="+",
        default=None,
        help="MMLU subset paths to evaluate (default: chemistry subsets)",
    )
    parser.add_argument(
        "--mmlu-label",
        default="chemistry",
        help="Result key label for the MMLU aggregate (default: chemistry)",
    )
    parser.add_argument(
        "--mmlu-batch-size",
        type=int,
        default=16,
        help="Batch size for MMLU-STEM and the custom MMLU aggregate (default: 16)",
    )
    parser.add_argument(
        "--output",
        default="/tmp/llama32_eval_checkpoints/results.json",
        help="where to write the JSON results",
    )
    args = parser.parse_args()

    checkpoints = list(zip(args.checkpoints, args.labels)) if args.checkpoints else CHECKPOINTS
    if args.checkpoints and not args.labels:
        checkpoints = [(path, Path(path).name) for path in args.checkpoints]

    mmlu_subsets = args.mmlu_subsets
    mmlu_label = args.mmlu_label
    mmlu_batch_size = args.mmlu_batch_size

    all_results = []
    for checkpoint, label in checkpoints:
        if not Path(checkpoint).exists():
            print(f"[warn] Skipping missing checkpoint: {checkpoint}")
            continue
        all_results.append(
            _eval_checkpoint(
                checkpoint,
                label,
                mmlu_subsets=mmlu_subsets,
                mmlu_label=mmlu_label,
                mmlu_batch_size=mmlu_batch_size,
            )
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== aggregated results ===")
    print(json.dumps(all_results, indent=2, sort_keys=True))
    print(f"\nWrote results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
