#!/usr/bin/env python3
"""Evaluate one saved ScaleSearch A/B checkpoint with the repository Evalution path."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import huggingface_hub  # noqa: E402
import torch  # noqa: E402
from transformers.utils import hub as transformers_hub  # noqa: E402


for _hub_name in ("create_repo", "hf_hub_download", "list_repo_tree", "snapshot_download"):
    if not hasattr(transformers_hub, _hub_name):
        setattr(transformers_hub, _hub_name, getattr(huggingface_hub, _hub_name))

import gptqmodel  # noqa: E402
from gptqmodel import BACKEND  # noqa: E402
from tests.eval import evaluate, get_eval_task_results  # noqa: E402


GPTQMODEL_SOURCE = Path(gptqmodel.__file__).resolve()
if REPO_ROOT not in GPTQMODEL_SOURCE.parents:
    raise RuntimeError(
        f"Evaluation must import GPT-QModel from {REPO_ROOT}, resolved {GPTQMODEL_SOURCE} instead."
    )


TASKS = {
    "arc_challenge": True,
    "mmlu_stem": False,
    "gsm8k_platinum_cot": True,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tasks", default=",".join(TASKS))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_path = Path(args.model).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    selected_tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    unknown_tasks = [task for task in selected_tasks if task not in TASKS]
    if unknown_tasks:
        raise ValueError(f"Unsupported tasks: {unknown_tasks}")

    summary = {
        "variant": args.variant,
        "model": str(model_path),
        "batch_size": args.batch_size,
        "backend": BACKEND.MARLIN.name,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gptqmodel": getattr(gptqmodel, "__version__", None),
            "gptqmodel_source": str(GPTQMODEL_SOURCE),
            "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "tasks": {},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for task in selected_tasks:
        start = time.perf_counter()
        result = evaluate(
            model_or_id_or_path=str(model_path),
            tasks=[task],
            batch_size=args.batch_size,
            backend=BACKEND.MARLIN,
            apply_chat_template=TASKS[task],
            model_args={
                "attn_implementation": "eager",
                "device": "cuda:0",
                "seed": 42,
                "random_seed": 42,
            },
            gen_kwargs="do_sample=false,temperature=0.0,top_p=1.0,top_k=50",
        )
        task_results = get_eval_task_results(result)
        summary["tasks"][task] = {
            "apply_chat_template": TASKS[task],
            "seconds": time.perf_counter() - start,
            "metrics": task_results.get(task, {}),
            "engine": result.get("engine", {}),
        }
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
