#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Quantize Qwen3-8B with the PR #151 calibration mix and evaluate it.

Recipe: GPTQ 4-bit, group_size 64, GAR (act_group_aware=True), desc_act=False,
scale_search=activation. Calibration dataset:
dataset/calibration_mix_128k_qwen3_8b/calibration.parquet (messages column).

Evaluation (MARLIN backend): gsm8k_platinum_cot, arc_challenge, mmlu_stem,
mmlu history subsets; MMLU tasks limited to 1024 rows.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CALIBRATION_PARQUET = REPO_ROOT / "dataset/calibration_mix_128k_qwen3_8b/calibration.parquet"

# Chat templates measurably hurt Qwen3-8B on generation/loglikelihood tasks;
# always evaluate these without a template.
NO_CHAT_TEMPLATE_TASKS = {"gsm8k_platinum_cot", "gsm8k_cot", "arc_challenge"}

MMLU_HISTORY_SUBSETS = [
    "humanities.high_school_european_history",
    "humanities.high_school_us_history",
    "humanities.high_school_world_history",
    "humanities.prehistory",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/monster/data/model/Qwen3-8B")
    parser.add_argument("--gpu", type=int, default=7, help="Physical (PCI-bus-ordered) GPU id.")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--output", default="/monster/data/model/qwen3_8b_gptq_w4g64_gar_actss_pr151mix")
    parser.add_argument("--mmlu-rows", type=int, default=1024)
    parser.add_argument("--gen-batch-size", type=int, default=32)
    parser.add_argument("--mmlu-batch-size", type=int, default=16)
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=120.0,
        help="Max seconds to wait for the requested GPU to become idle before failing.",
    )
    parser.add_argument(
        "--embed-bits",
        type=int,
        default=None,
        help="Bits for the embed/lm_head stage (default: same as --bits).",
    )
    parser.add_argument(
        "--embed-group-size",
        type=int,
        default=None,
        help="Group size for the embed/lm_head stage (default: same as --group-size).",
    )
    parser.add_argument("--skip-quant", action="store_true", help="Evaluate an existing checkpoint only.")
    parser.add_argument("--skip-eval", action="store_true", help="Quantize/save only.")
    parser.add_argument(
        "--dense-baseline",
        action="store_true",
        help="Evaluate the dense BF16 model at --model-path instead of the quantized checkpoint.",
    )
    parser.add_argument(
        "--quant-embed-lm-head",
        action="store_true",
        help=(
            "Second post-quant stage: load the quantized checkpoint at --output, requantize with "
            "embed+lm_head quantization (same quantize config), save to <output>_embed_lmhead, and "
            "evaluate that checkpoint."
        ),
    )
    return parser.parse_args()


def _idle_gate(
    physical_gpu: int,
    samples: int = 3,
    interval: float = 2.0,
    memory_slack_mb: int = 256,
    timeout: float = 120.0,
) -> dict:
    accepted = None
    consecutive = 0
    last_seen = None
    deadline = time.monotonic() + timeout
    while consecutive < samples:
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"physical GPU {physical_gpu} did not become idle within {timeout:.0f}s: "
                f"last sample {last_seen} (idle contract: util==0%, mem<= {memory_slack_mb}MiB "
                f"for {samples} consecutive samples)"
            )
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,pci.bus_id,uuid,name,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            idx, pci, uuid, name, mem_used, util = parts
            if int(idx) != physical_gpu:
                continue
            util = int(util)
            mem_used = int(mem_used)
            print(f"[preflight] GPU {idx} ({name}) util={util}% mem_used={mem_used}MiB", flush=True)
            last_seen = {"pci": pci, "uuid": uuid, "name": name, "util_pct": util, "memory_used_mb": mem_used}
            if util == 0 and mem_used <= memory_slack_mb:
                if accepted is None:
                    accepted = {
                        "physical_id": int(idx),
                        "pci": pci,
                        "uuid": uuid,
                        "name": name,
                        "memory_used_mb": mem_used,
                    }
                consecutive += 1
            else:
                consecutive = 0
                accepted = None
            break
        else:
            raise RuntimeError(f"physical GPU {physical_gpu} not found in nvidia-smi output")
        if consecutive < samples:
            time.sleep(interval)
    print(
        f"[preflight] Accepted GPU {accepted['physical_id']} pci={accepted['pci']} uuid={accepted['uuid']} "
        f"after {samples} idle samples (util==0%, mem<= {memory_slack_mb}MiB)",
        flush=True,
    )
    return accepted


def _load_calibration_rows() -> list[dict]:
    import pandas as pd

    if not CALIBRATION_PARQUET.exists():
        raise FileNotFoundError(f"Calibration parquet not found at {CALIBRATION_PARQUET}")
    df = pd.read_parquet(CALIBRATION_PARQUET)
    rows = [{"messages": list(messages)} for messages in df["messages"].tolist()]
    print(f"[data] Loaded {len(rows)} calibration rows from {CALIBRATION_PARQUET}", flush=True)
    return rows


def _quantize(args: argparse.Namespace) -> None:
    from gptqmodel import GPTQModel
    from gptqmodel.quantization import FORMAT, METHOD
    from gptqmodel.quantization.config import QuantizeConfig

    quantize_config = QuantizeConfig(
        quant_method=METHOD.GPTQ,
        format=FORMAT.GPTQ,
        bits=args.bits,
        group_size=args.group_size,
        desc_act=False,
        act_group_aware=True,
        scale_search="activation",
    )
    print(f"[quant] QuantizeConfig: {quantize_config}", flush=True)

    print(f"[load] Loading dense model from {args.model_path} ...", flush=True)
    model = GPTQModel.load(
        args.model_path,
        quantize_config=quantize_config,
        trust_remote_code=False,
        dtype="auto",
        device_map="auto",
    )

    calibration_rows = _load_calibration_rows()
    start = time.time()
    model.quantize(calibration_rows, batch_size=1, backend="auto")
    print(f"[quant] Quantization finished in {time.time() - start:.1f}s", flush=True)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    print(f"[save] Saving quantized model to {output} ...", flush=True)
    model.save(str(output))
    del model
    import torch

    torch.cuda.empty_cache()


def _requantize_embed_lm_head(args: argparse.Namespace) -> str:
    from gptqmodel import GPTQModel
    from gptqmodel.quantization import QuantizeEmbed

    print(f"[requant] Loading quantized checkpoint {args.output} ...", flush=True)
    model = GPTQModel.load(
        args.output,
        device_map="auto",
        trust_remote_code=False,
    )

    # GPTQModel.load ignores a caller quantize_config for already-quantized
    # checkpoints, so apply the embed/lm_head overrides (same settings as the
    # main quant config, not the looper's default 8-bit/g32) after loading.
    embed_bits = args.embed_bits if args.embed_bits is not None else args.bits
    embed_group_size = args.embed_group_size if args.embed_group_size is not None else args.group_size
    embed_lm_head_cfg = {
        "bits": embed_bits,
        "group_size": embed_group_size,
        "sym": True,
        "desc_act": False,
        "act_group_aware": True,
        "scale_search": "activation",
    }
    if model.quantize_config.dynamic is None:
        model.quantize_config.dynamic = {}
    model.quantize_config.dynamic["model.embed_tokens"] = dict(embed_lm_head_cfg)
    model.quantize_config.dynamic["lm_head"] = dict(embed_lm_head_cfg)
    for name in ("model.embed_tokens", "lm_head"):
        effective = model.quantize_config.dynamic_get(name, default=None)
        assert effective == embed_lm_head_cfg, f"dynamic override for {name} not effective: {effective}"
    print(f"[requant] effective dynamic overrides: {model.quantize_config.dynamic}", flush=True)

    calibration_rows = _load_calibration_rows()
    start = time.time()
    model.requantize(calibration=calibration_rows, batch_size=1, embed_quant_mode=QuantizeEmbed.BOTH)
    print(f"[requant] embed+lm_head requantization finished in {time.time() - start:.1f}s", flush=True)

    suffix = f"_embed_lmhead_w{embed_bits}g{embed_group_size}"
    output = Path(str(args.output) + suffix)
    output.mkdir(parents=True, exist_ok=True)
    print(f"[save] Saving requantized model to {output} ...", flush=True)
    model.save(str(output))
    del model
    import torch

    torch.cuda.empty_cache()
    return str(output)


def _evaluate(args: argparse.Namespace, checkpoint_override: str | None = None) -> dict:
    from gptqmodel.utils.backend import BACKEND
    from tests.eval import evaluate, format_eval_result_table, get_eval_task_results

    if checkpoint_override is not None:
        checkpoint = checkpoint_override
    elif args.dense_baseline:
        checkpoint = args.model_path
    else:
        checkpoint = args.output
    backend = BACKEND.AUTO if args.dense_baseline else BACKEND.MARLIN
    results: dict[str, dict] = {}

    def run(task: str, apply_chat_template: bool, batch_size: int, suite_kwargs: dict | None = None) -> dict:
        if task in NO_CHAT_TEMPLATE_TASKS:
            apply_chat_template = False
        print(
            f"\n[eval] task={task} chat_template={apply_chat_template} batch_size={batch_size} "
            f"suite_kwargs={suite_kwargs}",
            flush=True,
        )
        start = time.time()
        result = evaluate(
            model_or_id_or_path=checkpoint,
            tasks=[task],
            backend=backend,
            batch_size=batch_size,
            apply_chat_template=apply_chat_template,
            gen_kwargs="do_sample=false,temperature=0.0,top_p=1.0,top_k=50",
            suite_kwargs=suite_kwargs or {},
            trust_remote_code=False,
        )
        print(format_eval_result_table(result), flush=True)
        print(f"[eval] task={task} took {time.time() - start:.1f}s", flush=True)
        task_results = get_eval_task_results(result)
        return next(iter(task_results.values())) if task_results else {}

    results["gsm8k_platinum_cot"] = run("gsm8k_platinum_cot", False, args.gen_batch_size)
    results["arc_challenge"] = run("arc_challenge", False, args.gen_batch_size)
    results["mmlu_stem"] = run(
        "mmlu_stem", False, args.mmlu_batch_size, suite_kwargs={"max_rows": args.mmlu_rows}
    )
    results["mmlu_history"] = run(
        "mmlu",
        False,
        args.mmlu_batch_size,
        suite_kwargs={"subsets": MMLU_HISTORY_SUBSETS, "max_rows": args.mmlu_rows},
    )
    return results


def main() -> None:
    args = _parse_args()

    gpu_info = _idle_gate(args.gpu, timeout=args.idle_timeout)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch

    torch.cuda.init()
    visible_uuid = str(torch.cuda.get_device_properties(0).uuid).replace("-", "").lower()
    requested_uuid = gpu_info["uuid"].removeprefix("GPU-").replace("-", "").lower()
    if visible_uuid != requested_uuid:
        raise RuntimeError(
            f"visible GPU UUID {visible_uuid} does not match requested physical GPU {gpu_info['uuid']}"
        )
    print(f"[preflight] cuda:0 == physical GPU {gpu_info['physical_id']} ({gpu_info['uuid']})", flush=True)

    if not args.skip_quant and not args.dense_baseline and not args.quant_embed_lm_head:
        _quantize(args)

    checkpoint_override = None
    if args.quant_embed_lm_head:
        checkpoint_override = _requantize_embed_lm_head(args)

    if args.skip_eval:
        print("[done] quantize-only run complete")
        return

    results = _evaluate(args, checkpoint_override=checkpoint_override)
    if args.quant_embed_lm_head:
        embed_bits = args.embed_bits if args.embed_bits is not None else args.bits
        embed_group_size = args.embed_group_size if args.embed_group_size is not None else args.group_size
        results_name = f"eval_results_embed_lmhead_w{embed_bits}g{embed_group_size}.json"
    elif args.dense_baseline:
        results_name = "eval_results_dense_bf16.json"
    else:
        results_name = "eval_results.json"
    Path(args.output).mkdir(parents=True, exist_ok=True)
    results_path = Path(args.output) / results_name
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print("\n=== final results ===")
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"[done] wrote {results_path}")


if __name__ == "__main__":
    main()
