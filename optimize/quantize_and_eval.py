#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Generic quantize + evaluate pipeline for coverage-mix calibration datasets.

Merges the Qwen3-8B (scripts/quantize_eval_qwen3_8b_pr151_mix.py) and
Laguna-S-2.1 (scripts/quantize_laguna_s21_w4g64_covmix.py +
scripts/eval_laguna_covmix.py) pipelines into one parameterized entrypoint.

Default recipe: GPTQ 4-bit, group_size 64, GAR (act_group_aware=True),
desc_act=False, scale_search=activation. Calibration comes from a parquet
with a `messages` column (as produced by optimize/calibration_coverage.py).

Dense models: pass a single GPU via --gpus. Large MoE models: pass multiple
GPUs plus --moe-routing-bypass / --vram-strategy balanced /
--calibration-data-device balanced.

Evaluation runs through Evalution (tests.eval): gsm8k_platinum_cot,
arc_challenge, mmlu_stem, and mmlu history subsets by default, with optional
fused inference (model.fuse(): QKV + gate/up) on a single eval GPU.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Chat templates measurably hurt some models (e.g. Qwen3-8B) on generation and
# loglikelihood tasks, while chat-tuned models (e.g. Laguna) need them for CoT
# generation; controlled by --chat-template.

MMLU_HISTORY_SUBSETS = [
    "humanities.high_school_european_history",
    "humanities.high_school_us_history",
    "humanities.high_school_world_history",
    "humanities.prehistory",
]

DEFAULT_TASKS = "gsm8k_platinum_cot,arc_challenge,mmlu_stem,mmlu_history"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True, help="Dense (or per-layer sharded) base checkpoint.")
    parser.add_argument("--output", required=True, help="Quantized checkpoint output directory.")
    parser.add_argument(
        "--calibration-parquet",
        required=True,
        help="Parquet with a `messages` column (see optimize/calibration_coverage.py artifacts).",
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated physical (PCI-bus-ordered) GPU ids used for quantization.",
    )
    parser.add_argument(
        "--eval-gpu",
        type=int,
        default=None,
        help="Physical GPU id for evaluation (default: first of --gpus).",
    )
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1, help="Calibration forward batch size.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=120.0,
        help="Max seconds to wait for the requested GPUs to become idle before failing.",
    )
    # MoE controls (Laguna-style large MoE models).
    parser.add_argument(
        "--moe-routing-bypass",
        action="store_true",
        help="Quantize with MoE ExpertsRoutingBypass so every expert sees all calibration tokens.",
    )
    parser.add_argument(
        "--moe-batch-size",
        type=int,
        default=None,
        help="ExpertsRoutingBypass module batch size; caps VRAM pressure while quantizing experts.",
    )
    parser.add_argument(
        "--vram-strategy",
        choices=["exclusive", "balanced"],
        default=None,
        help="dense/moe VramStrategy override (multi-GPU MoE quantization wants `balanced`).",
    )
    parser.add_argument(
        "--calibration-data-device",
        default=None,
        help='Calibration activation placement, e.g. "balanced" to round-robin across GPUs.',
    )
    # Embed/lm_head second stage.
    parser.add_argument(
        "--quant-embed-lm-head",
        action="store_true",
        help=(
            "Second post-quant stage: load the quantized checkpoint at --output, requantize with "
            "embed+lm_head quantization, save to <output>_embed_lmhead_w<bits>g<gs>, and evaluate it."
        ),
    )
    parser.add_argument("--embed-bits", type=int, default=None, help="Embed/lm_head bits (default: --bits).")
    parser.add_argument(
        "--embed-group-size", type=int, default=None, help="Embed/lm_head group size (default: --group-size)."
    )
    # Eval controls.
    parser.add_argument("--tasks", default=DEFAULT_TASKS, help="Comma-separated eval task ids.")
    parser.add_argument("--mmlu-rows", type=int, default=1024)
    parser.add_argument("--gen-batch-size", type=int, default=32)
    parser.add_argument("--mmlu-batch-size", type=int, default=16)
    parser.add_argument(
        "--chat-template",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply the model chat template to generation tasks (loglikelihood tasks never use it).",
    )
    parser.add_argument(
        "--fuse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply model.fuse() (QKV + gate/up) before evaluation for faster inference.",
    )
    parser.add_argument("--skip-quant", action="store_true", help="Evaluate an existing checkpoint only.")
    parser.add_argument("--skip-eval", action="store_true", help="Quantize/save only.")
    parser.add_argument(
        "--dense-baseline",
        action="store_true",
        help="Evaluate the dense model at --model-path instead of the quantized checkpoint.",
    )
    return parser.parse_args()


def _idle_gate(
    physical_gpus: list[int],
    samples: int = 3,
    interval: float = 2.0,
    memory_slack_mb: int = 256,
    timeout: float = 120.0,
) -> dict[int, dict]:
    accepted: dict[int, dict] = {}
    consecutive = 0
    last_seen: dict[int, dict] = {}
    deadline = time.monotonic() + timeout
    wanted = set(physical_gpus)
    while consecutive < samples:
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"physical GPUs {physical_gpus} did not become idle within {timeout:.0f}s: "
                f"last samples {last_seen} (idle contract: util==0%, mem<= {memory_slack_mb}MiB "
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
        seen: set[int] = set()
        all_idle = True
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            idx, pci, uuid, name, mem_used, util = parts
            idx = int(idx)
            if idx not in wanted:
                continue
            seen.add(idx)
            util = int(util)
            mem_used = int(mem_used)
            print(f"[preflight] GPU {idx} ({name}) util={util}% mem_used={mem_used}MiB", flush=True)
            last_seen[idx] = {"pci": pci, "uuid": uuid, "name": name, "util_pct": util, "memory_used_mb": mem_used}
            if util == 0 and mem_used <= memory_slack_mb:
                accepted[idx] = {
                    "physical_id": idx,
                    "pci": pci,
                    "uuid": uuid,
                    "name": name,
                    "memory_used_mb": mem_used,
                }
            else:
                all_idle = False
        missing = wanted - seen
        if missing:
            raise RuntimeError(f"physical GPUs {sorted(missing)} not found in nvidia-smi output")
        if all_idle:
            consecutive += 1
        else:
            consecutive = 0
            accepted = {}
        if consecutive < samples:
            time.sleep(interval)
    for idx in sorted(accepted):
        info = accepted[idx]
        print(
            f"[preflight] Accepted GPU {idx} pci={info['pci']} uuid={info['uuid']} "
            f"after {samples} idle samples (util==0%, mem<= {memory_slack_mb}MiB)",
            flush=True,
        )
    return accepted


def _verify_visible_uuids(physical_gpus: list[int], gpu_infos: dict[int, dict]) -> None:
    import torch

    torch.cuda.init()
    for local_idx, physical_id in enumerate(physical_gpus):
        visible_uuid = str(torch.cuda.get_device_properties(local_idx).uuid).replace("-", "").lower()
        requested_uuid = gpu_infos[physical_id]["uuid"].removeprefix("GPU-").replace("-", "").lower()
        if visible_uuid != requested_uuid:
            raise RuntimeError(
                f"visible GPU cuda:{local_idx} UUID {visible_uuid} does not match requested "
                f"physical GPU {physical_id} ({gpu_infos[physical_id]['uuid']})"
            )
        print(
            f"[preflight] cuda:{local_idx} == physical GPU {physical_id} ({gpu_infos[physical_id]['uuid']})",
            flush=True,
        )


def _load_calibration_rows(parquet_path: Path) -> list[dict]:
    import pandas as pd

    if not parquet_path.exists():
        raise FileNotFoundError(f"Calibration parquet not found at {parquet_path}")
    df = pd.read_parquet(parquet_path)
    rows = [{"messages": list(messages)} for messages in df["messages"].tolist()]
    print(f"[data] Loaded {len(rows)} calibration rows from {parquet_path}", flush=True)
    return rows


def _build_quantize_config(args: argparse.Namespace):
    from gptqmodel.quantization import FORMAT, METHOD
    from gptqmodel.quantization.config import (
        ExpertsRoutingBypass,
        MoEConfig,
        QuantizeConfig,
        VramStrategy,
    )

    kwargs: dict = {}
    if args.moe_routing_bypass:
        kwargs["moe"] = MoEConfig(routing=ExpertsRoutingBypass(batch_size=args.moe_batch_size))
    if args.vram_strategy is not None:
        strategy = VramStrategy(args.vram_strategy)
        kwargs["dense_vram_strategy"] = strategy
        kwargs["moe_vram_strategy"] = strategy
    if args.calibration_data_device is not None:
        kwargs["calibration_data_device"] = args.calibration_data_device

    return QuantizeConfig(
        quant_method=METHOD.GPTQ,
        format=FORMAT.GPTQ,
        bits=args.bits,
        group_size=args.group_size,
        desc_act=False,
        act_group_aware=True,
        scale_search="activation",
        **kwargs,
    )


def _quantize(args: argparse.Namespace) -> None:
    from gptqmodel import GPTQModel

    quantize_config = _build_quantize_config(args)
    print(f"[quant] QuantizeConfig: {quantize_config}", flush=True)

    print(f"[load] Loading dense model from {args.model_path} ...", flush=True)
    model = GPTQModel.load(
        args.model_path,
        quantize_config=quantize_config,
        trust_remote_code=args.trust_remote_code,
        dtype="auto",
        device_map="auto",
    )

    calibration_rows = _load_calibration_rows(Path(args.calibration_parquet))
    start = time.time()
    model.quantize(calibration_rows, batch_size=args.batch_size, backend="auto")
    print(f"[quant] Quantization finished in {time.time() - start:.1f}s", flush=True)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    print(f"[save] Saving quantized model to {output} ...", flush=True)
    model.save(str(output))
    del model
    import torch

    torch.cuda.empty_cache()
    print("[done] quantize + save complete", flush=True)


def _requantize_embed_lm_head(args: argparse.Namespace) -> str:
    from gptqmodel import GPTQModel
    from gptqmodel.quantization import QuantizeEmbed

    print(f"[requant] Loading quantized checkpoint {args.output} ...", flush=True)
    model = GPTQModel.load(
        args.output,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )

    # GPTQModel.load ignores a caller quantize_config for already-quantized
    # checkpoints, so apply the embed/lm_head overrides after loading.
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

    calibration_rows = _load_calibration_rows(Path(args.calibration_parquet))
    start = time.time()
    model.requantize(calibration=calibration_rows, batch_size=args.batch_size, embed_quant_mode=QuantizeEmbed.BOTH)
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


class _FusedEnginePatch:
    """Swap evalution.GPTQModel for a subclass that fuses (QKV + gate/up) after load.

    tests.eval builds its engine internally from evalution.GPTQModel, so the fused engine
    is installed via attribute patch for the duration of the evaluation calls.
    """

    def __enter__(self):
        import evalution
        from evalution.engines.gptqmodel_engine import GPTQModel as _Engine

        class FusedGPTQModel(_Engine):
            def build(self, model):
                session = super().build(model)
                fused = session.model_wrapper.fuse()
                print(f"[eval] model.fuse() applied: {fused}", flush=True)
                return session

        self._evalution = evalution
        self._original = evalution.GPTQModel
        evalution.GPTQModel = FusedGPTQModel
        return self

    def __exit__(self, *exc):
        self._evalution.GPTQModel = self._original
        return False


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
    fuse = args.fuse and not args.dense_baseline
    # paged|flash_attention_2 enables FA2 plus transformers continuous (paged) batching in the
    # Evalution GPTQModel engine; it degrades gracefully to paged|sdpa when FA2 is unavailable.
    model_args = {"attn_implementation": "paged|flash_attention_2"}

    results: dict[str, dict] = {}

    def run(task: str, apply_chat_template: bool, batch_size: int, suite_kwargs: dict | None = None) -> dict:
        print(
            f"\n[eval] task={task} chat_template={apply_chat_template} batch_size={batch_size} "
            f"suite_kwargs={suite_kwargs}",
            flush=True,
        )
        start = time.time()
        with _FusedEnginePatch() if fuse else contextlib.nullcontext():
            result = evaluate(
                model_or_id_or_path=checkpoint,
                tasks=[task],
                backend=backend,
                batch_size=batch_size,
                apply_chat_template=apply_chat_template,
                gen_kwargs="do_sample=false,temperature=0.0,top_p=1.0,top_k=50",
                suite_kwargs=suite_kwargs or {},
                trust_remote_code=args.trust_remote_code,
                model_args=model_args,
            )
        print(format_eval_result_table(result), flush=True)
        print(f"[eval] task={task} took {time.time() - start:.1f}s", flush=True)
        task_results = get_eval_task_results(result)
        return next(iter(task_results.values())) if task_results else {}

    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        if task == "mmlu_stem":
            results[task] = run("mmlu_stem", False, args.mmlu_batch_size, suite_kwargs={"max_rows": args.mmlu_rows})
        elif task == "mmlu_history":
            results[task] = run(
                "mmlu",
                False,
                args.mmlu_batch_size,
                suite_kwargs={"subsets": MMLU_HISTORY_SUBSETS, "max_rows": args.mmlu_rows},
            )
        else:
            results[task] = run(task, args.chat_template, args.gen_batch_size)
    return results


def main() -> None:
    args = _parse_args()
    physical_gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    eval_gpu = args.eval_gpu if args.eval_gpu is not None else physical_gpus[0]

    quant_stage = not args.skip_quant and not args.dense_baseline and not args.quant_embed_lm_head
    stage_gpus = physical_gpus if quant_stage else [eval_gpu]

    gpu_infos = _idle_gate(stage_gpus, timeout=args.idle_timeout)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in stage_gpus)
    _verify_visible_uuids(stage_gpus, gpu_infos)

    if quant_stage:
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
