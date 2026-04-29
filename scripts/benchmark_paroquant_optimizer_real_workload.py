#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Any

import torch
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel.quantization.paroquant.optimization import optimize_paroquant_linear
from gptqmodel.utils.paroquant_benchmark import (
    _normalize_model_dtype,
    load_nm_calibration,
)


DEFAULT_MODEL = "/monster/data/model/Llama-3.2-1B-Instruct"
DEFAULT_MODULES = ("model.layers.0.self_attn.q_proj", "model.layers.0.mlp.down_proj")


@dataclass(frozen=True)
class BenchmarkCase:
    label: str
    stage_impl: str
    pair_impl: str
    quantizer_impl: str


CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase("reference", "reference", "reference", "reference"),
    BenchmarkCase("pair_fast", "reference", "fast", "reference"),
    BenchmarkCase("stage_pair_fast", "fast", "fast", "reference"),
    BenchmarkCase("stage_fast", "fast", "reference", "reference"),
    BenchmarkCase("quant_fast", "reference", "reference", "fast"),
    BenchmarkCase("all_fast", "fast", "fast", "fast"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ParoQuant optimizer implementations on real calibration activations."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--module",
        dest="modules",
        action="append",
        default=None,
        help="Fully-qualified linear module path. Can be passed multiple times.",
    )
    parser.add_argument("--model-dtype", default="fp16")
    parser.add_argument("--calibration-rows", type=int, default=64)
    parser.add_argument("--capture-rows", type=int, default=2048)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--krot", type=int, default=8)
    parser.add_argument("--pair-ratio", type=float, default=0.5)
    parser.add_argument("--train-rows", type=int, default=2048)
    parser.add_argument("--val-rows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rotation-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--rotation-lr", type=float, default=0.05)
    parser.add_argument("--weight-lr", type=float, default=1e-5)
    parser.add_argument("--quantizer-lr", type=float, default=1e-6)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-fused-rotation", action="store_true")
    return parser.parse_args()


def _get_named_module(model, module_name: str):
    module_map = dict(model.named_modules())
    if module_name not in module_map:
        raise KeyError(f"Module `{module_name}` not found.")
    return module_map[module_name]


def _tokenize_calibration_sample(tokenizer, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "input_ids" in sample:
        input_ids = torch.as_tensor(sample["input_ids"], dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        attention_mask = sample.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        else:
            attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    if "messages" in sample:
        rendered = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        tokenized = tokenizer(rendered, add_special_tokens=True, return_tensors="pt")
        return {
            "input_ids": tokenized["input_ids"].to(dtype=torch.long),
            "attention_mask": tokenized.get("attention_mask", torch.ones_like(tokenized["input_ids"])).to(dtype=torch.long),
        }

    if "text" in sample:
        tokenized = tokenizer(sample["text"], add_special_tokens=True, return_tensors="pt")
        return {
            "input_ids": tokenized["input_ids"].to(dtype=torch.long),
            "attention_mask": tokenized.get("attention_mask", torch.ones_like(tokenized["input_ids"])).to(dtype=torch.long),
        }

    raise ValueError(f"Unsupported calibration sample keys: {sorted(sample.keys())}")


def _capture_module_inputs(
    model,
    tokenizer,
    module_names: list[str],
    calibration_dataset: list[dict[str, Any]],
    *,
    max_rows: int,
) -> dict[str, torch.Tensor]:
    module_names_set = set(module_names)
    captured: dict[str, list[torch.Tensor]] = {name: [] for name in module_names}
    captured_rows = {name: 0 for name in module_names}
    hooks = []

    def make_hook(name: str):
        def hook(_module, inputs):
            if not inputs or captured_rows[name] >= max_rows:
                return
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).cpu()
            remaining = max_rows - captured_rows[name]
            if remaining <= 0:
                return
            piece = x[:remaining].contiguous()
            if piece.numel() == 0:
                return
            captured[name].append(piece)
            captured_rows[name] += piece.shape[0]

        return hook

    for name in module_names:
        module = _get_named_module(model, name)
        hooks.append(module.register_forward_pre_hook(make_hook(name)))

    model_device = next(model.parameters()).device
    try:
        for sample in calibration_dataset:
            if all(count >= max_rows for count in captured_rows.values()):
                break
            tokenized = _tokenize_calibration_sample(tokenizer, sample)
            with torch.inference_mode():
                model(
                    input_ids=tokenized["input_ids"].to(device=model_device),
                    attention_mask=tokenized["attention_mask"].to(device=model_device),
                )
    finally:
        for hook in hooks:
            hook.remove()

    flattened: dict[str, torch.Tensor] = {}
    for name in module_names_set:
        pieces = captured[name]
        if not pieces:
            raise RuntimeError(f"Failed to capture calibration activations for `{name}`.")
        flattened[name] = torch.cat(pieces, dim=0)[:max_rows].contiguous()
    return flattened


def _bench_one_case(
    *,
    case: BenchmarkCase,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    inputs: torch.Tensor,
    args: argparse.Namespace,
    run_idx: int,
) -> dict[str, float | str]:
    seed = int(args.seed) + (run_idx * 1000)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(weight.device)

    start = time.perf_counter()
    result = optimize_paroquant_linear(
        weight=weight,
        bias=bias,
        inputs=inputs,
        bits=args.bits,
        group_size=args.group_size,
        sym=True,
        krot=args.krot,
        pair_ratio=args.pair_ratio,
        train_rows=args.train_rows,
        val_rows=args.val_rows,
        batch_size=args.batch_size,
        rotation_epochs=args.rotation_epochs,
        finetune_epochs=args.finetune_epochs,
        rotation_lr=args.rotation_lr,
        weight_lr=args.weight_lr,
        quantizer_lr=args.quantizer_lr,
        seed=seed,
        fused_rotation=not args.no_fused_rotation,
        stage_impl=case.stage_impl,
        pair_impl=case.pair_impl,
        quantizer_impl=case.quantizer_impl,
    )
    wall_s = time.perf_counter() - start
    peak_bytes = 0
    if torch.cuda.is_available():
        peak_bytes = int(torch.cuda.max_memory_allocated(weight.device))
    return {
        "label": case.label,
        "wall_s": wall_s,
        "train_loss": float(result.train_loss),
        "val_loss": float(result.val_loss),
        "peak_gb": peak_bytes / (1024**3),
    }


def _summarize_runs(module_name: str, runs: list[dict[str, float | str]]) -> list[list[str]]:
    baseline = next(run for run in runs if run["label"] == "reference")
    baseline_wall = float(baseline["wall_s"])
    rows = []
    for label in [case.label for case in CASES]:
        selected = [run for run in runs if run["label"] == label]
        wall_values = [float(run["wall_s"]) for run in selected]
        train_values = [float(run["train_loss"]) for run in selected]
        val_values = [float(run["val_loss"]) for run in selected]
        peak_values = [float(run["peak_gb"]) for run in selected]
        median_wall = statistics.median(wall_values)
        rows.append(
            [
                module_name,
                label,
                f"{median_wall:.3f}",
                f"{statistics.mean(wall_values):.3f}",
                f"{baseline_wall / median_wall:.3f}x" if median_wall > 0 else "",
                f"{statistics.mean(train_values):.6f}",
                f"{statistics.mean(val_values):.6f}",
                f"{statistics.mean(peak_values):.3f}",
            ]
        )
    return rows


def main() -> int:
    args = parse_args()
    modules = args.modules or list(DEFAULT_MODULES)
    normalized_dtype = _normalize_model_dtype(args.model_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=False,
        torch_dtype=normalized_dtype,
        low_cpu_mem_usage=True,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    calibration_dataset = load_nm_calibration(args.calibration_rows)

    try:
        captured = _capture_module_inputs(
            model,
            tokenizer,
            modules,
            calibration_dataset,
            max_rows=max(args.capture_rows, args.train_rows + args.val_rows),
        )
        rows: list[list[str]] = []
        for module_name in modules:
            module = _get_named_module(model, module_name)
            weight = module.weight.detach().to(device=module.weight.device, dtype=torch.float32).contiguous()
            bias = None
            if getattr(module, "bias", None) is not None:
                bias = module.bias.detach().to(device=module.weight.device, dtype=torch.float32).contiguous()
            inputs = captured[module_name].to(device=module.weight.device, dtype=torch.float32).contiguous()

            runs: list[dict[str, float | str]] = []
            for run_idx in range(args.repeats):
                for case in CASES:
                    runs.append(
                        _bench_one_case(
                            case=case,
                            weight=weight,
                            bias=bias,
                            inputs=inputs,
                            args=args,
                            run_idx=run_idx,
                        )
                    )
            rows.extend(_summarize_runs(module_name, runs))

        print(
            tabulate(
                rows,
                headers=["module", "case", "median_s", "mean_s", "vs_ref", "train_loss", "val_loss", "peak_gb"],
                tablefmt="grid",
            )
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
