#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from typing import Any, Literal

import torch
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel.quantization.paroquant.optimization import (
    optimize_paroquant_linear,
    _build_random_rotation_buffers_cached_cpu,
    _clear_random_rotation_buffers_cache,
    _warm_random_rotation_buffers_cache,
)
from gptqmodel.utils.paroquant_benchmark import (
    _normalize_model_dtype,
    load_nm_calibration,
)


CacheStrategy = Literal["miss", "fixed", "fixed_preload"]


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
    BenchmarkCase("all_fast", "fast", "fast", "fast"),
)

DEFAULT_MODEL = "/monster/data/model/Llama-3.2-1B-Instruct"
DEFAULT_MODULES = (
    "self_attn.q_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.down_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark ParoQuant pair-cache behavior on real calibration activations, "
            "across explicit cache strategies."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layers", type=int, default=3, help="Number of decoder layers to evaluate from layer 0.")
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
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cache-strategy",
        choices=("miss", "fixed", "fixed_preload"),
        default="fixed_preload",
        help=(
            "miss: force one pair-cache miss per module call; "
            "fixed: keep seed fixed but do not preload schedule cache; "
            "fixed_preload: keep seed fixed and preload per-config schedules once."
        ),
    )
    parser.add_argument(
        "--no-fused-rotation",
        action="store_true",
        help="Pass fused_rotation=False to optimizer.",
    )
    return parser.parse_args()


def _get_named_module(model, module_name: str):
    module_map = dict(model.named_modules())
    if module_name not in module_map:
        raise KeyError(f"Module `{module_name}` not found.")
    return module_map[module_name]


def _module_list_for_layers(layer_count: int, module_suffixes: tuple[str, ...]) -> list[str]:
    return [f"model.layers.{layer_idx}.{suffix}" for layer_idx in range(layer_count) for suffix in module_suffixes]


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


def _precompute_pair_cache(
    *,
    modules: list[str],
    module_infos: dict[str, tuple[int, int, int, float, int]],
    cache_keys: set[tuple[int, int, int, float, int]],
    device: torch.device,
) -> None:
    _clear_random_rotation_buffers_cache()
    for in_features, group_size, krot, pair_ratio, seed in sorted(cache_keys):
        _warm_random_rotation_buffers_cache(
            in_features=in_features,
            group_size=group_size,
            krot=krot,
            pair_ratio=pair_ratio,
            seed=seed,
        )


def _run_case(
    *,
    case: BenchmarkCase,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    inputs: torch.Tensor,
    args: argparse.Namespace,
    run_idx: int,
    case_idx: int,
    module_name: str,
    module_seq: int,
    cache_strategy: CacheStrategy,
    cache_key: tuple[int, int, int, float, int],
) -> dict[str, Any]:
    base_seed = int(args.seed)
    seed = base_seed if cache_strategy != "miss" else base_seed + case_idx * 97 + run_idx * 1000 + module_seq * 17

    if cache_strategy == "miss" and case.pair_impl == "fast":
        _clear_random_rotation_buffers_cache()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    before = None
    if case.pair_impl == "fast":
        before = _build_random_rotation_buffers_cached_cpu.cache_info()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    if start is not None:
        torch.cuda.synchronize()
        start.record()
    else:
        t0 = torch.cuda.Event(enable_timing=False)
        t0.record() if torch.cuda.is_available() else None
        t0.synchronize() if torch.cuda.is_available() else None

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

    if torch.cuda.is_available():
        end.record()
        torch.cuda.synchronize()
        wall_s = float(start.elapsed_time(end)) / 1e3
    else:
        end_fallback = torch.cuda.Event(enable_timing=False) if torch.cuda.is_available() else None
        wall_s = float(end_fallback.elapsed_time(t0) / 1e3) if end_fallback is not None else 0.0

    after = None
    cache_hit = None
    if case.pair_impl == "fast":
        after = _build_random_rotation_buffers_cached_cpu.cache_info()
        if before is not None and after is not None:
            cache_hit = (after.hits - before.hits) > (after.misses - before.misses)
    peak_bytes = int(torch.cuda.max_memory_allocated(weight.device)) if torch.cuda.is_available() else 0

    return {
        "label": case.label,
        "module": module_name,
        "wall_s": wall_s,
        "train_loss": float(result.train_loss),
        "val_loss": float(result.val_loss),
        "peak_gb": peak_bytes / (1024**3),
        "cache_hit": None if cache_hit is None else cache_hit,
        "cache_miss": None if cache_hit is None else (not cache_hit),
        "cache_key": cache_key,
    }


def _collect_summary(
    *,
    module_label: str,
    module_results: list[dict[str, Any]],
) -> list[str]:
    """
    Return rows keyed by module-label (e.g. self_attn.q_proj at layer0..N).
    """
    ref = [r for r in module_results if r["label"] == "reference"]
    if not ref:
        return []
    ref_wall = statistics.median([float(r["wall_s"]) for r in ref])
    rows: list[str] = []

    for label in [case.label for case in CASES]:
        selected = [r for r in module_results if r["label"] == label]
        if not selected:
            continue
        wall_values = [float(r["wall_s"]) for r in selected]
        cache_hits = [r["cache_hit"] for r in selected if r["cache_hit"] is not None]
        median_wall = statistics.median(wall_values)
        rows.append(
            (
                module_label,
                label,
                f"{median_wall:.3f}",
                f"{ref_wall / median_wall:.3f}x" if median_wall > 0 else "",
                f"{statistics.mean(wall_values):.3f}",
                f"{sum(1 for v in cache_hits if v)}/{len(cache_hits)}"
                if cache_hits else "N/A",
            )
        )
    return rows


def main() -> int:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    model.to(device)
    model.eval()

    modules = _module_list_for_layers(args.layers, DEFAULT_MODULES)
    calibration_dataset = load_nm_calibration(args.calibration_rows)
    max_rows = max(args.capture_rows, args.train_rows + args.val_rows)

    try:
        captured = _capture_module_inputs(model, tokenizer, modules, calibration_dataset, max_rows=max_rows)
        cache_mode = args.cache_strategy

        if cache_mode == "fixed_preload":
            unique_cache_keys = set()
            for module_name in modules:
                module = _get_named_module(model, module_name)
                cache_seed = int(args.seed)
                unique_cache_keys.add(
                    (
                        module.weight.shape[1],
                        args.group_size,
                        args.krot,
                        float(args.pair_ratio),
                        cache_seed,
                    )
                )
            _precompute_pair_cache(
                modules=modules,
                module_infos={},
                cache_keys=unique_cache_keys,
                device=device,
            )
        elif cache_mode in {"fixed", "miss"}:
            _clear_random_rotation_buffers_cache()

        runs: list[dict[str, Any]] = []
        for run_idx in range(args.repeats):
            for module_idx, module_name in enumerate(modules):
                module = _get_named_module(model, module_name)
                weight = module.weight.detach().to(device=module.weight.device, dtype=torch.float32).contiguous()
                bias = module.bias.detach().to(device=module.weight.device, dtype=torch.float32).contiguous() if getattr(module, "bias", None) is not None else None
                inputs = captured[module_name].to(device=module.weight.device, dtype=torch.float32).contiguous()
                cache_seed = int(args.seed)
                cache_key = (weight.shape[1], args.group_size, args.krot, float(args.pair_ratio), cache_seed)

                for case_idx, case in enumerate(CASES):
                    runs.append(
                        _run_case(
                            case=case,
                            weight=weight,
                            bias=bias,
                            inputs=inputs,
                            args=args,
                            run_idx=run_idx,
                            case_idx=case_idx,
                            module_name=module_name,
                            module_seq=module_idx,
                            cache_strategy=cache_mode,
                            cache_key=cache_key,
                        )
                    )

        # summarize by short module name (relative to layer block) and full case
        by_rel = {}
        for run in runs:
            rel = ".".join(run["module"].split(".")[-2:])
            by_rel.setdefault(rel, []).append(run)

        rows = []
        for rel, rel_runs in sorted(by_rel.items()):
            rows.extend(_collect_summary(module_label=rel, module_results=rel_runs))

        print(f"\ncache_strategy={cache_mode}, runs={args.repeats}, layers={args.layers}")
        print(
            tabulate(
                rows,
                headers=["module", "case", "median_s", "vs_ref", "mean_s", "pair_cache_hit_count"],
                tablefmt="grid",
            )
        )

        print(f"\nCache Hits Detail (cache_strategy={cache_mode})")
        details = []
        for rel, rel_runs in sorted(by_rel.items()):
            rel_pairs = [r for r in rel_runs if r["label"] in {"pair_fast", "all_fast"}]
            if not rel_pairs:
                continue
            cold = [float(r["wall_s"]) for r in rel_pairs if r["cache_hit"] is False]
            warm = [float(r["wall_s"]) for r in rel_pairs if r["cache_hit"] is True]
            total = [r["cache_hit"] for r in rel_pairs if r["cache_hit"] is not None]
            if total:
                hit_rate = 100.0 * sum(1 for v in total if v) / len(total)
            else:
                hit_rate = None
            details.append(
                [
                    rel,
                    cache_mode,
                    f"{sum(1 for v in total if v == True)}/{len(total)}" if total else "N/A",
                    f"{(sum(1 for v in total if v is False) / max(1, len(total)) * 100.0):.1f}%" if total else "N/A",
                    f"{statistics.mean(cold):.3f}" if cold else "N/A",
                    f"{statistics.mean(warm):.3f}" if warm else "N/A",
                ]
            )

        print(
            tabulate(
                details,
                headers=[
                    "module",
                    "cache_strategy",
                    "pair_cache_hits",
                    "miss_rate_%",
                    "cold_ms (miss)",
                    "warm_ms (hit)",
                ],
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
