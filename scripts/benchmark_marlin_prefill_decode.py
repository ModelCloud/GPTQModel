#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark packed-weight Marlin prefill and ordinary Marlin decode."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear


DEFAULT_MODEL = (
    "/monster/data/model/Llama-3.2-1B-Instruct/"
    "gptq_4bits_10-26_15-59-54_maxlen2048_ns128_descFalse_damp0.005"
)


def _parse_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive integers"
        )
    return values


def _dtype(value: str) -> torch.dtype:
    return torch.float16 if value == "fp16" else torch.bfloat16


def _sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _load_model(
    path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, Any, Any]:
    wrapper = GPTQModel.load(
        path,
        backend=BACKEND.MARLIN,
        device=str(device),
        dtype=dtype,
        attn_implementation="eager",
    )
    model = getattr(wrapper, "model", wrapper).eval()
    tokenizer = getattr(wrapper, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("the loaded GPTQModel did not expose its tokenizer")
    return wrapper, model, tokenizer


def _projection_inventory(
    model: Any,
) -> tuple[dict[tuple[int, int], MarlinLinear], Counter[tuple[int, int]]]:
    representatives: dict[tuple[int, int], MarlinLinear] = {}
    counts: Counter[tuple[int, int]] = Counter()
    for module in model.modules():
        if isinstance(module, MarlinLinear):
            shape = (module.in_features, module.out_features)
            representatives.setdefault(shape, module)
            counts[shape] += 1
    if not representatives:
        raise RuntimeError("the model contains no MarlinLinear modules")
    return representatives, counts


def _select_route(modules: list[MarlinLinear], enabled: bool) -> None:
    for module in modules:
        module.packed_prefill = enabled


def _time_projection(
    module: MarlinLinear,
    x: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
    samples: int,
    device: torch.device,
) -> tuple[float, torch.Tensor, list[float]]:
    with torch.inference_mode():
        for _ in range(warmup):
            module(x)
        _sync(device)
        timings = []
        output = module(x)
        for _ in range(samples):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                output = module(x)
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end) / iterations)
    return statistics.median(timings), output.detach().cpu(), timings


def _benchmark_layer_ab(
    representatives: dict[tuple[int, int], MarlinLinear],
    counts: Counter[tuple[int, int]],
    modules: list[MarlinLinear],
    *,
    rows_values: list[int],
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iterations: int,
    samples: int,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    results = []
    candidate_outputs = {}
    with torch.inference_mode():
        for shape_index, ((in_features, out_features), module) in enumerate(
            sorted(representatives.items())
        ):
            for rows in rows_values:
                generator = torch.Generator(device=device)
                generator.manual_seed(1729 + shape_index * 1000 + rows)
                x = torch.randn(
                    (rows, in_features),
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )

                _select_route(modules, False)
                baseline_ms, baseline_output, baseline_samples = _time_projection(
                    module,
                    x,
                    warmup=warmup,
                    iterations=iterations,
                    samples=samples,
                    device=device,
                )
                _select_route(modules, True)
                candidate_ms, candidate_output, candidate_samples = _time_projection(
                    module,
                    x,
                    warmup=warmup,
                    iterations=iterations,
                    samples=samples,
                    device=device,
                )

                delta = (candidate_output.float() - baseline_output.float()).abs()
                tensor_name = f"m{rows}_k{in_features}_n{out_features}"
                candidate_outputs[tensor_name] = candidate_output
                results.append(
                    {
                        "m": rows,
                        "k": in_features,
                        "n": out_features,
                        "model_projection_count": counts[(in_features, out_features)],
                        "baseline_ms_median": baseline_ms,
                        "packed_prefill_ms_median": candidate_ms,
                        "speedup": baseline_ms / candidate_ms,
                        "baseline_samples_ms": baseline_samples,
                        "packed_prefill_samples_ms": candidate_samples,
                        "max_abs": delta.max().item(),
                        "mean_abs": delta.mean().item(),
                        "allclose_atol_0_05_rtol_0_005": torch.allclose(
                            candidate_output,
                            baseline_output,
                            atol=0.05,
                            rtol=0.005,
                        ),
                    }
                )
    return results, candidate_outputs


def _build_prompt(
    tokenizer: Any,
    *,
    prompt_tokens: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    seed_text = (
        "Explain how a weight-only quantized transformer handles prompt prefill "
        "and autoregressive decode, including matrix shapes and memory traffic. "
    )
    text = seed_text
    while len(tokenizer(text, add_special_tokens=False)["input_ids"]) < prompt_tokens:
        text += seed_text
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=prompt_tokens,
        padding="max_length",
        return_tensors="pt",
    )
    return {name: tensor.to(device) for name, tensor in encoded.items()}


def _prefill_decode_once(
    model: Any,
    inputs: dict[str, torch.Tensor],
    *,
    decode_tokens: int,
    device: torch.device,
) -> tuple[dict[str, float], list[int]]:
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)
    decode_start = torch.cuda.Event(enable_timing=True)
    decode_end = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        prefill_start.record()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        prefill_end.record()
        prefill_end.synchronize()

        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
        generated = []
        decode_attention_mask = attention_mask
        decode_start.record()
        for _ in range(decode_tokens):
            generated.extend(next_token.flatten().tolist())
            decode_attention_mask = torch.cat(
                (
                    decode_attention_mask,
                    torch.ones(
                        (decode_attention_mask.shape[0], 1),
                        dtype=decode_attention_mask.dtype,
                        device=device,
                    ),
                ),
                dim=1,
            )
            outputs = model(
                input_ids=next_token,
                attention_mask=decode_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
        decode_end.record()
        decode_end.synchronize()

    prefill_ms = prefill_start.elapsed_time(prefill_end)
    decode_ms = decode_start.elapsed_time(decode_end)
    return (
        {
            "prefill_ms": prefill_ms,
            "prefill_tps": input_ids.numel() * 1000.0 / prefill_ms,
            "decode_ms_per_token": decode_ms / decode_tokens,
            "decode_tps": decode_tokens * 1000.0 / decode_ms,
        },
        generated,
    )


def _benchmark_end_to_end_ab(
    model: Any,
    tokenizer: Any,
    modules: list[MarlinLinear],
    *,
    prompt_tokens: int,
    decode_tokens: int,
    warmup: int,
    runs: int,
    device: torch.device,
) -> dict[str, Any]:
    inputs = _build_prompt(tokenizer, prompt_tokens=prompt_tokens, device=device)
    samples: dict[str, list[dict[str, float]]] = {
        "marlin": [],
        "packed_prefill": [],
    }
    generated: dict[str, list[int]] = {}

    for label, enabled in (("marlin", False), ("packed_prefill", True)):
        _select_route(modules, enabled)
        for _ in range(warmup):
            _prefill_decode_once(
                model,
                inputs,
                decode_tokens=decode_tokens,
                device=device,
            )

    labels = (("marlin", False), ("packed_prefill", True))
    for run_index in range(runs):
        order = labels if run_index % 2 == 0 else tuple(reversed(labels))
        for label, enabled in order:
            _select_route(modules, enabled)
            sample, token_ids = _prefill_decode_once(
                model,
                inputs,
                decode_tokens=decode_tokens,
                device=device,
            )
            samples[label].append(sample)
            generated[label] = token_ids

    summaries = {}
    for label, values in samples.items():
        summaries[label] = {
            "prefill_ms_median": statistics.median(
                item["prefill_ms"] for item in values
            ),
            "prefill_tps_median": statistics.median(
                item["prefill_tps"] for item in values
            ),
            "decode_ms_per_token_median": statistics.median(
                item["decode_ms_per_token"] for item in values
            ),
            "decode_tps_median": statistics.median(
                item["decode_tps"] for item in values
            ),
            "generated_token_ids": generated[label],
            "samples": values,
        }

    paired_prefill = [
        baseline["prefill_ms"] / candidate["prefill_ms"]
        for baseline, candidate in zip(samples["marlin"], samples["packed_prefill"])
    ]
    paired_decode = [
        baseline["decode_ms_per_token"] / candidate["decode_ms_per_token"]
        for baseline, candidate in zip(samples["marlin"], samples["packed_prefill"])
    ]
    return {
        "prompt_tokens": prompt_tokens,
        "decode_tokens": decode_tokens,
        **summaries,
        "prefill_speedup_from_medians": (
            summaries["marlin"]["prefill_ms_median"]
            / summaries["packed_prefill"]["prefill_ms_median"]
        ),
        "prefill_paired_speedup_median": statistics.median(paired_prefill),
        "decode_paired_speedup_median": statistics.median(paired_decode),
        "packed_prefill_wins": sum(
            candidate["prefill_ms"] < baseline["prefill_ms"]
            for baseline, candidate in zip(samples["marlin"], samples["packed_prefill"])
        ),
        "generated_tokens_equal": (generated["marlin"] == generated["packed_prefill"]),
    }


def _compare_tensors(
    actual: dict[str, torch.Tensor],
    reference_path: Path,
) -> dict[str, dict[str, float | bool]]:
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    comparisons = {}
    for name, output in actual.items():
        expected = reference[name]
        delta = (output.float() - expected.float()).abs()
        comparisons[name] = {
            "max_abs": delta.max().item(),
            "mean_abs": delta.mean().item(),
            "allclose_atol_0_05_rtol_0_005": torch.allclose(
                output,
                expected,
                atol=0.05,
                rtol=0.005,
            ),
        }
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--m-values",
        type=_parse_ints,
        default=_parse_ints("1,128,256,512,1024,2048"),
    )
    parser.add_argument(
        "--prompt-lengths",
        type=_parse_ints,
        default=_parse_ints("128,512,1024,2048"),
    )
    parser.add_argument("--decode-tokens", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--layer-samples", type=int, default=9)
    parser.add_argument("--model-warmup", type=int, default=3)
    parser.add_argument("--model-runs", type=int, default=31)
    parser.add_argument("--packed-prefill-min-rows", type=int)
    parser.add_argument("--packed-prefill-config", type=int)
    parser.add_argument("--skip-layer", action="store_true")
    parser.add_argument("--skip-end-to-end", action="store_true")
    parser.add_argument("--tensor-out", type=Path)
    parser.add_argument("--reference-tensors", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if args.packed_prefill_min_rows is not None:
        if args.packed_prefill_min_rows < 1:
            parser.error("--packed-prefill-min-rows must be positive")
        os.environ["GPTQMODEL_MARLIN_PACKED_PREFILL_MIN_ROWS"] = str(
            args.packed_prefill_min_rows
        )
    if args.packed_prefill_config is not None:
        if not 0 <= args.packed_prefill_config <= 4:
            parser.error("--packed-prefill-config must be between 0 and 4")
        os.environ["GPTQMODEL_MARLIN_PACKED_PREFILL_CONFIG"] = str(
            args.packed_prefill_config
        )

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Marlin benchmarking requires a CUDA device")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before_load = torch.cuda.memory_allocated(device)
    wrapper, model, tokenizer = _load_model(
        args.model,
        device=device,
        dtype=_dtype(args.dtype),
    )
    properties = torch.cuda.get_device_properties(device)
    modules = [module for module in model.modules() if isinstance(module, MarlinLinear)]
    representatives, counts = _projection_inventory(model)

    packed_weight_bytes = sum(
        module.qweight.numel() * module.qweight.element_size() for module in modules
    )
    payload: dict[str, Any] = {
        "model": args.model,
        "device": str(device),
        "device_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "multiprocessor_count": properties.multi_processor_count,
        "dtype": args.dtype,
        "dispatch_environment": {
            "packed_prefill": os.environ.get(
                "GPTQMODEL_MARLIN_PACKED_PREFILL",
                "<unset: automatic>",
            ),
            "packed_prefill_effective_on_load": all(
                module.packed_prefill for module in modules
            ),
            "packed_prefill_min_rows": os.environ.get(
                "GPTQMODEL_MARLIN_PACKED_PREFILL_MIN_ROWS",
                "1024",
            ),
            "packed_prefill_config": os.environ.get(
                "GPTQMODEL_MARLIN_PACKED_PREFILL_CONFIG",
                "0",
            ),
        },
        "marlin_module_count": len(modules),
        "projection_counts": {
            f"k{k}_n{n}": count for (k, n), count in sorted(counts.items())
        },
        "packed_qweight_bytes": packed_weight_bytes,
        "persistent_prefill_cache_bytes": 0,
        "model_allocated_bytes": (
            torch.cuda.memory_allocated(device) - allocated_before_load
        ),
        "model_load_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) - allocated_before_load
        ),
    }

    output_tensors: dict[str, torch.Tensor] = {}
    if not args.skip_layer:
        layer_results, output_tensors = _benchmark_layer_ab(
            representatives,
            counts,
            modules,
            rows_values=args.m_values,
            dtype=_dtype(args.dtype),
            device=device,
            warmup=args.warmup,
            iterations=args.iterations,
            samples=args.layer_samples,
        )
        payload["layer_results"] = layer_results
        payload["weighted_projection_ms"] = {
            str(rows): {
                "marlin": sum(
                    item["baseline_ms_median"] * item["model_projection_count"]
                    for item in layer_results
                    if item["m"] == rows
                ),
                "packed_prefill": sum(
                    item["packed_prefill_ms_median"] * item["model_projection_count"]
                    for item in layer_results
                    if item["m"] == rows
                ),
            }
            for rows in args.m_values
        }

    if not args.skip_end_to_end:
        payload["end_to_end_ab"] = [
            _benchmark_end_to_end_ab(
                model,
                tokenizer,
                modules,
                prompt_tokens=prompt_tokens,
                decode_tokens=args.decode_tokens,
                warmup=args.model_warmup,
                runs=args.model_runs,
                device=device,
            )
            for prompt_tokens in args.prompt_lengths
        ]

    payload["execution_peak_allocated_bytes"] = (
        torch.cuda.max_memory_allocated(device) - allocated_before_load
    )
    if args.tensor_out is not None:
        args.tensor_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(output_tensors, args.tensor_out)
        payload["tensor_out"] = str(args.tensor_out)
    if args.reference_tensors is not None:
        payload["tensor_comparison"] = _compare_tensors(
            output_tensors,
            args.reference_tensors,
        )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))

    close = getattr(wrapper, "close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    main()
