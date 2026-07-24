#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark one GPT-QModel revision's ParoQuant runtime on the complete Qwen3-8B linear shape set."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
from tabulate import tabulate

import gptqmodel
from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear
from gptqmodel.nn_modules.qlinear.paroquant_triton import ParoQuantTritonLinear
from gptqmodel.quantization.paroquant.optimization import build_identity_rotation_buffers


@dataclass(frozen=True)
class Projection:
    name: str
    in_features: int
    out_features: int
    calls_per_layer: int


QWEN3_8B_PROJECTIONS = (
    Projection("q_o", 4096, 4096, 2),
    Projection("k_v", 4096, 1024, 2),
    Projection("gate_up", 4096, 12288, 2),
    Projection("down", 12288, 4096, 1),
)
QWEN3_8B_LAYERS = 36
DEFAULT_ROWS = (1, 2, 4, 8, 16, 32)


def _git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() or None


def _packed_random(shape: tuple[int, ...]) -> torch.Tensor:
    """Generate deterministic packed INT4 words directly without an unpacked KxN temporary."""
    return torch.randint(-(2**31), 2**31 - 1, shape, dtype=torch.int32)


def _make_buffers(
    projection: Projection,
    *,
    dtype: torch.dtype,
    seed: int,
    group_size: int = 128,
    krot: int = 8,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    groups = projection.in_features // group_size
    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=projection.in_features,
        group_size=group_size,
        krot=krot,
        dtype=dtype,
    )
    theta.uniform_(-0.2, 0.2)
    channel_scales.uniform_(0.75, 1.25)
    return {
        "qweight": _packed_random((projection.in_features, projection.out_features // 8)),
        "qzeros": _packed_random((groups, projection.out_features // 8)),
        "scales": ((torch.rand(groups, projection.out_features) * 0.04) + 0.01).to(dtype),
        "bias": (torch.randn(projection.out_features) * 0.1).to(dtype),
        "pairs": pairs,
        "theta": theta,
        "channel_scales": channel_scales,
    }


def _build_module(
    module_cls,
    projection: Projection,
    buffers: dict[str, torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
    group_size: int = 128,
    krot: int = 8,
):
    module = module_cls(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=projection.in_features,
        out_features=projection.out_features,
        bias=True,
        register_buffers=True,
        krot=krot,
    ).to(device=device, dtype=dtype)
    for name, value in buffers.items():
        getattr(module, name).copy_(value.to(device))
    module.post_init()
    module.eval()
    return module


def _stats(samples_us: list[float]) -> dict[str, float]:
    ordered = sorted(samples_us)
    return {
        "p50_us": statistics.median(samples_us),
        "mean_us": statistics.mean(samples_us),
        "p95_us": ordered[int(0.95 * (len(ordered) - 1))],
        "min_us": ordered[0],
        "max_us": ordered[-1],
        "std_us": statistics.stdev(samples_us) if len(samples_us) > 1 else 0.0,
    }


def _benchmark(module, x: torch.Tensor, *, warmup: int, iters: int) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            module(x)
        torch.cuda.synchronize(x.device)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for index in range(iters):
            starts[index].record()
            module(x)
            ends[index].record()
        ends[-1].synchronize()
    return _stats([starts[index].elapsed_time(ends[index]) * 1000.0 for index in range(iters)])


def _selected_plan(module: ParoQuantTritonLinear) -> str:
    plans = sorted(set(module._plan_cache.values()))
    return "/".join(plans) if plans else "uncached"


def _run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cases = []
    table_rows = []

    for projection_index, projection in enumerate(QWEN3_8B_PROJECTIONS):
        buffers = _make_buffers(projection, dtype=dtype, seed=args.seed + projection_index)
        reference = _build_module(ParoLinear, projection, buffers, device=device, dtype=dtype)
        candidate = _build_module(ParoQuantTritonLinear, projection, buffers, device=device, dtype=dtype)
        candidate.paroquant_triton_autotune_enabled = True
        candidate.paroquant_triton_autotune_warmup = args.autotune_warmup
        candidate.paroquant_triton_autotune_iters = args.autotune_iters
        candidate.paroquant_triton_autotune_margin = args.autotune_margin

        for rows in args.rows:
            candidate.clear_autotune()
            torch.manual_seed(args.seed + projection_index * 1000 + rows)
            torch.cuda.manual_seed_all(args.seed + projection_index * 1000 + rows)
            x = torch.randn((1, rows, projection.in_features), device=device, dtype=dtype)
            with torch.inference_mode():
                expected = reference(x)
                actual = candidate(x)
                repeated = candidate(x)
            torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
            torch.testing.assert_close(actual, expected, rtol=args.rtol, atol=args.atol)
            difference = (actual - expected).abs().float()
            reference_mean_abs = expected.abs().float().mean().item()
            relative_mean_abs = difference.mean().item() / max(reference_mean_abs, torch.finfo(torch.float32).tiny)
            if relative_mean_abs > args.max_relative_mean_drift:
                raise AssertionError(
                    f"{projection.name}/M{rows} relative mean drift {relative_mean_abs:.6f} "
                    f"exceeded {args.max_relative_mean_drift:.6f}"
                )

            timing = _benchmark(candidate, x, warmup=args.warmup, iters=args.iters)
            case = {
                "case_key": f"{projection.name}:m{rows}",
                "projection": projection.name,
                "rows": rows,
                "in_features": projection.in_features,
                "out_features": projection.out_features,
                "calls_per_layer": projection.calls_per_layer,
                "selected_plan": _selected_plan(candidate),
                "timing": timing,
                "accuracy": {
                    "mismatched": int(difference.count_nonzero().item()),
                    "max_abs": difference.max().item(),
                    "mean_abs": difference.mean().item(),
                    "reference_mean_abs": reference_mean_abs,
                    "relative_mean_abs": relative_mean_abs,
                    "output_sum": actual.float().sum().item(),
                },
            }
            cases.append(case)
            table_rows.append(
                [
                    projection.name,
                    f"{projection.in_features}->{projection.out_features}",
                    rows,
                    case["selected_plan"],
                    (
                        f"{timing['p50_us']:.3f}/{timing['mean_us']:.3f}/"
                        f"{timing['p95_us']:.3f}"
                    ),
                    (
                        f"{difference.max().item():.6f}/{difference.mean().item():.6f}/"
                        f"{relative_mean_abs:.6f}"
                    ),
                ]
            )
            del x, expected, actual, repeated, difference

        del reference, candidate, buffers
        torch.cuda.empty_cache()

    properties = torch.cuda.get_device_properties(device)
    payload = {
        "label": args.label,
        "git_revision": _git_revision(),
        "package_path": str(Path(gptqmodel.__file__).resolve()),
        "dtype": str(dtype).removeprefix("torch."),
        "device": properties.name,
        "device_uuid": str(properties.uuid),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "sm_count": properties.multi_processor_count,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "triton_version": __import__("triton").__version__,
        "seed": args.seed,
        "warmup": args.warmup,
        "iters": args.iters,
        "autotune_warmup": args.autotune_warmup,
        "autotune_iters": args.autotune_iters,
        "autotune_margin": args.autotune_margin,
        "rows": args.rows,
        "qwen3_8b": {
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "num_hidden_layers": QWEN3_8B_LAYERS,
        },
        "cases": cases,
    }
    print(
        tabulate(
            table_rows,
            headers=("projection", "K->N", "M", "selected plan", "p50/mean/p95 us", "ref max/mean/relative"),
            tablefmt="grid",
        )
    )
    return payload


def _median_case_map(paths: list[Path]) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    by_key: dict[str, list[dict[str, object]]] = {}
    for payload in payloads:
        for case in payload["cases"]:
            by_key.setdefault(case["case_key"], []).append(case)
    aggregated = {}
    for case_key, repeats in by_key.items():
        template = repeats[0]
        timing = {
            metric: statistics.median(repeat["timing"][metric] for repeat in repeats)
            for metric in ("p50_us", "mean_us", "p95_us")
        }
        aggregated[case_key] = {
            **{key: value for key, value in template.items() if key not in {"timing", "accuracy"}},
            "selected_plan": "/".join(sorted({repeat["selected_plan"] for repeat in repeats})),
            "timing": timing,
            "accuracy": {
                "max_abs": max(repeat["accuracy"]["max_abs"] for repeat in repeats),
                "mean_abs": max(repeat["accuracy"]["mean_abs"] for repeat in repeats),
                "relative_mean_abs": max(repeat["accuracy"]["relative_mean_abs"] for repeat in repeats),
            },
        }
    return aggregated, payloads


def _compare(args: argparse.Namespace) -> dict[str, object]:
    baseline, baseline_payloads = _median_case_map(args.baseline_json)
    candidate, candidate_payloads = _median_case_map(args.candidate_json)
    if baseline.keys() != candidate.keys():
        raise ValueError("baseline and candidate JSON files contain different case sets")
    case_rows = []
    case_results = []
    for case_key in sorted(baseline, key=lambda key: (baseline[key]["rows"], key)):
        base_case = baseline[case_key]
        candidate_case = candidate[case_key]
        p50_speedup = base_case["timing"]["p50_us"] / candidate_case["timing"]["p50_us"]
        mean_speedup = base_case["timing"]["mean_us"] / candidate_case["timing"]["mean_us"]
        case_rows.append(
            [
                base_case["projection"],
                f"{base_case['in_features']}->{base_case['out_features']}",
                base_case["rows"],
                base_case["selected_plan"],
                candidate_case["selected_plan"],
                f"{base_case['timing']['p50_us']:.3f}",
                f"{candidate_case['timing']['p50_us']:.3f}",
                f"{p50_speedup:.3f}x",
                f"{base_case['timing']['mean_us']:.3f}",
                f"{candidate_case['timing']['mean_us']:.3f}",
                f"{mean_speedup:.3f}x",
            ]
        )
        case_results.append(
            {
                "case_key": case_key,
                "baseline": base_case,
                "candidate": candidate_case,
                "p50_speedup": p50_speedup,
                "mean_speedup": mean_speedup,
            }
        )

    aggregate_rows = []
    aggregate_results = []
    all_rows = sorted({case["rows"] for case in baseline.values()})
    for rows in all_rows:
        aggregate = {}
        for metric in ("p50_us", "mean_us"):
            baseline_layer_us = sum(
                case["timing"][metric] * case["calls_per_layer"]
                for case in baseline.values()
                if case["rows"] == rows
            )
            candidate_layer_us = sum(
                case["timing"][metric] * case["calls_per_layer"]
                for case in candidate.values()
                if case["rows"] == rows
            )
            prefix = metric.removesuffix("_us")
            aggregate[f"baseline_{prefix}_layer_linear_us"] = baseline_layer_us
            aggregate[f"candidate_{prefix}_layer_linear_us"] = candidate_layer_us
            aggregate[f"baseline_{prefix}_36_layer_linear_us"] = baseline_layer_us * QWEN3_8B_LAYERS
            aggregate[f"candidate_{prefix}_36_layer_linear_us"] = candidate_layer_us * QWEN3_8B_LAYERS
            aggregate[f"{prefix}_speedup"] = baseline_layer_us / candidate_layer_us
            aggregate[f"baseline_projected_{prefix}_linear_tokens_per_second"] = (
                rows * 1e6 / (baseline_layer_us * QWEN3_8B_LAYERS)
            )
            aggregate[f"candidate_projected_{prefix}_linear_tokens_per_second"] = (
                rows * 1e6 / (candidate_layer_us * QWEN3_8B_LAYERS)
            )
        aggregate_rows.append(
            [
                rows,
                f"{aggregate['baseline_p50_36_layer_linear_us']:.3f}",
                f"{aggregate['candidate_p50_36_layer_linear_us']:.3f}",
                f"{aggregate['p50_speedup']:.3f}x",
                f"{aggregate['baseline_projected_p50_linear_tokens_per_second']:.1f}",
                f"{aggregate['candidate_projected_p50_linear_tokens_per_second']:.1f}",
                f"{aggregate['baseline_mean_36_layer_linear_us']:.3f}",
                f"{aggregate['candidate_mean_36_layer_linear_us']:.3f}",
                f"{aggregate['mean_speedup']:.3f}x",
            ]
        )
        aggregate_results.append({"rows": rows, **aggregate})

    print("Per projection")
    print(
        tabulate(
            case_rows,
            headers=(
                "projection",
                "K->N",
                "M",
                "main plan",
                "current plan",
                "main p50 us",
                "current p50 us",
                "p50 speedup",
                "main mean us",
                "current mean us",
                "mean speedup",
            ),
            tablefmt="grid",
        )
    )
    print()
    print("Weighted Qwen3-8B linear stack (2x Q/O, 2x K/V, 2x gate/up, 1x down; 36 layers)")
    print(
        tabulate(
            aggregate_rows,
            headers=(
                "M",
                "main p50 36L us",
                "current p50 36L us",
                "p50 speedup",
                "main p50 tok/s",
                "current p50 tok/s",
                "main mean 36L us",
                "current mean 36L us",
                "mean speedup",
            ),
            tablefmt="grid",
        )
    )
    return {
        "baseline_label": baseline_payloads[0]["label"],
        "candidate_label": candidate_payloads[0]["label"],
        "baseline_revisions": sorted({payload["git_revision"] for payload in baseline_payloads}),
        "candidate_revisions": sorted({payload["git_revision"] for payload in candidate_payloads}),
        "baseline_repeats": len(baseline_payloads),
        "candidate_repeats": len(candidate_payloads),
        "cases": case_results,
        "aggregates": aggregate_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--rows", type=int, action="append")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--autotune-warmup", type=int, default=10)
    parser.add_argument("--autotune-iters", type=int, default=20)
    parser.add_argument("--autotune-margin", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.25)
    parser.add_argument("--max-relative-mean-drift", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline-json", type=Path, action="append")
    parser.add_argument("--candidate-json", type=Path, action="append")
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    comparing = args.baseline_json is not None or args.candidate_json is not None
    if comparing:
        if not args.baseline_json or not args.candidate_json:
            raise ValueError("comparison requires at least one baseline and candidate JSON")
        payload = _compare(args)
    else:
        if args.label is None:
            raise ValueError("--label is required for a benchmark run")
        args.rows = DEFAULT_ROWS if args.rows is None else tuple(args.rows)
        if any(rows <= 0 for rows in args.rows):
            raise ValueError("rows must be positive")
        if args.warmup < 0 or args.iters <= 0:
            raise ValueError("warmup must be non-negative and iters must be positive")
        payload = _run_benchmark(args)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print()
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
