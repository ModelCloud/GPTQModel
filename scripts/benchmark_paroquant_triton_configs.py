#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Compare ParoQuant mega-kernel launch configurations with paired CUDA-graph timing."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from tabulate import tabulate

from benchmark_paroquant_triton_ab import BenchCase, _build_module, _make_quant_buffers
from gptqmodel.nn_modules.qlinear.paroquant_triton import ParoQuantTritonLinear
from gptqmodel.quantization.paroquant.modules.triton.gemm import (
    FP32_ACCUM,
    PAROQUANT_MEGAKERNEL_GROUP_SIZE,
    _paroquant_rotation_gemm_triton,
    get_same_device_cm,
    paroquant_rotation_gemm_splitk_kernel,
    paroquant_rotation_gemm_splitk_two_n_tiles_pair_counter_kernel,
    paroquant_rotation_gemm_splitk_two_n_tiles_kernel,
)


@dataclass(frozen=True)
class Variant:
    name: str
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int
    partner_dtype: str
    loop_unroll_factor: int
    explicit_fma: bool
    prefetch_first_partner: bool
    prefetch_packed_weight: bool
    split_k: int
    output_tiles_per_cta: int
    maxnreg: int | None
    pair_counter: bool
    atomic_counter_reset: bool
    prefetch_first_weight: bool


def _parse_variant(value: str) -> Variant:
    fields = value.split(":")
    if len(fields) not in {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}:
        raise argparse.ArgumentTypeError(
            "variant must be "
            "NAME:BLOCK_M:BLOCK_N:WARPS:STAGES:PARTNER_DTYPE"
            "[:LOOP_UNROLL[:EXPLICIT_FMA[:PREFETCH_FIRST[:PREFETCH_WEIGHT"
            "[:SPLIT_K[:OUTPUT_TILES[:MAXNREG[:PAIR_COUNTER[:ATOMIC_RESET"
            "[:PREFETCH_FIRST_WEIGHT]]]]]]]]]]"
        )
    name, block_m, block_n, num_warps, num_stages, partner_dtype = fields[:6]
    if partner_dtype not in {"global32", "local16", "local8"}:
        raise argparse.ArgumentTypeError("partner dtype must be global32, local16, or local8")
    try:
        loop_unroll_factor = int(fields[6]) if len(fields) >= 7 else 1
        explicit_fma_value = int(fields[7]) if len(fields) >= 8 else 0
        prefetch_first_value = int(fields[8]) if len(fields) >= 9 else 0
        prefetch_weight_value = int(fields[9]) if len(fields) >= 10 else 0
        split_k = int(fields[10]) if len(fields) >= 11 else 1
        output_tiles_per_cta = int(fields[11]) if len(fields) >= 12 else 1
        maxnreg = int(fields[12]) if len(fields) >= 13 else None
        pair_counter_value = int(fields[13]) if len(fields) >= 14 else 0
        atomic_reset_value = int(fields[14]) if len(fields) >= 15 else pair_counter_value
        prefetch_first_weight_value = int(fields[15]) if len(fields) == 16 else pair_counter_value
        if loop_unroll_factor < 1:
            raise ValueError("loop unroll factor must be positive")
        if explicit_fma_value not in {0, 1}:
            raise ValueError("explicit FMA flag must be 0 or 1")
        if prefetch_first_value not in {0, 1}:
            raise ValueError("prefetch-first flag must be 0 or 1")
        if prefetch_weight_value not in {0, 1}:
            raise ValueError("prefetch-weight flag must be 0 or 1")
        if split_k < 1:
            raise ValueError("split-K factor must be positive")
        if output_tiles_per_cta not in {1, 2}:
            raise ValueError("output tiles per CTA must be 1 or 2")
        if maxnreg is not None and maxnreg < 1:
            raise ValueError("maximum register count must be positive")
        if pair_counter_value not in {0, 1}:
            raise ValueError("pair-counter flag must be 0 or 1")
        if atomic_reset_value not in {0, 1}:
            raise ValueError("atomic-reset flag must be 0 or 1")
        if prefetch_first_weight_value not in {0, 1}:
            raise ValueError("prefetch-first-weight flag must be 0 or 1")
        if pair_counter_value and (split_k == 1 or output_tiles_per_cta != 2):
            raise ValueError("pair-counter variants require split K and two output tiles per CTA")
        if len(fields) >= 15 and not pair_counter_value:
            raise ValueError("explicit paired-kernel flags require a pair-counter variant")
        return Variant(
            name=name,
            block_m=int(block_m),
            block_n=int(block_n),
            num_warps=int(num_warps),
            num_stages=int(num_stages),
            partner_dtype=partner_dtype,
            loop_unroll_factor=loop_unroll_factor,
            explicit_fma=bool(explicit_fma_value),
            prefetch_first_partner=bool(prefetch_first_value),
            prefetch_packed_weight=bool(prefetch_weight_value),
            split_k=split_k,
            output_tiles_per_cta=output_tiles_per_cta,
            maxnreg=maxnreg,
            pair_counter=bool(pair_counter_value),
            atomic_counter_reset=bool(atomic_reset_value),
            prefetch_first_weight=bool(prefetch_first_weight_value),
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid numeric variant field: {exc}") from exc


def _dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "fp16" else torch.bfloat16


def _parse_exact_pair(value: str) -> tuple[str, str]:
    try:
        baseline, candidate = value.split(":", maxsplit=1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("exact pair must be BASELINE:CANDIDATE") from exc
    if not baseline or not candidate or baseline == candidate:
        raise argparse.ArgumentTypeError("exact pair must name two distinct variants")
    return baseline, candidate


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


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def _partner_for_variant(module: ParoQuantTritonLinear, variant: Variant) -> torch.Tensor:
    if module._megakernel_partner is None or module._megakernel_decode_partner is None:
        raise RuntimeError("rotation metadata was not initialized")
    if variant.partner_dtype == "global32":
        return module._megakernel_partner
    local_partner = torch.remainder(module._megakernel_partner, module.group_size)
    if variant.partner_dtype == "local16":
        return local_partner.to(torch.int16)
    return local_partner.to(torch.int8)


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda", args.device)
    dtype = _dtype(args.dtype)
    case = BenchCase(
        case_id=f"config_m{args.m}_k{args.k}_n{args.n}",
        batch=1,
        seq=args.m,
        in_features=args.k,
        out_features=args.n,
        group_size=128,
        krot=args.krot,
    )
    buffers = _make_quant_buffers(case, dtype=dtype)
    module = _build_module(ParoQuantTritonLinear, case, buffers, device, dtype=dtype)
    input_tensor = torch.randn((args.m, args.k), device=device, dtype=dtype)

    with torch.inference_mode():
        metadata = module._megakernel_rotation_metadata(input_tensor, decode=args.m <= 8)
        if metadata is None:
            raise RuntimeError("rotation metadata is unavailable")
        _, cos, sin, channel_scales = metadata
        module._ensure_runtime_dtype(device=device, dtype=dtype)

        functions = {}
        scratch = {}
        for variant in args.variant:
            partner = _partner_for_variant(module, variant)
            if variant.split_k > 1:
                k_blocks_per_split = (
                    args.k // PAROQUANT_MEGAKERNEL_GROUP_SIZE
                ) // variant.split_k
                num_pid_n = (args.n + variant.block_n - 1) // variant.block_n
                if variant.output_tiles_per_cta == 2 and (
                    k_blocks_per_split != 1 or num_pid_n % 2 != 0
                ):
                    raise ValueError(
                        "two-output-tile split-K variants require one K block per split and an even output-tile count"
                    )
                num_tiles = ((args.m + variant.block_m - 1) // variant.block_m) * (
                    (args.n + variant.block_n - 1) // variant.block_n
                )
                partials = torch.empty(
                    num_tiles * variant.split_k * variant.block_m * variant.block_n,
                    device=device,
                    dtype=torch.float32,
                )
                counters = torch.zeros(num_tiles, device=device, dtype=torch.int32)
                scratch[variant.name] = (partials, counters)

            def invoke(variant=variant, partner=partner):
                if variant.split_k > 1:
                    partials, counters = scratch[variant.name]
                    if partner.dtype not in {torch.int8, torch.int16}:
                        raise ValueError("split-K benchmark variants require local partner indices")
                    result = input_tensor.new_empty((args.m, args.n))
                    bias_arg = result if module.bias is None else module.bias
                    num_pid_m = (args.m + variant.block_m - 1) // variant.block_m
                    num_pid_n = (args.n + variant.block_n - 1) // variant.block_n
                    num_cta_tiles = num_pid_m * (
                        (num_pid_n + variant.output_tiles_per_cta - 1)
                        // variant.output_tiles_per_cta
                    )
                    grid = (num_cta_tiles * variant.split_k,)
                    if variant.pair_counter:
                        kernel = paroquant_rotation_gemm_splitk_two_n_tiles_pair_counter_kernel
                    elif variant.output_tiles_per_cta == 2:
                        kernel = paroquant_rotation_gemm_splitk_two_n_tiles_kernel
                    else:
                        kernel = paroquant_rotation_gemm_splitk_kernel
                    with get_same_device_cm(module.qweight):
                        kernel[grid](
                            input_tensor,
                            module.qweight,
                            result,
                            module.qzeros,
                            module.scales,
                            partner,
                            cos,
                            sin,
                            channel_scales,
                            bias_arg,
                            partials,
                            counters,
                            args.m,
                            args.n,
                            args.k,
                            BLOCK_SIZE_M=variant.block_m,
                            BLOCK_SIZE_N=variant.block_n,
                            BLOCK_SIZE_K=PAROQUANT_MEGAKERNEL_GROUP_SIZE,
                            KROT=args.krot,
                            SPLIT_K=variant.split_k,
                            K_BLOCKS_PER_SPLIT=(args.k // PAROQUANT_MEGAKERNEL_GROUP_SIZE) // variant.split_k,
                            HAS_BIAS=module.bias is not None,
                            INPUT_IS_BF16=input_tensor.dtype == torch.bfloat16,
                            num_warps=variant.num_warps,
                            num_stages=variant.num_stages,
                            maxnreg=variant.maxnreg,
                            **(
                                {
                                    "PREFETCH_FIRST_WEIGHT": variant.prefetch_first_weight,
                                    "PREFETCH_SECOND_WEIGHT": variant.prefetch_packed_weight,
                                    "ATOMIC_COUNTER_RESET": variant.atomic_counter_reset,
                                }
                                if variant.pair_counter
                                else {}
                            ),
                        )
                    return result
                return _paroquant_rotation_gemm_triton(
                    input_tensor,
                    module.qweight,
                    module.scales,
                    module.qzeros,
                    partner,
                    cos,
                    sin,
                    channel_scales,
                    module.bias,
                    block_size_m=variant.block_m,
                    block_size_n=variant.block_n,
                    num_warps=variant.num_warps,
                    num_stages=variant.num_stages,
                    loop_unroll_factor=variant.loop_unroll_factor,
                    explicit_fma=variant.explicit_fma,
                    prefetch_first_partner=variant.prefetch_first_partner,
                    prefetch_packed_weight=variant.prefetch_packed_weight,
                    fp32_accum=FP32_ACCUM,
                )

            functions[variant.name] = invoke

        eager_outputs = {}
        for variant in args.variant:
            for _ in range(args.eager_warmup):
                eager_outputs[variant.name] = functions[variant.name]()
        torch.cuda.synchronize(device)

        reference = eager_outputs[args.variant[0].name]
        accuracy = {}
        for variant in args.variant:
            difference = (eager_outputs[variant.name] - reference).abs().float()
            accuracy[variant.name] = {
                "mismatched": int((difference != 0).sum().item()),
                "max_abs": difference.max().item(),
                "mean_abs": difference.mean().item(),
            }
            if variant.split_k == 1:
                torch.testing.assert_close(eager_outputs[variant.name], reference, rtol=0, atol=0)
            else:
                print(
                    f"{variant.name} accuracy: mismatched={(difference != 0).sum().item()}/{difference.numel()}, "
                    f"max_abs={difference.max().item():.6f}, mean_abs={difference.mean().item():.6f}"
                )
                torch.testing.assert_close(eager_outputs[variant.name], reference, rtol=0.01, atol=2.0)
                if difference.mean().item() > args.max_mean_drift:
                    raise AssertionError(
                        f"split-K mean drift exceeded {args.max_mean_drift:.6f}: "
                        f"{difference.mean().item():.6f}"
                    )

        exact_pair_accuracy = {}
        for baseline_name, candidate_name in args.exact_pair:
            difference = (eager_outputs[candidate_name] - eager_outputs[baseline_name]).abs().float()
            exact_pair_accuracy[f"{baseline_name}:{candidate_name}"] = {
                "mismatched": int((difference != 0).sum().item()),
                "max_abs": difference.max().item(),
                "mean_abs": difference.mean().item(),
            }
            torch.testing.assert_close(
                eager_outputs[candidate_name],
                eager_outputs[baseline_name],
                rtol=0,
                atol=0,
            )

        if args.profile_variant is not None:
            if args.profile_variant not in functions:
                raise ValueError(f"unknown profile variant: {args.profile_variant}")
            torch.cuda.cudart().cudaProfilerStart()
            for _ in range(args.profile_launches):
                functions[args.profile_variant]()
            torch.cuda.synchronize(device)
            torch.cuda.cudart().cudaProfilerStop()
            # Kernel replay restores profiler-owned memory snapshots, so explicitly reestablish the benchmark's
            # empty atomic-counter invariant before subsequent CUDA graph capture.
            for _, counters in scratch.values():
                counters.zero_()
            torch.cuda.synchronize(device)

        graphs = {}
        graph_outputs = {}
        for variant in args.variant:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_outputs[variant.name] = functions[variant.name]()
            graphs[variant.name] = graph
        torch.cuda.synchronize(device)

        for _ in range(args.graph_warmup):
            for variant in args.variant:
                graphs[variant.name].replay()
        torch.cuda.synchronize(device)

        for variant in args.variant:
            torch.testing.assert_close(
                graph_outputs[variant.name],
                eager_outputs[variant.name],
                rtol=0,
                atol=0,
            )

        events = {
            variant.name: (
                [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)],
                [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)],
            )
            for variant in args.variant
        }
        names = [variant.name for variant in args.variant]
        for iteration in range(args.iters):
            order = names if iteration % 2 == 0 else list(reversed(names))
            for name in order:
                starts, ends = events[name]
                starts[iteration].record()
                graphs[name].replay()
                ends[iteration].record()
        torch.cuda.synchronize(device)

    stats = {}
    for variant in args.variant:
        starts, ends = events[variant.name]
        samples_us = [starts[index].elapsed_time(ends[index]) * 1000.0 for index in range(args.iters)]
        stats[variant.name] = _stats(samples_us)

    baseline = stats[args.variant[0].name]
    rows = []
    for variant in args.variant:
        result = stats[variant.name]
        rows.append(
            [
                variant.name,
                variant.block_m,
                variant.block_n,
                variant.num_warps,
                variant.num_stages,
                variant.partner_dtype,
                variant.loop_unroll_factor,
                str(variant.explicit_fma),
                str(variant.prefetch_first_partner),
                str(variant.prefetch_packed_weight),
                variant.split_k,
                variant.output_tiles_per_cta,
                "-" if variant.maxnreg is None else variant.maxnreg,
                str(variant.pair_counter),
                str(variant.atomic_counter_reset),
                str(variant.prefetch_first_weight),
                f"{result['p50_us']:.3f}",
                f"{result['mean_us']:.3f}",
                f"{result['p95_us']:.3f}",
                f"{baseline['p50_us'] / result['p50_us']:.4f}x",
                f"{baseline['mean_us'] / result['mean_us']:.4f}x",
            ]
        )
    print(
        tabulate(
            rows,
            headers=(
                "variant",
                "BM",
                "BN",
                "warps",
                "stages",
                "partner",
                "unroll",
                "FMA",
                "prefetch first",
                "prefetch weight",
                "split K",
                "output tiles",
                "maxnreg",
                "pair counter",
                "atomic reset",
                "prefetch first weight",
                "p50 us",
                "mean us",
                "p95 us",
                "p50",
                "mean",
            ),
            tablefmt="plain",
        )
    )

    props = torch.cuda.get_device_properties(device)
    return {
        "git_revision": _git_revision(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": args.device,
        "device": props.name,
        "device_uuid": str(props.uuid),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "sm_count": props.multi_processor_count,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "triton_version": __import__("triton").__version__,
        "dtype": args.dtype,
        "shape": {"m": args.m, "k": args.k, "n": args.n},
        "krot": args.krot,
        "seed": args.seed,
        "fp32_accum": FP32_ACCUM,
        "eager_warmup": args.eager_warmup,
        "graph_warmup": args.graph_warmup,
        "iters": args.iters,
        "ordering": "AB/BA alternating",
        "profile_variant": args.profile_variant,
        "profile_launches": args.profile_launches,
        "graph_matches_eager_exactly": True,
        "accuracy": accuracy,
        "exact_pair_accuracy": exact_pair_accuracy,
        "variants": [asdict(variant) for variant in args.variant],
        "stats": stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="CUDA index within the visible device set")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--krot", type=int, choices=(1, 8), default=8)
    parser.add_argument("--variant", type=_parse_variant, action="append", required=True)
    parser.add_argument("--eager-warmup", type=int, default=20)
    parser.add_argument("--graph-warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-mean-drift", type=float, default=0.003)
    parser.add_argument(
        "--exact-pair",
        type=_parse_exact_pair,
        action="append",
        default=[],
        help="Require bit-exact eager output for BASELINE:CANDIDATE; may be repeated",
    )
    parser.add_argument("--profile-variant")
    parser.add_argument("--profile-launches", type=int, default=1)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if len({variant.name for variant in args.variant}) != len(args.variant):
        raise ValueError("variant names must be unique")
    variant_names = {variant.name for variant in args.variant}
    if any(name not in variant_names for pair in args.exact_pair for name in pair):
        raise ValueError("every exact-pair name must identify a configured variant")
    if args.variant[0].split_k != 1:
        raise ValueError("the first variant must be a non-split reference")

    payload = run(args)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
