#!/usr/bin/env python3
"""H100 experiment: packed transition extraction versus the last cooperative prototype."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("MAX_JOBS", "8")
os.environ.setdefault("NINJAFLAGS", "-j8")
os.environ.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "8")
os.environ.setdefault("NVCC_THREADS", "2")

from scripts import benchmark_qvq_cuda_lr as benchmark_utils
from scripts import benchmark_qvq_lr_vs_gptq_llama32_1b as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, default=1)
    parser.add_argument("--shapes", nargs="+", choices=[case.name for case in base.LLAMA32_1B_SHAPES],
                        default=[case.name for case in base.LLAMA32_1B_SHAPES])
    parser.add_argument("--m", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--bits", nargs="+", type=float, default=[2.0, 2.5])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--correctness-seeds", type=int, default=3)
    parser.add_argument("--correctness-repeats", type=int, default=10)
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark_artifacts/qvq_lr_exp_packed_h100.json"))
    parser.add_argument("--markdown-output", type=Path,
                        default=Path("benchmark_artifacts/qvq_lr_exp_packed_h100.md"))
    return parser.parse_args()


def load_kernel(
    *,
    name: str,
    namespace: str,
    source: str,
    build_root_env: str,
    default_build_root: str,
    display_name: str,
):
    from gptqmodel.utils.cpp import (
        TorchOpsJitExtension,
        default_jit_cflags,
        default_jit_cuda_cflags,
    )

    extension = TorchOpsJitExtension(
        name=name,
        namespace=namespace,
        required_ops=("gemv_lr",),
        sources=[str(REPO_ROOT / source)],
        build_root_env=build_root_env,
        default_build_root=default_build_root,
        display_name=display_name,
        extra_cflags=lambda: default_jit_cflags(enable_bf16=True),
        extra_cuda_cflags=lambda: default_jit_cuda_cflags(
            enable_bf16=True,
            include_lineinfo=True,
            include_nvcc_threads=True,
            nvcc_threads=2,
            include_split_compile=True,
            include_fast_compile=True,
            include_ptxas_optimizations=True,
            include_ptxas_verbosity=False,
            include_fatbin_compression=True,
            include_diag_suppress=True,
        ),
        force_rebuild_env=f"{build_root_env}_FORCE_REBUILD",
        verbose_env="GPTQMODEL_EXT_VERBOSE",
        requires_cuda=True,
    )
    if not extension.load():
        raise RuntimeError(extension.last_error_message())
    return extension.op("gemv_lr")


def timing(torch, call, *, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(iterations)]
    graph = torch.cuda.CUDAGraph()
    held = None
    with torch.cuda.graph(graph):
        for start, end in zip(starts, ends, strict=True):
            start.record()
            held = call()
            end.record()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    values = sorted(start.elapsed_time(end) for start, end in zip(starts, ends, strict=True))
    del held
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": values[min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)],
        "min_ms": values[0],
        "max_ms": values[-1],
    }


def error_metrics(torch, actual, reference) -> dict[str, float | bool]:
    actual_f = actual.float()
    reference_f = reference.float()
    delta = actual_f - reference_f
    return {
        "finite": bool(torch.isfinite(actual_f).all()),
        "mae": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "max_abs": float(delta.abs().max()),
        "rel_l2": float(torch.linalg.vector_norm(delta) /
                        torch.linalg.vector_norm(reference_f).clamp_min(1e-12)),
    }


def validate_repeated(torch, call, reference, *, repeats: int) -> dict[str, float | bool]:
    outputs = []
    worst = {"finite": True, "mae": 0.0, "rmse": 0.0, "max_abs": 0.0, "rel_l2": 0.0}
    for _ in range(repeats):
        output = call()
        torch.cuda.synchronize()
        metrics = error_metrics(torch, output, reference)
        outputs.append(output)
        worst["finite"] = bool(worst["finite"] and metrics["finite"])
        for key in ("mae", "rmse", "max_abs", "rel_l2"):
            worst[key] = max(float(worst[key]), float(metrics[key]))
    first = outputs[0]
    worst["run_spread_max_abs"] = max(
        (float((output - first).abs().max()) for output in outputs[1:]),
        default=0.0,
    )
    if not worst["finite"] or float(worst["max_abs"]) > 2e-3:
        raise AssertionError(f"prototype failed dense-reference gate: {worst}")
    return worst


def markdown(payload: dict) -> str:
    lines = [
        "# Packed-transition cooperative decode experiment — H100",
        "",
        "The candidate consumes the unchanged W2/W2.5 trellis and bank-selector payload.",
        "`Better than last` means faster than the same-run first cooperative prototype.",
        "",
        "| Shape | M | K | N | W | Candidate | Median ms | P95 ms | TFLOP/s | xLast | xMarlin | xMachete | Better than last | Max abs | Run spread |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['shape']} | {row['m']} | {row['k']} | {row['n']} | {row['bits']:g} | "
            f"{row['candidate']} | {row['median_ms']:.4f} | {row['p95_ms']:.4f} | "
            f"{row['logical_tflops']:.3f} | {row.get('speedup_vs_last', 1.0):.3f}x | "
            f"{row.get('speedup_vs_marlin', float('nan')):.3f}x | "
            f"{row.get('speedup_vs_machete', float('nan')):.3f}x | "
            f"{'yes' if row.get('better_than_last', False) else 'no'} | "
            f"{row['max_abs']:.3g} | {row.get('run_spread_max_abs', 0.0):.3g} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    hardware = benchmark_utils._idle_preflight(args.physical_gpu, 3, 1.0, 8)
    os.environ["CUDA_VISIBLE_DEVICES"] = hardware["uuid"]

    import torch

    from gptqmodel.quantization.qvq_rates import qvq_transition_bits
    from gptqmodel.utils.marlin_scalar_type import scalar_types
    from gptqmodel.utils.qvq_cuda import _pgc16_levels

    base._visible_gpu_matches(torch, hardware)
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    if (properties.major, properties.minor) != (9, 0):
        raise RuntimeError(f"prototype requires H100 compute capability 9.0, got {properties.major}.{properties.minor}")
    print(
        f"H100 physical={args.physical_gpu} pci={hardware['pci.bus_id']} uuid={hardware['uuid']} "
        f"name={properties.name} cc={properties.major}.{properties.minor} "
        f"sms={properties.multi_processor_count} memory={properties.total_memory} "
        f"torch={torch.__version__} cuda={torch.version.cuda}",
        flush=True,
    )
    last_op = load_kernel(
        name="qvq_lr_cooperative_proto_ops",
        namespace="gptqmodel_qvq_proto",
        source="benchmark_artifacts/qvq_lr_cooperative_proto.cu",
        build_root_env="GPTQMODEL_QVQ_LR_PROTO_BUILD_ROOT",
        default_build_root="/tmp/qvq-jit-current/qvq_lr_cooperative_proto",
        display_name="QVQ local-ring cooperative prototype",
    )
    candidate_op = load_kernel(
        name="qvq_lr_exp_packed_ops",
        namespace="gptqmodel_qvq_exp_packed",
        source="benchmark_artifacts/qvq_lr_exp_packed.cu",
        build_root_env="GPTQMODEL_QVQ_LR_EXP_PACKED_BUILD_ROOT",
        default_build_root="/tmp/qvq-jit-current/qvq_lr_exp_packed",
        display_name="QVQ packed-transition experiment",
    )
    torch.cuda.synchronize()

    cases = [base._shape_by_name(name) for name in args.shapes]
    levels = _pgc16_levels(device, "pgc16-v1")
    correctness = []
    for shape_index, case in enumerate(cases):
        for bits in args.bits:
            transition_bits = qvq_transition_bits(bits, vector_size=2)
            for seed_index in range(args.correctness_seeds):
                seed = 20260830 + shape_index * 1000 + int(bits * 10) + seed_index * 100_000
                trellis, bank_ids, dense, _ = base._qvq_payload(
                    torch, case, bits=bits, seed=seed, device=device
                )
                input_generator = torch.Generator().manual_seed(seed + 17)
                for m in args.m:
                    x = (torch.randn((m, case.in_features), generator=input_generator, dtype=torch.float32) * 0.1).to(
                        device=device, dtype=torch.float16
                    )
                    reference = x.float() @ dense.float()

                    def correctness_call(
                        x=x,
                        trellis=trellis,
                        levels=levels,
                        bank_ids=bank_ids,
                        transition_bits=transition_bits,
                        out_features=case.out_features,
                    ):
                        return candidate_op(
                            x, trellis, levels, transition_bits, out_features, True, bank_ids, 3, 0
                        )

                    metrics = validate_repeated(
                        torch, correctness_call, reference, repeats=args.correctness_repeats
                    )
                    correctness.append({
                        "shape": case.name,
                        "m": m,
                        "k": case.in_features,
                        "n": case.out_features,
                        "bits": bits,
                        "seed": seed,
                        **metrics,
                    })
                print(
                    f"correctness {case.name} W{bits:g} seed={seed}: "
                    f"M={args.m} passed {args.correctness_repeats} repeats",
                    flush=True,
                )
                del trellis, bank_ids, dense

    base._pre_timing_exclusivity_gate(
        physical_gpu=args.physical_gpu,
        gpu_uuid=hardware["uuid"],
        samples=3,
        interval=1.0,
    )

    rows = []
    for shape_index, case in enumerate(cases):
        generator = torch.Generator().manual_seed(20260830 + shape_index * 1000)
        inputs = {
            m: (torch.randn((m, case.in_features), generator=generator, dtype=torch.float32) * 0.1)
            .to(device=device, dtype=torch.float16)
            for m in args.m
        }
        gptq_source = base._gptq_source(
            torch,
            case,
            group_size=base.GPTQ_GROUP_SIZE,
            seed=20260830 + shape_index * 1000 + 1,
            dtype=torch.float16,
        )
        dense_gptq = base._dense_gptq_weight(
            torch, gptq_source, device=device, weight_type=scalar_types.uint4b8
        )
        marlin, _ = base._build_gptq_module(
            torch, "gptq_marlin", case, group_size=base.GPTQ_GROUP_SIZE,
            source=gptq_source, device=device, dtype=torch.float16
        )
        machete, _ = base._build_gptq_module(
            torch, "gptq_machete", case, group_size=base.GPTQ_GROUP_SIZE,
            source=gptq_source, device=device, dtype=torch.float16
        )
        baseline_times = {}
        for m, x in inputs.items():
            reference = x.float() @ dense_gptq.float()
            for name, module in (("marlin", marlin), ("machete", machete)):
                call = lambda module=module, x=x: module(x)
                actual = call()
                torch.cuda.synchronize()
                metrics = error_metrics(torch, actual, reference)
                if not metrics["finite"] or float(metrics["max_abs"]) > 2e-2 + 2e-2 * float(reference.abs().max()):
                    raise AssertionError(f"{name} correctness failure: {metrics}")
                baseline_times[(name, m)] = timing(
                    torch, call, warmup=args.warmup, iterations=args.iterations
                )["median_ms"]

        for bits in args.bits:
            transition_bits = qvq_transition_bits(bits, vector_size=2)
            trellis, bank_ids, dense, _ = base._qvq_payload(
                torch,
                case,
                bits=bits,
                seed=20260830 + shape_index * 1000 + int(bits * 10),
                device=device,
            )
            for m, x in inputs.items():
                reference = x.float() @ dense.float()

                def last_call(
                    x=x,
                    trellis=trellis,
                    levels=levels,
                    bank_ids=bank_ids,
                    transition_bits=transition_bits,
                    out_features=case.out_features,
                ):
                    return last_op(
                        x,
                        trellis,
                        levels,
                        transition_bits,
                        out_features,
                        True,
                        bank_ids,
                        3,
                        0,
                    )

                def candidate_call(
                    x=x,
                    trellis=trellis,
                    levels=levels,
                    bank_ids=bank_ids,
                    transition_bits=transition_bits,
                    out_features=case.out_features,
                ):
                    return candidate_op(
                        x,
                        trellis,
                        levels,
                        transition_bits,
                        out_features,
                        True,
                        bank_ids,
                        3,
                        0,
                    )
                last_output = last_call()
                torch.cuda.synchronize()
                last_error = error_metrics(torch, last_output, reference)
                if not last_error["finite"] or float(last_error["max_abs"]) > 2e-3:
                    raise AssertionError(f"last prototype correctness failure: {last_error}")
                candidate_error = validate_repeated(
                    torch, candidate_call, reference, repeats=args.correctness_repeats
                )
                last_timing = timing(
                    torch, last_call, warmup=args.warmup, iterations=args.iterations
                )
                candidate_timing = timing(
                    torch, candidate_call, warmup=args.warmup, iterations=args.iterations
                )
                logical_flops = 2 * m * case.in_features * case.out_features
                for candidate, measured, metrics in (
                    ("last_cooperative", last_timing, last_error),
                    ("packed_experiment", candidate_timing, candidate_error),
                ):
                    row = {
                        "shape": case.name,
                        "m": m,
                        "k": case.in_features,
                        "n": case.out_features,
                        "bits": bits,
                        "candidate": candidate,
                        **measured,
                        **metrics,
                    }
                    row["logical_tflops"] = logical_flops / (measured["median_ms"] * 1e9)
                    row["speedup_vs_last"] = last_timing["median_ms"] / measured["median_ms"]
                    row["speedup_vs_marlin"] = baseline_times[("marlin", m)] / measured["median_ms"]
                    row["speedup_vs_machete"] = baseline_times[("machete", m)] / measured["median_ms"]
                    row["better_than_last"] = (
                        candidate == "packed_experiment"
                        and measured["median_ms"] < last_timing["median_ms"]
                    )
                    rows.append(row)
                print(
                    f"complete {case.name} M{m} W{bits:g}: last={last_timing['median_ms']:.4f}ms "
                    f"packed={candidate_timing['median_ms']:.4f}ms "
                    f"speedup={last_timing['median_ms']/candidate_timing['median_ms']:.3f}x "
                    f"max_abs={candidate_error['max_abs']:.3g}",
                    flush=True,
                )

        del inputs, gptq_source, dense_gptq, marlin, machete

    payload = {
        "commit": benchmark_utils._git_commit(),
        "last_source": "benchmark_artifacts/qvq_lr_cooperative_proto.cu",
        "candidate_source": "benchmark_artifacts/qvq_lr_exp_packed.cu",
        "storage_change_bits_per_weight": 0.0,
        "physical_gpu": args.physical_gpu,
        "hardware": hardware,
        "device": {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "sm_count": properties.multi_processor_count,
            "total_memory": properties.total_memory,
        },
        "software": {"torch": torch.__version__, "cuda": torch.version.cuda},
        "timing": "one CUDA Graph replay with internal external CUDA events; host launch gaps excluded",
        "args": vars(args) | {"output": str(args.output), "markdown_output": str(args.markdown_output)},
        "correctness": correctness,
        "rows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    args.markdown_output.write_text(markdown(payload), encoding="utf-8")
    print(f"wrote {args.output} and {args.markdown_output}", flush=True)


if __name__ == "__main__":
    main()
