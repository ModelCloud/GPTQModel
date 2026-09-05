#!/usr/bin/env python3
"""Paired experimental AMD butterfly/library comparison; does not enable production dispatch.

Use --full-sweep for the complete Qwen3.8-27B shape/rate/M matrix. Optional AITER
solution indices are build-specific research knobs, not portable defaults.
"""

import argparse
import ast
import hashlib
import itertools
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

from benchmark_qvq_p32_amd import _idle_preflight, _rocm_snapshot, _timing_recheck
from benchmark_qvq_p32_amd_dispatch_sweep import QWEN38_27B_SHAPES, REQUESTED_M
from benchmark_qvq_p32_amd_fold_ceiling import SHAPE_AXES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=1.0)
    parser.add_argument("--idle-memory-tolerance-mib", type=int, default=1024)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--full-sweep", action="store_true")
    parser.add_argument("--baseline-forward-commit", help="Compare the folded-forward method from this git revision")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--butterfly", choices=("none", "gather", "split"), default="split"
    )
    parser.add_argument("--block-rows", type=int, choices=(4, 8, 16, 32), default=4)
    parser.add_argument("--hipblaslt-solution", type=int, default=-1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 2 or args.warmup < 1:
        parser.error("Require iterations >= 2 and warmup >= 1")
    hardware, valid = _idle_preflight(args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["HIP_VISIBLE_DEVICES"] = str(args.physical_gpu)
    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/qvq-triton-butterfly-experiment")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    import torch
    import triton
    from qvq_p32_amd_butterfly_experiment import fht128_kernel, fht128_split_kernel

    import gptqmodel.nn_modules.qlinear.qvq as qvq_module
    from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
    from gptqmodel.quantization.qvq import pack_qvq_binary_bank_ids
    from gptqmodel.quantization.qvq_rates import qvq_words_per_tile
    from gptqmodel.quantization.rotation.hadamard_utils import matmul_hadU
    from gptqmodel.utils.qvq_amd import qvq_p32_amd_folded_case_supported

    props = torch.cuda.get_device_properties(0)
    if not torch.version.hip or props.gcnArchName.split(":")[0] != "gfx950":
        raise RuntimeError("This experiment requires gfx950")
    original_mm = torch.mm
    candidate_forward = QVQLinear._qvq_amd_folded_forward
    baseline_forward = candidate_forward
    if args.baseline_forward_commit:
        revision = subprocess.check_output(
            ["git", "rev-parse", "--verify", args.baseline_forward_commit + "^{commit}"], cwd=root, text=True,
        ).strip()
        source = subprocess.check_output(
            ["git", "show", revision + ":gptqmodel/nn_modules/qlinear/qvq.py"], cwd=root, text=True,
        )
        cls = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "QVQLinear")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_qvq_amd_folded_forward")
        namespace = dict(vars(qvq_module))
        # Explicit opt-in to trusted local repository code, never network input.
        exec(compile(ast.Module(body=[method], type_ignores=[]), "<baseline-folded-forward>", "exec"), namespace)  # noqa: S102
        baseline_forward = namespace["_qvq_amd_folded_forward"]
    hipb_mm = None
    if args.hipblaslt_solution >= 0:
        import aiter
        from aiter.ops.gradlib import _hipb_mm

        aiter.hipb_create_extension()
        hipb_mm = _hipb_mm
    kernel = fht128_split_kernel if args.butterfly == "split" else fht128_kernel

    def candidate_mm(x, w, *mm_args, **kwargs):
        # Only intercept the two exact production calls; all other shapes and
        # contracts pass through unchanged. Scoped to this single-threaded tool.
        if (
            hipb_mm is not None
            and x.shape == (1024, 17408)
            and w.shape == (17408, 5120)
            and x.dtype == w.dtype == torch.float16
            and x.is_contiguous()
            and w.is_contiguous()
            and not mm_args
            and kwargs == {"out_dtype": torch.float32}
        ):
            y = torch.empty((1024, 5120), device=x.device, dtype=torch.float32)
            hipb_mm(x, w, args.hipblaslt_solution, y)
            return y
        if (
            args.butterfly != "none"
            and x.shape == (40960, 128)
            and w.shape == (128, 128)
            and x.dtype == w.dtype == torch.float32
            and x.is_contiguous()
            and w.is_contiguous()
            and not mm_args
            and not kwargs
        ):
            y = torch.empty_like(x)
            kernel[(triton.cdiv(x.shape[0], args.block_rows),)](
                x,
                y,
                x.shape[0],
                args.block_rows,
                num_warps=4,
            )
            return y
        return original_mm(x, w, *mm_args, **kwargs)

    def select(name):
        QVQLinear._qvq_amd_folded_forward = baseline_forward if name == "baseline" else candidate_forward
        torch.mm = (candidate_mm if name == "candidate"
                    and (args.butterfly != "none" or hipb_mm is not None) else original_mm)

    shapes = QWEN38_27B_SHAPES if args.full_sweep else [("mlp_down", 17408, 5120)]
    m_values = REQUESTED_M if args.full_sweep else [1024]
    permitted = set(hardware["process_ids"]) | set(
        _rocm_snapshot(args.physical_gpu)["process_ids"]
    )
    report = {
        "baseline": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "retained_kernel_baseline": "c89459e3",
        "hardware": hardware,
        "software": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": triton.__version__,
            "gpu": props.name,
            "arch": props.gcnArchName,
            "cu_count": props.multi_processor_count,
        },
        "config": vars(args) | {"output": str(args.output)},
        "valid": valid,
        "rows": [],
        "kernel_source_sha256": hashlib.sha256(
            (root / "scripts/qvq_p32_amd_butterfly_experiment.py").read_bytes()
        ).hexdigest(),
        "qvq_source_sha256": hashlib.sha256(
            (root / "gptqmodel/nn_modules/qlinear/qvq.py").read_bytes()
        ).hexdigest(),
        "status": "exploratory; not production promotion or model-quality evidence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for (shape, k, n), bits in itertools.product(shapes, (2.0, 2.5, 3.0, 3.5)):
            torch.mm = original_mm
            ih, oh = SHAPE_AXES[shape]
            g = torch.Generator(device="cuda").manual_seed(
                20260904 + int(bits * 10) + k + n
            )
            tiles = k * n // 256
            packed = torch.randint(
                -(1 << 31),
                1 << 31,
                (tiles, qvq_words_per_tile(bits, vector_size=2)),
                device="cuda",
                dtype=torch.int32,
                generator=g,
            )
            banks = pack_qvq_binary_bank_ids(
                torch.randint(
                    0,
                    2,
                    (tiles * 8,),
                    device="cuda",
                    dtype=torch.uint8,
                    generator=g,
                )
            )
            layer = QVQLinear(
                bits=bits,
                in_features=k,
                out_features=n,
                bank_count=2,
                v2b2_p32=True,
                input_hadamard=ih,
                output_hadamard=oh,
                tensors={
                    "trellis": packed,
                    "bank_ids": banks,
                    "SU": torch.ones(k, device="cuda"),
                    "SV": torch.ones(n, device="cuda"),
                    "bank_alt_id": torch.tensor([3], device="cuda", dtype=torch.uint8),
                },
            ).eval()
            inner = layer.get_inner_weight_tensor()
            for m in m_values:
                torch.mm = original_mm
                x = (
                    torch.randn((m, k), device="cuda", dtype=torch.float16, generator=g)
                    * 0.01
                ).contiguous()
                ref = matmul_hadU(x.float()) if ih else x.float()
                ref = ref @ inner
                if oh:
                    ref = matmul_hadU(ref)
                outputs = {}
                for name, fn in [
                    ("baseline", original_mm),
                    ("candidate", candidate_mm),
                ]:
                    select(name)
                    outputs[name] = layer(x)
                    for _ in range(args.warmup):
                        layer(x)
                torch.cuda.synchronize()
                _, okay = _timing_recheck(args, permitted)
                report["valid"] &= okay
                records = []
                for i in range(args.iterations):
                    order = (
                        ["baseline", "candidate"]
                        if i % 2 == 0
                        else ["candidate", "baseline"]
                    )
                    for name in order:
                        records.append(
                            (
                                name,
                                torch.cuda.Event(enable_timing=True),
                                torch.cuda.Event(enable_timing=True),
                            )
                        )
                for name, start, end in records:
                    select(name)
                    start.record()
                    layer(x)
                    end.record()
                torch.cuda.synchronize()
                row = {
                    "shape": shape,
                    "bits": bits,
                    "m": m,
                    "k": k,
                    "n": n,
                    "dtype": "float16",
                }
                for name, value in outputs.items():
                    times = sorted(
                        s.elapsed_time(e) for label, s, e in records if label == name
                    )
                    delta = value.float() - ref
                    row[name] = {
                        "median_ms": statistics.median(times),
                        "mean_ms": statistics.mean(times),
                        "p95_ms": times[int(0.95 * (len(times) - 1))],
                        "min_ms": times[0],
                        "max_ms": times[-1],
                        "max_abs": delta.abs().max().item(),
                        "mean_abs": delta.abs().mean().item(),
                        "relative_l2": (
                            delta.norm() / ref.norm().clamp_min(1e-12)
                        ).item(),
                    }
                row["speedup"] = (
                    row["baseline"]["median_ms"] / row["candidate"]["median_ms"]
                )
                row["accuracy_basis"] = (
                    "canonical_fp32"
                    if qvq_p32_amd_folded_case_supported(m, k, n)
                    else "exact_baseline"
                )
                row["accuracy_pass"] = (
                    row["candidate"]["max_abs"] <= 0.002
                    if row["accuracy_basis"] == "canonical_fp32"
                    else torch.equal(outputs["candidate"], outputs["baseline"])
                )
                if (m, k, n) == (1024, 17408, 5120):
                    select("candidate")
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        xs = x.clone()
                        ys = layer(xs)
                    torch.cuda.current_stream().wait_stream(stream)
                    row["stream_equal"] = torch.equal(ys, outputs["candidate"])
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        yg = layer(x)
                    graph.replay()
                    torch.cuda.synchronize()
                    row["graph_equal"] = torch.equal(yg, outputs["candidate"])
                    del xs, ys, yg, graph
                report["rows"].append(row)
                args.output.write_text(json.dumps(report, indent=2))
                print(
                    f"{shape:12} W{bits:g} M={m:4} K={k:5} N={n:5} fp16 "
                    f"baseline={row['baseline']['median_ms']:.6f}ms "
                    f"candidate={row['candidate']['median_ms']:.6f}ms "
                    f"speedup={row['speedup']:.3f}x pass={row['accuracy_pass']}",
                    flush=True,
                )
                del x, ref, outputs
            del layer, packed, banks, inner
            torch.cuda.empty_cache()
        report["completed"] = True
        args.output.write_text(json.dumps(report, indent=2))
    except Exception as exc:
        report["valid"] = False
        report["completed"] = False
        report["error"] = repr(exc)
        args.output.write_text(json.dumps(report, indent=2))
        raise
    finally:
        torch.mm = original_mm
        QVQLinear._qvq_amd_folded_forward = candidate_forward


if __name__ == "__main__":
    main()
