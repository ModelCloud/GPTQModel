#!/usr/bin/env python3
"""Paired experimental AMD butterfly/library comparison; does not enable production dispatch.

Use --full-sweep for the complete Qwen3.8-27B shape/rate/M matrix. Optional AITER
solution indices are build-specific research knobs, not portable defaults.
"""

import argparse
import ast
import copy
import hashlib
import importlib.util
import itertools
import json
import os
import statistics
import subprocess
import sys
import tempfile
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
    parser.add_argument("--trim-composite", action="store_true", help="Experimentally remove padded recovery math")
    parser.add_argument("--recovery-warps", type=int, choices=(4, 8), default=8)
    parser.add_argument("--gemv-full-k", action="store_true")
    parser.add_argument("--gemv-dot2", action="store_true", help="Packed FP16 dot products accumulated in FP32")
    parser.add_argument("--gemv-dot2-loop", action="store_true", help="Reduce once after the packed-dot K loop")
    parser.add_argument("--gemv-gluon", action="store_true", help="Use explicit layouts for the packed-dot K loop")
    parser.add_argument("--inplace-correction", action="store_true",
                        help="Reuse the private FP32 primary output for residual addmm")
    parser.add_argument("--fused-correction", action="store_true", help="Experimental shared-X Gluon residual GEMM")
    parser.add_argument("--fused-interleave", action="store_true", help="Interleave high/low into one FP32 accumulator")
    parser.add_argument("--fused-keep-masks", action="store_true", help="Control: retain masks on divisible fused tiles")
    parser.add_argument("--fused-prefetch", action="store_true", help="Use explicit double-buffered global-to-LDS copies")
    parser.add_argument("--fused-block-m", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--fused-block-n", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--fused-block-k", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument("--torch-profile-dir", type=Path, help="Capture warmed paired operator mapping traces")
    parser.add_argument("--gemv-loop-k", type=int, choices=(256, 512, 1024), default=512)
    parser.add_argument("--gemv-split-k", action="store_true", help="Remove padded arithmetic from full-K GEMV")
    parser.add_argument("--graph-execute", action="store_true", help="Stage fresh inputs and clone graph outputs")
    parser.add_argument("--aiter-skinny", choices=("none", "wv", "llmm1"), default="none")
    parser.add_argument("--aiter-direct", action="store_true", help="Measure private native binding with cached weight metadata")
    parser.add_argument("--gemv-block-n", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--shapes", nargs="+", choices=[shape[0] for shape in QWEN38_27B_SHAPES])
    parser.add_argument("--m-values", nargs="+", type=int, choices=REQUESTED_M)
    parser.add_argument("--folded-residual-ceiling", action="store_true",
                        help="Benchmark a raw cached high+residual operator, not production dispatch")
    parser.add_argument("--folded-direct-ceiling", action="store_true",
                        help="Bypass layer guards for the unchanged cached production operator")
    parser.add_argument("--baseline-forward-commit", help="Compare the folded-forward method from this git revision")
    parser.add_argument("--baseline-amd-commit", help="Use isolated kernel module and layer caches from this git revision")
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
    if args.folded_direct_ceiling and args.folded_residual_ceiling:
        parser.error("Choose only one raw operator ceiling")
    if args.graph_execute and args.aiter_skinny != "none":
        parser.error("Graph staging and skinny dispatch are separate experiments")
    if args.aiter_direct and args.aiter_skinny == "none":
        parser.error("--aiter-direct requires --aiter-skinny and a prebuilt AITER module_custom")
    if sum((args.inplace_correction, args.fused_correction, args.graph_execute, args.aiter_skinny != "none")) > 1:
        parser.error("Choose one correction, graph staging, or skinny dispatch experiment")
    hardware, valid = _idle_preflight(args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["HIP_VISIBLE_DEVICES"] = str(args.physical_gpu)
    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/qvq-triton-butterfly-experiment")
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    import torch
    import triton
    from qvq_p32_amd_butterfly_experiment import (
        composite_trim_kernel,
        fht128_kernel,
        fht128_split_kernel,
        folded_gemv_dot2_gluon_kernel,
        folded_gemv_dot2_kernel,
        folded_gemv_dot2_loop_kernel,
        folded_gemv_full_k_kernel,
        folded_gemv_split_k_kernel,
        folded_residual_gemm_gluon_kernel,
    )
    from qvq_p32_amd_prefetch_experiment import folded_residual_prefetch_kernel

    import gptqmodel.nn_modules.qlinear.qvq as qvq_module
    import gptqmodel.utils.qvq_amd as candidate_amd
    from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
    from gptqmodel.quantization.qvq import pack_qvq_binary_bank_ids
    from gptqmodel.quantization.qvq_rates import qvq_words_per_tile
    from gptqmodel.quantization.rotation.hadamard_utils import matmul_hadU
    from gptqmodel.utils.qvq_amd import qvq_p32_amd_folded_case_supported

    baseline_amd = candidate_amd
    original_recovery = candidate_amd._qvq_p32_composite_recovery_gfx950_kernel
    original_gemv = candidate_amd._qvq_p32_folded_gemv_gfx950_kernel
    original_execute = candidate_amd._qvq_p32_folded_execute
    aiter_skinny = None
    if args.aiter_skinny != "none":
        import aiter

        aiter_skinny = aiter.wvSpltK if args.aiter_skinny == "wv" else aiter.LLMM1
        if args.aiter_direct:
            from aiter.jit.core import _pybind_develop_hooks, get_module

            skinny_module = get_module("module_custom")
            skinny_native = getattr(skinny_module, "wvSpltK" if args.aiter_skinny == "wv" else "LLMM1")
            skinny_convert, _, skinny_raw_stream, skinny_current_device = _pybind_develop_hooks()
    baseline_source_dir = None
    if args.baseline_amd_commit:
        revision = subprocess.check_output(
            ["git", "rev-parse", "--verify", args.baseline_amd_commit + "^{commit}"], cwd=root, text=True,
        ).strip()
        source = subprocess.check_output(
            ["git", "show", revision + ":gptqmodel/utils/qvq_amd.py"], cwd=root, text=True,
        )
        # A real generated snapshot is required for Triton's inspect/JIT source
        # lookup. This is trusted local git code, with independent module globals.
        baseline_source_dir = tempfile.TemporaryDirectory(prefix="qvq-amd-baseline-")
        snapshot = Path(baseline_source_dir.name) / "qvq_amd.py"
        snapshot.write_text(source)
        spec = importlib.util.spec_from_file_location("gptqmodel.utils._qvq_amd_benchmark_baseline", snapshot)
        baseline_amd = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline_amd
        spec.loader.exec_module(baseline_amd)

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

    class TrimRecovery:
        def __getitem__(self, grid):
            def launch(*positional, **kwargs):
                kwargs["num_warps"] = args.recovery_warps
                return composite_trim_kernel[grid](*positional, **kwargs)
            return launch

    trimmed_recovery = TrimRecovery()

    class FullKGemv:
        def __getitem__(self, grid):
            def launch(*positional, **kwargs):
                size_n = grid[0] * kwargs["block_n"]
                kwargs["block_n"] = args.gemv_block_n
                kwargs["block_k"] = triton.next_power_of_2(kwargs["size_k"])
                if args.gemv_dot2_loop or args.gemv_gluon:
                    kwargs["block_k"] = args.gemv_loop_k
                gemv = (folded_gemv_dot2_gluon_kernel if args.gemv_gluon else
                        folded_gemv_dot2_loop_kernel if args.gemv_dot2_loop else
                        folded_gemv_dot2_kernel if args.gemv_dot2 else
                        folded_gemv_split_k_kernel if args.gemv_split_k else folded_gemv_full_k_kernel)
                return gemv[(size_n // args.gemv_block_n,)](*positional, **kwargs)
            return launch

    full_k_gemv = FullKGemv()

    graph_entries = {}
    skinny_weights = {}

    def skinny_execute(x, operand, residual_operand=None, composite_recovery=None, **kwargs):
        m, k = x.shape
        n = kwargs["out_features"]
        if (k, n) == (6144, 5120) and m <= 32:
            residual_operand = None
        if (composite_recovery is not None or residual_operand is not None or kwargs["output_fp32"]
                or m > (4 if args.aiter_skinny == "wv" else 1) or not operand.T.is_contiguous()):
            return original_execute(x, operand, residual_operand, composite_recovery, **kwargs)
        output = torch.empty((m, n), device=x.device, dtype=x.dtype)
        if args.aiter_direct:
            weight_entry = skinny_weights.get(id(operand))
            if weight_entry is None:
                weight_entry = (operand, skinny_convert(operand.T))
                skinny_weights[id(operand)] = weight_entry
            # Same stream-setting contract as AITER's develop=True wrapper.
            skinny_module._set_current_hip_stream(skinny_raw_stream(skinny_current_device()))
            if args.aiter_skinny == "wv":
                skinny_native(weight_entry[1], skinny_convert(x), skinny_convert(output), m, props.multi_processor_count)
            else:
                skinny_native(weight_entry[1], skinny_convert(x), skinny_convert(output), 4)
            return output
        if args.aiter_skinny == "wv":
            aiter_skinny(operand.T, x, output, m, props.multi_processor_count)
        else:
            aiter_skinny(operand.T, x, output, 4)
        return output

    def graph_execute(x, operand, residual_operand=None, composite_recovery=None, **kwargs):
        # Single-threaded benchmark only. Preserve outer graph capture and autograd
        # by using the ordinary operator; production needs a bounded cache/lifetime policy.
        if (torch.cuda.is_current_stream_capturing() or x.requires_grad or operand.requires_grad
                or (residual_operand is not None and residual_operand.requires_grad)):
            return original_execute(x, operand, residual_operand, composite_recovery, **kwargs)
        key = (id(operand), id(residual_operand), id(composite_recovery), tuple(x.shape), x.dtype,
               x.device, torch.cuda.current_stream().cuda_stream, tuple(sorted(kwargs.items())))
        entry = graph_entries.get(key)
        if entry is None:
            static_x = torch.empty_like(x)
            static_x.copy_(x)
            original_execute(static_x, operand, residual_operand, composite_recovery, **kwargs)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_y = original_execute(static_x, operand, residual_operand, composite_recovery, **kwargs)
            entry = (static_x, static_y, graph, operand, residual_operand, composite_recovery)
            graph_entries[key] = entry
        static_x, static_y, graph = entry[:3]
        static_x.copy_(x)
        graph.replay()
        return static_y.clone()

    def inplace_correction_execute(x, operand, residual_operand=None, composite_recovery=None, **kwargs):
        if (composite_recovery is not None or residual_operand is None or x.shape[0] < 64
                or (torch.is_grad_enabled() and any(t.requires_grad for t in (x, operand, residual_operand)))):
            return original_execute(x, operand, residual_operand, composite_recovery, **kwargs)
        primary = torch.mm(x, operand, out_dtype=torch.float32)
        torch.addmm(primary, x, residual_operand, out_dtype=torch.float32, out=primary)
        return primary if kwargs["output_fp32"] else primary.to(x.dtype)

    def fused_correction_execute(x, operand, residual_operand=None, composite_recovery=None, **kwargs):
        if (composite_recovery is not None or residual_operand is None or x.shape[0] < 64
                or not x.is_contiguous() or not operand.T.is_contiguous() or not residual_operand.T.is_contiguous()
                or (torch.is_grad_enabled() and any(t.requires_grad for t in (x, operand, residual_operand)))):
            return original_execute(x, operand, residual_operand, composite_recovery, **kwargs)
        m, k = x.shape
        n = operand.shape[1]
        output = torch.empty((m, n), device=x.device, dtype=torch.float32 if kwargs["output_fp32"] else x.dtype)
        kernel = folded_residual_prefetch_kernel if args.fused_prefetch else folded_residual_gemm_gluon_kernel
        kernel[(triton.cdiv(m, args.fused_block_m), triton.cdiv(n, args.fused_block_n))](
            x, operand, residual_operand, output, m, n, k,
            args.fused_block_m, args.fused_block_n, args.fused_block_k, args.fused_interleave, not args.fused_keep_masks,
            num_warps=4, num_stages=2,
        )
        return output

    def select(name):
        candidate_amd._qvq_p32_folded_execute = (
            fused_correction_execute if name == "candidate" and args.fused_correction else
            inplace_correction_execute if name == "candidate" and args.inplace_correction else
            graph_execute if name == "candidate" and args.graph_execute else
            skinny_execute if name == "candidate" and aiter_skinny is not None else original_execute
        )
        candidate_amd._qvq_p32_folded_gemv_gfx950_kernel = (
            full_k_gemv if name == "candidate" and (
                args.gemv_full_k or args.gemv_split_k or args.gemv_dot2 or args.gemv_dot2_loop or args.gemv_gluon)
            else original_gemv
        )
        candidate_amd._qvq_p32_composite_recovery_gfx950_kernel = (
            trimmed_recovery if name == "candidate" and args.trim_composite else original_recovery
        )
        sys.modules["gptqmodel.utils.qvq_amd"] = baseline_amd if name == "baseline" else candidate_amd
        QVQLinear._qvq_amd_folded_forward = baseline_forward if name == "baseline" else candidate_forward
        torch.mm = (candidate_mm if name == "candidate"
                    and (args.butterfly != "none" or hipb_mm is not None) else original_mm)

    shapes = QWEN38_27B_SHAPES if args.full_sweep else [("mlp_down", 17408, 5120)]
    m_values = REQUESTED_M if args.full_sweep else [1024]
    if args.shapes:
        shapes = [shape for shape in QWEN38_27B_SHAPES if shape[0] in args.shapes]
    if args.m_values:
        m_values = args.m_values
    permitted = set(hardware["process_ids"]) | set(
        _rocm_snapshot(args.physical_gpu)["process_ids"]
    )
    candidate_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    report = {
        "baseline": args.baseline_amd_commit or args.baseline_forward_commit or candidate_revision,
        "candidate_revision": candidate_revision,
        "isolated_baseline_caches": bool(args.baseline_amd_commit),
        "retained_kernel_baseline": "c89459e3",
        "hardware": hardware,
        "software": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": triton.__version__,
            "gpu": props.name,
            "arch": props.gcnArchName,
            "cu_count": props.multi_processor_count,
            "aiter_source": aiter.__file__ if aiter_skinny is not None else None,
            "aiter_jit_dir": os.environ.get("AITER_JIT_DIR") if aiter_skinny is not None else None,
        },
        "config": vars(args) | {"output": str(args.output),
                                "torch_profile_dir": str(args.torch_profile_dir) if args.torch_profile_dir else None},
        "valid": valid,
        "rows": [],
        "kernel_source_sha256": hashlib.sha256(
            (root / "scripts/qvq_p32_amd_butterfly_experiment.py").read_bytes()
        ).hexdigest(),
        "qvq_source_sha256": hashlib.sha256(
            (root / "gptqmodel/nn_modules/qlinear/qvq.py").read_bytes()
        ).hexdigest(),
        "amd_source_sha256": hashlib.sha256((root / "gptqmodel/utils/qvq_amd.py").read_bytes()).hexdigest(),
        "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prefetch_source_sha256": hashlib.sha256(
            (root / "scripts/qvq_p32_amd_prefetch_experiment.py").read_bytes()
        ).hexdigest(),
        "status": "exploratory; not production promotion or model-quality evidence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for (shape, k, n), bits in itertools.product(shapes, (2.0, 2.5, 3.0, 3.5)):
            graph_entries.clear()
            skinny_weights.clear()
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
            baseline_layer = copy.deepcopy(layer) if args.baseline_amd_commit else layer
            inner = layer.get_inner_weight_tensor()
            high = low = None
            if args.folded_residual_ceiling:
                folded = inner.T.contiguous()
                if ih:
                    folded = matmul_hadU(folded, transpose=True)
                if oh:
                    folded = matmul_hadU(folded.T.contiguous()).T.contiguous()
                high = folded.to(torch.float16).contiguous()
                low = (folded - high.float()).to(torch.float16).contiguous()
                del folded
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
                direct_cache = None
                if args.folded_direct_ceiling and qvq_p32_amd_folded_case_supported(m, k, n):
                    select("candidate")
                    layer(x)
                    direct_cache = layer._qvq_amd_folded_hot_cache
                def run(
                    name, input_tensor=x, high=high, low=low, layer=layer, baseline_layer=baseline_layer,
                    direct_cache=direct_cache, m=m, k=k, n=n,
                ):
                    if name == "candidate" and direct_cache is not None:
                        return candidate_amd._qvq_p32_folded_execute(
                            input_tensor, direct_cache[24], direct_cache[25], direct_cache[26],
                            out_features=n,
                            output_fp32=candidate_amd.qvq_p32_amd_folded_prefers_fp32_output(m, k, n),
                        ).to(input_tensor.dtype)
                    if name == "candidate" and args.folded_residual_ceiling:
                        if args.fused_correction:
                            return fused_correction_execute(input_tensor, high.T, low.T,
                                                            out_features=n, output_fp32=False)
                        primary = original_mm(input_tensor, high.T, out_dtype=torch.float32)
                        return torch.addmm(primary, input_tensor, low.T, out_dtype=torch.float32).to(torch.float16)
                    return (baseline_layer if name == "baseline" else layer)(input_tensor)
                outputs = {}
                for name, fn in [
                    ("baseline", original_mm),
                    ("candidate", candidate_mm),
                ]:
                    select(name)
                    outputs[name] = run(name)
                    for _ in range(args.warmup):
                        run(name)
                torch.cuda.synchronize()
                if args.torch_profile_dir:
                    args.torch_profile_dir.mkdir(parents=True, exist_ok=True)
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
                        for name in ("baseline", "candidate"):
                            select(name)
                            with torch.profiler.record_function(f"qvq_{name}"):
                                run(name)
                            torch.cuda.synchronize()
                    prof.export_chrome_trace(str(args.torch_profile_dir / f"{shape}_w{bits}_m{m}.json"))
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
                    run(name)
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
                row["fp16_reference_rounding_floor"] = (ref.to(torch.float16).float() - ref).abs().max().item()
                row["raw_candidate"] = args.folded_residual_ceiling or direct_cache is not None
                row["candidate_weight_cache_bytes"] = 4 * k * n if args.folded_residual_ceiling else None
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
                    if args.folded_residual_ceiling or qvq_p32_amd_folded_case_supported(m, k, n)
                    else "exact_baseline"
                )
                row["accuracy_pass"] = (
                    row["candidate"]["max_abs"] <= 0.002
                    if row["accuracy_basis"] == "canonical_fp32"
                    else torch.equal(outputs["candidate"], outputs["baseline"])
                )
                row["exact_baseline_equal"] = torch.equal(outputs["candidate"], outputs["baseline"])
                if (args.graph_execute or args.inplace_correction
                        or (args.fused_correction and args.folded_residual_ceiling)
                        or ((k, n) == (6144, 5120) and m >= 64)):
                    select("candidate")
                    saved = outputs["candidate"].clone()
                    changed = run("candidate", -x)
                    row["fresh_input_max_abs"] = (changed.float() + ref).abs().max().item()
                    row["previous_output_unchanged"] = torch.equal(saved, outputs["candidate"])
                    row["accuracy_pass"] &= row["fresh_input_max_abs"] <= 0.002
                    row["accuracy_pass"] &= row["previous_output_unchanged"]
                    row["graph_staging_bytes"] = sum(
                        e[0].numel() * e[0].element_size() + e[1].numel() * e[1].element_size()
                        for e in graph_entries.values()
                    )
                    del saved, changed
                if aiter_skinny is not None and not qvq_p32_amd_folded_case_supported(m, k, n):
                    row["graph_check_status"] = "unchanged fallback: host validation is not graph-capture-safe"
                if (args.graph_execute or (aiter_skinny is not None and m <= 4) or args.gemv_dot2
                        or (args.fused_correction and args.folded_residual_ceiling)
                        or ((args.gemv_dot2_loop or args.gemv_gluon) and m == 1)
                        or ((k, n) == (17408, 5120) and m >= 1024)
                        or ((k, n) == (6144, 5120) and m >= 64)):
                    select("candidate")
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        xs = x.clone()
                        ys = run("candidate", xs)
                    torch.cuda.current_stream().wait_stream(stream)
                    row["stream_equal"] = torch.equal(ys, outputs["candidate"])
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        yg = run("candidate")
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
                del x, ref, outputs, run
            del layer, baseline_layer, packed, banks, inner, high, low
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
        candidate_amd._qvq_p32_composite_recovery_gfx950_kernel = original_recovery
        candidate_amd._qvq_p32_folded_gemv_gfx950_kernel = original_gemv
        candidate_amd._qvq_p32_folded_execute = original_execute
        graph_entries.clear()
        skinny_weights.clear()
        QVQLinear._qvq_amd_folded_forward = candidate_forward
        sys.modules["gptqmodel.utils.qvq_amd"] = candidate_amd
        if baseline_source_dir is not None:
            baseline_source_dir.cleanup()


if __name__ == "__main__":
    main()
