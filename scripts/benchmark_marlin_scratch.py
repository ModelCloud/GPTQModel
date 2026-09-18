#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Paired, allocation-audited Marlin temporary-buffer benchmark.

No GPU is required for --list-cases or to emit an explicit not_run report.
Graph trials use exclusively owned explicit scratch; the eager context rejects capture.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

M_VALUES = (1, 8, 16, 17, 32, 64, 128, 512, 2048)
SHAPES = (("square", 4096, 4096), ("narrow", 4096, 1024),
          ("long_k", 14336, 4096), ("padded", 4128, 4088))


def matrix(dtypes, methods, m_values):
    return [dict(shape=name, k=k, n=n, m=m, dtype=dtype, method=method,
                 act_order=act, fp32=True)
            for dtype in dtypes for method in methods
            for name, k, n in SHAPES
            for act in ((False, True) if method == "gptq" and name != "padded" else (False,))
            for m in m_values]


def measure(fn, torch, device, iters):
    """Report CPU enqueue, synchronized end-to-end, and GPU event span separately."""
    torch.cuda.synchronize(device)
    start = time.perf_counter_ns()
    for _ in range(iters):
        fn()
    submitted = time.perf_counter_ns()
    torch.cuda.synchronize(device)
    completed = time.perf_counter_ns()
    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
             for _ in range(iters)]
    for a, b in pairs:
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize(device)
    return dict(cpu_submit_us=(submitted - start) / iters / 1000,
                forward_us=(completed - start) / iters / 1000,
                gpu_span_us=statistics.median(a.elapsed_time(b) * 1000 for a, b in pairs))


def allocation_audit(fn, torch, device, path, iters=5):
    """Allocator history counts allocation requests, separately from driver mallocs."""
    torch.cuda.synchronize(device)
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    history_error = None
    try:
        torch.cuda.memory._record_memory_history(enabled="all", context=None, stacks="python",
                                                 max_entries=100000)
    except Exception as exc:
        history_error = str(exc)
    try:
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                profile_memory=True) as prof:
            if history_error is None:
                history_start = len(torch.cuda.memory._snapshot()["device_traces"][device_index])
            for _ in range(iters):
                fn()
            torch.cuda.synchronize(device)
            if history_error is None:
                allocation_events = torch.cuda.memory._snapshot()["device_traces"][device_index][history_start:]
        prof.export_chrome_trace(str(path))
        trace = json.loads(path.read_text())
        events = trace["traceEvents"]
        kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
        driver_allocs = [e for e in events if e.get("ph") == "X" and
                        any(s in e.get("name", "") for s in ("cudaMalloc", "cuMemCreate", "cuMemAlloc"))]
        result = dict(kernel_us_per_call=sum(e.get("dur", 0) for e in kernels) / iters,
                      kernel_count_per_call=len(kernels) / iters,
                      profiled_calls=iters,
                      driver_alloc_events=len(driver_allocs), profiler_trace=str(path),
                      driver_alloc_events_per_call=len(driver_allocs) / iters,
                      aten_empty_calls=sum(e.count for e in prof.key_averages() if e.key == "aten::empty"))
        if history_error is None:
            allocs = [e for e in allocation_events if e["action"] == "alloc"]
            result.update(allocator_requests_per_call=len(allocs) / iters,
                          allocator_requested_bytes_per_call=sum(e["size"] for e in allocs) / iters)
        else:
            result["allocator_history_unavailable"] = history_error
        return result
    finally:
        if history_error is None:
            torch.cuda.memory._record_memory_history(enabled=None)


def graph_trial(layer, x, buffers, torch, utils, args, trace_path):
    """Sequential graphs use the same kernel settings and module lock workspace."""
    original = utils.gptq_marlin_gemm
    def gemm(*positional, **kwargs):
        return original(*positional, **kwargs, c_tmp=buffers[0], a_tmp=buffers[1])
    manager = patch.object(utils, "gptq_marlin_gemm", gemm) if buffers else contextlib.nullcontext()
    with manager:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(args.warmup):
                layer(x)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(x.device)
        capture_start = time.perf_counter_ns()
        with torch.cuda.graph(graph):
            result = layer(x)
        torch.cuda.synchronize(x.device)
        capture_us = (time.perf_counter_ns() - capture_start) / 1000
        for _ in range(10):
            x.add_(.001)
            graph.replay()
            eager = layer(x)
            torch.testing.assert_close(result, eager, rtol=.05, atol=.05)
        data = measure(graph.replay, torch, x.device, args.iters)
        torch.cuda.reset_peak_memory_stats(x.device)
        audit = allocation_audit(graph.replay, torch, x.device, trace_path)
        data.update(capture_us=capture_us, allocation_audit=audit,
                    resident_process_bytes=torch.cuda.memory_allocated(x.device),
                    peak_process_bytes=torch.cuda.max_memory_allocated(x.device),
                    explicit_scratch_bytes=sum(b.numel() * b.element_size() for b in buffers) if buffers else 0)
        # Owners remain alive until queued replay finishes; no concurrently replayed graphs.
        torch.cuda.synchronize(x.device)
        return data


def run_case(case, torch, utils, args, artifact):
    from scripts.marlin_scratch_fixture import gemm_arguments, make_layer
    dtype = torch.float16 if case["dtype"] == "fp16" else torch.bfloat16
    device = torch.device(args.device)
    major, minor = torch.cuda.get_device_capability(device)
    if ((major, minor) < (7, 5) or
            ((case["method"] == "awq" or dtype == torch.bfloat16) and major < 8)):
        return dict(**case, status="excluded", reason="unsupported device/dtype")
    layer, dense, bias = make_layer(case["method"], dtype, case["k"], case["n"],
                                   act_order=case["act_order"], device=device, bias=False, seed=args.seed)
    layer.fp32 = True
    generator = torch.Generator(device=device).manual_seed(args.seed)
    x = torch.randn(case["m"], case["k"], device=device, dtype=dtype,
                    generator=generator) / case["k"]**.5
    expected = x @ dense
    torch.testing.assert_close(layer(x), expected, rtol=.05, atol=.05)
    del expected, dense, bias
    context = utils.MarlinScratchContext(device, max_cached_bytes=args.cache_bytes)
    samples = {"temporary": [], "context": []}
    # Capture compilation separately; first/growth costs below exclude JIT.
    for _ in range(args.warmup):
        layer(x)
    initial = {}
    with context:
        for label, value in (("first_request", x[:1]), ("growth", x)):
            initial[label] = measure(lambda: layer(value), torch, device, 1)
        expected = layer(x)
    torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
    del expected
    audit = {}
    rng = random.Random(args.seed)
    for trial in range(args.rounds):
        order = ["temporary", "context"]
        rng.shuffle(order)
        for variant in order:
            with context if variant == "context" else contextlib.nullcontext():
                for _ in range(args.warmup):
                    layer(x)
                samples[variant].append(measure(lambda: layer(x), torch, device, args.iters))
                if trial == 0:
                    torch.cuda.synchronize(device)
                    before = torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    measured = allocation_audit(lambda: layer(x), torch, device,
                                                artifact / f"{variant}.trace.json")
                    measured.update(resident_process_bytes=before,
                                    peak_process_bytes=torch.cuda.max_memory_allocated(device))
                    audit[variant] = measured
    median = {v: {metric: statistics.median(r[metric] for r in rows)
                  for metric in rows[0]} for v, rows in samples.items()}
    ratios = [a["forward_us"] / b["forward_us"]
              for a, b in zip(samples["temporary"], samples["context"])]
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    context.clear()
    cache_resident = before - torch.cuda.memory_allocated(device)
    audit["temporary"].update(
        inactive_context_bytes_in_process=cache_resident,
        resident_excluding_inactive_context_bytes=audit["temporary"]["resident_process_bytes"] - cache_resident,
        peak_excluding_inactive_context_bytes=audit["temporary"]["peak_process_bytes"] - cache_resident)
    gemm = gemm_arguments(layer, x)
    c_count, a_count = utils.marlin_scratch_sizes(
        gemm["a"], gemm["size_m"], gemm["size_k"], True, case["act_order"])
    requirements = dict(executed_m=gemm["size_m"], executed_k=gemm["size_k"],
                        executed_n=gemm["size_n"], c_tmp_elements=c_count,
                        a_tmp_elements=a_count, c_tmp_bytes=c_count * 4,
                        a_tmp_bytes=a_count * x.element_size())
    del gemm
    graph = {}
    if args.graph:
        original_input = x.clone()
        graph["temporary"] = graph_trial(layer, x, None, torch, utils, args,
                                         artifact / "graph-temporary.trace.json")
        x.copy_(original_input)
        explicit = (torch.empty(c_count, device=device, dtype=torch.float32),
                    torch.empty(a_count, device=device, dtype=dtype))
        graph["explicit_scratch"] = graph_trial(layer, x, explicit, torch, utils, args,
                                                artifact / "graph-explicit.trace.json")
        x.copy_(original_input)
    return dict(**case, status="complete", samples=samples, medians=median,
                forward_speedup_median=statistics.median(ratios),
                forward_speedup_min=min(ratios), forward_speedup_max=max(ratios),
                allocation_audit=audit, context_resident_bytes=cache_resident,
                initial=initial, graph=graph, scratch_requirements=requirements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "both"), default="both")
    parser.add_argument("--method", choices=("gptq", "awq", "both"), default="both")
    parser.add_argument("--m-values", default=",".join(map(str, M_VALUES)))
    parser.add_argument("--cache-bytes", type=int, default=64 << 20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/tmp/marlin-scratch-benchmark"))
    args = parser.parse_args()
    if min(args.iters, args.rounds, args.warmup) < 1 or args.cache_bytes < 0:
        parser.error("iterations, rounds and warmup must be positive; cache bytes nonnegative")
    cases = matrix(("fp16", "bf16") if args.dtype == "both" else (args.dtype,),
                   ("gptq", "awq") if args.method == "both" else (args.method,),
                   tuple(map(int, args.m_values.split(","))))
    if any(c["m"] <= 0 for c in cases):
        parser.error("M values must be positive")
    if args.list_cases:
        print(json.dumps(cases, indent=2))
        return 0
    import torch
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(command=sys.argv, python=platform.python_version(), torch=torch.__version__,
                    torch_cuda=torch.version.cuda, cache_bytes=args.cache_bytes,
                    seed=args.seed, protocol="same layer/input/dtype/dispatch/FP32; randomized paired eager trials",
                    graph_scope="sequential exclusive explicit scratch; eager manager disallows capture",
                    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    rows=[])
    sources = [Path(__file__), ROOT / "scripts/marlin_scratch_fixture.py",
               ROOT / "gptqmodel/utils/marlin.py", ROOT / "gptqmodel/utils/marlin_scratch.py",
               *sorted((ROOT / "gptqmodel_ext/marlin").glob("*scratch*")),
               *sorted((ROOT / "gptqmodel_ext/marlin").glob("gptq_marlin*")),
               *sorted((ROOT / "gptqmodel_ext/marlin").glob("marlin_torch*"))]
    manifest["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sources if p.is_file()}
    (args.output / "source.diff").write_text(subprocess.check_output(["git", "diff"], cwd=ROOT, text=True))
    if not torch.cuda.is_available():
        manifest["rows"] = [dict(**case, status="not_run", reason="CUDA unavailable") for case in cases]
    else:
        from gptqmodel.utils import marlin as utils
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        torch.manual_seed(args.seed)
        props = torch.cuda.get_device_properties(device)
        manifest["gpu"] = dict(name=props.name, uuid=str(getattr(props, "uuid", None)), sms=props.multi_processor_count,
                               memory=props.total_memory, capability=torch.cuda.get_device_capability(device))
        manifest["nvidia_smi"] = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
        with torch.inference_mode():
            for index, case in enumerate(cases):
                artifact = args.output / f"case-{index:04d}"
                artifact.mkdir(exist_ok=True)
                try:
                    row = run_case(case, torch, utils, args, artifact)
                except Exception as exc:
                    row = dict(**case, status="failed", error=repr(exc))
                manifest["rows"].append(row)
                (args.output / "results.json").write_text(json.dumps(manifest, indent=2))
                print(f"{index + 1}/{len(cases)} {case}: {row['status']}", flush=True)
    (args.output / "results.json").write_text(json.dumps(manifest, indent=2))
    print(f"Results: {args.output / 'results.json'}; "
          f"complete={sum(r['status'] == 'complete' for r in manifest['rows'])}, "
          f"not_run={sum(r['status'] == 'not_run' for r in manifest['rows'])}, "
          f"failed={sum(r['status'] == 'failed' for r in manifest['rows'])}")
    return 1 if any(r["status"] == "failed" for r in manifest["rows"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
