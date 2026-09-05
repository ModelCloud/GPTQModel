#!/usr/bin/env python3
"""Establish the eager ROCm banked-Viterbi baseline for native YAQA work.

Synthetic fixtures measure kernel behavior only, not YAQA model quality.
The baseline includes emission, recurrence, bank transitions and traceback.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from benchmark_qvq_p32_amd import _idle_preflight, _timing_recheck


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-gpu", type=int, default=0)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=1.0)
    parser.add_argument("--idle-memory-tolerance-mib", type=int, default=1024)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--bits", type=float, nargs="+", choices=[2, 2.5, 3, 3.5], default=[2, 2.5, 3, 3.5])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--native", action="store_true", help="Compare experimental native recurrence with eager")
    parser.add_argument("--public-native", action="store_true", help="Include public validation and dispatch")
    parser.add_argument("--q-chunk", type=int, choices=[4, 8, 16, 32], default=4)
    parser.add_argument("--graph", action="store_true", help="Time native replay separately from preparation")
    parser.add_argument("--codebook-dtype", choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--closed", action="store_true")
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch_sizes) < 1 or args.warmup < 1 or args.iterations < 2:
        parser.error("Require positive batches/warmup and at least two timing iterations")
    if args.graph and not args.native:
        parser.error("Experimental launch modes require --native")
    if args.public_native and (not args.native or args.graph or args.q_chunk != 4):
        parser.error("--public-native requires --native, no graph, and the default q-chunk")
    args.allow_busy = False
    hardware, valid = _idle_preflight(args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["HIP_VISIBLE_DEVICES"] = str(args.physical_gpu)
    # The oracle must remain eager even when the caller's environment opts in.
    os.environ["GPTQMODEL_QVQ_AMD_NATIVE_QUANTIZATION"] = "0"
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch

    from gptqmodel.quantization.qvq import batched_v2b2_p32_viterbi_quantize

    props = torch.cuda.get_device_properties(0)
    if torch.version.hip is None or props.gcnArchName.split(":")[0] != "gfx950":
        raise RuntimeError("This baseline requires the target gfx950 ROCm GPU")
    report = {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_sha256": hashlib.sha256((root / "gptqmodel/quantization/qvq.py").read_bytes()).hexdigest(),
        "native_source_sha256": hashlib.sha256((root / "gptqmodel/utils/qvq_yaqa_amd.py").read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "hardware": hardware,
        "software": {"torch": torch.__version__, "hip": torch.version.hip, "gpu": props.name,
                     "arch": props.gcnArchName, "cu_count": props.multi_processor_count},
        "config": vars(args) | {"output": str(args.output),
                                "profile_dir": str(args.profile_dir) if args.profile_dir else None},
        "scope": "banked Viterbi: --public-native includes public validation; other native modes use trusted inputs; "
                 "graph replay excludes preparation; not full YAQA or model-quality evidence",
        "valid": valid, "completed": False, "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for batch in args.batch_sizes:
            for bits in args.bits:
                seed = 20260905 + batch * 100 + int(bits * 10)
                generator = torch.Generator(device="cuda").manual_seed(seed)
                sequences = torch.randn((batch, 128, 2), generator=generator, device="cuda", dtype=torch.float32)
                codebooks = torch.randn((2, 65536, 2), generator=generator, device="cuda", dtype=torch.float32)
                codebooks = codebooks.to(torch.float16 if args.codebook_dtype == "fp16" else torch.float32)
                overlap = torch.full((batch,), 13, device="cuda", dtype=torch.int64) if args.closed else None
                weights = torch.rand((batch, 128), generator=generator, device="cuda") if args.weighted else None

                def run(sequences=sequences, codebooks=codebooks, bits=bits, overlap=overlap, weights=weights):
                    return batched_v2b2_p32_viterbi_quantize(sequences, codebooks, bits=bits,
                                                          overlap=overlap, step_weights=weights)

                expected = run()
                reference_run = run
                if args.native:
                    from gptqmodel.utils.qvq_yaqa_amd import banked_viterbi_trusted

                    def run(sequences=sequences, codebooks=codebooks, bits=bits, overlap=overlap, weights=weights):
                        return banked_viterbi_trusted(sequences, codebooks, bits=bits, q_chunk=args.q_chunk,
                                                     overlap=overlap, step_weights=weights)
                if args.public_native:
                    def run(reference_run=reference_run):
                        os.environ["GPTQMODEL_QVQ_AMD_NATIVE_QUANTIZATION"] = "1"
                        try:
                            return reference_run()
                        finally:
                            os.environ["GPTQMODEL_QVQ_AMD_NATIVE_QUANTIZATION"] = "0"
                for _ in range(args.warmup):
                    run()
                torch.cuda.synchronize()
                if args.graph:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        graph_output = run()

                    def run(graph=graph, graph_output=graph_output):
                        graph.replay()
                        return graph_output

                    run()
                    torch.cuda.synchronize()
                _timing_recheck(args, {os.getpid()})
                event_ms, wall_ms = [], []
                exact = True
                for _ in range(args.iterations):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    wall_start = time.perf_counter()
                    start.record()
                    actual = run()
                    end.record()
                    end.synchronize()
                    wall_ms.append((time.perf_counter() - wall_start) * 1000)
                    event_ms.append(start.elapsed_time(end))
                    exact &= all(torch.equal(getattr(actual, name), getattr(expected, name)) for name in
                                 ("states", "values", "squared_error", "segment_bank_ids"))
                row = {"batch": batch, "steps": 128, "vector_size": 2, "banks": 2, "states": 65536,
                       "bits": bits, "seed": seed, "repeat_exact": exact,
                       "event_median_ms": statistics.median(event_ms), "event_samples_ms": event_ms,
                       "wall_median_ms": statistics.median(wall_ms), "wall_samples_ms": wall_ms,
                       "reference_sha256": {name: hashlib.sha256(
                           getattr(expected, name).cpu().contiguous().numpy().tobytes()).hexdigest()
                           for name in ("states", "values", "squared_error", "segment_bank_ids")}}
                if args.public_native:
                    reference_ms = []
                    for _ in range(args.iterations):
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        reference_run()
                        end.record()
                        end.synchronize()
                        reference_ms.append(start.elapsed_time(end))
                    row["eager_event_samples_ms"] = reference_ms
                    row["eager_event_median_ms"] = statistics.median(reference_ms)
                    row["public_speedup"] = row["eager_event_median_ms"] / row["event_median_ms"]
                if args.profile_dir:
                    _timing_recheck(args, {os.getpid()})
                    args.profile_dir.mkdir(parents=True, exist_ok=True)
                    with (
                        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                          torch.profiler.ProfilerActivity.CUDA]) as prof,
                        torch.profiler.record_function("qvq_banked_native" if args.native else "qvq_banked_torch_reference"),
                    ):
                        run()
                        torch.cuda.synchronize()
                    trace = args.profile_dir / f"b{batch}_w{bits}.json"
                    prof.export_chrome_trace(str(trace))
                    row["trace"] = str(trace)
                report["rows"].append(row)
                report["valid"] &= exact
                args.output.write_text(json.dumps(report, indent=2))
                mode = "native" if args.native else "eager"
                print(f"B={batch} W{bits:g} {mode}={row['event_median_ms']:.3f}ms "
                      f"wall={row['wall_median_ms']:.3f}ms exact={exact}", flush=True)
        report["completed"] = True
    except Exception as exc:
        report["valid"] = False
        report["error"] = repr(exc)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
