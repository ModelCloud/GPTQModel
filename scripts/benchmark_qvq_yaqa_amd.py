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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch_sizes) < 1 or args.warmup < 1 or args.iterations < 2:
        parser.error("Require positive batches/warmup and at least two timing iterations")
    args.allow_busy = False
    hardware, valid = _idle_preflight(args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["HIP_VISIBLE_DEVICES"] = str(args.physical_gpu)
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
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "hardware": hardware,
        "software": {"torch": torch.__version__, "hip": torch.version.hip, "gpu": props.name,
                     "arch": props.gcnArchName, "cu_count": props.multi_processor_count},
        "config": vars(args) | {"output": str(args.output),
                                "profile_dir": str(args.profile_dir) if args.profile_dir else None},
        "scope": "eager banked Viterbi baseline, not full YAQA or model-quality evidence",
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

                def run(sequences=sequences, codebooks=codebooks, bits=bits):
                    return batched_v2b2_p32_viterbi_quantize(sequences, codebooks, bits=bits)

                expected = run()
                for _ in range(args.warmup):
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
                if args.profile_dir:
                    _timing_recheck(args, {os.getpid()})
                    args.profile_dir.mkdir(parents=True, exist_ok=True)
                    with (
                        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                          torch.profiler.ProfilerActivity.CUDA]) as prof,
                        torch.profiler.record_function("qvq_banked_torch_reference"),
                    ):
                        run()
                        torch.cuda.synchronize()
                    trace = args.profile_dir / f"b{batch}_w{bits}.json"
                    prof.export_chrome_trace(str(trace))
                    row["trace"] = str(trace)
                report["rows"].append(row)
                report["valid"] &= exact
                args.output.write_text(json.dumps(report, indent=2))
                print(f"B={batch} W{bits:g} eager={row['event_median_ms']:.3f}ms "
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
