#!/usr/bin/env python3
"""Paired complete small YAQA solves; synthetic kernel evidence, not model quality."""

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
    parser.add_argument("--idle-interval", type=float, default=1)
    parser.add_argument("--idle-memory-tolerance-mib", type=int, default=1024)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--bits", type=float, nargs="+", default=[2, 2.5, 3, 3.5])
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 2 or any(size < 16 or size % 16 for size in args.sizes):
        parser.error("Require two or more iterations and positive multiples of 16")
    args.allow_busy = False
    hardware, valid = _idle_preflight(args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["HIP_VISIBLE_DEVICES"] = str(args.physical_gpu)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch

    from gptqmodel.quantization.qvq import yaqa_inner_v2b2_p32
    from gptqmodel.quantization.qvq_codecs import pgc16_codebook_v2_bank

    report = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "scope": "complete YAQA family reselect/full-sampling solve, synthetic SPD Hessians and weights; "
                       "not model calibration, preprocessing, serialization or end-to-end model quantization",
              "config": vars(args) | {"output": str(args.output)}, "hardware": hardware,
              "valid": valid, "completed": False, "rows": [],
              "torch": torch.__version__, "hip": torch.version.hip,
              "source_sha256": {str(path): hashlib.sha256((root / path).read_bytes()).hexdigest() for path in
                                ("gptqmodel/quantization/qvq.py", "gptqmodel/utils/qvq_yaqa_amd.py")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for size in args.sizes:
            for bits in args.bits:
                generator = torch.Generator(device="cuda").manual_seed(20260908 + size)
                weight = torch.randn((size, size), device="cuda", generator=generator)
                a = torch.randn((size, size), device="cuda", generator=generator)
                b = torch.randn((size, size), device="cuda", generator=generator)
                h_in = a @ a.T + torch.eye(size, device="cuda")
                h_out = b @ b.T + torch.eye(size, device="cuda")
                library = tuple(pgc16_codebook_v2_bank(bank, device="cuda", bits=bits) for bank in range(4))

                def run(weight=weight, h_in=h_in, h_out=h_out, library=library, bits=bits):
                    return yaqa_inner_v2b2_p32(weight, h_in, h_out, library, bits=bits)

                os.environ["GPTQMODEL_QVQ_AMD_NATIVE_QUANTIZATION"] = "0"
                expected = run()
                os.environ["GPTQMODEL_QVQ_AMD_NATIVE_QUANTIZATION"] = "1"
                actual = run()
                exact = all(torch.equal(a, e) for a, e in zip(actual, expected, strict=True))
                torch.cuda.synchronize()
                _timing_recheck(args, {os.getpid()})
                samples = {"eager": [], "native": []}
                for iteration in range(args.iterations):
                    for mode in (("eager", "native") if iteration % 2 == 0 else ("native", "eager")):
                        os.environ["GPTQMODEL_QVQ_AMD_NATIVE_QUANTIZATION"] = str(int(mode == "native"))
                        start = time.perf_counter()
                        actual = run()
                        torch.cuda.synchronize()
                        samples[mode].append((time.perf_counter() - start) * 1000)
                        exact &= all(torch.equal(a, e) for a, e in zip(actual, expected, strict=True))
                row = {"in_features": size, "out_features": size, "bits": bits, "exact": exact,
                       "wall_ms": samples, "median_ms": {k: statistics.median(v) for k, v in samples.items()}}
                row["speedup"] = row["median_ms"]["eager"] / row["median_ms"]["native"]
                report["rows"].append(row)
                report["valid"] &= exact
                args.output.write_text(json.dumps(report, indent=2))
                print(f"YAQA {size}x{size} W{bits:g}: {row['speedup']:.2f}x exact={exact}", flush=True)
        report["completed"] = True
    except Exception as exc:
        report["valid"] = False
        report["error"] = repr(exc)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
