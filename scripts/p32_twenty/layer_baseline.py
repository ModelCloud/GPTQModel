# ruff: noqa: B023
# Timing callbacks execute synchronously within their current loop iteration; none escape.
"""Matched-device P32 layer timing on captured real FP32-teacher activations."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SNAPSHOT = Path(
    "/root/qvq-results/calibration-fisher-frontier-wave14-v1/llama32-1b-f6_yaqa125x_seed7"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--fused-block-m", type=int, choices=(16,32,64))
    parser.add_argument("--fused-block-n", type=int, choices=(32,64), default=32)
    parser.add_argument("--fused-split", type=int, default=1)
    parser.add_argument("--module")
    parser.add_argument("--row-reuse", type=int, choices=(1, 2, 4, 8, 16))
    parser.add_argument(
        "--m-values", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 128, 512, 2048]
    )
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(SNAPSHOT):
        parser.error("Output must be outside the snapshot")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.uuid
    for sample in range(3):
        row = subprocess.check_output(
            [
                "nvidia-smi",
                "--id=" + args.uuid,
                "--query-gpu=uuid,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        f = [s.strip() for s in row.split(",")]
        procs = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        if f[0] != args.uuid or int(f[1]) > 8 or int(f[2]) != 0 or args.uuid in procs:
            raise RuntimeError("Idle preflight failed: " + row)
        print("IDLE", sample + 1, row, flush=True)
        time.sleep(1)
    import torch
    from safetensors import safe_open

    from gptqmodel.quantization.qvq import (
        decode_p32_window_tiles,
        reconstruct_qvq_inner_weight,
        repack_p32_planar_to_window,
        unpack_p32_window_states,
    )
    from gptqmodel.quantization.qvq_codecs import pgc16_levels_for_version
    from gptqmodel.quantization.rotation.hadamard_utils import matmul_hadU
    from gptqmodel.utils.qvq_ampere_cuda import qvq_p32_window_ampere
    from gptqmodel.utils.qvq_cuda import qvq_cuda_gemv
    from scripts.p32_twenty.scorecard import layer_metrics

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")
    cfg = json.loads((SNAPSHOT / "quantize_config.json").read_text())
    idx = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    def read(name):
        with safe_open(
            str(SNAPSHOT / idx[name]), framework="pt", device="cpu"
        ) as handle:
            return handle.get_tensor(name).to("cuda")

    def exclusive():
        procs = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        for row in procs.splitlines():
            f = [s.strip() for s in row.split(",")]
            if f[0] == args.uuid and int(f[1]) != os.getpid():
                raise RuntimeError("Foreign process before timed region")

    def timing(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        exclusive()
        samples = []
        for _ in range(20):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            out = fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
            del out
        return {"median_ms": sorted(samples)[10], "samples_ms": samples}

    captured = json.loads(
        Path("/root/p32-timing-activations/capture.json").read_text()
    )["modules"]
    modules = [m for m in captured if m["module"] + ".bank_alt_id" in idx]
    report = {
        "scope": "layer baseline and window candidate; experiment 1/10 partial evidence; profiler and model gates pending",
        "uuid": args.uuid,
        "worker": args.worker,
        "torch": torch.__version__,
        "device": str(torch.cuda.get_device_properties(0)),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected = [item for item in modules if item["module"] == args.module] if args.module else modules[args.worker :: 4]
    if not selected:
        raise ValueError(f"No captured P32 modules match {args.module!r}")
    for item in selected:
        prefix = item["module"]
        cap = torch.load(item["path"], weights_only=True)
        xall = cap["input"].reshape(-1, cap["input"].shape[-1]).cuda()
        t, su, sv, bank, alt = (
            read(prefix + "." + k)
            for k in ("trellis", "SU", "SV", "bank_ids", "bank_alt_id")
        )
        bits = t.shape[-1] / 8
        N = sv.numel()
        K = su.numel()
        aid = alt.item()
        levels = pgc16_levels_for_version(cfg["codebook"]).cuda().contiguous()
        window = repack_p32_planar_to_window(t, bits=bits)
        inner = reconstruct_qvq_inner_weight(
            t,
            bits=bits,
            in_features=K,
            out_features=N,
            bank_ids=bank,
            bank_alt_id=alt,
            v2b2_p32=True,
        )
        decode_stages = (
            {}
            if args.profile
            else {
                "extract_states": timing(
                    lambda: unpack_p32_window_states(window, bits=bits)
                ),
                "extract_banks_lookup": timing(
                    lambda: decode_p32_window_tiles(
                        window, bits=bits, bank_ids=bank, bank_alt_id=alt
                    )
                ),
            }
        )
        for m in args.m_values:
            x = xall[:m].float()
            transformed = matmul_hadU(x * su.float()).half().contiguous()

            def planar(z):
                return qvq_cuda_gemv(
                    z,
                    t,
                    bits,
                    out_features=N,
                    output_fp32=True,
                    bank_ids=bank,
                    v2b2_p32=True,
                    bank_alt_id=aid,
                )

            def ampere(z):
                if args.row_reuse and z.shape[0] > args.row_reuse:
                    return torch.cat([
                        qvq_p32_window_ampere(
                            chunk, window, levels, bank, bits,
                            out_features=N, bank_alt_id=aid,
                        ) for chunk in z.split(args.row_reuse)
                    ], dim=0)
                return qvq_p32_window_ampere(
                    z, window, levels, bank, bits, out_features=N, bank_alt_id=aid
                )

            def full(fn):
                z = matmul_hadU(x * su.float()).half().contiguous()
                return matmul_hadU(fn(z).float()) * sv.float()

            teacher = (
                matmul_hadU(matmul_hadU(x * su.float()) @ inner.float()) * sv.float()
            )
            base = full(planar)
            candidate = full(ampere)
            if args.profile:
                for _ in range(10):
                    planar(transformed)
                    ampere(transformed)
                torch.cuda.synchronize()
                exclusive()
                torch.cuda.cudart().cudaProfilerStart()
                planar(transformed)
                ampere(transformed)
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
                report["rows"].append(
                    {
                        "module": prefix,
                        "M": m,
                        "K": K,
                        "N": N,
                        "bits": bits,
                        "baseline_metrics": layer_metrics(base, teacher),
                        "candidate_metrics": layer_metrics(candidate, teacher),
                        "profile_scope": "one planar and one Ampere inner call after warmup",
                    }
                )
                continue
            row = {
                "module": prefix,
                "M": m,
                "K": K,
                "N": N,
                "bits": bits,
                "row_reuse_limit": args.row_reuse,
                "row_reuse_scope": "existing-kernel row-group ablation; includes extra launches and concatenation",
                "output_dtype": "float32",
                "operand_dtype": "float16",
                "transforms_dtype": "float32",
                "baseline_metrics": layer_metrics(base, teacher),
                "candidate_metrics": layer_metrics(candidate, teacher),
                "candidate_vs_baseline": layer_metrics(candidate, base),
                "decode_stages": decode_stages,
                "input_scale": timing(lambda: x * su.float()),
                "input_hadamard": timing(lambda: matmul_hadU(x)),
                "baseline_inner": timing(lambda: planar(transformed)),
                "candidate_inner": timing(lambda: ampere(transformed)),
                "baseline_full": timing(lambda: full(planar)),
                "candidate_full": timing(lambda: full(ampere)),
                "dense_inner_fp32": timing(lambda: transformed.float() @ inner.float()),
                "payload_bytes": t.numel() * t.element_size(),
                "resident_reference_bytes": inner.numel() * inner.element_size(),
            }
            if args.fused_block_m:
                from scripts.p32_twenty.fused_window_gemm import fused_window_mm

                def fused(z):
                    return fused_window_mm(
                        z, window, levels, bank, bits, out_features=N,
                        bank_alt_id=aid, block_m=args.fused_block_m,
                        block_n=args.fused_block_n, split=args.fused_split,
                    )

                fused_output = full(fused)
                row["fused"] = {
                    "block_m": args.fused_block_m,
                    "block_n": args.fused_block_n,
                    "split": args.fused_split,
                    "metrics": layer_metrics(fused_output, teacher),
                    "vs_window": layer_metrics(fused_output, candidate),
                    "inner": timing(lambda: fused(transformed)),
                    "full": timing(lambda: full(fused)),
                }
                row["fused"]["speedup_vs_window"] = (
                    row["candidate_full"]["median_ms"] / row["fused"]["full"]["median_ms"]
                )
            row["full_speedup"] = (
                row["baseline_full"]["median_ms"] / row["candidate_full"]["median_ms"]
            )
            report["rows"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                "CASE",
                prefix,
                m,
                "speedup",
                row["full_speedup"],
                "gate",
                row["candidate_metrics"]["local_tolerance_pass"],
                flush=True,
            )
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
