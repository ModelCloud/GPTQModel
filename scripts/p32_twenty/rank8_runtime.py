"""Rank8 fused correction expansion/add and blockwise-FP32 input projection.

Native W4A16 execution is unchanged and remains a separate call. This is NOT
fusion into the native GEMM itself. Both eager and CUDA Graph timings include
native base, correction, conversions and final FP16 output.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uuid", required=True)
    parser.add_argument(
        "--export",
        type=Path,
        default=Path("/root/p32-low-rank/worker2/rank8-tail-float16.pt"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-m", type=int, default=16)
    args = parser.parse_args()
    from scripts.p32_twenty.low_rank_sweep import SNAPSHOT, sha

    if args.output.resolve().is_relative_to(SNAPSHOT.resolve()):
        parser.error("External output required")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.uuid
    for _ in range(3):
        row = subprocess.check_output(
            [
                "nvidia-smi",
                "--id=" + args.uuid,
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).split(",")
        apps = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        if int(row[0]) > 8 or int(row[1]) or args.uuid in apps:
            raise RuntimeError("GPU not idle")
        time.sleep(1)
    import torch
    import triton
    import triton.language as tl
    from safetensors import safe_open

    from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
    from gptqmodel.quantization.qvq import (
        reconstruct_qvq_inner_weight,
        repack_p32_planar_to_window,
    )
    from gptqmodel.quantization.qvq_codecs import pgc16_levels_for_version
    from gptqmodel.quantization.rotation.hadamard_utils import matmul_hadU
    from gptqmodel.utils.qvq_ampere_cuda import qvq_p32_window_ampere
    from scripts.p32_twenty.recovered_linear import RecoveredLinear
    from scripts.p32_twenty.recovery_epilogue import expansion_add
    from scripts.p32_twenty.scorecard import layer_metrics

    @triton.jit
    def project_blockwise(
        X,
        A,
        H,
        M: tl.constexpr,
        K: tl.constexpr,
        R: tl.constexpr,
        PROMOTE: tl.constexpr,
    ):
        rows = tl.program_id(0) * 16 + tl.arange(0, 16)
        rank = tl.arange(0, 16)
        kk = tl.arange(0, 16)
        total = tl.full((16, 16), 0, tl.float32)
        partial = tl.full((16, 16), 0, tl.float32)
        for offset in range(0, K, 16):
            x = tl.load(
                X + rows[:, None] * K + offset + kk[None, :], rows[:, None] < M, 0
            )
            a = tl.load(
                A + (offset + kk[:, None]) * R + rank[None, :], rank[None, :] < R, 0
            )
            partial = tl.dot(x, a, partial, out_dtype=tl.float32)
            if (offset + 16) % PROMOTE == 0:
                total = total + partial
                partial = tl.full((16, 16), 0, tl.float32)
        tl.store(
            H + rows[:, None] * R + rank[None, :],
            total,
            (rows[:, None] < M) & (rank[None, :] < R),
        )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if torch.cuda.get_device_capability() != (8, 0):
        raise ValueError("sm80-only experiment")
    op = RecoveredLinear(args.export)
    module = op.source_module
    k, n = op.in_features, op.out_features
    r = op.a.shape[1]
    if (
        r not in [8, 12, 16]
        or op.a.dtype != torch.float16
        or op.b.dtype != torch.float16
        or k % 256
    ):
        raise ValueError("Expected FP16 rank8/12/16 factors and K256 alignment")
    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    tensors = {}
    for s in ["trellis", "SU", "SV", "bank_ids", "bank_alt_id"]:
        key = module + "." + s
        with safe_open(str(SNAPSHOT / index[key]), framework="pt", device="cpu") as f:
            tensors[s] = f.get_tensor(key).cuda()
    t, su, sv, bank, alt = [
        tensors[s] for s in ["trellis", "SU", "SV", "bank_ids", "bank_alt_id"]
    ]
    bits = t.shape[-1] / 8
    aid = int(alt.item())
    inner = reconstruct_qvq_inner_weight(
        t,
        bits=bits,
        in_features=k,
        out_features=n,
        bank_ids=bank,
        bank_alt_id=alt,
        v2b2_p32=True,
    ).float()
    payload = repack_p32_planar_to_window(t, bits=bits)
    cfg = json.loads((SNAPSHOT / "quantize_config.json").read_text())
    levels = pgc16_levels_for_version(cfg["codebook"]).cuda()
    window = QVQLinear(
        bits=bits,
        in_features=k,
        out_features=n,
        tensors=tensors,
        bank_count=2,
        v2b2_p32=True,
        codebook_version=cfg["codebook"],
    ).eval()

    def inner_forward(self, z, **kw):
        if kw:
            raise ValueError("Unexpected ordered partials")
        return qvq_p32_window_ampere(
            z.contiguous(), payload, levels, bank, bits, out_features=n, bank_alt_id=aid
        )

    window._inner_forward = MethodType(inner_forward, window)
    xall = (
        torch.load(
            Path("/root/p32-timing-activations") / (module + ".pt"), weights_only=True
        )["input"]
        .reshape(-1, k)
        .cuda()
        .half()
    )

    def fused(x, promotion=0):
        base = op.base(x.bfloat16())
        if promotion:
            hidden = torch.empty((len(x), r), device=x.device, dtype=torch.float16)
            project_blockwise[(triton.cdiv(len(x), 16),)](
                x, op.a, hidden, len(x), k, r, promotion, num_warps=4
            )
        else:
            hidden = x @ op.a
        out = torch.empty((len(x), n), device=x.device, dtype=torch.float16)
        expansion_add[(triton.cdiv(len(x), 16), triton.cdiv(n, 32))](
            hidden, op.b, base, out, len(x), n, r, num_warps=4
        )
        return out

    integrated = RecoveredLinear(args.export, fused_expansion=True)
    variants = {"window": window, "separate": op, "fused_expansion": fused, "integrated": integrated}
    for promotion in [16, 32, 64, 128, 256]:
        variants[f"block_fp32_{promotion}"] = lambda x, promotion=promotion: fused(
            x, promotion
        )

    def exclusive():
        apps = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        for row in apps.splitlines():
            gpu, pid = [s.strip() for s in row.split(",")]
            if gpu == args.uuid and int(pid) != os.getpid():
                raise RuntimeError("Foreign GPU process")

    def timing(fn):
        exclusive()
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(20):
            a, b = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            a.record()
            fn()
            b.record()
            b.synchronize()
            samples.append(a.elapsed_time(b))
        return {"median_ms": sorted(samples)[10], "samples_ms": samples}

    report = {
        "scope": __doc__,
        "uuid": args.uuid,
        "module": module,
        "export": str(args.export),
        "export_sha256": sha(args.export),
        "logical_rank": r,
        "physical_correction_mma_rank": 16,
        "source_sha256": sha(Path(__file__)),
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        if args.profile:
            x = xall[: args.profile_m]
            for _ in range(10):
                for key in ["window", "separate", "fused_expansion", "block_fp32_64"]:
                    variants[key](x)
            torch.cuda.synchronize()
            exclusive()
            torch.cuda.cudart().cudaProfilerStart()
            for key in ["window", "separate", "fused_expansion", "block_fp32_64"]:
                variants[key](x)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            report.update(complete=True, profile_m=args.profile_m)
        else:
            for m in [1, 2, 4, 8, 16, 32, 128, 512, 2048]:
                x = xall[:m]
                teacher = (
                    matmul_hadU(matmul_hadU(x.float() * su.float()) @ inner)
                    * sv.float()
                )
                wy = window(x)
                reference = op(x)
                for name, fn in variants.items():
                    output = fn(x)
                    row = {
                        "M": m,
                        "variant": name,
                        "teacher_metrics": layer_metrics(output, teacher),
                        "window_metrics": layer_metrics(output, wy),
                        "separate_metrics": layer_metrics(output, reference),
                        "eager": timing(lambda fn=fn, x=x: fn(x)),
                    }
                    # Input is static and native/correction output buffers are captured.
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        graph_output = fn(x)
                    graph.replay()
                    torch.cuda.synchronize()
                    row["graph_equal"] = torch.equal(graph_output, output)
                    if not row["graph_equal"]:
                        raise AssertionError("Graph replay changed outputs")
                    row["graph"] = timing(graph.replay)
                    report["rows"].append(row)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    del graph, graph_output
                print("M", m, "complete", flush=True)
            report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
