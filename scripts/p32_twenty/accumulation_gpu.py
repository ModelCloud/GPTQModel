"""Experiments7/8 arithmetic isolation on real P32 weights and activations.

Decoded weights are resident: timings exclude decoding and are not end-to-end
P32 speedups. FP16 partials use FP16 MMA output; BF16 partials are emulated by
rounding FP32 MMA results every K16 step (not native BF16 accumulation).
"""

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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uuid", required=True)
    p.add_argument("--module", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.resolve().is_relative_to(SNAPSHOT.resolve()):
        p.error("Output cannot be inside snapshot")
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
        procs = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        if int(row[0]) > 8 or int(row[1]) or args.uuid in procs:
            raise RuntimeError("GPU not idle")
        time.sleep(1)
    import torch
    import triton
    import triton.language as tl
    from safetensors import safe_open

    from gptqmodel.quantization.qvq import reconstruct_qvq_inner_weight
    from gptqmodel.quantization.rotation.hadamard_utils import matmul_hadU
    from scripts.p32_twenty.scorecard import layer_metrics

    @triton.jit
    def gemm(
        X,
        W,
        Y,
        M: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        MODE: tl.constexpr,
        PROMOTE: tl.constexpr,
    ):
        rows = tl.program_id(0) * 16 + tl.arange(0, 16)
        cols = tl.program_id(1) * 32 + tl.arange(0, 32)
        kk = tl.arange(0, 16)
        total = tl.full((16, 32), 0, tl.float32)
        if MODE == 1:
            partial = tl.full((16, 32), 0, tl.float16)
        else:
            partial = tl.full((16, 32), 0, tl.float32)
        for base in range(0, K, 16):
            a = tl.load(
                X + rows[:, None] * K + base + kk[None, :], rows[:, None] < M, 0
            )
            b = tl.load(
                W + (base + kk[:, None]) * N + cols[None, :], cols[None, :] < N, 0
            )
            if MODE == 0:
                total = tl.dot(a, b, total, out_dtype=tl.float32)
            else:
                if MODE == 1:
                    partial = tl.dot(a, b, partial, out_dtype=tl.float16)
                else:
                    partial = tl.dot(a, b, partial, out_dtype=tl.float32)
                    if MODE == 2:
                        partial = partial.to(tl.bfloat16).to(tl.float32)
                if (base + 16) % PROMOTE == 0:
                    total = total + partial.to(tl.float32)
                    partial = tl.full(
                        (16, 32), 0, tl.float16 if MODE == 1 else tl.float32
                    )
        tl.store(
            Y + rows[:, None] * N + cols[None, :],
            total,
            (rows[:, None] < M) & (cols[None, :] < N),
        )

    torch.backends.cuda.matmul.allow_tf32 = False
    idx = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    def read(s):
        key = args.module + "." + s
        with safe_open(str(SNAPSHOT / idx[key]), framework="pt", device="cpu") as f:
            return f.get_tensor(key).cuda()

    t, su, sv, bank, alt = [
        read(s) for s in ["trellis", "SU", "SV", "bank_ids", "bank_alt_id"]
    ]
    k, n = su.numel(), sv.numel()
    inner = reconstruct_qvq_inner_weight(
        t,
        bits=t.shape[-1] / 8,
        in_features=k,
        out_features=n,
        bank_ids=bank,
        bank_alt_id=alt,
        v2b2_p32=True,
    ).float()
    half = inner.half().contiguous()
    assert torch.equal(half.float(), inner), (
        "Inner codebook narrowing must be exact for this isolation"
    )
    cap = torch.load(
        Path("/root/p32-timing-activations") / (args.module + ".pt"), weights_only=True
    )
    xall = cap["input"].reshape(-1, k).cuda().float()
    report = {
        "scope": __doc__,
        "module": args.module,
        "uuid": args.uuid,
        "resident_decoded_weight_bytes": half.numel() * half.element_size(),
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def timing(fn):
        procs = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        for line in procs.splitlines():
            gpu, pid = [v.strip() for v in line.split(",")]
            if gpu == args.uuid and int(pid) != os.getpid():
                raise RuntimeError("Foreign GPU process")
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(10):
            a, b = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            a.record()
            fn()
            b.record()
            b.synchronize()
            times.append(a.elapsed_time(b))
        return times

    with torch.no_grad():
        for m in [1, 2, 4, 8, 16, 32, 128, 512, 2048]:
            x = xall[:m]
            z = matmul_hadU(x * su.float()).half().contiguous()
            teacher = matmul_hadU(matmul_hadU(x * su.float()) @ inner) * sv.float()
            out = torch.empty((m, n), device="cuda", dtype=torch.float32)
            for mode in [0, 1, 2, 3]:
                for promotion in [16] if mode == 0 else [16, 32, 64, 128, 256]:

                    def run(z=z, out=out, m=m, mode=mode, promotion=promotion):
                        gemm[(triton.cdiv(m, 16), triton.cdiv(n, 32))](
                            z, half, out, m, k, n, mode, promotion, num_warps=4
                        )

                    run()
                    candidate = matmul_hadU(out) * sv.float()
                    metrics = layer_metrics(candidate, teacher)
                    report["rows"].append(
                        {
                            "M": m,
                            "mode": mode,
                            "promotion_k": promotion,
                            "metrics": metrics,
                            "inner_samples_ms": timing(run),
                        }
                    )
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
            print("M", m, "complete", flush=True)
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
