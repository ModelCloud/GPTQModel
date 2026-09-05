"""Experiment 19: deployed W4A16 kernel plus output-fitted low-rank recovery.

INT4 storage with BF16 activation/compute; this is not INT4 activation IMMA.
Calibration and held-out evaluation captures are separate. Snapshot is read only.
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uuid", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--fp16-boundary", action="store_true")
    parser.add_argument("--joint-steps", type=int, default=0)
    parser.add_argument("--joint-rank", type=int, choices=(16, 32, 64, 128), default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=Path("/root/p32-recovery-calibration/activations"),
    )
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(SNAPSHOT.resolve()):
        parser.error("Output must be outside teacher snapshot")
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
        if int(row[0]) > 8 or int(row[1]) != 0 or args.uuid in procs:
            raise RuntimeError("GPU must be idle")
        time.sleep(1)
    import torch
    from safetensors import safe_open
    from torchao.quantization import Int4WeightOnlyConfig, quantize_
    from torchao.quantization.quantize_.workflows.int4.int4_packing_format import (
        Int4PackingFormat,
    )

    from gptqmodel.quantization.qvq import (
        reconstruct_qvq_inner_weight,
        repack_p32_planar_to_window,
    )
    from gptqmodel.quantization.qvq_codecs import pgc16_levels_for_version
    from gptqmodel.quantization.rotation.hadamard_utils import matmul_hadU
    from gptqmodel.utils.qvq_ampere_cuda import qvq_p32_window_ampere
    from gptqmodel.utils.qvq_cuda import qvq_cuda_gemv
    from scripts.p32_twenty.scorecard import layer_metrics

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    def read(suffix):
        key = args.module + "." + suffix
        with safe_open(str(SNAPSHOT / index[key]), framework="pt", device="cpu") as f:
            return f.get_tensor(key).cuda()

    trellis, su, sv, bank, alt = [
        read(k) for k in ["trellis", "SU", "SV", "bank_ids", "bank_alt_id"]
    ]
    bits = trellis.shape[-1] / 8
    k, n = su.numel(), sv.numel()
    inner = reconstruct_qvq_inner_weight(
        trellis,
        bits=bits,
        in_features=k,
        out_features=n,
        bank_ids=bank,
        bank_alt_id=alt,
        v2b2_p32=True,
    ).float()
    folded = (
        matmul_hadU(matmul_hadU(inner.T.contiguous()).T.contiguous())
        * su.float()[:, None]
        * sv.float()[None, :]
    )
    linear = torch.nn.Linear(
        k, n, bias=False, device="cuda", dtype=torch.bfloat16
    ).eval()
    linear.weight.data.copy_(folded.T)
    quantize_(
        linear,
        Int4WeightOnlyConfig(
            group_size=128, int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D
        ),
    )

    def native(x):
        return linear(x.bfloat16()).float()

    def planar(x):
        z = matmul_hadU(x.float() * su.float()).half().contiguous()
        y = qvq_cuda_gemv(
            z,
            trellis,
            bits,
            out_features=n,
            output_fp32=True,
            bank_ids=bank,
            v2b2_p32=True,
            bank_alt_id=int(alt.item()),
        )
        return matmul_hadU(y) * sv.float()

    window = repack_p32_planar_to_window(trellis, bits=bits)
    cfg = json.loads((SNAPSHOT / "quantize_config.json").read_text())
    levels = pgc16_levels_for_version(cfg["codebook"]).cuda()

    def window_full(x):
        z = matmul_hadU(x.float() * su.float()).half().contiguous()
        y = qvq_p32_window_ampere(
            z, window, levels, bank, bits, out_features=n, bank_alt_id=int(alt.item())
        )
        return matmul_hadU(y) * sv.float()

    def capture(root):
        item = torch.load(Path(root) / (args.module + ".pt"), weights_only=True)
        return item["input"].reshape(-1, k).cuda().float(), item[
            "teacher_output"
        ].reshape(-1, n).cuda().float()

    xc, yc = capture(args.calibration_root)
    xe, ye = capture("/root/p32-timing-activations")
    if args.fp16_boundary:
        xc, xe = xc.half().float(), xe.half().float()
        # Recompute the canonical reference on IDENTICAL rounded input values.
        yc = matmul_hadU(matmul_hadU(xc * su.float()) @ inner) * sv.float()
        ye = matmul_hadU(matmul_hadU(xe * su.float()) @ inner) * sv.float()

    def output_cast(y):
        return y.half().float() if args.fp16_boundary else y

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": 20 if args.joint_steps else 19,
        "joint_steps": args.joint_steps,
        "fp16_boundary": args.fp16_boundary,
        "module": args.module,
        "uuid": args.uuid,
        "scope": "one real projection; actual TorchAO INT4-storage BF16-activation tinygemm plus FP32 low-rank GEMMs; not INT4 activation IMMA or full-model evaluation",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "calibration": str(args.calibration_root),
        "calibration_rows": xc.shape[0],
        "evaluation": "separate C4 teacher activation capture; never used in fitting",
        "activation_rcond": 1e-5,
        "rows": [],
    }

    def publish():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def timing(fn):
        procs = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        for row in procs.splitlines():
            gpu, pid = [x.strip() for x in row.split(",")]
            if gpu == args.uuid and int(pid) != os.getpid():
                raise RuntimeError("Foreign process before timing")
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(10):
            a, b = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            a.record()
            fn()
            b.record()
            b.synchronize()
            samples.append(a.elapsed_time(b))
        return samples

    with torch.no_grad():
        report["folded_calibration_metrics"] = layer_metrics(xc @ folded, yc)
        report["folded_evaluation_metrics"] = layer_metrics(xe @ folded, ye)
        yn = native(xc)
        z = yc.double() - yn.double()
        # Reduced-rank regression on real deployed output residual, not dequantized weights.
        u, s, vh = torch.linalg.svd(xc.double(), full_matrices=False)
        p = int((s > s[0] * report["activation_rcond"]).sum())
        left, values, right = torch.linalg.svd(u[:, :p].T @ z, full_matrices=False)
        rank = min(args.joint_rank if args.joint_steps else 128, p, n)
        aa = ((vh[:p].T / s[:p]) @ (left[:, :rank] * values[:rank])).float()
        bb = right[:rank].float()
        report["activation_rank"] = p
        report["native_calibration_metrics"] = layer_metrics(yn, yc)

        if args.joint_steps:
            if args.joint_steps < 0:
                raise ValueError("Joint steps must be nonnegative")
            history = []

            def measure_iteration(step):
                prediction = output_cast(native(xc) + (xc @ aa) @ bb)
                loss = float((prediction.double() - yc.double()).square().mean())
                history.append(
                    {
                        "step": step,
                        "calibration_mse": loss,
                        "calibration_metrics": layer_metrics(prediction, yc),
                        "evaluation_metrics": layer_metrics(
                            output_cast(native(xe) + (xe @ aa) @ bb), ye
                        ),
                    }
                )
                return loss

            best_loss = measure_iteration(0)
            best = (linear, aa, bb, 0)
            for step in range(1, args.joint_steps + 1):
                # Quantize a new independent base; never mutate teacher tensors.
                target = folded - aa @ bb
                linear = torch.nn.Linear(
                    k, n, bias=False, device="cuda", dtype=torch.bfloat16
                ).eval()
                linear.weight.copy_(target.T)
                quantize_(
                    linear,
                    Int4WeightOnlyConfig(
                        group_size=128,
                        int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D,
                    ),
                )
                residual = yc.double() - native(xc).double()
                left, values, right = torch.linalg.svd(
                    u[:, :p].T @ residual, full_matrices=False
                )
                aa = ((vh[:p].T / s[:p]) @ (left[:, :rank] * values[:rank])).float()
                bb = right[:rank].float()
                loss = measure_iteration(step)
                if loss < best_loss:
                    best_loss = loss
                    best = (linear, aa, bb, step)
                print("JOINT_STEP", args.module, step, loss, flush=True)
            linear, aa, bb, selected_step = best
            report["joint_history"] = history
            report["selected_step"] = selected_step
            report["selection_rule"] = (
                "minimum actual calibration-output MSE at fixed rank; evaluation never selects the iteration"
            )
            report["final_native_calibration_metrics"] = layer_metrics(native(xc), yc)

        # Flatten actual tensor-subclass payloads recursively for stored-byte accounting.
        def leaves(t):
            if hasattr(t, "__tensor_flatten__"):
                names, _ = t.__tensor_flatten__()
                return [leaf for name in names for leaf in leaves(getattr(t, name))]
            return [t]

        tensors = leaves(linear.weight)
        seen = set()
        payload = 0
        for t in tensors:
            key = (str(t.device), t.untyped_storage().data_ptr())
            if key not in seen:
                payload += t.untyped_storage().nbytes()
                seen.add(key)
        report["native_payload_bytes"] = payload
        report["storage_scope"] = (
            "Actual packed tensor/scale storage plus FP32 factors; Python format/config metadata and full-model overhead not included, so not complete effective BPW"
        )
        torch.save(
            {
                "native_state_dict": linear.state_dict(),
                "a": aa.cpu(),
                "b": bb.cpu(),
                "group_size": 128,
                "format": "tile_packed_to_4d",
                "module": args.module,
            },
            args.output / "export.pt",
        )
        for m in [1, 2, 4, 8, 16, 32, 128, 512, 2048]:
            x, y = xe[:m], ye[:m]
            native_y = native(x)
            base = planar(x)
            row = {
                "M": m,
                "planar_metrics": layer_metrics(base, y),
                "native_metrics": layer_metrics(native_y, y),
                "window_metrics": layer_metrics(window_full(x), y),
                "window_ms": timing(lambda x=x: window_full(x)),
                "planar_ms": timing(lambda x=x: planar(x)),
                "native_ms": timing(lambda x=x: native(x)),
                "ranks": [],
            }
            for r in [args.joint_rank] if args.joint_steps else [16, 32, 64, 128]:
                rr = min(r, rank)
                a, b = aa[:, :rr].contiguous(), bb[:rr].contiguous()

                def corrected(x=x, a=a, b=b):
                    return output_cast(native(x) + (x @ a) @ b)

                row["ranks"].append(
                    {
                        "requested_rank": r,
                        "actual_rank": rr,
                        "metrics": layer_metrics(corrected(), y),
                        "calibration_metrics": layer_metrics(
                            output_cast(native(xc) + (xc @ a) @ b), yc
                        ),
                        "samples_ms": timing(corrected),
                        "factor_bytes": (a.numel() + b.numel()) * 4,
                        "tensor_payload_bpw": 8
                        * (payload + (a.numel() + b.numel()) * 4)
                        / (k * n),
                    }
                )
            report["rows"].append(row)
            publish()
            print("ROW", args.module, m, flush=True)
    report["export_file_bytes"] = (args.output / "export.pt").stat().st_size
    report["complete"] = True
    publish()


if __name__ == "__main__":
    main()
