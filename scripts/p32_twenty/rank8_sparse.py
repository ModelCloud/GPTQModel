"""Target the original rank8 failure channel with historical-calibration sparse fits.

Original held-out channel localization makes that case development evidence.
Only historical Fisher rows determine support correlations and coefficient fits.
Broader replay and model tasks must independently assess every new export.
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uuid", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--export",
        type=Path,
        default=Path("/root/p32-low-rank/worker2/rank8-tail-float16.pt"),
    )
    p.add_argument("--channels", type=int, choices=[1, 2, 4, 8], default=1)
    args = p.parse_args()
    from scripts.p32_twenty.low_rank_sweep import SNAPSHOT, sha

    if args.output.resolve().is_relative_to(SNAPSHOT.resolve()):
        p.error("External output required")
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
    from scripts.p32_twenty.scorecard import layer_metrics

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    source = args.export
    digest = sha(source)
    op = RecoveredLinear(source)
    bundle = torch.load(source, map_location="cpu", weights_only=False)
    module = op.source_module
    k, n = op.in_features, op.out_features
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
            raise ValueError("Unexpected partials")
        return qvq_p32_window_ampere(
            z.contiguous(), payload, levels, bank, bits, out_features=n, bank_alt_id=aid
        )

    window._inner_forward = MethodType(inner_forward, window)

    def capture(root):
        return (
            torch.load(Path(root) / (module + ".pt"), weights_only=True)["input"]
            .reshape(-1, k)
            .cuda()
            .half()
        )

    xc = capture("/root/p32-recovery-calibration/activations16x512")
    xe = capture("/root/p32-timing-activations")

    def teacher(x):
        return matmul_hadU(matmul_hadU(x.float() * su.float()) @ inner) * sv.float()

    def raw(x):
        base = op.base(x.bfloat16()).float()
        if op.a.shape[1] == 0:
            return base
        return base + ((x @ op.a) @ op.b).float()

    report = {
        "scope": __doc__,
        "module": module,
        "K": k,
        "N": n,
        "source": str(source),
        "source_sha256": digest,
        "calibration": "same historical 8192-token Fisher capture",
        "candidates": [],
    }
    args.output.mkdir(parents=True, exist_ok=True)

    def timing(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(20):
            a, b = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            a.record()
            fn()
            b.record()
            b.synchronize()
            times.append(a.elapsed_time(b))
        return {"median_ms": sorted(times)[10], "samples_ms": times}

    with torch.no_grad():
        original_error = (op(xe).float() - window(xe).float()).abs()
        worst = int(original_error.argmax())
        channel = worst % n
        report["target_channel"] = channel
        channels = torch.argsort(original_error.amax(0), descending=True, stable=True)[
            : args.channels
        ]
        report["target_channels"] = channels.cpu().tolist()
        report["original_worst_token"] = worst // n
        report["original_window_max"] = float(original_error.max())
        yc, ye = teacher(xc), teacher(xe)
        z = yc[:, channels].double() - raw(xc)[:, channels].double()
        x = xc.double()
        energy = x.square().sum(0)
        score = (x.T @ z).square() / energy.clamp_min(1e-30)[:, None]
        support = torch.argsort(score.flatten(), descending=True, stable=True)[:512]
        for budget in [0, 1, 4, 8, 16, 32, 64, 128, 256, 512]:
            selected = support[:budget]
            positions = selected // args.channels
            output_positions = selected % args.channels
            values = torch.empty(budget, device="cuda", dtype=torch.float32)
            for column in range(args.channels):
                chosen = output_positions == column
                if not chosen.any():
                    continue
                u, singular, vh = torch.linalg.svd(
                    x[:, positions[chosen]], full_matrices=False
                )
                keep = singular > singular[0] * 1e-5
                values[chosen] = (
                    vh[keep].T @ ((u[:, keep].T @ z[:, column]) / singular[keep])
                ).float()
            export = dict(bundle)
            export.update(
                sparse_input_indices=positions.int().cpu(),
                sparse_output_indices=channels[output_positions].int().cpu(),
                sparse_values=values.cpu(),
                sparse_fit="calibration normalized correlation support; FP64 least-squares rcond1e-5; original development-case target channel",
            )
            path = args.output / f"nnz{budget}.pt"
            torch.save(export, path)
            candidate = RecoveredLinear(path)
            item = {
                "rank": op.a.shape[1],
                "fit": "targeted_sparse",
                "factor_dtype": "float16",
                "nnz": budget,
                "export": str(path),
                "export_sha256": sha(path),
                "serialized_operator_bytes": path.stat().st_size,
                "serialized_operator_bpw": 8 * path.stat().st_size / (k * n),
                "calibration_vs_teacher": layer_metrics(candidate(xc), yc),
                "rows": [],
            }
            for m in [1, 2, 4, 8, 16, 32, 128, 512, 2048]:
                xx = xe[:m]
                actual = candidate(xx)
                direct = raw(xx)
                if budget:
                    direct.index_add_(
                        1,
                        export["sparse_output_indices"].cuda().long(),
                        xx.float()[:, positions] * values,
                    )
                equal = torch.equal(direct.half(), actual)
                repeat = torch.equal(candidate(xx), actual)
                item["rows"].append(
                    {
                        "M": m,
                        "teacher_metrics": layer_metrics(actual, ye[:m]),
                        "window_metrics": layer_metrics(actual, window(xx)),
                        "reload_equal": equal,
                        "repeat_equal": repeat,
                        "latency": timing(
                            lambda xx=xx, candidate=candidate: candidate(xx)
                        ),
                    }
                )
            report["candidates"].append(item)
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(
                "SPARSE",
                budget,
                sum(r["window_metrics"]["local_tolerance_pass"] for r in item["rows"]),
                flush=True,
            )
    report["source_unchanged"] = sha(source) == digest
    if not report["source_unchanged"]:
        raise AssertionError("Source mutated")
    report["complete"] = True
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
