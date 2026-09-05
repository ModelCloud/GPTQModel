"""Experiments 31/32/33/35: fixed deployed base, small output-residual ranks.

No quantizer is invoked. Inputs are the pinned historical Fisher activation
capture; held-out C4 is evaluation only. Standalone window uses the production
QVQLinear forward and the same inner override as the window model study.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SNAPSHOT = Path(
    "/root/qvq-results/calibration-fisher-frontier-wave14-v1/llama32-1b-f6_yaqa125x_seed7"
)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uuid", required=True)
    p.add_argument(
        "--base-export",
        type=Path,
        default=Path("/root/p32-native-joint/down_proj/rank16.pt"),
    )
    p.add_argument(
        "--calibration-root",
        type=Path,
        default=Path("/root/p32-recovery-calibration/activations16x512"),
    )
    p.add_argument("--calibration-tokens", type=int, default=8192)
    p.add_argument("--calibration-prefix", action="store_true")
    p.add_argument("--ranks", nargs="+", type=int, default=[0, 2, 4, 6, 8, 12, 16])
    p.add_argument(
        "--fits",
        nargs="+",
        choices=["l2", "tail", "truncate"],
        default=["l2", "tail", "truncate"],
    )
    p.add_argument("--tail-steps", type=int, default=40)
    p.add_argument("--tail-alpha", type=float, default=9.0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.resolve().is_relative_to(SNAPSHOT.resolve()) or any(
        r not in (0, 2, 4, 6, 8, 12, 16) for r in args.ranks
    ):
        p.error("Use an external output directory and the declared rank grid")
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
            raise RuntimeError("Target GPU must be idle")
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

    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    args.output.mkdir(parents=True, exist_ok=True)
    source_sha = sha(args.base_export)
    bundle = torch.load(args.base_export, weights_only=False, map_location="cpu")
    module = bundle["module"]
    k, n = bundle["in_features"], bundle["out_features"]
    base = RecoveredLinear(args.base_export).eval()
    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    shard_paths = sorted(
        {
            SNAPSHOT / index[module + "." + s]
            for s in ["trellis", "SU", "SV", "bank_ids", "bank_alt_id"]
        }
    )
    shard_hashes = {str(path): sha(path) for path in shard_paths}

    def read(s):
        key = module + "." + s
        with safe_open(str(SNAPSHOT / index[key]), framework="pt", device="cpu") as f:
            return f.get_tensor(key).cuda()

    tensors = {s: read(s) for s in ["trellis", "SU", "SV", "bank_ids", "bank_alt_id"]}
    t, su, sv, bank, alt = [
        tensors[s] for s in ["trellis", "SU", "SV", "bank_ids", "bank_alt_id"]
    ]
    bits, aid = t.shape[-1] / 8, int(alt.item())
    cfg = json.loads((SNAPSHOT / "quantize_config.json").read_text())
    inner = reconstruct_qvq_inner_weight(
        t,
        bits=bits,
        in_features=k,
        out_features=n,
        bank_ids=bank,
        bank_alt_id=alt,
        v2b2_p32=True,
    ).float()
    window = repack_p32_planar_to_window(t, bits=bits)
    levels = pgc16_levels_for_version(cfg["codebook"]).cuda()
    window_layer = QVQLinear(
        bits=bits,
        in_features=k,
        out_features=n,
        tensors=tensors,
        bank_count=2,
        v2b2_p32=True,
        codebook_version=cfg["codebook"],
    ).eval()

    def window_inner(
        self, x, *, return_ordered_partials=False, ordered_split_count=None
    ):
        if return_ordered_partials or ordered_split_count is not None:
            raise ValueError("This experiment uses the window model ordinary forward")
        return qvq_p32_window_ampere(
            x.contiguous(), window, levels, bank, bits, out_features=n, bank_alt_id=aid
        )

    window_layer._inner_forward = MethodType(window_inner, window_layer)

    def teacher(x):
        return matmul_hadU(matmul_hadU(x.float() * su.float()) @ inner) * sv.float()

    def native(x):
        return base.base(x.bfloat16()).float()

    def capture(root):
        path = root / (module + ".pt")
        cap = torch.load(path, weights_only=True)
        return cap["input"].reshape(-1, k).cuda().half(), {
            "path": str(path),
            "sha256": sha(path),
        }

    xc, calmeta = capture(args.calibration_root)
    xe, evalmeta = capture(Path("/root/p32-timing-activations"))
    if args.calibration_prefix:
        calmeta["available_rows"] = len(xc)
        xc = xc[: args.calibration_tokens]
        calmeta["selected_prefix_rows"] = len(xc)
    if args.calibration_tokens < 1 or len(xc) != args.calibration_tokens:
        raise ValueError("Calibration capture does not match the declared token count")

    def native_fingerprint(weight):
        h = hashlib.sha256()

        def visit(t, name):
            if hasattr(t, "__tensor_flatten__"):
                names, metadata = t.__tensor_flatten__()
                h.update((name + repr(metadata)).encode())
                for child in names:
                    visit(getattr(t, child), name + "." + child)
            else:
                c = t.detach().cpu().contiguous()
                h.update((name + str(c.dtype) + str(tuple(c.shape))).encode())
                h.update(c.view(torch.uint8).numpy().tobytes())

        visit(weight, "weight")
        return h.hexdigest()

    fingerprint = native_fingerprint(base.base.weight)
    report = {
        "experiments": [31, 32, 33, 35],
        "module": module,
        "K": k,
        "N": n,
        "uuid": args.uuid,
        "torch": torch.__version__,
        "device": str(torch.cuda.get_device_properties(0)),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_export": str(args.base_export),
        "source_sha256": source_sha,
        "fixed_native_fingerprint": fingerprint,
        "calibration": calmeta,
        "evaluation": evalmeta,
        "calibration_rows": len(xc),
        "evaluation_rows": len(xe),
        "ranks": args.ranks,
        "activation_rcond": 1e-5,
        "teacher": "Canonical FP32 on identical FP16-rounded inputs",
        "window_reference": "Production QVQLinear forward with lossless-window inner override, FP16 input/output",
        "native": "Actual TorchAO W4A16 BF16 tinygemm; no native base requantization",
        "factor_arithmetic": "Input cast to A dtype, first GEMM, hidden cast to B dtype, second GEMM; correction upcast FP32 and added to BF16-native output upcast FP32, final FP16 output",
        "tail": {
            "steps": args.tail_steps,
            "alpha": args.tail_alpha,
            "quantile": 0.999,
            "selection": "minimum fixed calibration tail-weighted actual-output MSE including initial fit; no held-out selection",
        },
        "candidates": [],
        "baselines": [],
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
        for line in procs.splitlines():
            gpu, pid = [s.strip() for s in line.split(",")]
            if gpu == args.uuid and int(pid) != os.getpid():
                raise RuntimeError("Foreign process during timing")
        with torch.no_grad():
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            result = []
            for _ in range(20):
                a, b = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                a.record()
                fn()
                b.record()
                b.synchronize()
                result.append(a.elapsed_time(b))
        return {"median_ms": sorted(result)[len(result) // 2], "samples_ms": result}

    with torch.no_grad():
        yc, ye = teacher(xc), teacher(xe)
        yn = native(xc)
        xcf = xc.float()
        residual = yc.double() - yn.double()
        print("FITTING_FP64_SVD", module, args.ranks, flush=True)
        u, s, vh = torch.linalg.svd(xcf.double(), full_matrices=False)
        effective = int((s > s[0] * report["activation_rcond"]).sum())
        left, values, right = torch.linalg.svd(
            u[:, :effective].T @ residual, full_matrices=False
        )
        aa = ((vh[:effective].T / s[:effective]) @ (left[:, :16] * values[:16])).float()
        bb = right[:16].float()
        report["activation_rank"] = effective
        report["residual_projected_singular_values"] = values.cpu().tolist()
        report["residual_total_energy"] = float(residual.square().sum())
        report["projected_residual_energy"] = float(values.square().sum())
        # QR/SVD computes the exact SVD of A16 B16 without allocating its dense K*N product.
        qa, ra = torch.linalg.qr(base.a.double(), mode="reduced")
        qb, rb = torch.linalg.qr(base.b.double().T, mode="reduced")
        tu, ts, tv = torch.linalg.svd(ra @ rb.T, full_matrices=False)
        trunc_a = (qa @ tu * ts).float()
        trunc_b = (tv @ qb.T).float()
        report["rank16_weight_correction_singular_values"] = ts.cpu().tolist()
        del u, s, vh, left, right, residual, qa, qb, ra, rb, tu, ts, tv
        for m in [1, 2, 4, 8, 16, 32, 128, 512, 2048]:
            x = xe[:m]
            wy = window_layer(x)
            report["baselines"].append(
                {
                    "M": m,
                    "window_vs_teacher": layer_metrics(wy, ye[:m]),
                    "window": timing(lambda x=x: window_layer(x)),
                    "rank16": timing(lambda x=x: base(x)),
                    "rank16_vs_teacher": layer_metrics(base(x), ye[:m]),
                }
            )
        publish()

    def predict_factors(a, b):
        return (yn + (xcf @ a) @ b).half().float()

    def tail_refit(a, b):
        if not a.shape[1]:
            return a, b, {"skipped": "rank zero has no trainable factors"}
        with torch.no_grad():
            initial = predict_factors(a, b)
            initial_error = (initial - yc).abs()
            threshold = float(torch.quantile(initial_error.flatten(), 0.999))
            weights = 1 + args.tail_alpha * (initial_error > threshold).float()
            denominator = float(((initial - yc).square() * weights).mean())
        a = a.clone().requires_grad_()
        b = b.clone().requires_grad_()
        optimizer = torch.optim.Adam(
            [
                {"params": [a], "lr": 0.01 * float(a.detach().square().mean().sqrt())},
                {"params": [b], "lr": 0.01 * float(b.detach().square().mean().sqrt())},
            ]
        )
        best = float("inf")
        best_pair = None
        history = []
        best_step = 0
        with torch.enable_grad():
            for step in range(args.tail_steps + 1):
                optimizer.zero_grad(set_to_none=True)
                loss = ((predict_factors(a, b) - yc).square() * weights).mean()
                value = float(loss.detach())
                if value < best:
                    best = value
                    best_pair = (a.detach().clone(), b.detach().clone())
                    best_step = step
                history.append(value)
                if step < args.tail_steps:
                    (loss / max(denominator, 1e-30)).backward()
                    optimizer.step()
        return *best_pair, {
            "threshold": threshold,
            "weighted_elements": int((weights > 1).sum()),
            "objective_history": history,
            "selected_step": best_step,
        }

    for rank in args.ranks:
        for fit in args.fits:
            if fit == "truncate":
                a, b = trunc_a[:, :rank].contiguous(), trunc_b[:rank].contiguous()
            else:
                a, b = aa[:, :rank].contiguous(), bb[:rank].contiguous()
            fitmeta = {}
            if fit == "tail":
                a, b, fitmeta = tail_refit(a, b)
            for dtype_name in ["float32", "float16"]:
                dtype = getattr(torch, dtype_name)
                export = dict(bundle)
                export.update(
                    a=a.detach().to(device="cpu", dtype=dtype).clone(),
                    b=b.detach().to(device="cpu", dtype=dtype).clone(),
                    actual_rank=rank,
                    fit=fit,
                    factor_dtype=dtype_name,
                    fixed_base_source_sha256=source_sha,
                    arithmetic=report["factor_arithmetic"],
                )
                path = args.output / f"rank{rank}-{fit}-{dtype_name}.pt"
                torch.save(export, path)
                operator = RecoveredLinear(path)
                if native_fingerprint(operator.base.weight) != fingerprint:
                    raise AssertionError("Fixed native base changed")
                item = {
                    "rank": rank,
                    "fit": fit,
                    "factor_dtype": dtype_name,
                    "fit_metadata": fitmeta,
                    "export": str(path),
                    "export_sha256": sha(path),
                    "serialized_operator_bytes": path.stat().st_size,
                    "serialized_operator_bpw": 8 * path.stat().st_size / (k * n),
                    "factor_bytes": (k + n)
                    * rank
                    * torch.tensor([], dtype=dtype).element_size(),
                    "optimal_l2_projected_energy_fraction_for_rank": float(
                        values[:rank].square().sum() / values.square().sum()
                    ),
                    "rows": [],
                }
                with torch.no_grad():
                    item["calibration_vs_teacher"] = layer_metrics(operator(xc), yc)
                    item["calibration_vs_window"] = layer_metrics(
                        operator(xc), window_layer(xc)
                    )
                    item["heldout_vs_teacher"] = layer_metrics(operator(xe), ye)
                    item["heldout_vs_window"] = layer_metrics(
                        operator(xe), window_layer(xe)
                    )
                    # Independent in-memory factors and the trusted reloaded operator must agree.
                    ad, bd = a.to(dtype), b.to(dtype)
                    for m in [1, 2, 4, 8, 16, 32, 128, 512, 2048]:
                        x = xe[:m]
                        direct = native(x)
                        if rank:
                            direct = direct + ((x.float().to(dtype) @ ad) @ bd).float()
                        direct = direct.half()
                        got = operator(x)
                        if not torch.equal(got, direct):
                            raise AssertionError("Export-reload changed output")
                        latency = timing(lambda x=x, operator=operator: operator(x))
                        baseline = next(r for r in report["baselines"] if r["M"] == m)
                        item["rows"].append(
                            {
                                "M": m,
                                "teacher_metrics": layer_metrics(got, ye[:m]),
                                "window_metrics": layer_metrics(got, window_layer(x)),
                                "reload_equal": True,
                                "latency": latency,
                                "speedup_vs_window": baseline["window"]["median_ms"]
                                / latency["median_ms"],
                                "speedup_vs_rank16": baseline["rank16"]["median_ms"]
                                / latency["median_ms"],
                            }
                        )
                report["candidates"].append(item)
                publish()
                print(
                    "CANDIDATE",
                    rank,
                    fit,
                    dtype_name,
                    "passes",
                    sum(
                        r["teacher_metrics"]["local_tolerance_pass"]
                        for r in item["rows"]
                    ),
                    "bpw",
                    item["serialized_operator_bpw"],
                    flush=True,
                )
                del operator
    report["source_export_unchanged"] = sha(args.base_export) == source_sha
    report["teacher_read_shards_unchanged"] = all(
        sha(Path(path)) == digest for path, digest in shard_hashes.items()
    )
    report["teacher_shard_sha256"] = shard_hashes
    if (
        not report["source_export_unchanged"]
        or not report["teacher_read_shards_unchanged"]
    ):
        raise AssertionError("Read-only source changed")
    report["complete"] = True
    publish()


if __name__ == "__main__":
    main()
