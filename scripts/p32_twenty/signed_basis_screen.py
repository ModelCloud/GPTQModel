"""Real F6 signed-basis screening; dense reconstruction is NOT a packed inference kernel."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.p32_twenty.low_rank_sweep import SNAPSHOT, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uuid", required=True)
    p.add_argument("--module", default="model.layers.0.mlp.down_proj")
    p.add_argument("--tile", type=int, choices=(16, 32, 64, 128), default=64)
    p.add_argument("--weighting", choices=("activation", "uniform"), default="activation")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(SNAPSHOT.resolve()):
        p.error("A new external output directory is required")
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
        if int(row[0]) > 8 or int(row[1]):
            raise RuntimeError("GPU not idle")
        time.sleep(1)
    import numpy as np
    import torch
    from safetensors import safe_open

    from gptqmodel.quantization.qvq import reconstruct_qvq_inner_weight
    from gptqmodel.quantization.rotation.hadamard_utils import matmul_hadU
    from scripts.p32_twenty.scorecard import layer_metrics

    torch.backends.cuda.matmul.allow_tf32 = False
    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    tensors = []
    hashes = {}
    for suffix in ("trellis", "SU", "SV", "bank_ids", "bank_alt_id"):
        key = args.module + "." + suffix
        path = SNAPSHOT / index[key]
        hashes.setdefault(str(path), sha(path))
        with safe_open(str(path), framework="pt", device="cpu") as f:
            tensors.append(f.get_tensor(key).cuda())
    t, su, sv, bank, alt = tensors
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
    folded = (
        matmul_hadU(matmul_hadU(inner.T.contiguous()).T.contiguous())
        * su.float()[:, None]
        * sv.float()[None, :]
    )

    def teacher(x):
        return matmul_hadU(matmul_hadU(x.float() * su.float()) @ inner) * sv.float()

    calpath = Path("/root/p32-recovery-calibration/activations16x512") / (
        args.module + ".pt"
    )
    evalpath = Path("/root/p32-timing-activations") / (args.module + ".pt")
    xc = (
        torch.load(calpath, weights_only=True)["input"]
        .reshape(-1, k)
        .cuda()
        .half()
        .float()
    )
    xe = (
        torch.load(evalpath, weights_only=True)["input"]
        .reshape(-1, k)
        .cuda()
        .half()
        .float()
    )
    if len(xc) != 8192 or k % args.tile:
        raise ValueError("Expected original 8192-token capture and whole tiles")
    args.output.mkdir(parents=True)
    report = {
        "scope": __doc__,
        "module": args.module,
        "uuid": args.uuid,
        "tile": args.tile,
        "calibration": str(calpath),
        "calibration_sha256": sha(calpath),
        "evaluation": str(evalpath),
        "evaluation_sha256": sha(evalpath),
        "objective": "greedy weight residual; not full output-aware optimum",
        "weighting": args.weighting,
        "candidates": [],
    }
    with torch.no_grad():
        residual = folded.T.contiguous().reshape(n, k // args.tile, args.tile)
        energy = (
            xc.square().mean(0).reshape(1, k // args.tile, args.tile).clamp_min(1e-30)
        )
        if args.weighting == "uniform":
            energy = torch.ones_like(energy)
        decoded = torch.zeros_like(residual)
        planes = []
        scales = []
        reference = teacher(xe)
        report["folding_metrics"] = layer_metrics(xe @ folded, reference)
        for rank in range(1, 17):
            signs = torch.where(residual >= 0, 1.0, -1.0)
            scale = (residual.abs() * energy).sum(-1) / energy.sum(-1)
            decoded += signs * scale[..., None]
            residual -= signs * scale[..., None]
            planes.append(
                np.packbits((signs > 0).cpu().numpy(), axis=-1, bitorder="little")
            )
            scales.append(scale.cpu())
            if rank not in (1, 2, 4, 8, 12, 16):
                continue
            bundle = {
                "module": args.module,
                "shape": [n, k],
                "tile": args.tile,
                "rank": rank,
                "signs": torch.from_numpy(np.stack(planes, axis=0)),
                "scales": torch.stack(scales),
                "bitorder": "little",
                "sign_mapping": "0=-1,1=+1",
            }
            path = args.output / f"rank{rank}.pt"
            torch.save(bundle, path)
            loaded = torch.load(path, weights_only=True)
            bits = np.unpackbits(
                loaded["signs"].numpy(), axis=-1, count=args.tile, bitorder="little"
            )
            restored = torch.zeros_like(decoded)
            for i in range(rank):
                restored += (
                    torch.from_numpy(bits[i].astype(np.float32) * 2 - 1).cuda()
                    * loaded["scales"][i].cuda()[..., None]
                )
            if not torch.equal(restored, decoded):
                raise AssertionError("Packed export reload changed reconstruction")
            w = restored.reshape(n, k).T.contiguous()
            cases = []
            for m in (1, 2, 4, 8, 16, 32, 128, 512, 2048):
                cases.append(
                    {
                        "M": m,
                        "teacher_metrics": layer_metrics(xe[:m] @ w, reference[:m]),
                    }
                )
            report["candidates"].append(
                {
                    "rank": rank,
                    "export": str(path),
                    "sha256": sha(path),
                    "serialized_bytes": path.stat().st_size,
                    "bpw": 8 * path.stat().st_size / (n * k),
                    "reload_equal": True,
                    "weighted_residual_energy": float((residual.square() * energy).sum()),
                    "weight_residual_max": float(residual.abs().max()),
                    "cases": cases,
                }
            )
            print(
                "RANK",
                rank,
                "passes",
                sum(x["teacher_metrics"]["local_tolerance_pass"] for x in cases),
                flush=True,
            )
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
    report["teacher_shards_unchanged"] = all(
        sha(Path(p)) == h for p, h in hashes.items()
    )
    if not report["teacher_shards_unchanged"]:
        raise AssertionError("Teacher changed")
    report["teacher_shard_sha256"] = hashes
    report["complete"] = True
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
