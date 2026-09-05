"""Replay fixed native exports on independent-document C4 activation captures."""

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
from scripts.p32_twenty.low_rank_sweep import SNAPSHOT, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uuid", required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument(
        "--captures", type=Path, default=Path("/root/p32-low-rank/all-down-heldout")
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.resolve().is_relative_to(SNAPSHOT.resolve()):
        p.error("Output must be outside teacher")
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
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    report = json.loads(args.report.read_text())
    if not report.get("complete"):
        raise ValueError("Fitting report not complete")
    module = report["module"]
    k, n = report["K"], report["N"]
    capture_manifest = json.loads((args.captures / "capture.json").read_text())
    if not capture_manifest.get("complete"):
        raise ValueError("Capture not complete")
    counts = capture_manifest["sequence_token_counts"]
    path = args.captures / (module + ".pt")
    x = torch.load(path, weights_only=True)["input"].reshape(-1, k).cuda().half()
    if sum(counts) != len(x):
        raise ValueError("Document boundaries mismatch")
    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    tensors = {}
    for suffix in ["trellis", "SU", "SV", "bank_ids", "bank_alt_id"]:
        key = module + "." + suffix
        with safe_open(str(SNAPSHOT / index[key]), framework="pt", device="cpu") as f:
            tensors[suffix] = f.get_tensor(key).cuda()
    t, su, sv, bank, alt = [
        tensors[s] for s in ["trellis", "SU", "SV", "bank_ids", "bank_alt_id"]
    ]
    bits = t.shape[-1] / 8
    aid = int(alt.item())
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
    payload = repack_p32_planar_to_window(t, bits=bits)
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
            raise ValueError("Unexpected ordered-partial request")
        return qvq_p32_window_ampere(
            z.contiguous(), payload, levels, bank, bits, out_features=n, bank_alt_id=aid
        )

    window._inner_forward = MethodType(inner_forward, window)
    result = {
        "scope": "Broader 16-document C4 replay, no fitting or candidate selection on these outputs; first eight source documents overlap original concatenated capture but contexts reset per document",
        "module": module,
        "uuid": args.uuid,
        "source_report": str(args.report),
        "capture_sha256": sha(path),
        "capture_manifest": capture_manifest,
        "documents": len(counts),
        "tokens": len(x),
        "candidates": [],
        "window": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def publish():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    with torch.no_grad():
        y = matmul_hadU(matmul_hadU(x.float() * su.float()) @ inner) * sv.float()
        cases = []
        offset = 0
        for doc, count in enumerate(counts):
            cases.append((f"document-{doc}", offset, offset + count))
            offset += count
        cases.extend(
            (f"prefix-M{m}", 0, m) for m in [1, 2, 4, 8, 16, 32, 128, 512, 2048]
        )
        window_outputs = {}
        for name, start, end in cases:
            wy = window(x[start:end])
            window_outputs[name] = wy
            result["window"].append(
                {
                    "case": name,
                    "rows": end - start,
                    "metrics": layer_metrics(wy, y[start:end]),
                }
            )
        for candidate in report["candidates"]:
            export = Path(candidate["export"])
            if sha(export) != candidate["export_sha256"]:
                raise AssertionError("Export changed")
            op = RecoveredLinear(export)
            item = {
                key: candidate[key]
                for key in ["rank", "fit", "factor_dtype", "export", "export_sha256"]
            }
            item["cases"] = []
            for name, start, end in cases:
                got = op(x[start:end])
                error = (got.float() - y[start:end]).abs()
                worst = int(error.argmax())
                item["cases"].append(
                    {
                        "case": name,
                        "rows": end - start,
                        "teacher_metrics": layer_metrics(got, y[start:end]),
                        "window_metrics": layer_metrics(got, window_outputs[name]),
                        "worst_token": start + worst // n,
                        "worst_channel": worst % n,
                    }
                )
            result["candidates"].append(item)
            publish()
            print(
                "REPLAY",
                candidate["rank"],
                candidate["fit"],
                candidate["factor_dtype"],
                sum(c["window_metrics"]["local_tolerance_pass"] for c in item["cases"]),
                len(cases),
                flush=True,
            )
    result["complete"] = True
    publish()


if __name__ == "__main__":
    main()
