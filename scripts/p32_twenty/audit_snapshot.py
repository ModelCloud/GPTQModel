"""Read-only snapshot representation checks, partitioned over physical GPUs."""

import argparse
import hashlib
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
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.worker < args.workers:
        parser.error("worker must be in [0, workers)")
    if args.output.resolve().is_relative_to(SNAPSHOT.resolve()):
        parser.error("results cannot be written into the snapshot")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.uuid
    for sample in range(3):
        row = subprocess.check_output(
            [
                "nvidia-smi",
                "--id=" + args.uuid,
                "--query-gpu=index,pci.bus_id,uuid,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        fields = [x.strip() for x in row.split(",")]
        if fields[2] != args.uuid or int(fields[3]) > 8 or int(fields[4]) != 0:
            raise RuntimeError("Idle gate failed: " + row)
        processes = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        if args.uuid in processes:
            raise RuntimeError("Foreign GPU process: " + processes)
        print("IDLE", sample + 1, row, "memory allowance=8MiB", flush=True)
        time.sleep(1)
    import torch
    from safetensors import safe_open

    from gptqmodel.quantization.qvq import (
        reconstruct_p32_window_inner_weight,
        reconstruct_qvq_inner_weight,
        repack_p32_planar_to_window,
        repack_p32_window_to_planar,
    )

    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]

    def read(name):
        with safe_open(str(SNAPSHOT / index[name]), framework="pt", device="cpu") as f:
            return f.get_tensor(name).to("cuda")

    def sha(path):
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

    shard_hashes = {name: sha(SNAPSHOT / name) for name in sorted(set(index.values()))}
    modules = sorted(
        name[:-8]
        for name in index
        if name.endswith(".trellis") and name[:-8] + ".bank_alt_id" in index
    )
    report = {
        "status": "correctness audit only; no performance or model-quality claim",
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "worker": args.worker,
        "workers": args.workers,
        "uuid": args.uuid,
        "device": str(torch.cuda.get_device_properties(0)),
        "torch": torch.__version__,
        "snapshot": str(SNAPSHOT),
        "shard_sha256_before": shard_hashes,
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for prefix in modules[args.worker :: args.workers]:
        trellis, bank, alt, su, sv = (
            read(prefix + "." + key)
            for key in ("trellis", "bank_ids", "bank_alt_id", "SU", "SV")
        )
        bits = trellis.shape[-1] / 8
        window = repack_p32_planar_to_window(trellis, bits=bits)
        back = repack_p32_window_to_planar(window, bits=bits)
        kwargs = {
            "bits": bits,
            "in_features": su.numel(),
            "out_features": sv.numel(),
            "bank_ids": bank,
            "bank_alt_id": alt,
        }
        canonical = reconstruct_qvq_inner_weight(trellis, v2b2_p32=True, **kwargs)
        decoded = reconstruct_p32_window_inner_weight(window, **kwargs)
        row = {
            "module": prefix,
            "bits": bits,
            "K": su.numel(),
            "N": sv.numel(),
            "exact_roundtrip": torch.equal(back, trellis),
            "exact_reconstructed_values": torch.equal(canonical, decoded),
            "finite": bool(torch.isfinite(canonical).all()),
            "payload_bytes": trellis.numel() * trellis.element_size(),
            "window_bytes": window.numel() * window.element_size(),
            "bank_alt_id": alt.item(),
        }
        report["rows"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        if not all(
            row[k] for k in ("exact_roundtrip", "exact_reconstructed_values", "finite")
        ):
            raise RuntimeError("Representation mismatch: " + prefix)
        del trellis, bank, alt, su, sv, window, back, canonical, decoded
    report["snapshot_unchanged"] = all(
        sha(SNAPSHOT / name) == value for name, value in shard_hashes.items()
    )
    if not report["snapshot_unchanged"]:
        raise RuntimeError("Snapshot shard identity changed during run")
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
