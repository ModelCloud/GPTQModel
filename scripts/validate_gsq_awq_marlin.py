"""Check real saved GSQ AWQ GEMM payloads through the Marlin runtime conversion."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from scripts.validate_qvq_gsq_layers import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output file")
    source = json.loads((args.run / "report.json").read_text())
    if source["state"] != "complete" or source["provenance"]["method"] != "awq":
        raise ValueError("Expected completed real AWQ run")
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not uuid.startswith("GPU-") or "," in uuid or not os.environ.get("GPU_ALLOCATOR_LEASE_ID"):
        raise ValueError("Run under one exclusive UUID allocator lease")
    for sample in range(3):
        inventory = subprocess.check_output([
            "nvidia-smi", "--id="+uuid,
            "--query-gpu=index,pci.bus_id,uuid,name,memory.used,memory.total,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits"], text=True).strip()
        fields = [v.strip() for v in inventory.split(",")]
        processes = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
        if fields[2] != uuid or int(fields[4]) > 8 or int(fields[6]) != 0 or uuid in processes:
            raise RuntimeError("Idle GPU preflight failed: " + inventory)
        print("IDLE", sample+1, inventory, flush=True)
        time.sleep(1)

    import torch

    from gptqmodel.nn_modules.qlinear.marlin_awq import AwqMarlinLinear
    from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    fixture_path = Path(source["provenance"]["awq_calibration"]) / "calibration.pt"
    if digest(fixture_path) != source["provenance"]["awq_calibration_sha256"]:
        raise ValueError("Real activation fixture changed")
    fixture = torch.load(fixture_path, weights_only=True, map_location="cpu")
    heldout = fixture["runtime_scaled_features"]["heldout"][0][0]
    report = {"state": "running", "source_report_sha256": digest(args.run / "report.json"),
              "inventory": inventory, "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": str(torch.cuda.get_device_properties(0)), "rows": [],
              "scope": "real selected-QKV AWQ GEMM load, Marlin repack and forward; no model-quality or speed claim",
              "source_hashes": {str(p): digest(p) for p in (
                  Path(__file__), Path("gptqmodel/nn_modules/qlinear/marlin_awq.py"),
                  Path("gptqmodel/utils/marlin.py"))},
              "kernel_source_hashes": {str(p): digest(p) for p in Path("gptqmodel_ext/marlin").rglob("*")
                                       if p.is_file() and p.suffix in (".cu", ".cuh", ".h", ".cpp")}}
    write_json(args.output, report)
    for name, layer in source["layers"].items():
        out_features, in_features = layer["shape"]
        for arm, record in layer["arms"].items():
            path = args.run / f"{name}.{arm}.pt"
            if digest(path) != record["payload_sha256"]:
                raise ValueError("Saved AWQ payload changed")
            state = torch.load(path, weights_only=True, map_location="cpu")
            canonical = dequantize_gemm(state["qweight"], state["qzeros"], state["scales"].float(), 4, 128).cuda()
            module = AwqMarlinLinear(bits=4, group_size=128, desc_act=False, sym=False,
                                    in_features=in_features, out_features=out_features, bias=False,
                                    dtype=torch.float16, register_buffers=True)
            module.load_state_dict(state, strict=True)
            if any(not torch.equal(v, module.state_dict()[k]) for k, v in state.items()):
                raise ValueError("GEMM payload changed during reload")
            module = module.cuda().eval()
            module.post_init()
            for tokens in (1, 16):
                x = heldout[:tokens].cuda()
                with torch.inference_mode():
                    actual = module(x).float()
                    reference = x.float() @ canonical
                error = (actual-reference).abs()
                row = {"module": name, "arm": arm, "tokens": tokens,
                       "mean_abs": error.mean().item(), "max_abs": error.max().item(),
                       "finite": bool(torch.isfinite(actual).all()), "payload_sha256": digest(path),
                       "mean_limit": .002, "max_limit": .046875}
                row["pass"] = row["finite"] and row["mean_abs"] <= .002 and row["max_abs"] <= .046875
                report["rows"].append(row)
                write_json(args.output, report)
                print("CASE", json.dumps(row), flush=True)
                if not row["pass"]:
                    raise ValueError("Marlin native parity gate failed")
            del module, canonical
    report["state"] = "complete"
    write_json(args.output, report)
    print("COMPLETE", len(report["rows"]), flush=True)


if __name__ == "__main__":
    main()
