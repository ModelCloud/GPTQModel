"""Execute prepared real QKV FP8 GSQ comparisons on a leased GPU."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from scripts.validate_qvq_gsq_layers import TARGETS, digest, write_json


def execute(output):
    provenance = json.loads((output / "provenance.json").read_text())
    if (output / "report.json").exists():
        raise ValueError("Execution already started; preserve it and prepare a fresh directory")
    for path, sha in provenance["files"].items():
        if digest(path) != sha:
            raise ValueError(f"Prepared source changed: {path}")
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not uuid.startswith("GPU-") or "," in uuid or not os.environ.get("GPU_ALLOCATOR_LEASE_ID"):
        raise ValueError("Requires one exclusive GPU allocator UUID lease")
    for _ in range(3):
        inventory = subprocess.check_output([
            "nvidia-smi", "--id=" + uuid,
            "--query-gpu=index,pci.bus_id,uuid,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True).strip()
        fields = [item.strip() for item in inventory.split(",")]
        processes = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
        if fields[2] != uuid or int(fields[4]) > 8 or int(fields[5]) or uuid in processes:
            raise RuntimeError("Idle GPU preflight failed: " + inventory)
        print("IDLE", inventory, flush=True)
        time.sleep(1)

    import torch
    from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
    from gptqmodel.quantization.gsq_fp8 import refine_fp8_weight

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    inputs = Path(provenance["inputs"])
    documents = json.loads((inputs / "inputs.json").read_text())
    props = torch.cuda.get_device_properties(0)
    report = {"state": "running", "provenance": provenance, "inventory": inventory,
              "runner_sha256": digest(__file__), "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "compute_capability": [props.major, props.minor], "sm_count": props.multi_processor_count,
              "memory_bytes": props.total_memory, "layers": {}}
    write_json(output / "report.json", report)
    for name in TARGETS:
        fixture = torch.load(inputs / f"{name}.inputs.pt", weights_only=True)
        teacher = fixture["weight"].cuda()
        rows, columns = teacher.shape
        train = torch.cat([x * provenance["train_source_weights"][row["source_name"]] ** .5
                           for x, row in zip(fixture["train"], documents["train"], strict=True)]).cuda()
        kwargs = dict(bits=8, group_size=-1, sym=True, desc_act=False,
                      in_features=columns, out_features=rows, bias=False)
        baseline = TorchFP8Linear(**kwargs)
        dense = torch.nn.Linear(columns, rows, bias=False)
        dense.weight.data.copy_(fixture["weight"])
        baseline.pack_original(dense, None, None)
        baseline = baseline.cuda()
        report["layers"][name] = {}
        for arm in provenance["arms"]:
            print("FP8_START", name, arm, tuple(teacher.shape), flush=True)
            started = time.monotonic()
            result = refine_fp8_weight(
                baseline.weight, baseline.weight_scale_inv, target=teacher,
                inputs=train if arm == "gsq_calibrated" else None,
                config={"enabled": arm != "baseline", "steps": provenance["steps"],
                        "candidates": provenance["candidates"], "seed": provenance["seed"]})
            packed = TorchFP8Linear(**kwargs)
            packed.weight = result["weight"].cpu()
            packed.weight_scale_inv = result["scale_inv"].cpu()
            path = output / f"{name}.{arm}.pt"
            torch.save(packed.state_dict(), path)
            restored = TorchFP8Linear(**kwargs)
            restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
            restored = restored.cuda()
            decoded = restored.dequantize_weight(dtype=torch.float32)
            errors, count = 0., 0
            checks = []
            with torch.no_grad():
                for index, values in enumerate(fixture["heldout"]):
                    x = values.cuda().float()
                    actual = restored(x)
                    reference = x @ decoded
                    drift = (actual-reference).abs()
                    check = {"document": index, "mean": drift.mean().item(), "max": drift.max().item()}
                    checks.append(check)
                    if not torch.isfinite(actual).all() or check["mean"] > .002 or check["max"] > .046875:
                        raise AssertionError(f"FP8 runtime parity failed: {check}")
                    errors += (actual.double() - (x @ teacher.T).double()).square().sum().item()
                    count += actual.numel()
            report["layers"][name][arm] = {
                "gsq": {key: result[key] for key in ("before", "after", "history")},
                "changed_codes": int((result["weight"].view(torch.uint8)
                                      != baseline.weight.view(torch.uint8)).sum()),
                "heldout_mse": errors/count, "runtime_checks": checks,
                "payload_sha256": digest(path), "seconds": time.monotonic()-started}
            write_json(output / "report.json", report)
            print("FP8_DONE", name, arm, errors/count, flush=True)
        del teacher, train, baseline, restored, decoded
        torch.cuda.empty_cache()
    report["state"] = "complete"
    write_json(output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    execute(parser.parse_args().output.resolve())
