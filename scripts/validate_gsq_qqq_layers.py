"""Real saved F6/seed7 QKV inputs through QQQ W4 quantize, reload and native forward.

Selected-projection reconstruction only; this does not measure full-model KL/Top-K.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from scripts.validate_qvq_gsq_layers import ROOT, TARGETS, digest, write_json


def prepare(args):
    import torch
    from safetensors import safe_open

    if args.output.exists():
        raise ValueError("Use a new output directory")
    source = json.loads((args.inputs / "provenance.json").read_text())
    rows = json.loads((args.inputs / "inputs.json").read_text())
    if digest(args.inputs / "inputs.json") != source["inputs_sha256"]:
        raise ValueError("Saved token selection changed")
    for path, sha in source["source_hashes"].items():
        if digest(path) != sha:
            raise ValueError(f"Calibration provenance changed: {path}")
    dense = Path(source["dense"]) / "model.safetensors"
    paths = [dense, args.inputs / "inputs.json", args.inputs / "provenance.json"]
    with safe_open(dense, framework="pt", device="cpu") as handle:
        for name in TARGETS:
            path = args.inputs / f"{name}.inputs.pt"
            fixture = torch.load(path, weights_only=True)
            if not torch.equal(fixture["weight"], handle.get_tensor(name + ".weight").float()):
                raise ValueError(f"Saved weights differ from dense model: {name}")
            for split in ("train", "heldout"):
                if len(fixture[split]) != len(rows[split]):
                    raise ValueError("Saved activation rows differ from tokens")
                for activation, row in zip(fixture[split], rows[split], strict=True):
                    if activation.shape != (len(row["input_ids"]), fixture["weight"].shape[1]):
                        raise ValueError("Saved activation dimensions differ from tokens/weights")
                    if not torch.isfinite(activation).all():
                        raise ValueError("Nonfinite saved activation")
            paths.append(path)
    sources = [Path(__file__), ROOT / "gptqmodel/quantization/qqq.py",
               ROOT / "gptqmodel/quantization/gsq_qqq.py", ROOT / "gptqmodel/quantization/config.py",
               ROOT / "gptqmodel/nn_modules/qlinear/qqq.py", ROOT / "gptqmodel_ext/qqq/qqq_gemm.cu"]
    args.output.mkdir(parents=True)
    write_json(args.output / "provenance.json", {
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "inputs": str(args.inputs), "steps": args.steps, "group_size": args.group_size,
        "seed": 7, "bits": 4, "arms": ["baseline", "gsq_fixed"],
        "scope": "full block-0 QKV; native W4A8 local reconstruction; no full-model propagation",
        "files": {str(p): digest(p) for p in paths + sources},
        "train_source_weights": source["train_source_weights"],
    })


def execute(args):
    provenance = json.loads((args.output / "provenance.json").read_text())
    if any(provenance[k] != getattr(args, k) for k in ("steps", "group_size")):
        raise ValueError("Prepared configuration changed")
    if provenance["inputs"] != str(args.inputs) or (args.output / "report.json").exists():
        raise ValueError("Changed inputs or existing execution; use a fresh prepared directory")
    for path, sha in provenance["files"].items():
        if digest(path) != sha:
            raise ValueError(f"Prepared file changed: {path}")
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not uuid.startswith("GPU-") or "," in uuid or not os.environ.get("GPU_ALLOCATOR_LEASE_ID"):
        raise ValueError("Requires one exclusive UUID GPU allocator lease")
    for _ in range(3):
        inventory = subprocess.check_output([
            "nvidia-smi", "--id=" + uuid,
            "--query-gpu=index,pci.bus_id,uuid,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True).strip()
        fields = [v.strip() for v in inventory.split(",")]
        processes = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
        if fields[2] != uuid or int(fields[4]) > 8 or int(fields[5]) != 0 or uuid in processes:
            raise RuntimeError("Idle preflight failed: " + inventory)
        print("IDLE", inventory, flush=True)
        time.sleep(1)

    import torch
    from gptqmodel.nn_modules.qlinear.qqq import QQQLinear, QQQTorchLinear
    from gptqmodel.quantization.config import GSQConfig, QQQConfig, QuantizeConfig
    from gptqmodel.quantization.qqq import QQQ

    torch.set_num_threads(4)
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    rows = json.loads((args.inputs / "inputs.json").read_text())
    report = {"state": "running", "provenance": provenance, "inventory": inventory,
              "torch": str(torch.__version__), "cuda": torch.version.cuda, "layers": {}}
    for name in TARGETS:
        fixture = torch.load(args.inputs / f"{name}.inputs.pt", weights_only=True)
        teacher = fixture["weight"].cuda().half()
        out_features, in_features = teacher.shape
        x = torch.cat([value * provenance["train_source_weights"][row["source_name"]] ** .5
                       for value, row in zip(fixture["train"], rows["train"], strict=True)]).cuda()
        report["layers"][name] = {}
        baseline_codes = None
        for arm in ("baseline", "gsq_fixed"):
            report["state"] = f"quantizing {name}/{arm}"
            write_json(args.output / "report.json", report)
            cfg = QQQConfig(bits=4, group_size=args.group_size, desc_act=False,
                            gsq=None if arm == "baseline" else GSQConfig(enabled=True, steps=args.steps, seed=7))
            config_path = args.output / f"{name}.{arm}.config.json"
            write_json(config_path, cfg.to_dict())
            cfg = QuantizeConfig.from_quant_config(json.loads(config_path.read_text()))
            layer = torch.nn.Linear(in_features, out_features, bias=False, device="cuda", dtype=torch.float16)
            layer.weight.data.copy_(teacher)
            task = QQQ(layer, cfg)
            task.quantizer.configure(4, perchannel=True, sym=True, mse=False, groupsize=args.group_size)
            task.add_batch(x, None)
            weight, scales, _, _, duration, _, _, extra, _ = task.quantize()
            layer.weight.data.copy_(weight)
            kwargs = dict(bits=4, group_size=args.group_size, sym=True, desc_act=False,
                          in_features=in_features, out_features=out_features, bias=False)
            reference = QQQTorchLinear(**kwargs)
            reference.pack(layer.cpu(), scales.cpu(), extra.cpu())
            path = args.output / f"{name}.{arm}.pt"
            torch.save(reference.state_dict(), path)
            state = torch.load(path, weights_only=True)
            if any(not torch.equal(value, state[key]) for key, value in reference.state_dict().items()):
                raise ValueError("Packed serialization changed state")
            native = QQQLinear(**kwargs)
            native.load_state_dict(state, strict=True)
            native = native.cuda().eval()
            native.post_init()
            reference = reference.cuda().eval()
            # Independently score the actual packed grid using explicit
            # activation residuals, without the optimizer's Hessian factor.
            with torch.inference_mode():
                integers, channel = reference._dequantize_weight_for_torch()
                canonical = (integers * channel).T
                quantized_x, token_scale = reference.dynamic_quant(x.half())
                deployed_x = quantized_x.float() * token_scale
                error = deployed_x @ (canonical - teacher.float()).T
                residual = (x.float() - deployed_x) @ teacher.float().T
                energy = (deployed_x @ teacher.float().T).square().sum()
                denominator = energy if energy > torch.finfo(torch.float32).eps else 1.0
                objective = float((error.square().sum() - 2 * (error * residual).sum()) / denominator)
            diagnostics = getattr(task, "gsq_diagnostics", None)
            if diagnostics is not None and abs(objective - diagnostics["after"]) > max(1e-8, abs(objective) * 1e-4):
                raise ValueError(f"Packed objective differs from GSQ diagnostics: {objective}, {diagnostics}")
            codes = reference._unpack_weight_codes().cpu()
            if baseline_codes is None:
                baseline_codes = codes.clone()
            checks = []
            with torch.inference_mode():
                for activation in fixture["heldout"]:
                    inputs = activation.cuda().half()
                    actual = native(inputs).float()
                    expected = reference(inputs).float()
                    delta = (actual - expected).abs()
                    target = activation.cuda().float() @ fixture["weight"].cuda().T
                    checks.append({"tokens": len(inputs), "mean_abs": delta.mean().item(),
                                   "max_abs": delta.max().item(),
                                   "mse": (actual - target).square().mean().item()})
                    if not torch.isfinite(actual).all() or delta.mean() > .002 or delta.max() > .046875:
                        raise ValueError("Native/reference localized gate failed")
            report["layers"][name][arm] = {
                "quantize_seconds": duration, "diagnostics": diagnostics, "packed_objective": objective,
                "payload_sha256": digest(path), "changed_codes": int((codes != baseline_codes).sum()),
                "heldout": checks, "mse": sum(r["mse"] * r["tokens"] for r in checks) /
                sum(r["tokens"] for r in checks),
            }
            write_json(args.output / "report.json", report)
            print("LAYER", name, arm, report["layers"][name][arm]["mse"], flush=True)
            task.free()
            del task, native, reference, layer
    report["state"] = "complete"
    write_json(args.output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--group-size", type=int, choices=(-1, 128), default=128)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    prepare(args) if args.prepare else execute(args)


if __name__ == "__main__":
    main()
