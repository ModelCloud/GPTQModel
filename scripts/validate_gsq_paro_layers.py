"""Execute prepared real QKV ParoQuant module-scope GSQ comparisons on a leased GPU."""

import argparse
import json
import os
import subprocess
import threading
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
    from gptqmodel.looper.named_module import NamedModule
    from gptqmodel.looper.paroquant_processor import ParoQuantProcessor
    from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear
    from gptqmodel.quantization.config import GSQConfig, ParoConfig, QuantizeConfig
    from gptqmodel.quantization.paroquant.optimization import _apply_inverse_rotation

    torch.set_num_threads(4)
    torch.manual_seed(7)
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
        out_features, in_features = teacher.shape
        weighted = [x * provenance["train_source_weights"][row["source_name"]] ** .5
                    for x, row in zip(fixture["train"], documents["train"], strict=True)]
        train, validation = torch.cat(weighted[:12]).cuda(), torch.cat(weighted[12:]).cuda()
        report["layers"][name] = {}
        baseline = None
        for arm in provenance["arms"]:
            print("PARO_START", name, arm, tuple(teacher.shape), flush=True)
            report["state"] = f"quantizing {name}/{arm}"
            write_json(output / "report.json", report)
            cfg = ParoConfig(bits=4, group_size=128, krot=8, opt_seed=7, opt_scope="module",
                             gsq=None if arm == "baseline" else GSQConfig(
                                 enabled=True, seed=7, steps=100, learn_scales=arm == "gsq_scales"))
            payload = cfg.to_dict()
            cfg = QuantizeConfig.from_quant_config(payload)
            write_json(output / f"{name}.{arm}.config.json", payload)
            processor = object.__new__(ParoQuantProcessor)
            processor.qcfg = cfg
            processor.lock = threading.Lock()
            processor.calculate_w_wq_diff = False
            processor._has_explicit_validation_calibration = True
            linear = torch.nn.Linear(in_features, out_features, bias=False, device="cuda", dtype=torch.float16)
            linear.weight.data.copy_(teacher)
            module = NamedModule(linear, name=name.rsplit(".", 1)[-1], full_name=name, layer_index=0)
            started = time.monotonic()
            losses = processor._quantize_one_module(module, train, validation)
            state = module.state
            kwargs = dict(bits=4, group_size=128, sym=cfg.sym, desc_act=False, in_features=in_features,
                          out_features=out_features, bias=False, register_buffers=True, krot=8)
            packed = ParoLinear(**kwargs)
            transport = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float16)
            transport.weight.data.copy_(state["pack_weight"])
            packed.pack(transport, state["q_scales"], state["q_zeros"])
            packed.pairs.copy_(state["pairs"])
            packed.theta.copy_(state["theta"])
            packed.channel_scales.copy_(state["channel_scales"].reshape_as(packed.channel_scales))
            path = output / f"{name}.{arm}.pt"
            torch.save(packed.state_dict(), path)
            loaded = torch.load(path, weights_only=True)
            restored = ParoLinear(**kwargs)
            restored.load_state_dict(loaded, strict=True)
            assert all(torch.equal(value, restored.state_dict()[key]) for key, value in loaded.items())
            if baseline is None:
                baseline = {key: value.clone() for key, value in loaded.items()}
            equal = all(torch.equal(value, baseline[key]) for key, value in loaded.items())
            restored = restored.cuda().eval()
            restored.post_init()
            # Runtime rotations cast metadata; derive the reference from exported
            # tensors, never from the optimizer's higher-precision pseudo-weight.
            from gptqmodel.quantization.gsq_scalar import affine_codes
            groups = torch.arange(in_features, device="cuda") // 128
            codes = affine_codes(state["pack_weight"].cuda(), state["q_scales"].cuda(),
                                 state["q_zeros"].cuda(), groups, 4, packing="awq_gemm")
            scales = state["q_scales"].cuda().float().repeat_interleave(128, dim=1)
            zeros = state["q_zeros"].cuda().float().repeat_interleave(128, dim=1)
            grid = (codes.float() - zeros) * scales
            reference_weight = _apply_inverse_rotation(grid, state["pairs"].cuda(), state["theta"].cuda().float(),
                                               group_size=128, fused_rotation=False)
            reference_weight *= state["channel_scales"].cuda().float().reshape(1, -1)
            checks, error_sum, count = [], 0., 0
            with torch.no_grad():
                for index, values in enumerate(fixture["heldout"]):
                    x = values.cuda().half()
                    actual = restored(x).float()
                    reference = x.float() @ reference_weight.T
                    delta = (actual - reference).abs()
                    check = {"document": index, "mean": delta.mean().item(), "max": delta.max().item()}
                    checks.append(check)
                    if not torch.isfinite(actual).all() or check["mean"] > .002 or check["max"] > .046875:
                        write_json(output / "failed-check.json", {"layer": name, "arm": arm, **check})
                        raise AssertionError(f"Native parity failed: {check}")
                    error_sum += (actual.double() - x.float().matmul(teacher.T).double()).square().sum().item()
                    count += actual.numel()
            report["layers"][name][arm] = {
                "train_loss": losses[0], "validation_loss": losses[1],
                "gsq": state.get("gsq_diagnostics"), "payload_equal_baseline": equal,
                "payload_sha256": digest(path), "native_checks": checks, "heldout_mse": error_sum / count,
                "seconds": time.monotonic() - started}
            write_json(output / "report.json", report)
            print("PARO_DONE", name, arm, error_sum / count, "baseline_equal", equal, flush=True)
            del module, linear, processor, restored, packed, reference_weight
            torch.cuda.empty_cache()
    report["state"] = "complete"
    write_json(output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    execute(parser.parse_args().output.resolve())
