"""Real F6/seed7 QKV scalar lifecycle, packed reload and held-out propagation.

This is a selected-projection experiment, not a complete scalar-model export.
All other projections use the original F6 snapshot's canonical QVQ operator.
"""

import argparse
import json
import os
import shutil
import subprocess
import time
import types
from pathlib import Path

from scripts.validate_qvq_gsq_layers import ROOT, SNAPSHOT, TARGETS, digest, prepare as prepare_inputs, write_json


ARMS = ("baseline", "gsq_fixed", "gsq_scales")


def prepare(args):
    # Reuse the historical-source audit and exact stratified token selection.
    args.gsq, args.gsq_lifecycle, args.target_bits = True, False, None
    prepare_inputs(args)
    provenance = json.loads((args.output / "provenance.json").read_text())
    sources = [Path(__file__), ROOT / "scripts/validate_qvq_gsq_layers.py",
               ROOT / "gptqmodel/quantization/config.py", ROOT / "gptqmodel/quantization/gsq_scalar.py",
               ROOT / "gptqmodel/quantization/gptq.py", ROOT / "gptqmodel/quantization/rtn.py",
               ROOT / "gptqmodel/nn_modules/qlinear/__init__.py", ROOT / "gptqmodel/nn_modules/qlinear/torch.py"]
    if args.method == "awq":
        sources += [ROOT / "gptqmodel/looper/awq_processor.py", ROOT / "gptqmodel/quantization/awq/quantize/scale.py",
                    ROOT / "gptqmodel/nn_modules/qlinear/torch_awq.py"]
        calibration_report = json.loads((args.awq_calibration / "report.json").read_text())
        if calibration_report["state"] != "complete" or calibration_report["inputs_sha256"] != digest(args.output / "inputs.json"):
            raise ValueError("AWQ calibration must be complete and use identical token selections")
        provenance["awq_calibration"] = str(args.awq_calibration)
        provenance["awq_calibration_sha256"] = digest(args.awq_calibration / "calibration.pt")
        if provenance["awq_calibration_sha256"] != calibration_report["calibration_sha256"]:
            raise ValueError("AWQ calibration fixture changed")
    provenance.update({
        "experiment": "scalar-qkv-lifecycle", "method": args.method, "bits": args.bits,
        "group_size": args.group_size, "arms": ARMS,
        "scope": "full QKV projections; GPTQ v2 packing; Torch GPU layer checks; F6 canonical full-model propagation",
        "source_code": {str(p): digest(p) for p in sources},
        "inputs_sha256": digest(args.output / "inputs.json"),
    })
    if args.method == "awq":
        provenance["scope"] = "full scaled QKV projections; AWQ GEMM packing; Torch GPU checks; F6 FP32 propagation with scaled RMSNorm"
        provenance["awq_weighting"] = "uniform tokens; source weights not applied"
    (args.output / "executed-source").mkdir()
    for p in sources:
        destination = args.output / "executed-source" / p.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, destination)
    write_json(args.output / "provenance.json", provenance)


def execute(args):
    provenance = json.loads((args.output / "provenance.json").read_text())
    expected = {"method": args.method, "bits": args.bits, "group_size": args.group_size,
                "steps": args.steps, "candidate_count": args.candidates,
                "dense": str(args.dense), "snapshot": str(args.snapshot)}
    if args.method == "awq":
        expected.update(awq_calibration=str(args.awq_calibration),
                        awq_calibration_sha256=digest(args.awq_calibration / "calibration.pt"))
    if any(provenance.get(k) != v for k, v in expected.items()):
        raise ValueError("Execution differs from the prepared scalar experiment")
    if (args.output / "report.json").exists():
        raise ValueError("Use a new output directory; never overwrite an earlier run")
    for path, sha in (provenance["source_hashes"] | provenance["file_hashes"] | provenance["source_code"]).items():
        if digest(path) != sha:
            raise ValueError(f"Prepared source changed: {path}")
    if digest(args.output / "inputs.json") != provenance["inputs_sha256"]:
        raise ValueError("Prepared tokens changed")
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not uuid.startswith("GPU-") or "," in uuid or not os.environ.get("GPU_ALLOCATOR_LEASE_ID"):
        raise ValueError("Run under one exclusive UUID GPU allocator lease")
    inventory = None
    for sample in range(3):
        inventory = subprocess.check_output([
            "nvidia-smi", "--id=" + uuid,
            "--query-gpu=index,pci.bus_id,uuid,name,memory.used,memory.total,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits"], text=True).strip()
        fields = [v.strip() for v in inventory.split(",")]
        procs = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
        if fields[2] != uuid or int(fields[4]) > 8 or int(fields[6]) != 0 or uuid in procs:
            raise RuntimeError("Idle GPU preflight failed: " + inventory)
        print("IDLE", sample + 1, inventory, flush=True)
        time.sleep(1)

    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    from gptqmodel import BACKEND
    from gptqmodel.looper.awq_processor import AWQProcessor
    from gptqmodel.looper.named_module import NamedModule
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization import AWQConfig, FORMAT, GPTQConfig, GSQConfig, QuantizeConfig, RTNConfig
    from gptqmodel.quantization.awq.quantize.scale import apply_clip
    from gptqmodel.quantization.awq.utils.packing_utils import dequantize_gemm
    from gptqmodel.quantization.gptq import GPTQ
    from gptqmodel.quantization.qvq import reconstruct_qvq_inner_weight
    from gptqmodel.quantization.rotation.hadamard_utils import matmul_hadU
    from gptqmodel.quantization.rtn import RTN
    from scripts.p32_twenty.scorecard import logits_metrics

    torch.set_num_threads(4)
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    data = json.loads((args.output / "inputs.json").read_text())
    report = {"state": "dense capture", "provenance": provenance, "inventory": inventory,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": str(torch.cuda.get_device_properties(0)), "layers": {}, "model": {}}
    write_json(args.output / "report.json", report)
    model = AutoModelForCausalLM.from_pretrained(
        args.dense, dtype=torch.float32, device_map={"": "cuda:0"},
        attn_implementation="eager", local_files_only=True).eval().requires_grad_(False)
    weights = {name: model.get_submodule(name).weight.detach().clone() for name in TARGETS}
    captures = {name: {"train": [], "heldout": []} for name in TARGETS}
    split = "train"
    hooks = []
    for name in TARGETS:
        def capture(module, values, name=name):
            captures[name][split].append(values[0].detach().reshape(-1, values[0].shape[-1]).clone())
        hooks.append(model.get_submodule(name).register_forward_pre_hook(capture))
    (args.output / "teacher").mkdir()
    with torch.inference_mode():
        for split in ("train", "heldout"):
            for i, row in enumerate(data[split]):
                ids = torch.tensor([row["input_ids"]], device="cuda")
                if split == "train":
                    model.model(ids, use_cache=False)
                else:
                    logits = model(ids, use_cache=False).logits[0].float()
                    if not torch.isfinite(logits).all():
                        raise ValueError("Nonfinite dense logits")
                    torch.save(logits.cpu(), args.output / "teacher" / f"{i}.pt")
                print("DENSE", split, i + 1, flush=True)
    for hook in hooks:
        hook.remove()

    decoded_arms = {arm: {} for arm in ARMS}
    awq = torch.load(args.awq_calibration / "calibration.pt", weights_only=True) if args.method == "awq" else None
    for name in TARGETS:
        dense = weights[name]
        out_features, in_features = dense.shape
        x = torch.cat([value * provenance["train_source_weights"][row["source_name"]]**0.5
                       for value, row in zip(captures[name]["train"], data["train"], strict=True)])
        fitting_weight = dense.half()
        if awq is not None:
            relative = name.removeprefix("model.layers.0.")
            fitting_weight = awq["scaled_block"][relative + ".weight"].cuda()
            x = torch.cat(awq["features"]["train"], dim=1)[0].cuda() / awq["scale"].cuda()
        torch.save({"weight": dense.cpu(), "train": [v.cpu() for v in captures[name]["train"]],
                    "heldout": [v.cpu() for v in captures[name]["heldout"]]}, args.output / f"{name}.inputs.pt")
        report["layers"][name] = {"shape": list(dense.shape), "arms": {}}
        for arm in ARMS:
            report["state"] = f"quantizing {name}/{arm}"
            write_json(args.output / "report.json", report)
            gsq = None if arm == "baseline" else GSQConfig(
                enabled=True, steps=args.steps, candidates=args.candidates, seed=7, learn_scales=arm == "gsq_scales")
            cls, cfg_cls = (GPTQ, GPTQConfig) if args.method == "gptq" else (RTN, RTNConfig)
            common = dict(bits=args.bits, group_size=args.group_size, format=FORMAT.GPTQ_V2, gsq=gsq)
            if args.method == "gptq":
                common.update(desc_act=True, act_group_aware=False, hessian={"length_aware": False})
            cfg = cfg_cls(**common)
            if awq is not None:
                cfg = AWQConfig(bits=4, group_size=128, sym=False, format=FORMAT.GEMM, gsq=gsq)
            path = args.output / f"{name}.{arm}.config.json"
            write_json(path, cfg.to_dict())
            restored_cfg = QuantizeConfig.from_quant_config(json.loads(path.read_text()))
            if restored_cfg.gsq != cfg.gsq:
                raise ValueError("GSQ config reload changed the control")
            layer = torch.nn.Linear(in_features, out_features, bias=False, device="cuda", dtype=torch.float16)
            layer.weight.data.copy_(fitting_weight)
            if awq is not None:
                task = AWQProcessor(tokenizer=None, qcfg=restored_cfg, calibration=None, prepare_dataset_func=None,
                                    calibration_concat_size=None, calibration_sort=None, batch_size=1,
                                    gptq_model=types.SimpleNamespace(rotary_embedding=None), model=None)
                named = NamedModule(layer, name=relative, full_name=name, layer_index=0)
                references = {relative: fitting_weight.detach().clone()}
                started = time.perf_counter()
                if relative.endswith("v_proj"):
                    container = torch.nn.Module()
                    container.add_module("projection", layer)
                    clip = task._search_best_clip(container, {"projection": layer}, {"projection": x})
                    apply_clip(container, clip)
                task.apply_quant({relative: named}, [], input_features={relative: x}, adjacent_references=references)
                named.stream_sync()
                wq, scales, zeros = layer.weight.detach(), named.state["q_scales"], named.state["q_zeros"]
                groups = torch.arange(in_features, dtype=torch.int32, device="cuda") // args.group_size
                duration = time.perf_counter()-started
                task.gsq_diagnostics = named.state.get("gsq_diagnostics")
            else:
                task = cls(layer, qcfg=restored_cfg)
                if args.method == "gptq":
                    task.quantizer.configure(perchannel=True)
                    task.add_batch(x, None)
                wq, scales, zeros, groups, duration, *_ = task.quantize()
            linear = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float16)
            linear.weight.data.copy_(wq.cpu())

            def new_packed():
                if awq is not None:
                    return AwqTorchLinear(bits=4, group_size=128, sym=False, desc_act=False,
                                          in_features=in_features, out_features=out_features, bias=False,
                                          register_buffers=True)
                return TorchLinear(bits=args.bits, group_size=args.group_size, sym=cfg.sym,
                                   desc_act=cfg.desc_act, in_features=in_features, out_features=out_features,
                                   bias=False, backend=BACKEND.TORCH, format=FORMAT.GPTQ_V2)

            packed = new_packed()
            if awq is not None:
                packed.pack(linear, scales.cpu(), zeros.cpu())
            else:
                packed.pack_original(linear, scales.cpu(), zeros.cpu(), groups.cpu())
            payload = args.output / f"{name}.{arm}.pt"
            torch.save(packed.state_dict(), payload)
            restored = new_packed()
            restored.load_state_dict(torch.load(payload, map_location="cpu", weights_only=True), strict=True)
            if any(not torch.equal(v, restored.state_dict()[k]) for k, v in packed.state_dict().items()):
                raise ValueError("Packed state changed on reload")
            if awq is not None:
                canonical = dequantize_gemm(restored.qweight, restored.qzeros, restored.scales.float(), 4, 128).T.cuda()
            else:
                codes, z = restored._unpack_continuous_codes()
                canonical = (restored.scales.float()[restored.g_idx] *
                             (codes.float() - z.float()[restored.g_idx])).T.cuda()
            decoded_arms[arm][name] = canonical
            restored = restored.cuda().eval()
            sample = captures[name]["heldout"][0][:16].half()
            if awq is not None:
                sample = awq["runtime_scaled_features"]["heldout"][0][0, :16].cuda()
            with torch.inference_mode():
                actual = restored(sample).float()
                reference = sample.float() @ canonical.T
            delta = (actual-reference).abs()
            parity = {"mean_abs": delta.mean().item(), "max_abs": delta.max().item(),
                      "finite": bool(torch.isfinite(actual).all()), "tokens": len(sample),
                      "mean_limit": 0.002, "max_limit": 0.046875}
            parity["pass"] = parity["finite"] and parity["mean_abs"] <= .002 and parity["max_abs"] <= .046875
            local_rows = []
            for row_index, inputs in enumerate(captures[name]["heldout"]):
                target = inputs @ dense.T
                deployed_inputs = inputs if awq is None else awq["runtime_scaled_features"]["heldout"][row_index][0].cuda().float()
                error = deployed_inputs @ canonical.T - target
                local_rows.append({"tokens": len(inputs), "mse": error.square().mean().item(),
                                   "nmse": (error.square().sum()/target.square().sum()).item()})
            packed_objective_target = fitting_weight.float()
            metric_inputs = x.float() if args.method in ("gptq", "awq") else torch.eye(in_features, device="cuda")
            packed_loss = float(((canonical-packed_objective_target) @ metric_inputs.T).square().sum() /
                                (packed_objective_target @ metric_inputs.T).square().sum())
            diagnostics = getattr(task, "gsq_diagnostics", None)
            objective_match = diagnostics is None or abs(packed_loss-diagnostics["after"]) <= max(1e-8, packed_loss*1e-4)
            record = {"quantize_seconds": duration, "gsq": diagnostics, "packed_objective": packed_loss,
                      "packed_objective_matches": objective_match, "runtime_parity": parity,
                      "payload_sha256": digest(payload), "payload_bytes": payload.stat().st_size,
                      "reload_exact": True, "heldout": local_rows}
            report["layers"][name]["arms"][arm] = record
            write_json(args.output / "report.json", report)
            print("LAYER", name, arm, json.dumps({k: v for k, v in record.items() if k != "heldout"}), flush=True)
            if not objective_match or not parity["pass"]:
                raise ValueError("Packed objective or localized runtime gate failed; report retained")
            del task, layer, packed, restored, wq, scales, zeros, groups, linear
        del x

    snapshot = args.snapshot / "qvq-p32"
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    cfg = json.loads((snapshot / "quantize_config.json").read_text())

    def read(name):
        with safe_open(str(snapshot / index[name]), framework="pt") as handle:
            return handle.get_tensor(name).cuda()

    class CanonicalQVQ(torch.nn.Module):
        def __init__(self, inner, su, sv):
            super().__init__()
            self.register_buffer("inner", inner.float())
            self.register_buffer("su", su.float())
            self.register_buffer("sv", sv.float())

        def forward(self, inputs):
            return matmul_hadU(matmul_hadU(inputs.float()*self.su) @ self.inner)*self.sv

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in index:
                param.copy_(read(name))
    if cfg.get("activation") or cfg.get("incoherence") != "rht":
        raise ValueError("Unsupported F6 transform")
    for name in sorted(index):
        if not name.endswith(".trellis"):
            continue
        prefix = name[:-8]
        trellis, su, sv = [read(prefix + "." + suffix) for suffix in ("trellis", "SU", "SV")]
        p32 = prefix + ".bank_alt_id" in index
        bank = read(prefix + ".bank_ids") if prefix + ".bank_ids" in index else None
        alt = read(prefix + ".bank_alt_id") if p32 else None
        inner = reconstruct_qvq_inner_weight(
            trellis, bits=trellis.shape[-1]/8, in_features=su.numel(), out_features=sv.numel(),
            bank_ids=bank, bank_alt_id=alt, codebook_version=cfg["codebook"], v2b2_p32=p32)
        parent, leaf = prefix.rsplit(".", 1)
        setattr(model.get_submodule(parent), leaf, CanonicalQVQ(inner, su, sv))
    report["snapshot_quantized_modules"] = sum(name.endswith(".trellis") for name in index)
    report["snapshot_p32_modules"] = sum(name.endswith(".bank_alt_id") for name in index)
    if awq is not None:
        model.model.layers[0].input_layernorm.weight.data.copy_(awq["scaled_block"]["input_layernorm.weight"])

    for arm in ARMS:
        for name in TARGETS:
            weight = decoded_arms[arm][name]
            linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False, device="cuda", dtype=torch.float32)
            linear.weight.data.copy_(weight)
            parent, leaf = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), leaf, linear.requires_grad_(False))
        rows = []
        for i, row in enumerate(data["heldout"]):
            with torch.inference_mode():
                teacher = torch.load(args.output / "teacher" / f"{i}.pt", weights_only=True).cuda()
                logits = model(torch.tensor([row["input_ids"]], device="cuda"), use_cache=False).logits[0]
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite model logits")
                chunks = [logits_metrics(logits[j:j+32], teacher[j:j+32]) for j in range(0, len(logits), 32)]
                metrics = {key: sum(c[key]*c["tokens"] for c in chunks)/len(logits)
                           for key in chunks[0] if key != "tokens"}
                metrics.update(tokens=len(logits), mse=(logits-teacher).square().mean().item())
                rows.append(metrics)
            total = sum(r["tokens"] for r in rows)
            report["model"][arm] = {"rows": rows, "complete": i+1 == len(data["heldout"]),
                                     "mean": {key: sum(r[key]*r["tokens"] for r in rows)/total
                                              for key in rows[0] if key != "tokens"}}
            report["state"] = f"evaluating {arm} {i+1}/{len(data['heldout'])}"
            write_json(args.output / "report.json", report)
            print("MODEL", arm, i+1, json.dumps(report["model"][arm]["mean"]), flush=True)
    report["paired_intervals"] = {}
    for arm in ARMS[1:]:
        report["paired_intervals"][arm] = {}
        for metric in ("kl_teacher_candidate", "mse", "top1_agreement", "top5_agreement", "top10_agreement"):
            values = torch.tensor([a[metric]-b[metric] for a, b in zip(
                report["model"][arm]["rows"], report["model"]["baseline"]["rows"], strict=True)])
            samples = torch.randint(len(values), (2000, len(values)), generator=torch.Generator().manual_seed(7))
            report["paired_intervals"][arm][metric] = {
                "mean_document_delta": values.mean().item(),
                "bootstrap95": values[samples].mean(1).quantile(torch.tensor([.025, .975])).tolist()}
    report["state"] = "complete"
    write_json(args.output / "report.json", report)
    print("COMPLETE", json.dumps({k: v["mean"] for k, v in report["model"].items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--method", choices=("gptq", "rtn", "awq"), required=True)
    parser.add_argument("--awq-calibration", type=Path)
    parser.add_argument("--bits", type=int, choices=(2, 3, 4, 8), default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--dense", type=Path, default=Path("/monster/data/model/Llama-3.2-1B-Instruct"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=16)
    parser.add_argument("--eval-rows", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--candidates", type=int, default=33)
    args = parser.parse_args()
    if args.method == "awq" and (args.awq_calibration is None or args.bits != 4 or args.group_size != 128):
        parser.error("AWQ experiment requires --awq-calibration, W4 and group size 128")
    if args.train_rows < 2 or args.train_rows % 2 or args.eval_rows < 2 or args.tokens < 2:
        parser.error("require positive even training rows and at least two evaluation rows/tokens")
    prepare(args) if args.prepare else execute(args)


if __name__ == "__main__":
    main()
