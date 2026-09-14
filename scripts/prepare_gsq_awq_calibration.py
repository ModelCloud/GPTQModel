"""Prepare real block-0 AWQ QKV scale search for the F6/seed7 GSQ comparison.

This produces a selected-layer calibration fixture, not a complete model.
"""

import argparse
import json
import os
import shutil
import subprocess
import time
import types
from pathlib import Path

from scripts.validate_qvq_gsq_layers import ROOT, digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dense", type=Path, default=Path("/monster/data/model/Llama-3.2-1B-Instruct"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory")
    provenance = json.loads((args.source / "provenance.json").read_text())
    token_path = args.source / "inputs.json"
    if digest(token_path) != provenance["inputs_sha256"]:
        raise ValueError("Source calibration tokens changed")
    if str(args.dense) != provenance["dense"]:
        raise ValueError("Dense source differs from audited F6 source")
    for path, expected in provenance["source_hashes"].items():
        if digest(path) != expected:
            raise ValueError(f"Audited model source changed: {path}")
    dense_hashes = {p: h for p, h in provenance["file_hashes"].items()
                    if Path(p).is_relative_to(args.dense)}
    if not dense_hashes:
        raise ValueError("Missing audited dense model hashes")
    for path, expected in dense_hashes.items():
        if digest(path) != expected:
            raise ValueError(f"Audited dense file changed: {path}")
    uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not uuid.startswith("GPU-") or "," in uuid or not os.environ.get("GPU_ALLOCATOR_LEASE_ID"):
        raise ValueError("Run under one exclusive UUID GPU allocator lease")
    for sample in range(3):
        inventory = subprocess.check_output([
            "nvidia-smi", "--id=" + uuid,
            "--query-gpu=index,pci.bus_id,uuid,name,memory.used,memory.total,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits"], text=True).strip()
        fields = [v.strip() for v in inventory.split(",")]
        processes = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
        if fields[2] != uuid or int(fields[4]) > 8 or int(fields[6]) != 0 or uuid in processes:
            raise RuntimeError("Idle GPU preflight failed: " + inventory)
        print("IDLE", sample + 1, inventory, flush=True)
        time.sleep(1)

    import torch
    from transformers import AutoModelForCausalLM

    from gptqmodel.looper.awq_processor import AWQProcessor
    from gptqmodel.quantization import AWQConfig
    from gptqmodel.quantization.awq.quantize.scale import apply_scale

    torch.set_num_threads(4)
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True)
    shutil.copy2(token_path, args.output / "inputs.json")
    shutil.copy2(args.source / "provenance.json", args.output / "source_provenance.json")
    sources = [Path(__file__), ROOT / "gptqmodel/looper/awq_processor.py",
               ROOT / "gptqmodel/quantization/awq/quantize/scale.py", ROOT / "gptqmodel/quantization/config.py"]
    for path in sources:
        dest = args.output / "executed-source" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    report = {"state": "loading", "source_hashes": {str(p): digest(p) for p in sources},
              "inventory": inventory, "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": str(torch.cuda.get_device_properties(0)), "seed": 7,
              "scope": "block-0 attention QKV scale search; calibration only; no GSQ quality conclusion",
              "dense_hashes": dense_hashes,
              "weighting": "uniform valid calibration tokens; no YAQA source weights",
              "inputs_sha256": digest(token_path)}
    write_json(args.output / "report.json", report)
    model = AutoModelForCausalLM.from_pretrained(args.dense, dtype=torch.float16,
                                                device_map={"": "cuda:0"}, attn_implementation="eager",
                                                local_files_only=True).eval().requires_grad_(False)
    block = model.model.layers[0]
    names = ["self_attn." + p + "_proj" for p in ("q", "k", "v")]
    linears = [block.get_submodule(n) for n in names]
    original = {n: p.detach().cpu().clone() for n, p in block.state_dict().items()}
    rows = json.loads(token_path.read_text())
    features = {}
    with torch.inference_mode():
        for split in ("train", "heldout"):
            features[split] = [block.input_layernorm(model.model.embed_tokens(
                torch.tensor([row["input_ids"]], device="cuda"))).detach() for row in rows[split]]
        x = torch.cat(features["train"], dim=1)
        lengths = [v.shape[1] for v in features["train"]]
        positions = torch.cat([torch.arange(n, device="cuda") for n in lengths])[None]
        mask = torch.full((1, 1, x.shape[1], x.shape[1]), torch.finfo(x.dtype).min,
                          device="cuda", dtype=x.dtype)
        start = 0
        separate = []
        for value, length in zip(features["train"], lengths, strict=True):
            causal = torch.full((length, length), torch.finfo(x.dtype).min,
                                device="cuda", dtype=x.dtype).triu(1)
            mask[0, 0, start:start+length, start:start+length] = causal
            pos = torch.arange(length, device="cuda")[None]
            separate.append(block.self_attn(value, attention_mask=causal[None, None],
                            position_embeddings=model.model.rotary_emb(value, pos))[0])
            start += length
        kwargs = {"attention_mask": mask, "position_embeddings": model.model.rotary_emb(x, positions)}
        joined = block.self_attn(x, **kwargs)[0]
        error = (joined.float() - torch.cat(separate, dim=1).float()).abs()
        report["document_isolation"] = {"mean": error.mean().item(), "max": error.max().item()}
        if not torch.isfinite(error).all() or error.mean() > .002 or error.max() > .046875:
            raise ValueError("Joined attention failed separate-document reference")
    cfg = AWQConfig(bits=4, group_size=128, sym=False)
    processor = AWQProcessor(tokenizer=None, qcfg=cfg, calibration=None, prepare_dataset_func=None,
                             calibration_concat_size=None, calibration_sort=None, batch_size=1,
                             gptq_model=types.SimpleNamespace(rotary_embedding=None), model=model)
    report.update(state="searching", config=cfg.to_dict(), documents=len(lengths), tokens=sum(lengths))
    write_json(args.output / "report.json", report)
    print("SEARCH", len(lengths), sum(lengths), flush=True)
    scales = processor._search_best_scale(block, block.input_layernorm, linears, x,
                                          module2inspect=block.self_attn, kwargs=kwargs)
    if any(not torch.equal(value, block.state_dict()[name].cpu()) for name, value in original.items()):
        raise ValueError("Scale search did not restore original block weights")
    apply_scale(block, [scales])
    with torch.inference_mode():
        transformed = x / scales[2].to(x.device)
        runtime_features = {split: [block.input_layernorm(model.model.embed_tokens(
            torch.tensor([row["input_ids"]], device="cuda"))).detach() for row in rows[split]]
            for split in ("train", "heldout")}
        runtime_x = torch.cat(runtime_features["train"], dim=1)
        input_error = (runtime_x.float()-transformed.float()).abs()
        report["scaled_input_rounding"] = {"mean": input_error.mean().item(), "max": input_error.max().item()}
        after = block.self_attn(runtime_x, **kwargs)[0]
        error = (after.float()-joined.float()).abs()
        report["folding_error"] = {"mean": error.mean().item(), "max": error.max().item()}
        if not torch.isfinite(error).all() or error.mean() > .002 or error.max() > .046875:
            raise ValueError("Scale folding failed original attention reference")
    torch.save({"original_block": original, "scaled_block": {n: p.cpu() for n, p in block.state_dict().items()},
                "runtime_scaled_features": {s: [v.cpu() for v in values] for s, values in runtime_features.items()},
                "scale": scales[2].cpu(), "features": {s: [v.cpu() for v in values]
                for s, values in features.items()}}, args.output / "calibration.pt")
    report.update(state="complete", scale_search_loss=float(scales[3]),
                  scale_min=float(scales[2].min()), scale_max=float(scales[2].max()),
                  calibration_sha256=digest(args.output / "calibration.pt"))
    write_json(args.output / "report.json", report)
    print("COMPLETE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
