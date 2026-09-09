"""Full F6/S7 QKV projection refinement and full-model reference propagation.

Preparation verifies historical data/checkpoint bindings before any GPU lease.
Execution uses the exact saved F6 payloads, scales, banks and dense parameters.
Only block-0 Q/K/V payloads change. No new production kernel or lifecycle is used.
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path(
    "/monster/data/model/qvq/"
    "modelcloud-qvq__llama-3.2-1b-instruct__f6-p32__qvq-p32-gguf-exl3__yaqa125x__seed7__"
    "20260904__commit5c5979194dc0__aff65a505e88"
)
TARGETS = [f"model.layers.0.self_attn.{kind}_proj" for kind in ("q", "k", "v")]


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def prepare(args):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.snapshot / "qvq-p32"
    run = json.loads((snapshot / "qvq_quantize_run.json").read_text())
    cfg = json.loads((snapshot / "quantize_config.json").read_text())
    if cfg["yaqa"]["seed"] != 7 or cfg["format"] != "qvq_v2b2_p32" or cfg["incoherence"] != "rht":
        raise ValueError("Expected F6/S7 RHT P32 snapshot")
    manifest = json.loads((args.snapshot / "snapshot_manifest.json").read_text())
    if manifest["model"]["experiment"] != "f6-p32" or manifest["model"]["seed"] != 7:
        raise ValueError("Snapshot is not F6/S7")
    calibration = args.snapshot / "calibration/source/yaqa182-nm10000.parquet"
    evaluation = ROOT / "dataset/divergence300-v1/divergence300-locked.jsonl"
    audit = ROOT / "dataset/calibration-fisher-scaling-v2/yaqa182_nm10000.disjointness.json"
    bindings = run["disjointness_manifest"]
    expected = {
        str(calibration): run["datasets"]["yaqa"]["content_sha256"],
        str(evaluation): bindings["evaluation_bindings"]["d300_locked"]["sha256"],
        str(audit): bindings["sha256"],
    }
    for path, sha in expected.items():
        if digest(path) != sha:
            raise ValueError(f"Historical source hash mismatch: {path}")
    if json.loads(audit.read_text())["status"] != "pass":
        raise ValueError("Historical disjointness audit failed")
    # Bind every shard, plus dense source and tokenizer, to this run.
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    file_hashes = {str(snapshot / p): digest(snapshot / p) for p in sorted(set(index.values()))}
    for path in (snapshot / "quantize_config.json", snapshot / "model.safetensors.index.json",
                 args.dense / "model.safetensors", args.dense / "config.json",
                 args.dense / "tokenizer.json", args.dense / "tokenizer_config.json"):
        file_hashes[str(path)] = digest(path)
    tokenizer = AutoTokenizer.from_pretrained(args.dense, local_files_only=True)
    rows = pq.read_table(calibration).to_pylist()
    rng = random.Random(7)
    selected = []
    for source in ("yaqa", "nm"):
        eligible = [i for i, row in enumerate(rows) if row["source_name"] == source]
        selected.extend(rng.sample(eligible, args.train_rows // 2))
    eval_rows = [json.loads(line) for line in evaluation.read_text().splitlines()]
    eval_indices = sorted(random.Random(7).sample(range(len(eval_rows)), args.eval_rows))

    def encode(row, index, split):
        ids = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False,
                                            return_dict=False)[:args.tokens]
        if len(ids) < 2:
            raise ValueError("Insufficient tokens")
        return {"source_row": index, "split": split, "source_name": row.get("source_name"),
                "input_ids": ids, "token_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}

    train = [encode(rows[i], i, "train") for i in selected]
    heldout = [encode(eval_rows[i], i, "heldout") for i in eval_indices]
    if {r["token_sha256"] for r in train} & {r["token_sha256"] for r in heldout}:
        raise ValueError("Tokenized refinement/evaluation overlap")
    if len({r["token_sha256"] for r in train + heldout}) != len(train + heldout):
        raise ValueError("Duplicate tokenized documents")
    write_json(args.output / "inputs.json", {"train": train, "heldout": heldout})
    write_json(args.output / "quantize_config.json", cfg)
    write_json(args.output / "provenance.json", {
        "snapshot": str(args.snapshot), "dense": str(args.dense), "historical_run_commit": run["commit"],
        "repository_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_hashes": expected, "file_hashes": file_hashes, "historical_disjointness_verified": True,
        "refinement": "16 stratified original-corpus documents by default; not a repeat of all 10178 YAQA rows",
        "seed": 7, "train_rows": len(train), "eval_rows": len(heldout), "token_cap": args.tokens,
        "targets": TARGETS, "candidate_count": args.candidates, "steps": args.steps,
        "train_tokens": sum(len(r["input_ids"]) for r in train),
        "heldout_tokens": sum(len(r["input_ids"]) for r in heldout),
        "train_source_weights": dict(cfg["yaqa"]["source_weights"]),
        "source_code": {str(p): digest(p) for p in
                        (Path(__file__), ROOT / "gptqmodel/quantization/qvq_gsq.py")},
        "scope": "full QKV projections; full-model propagation; FP32 canonical reference backend",
    })
    print("PREPARED", args.output, flush=True)


def execute(args):
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

    from gptqmodel.quantization.qvq import (
        decode_p32_window_tiles,
        reconstruct_p32_window_inner_weight,
        reconstruct_qvq_inner_weight,
        repack_p32_planar_to_window,
        repack_p32_window_to_planar,
        rht_preprocess_weight,
    )
    from gptqmodel.quantization.qvq_gsq import refine_p32_candidates
    from gptqmodel.quantization.rotation.hadamard_utils import matmul_hadU
    from scripts.p32_twenty.scorecard import logits_metrics

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    data = json.loads((args.output / "inputs.json").read_text())
    provenance = json.loads((args.output / "provenance.json").read_text())
    if str(args.snapshot) != provenance["snapshot"] or str(args.dense) != provenance["dense"]:
        raise ValueError("Execution model paths differ from prepared contract")
    for path, sha in (provenance["file_hashes"] | provenance["source_hashes"]).items():
        if digest(path) != sha:
            raise ValueError(f"Prepared input changed before execution: {path}")
    if provenance["candidate_count"] != args.candidates or provenance["steps"] != args.steps:
        raise ValueError("Execution parameters differ from prepared contract")
    cfg = json.loads((args.output / "quantize_config.json").read_text())
    snapshot = args.snapshot / "qvq-p32"
    if cfg != json.loads((snapshot / "quantize_config.json").read_text()):
        raise ValueError("Prepared quantization config differs from snapshot")
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    report = {"provenance": provenance, "inventory": inventory, "torch": torch.__version__,
              "cuda": torch.version.cuda, "gpu": str(torch.cuda.get_device_properties(0)),
              "backend": "FP32 canonical QVQ RHT/window reference; no native-kernel performance claim",
              "layers": {}, "model": {}, "state": "dense reference"}
    write_json(args.output / "report.json", report)

    def read(name):
        with safe_open(str(snapshot / index[name]), framework="pt") as handle:
            return handle.get_tensor(name).cuda()

    def local_metrics(a, b):
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("Nonfinite layer outputs")
        mse = (a.float() - b.float()).square().mean().item()
        return {"mse": mse, "nmse": mse / b.float().square().mean().item()}

    model = AutoModelForCausalLM.from_pretrained(
        args.dense, dtype=torch.float32, device_map={"": "cuda:0"},
        attn_implementation="eager", local_files_only=True).eval()
    model.requires_grad_(False)
    dense_weights = {name: model.get_submodule(name).weight.detach().clone() for name in TARGETS}
    captures = {name: {"train": [], "heldout": []} for name in TARGETS}
    split = "train"
    hooks = []
    for name in TARGETS:
        def capture(module, values, name=name):
            captures[name][split].append(values[0].detach().reshape(-1, values[0].shape[-1]).clone())
        hooks.append(model.get_submodule(name).register_forward_pre_hook(capture))
    (args.output / "teacher").mkdir(exist_ok=False)
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
                print("DENSE", split, i + 1, "/", len(data[split]), flush=True)
    for hook in hooks:
        hook.remove()
    # Snapshot-owned endpoints/norms must be copied, not silently left as the source model's.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in index:
                param.copy_(read(name))

    class CanonicalLinear(torch.nn.Module):
        def __init__(self, inner, su, sv):
            super().__init__()
            self.register_buffer("inner", inner.float())
            self.register_buffer("su", su.float())
            self.register_buffer("sv", sv.float())

        def forward(self, x):
            return matmul_hadU(matmul_hadU(x.float() * self.su) @ self.inner) * self.sv

    if cfg.get("activation") or cfg.get("incoherence") != "rht":
        raise ValueError("Unsupported snapshot transform/activation contract")
    saved = {}
    for name in sorted(index):
        if not name.endswith(".trellis"):
            continue
        prefix = name[:-8]
        trellis, su, sv = [read(prefix + "." + suffix) for suffix in ("trellis", "SU", "SV")]
        p32 = prefix + ".bank_alt_id" in index
        bank = read(prefix + ".bank_ids") if prefix + ".bank_ids" in index else None
        alt = read(prefix + ".bank_alt_id") if p32 else None
        bits = trellis.shape[-1] / 8
        kw = dict(bits=bits, in_features=su.numel(), out_features=sv.numel(),
                  bank_ids=bank, bank_alt_id=alt, codebook_version=cfg["codebook"])
        inner = reconstruct_qvq_inner_weight(trellis, v2b2_p32=p32, **kw)
        if prefix in TARGETS:
            if not p32:
                raise ValueError("Expected target P32 projection")
            window = repack_p32_planar_to_window(trellis, bits=bits)
            if not torch.equal(reconstruct_p32_window_inner_weight(window, **kw), inner):
                raise ValueError("Baseline planar/window decode differs")
            saved[prefix] = {"base": window, "kw": kw, "su": su, "sv": sv}
        parent, leaf = prefix.rsplit(".", 1)
        setattr(model.get_submodule(parent), leaf, CanonicalLinear(inner, su, sv))
        print("F6_RECONSTRUCTED", prefix, flush=True)
    report["snapshot_quantized_modules"] = sum(name.endswith(".trellis") for name in index)
    report["snapshot_p32_modules"] = sum(name.endswith(".bank_alt_id") for name in index)

    def decode_tiles(words, kw):
        return decode_p32_window_tiles(words, **{key: kw[key] for key in
                                                ("bits", "bank_ids", "bank_alt_id", "codebook_version")})

    def greedy_fit(candidates, x, target, kw):
        # Exact hard coordinate descent: update disjoint output blocks together,
        # sweep full input blocks sequentially, maintaining full output residual.
        k, n = target.shape
        decoded = torch.stack([decode_tiles(c, kw) for c in candidates]).reshape(
            args.candidates, k // 16, n // 16, 16, 16)
        current = decoded[0].clone()
        residual = x @ (current.permute(0, 2, 1, 3).reshape(k, n) - target)
        choices = torch.zeros(k // 16, n // 16, dtype=torch.long, device=x.device)
        for sweep in range(3):
            for ib in range(k // 16):
                delta = decoded[:, ib] - current[ib].unsqueeze(0)
                outputs = torch.einsum("ti,cnij->cntj", x[:, ib * 16:(ib + 1) * 16], delta)
                old = residual.reshape(x.shape[0], n // 16, 16).permute(1, 0, 2)
                costs = (outputs + old.unsqueeze(0)).square().sum((2, 3))
                best = costs.argmin(0)
                ids = torch.arange(n // 16, device=x.device)
                improved = costs[best, ids] < old.square().sum((1, 2))
                chosen = torch.where(improved, best, choices[ib])
                update = outputs[chosen, ids]
                residual += update.permute(1, 0, 2).reshape_as(residual)
                current[ib] = decoded[chosen, ib, ids]
                choices[ib] = chosen
            print("GREEDY", sweep + 1, float(residual.square().mean()), flush=True)
        return candidates[choices.flatten(), torch.arange(k * n // 256, device=x.device)].contiguous()

    def evaluate(arm):
        rows = []
        for i, row in enumerate(data["heldout"]):
            with torch.inference_mode():
                teacher = torch.load(args.output / "teacher" / f"{i}.pt", weights_only=True).cuda()
                logits = model(torch.tensor([row["input_ids"]], device="cuda"), use_cache=False).logits[0]
                chunks = [logits_metrics(logits[j:j + 32], teacher[j:j + 32]) for j in range(0, len(logits), 32)]
                metrics = {key: sum(c[key] * c["tokens"] for c in chunks) / len(logits)
                           for key in chunks[0] if key != "tokens"}
                metrics.update(local_metrics(logits, teacher))
                metrics["tokens"] = len(logits)
                rows.append(metrics)
                del logits, teacher
            report["model"][arm] = {"rows": rows, "complete": False}
            report["state"] = f"{arm} evaluation {i + 1}/{len(data['heldout'])}"
            write_json(args.output / "report.json", report)
            print("MODEL", arm, i + 1, json.dumps(metrics), flush=True)
        total = sum(r["tokens"] for r in rows)
        report["model"][arm] = {"rows": rows, "complete": True,
                                 "mean": {key: sum(r[key] * r["tokens"] for r in rows) / total
                                          for key in rows[0] if key != "tokens"}, "tokens": total}
        write_json(args.output / "report.json", report)

    evaluate("f6_seed7")
    for name in TARGETS:
        state = saved[name]
        kw, base = state["kw"], state["base"]
        su, sv = state["su"].float(), state["sv"].float()
        # With constant |SV|, orthogonality makes inner and full-output NMSE
        # identical. Reject anything else rather than silently changing loss.
        if not torch.allclose(sv.abs(), sv.abs()[0].expand_as(sv), rtol=1e-6, atol=0):
            raise ValueError("Nonuniform output scales require a full-output training objective")
        target = rht_preprocess_weight(dense_weights[name], su.reciprocal(), sv.reciprocal())
        sources = provenance["train_source_weights"]
        x = torch.cat([matmul_hadU(a * su) * sources[row["source_name"]] ** 0.5
                       for a, row in zip(captures[name]["train"], data["train"], strict=True)]).detach()
        candidates = base.unsqueeze(0).repeat(args.candidates, 1, 1)
        gen = torch.Generator(device="cuda").manual_seed(7)
        tiles = torch.arange(len(base), device="cuda")
        for c in range(1, args.candidates):
            bit = torch.randint(base.shape[1] * 32, (len(base),), device="cuda", generator=gen)
            candidates[c, tiles, bit // 32] ^= (torch.ones_like(bit) << (bit % 32)).to(torch.int32)
        report["state"] = "refining " + name
        write_json(args.output / "report.json", report)
        def progress(step, value):
            if step % 10 == 0:
                print("GSQ", name, step, value, flush=True)
        result = refine_p32_candidates(
            candidates, bits=kw["bits"], bank_ids=kw["bank_ids"], bank_alt_id=kw["bank_alt_id"],
            codebook_version=kw["codebook_version"], target=target, inputs=x,
            steps=args.steps, seed=7, progress=progress)
        state["gsq"] = result.window_words
        state["greedy"] = greedy_fit(candidates, x, target, kw)
        layer = {"shape_out_in": list(dense_weights[name].shape), "bits": kw["bits"],
                 "calibration_before": result.calibration_before, "calibration_after": result.calibration_after,
                 "payload_bytes": base.numel() * base.element_size(), "heldout": {}}
        for arm in ("base", "gsq", "greedy"):
            words = state[arm]
            if not torch.equal(words, repack_p32_planar_to_window(
                repack_p32_window_to_planar(words, bits=kw["bits"]), bits=kw["bits"]
            )):
                raise ValueError("Candidate window roundtrip failed")
            restored = reconstruct_p32_window_inner_weight(words, **kw)
            operator = CanonicalLinear(restored, su, sv)
            layer["heldout"][arm] = [local_metrics(operator(a), a @ dense_weights[name].T)
                                        for a in captures[name]["heldout"]]
        export = {key: state[key].cpu() for key in ("base", "gsq", "greedy", "su", "sv")}
        export.update(bank_ids=kw["bank_ids"].cpu(), bank_alt_id=kw["bank_alt_id"].cpu())
        path = args.output / (name + ".pt")
        torch.save(export, path)
        restored = torch.load(path, weights_only=True)
        for arm in ("base", "gsq", "greedy"):
            if not torch.equal(restored[arm].cuda(), state[arm]):
                raise ValueError("Payload reload differs")
            state[arm] = restored[arm].cuda()
        layer["export_sha256"] = digest(path)
        layer["roundtrip_and_reload_exact"] = True
        report["layers"][name] = layer
        write_json(args.output / "report.json", report)
        print("LAYER_COMPLETE", name, json.dumps(layer), flush=True)
        del candidates, x, target, result
    for arm in ("gsq", "greedy"):
        for name in TARGETS:
            module = model.get_submodule(name)
            module.inner = reconstruct_p32_window_inner_weight(saved[name][arm], **saved[name]["kw"])
        evaluate(arm)
    report["paired_intervals"] = {}
    for arm in ("gsq", "greedy"):
        report["paired_intervals"][arm] = {}
        for key in ("kl_teacher_candidate", "mse", "nmse", "top1_agreement", "top5_agreement", "top10_agreement"):
            delta = torch.tensor([a[key] - b[key] for a, b in zip(
                report["model"][arm]["rows"], report["model"]["f6_seed7"]["rows"], strict=True)])
            boot = torch.randint(len(delta), (2000, len(delta)), generator=torch.Generator().manual_seed(7))
            interval = delta[boot].mean(1).quantile(torch.tensor([0.025, 0.975])).tolist()
            positive = interval[1] < 0 if key in ("kl_teacher_candidate", "mse", "nmse") else interval[0] > 0
            negative = interval[0] > 0 if key in ("kl_teacher_candidate", "mse", "nmse") else interval[1] < 0
            report["paired_intervals"][arm][key] = {
                "mean_document_delta": delta.mean().item(), "bootstrap95": interval,
                "classification": "clear positive" if positive else "clear negative" if negative else "noise-consistent"}
    report["state"] = "complete"
    write_json(args.output / "report.json", report)
    print("COMPLETE", json.dumps({k: v["mean"] for k, v in report["model"].items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--dense", type=Path, default=Path("/monster/data/model/Llama-3.2-1B-Instruct"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=16)
    parser.add_argument("--eval-rows", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--candidates", type=int, default=33)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.train_rows < 2 or args.train_rows % 2 or args.eval_rows < 2 or args.tokens < 2:
        parser.error("Use positive even train rows, >=2 eval rows and >=2 tokens")
    if args.candidates < 2 or args.steps < 1:
        parser.error("Use >=2 candidates and positive steps")
    if args.output.resolve().is_relative_to(args.snapshot.resolve()):
        parser.error("Keep partial experiment outputs outside the snapshot")
    if not args.prepare and (args.output / "report.json").exists():
        parser.error("Refusing to overwrite an existing run; prepare a new output directory")
    prepare(args) if args.prepare else execute(args)


if __name__ == "__main__":
    main()
