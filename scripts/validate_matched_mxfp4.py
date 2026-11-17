# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Bounded CPU model qualification for the matched solver/format public API."""

import argparse
import gc
import hashlib
import json
import math
import resource
import time
from pathlib import Path

import torch
from datasets import load_dataset

from gptqmodel import GPTQModel, MatchedQuantizeConfig, Matcher, formats, solvers
from gptqmodel.nn_modules.qlinear.mxfp4 import MXFP4Linear


def token_rows(tokenizer, split, count, length):
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    tokens = []
    for text in dataset["text"]:
        if text.strip():
            tokens.extend(tokenizer(text + "\n", add_special_tokens=False)["input_ids"])
        if len(tokens) >= count * length:
            break
    return [{"input_ids": torch.tensor([tokens[index * length:(index + 1) * length]])}
            for index in range(count)]


@torch.inference_mode()
def evaluate(model, rows):
    loss, tokens, predictions, hashes = 0., 0, [], []
    for batch in rows:
        ids = batch["input_ids"]
        logits = model.model(**batch, use_cache=False).logits.float()
        assert torch.isfinite(logits).all()
        count = ids.numel() - len(ids)
        loss += float(torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum"))
        tokens += count
        predictions.append(logits.argmax(-1))
        hashes.append(hashlib.sha256(logits.numpy().tobytes()).hexdigest())
        del logits
    return {"nll": loss / tokens, "perplexity": math.exp(loss / tokens), "tokens": tokens,
            "logit_sha256": hashes}, torch.cat(predictions)


def payload_hashes(model):
    return {name: hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for name, tensor in model.model.state_dict().items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-samples", type=int, default=16)
    parser.add_argument("--evaluation-samples", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(11)
    model = GPTQModel.load(args.model, device="cpu", dtype=torch.float32, attn_implementation="eager")
    calibration = token_rows(model.tokenizer, "train", args.calibration_samples, args.sequence_length)
    heldout = token_rows(model.tokenizer, "test", args.evaluation_samples, args.sequence_length)
    baseline, reference_predictions = evaluate(model, heldout)
    print("Dense baseline:", baseline, flush=True)
    targets = [MatchedQuantizeConfig(matcher=Matcher.glob("model.layers.*.*.*"),
                                    solver=solvers.GPTQ(), format=formats.MXFP4())]
    dense_bytes = sum(t.numel() * t.element_size() for t in model.model.parameters())
    started = time.perf_counter()
    records = model.quantize(calibration, config=targets)
    elapsed = time.perf_counter() - started
    quantized, predictions = evaluate(model, heldout)
    expected_payloads = payload_hashes(model)
    packed = [module for module in model.model.modules() if isinstance(module, MXFP4Linear)]
    assert len(packed) == len(records) == model.model.config.num_hidden_layers * 7
    packed_bytes = sum(module.qweight.numel() + module.scales.numel() for module in packed)
    assert all(not hasattr(module, "weight") for module in packed)
    prompt = model.tokenizer("The capital of France is", return_tensors="pt")
    with torch.inference_mode():
        generated = model.generate(**prompt, do_sample=False, max_new_tokens=16)
    generation = model.tokenizer.decode(generated[0], skip_special_tokens=True)
    checkpoint = args.output / "checkpoint"
    model.save(str(checkpoint), max_shard_size="500MB")
    del model, packed
    gc.collect()
    restored = GPTQModel.load(str(checkpoint), device="cpu")
    reload_metrics, _ = evaluate(restored, heldout)
    assert reload_metrics["logit_sha256"] == quantized["logit_sha256"]
    assert payload_hashes(restored) == expected_payloads
    with torch.inference_mode():
        regenerated = restored.generate(**prompt, do_sample=False, max_new_tokens=16)
    assert torch.equal(regenerated, generated)
    second_checkpoint = args.output / "checkpoint-resaved"
    restored.save(str(second_checkpoint), max_shard_size="500MB")
    del restored
    gc.collect()
    restored = GPTQModel.load(str(second_checkpoint), device="cpu")
    resaved_metrics, _ = evaluate(restored, heldout)
    assert resaved_metrics["logit_sha256"] == quantized["logit_sha256"]
    assert payload_hashes(restored) == expected_payloads
    report = {
        "model": args.model, "torch": torch.__version__, "device": "cpu", "threads": args.threads,
        "calibration_dataset": "Salesforce/wikitext/wikitext-2-raw-v1/train",
        "calibration_tokens": args.calibration_samples * args.sequence_length,
        "evaluation_dataset": "Salesforce/wikitext/wikitext-2-raw-v1/test",
        "dense": baseline, "gptq_mxfp4": quantized,
        "argmax_agreement": float((predictions == reference_predictions).float().mean()),
        "target_modules": len(records), "quantization_seconds": elapsed,
        "dense_parameter_bytes": dense_bytes, "packed_target_bytes": packed_bytes,
        "checkpoint_bytes": sum(path.stat().st_size for path in checkpoint.glob("*.safetensors")),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "reload_payloads_exact": True, "reload_logits_exact": True, "reload_generation_exact": True,
        "resave_payloads_exact": True, "resave_logits_exact": True,
        "generation": generation, "modules": records,
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "modules"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
