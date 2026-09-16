#!/usr/bin/env python3
"""Run a disjoint FineWeb-Edu staged-objective smoke test on one P32 projection."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel.quantization.gsq_training import fit_reconstruction_stage
from gptqmodel.quantization.gsq_training_qvq import GSQP32TrainingModule
from gptqmodel.quantization.qvq import (
    repack_p32_planar_to_window,
    rht_preprocess_weight,
)
from gptqmodel.quantization.qvq_gsq import fisher_screened_trellis_candidates


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-model", type=Path, default=Path("/monster/data/model/Llama-3.2-1B-Instruct"))
    parser.add_argument("--qvq-model", type=Path,
                        default=Path("/root/qvq-results/w3-p32-gsq-ab-20260915/gsq"))
    parser.add_argument("--module", default="model.layers.0.mlp.down_proj")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--train-samples", type=int, default=8)
    parser.add_argument("--validation-samples", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=33)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_tensor(checkpoint, key, device):
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(checkpoint / index[key]), framework="pt", device=str(device)) as handle:
        return handle.get_tensor(key)


def fineweb_chunks(tokenizer, *, count, length, seed):
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True,
    ).shuffle(seed=seed, buffer_size=10_000)
    buffer = []
    chunks = []
    iterator = iter(dataset)
    try:
        for row in iterator:
            buffer.extend(tokenizer(row["text"], add_special_tokens=False)["input_ids"])
            while len(buffer) >= length:
                chunks.append(torch.tensor(buffer[:length], dtype=torch.long))
                del buffer[:length]
                if len(chunks) == count:
                    return chunks
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
    raise RuntimeError("FineWeb-Edu stream ended before enough fixed-length chunks were built")


def digest(tokens):
    return hashlib.sha256(tokens.numpy().tobytes()).hexdigest()


def capture_projection_inputs(model, module_name, documents, device):
    module = model.get_submodule(module_name)
    captured = []

    class CaptureComplete(Exception):
        pass

    def hook(_module, args):
        captured.append(args[0].detach().float().cpu())
        raise CaptureComplete

    handle = module.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            for tokens in documents:
                try:
                    model(input_ids=tokens[None].to(device), use_cache=False)
                except CaptureComplete:
                    pass
    finally:
        handle.remove()
    if len(captured) != len(documents):
        raise RuntimeError("projection capture did not produce one activation per document")
    return captured


def main():
    args = parse_args()
    if args.train_samples < 1 or args.validation_samples < 1 or args.sequence_length < 1:
        raise ValueError("training, validation and sequence lengths must be positive")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.dense_model)
    documents = fineweb_chunks(
        tokenizer,
        count=args.train_samples + args.validation_samples,
        length=args.sequence_length,
        seed=args.seed,
    )
    train_documents = documents[:args.train_samples]
    validation_documents = documents[args.train_samples:]
    train_hashes = [digest(tokens) for tokens in train_documents]
    validation_hashes = [digest(tokens) for tokens in validation_documents]
    if set(train_hashes) & set(validation_hashes):
        raise RuntimeError("FineWeb-Edu train and held-out token chunks overlap")

    model = AutoModelForCausalLM.from_pretrained(
        args.dense_model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to(device).eval()
    dense_projection = model.get_submodule(args.module)
    dense_weight = dense_projection.weight.detach().float()
    capture_started = time.perf_counter()
    activations = capture_projection_inputs(model, args.module, documents, device)
    capture_seconds = time.perf_counter() - capture_started
    del model
    torch.cuda.empty_cache()

    names = ("trellis", "SU", "SV", "bank_ids", "bank_alt_id")
    trellis, SU, SV, bank_ids, bank_alt_id = (
        load_tensor(args.qvq_model, f"{args.module}.{name}", device) for name in names
    )
    baseline = repack_p32_planar_to_window(trellis, bits=3)
    target = rht_preprocess_weight(dense_weight, SU.reciprocal(), SV.reciprocal()).float()
    input_metric = torch.eye(target.shape[0], device=device)
    output_metric = torch.eye(target.shape[1], device=device)
    candidates, decoded, indices, deltas, shifts = fisher_screened_trellis_candidates(
        baseline,
        count=args.candidates,
        seed=args.seed,
        bits=3,
        layout="p32_window",
        target=target,
        input_hessian=input_metric,
        output_hessian=output_metric,
        bank_ids=bank_ids,
        bank_alt_id=bank_alt_id,
        return_decoded=True,
        return_sparse=True,
        return_shifts=True,
    )
    del input_metric, output_metric, target
    quantizer = GSQP32TrainingModule(
        candidates,
        decoded[0],
        indices,
        deltas,
        shifts,
        bits=3,
        bank_ids=bank_ids,
        bank_alt_id=bank_alt_id,
        in_features=dense_weight.shape[1],
        out_features=dense_weight.shape[0],
        SU=SU,
        SV=SV,
        seed=args.seed,
    )

    def stage_batches(values):
        return [[(
            value.to(device, non_blocking=True),
            value.shape[0] * value.shape[1] * dense_weight.shape[0],
        )] for value in values]

    train_batches = stage_batches(activations[:args.train_samples])
    validation_batches = stage_batches(activations[args.train_samples:])

    def objective(inputs, weights):
        teacher = torch.nn.functional.linear(inputs, dense_weight)
        student = torch.nn.functional.linear(inputs, weights["weight"])
        return torch.nn.functional.mse_loss(student, teacher)

    result = fit_reconstruction_stage(
        {"weight": quantizer},
        train_batches,
        objective,
        epochs=args.epochs,
        seed=args.seed,
        assignment_lr=1e-4,
        scale_lr=5e-5,
        weight_decay=1.,
        betas=(.9, .95),
        temperature=(2., .05),
        multiplier=(100., 500.),
        min_lr=.1,
        decay="cosine",
        validation_batches=validation_batches,
        restore_best=True,
    )
    state = quantizer.hard_state()
    payload = {
        "module": args.module,
        "device": torch.cuda.get_device_name(),
        "dataset": "HuggingFaceFW/fineweb-edu/sample-10BT",
        "sequence_length": args.sequence_length,
        "train_samples": args.train_samples,
        "validation_samples": args.validation_samples,
        "train_tokens": args.train_samples * args.sequence_length,
        "validation_tokens": args.validation_samples * args.sequence_length,
        "train_hashes": train_hashes,
        "validation_hashes": validation_hashes,
        "strict_disjoint": not bool(set(train_hashes) & set(validation_hashes)),
        "epochs": args.epochs,
        "updates": args.epochs * args.train_samples,
        "candidates": args.candidates,
        "capture_seconds": capture_seconds,
        "fit_seconds": result["elapsed_seconds"],
        "train_hard_loss_before": result["hard_loss_before"],
        "train_hard_loss_after": result["hard_loss_after"],
        "validation_hard_loss_before": result["validation_hard_loss_before"],
        "validation_hard_loss_after": result["validation_hard_loss_after"],
        "best_validation_hard_loss": result["best_validation_hard_loss"],
        "best_validation_epoch": result["best_validation_epoch"],
        "changed_tiles": int((state["choices"] != 0).sum()),
        "scale_max_abs_delta": float((state["SV"] - SV).abs().max()),
        "roundtrip": bool(torch.equal(quantizer.adapter.pack(quantizer.adapter.unpack(state["words"])),
                                      state["words"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
