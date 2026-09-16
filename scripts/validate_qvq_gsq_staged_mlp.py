#!/usr/bin/env python3
"""Train one joint Llama MLP stage from legal W3/P32 QVQ payloads."""

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel.looper.gsq_training_capture import (
    capture_llama_gsq_inputs,
    prepare_llama_gsq_capture,
)
from gptqmodel.quantization.gsq_training import (
    fit_reconstruction_stage,
    reconstruction_stage_loss,
)
from gptqmodel.quantization.gsq_training_qvq import (
    p32_payload_weight,
    p32_training_module_from_payload,
    p32_training_module_from_words,
)
from gptqmodel.quantization.qvq import (
    repack_p32_planar_to_window,
    repack_p32_window_to_planar,
)
from scripts.validate_qvq_gsq_staged_projection import (
    digest,
    fineweb_chunks,
    load_tensor,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-model", type=Path, default=Path("/monster/data/model/Llama-3.2-1B-Instruct"))
    parser.add_argument("--qvq-model", type=Path,
                        default=Path("/root/qvq-results/w3-p32-gsq-ab-20260915/gsq"))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=2)
    parser.add_argument("--validation-samples", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--disable-fast-training-hadamard",
        action="store_true",
        help="Use the eager FP32 Hadamard oracle during GSQ optimization",
    )
    parser.add_argument("--token-cache", type=Path)
    parser.add_argument("--state-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def projection_payload(checkpoint, name, device):
    keys = ("trellis", "SU", "SV", "bank_ids", "bank_alt_id")
    return tuple(load_tensor(checkpoint, f"{name}.{key}", device) for key in keys)


def main():
    args = parse_args()
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.dense_model)
    document_count = args.train_samples + args.validation_samples
    if args.token_cache is not None and args.token_cache.is_file():
        cached = load_file(args.token_cache)
        if "tokens" in cached:
            documents = [row.long() for row in cached["tokens"][:document_count]]
        else:
            documents = [cached[f"tokens.{index:05d}"] for index in range(document_count)]
        if any(tokens.shape != (args.sequence_length,) for tokens in documents):
            raise ValueError("cached FineWeb-Edu token chunks have the wrong sequence length")
    else:
        documents = fineweb_chunks(
            tokenizer,
            count=document_count,
            length=args.sequence_length,
            seed=args.seed,
        )
        if args.token_cache is not None:
            args.token_cache.parent.mkdir(parents=True, exist_ok=True)
            save_file(
                {f"tokens.{index:05d}": tokens for index, tokens in enumerate(documents)},
                args.token_cache,
                metadata={
                    "dataset": "HuggingFaceFW/fineweb-edu/sample-10BT",
                    "seed": str(args.seed),
                    "sequence_length": str(args.sequence_length),
                },
            )
    train_hashes = [digest(tokens) for tokens in documents[:args.train_samples]]
    validation_hashes = [digest(tokens) for tokens in documents[args.train_samples:]]
    if set(train_hashes) & set(validation_hashes):
        raise RuntimeError("FineWeb-Edu train and held-out token chunks overlap")

    model = AutoModelForCausalLM.from_pretrained(
        args.dense_model, dtype=torch.bfloat16, low_cpu_mem_usage=True,
    ).to(device).eval()
    layer_prefix = f"model.layers.{args.layer}"
    layer = model.model.layers[args.layer]

    # The MLP teacher and student share the already-quantized attention, which
    # is the paper's sequential stage boundary.
    for leaf in ("q_proj", "k_proj", "v_proj", "o_proj"):
        name = f"{layer_prefix}.self_attn.{leaf}"
        projection = layer.self_attn.get_submodule(leaf)
        trellis, SU, SV, bank_ids, bank_alt_id = projection_payload(args.qvq_model, name, device)
        quantized = p32_payload_weight(
            trellis, SU, SV, bank_ids, bank_alt_id,
            in_features=projection.in_features,
            out_features=projection.out_features,
        )
        with torch.no_grad():
            projection.weight.copy_(quantized.to(projection.weight.dtype))

    capture_started = time.perf_counter()
    source_documents = [{"input_ids": tokens.tolist()} for tokens in documents]
    capture = capture_llama_gsq_inputs(model, source_documents, layer_index=args.layer)
    prepared, batches = prepare_llama_gsq_capture(layer, capture, device=device)
    capture_seconds = time.perf_counter() - capture_started
    del model, capture
    torch.cuda.empty_cache()

    quantizers = {}
    projection_constants = {}
    initial_words = {}
    initial_scales = {}
    candidate_seconds = 0.
    training_hadamard_backends = set()
    for leaf in ("gate_proj", "up_proj", "down_proj"):
        short_name = f"mlp.{leaf}"
        full_name = f"{layer_prefix}.{short_name}"
        projection = prepared.get_submodule(short_name)
        trellis, SU, SV, bank_ids, bank_alt_id = projection_payload(args.qvq_model, full_name, device)
        key = f"{short_name}.weight"
        projection_constants[key] = (
            SU, bank_ids, bank_alt_id, projection.weight.detach().float(),
        )
        initial_words[key] = repack_p32_planar_to_window(trellis, bits=3)
        initial_scales[key] = SV.detach().clone()
        started = time.perf_counter()
        quantizers[key] = p32_training_module_from_payload(
            trellis,
            SU,
            SV,
            bank_ids,
            bank_alt_id,
            projection.weight.detach().float(),
            candidates=args.candidates,
            seed=args.seed,
            fast_hadamard=not args.disable_fast_training_hadamard,
        )
        training_hadamard_backends.add(
            quantizers[key].training_hadamard_backend
        )
        candidate_seconds += time.perf_counter() - started

    def grouped(values):
        return [[((hidden, kwargs, None), hidden.numel())] for hidden, kwargs in values]

    train_batches = grouped(batches[:args.train_samples])
    validation_batches = grouped(batches[args.train_samples:])

    def objective(batch, weights):
        hidden, kwargs, mask = batch
        return reconstruction_stage_loss(
            prepared,
            (hidden,),
            kwargs,
            student_weights=weights,
            output_mask=mask,
        )

    round_results = []
    fit_seconds = 0.
    for round_index in range(args.rounds):
        result = fit_reconstruction_stage(
            quantizers,
            train_batches,
            objective,
            epochs=args.epochs,
            seed=args.seed + round_index,
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
        fit_seconds += result["elapsed_seconds"]
        states = {name: quantizer.hard_state() for name, quantizer in quantizers.items()}
        round_results.append({
            "round": round_index + 1,
            "train_hard_loss_before": result["hard_loss_before"],
            "train_hard_loss_after": result["hard_loss_after"],
            "validation_hard_loss_before": result["validation_hard_loss_before"],
            "validation_hard_loss_after": result["validation_hard_loss_after"],
            "best_validation_hard_loss": result["best_validation_hard_loss"],
            "best_validation_epoch": result["best_validation_epoch"],
            "changed_tiles_this_round": {
                name: int((state["choices"] != 0).sum()) for name, state in states.items()
            },
        })
        if round_index + 1 == args.rounds:
            break
        next_quantizers = {}
        for name, state in states.items():
            SU, bank_ids, bank_alt_id, teacher_weight = projection_constants[name]
            started = time.perf_counter()
            next_quantizers[name] = p32_training_module_from_words(
                state["words"],
                SU,
                state["SV"],
                bank_ids,
                bank_alt_id,
                teacher_weight,
                candidates=args.candidates,
                seed=args.seed + round_index + 1,
                fast_hadamard=not args.disable_fast_training_hadamard,
            )
            training_hadamard_backends.add(
                next_quantizers[name].training_hadamard_backend
            )
            candidate_seconds += time.perf_counter() - started
        quantizers = next_quantizers
    states = {name: quantizer.hard_state() for name, quantizer in quantizers.items()}
    if args.state_output is not None:
        state_tensors = {}
        for name, state in states.items():
            module_name = f"{layer_prefix}.{name.removesuffix('.weight')}"
            state_tensors[f"{module_name}.trellis"] = repack_p32_window_to_planar(
                state["words"], bits=3,
            ).cpu()
            state_tensors[f"{module_name}.SV"] = state["SV"].cpu()
        args.state_output.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            state_tensors,
            args.state_output,
            metadata={
                "format": "qvq_w3_p32_gsq_hard_state",
                "layer": str(args.layer),
                "rounds": str(args.rounds),
                "train_token_hashes": ",".join(train_hashes),
                "validation_token_hashes": ",".join(validation_hashes),
            },
        )
    payload = {
        "layer": args.layer,
        "stage": "joint_mlp_full_block",
        "attention_boundary": "hard_qvq_qkvo_shared_teacher_student",
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
        "rounds": args.rounds,
        "updates": args.rounds * args.epochs * args.train_samples,
        "candidates": args.candidates,
        "training_hadamard_backends": sorted(training_hadamard_backends),
        "capture_seconds": capture_seconds,
        "candidate_seconds": candidate_seconds,
        "fit_seconds": fit_seconds,
        "train_hard_loss_before": round_results[0]["train_hard_loss_before"],
        "train_hard_loss_after": round_results[-1]["train_hard_loss_after"],
        "validation_hard_loss_before": round_results[0]["validation_hard_loss_before"],
        "validation_hard_loss_after": round_results[-1]["validation_hard_loss_after"],
        "best_validation_hard_loss": result["best_validation_hard_loss"],
        "best_validation_epoch": result["best_validation_epoch"],
        "round_results": round_results,
        "changed_tiles": {
            name: int((state["words"] != initial_words[name]).any(-1).sum())
            for name, state in states.items()
        },
        "scale_max_abs_delta": {
            name: float((state["SV"] - initial_scales[name]).abs().max())
            for name, state in states.items()
        },
        "roundtrip": all(torch.equal(
            quantizers[name].adapter.pack(quantizers[name].adapter.unpack(state["words"])),
            state["words"],
        ) for name, state in states.items()),
        "state_output": None if args.state_output is None else str(args.state_output.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
