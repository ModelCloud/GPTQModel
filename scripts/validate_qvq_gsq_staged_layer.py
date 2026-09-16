#!/usr/bin/env python3
"""Train a complete Q/K -> V/O -> MLP staged QVQ layer."""

import argparse
import atexit
import copy
import json
import os
import time
from pathlib import Path

# cuBLAS requires a workspace contract for bitwise deterministic GEMM. This
# must be present before the first CUDA context is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel.looper.gsq_training_capture import (
    capture_llama_gsq_inputs,
    prepare_llama_gsq_capture,
)
from gptqmodel.quantization.gsq_batching import llama_stage_batches
from gptqmodel.quantization.gsq_training import (
    LlamaGSQAttentionStage,
    evaluate_hard_stage,
    fit_reconstruction_stage,
    prepare_qk_calibration_factor,
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
from scripts.validate_qvq_gsq_staged_mlp import projection_payload
from scripts.validate_qvq_gsq_staged_projection import digest, fineweb_chunks

PROJECTIONS = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)


class DocumentSlice:
    """Lazy contiguous view over captured decoder documents."""

    def __init__(self, documents, start, stop):
        self.documents = documents
        self.start = start
        self.stop = stop
        self.offloaded = getattr(documents, "offloaded", False)
        self.fixed_sequence_length = getattr(documents, "fixed_sequence_length", None)
        self.has_attention_mask = getattr(documents, "has_attention_mask", None)

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self):
            raise IndexError(index)
        return self.documents[self.start + index]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-model", type=Path,
                        default=Path("/monster/data/model/Llama-3.2-1B-Instruct"))
    parser.add_argument("--qvq-model", type=Path,
                        default=Path("/root/qvq-results/w3-p32-gsq-ab-20260915/gsq"))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--train-samples", type=int, default=8)
    parser.add_argument("--validation-samples", type=int, default=4)
    parser.add_argument("--qk-samples", type=int, default=0,
                        help="Separate post-validation Q/K metric rows; zero reuses training rows")
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--validation-offset", type=int)
    parser.add_argument("--qk-offset", type=int)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--qk-steps", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=33)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--offload-capture", action="store_true")
    parser.add_argument("--capture-directory", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-nondeterministic", action="store_true")
    parser.add_argument("--token-cache", type=Path)
    parser.add_argument("--prefix-state", type=Path)
    parser.add_argument("--state-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def documents_for_run(args, tokenizer):
    count = args.train_samples + args.validation_samples + args.qk_samples
    validation_offset = (
        args.train_offset + args.train_samples
        if args.validation_offset is None else args.validation_offset
    )
    qk_offset = (
        validation_offset + args.validation_samples
        if args.qk_offset is None else args.qk_offset
    )
    ranges = (
        (args.train_offset, args.train_samples),
        (validation_offset, args.validation_samples),
        (qk_offset, args.qk_samples),
    )
    if any(start < 0 for start, _ in ranges):
        raise ValueError("FineWeb-Edu token-cache offsets must be nonnegative")
    if args.token_cache is not None and args.token_cache.is_file():
        cached = load_file(args.token_cache)
        if "tokens" in cached:
            selected = [
                cached["tokens"][start:start + size]
                for start, size in ranges if size
            ]
            rows = torch.cat(selected) if selected else cached["tokens"][:0]
            documents = [row[:args.sequence_length].long() for row in rows]
        else:
            if ranges != ((0, args.train_samples),
                          (args.train_samples, args.validation_samples),
                          (args.train_samples + args.validation_samples, args.qk_samples)):
                raise ValueError("per-document token caches support only contiguous split offsets")
            documents = [cached[f"tokens.{index:05d}"] for index in range(count)]
    else:
        documents = fineweb_chunks(
            tokenizer, count=count, length=args.sequence_length, seed=args.seed,
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
    if any(tokens.shape != (args.sequence_length,) for tokens in documents):
        raise ValueError("FineWeb-Edu token cache has the wrong sequence length")
    if len(documents) != count:
        raise ValueError("FineWeb-Edu token cache does not contain all requested split rows")
    return documents


def fitting_options(args, *, epochs, seed, validation_batches):
    return {
        "epochs": epochs,
        "seed": seed,
        "assignment_lr": 1e-4,
        "scale_lr": 5e-5,
        "weight_decay": 1.,
        "betas": (.9, .95),
        "temperature": (2., .05),
        "multiplier": (100., 500.),
        "min_lr": .1,
        "decay": "cosine",
        "validation_batches": validation_batches,
        "restore_best": True,
    }


def main():
    args = parse_args()
    if not args.allow_nondeterministic:
        torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.dense_model, local_files_only=True)
    documents = documents_for_run(args, tokenizer)
    train_hashes = [digest(tokens) for tokens in documents[:args.train_samples]]
    validation_end = args.train_samples + args.validation_samples
    validation_hashes = [digest(tokens) for tokens in documents[args.train_samples:validation_end]]
    qk_hashes = [digest(tokens) for tokens in documents[validation_end:]]
    if set(train_hashes) & set(validation_hashes):
        raise RuntimeError("FineWeb-Edu train and heldout token chunks overlap")
    if (set(train_hashes) & set(qk_hashes)) or (set(validation_hashes) & set(qk_hashes)):
        raise RuntimeError("FineWeb-Edu Q/K metric chunks overlap train or heldout chunks")

    model = AutoModelForCausalLM.from_pretrained(
        args.dense_model, dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True,
    ).to(device).eval()
    layer_prefix = f"model.layers.{args.layer}"
    prefix_tensors = {}
    if args.prefix_state is not None:
        prefix_tensors = load_file(args.prefix_state)
        prefix_modules = sorted(
            name.removesuffix(".trellis")
            for name in prefix_tensors
            if name.endswith(".trellis")
        )
        if not prefix_modules:
            raise ValueError("prefix state contains no P32 trellis payloads")
        for name in prefix_modules:
            parts = name.split(".")
            if len(parts) < 4 or parts[:2] != ["model", "layers"]:
                raise ValueError(f"invalid prefix-state module name: {name}")
            if int(parts[2]) >= args.layer:
                raise ValueError("prefix state must contain only preceding decoder layers")
    # Every preceding layer participates in the quantized prefix. A missing
    # state entry means that layer was globally rejected and must replay its
    # original QVQ payload, never the dense source weight.
    for prior_layer in range(args.layer):
        for projection_name in PROJECTIONS:
            name = f"model.layers.{prior_layer}.{projection_name}"
            projection = model.get_submodule(name)
            trellis, SU, SV, bank_ids, bank_alt_id = projection_payload(
                args.qvq_model, name, device,
            )
            selected_trellis = prefix_tensors.get(f"{name}.trellis", trellis).to(device)
            selected_sv = prefix_tensors.get(f"{name}.SV", SV).to(device)
            quantized = p32_payload_weight(
                selected_trellis,
                SU,
                selected_sv,
                bank_ids,
                bank_alt_id,
                in_features=projection.in_features,
                out_features=projection.out_features,
            )
            with torch.no_grad():
                projection.weight.copy_(quantized.to(projection.weight.dtype))
    capture_started = time.perf_counter()
    capture_directory = args.capture_directory
    if args.offload_capture and capture_directory is None:
        capture_directory = args.output.parent / f"capture-layer{args.layer}"
    capture = capture_llama_gsq_inputs(
        model,
        [{"input_ids": tokens.tolist()} for tokens in documents],
        layer_index=args.layer,
        offload_to_cpu=args.offload_capture,
        offload_directory=capture_directory if args.offload_capture else None,
    )
    prepared, batches = prepare_llama_gsq_capture(
        model.model.layers[args.layer], capture, device=device,
    )
    cleanup_capture = getattr(capture, "cleanup", None)
    if cleanup_capture is not None:
        atexit.register(cleanup_capture)
    else:
        del capture
    capture_seconds = time.perf_counter() - capture_started
    del model
    torch.cuda.empty_cache()
    dense_layer = copy.deepcopy(prepared).eval()
    train_documents = DocumentSlice(batches, 0, args.train_samples)
    validation_documents = DocumentSlice(
        batches, args.train_samples, args.train_samples + args.validation_samples,
    )
    qk_documents = (
        DocumentSlice(batches, validation_end, validation_end + args.qk_samples)
        if args.qk_samples else train_documents
    )

    constants = {}
    initial_words = {}
    initial_scales = {}
    baseline_weights = {}
    candidate_seconds = 0.
    for name in PROJECTIONS:
        full_name = f"{layer_prefix}.{name}"
        projection = prepared.get_submodule(name)
        trellis, SU, SV, bank_ids, bank_alt_id = projection_payload(
            args.qvq_model, full_name, device,
        )
        constants[name] = (
            trellis, SU, bank_ids, bank_alt_id, projection.weight.detach().float(),
        )
        initial_words[name] = repack_p32_planar_to_window(trellis, bits=3)
        initial_scales[name] = SV.detach().clone()
        baseline_weights[f"{name}.weight"] = p32_payload_weight(
            trellis, SU, SV, bank_ids, bank_alt_id,
            in_features=projection.in_features,
            out_features=projection.out_features,
        )

    def new_quantizer(name, words=None, scales=None, round_index=0):
        nonlocal candidate_seconds
        trellis, SU, bank_ids, bank_alt_id, teacher_weight = constants[name]
        started = time.perf_counter()
        if words is None:
            quantizer = p32_training_module_from_payload(
                trellis, SU, initial_scales[name], bank_ids, bank_alt_id, teacher_weight,
                candidates=args.candidates, seed=args.seed + round_index,
            )
        else:
            quantizer = p32_training_module_from_words(
                words, SU, scales, bank_ids, bank_alt_id, teacher_weight,
                candidates=args.candidates, seed=args.seed + round_index,
            )
        candidate_seconds += time.perf_counter() - started
        return quantizer

    implicit_causal = prepared.self_attn.config._attn_implementation == "sdpa"
    train_stage_batches = llama_stage_batches(
        train_documents,
        batch_size=args.batch_size,
        microbatch_size=args.microbatch_size,
        device=device,
        implicit_causal=implicit_causal,
        lazy=args.offload_capture,
    )
    validation_stage_batches = llama_stage_batches(
        validation_documents,
        batch_size=args.batch_size,
        microbatch_size=args.microbatch_size,
        device=device,
        implicit_causal=implicit_causal,
        lazy=args.offload_capture,
    )
    stage_records = {}
    accepted_states = {}
    fit_seconds = 0.

    train_factor, train_dead = prepare_qk_calibration_factor(
        (qk_documents[index][0] for index in range(len(qk_documents))),
        device=device,
        transform=prepared.input_layernorm,
    )
    validation_factor, validation_dead = prepare_qk_calibration_factor(
        (validation_documents[index][0] for index in range(len(validation_documents))),
        device=device,
        transform=prepared.input_layernorm,
    )
    for name in PROJECTIONS[:2]:
        quantizer = new_quantizer(name)
        target = constants[name][-1].clone()

        def qk_objective(batch, weights, target=target):
            factor, dead = batch
            error = (target - weights["weight"]).masked_fill(dead[None], 0.)
            return (error @ factor).square().sum()

        rounds = []
        for round_index in range(args.rounds):
            result = fit_reconstruction_stage(
                {"weight": quantizer},
                [[((train_factor, train_dead), 1)]],
                qk_objective,
                **fitting_options(
                    args,
                    epochs=args.qk_steps,
                    seed=args.seed + round_index,
                    validation_batches=[[((validation_factor, validation_dead), 1)]],
                ),
            )
            fit_seconds += result["elapsed_seconds"]
            state = quantizer.hard_state()
            rounds.append({
                "round": round_index + 1,
                "train_before": result["hard_loss_before"],
                "train_after": result["hard_loss_after"],
                "validation_before": result["validation_hard_loss_before"],
                "validation_after": result["validation_hard_loss_after"],
                "best_epoch": result["best_validation_epoch"],
                "changed_tiles": int((state["choices"] != 0).sum()),
            })
            if round_index + 1 < args.rounds:
                quantizer = new_quantizer(
                    name, state["words"], state["SV"], round_index + 1,
                )
        accepted_states[name] = state
        with torch.no_grad():
            prepared.get_submodule(name).weight.copy_(quantizer.hard_weight().to(torch.bfloat16))
        stage_records[name] = rounds

    def fit_joint_stage(stage_name, names, stage):
        nonlocal fit_seconds
        quantizers = {f"{name}.weight": new_quantizer(name) for name in names}

        def objective(batch, weights):
            hidden, kwargs, mask = batch
            return reconstruction_stage_loss(
                stage, (hidden,), kwargs, student_weights=weights, output_mask=mask,
            )

        rounds = []
        for round_index in range(args.rounds):
            result = fit_reconstruction_stage(
                quantizers,
                train_stage_batches,
                objective,
                **fitting_options(
                    args,
                    epochs=args.epochs,
                    seed=args.seed + round_index,
                    validation_batches=validation_stage_batches,
                ),
            )
            fit_seconds += result["elapsed_seconds"]
            states = {name: quantizer.hard_state() for name, quantizer in quantizers.items()}
            rounds.append({
                "round": round_index + 1,
                "train_before": result["hard_loss_before"],
                "train_after": result["hard_loss_after"],
                "validation_before": result["validation_hard_loss_before"],
                "validation_after": result["validation_hard_loss_after"],
                "best_epoch": result["best_validation_epoch"],
                "changed_tiles": {
                    name: int((state["choices"] != 0).sum()) for name, state in states.items()
                },
            })
            if round_index + 1 < args.rounds:
                quantizers = {
                    key: new_quantizer(
                        key.removesuffix(".weight"),
                        state["words"],
                        state["SV"],
                        round_index + 1,
                    )
                    for key, state in states.items()
                }
        for key, quantizer in quantizers.items():
            name = key.removesuffix(".weight")
            accepted_states[name] = quantizer.hard_state()
            with torch.no_grad():
                prepared.get_submodule(name).weight.copy_(quantizer.hard_weight().to(torch.bfloat16))
        stage_records[stage_name] = rounds

    fit_joint_stage(
        "attention",
        PROJECTIONS[2:4],
        LlamaGSQAttentionStage(prepared),
    )
    fit_joint_stage("mlp", PROJECTIONS[4:], prepared)

    final_weights = {}
    state_tensors = dict(prefix_tensors)
    for name, state in accepted_states.items():
        _, SU, bank_ids, bank_alt_id, teacher_weight = constants[name]
        quantizer = p32_training_module_from_words(
            state["words"], SU, state["SV"], bank_ids, bank_alt_id, teacher_weight,
            candidates=2, seed=args.seed,
        )
        final_weights[f"{name}.weight"] = quantizer.hard_weight()
        full_name = f"{layer_prefix}.{name}"
        state_tensors[f"{full_name}.trellis"] = repack_p32_window_to_planar(
            state["words"], bits=3,
        ).cpu()
        state_tensors[f"{full_name}.SV"] = state["SV"].cpu()

    def full_block_objective(batch, weights):
        hidden, kwargs, mask = batch
        return reconstruction_stage_loss(
            dense_layer, (hidden,), kwargs, student_weights=weights, output_mask=mask,
        )

    baseline_validation = evaluate_hard_stage(
        {}, validation_stage_batches,
        lambda batch, _weights: full_block_objective(batch, baseline_weights),
    )
    final_validation = evaluate_hard_stage(
        {}, validation_stage_batches,
        lambda batch, _weights: full_block_objective(batch, final_weights),
    )
    candidate_full_block_validation = final_validation
    layer_accepted = final_validation <= baseline_validation
    if not layer_accepted:
        final_validation = baseline_validation
        final_weights = baseline_weights
        for name in PROJECTIONS:
            accepted_states[name] = {
                "words": initial_words[name],
                "SV": initial_scales[name],
            }
            trellis, _, _, _, _ = constants[name]
            full_name = f"{layer_prefix}.{name}"
            state_tensors[f"{full_name}.trellis"] = trellis.cpu()
            state_tensors[f"{full_name}.SV"] = initial_scales[name].cpu()
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
            "qk_metric_token_hashes": ",".join(qk_hashes),
        },
    )
    payload = {
        "layer": args.layer,
        "stage_order": ["q_proj", "k_proj", "joint_v_o_attention", "joint_mlp_full_block"],
        "device": torch.cuda.get_device_name(),
        "dataset": "HuggingFaceFW/fineweb-edu/sample-10BT",
        "train_tokens": args.train_samples * args.sequence_length,
        "validation_tokens": args.validation_samples * args.sequence_length,
        "qk_metric_tokens": (
            args.qk_samples if args.qk_samples else args.train_samples
        ) * args.sequence_length,
        "split_offsets": {
            "train": args.train_offset,
            "validation": (
                args.train_offset + args.train_samples
                if args.validation_offset is None else args.validation_offset
            ),
            "qk": (
                (args.train_offset + args.train_samples
                 if args.validation_offset is None else args.validation_offset)
                + args.validation_samples
                if args.qk_offset is None else args.qk_offset
            ),
        },
        "train_hashes": train_hashes,
        "validation_hashes": validation_hashes,
        "qk_metric_hashes": qk_hashes if args.qk_samples else train_hashes,
        "strict_disjoint": not bool(set(train_hashes) & set(validation_hashes)),
        "epochs": args.epochs,
        "qk_steps": args.qk_steps,
        "rounds": args.rounds,
        "candidates": args.candidates,
        "batch_size": args.batch_size,
        "microbatch_size": args.microbatch_size,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "offload_capture": args.offload_capture,
        "capture_seconds": capture_seconds,
        "candidate_seconds": candidate_seconds,
        "fit_seconds": fit_seconds,
        "stages": stage_records,
        "full_block_validation_before": baseline_validation,
        "full_block_validation_candidate": candidate_full_block_validation,
        "full_block_validation_after": final_validation,
        "layer_accepted": layer_accepted,
        "full_block_validation_improvement_percent": 100. * (
            baseline_validation - final_validation
        ) / baseline_validation,
        "changed_tiles": {
            name: int((state["words"] != initial_words[name]).any(-1).sum())
            for name, state in accepted_states.items()
        },
        "scale_max_abs_delta": {
            name: float((state["SV"] - initial_scales[name]).abs().max())
            for name, state in accepted_states.items()
        },
        "state_output": str(args.state_output.resolve()),
        "prefix_state": None if args.prefix_state is None else str(args.prefix_state.resolve()),
        "cumulative_state_tensors": len(state_tensors),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))
    if cleanup_capture is not None:
        cleanup_capture()
        atexit.unregister(cleanup_capture)


if __name__ == "__main__":
    main()
