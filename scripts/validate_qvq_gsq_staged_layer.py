#!/usr/bin/env python3
"""Train a complete Q/K -> V/O -> MLP staged QVQ layer."""

import argparse
import atexit
import concurrent.futures
import copy
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

# cuBLAS requires a workspace contract for bitwise deterministic GEMM. This
# must be present before the first CUDA context is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for import_root in (SCRIPT_DIR, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from gpu_idle_preflight import (
    add_gpu_idle_preflight_args,
    bootstrap_gpu_idle_preflight,
    recheck_gpu_exclusivity,
)

_GPU_IDLE_PREFLIGHT = bootstrap_gpu_idle_preflight() if __name__ == "__main__" else None

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
    LlamaGSQMLPStage,
    evaluate_hard_stage,
    fit_reconstruction_stage,
    prepare_qk_calibration_factor,
    reconstruction_stage_loss,
    reconstruction_stage_student_loss,
)
from gptqmodel.quantization.gsq_training_qvq import (
    p32_payload_weight,
    p32_training_module_from_payload,
    p32_training_module_from_words,
)
from gptqmodel.quantization.qvq import (
    repack_p32_planar_to_window,
    repack_p32_window_to_planar,
    rht_preprocess_weight,
    rht_reconstruct_weight,
)
from gptqmodel.quantization.qvq_gsq import refine_trellis_candidates
from gptqmodel.utils.hadamard import hadamard_transform
from scripts.validate_qvq_gsq_staged_mlp import projection_payload
from scripts.validate_qvq_gsq_staged_projection import digest, fineweb_chunks

PROJECTIONS = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)


def two_sided_normalized_hadamard(matrix):
    """Return H @ matrix @ H for the normalized Walsh-Hadamard matrix H."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("two-sided Hadamard input must be square")
    width = matrix.shape[0]
    if width < 8 or width > 32768 or width & (width - 1):
        raise ValueError("two-sided Hadamard width must be a supported power of two")
    scale = 1. / math.sqrt(width)
    right = hadamard_transform(matrix.contiguous(), scale)
    return hadamard_transform(right.T.contiguous(), scale).T.contiguous()


def closed_form_qk_scales(target, unscaled, factor, dead):
    """Solve independent output scales for ||(target - diag(s) W) D F||²."""
    if target.shape != unscaled.shape or target.ndim != 2:
        raise ValueError("Q/K scale solve requires matching rank-two weights")
    if factor.shape != (target.shape[1], target.shape[1]) or dead.shape != (target.shape[1],):
        raise ValueError("Q/K scale solve metric does not match the input width")
    target_factor = target.masked_fill(dead[None], 0.) @ factor
    unscaled_factor = unscaled.masked_fill(dead[None], 0.) @ factor
    return (
        (target_factor * unscaled_factor).sum(1)
        / unscaled_factor.square().sum(1).clamp_min(torch.finfo(torch.float32).tiny)
    )


def select_qk_pair(q_alternatives, k_alternatives, objective):
    """Select the Q/K Cartesian pair with the lowest downstream objective."""
    if not q_alternatives or not k_alternatives:
        raise ValueError("Q/K pair selection requires non-empty alternatives")
    pairs = []
    losses = []
    for q_alternative in q_alternatives:
        for k_alternative in k_alternatives:
            pairs.append((q_alternative, k_alternative))
            losses.append(objective(q_alternative, k_alternative))
    if any(isinstance(loss, torch.Tensor) for loss in losses):
        if not all(isinstance(loss, torch.Tensor) and loss.numel() == 1 for loss in losses):
            raise TypeError("Q/K downstream replay losses must all be scalar tensors")
        loss_values = torch.stack(losses).detach().cpu().tolist()
    else:
        loss_values = losses
    measurements = []
    for (q_alternative, k_alternative), loss in zip(pairs, loss_values):
        if not math.isfinite(loss):
            raise ValueError("Q/K downstream replay produced a nonfinite loss")
        measurements.append({
                "q_selection": q_alternative["selection"],
                "k_selection": k_alternative["selection"],
                "loss": loss,
        })
    best_index = min(range(len(measurements)), key=lambda index: measurements[index]["loss"])
    k_count = len(k_alternatives)
    return (
        q_alternatives[best_index // k_count],
        k_alternatives[best_index % k_count],
        measurements,
    )


def unique_qk_alternatives(alternatives):
    """Remove states that serialize to the same P32 payload and scales."""
    unique = []
    for alternative in alternatives:
        if not any(
            torch.equal(alternative["state"]["words"], accepted["state"]["words"])
            and torch.equal(alternative["state"]["SV"], accepted["state"]["SV"])
            for accepted in unique
        ):
            unique.append(alternative)
    return unique


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
    parser.add_argument(
        "--qk-hard-eval-interval", type=int, default=100,
        help="Updates between exact structured Q/K hard-oracle checkpoints",
    )
    parser.add_argument(
        "--qk-hard-dense-verify-topk", type=int, default=4,
        help="Dense-FP32 verification candidates after structured Q/K hard scoring",
    )
    parser.add_argument(
        "--qk-cuda-graph-updates-per-replay", type=int, default=4,
        help="Exact Fisher updates captured per Q/K CUDA graph replay when cadences align",
    )
    parser.add_argument(
        "--qk-soft-dtype",
        choices=("float32", "bfloat16"),
        default="bfloat16",
        help="Soft exact-Fisher relaxation dtype; hard Fisher and held-out guards remain FP32",
    )
    parser.add_argument(
        "--mlp-soft-dtype", choices=("float32", "bfloat16"), default="float32",
        help="Relaxed MLP weight reconstruction dtype; attention remains FP32",
    )
    parser.add_argument(
        "--mlp-fp32-tail-epochs", type=int, default=0,
        help="Final MLP epochs reconstructed in FP32 after an optional BF16 prefix",
    )
    parser.add_argument(
        "--attention-validation-tail-epochs", type=int, default=6,
        help="Validate hard attention checkpoints only during this many final epochs",
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=33)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--offload-capture", action="store_true")
    parser.add_argument("--capture-directory", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--allow-nondeterministic", action="store_true")
    parser.add_argument(
        "--disable-fast-training-hadamard",
        action="store_true",
        help="Use the eager FP32 Hadamard oracle during GSQ optimization",
    )
    parser.add_argument(
        "--parallel-candidate-build",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build independent projection candidate banks on concurrent CUDA streams",
    )
    parser.add_argument(
        "--fused-qk-fisher",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the fused exact-Fisher P32 optimizer and closed-form Q/K scales",
    )
    parser.add_argument("--token-cache", type=Path)
    parser.add_argument("--prefix-state", type=Path)
    parser.add_argument("--state-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    add_gpu_idle_preflight_args(parser)
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


def fitting_options(args, *, epochs, seed, validation_batches, fp32_tail_epochs=0,
                    validation_start_epoch=0):
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
        "fp32_tail_epochs": fp32_tail_epochs,
        "validation_start_epoch": validation_start_epoch,
        "export_weights": False,
    }


def main():
    args = parse_args()
    if args.attention_validation_tail_epochs < 1:
        raise ValueError("attention validation tail epochs must be positive")
    if args.qk_hard_eval_interval < 1:
        raise ValueError("Q/K hard evaluation interval must be positive")
    if args.qk_cuda_graph_updates_per_replay < 1:
        raise ValueError("Q/K CUDA graph updates per replay must be positive")
    if args.fused_qk_fisher and args.rounds != 1:
        raise ValueError("fused Q/K Fisher fitting currently requires exactly one P32 round")
    if not args.allow_nondeterministic:
        torch.use_deterministic_algorithms(True)
        # GSQ's large workspaces are fully overwritten before they are read.
        # Avoid deterministic mode's NaN poison fills for torch.empty buffers;
        # the poison is a debug guard, not part of deterministic execution.
        torch.utils.deterministic.fill_uninitialized_memory = False
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

    training_hadamard_backends = set()
    quantizer_metadata_lock = threading.Lock()
    candidate_wall_seconds = 0.

    def new_quantizer(name, words=None, scales=None, round_index=0, training_dtype="float32"):
        nonlocal candidate_seconds
        trellis, SU, bank_ids, bank_alt_id, teacher_weight = constants[name]
        started = time.perf_counter()
        if words is None:
            quantizer = p32_training_module_from_payload(
                trellis, SU, initial_scales[name], bank_ids, bank_alt_id, teacher_weight,
                candidates=args.candidates, seed=args.seed + round_index,
                fast_hadamard=not args.disable_fast_training_hadamard,
                training_dtype=getattr(torch, training_dtype),
            )
        else:
            quantizer = p32_training_module_from_words(
                words, SU, scales, bank_ids, bank_alt_id, teacher_weight,
                candidates=args.candidates, seed=args.seed + round_index,
                fast_hadamard=not args.disable_fast_training_hadamard,
                training_dtype=getattr(torch, training_dtype),
            )
        with quantizer_metadata_lock:
            training_hadamard_backends.add(quantizer.training_hadamard_backend)
            candidate_seconds += time.perf_counter() - started
        return quantizer

    def new_quantizers(names, *, training_dtype="float32"):
        """Build independent projection candidate banks on concurrent streams."""
        nonlocal candidate_wall_seconds
        names = tuple(names)
        if len(names) == 1 or not args.parallel_candidate_build:
            started = time.perf_counter()
            result = {
                name: new_quantizer(name, training_dtype=training_dtype)
                for name in names
            }
            candidate_wall_seconds += time.perf_counter() - started
            return result
        ready = torch.cuda.Event()
        ready.record()
        streams = {name: torch.cuda.Stream() for name in names}

        def build(name):
            stream = streams[name]
            with torch.cuda.stream(stream):
                stream.wait_event(ready)
                quantizer = new_quantizer(name, training_dtype=training_dtype)
            stream.synchronize()
            return name, quantizer

        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(names)) as executor:
            result = dict(executor.map(build, names))
        candidate_wall_seconds += time.perf_counter() - started
        return result

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
    qk_alternatives = {}
    qk_pair_guards = {}
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
    timing_exclusivity = (
        recheck_gpu_exclusivity(_GPU_IDLE_PREFLIGHT)
        if _GPU_IDLE_PREFLIGHT is not None else None
    )
    qk_quantizers = new_quantizers(PROJECTIONS[:2])
    qk_streams = {name: torch.cuda.Stream() for name in PROJECTIONS[:2]}
    qk_capture_barrier = threading.Barrier(2) if args.fused_qk_fisher else None

    def fit_qk_projection(name):
        quantizer = qk_quantizers[name]
        target = constants[name][-1].clone()

        def qk_objective(batch, weights):
            factor, dead = batch
            error = (target - weights["weight"]).masked_fill(dead[None], 0.)
            return (error @ factor).square().sum()

        rounds = []
        with torch.cuda.stream(qk_streams[name]):
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
            weight = quantizer.hard_weight().to(torch.bfloat16)
        qk_streams[name].synchronize()
        return name, quantizer, state, weight, rounds

    def fit_qk_projection_fused(name):
        quantizer = qk_quantizers[name]
        target = constants[name][-1].clone()
        masked_factor = train_factor.masked_fill(train_dead[:, None], 0.)
        hessian = masked_factor @ masked_factor.T

        def solve_scales(inner):
            unscaled = rht_reconstruct_weight(
                inner,
                quantizer.SU,
                torch.ones_like(quantizer.scales),
            )
            return closed_form_qk_scales(target, unscaled, train_factor, train_dead)

        fisher_scales = solve_scales(quantizer.baseline_matrix)

        input_metric = two_sided_normalized_hadamard(
            quantizer.SU[:, None] * hessian * quantizer.SU[None, :],
        )
        output_metric = two_sided_normalized_hadamard(torch.diag(fisher_scales.square()))
        inner_target = rht_preprocess_weight(
            target,
            quantizer.SU.reciprocal(),
            fisher_scales.reciprocal(),
        ).float()
        optimized = refine_trellis_candidates(
            quantizer.candidates,
            bits=3,
            bank_ids=quantizer.bank_ids,
            bank_alt_id=quantizer.bank_alt_id,
            target=inner_target,
            inputs=inner_target.new_zeros((1, inner_target.shape[0])),
            enabled=True,
            steps=args.qk_steps,
            learning_rate=1e-4,
            temperature_start=2.,
            temperature_end=.05,
            kappa_start=100.,
            kappa_end=500.,
            weight_decay=1.,
            soft_dtype=args.qk_soft_dtype,
            coordinate_sweeps=0,
            hard_eval_interval=min(args.qk_hard_eval_interval, args.qk_steps),
            relaxation_patience=0,
            decoded_candidates=None,
            sparse_candidate_indices=quantizer.sparse_indices.permute(1, 0, 2).contiguous(),
            sparse_candidate_deltas=quantizer.sparse_deltas.permute(1, 0, 2).contiguous(),
            sparse_candidate_shifts=quantizer.sparse_shifts.T.contiguous(),
            input_metric=input_metric,
            output_metric=output_metric,
            output_metric_hadamard_diagonal=fisher_scales.square(),
            hard_dense_verify_topk=args.qk_hard_dense_verify_topk,
            cuda_graph_updates_per_replay=args.qk_cuda_graph_updates_per_replay,
            capture_barrier=qk_capture_barrier,
            seed=args.seed,
        )
        inner = quantizer.adapter.inner(
            optimized.words,
            quantizer.in_features,
            quantizer.out_features,
            quantizer.bank_ids,
            quantizer.bank_alt_id,
        )
        scales = solve_scales(inner)
        state = {
            "words": optimized.words,
            "choices": optimized.choices,
            "SV": scales,
        }
        candidate_weight = rht_reconstruct_weight(
            inner, quantizer.SU, scales,
        )
        baseline_scale_weight = rht_reconstruct_weight(
            quantizer.baseline_matrix, quantizer.SU, fisher_scales,
        )

        def qk_loss(factor, dead, candidate):
            error = (target - candidate).masked_fill(dead[None], 0.)
            return float((error @ factor).square().sum())

        baseline = quantizer.hard_weight_for_evaluation()
        train_before = qk_loss(train_factor, train_dead, baseline)
        validation_before = qk_loss(validation_factor, validation_dead, baseline)
        alternatives = [
            {
                "validation_loss": validation_before,
                "train_loss": train_before,
                "state": quantizer.hard_state(),
                "weight": baseline,
                "selection": "original",
            },
            {
                "validation_loss": qk_loss(
                    validation_factor, validation_dead, baseline_scale_weight,
                ),
                "train_loss": qk_loss(train_factor, train_dead, baseline_scale_weight),
                "state": {
                    "words": quantizer.candidates[0].clone(),
                    "choices": torch.zeros_like(optimized.choices),
                    "SV": fisher_scales,
                },
                "weight": baseline_scale_weight,
                "selection": "closed_form_scale",
            },
            {
                "validation_loss": qk_loss(
                    validation_factor, validation_dead, candidate_weight,
                ),
                "train_loss": qk_loss(train_factor, train_dead, candidate_weight),
                "state": state,
                "weight": candidate_weight,
                "selection": "fused_p32_and_closed_form_scale",
            },
        ]
        selected = min(
            alternatives, key=lambda alternative: alternative["validation_loss"],
        )
        validation_after = selected["validation_loss"]
        train_after = selected["train_loss"]
        state = selected["state"]
        selected_weight = selected["weight"]
        selection = selected["selection"]
        weight = selected_weight.to(torch.bfloat16)
        if validation_after > validation_before:
            state = quantizer.hard_state()
            weight = baseline.to(torch.bfloat16)
            train_after = train_before
            validation_after = validation_before
        rounds = [{
            "round": 1,
            "train_before": train_before,
            "train_after": train_after,
            "validation_before": validation_before,
            "validation_after": validation_after,
            "best_epoch": args.qk_steps - 1 if selection.startswith("fused_p32") else -1,
            "changed_tiles": int((state["choices"] != 0).sum()),
            "optimizer": "fused_exact_fisher_closed_form_scale",
            "selection": selection,
            "relaxation_requested_steps": args.qk_steps,
            "relaxation_completed_steps": optimized.diagnostics["completed_steps"],
            "relaxation_changed_tiles": optimized.diagnostics["relaxation_changed_tiles"],
            "relaxation_soft_dtype": optimized.diagnostics["soft_dtype"],
            "hard_oracle": optimized.diagnostics["hard_oracle"],
            "hard_dense_verify_topk": optimized.diagnostics["hard_dense_verify_topk"],
        }]
        return name, quantizer, state, weight, rounds, alternatives

    def fit_qk_projection_fused_on_stream(name):
        stream = qk_streams[name]
        with torch.cuda.stream(stream):
            result = fit_qk_projection_fused(name)
        stream.synchronize()
        return result

    qk_started = time.perf_counter()
    torch.cuda.current_stream().synchronize()
    qk_fit = (
        fit_qk_projection_fused_on_stream
        if args.fused_qk_fisher else fit_qk_projection
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        qk_results = list(executor.map(qk_fit, PROJECTIONS[:2]))
    qk_fit_seconds = time.perf_counter() - qk_started
    fit_seconds += qk_fit_seconds
    for result in qk_results:
        name, quantizer, state, weight, rounds, *fused_alternatives = result
        accepted_states[name] = state
        with torch.no_grad():
            prepared.get_submodule(name).weight.copy_(weight)
        stage_records[name] = rounds
        if fused_alternatives:
            qk_alternatives[name] = unique_qk_alternatives(fused_alternatives[0])

    block_parameters = {
        name: value.detach() for name, value in dense_layer.named_parameters()
    }
    block_buffers = {
        name: value.detach().clone() for name, value in dense_layer.named_buffers()
    }
    validation_teacher_batches = None

    def validation_teachers():
        nonlocal validation_teacher_batches
        if validation_teacher_batches is None:
            validation_teacher_batches = []
            with torch.no_grad():
                for batch_index in range(len(validation_stage_batches)):
                    teacher_microbatches = []
                    for batch, count in validation_stage_batches[batch_index]:
                        hidden, kwargs, mask = batch
                        teacher = dense_layer(hidden, **kwargs)
                        teacher_microbatches.append(
                            ((batch, teacher if mask is None else teacher[mask]), count),
                        )
                    validation_teacher_batches.append(teacher_microbatches)
        return validation_teacher_batches

    def qk_pair_replay_weights(q_alternative, k_alternative, downstream_weights):
        return {
            **downstream_weights,
            "self_attn.q_proj.weight": q_alternative["weight"],
            "self_attn.k_proj.weight": k_alternative["weight"],
        }

    def qk_pair_replay_objective(q_alternative, k_alternative, downstream_weights):
        weights = qk_pair_replay_weights(q_alternative, k_alternative, downstream_weights)
        student_state = {**block_parameters}
        for name, value in weights.items():
            parameter = block_parameters[name]
            student_state[name] = value.to(device=parameter.device, dtype=parameter.dtype)
        weighted_loss = torch.zeros((), device=device)
        elements = 0
        with torch.no_grad():
            for teacher_microbatches in validation_teachers():
                for (batch, teacher), count in teacher_microbatches:
                    hidden, kwargs, mask = batch
                    student = torch.func.functional_call(
                        dense_layer,
                        (student_state, block_buffers),
                        (hidden,),
                        kwargs,
                    )
                    student = student if mask is None else student[mask]
                    weighted_loss += torch.nn.functional.mse_loss(student, teacher) * count
                    elements += count
        if not elements:
            raise ValueError("Q/K downstream replay requires held-out output elements")
        return weighted_loss / elements

    def install_qk_pair(q_alternative, k_alternative):
        for name, alternative in zip(PROJECTIONS[:2], (q_alternative, k_alternative)):
            accepted_states[name] = alternative["state"]
            with torch.no_grad():
                prepared.get_submodule(name).weight.copy_(
                    alternative["weight"].to(torch.bfloat16),
                )
            record = stage_records[name][-1]
            record["selection"] = alternative["selection"]
            record["train_after"] = alternative["train_loss"]
            record["validation_after"] = alternative["validation_loss"]
            record["changed_tiles"] = int(
                (alternative["state"]["choices"] != 0).sum(),
            )

    def materialize_state(name, state):
        _, SU, bank_ids, bank_alt_id, teacher_weight = constants[name]
        if name in qk_quantizers:
            quantizer = qk_quantizers[name]
            inner = quantizer.adapter.inner(
                state["words"], teacher_weight.shape[1], teacher_weight.shape[0],
                bank_ids, bank_alt_id,
            )
            return rht_reconstruct_weight(inner, SU, state["SV"])
        return p32_training_module_from_words(
            state["words"], SU, state["SV"], bank_ids, bank_alt_id, teacher_weight,
            candidates=2, seed=args.seed,
        ).hard_weight()

    qk_pair_guard_seconds = 0.
    if len(qk_alternatives) == 2:
        guard_started = time.perf_counter()
        initial_downstream_weights = {
            f"{name}.weight": baseline_weights[f"{name}.weight"]
            for name in PROJECTIONS[2:]
        }
        q_alternative, k_alternative, measurements = select_qk_pair(
            qk_alternatives[PROJECTIONS[0]],
            qk_alternatives[PROJECTIONS[1]],
            lambda q, k: qk_pair_replay_objective(
                q, k, initial_downstream_weights,
            ),
        )
        qk_pair_guards["before_downstream_fit"] = {
            "applied": True,
            "q_selection": q_alternative["selection"],
            "k_selection": k_alternative["selection"],
            "loss": min(measurement["loss"] for measurement in measurements),
            "measurements": measurements,
        }
        install_qk_pair(q_alternative, k_alternative)
        qk_pair_guard_seconds += time.perf_counter() - guard_started

    def cache_stage_batches(source_batches, stage, prefix_stage=None):
        cached = []
        with torch.no_grad():
            for batch_index in range(len(source_batches)):
                cached_microbatches = []
                for batch, count in source_batches[batch_index]:
                    hidden, kwargs, mask = batch
                    if prefix_stage is None:
                        stage_hidden, stage_kwargs = hidden, kwargs
                    else:
                        stage_hidden = prefix_stage(hidden, **kwargs)
                        stage_kwargs = {}
                    teacher = stage(stage_hidden, **stage_kwargs)
                    cached_microbatches.append(
                        ((stage_hidden, stage_kwargs, mask, teacher), count),
                    )
                cached.append(cached_microbatches)
        return cached

    def fit_joint_stage(stage_name, names, stage, prefix_stage=None):
        nonlocal fit_seconds
        training_dtype = args.mlp_soft_dtype if stage_name == "mlp" else "float32"
        quantizers = {
            f"{name}.weight": quantizer
            for name, quantizer in new_quantizers(
                names, training_dtype=training_dtype,
            ).items()
        }

        cache_started = time.perf_counter()
        stage_train_batches = cache_stage_batches(
            train_stage_batches, stage, prefix_stage,
        )
        stage_validation_batches = cache_stage_batches(
            validation_stage_batches, stage, prefix_stage,
        )
        cache_seconds = time.perf_counter() - cache_started

        def objective(batch, weights):
            hidden, kwargs, mask, teacher = batch
            return reconstruction_stage_student_loss(
                stage, (hidden,), kwargs, student_weights=weights,
                teacher=teacher, output_mask=mask,
            )

        rounds = []
        for round_index in range(args.rounds):
            validation_start_epoch = 0
            if stage_name == "attention":
                validation_start_epoch = max(
                    0, args.epochs - args.attention_validation_tail_epochs,
                )
            elif (args.mlp_soft_dtype == "bfloat16"
                  and args.mlp_fp32_tail_epochs):
                validation_start_epoch = max(
                    0, args.epochs - args.mlp_fp32_tail_epochs - 1,
                )
            torch.cuda.nvtx.range_push(f"gsq.stage.{stage_name}.round_{round_index + 1}")
            try:
                result = fit_reconstruction_stage(
                    quantizers,
                    stage_train_batches,
                    objective,
                    **fitting_options(
                        args,
                        epochs=args.epochs,
                        seed=args.seed + round_index,
                        validation_batches=stage_validation_batches,
                        fp32_tail_epochs=(
                            args.mlp_fp32_tail_epochs if stage_name == "mlp" else 0
                        ),
                        validation_start_epoch=validation_start_epoch,
                    ),
                )
            finally:
                torch.cuda.nvtx.range_pop()
            round_seconds = result["elapsed_seconds"] + (
                cache_seconds if round_index == 0 else 0.
            )
            fit_seconds += round_seconds
            states = {name: quantizer.hard_state() for name, quantizer in quantizers.items()}
            rounds.append({
                "round": round_index + 1,
                "train_before": result["hard_loss_before"],
                "train_after": result["hard_loss_after"],
                "validation_before": result["validation_hard_loss_before"],
                "validation_after": result["validation_hard_loss_after"],
                "best_epoch": result["best_validation_epoch"],
                "elapsed_seconds": round_seconds,
                "teacher_cache_seconds": cache_seconds if round_index == 0 else 0.,
                "fp32_tail_epochs": result["fp32_tail_epochs"],
                "validation_start_epoch": result["validation_start_epoch"],
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
                        training_dtype=training_dtype,
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
    fit_joint_stage(
        "mlp",
        PROJECTIONS[4:],
        LlamaGSQMLPStage(prepared),
        prefix_stage=LlamaGSQAttentionStage(prepared),
    )

    if len(qk_alternatives) == 2:
        guard_started = time.perf_counter()
        downstream_weights = {
            f"{name}.weight": materialize_state(name, accepted_states[name])
            for name in PROJECTIONS[2:]
        }
        for name in PROJECTIONS[:2]:
            for alternative in qk_alternatives[name]:
                alternative["weight"] = materialize_state(name, alternative["state"])
        q_alternative, k_alternative, measurements = select_qk_pair(
            qk_alternatives[PROJECTIONS[0]],
            qk_alternatives[PROJECTIONS[1]],
            lambda q, k: qk_pair_replay_objective(q, k, downstream_weights),
        )
        qk_pair_guards["after_downstream_fit"] = {
            "applied": True,
            "q_selection": q_alternative["selection"],
            "k_selection": k_alternative["selection"],
            "loss": min(measurement["loss"] for measurement in measurements),
            "measurements": measurements,
        }
        install_qk_pair(q_alternative, k_alternative)
        qk_pair_guard_seconds += time.perf_counter() - guard_started
    fit_seconds += qk_pair_guard_seconds

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
        "qk_hard_eval_interval": args.qk_hard_eval_interval,
        "qk_cuda_graph_updates_per_replay": args.qk_cuda_graph_updates_per_replay,
        "qk_soft_dtype": args.qk_soft_dtype,
        "mlp_soft_dtype": args.mlp_soft_dtype,
        "mlp_fp32_tail_epochs": args.mlp_fp32_tail_epochs,
        "attention_validation_tail_epochs": args.attention_validation_tail_epochs,
        "qk_hard_dense_verify_topk": args.qk_hard_dense_verify_topk,
        "rounds": args.rounds,
        "candidates": args.candidates,
        "batch_size": args.batch_size,
        "microbatch_size": args.microbatch_size,
        "training_hadamard_backends": sorted(training_hadamard_backends),
        "fused_qk_fisher": args.fused_qk_fisher,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_fill_uninitialized_memory": (
            torch.utils.deterministic.fill_uninitialized_memory
        ),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "python_gil_enabled": (
            sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else None
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "offload_capture": args.offload_capture,
        "parallel_candidate_build": args.parallel_candidate_build,
        "gpu_idle_preflight": (
            _GPU_IDLE_PREFLIGHT.as_dict() if _GPU_IDLE_PREFLIGHT is not None else None
        ),
        "gpu_timing_exclusivity": timing_exclusivity,
        "capture_seconds": capture_seconds,
        "candidate_seconds": candidate_seconds,
        "candidate_wall_seconds": candidate_wall_seconds,
        "fit_seconds": fit_seconds,
        "fit_stage_seconds": {
            "qk": qk_fit_seconds,
            "attention": sum(
                record["elapsed_seconds"] for record in stage_records["attention"]
            ),
            "mlp": sum(record["elapsed_seconds"] for record in stage_records["mlp"]),
            "qk_pair_guard": qk_pair_guard_seconds,
        },
        "stages": stage_records,
        "qk_pair_guards": qk_pair_guards,
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
