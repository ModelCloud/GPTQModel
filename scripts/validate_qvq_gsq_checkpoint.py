#!/usr/bin/env python3
"""Install a hard staged-GSQ state and evaluate a fresh QVQ reload."""

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel import BACKEND, GPTQModel
from scripts.validate_qvq_gsq_staged_projection import digest, fineweb_chunks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-model", type=Path,
                        default=Path("/monster/data/model/Llama-3.2-1B-Instruct"))
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--evaluation-samples", type=int, default=4)
    parser.add_argument("--evaluation-seed", type=int, default=991)
    parser.add_argument("--token-cache", type=Path)
    parser.add_argument("--token-offset", type=int, default=0)
    parser.add_argument("--role", choices=("validation", "report_only"), default="report_only")
    parser.add_argument(
        "--incumbent-evaluation",
        type=Path,
        help="validation JSON for the currently accepted prefix",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def state_metadata(path):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return handle.metadata() or {}


def install_state(source, destination, state_path):
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, copy_function=os.link)
    index = json.loads((source / "model.safetensors.index.json").read_text())
    replacements = load_file(state_path)
    by_shard = {}
    for name, value in replacements.items():
        if name not in index["weight_map"]:
            raise KeyError(f"GSQ state tensor is absent from checkpoint: {name}")
        by_shard.setdefault(index["weight_map"][name], {})[name] = value

    unchanged_tensors = 0
    for shard_name, shard_replacements in by_shard.items():
        source_shard = source / shard_name
        with safe_open(str(source_shard), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            tensors = {name: handle.get_tensor(name) for name in handle}
        for name, value in shard_replacements.items():
            original = tensors[name]
            if original.shape != value.shape or original.dtype != value.dtype:
                raise ValueError(
                    f"GSQ state tensor mismatch for {name}: "
                    f"checkpoint={original.shape}/{original.dtype}, state={value.shape}/{value.dtype}"
                )
            tensors[name] = value.contiguous()
        temp = destination / f".{shard_name}.gsq.tmp"
        save_file(tensors, temp, metadata=metadata)
        os.replace(temp, destination / shard_name)

        with safe_open(str(destination / shard_name), framework="pt", device="cpu") as output:
            for name, original in tensors.items():
                actual = output.get_tensor(name)
                if not torch.equal(actual, original):
                    raise AssertionError(f"checkpoint rewrite changed tensor unexpectedly: {name}")
                if name not in shard_replacements:
                    unchanged_tensors += 1

    installed = load_file(state_path)
    for name, expected in installed.items():
        shard = destination / index["weight_map"][name]
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            if not torch.equal(handle.get_tensor(name), expected):
                raise AssertionError(f"installed GSQ tensor does not match hard state: {name}")
    return len(installed), unchanged_tensors, sorted(by_shard)


def evaluation_documents(args, tokenizer):
    if args.token_offset < 0:
        raise ValueError("token cache offset must be nonnegative")
    if args.token_cache is not None and args.token_cache.is_file():
        cached = load_file(args.token_cache)
        if "tokens" in cached:
            rows = cached["tokens"][args.token_offset:args.token_offset + args.evaluation_samples]
            documents = [row[:args.sequence_length].long() for row in rows]
        else:
            documents = [
                cached[f"tokens.{index:05d}"]
                for index in range(args.token_offset, args.token_offset + args.evaluation_samples)
            ]
    else:
        documents = fineweb_chunks(
            tokenizer,
            count=args.evaluation_samples,
            length=args.sequence_length,
            seed=args.evaluation_seed,
        )
        if args.token_cache is not None:
            args.token_cache.parent.mkdir(parents=True, exist_ok=True)
            save_file(
                {f"tokens.{index:05d}": tokens for index, tokens in enumerate(documents)},
                args.token_cache,
                metadata={
                    "dataset": "HuggingFaceFW/fineweb-edu/sample-10BT",
                    "role": "report_only_evaluation",
                    "seed": str(args.evaluation_seed),
                    "sequence_length": str(args.sequence_length),
                },
            )
    if len(documents) != args.evaluation_samples:
        raise ValueError("evaluation token cache contains too few rows")
    if any(tokens.shape != (args.sequence_length,) for tokens in documents):
        raise ValueError("evaluation token cache has the wrong sequence length")
    return documents


@torch.inference_mode()
def evaluate(teacher, student, documents, device):
    totals = {"forward_kld": 0., "logit_mse": 0., "cross_entropy": 0., "top1_agreement": 0.}
    token_count = 0
    output_hash = hashlib.sha256()
    for tokens in documents:
        input_ids = tokens[None].to(device)
        teacher_logits = teacher(input_ids=input_ids, use_cache=False).logits[:, :-1].float()
        student_logits = student(input_ids=input_ids, use_cache=False).logits[:, :-1].float()
        count = student_logits.shape[1]
        teacher_log_prob = F.log_softmax(teacher_logits, dim=-1)
        student_log_prob = F.log_softmax(student_logits, dim=-1)
        totals["forward_kld"] += float(
            (teacher_log_prob.exp() * (teacher_log_prob - student_log_prob)).sum(-1).sum()
        )
        totals["logit_mse"] += float(F.mse_loss(student_logits, teacher_logits, reduction="sum"))
        totals["cross_entropy"] += float(F.cross_entropy(
            student_logits.reshape(-1, student_logits.shape[-1]),
            input_ids[:, 1:].reshape(-1),
            reduction="sum",
        ))
        totals["top1_agreement"] += float(
            student_logits.argmax(-1).eq(teacher_logits.argmax(-1)).sum()
        )
        output_hash.update(student_logits.half().cpu().numpy().tobytes())
        token_count += count
        del teacher_logits, student_logits, teacher_log_prob, student_log_prob
    result = {
        "forward_kld": totals["forward_kld"] / token_count,
        "logit_mse": totals["logit_mse"] / token_count,
        "cross_entropy": totals["cross_entropy"] / token_count,
        "top1_agreement": totals["top1_agreement"] / token_count,
        "tokens": token_count,
        "logit_sha256": output_hash.hexdigest(),
    }
    result["perplexity"] = math.exp(result["cross_entropy"])
    return result


def strict_global_guard(incumbent, candidate):
    """Require every dense-teacher endpoint to avoid regression."""
    improvements = {
        key: 100. * (incumbent[key] - candidate[key]) / incumbent[key]
        for key in ("forward_kld", "logit_mse", "cross_entropy", "perplexity")
    }
    improvements["top1_agreement_points"] = 100. * (
        candidate["top1_agreement"] - incumbent["top1_agreement"]
    )
    accepted = all(value >= 0. for value in improvements.values())
    return accepted, improvements


def main():
    args = parse_args()
    state_meta = state_metadata(args.state)
    tokenizer = AutoTokenizer.from_pretrained(args.dense_model, local_files_only=True)
    documents = evaluation_documents(args, tokenizer)
    hashes = [digest(tokens) for tokens in documents]
    train_hashes = set(filter(None, state_meta.get("train_token_hashes", "").split(",")))
    validation_hashes = set(filter(None, state_meta.get("validation_token_hashes", "").split(",")))
    qk_hashes = set(filter(None, state_meta.get("qk_metric_token_hashes", "").split(",")))
    actual_hashes = set(hashes)
    overlap = (train_hashes | validation_hashes | qk_hashes) & actual_hashes
    if args.role == "report_only" and overlap:
        raise RuntimeError(f"report-only evaluation overlaps GSQ selection data: {sorted(overlap)}")
    if args.role == "validation":
        forbidden = (train_hashes | qk_hashes) & actual_hashes
        if forbidden or not actual_hashes <= validation_hashes:
            raise RuntimeError("validation evaluation does not match the declared heldout selection split")
    if args.incumbent_evaluation is not None and args.role != "validation":
        raise ValueError("an incumbent evaluation is only valid for the validation role")

    # Validate the split before creating the copy-on-write checkpoint. Invalid
    # selection data must not leave behind an apparently usable candidate.
    installed, unchanged, shards = install_state(
        args.source_checkpoint.resolve(),
        args.output_checkpoint.resolve(),
        args.state.resolve(),
    )

    device = torch.device("cuda")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.dense_model,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device).eval()
    baseline_wrapper = GPTQModel.load(
        str(args.source_checkpoint),
        backend=BACKEND.QVQ,
        dtype=torch.float16,
        device_map={"": "cuda:0"},
        attn_implementation="eager",
        local_files_only=True,
    )
    baseline = baseline_wrapper.model.eval()
    baseline_metrics = evaluate(teacher, baseline, documents, device)
    del baseline, baseline_wrapper
    torch.cuda.empty_cache()

    candidate_wrapper = GPTQModel.load(
        str(args.output_checkpoint),
        backend=BACKEND.QVQ,
        dtype=torch.float16,
        device_map={"": "cuda:0"},
        attn_implementation="eager",
        local_files_only=True,
    )
    candidate = candidate_wrapper.model.eval()
    candidate_metrics = evaluate(teacher, candidate, documents, device)
    improvements = {}
    for key in ("forward_kld", "logit_mse", "cross_entropy", "perplexity"):
        improvements[key] = 100. * (baseline_metrics[key] - candidate_metrics[key]) / baseline_metrics[key]
    improvements["top1_agreement_points"] = 100. * (
        candidate_metrics["top1_agreement"] - baseline_metrics["top1_agreement"]
    )

    global_guard = None
    if args.incumbent_evaluation is not None:
        incumbent_payload = json.loads(args.incumbent_evaluation.read_text())
        if incumbent_payload.get("role") != "validation_evaluation":
            raise ValueError("incumbent must be a validation evaluation")
        if incumbent_payload.get("evaluation_hashes") != hashes:
            raise ValueError("incumbent and candidate validation rows differ")
        incumbent_metrics = incumbent_payload["candidate"]
        accepted, against_incumbent = strict_global_guard(incumbent_metrics, candidate_metrics)
        global_guard = {
            "incumbent_evaluation": str(args.incumbent_evaluation.resolve()),
            "accepted": accepted,
            "relative_improvement_percent": against_incumbent,
        }

    payload = {
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "candidate_checkpoint": str(args.output_checkpoint.resolve()),
        "state": str(args.state.resolve()),
        "installed_tensors": installed,
        "unchanged_tensors_verified_in_rewritten_shards": unchanged,
        "rewritten_shards": shards,
        "runtime": {"backend": "QVQ", "device": torch.cuda.get_device_name()},
        "dataset": "HuggingFaceFW/fineweb-edu/sample-10BT",
        "role": f"{args.role}_evaluation",
        "evaluation_seed": args.evaluation_seed,
        "evaluation_hashes": hashes,
        "strict_disjoint_from_train_qk_and_validation": not bool(overlap),
        "validation_membership_verified": args.role == "validation",
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "relative_improvement_percent": improvements,
        "global_guard": global_guard,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
