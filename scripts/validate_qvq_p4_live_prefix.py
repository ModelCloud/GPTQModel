#!/usr/bin/env python3
"""Validate one localized P4 proposal behind a real packed QVQ live prefix."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.quantization.qvq import QVQLinearQuantizationResult, quantize_qvq_linear
from scripts.analyze_gptq_low_bit_grid import load_nm_evaluation_batch, tensor_metrics
from scripts.compare_qvq_codecs_llama_qkvo import (
    _install_qvq_prefix_artifact,
    _load_qvq_prefix_artifact,
    _load_yaqa_factor_cache,
    _save_qvq_prefix_artifact,
    _unpadded_evaluation_rows,
    _WeightedMetricAccumulator,
)


class _ForwardCaptured(RuntimeError):
    """Private control-flow exception used to stop at the target projection."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--prefix-artifact",
        type=Path,
        nargs="+",
        required=True,
        help="One or more compatible packed prefix artifacts installed atomically in the listed order.",
    )
    parser.add_argument("--yaqa-factor-cache", type=Path, required=True)
    parser.add_argument("--yaqa-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-artifact", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--target", default="model.layers.1.self_attn.q_proj")
    parser.add_argument("--bits", type=float, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--search-row-offset", type=int, default=1602)
    parser.add_argument("--search-rows", type=int, default=8)
    parser.add_argument("--confirmation-row-offset", type=int, default=1610)
    parser.add_argument("--confirmation-rows", type=int, default=8)
    parser.add_argument("--evaluation-row-offset", type=int, default=1618)
    parser.add_argument("--evaluation-rows", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--trellis-batch-size", type=int, default=8)
    parser.add_argument("--ranks", nargs="+", type=int, default=(8, 16, 32))
    parser.add_argument("--alphas", nargs="+", type=float, default=(0.25, 0.5, 1.0))
    parser.add_argument("--max-segments", type=int, default=8)
    parser.add_argument("--max-changes", type=int, default=1)
    parser.add_argument("--replay-candidates", type=int, default=0)
    parser.add_argument("--direct-replay-candidates", type=int, default=0)
    parser.add_argument("--replay-folds", type=int, default=1)
    parser.add_argument("--topn-regression-limit", type=float, default=0.0025)
    parser.add_argument("--minimum-relative-kl-improvement", type=float, default=0.001)
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Serialize ordinary V2B2-P32+YAQA without localized P4 candidate generation.",
    )
    return parser


def _validate_disjoint_splits(splits: Mapping[str, tuple[int, int]]) -> None:
    occupied: list[tuple[int, int, str]] = []
    for name, (offset, rows) in splits.items():
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError(f"{name} row offset must be a nonnegative integer")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
            raise ValueError(f"{name} row count must be a positive integer")
        occupied.append((offset, offset + rows, name))
    occupied.sort()
    for (_, left_stop, left_name), (right_start, _, right_name) in pairwise(occupied):
        if right_start < left_stop:
            raise ValueError(f"{left_name} and {right_name} rows must be disjoint")


def _load_rows(
    tokenizer,
    *,
    dataset: Path,
    offset: int,
    rows: int,
    max_length: int | None,
    device: torch.device,
) -> tuple[tuple[dict[str, torch.Tensor], ...], dict[str, object]]:
    encoded, stats = load_nm_evaluation_batch(
        tokenizer,
        dataset_path=dataset,
        row_offset=offset,
        rows=rows,
        max_length=max_length,
    )
    return _unpadded_evaluation_rows(encoded, device), stats


@torch.inference_mode()
def _capture_target_inputs(
    model: torch.nn.Module,
    target: torch.nn.Module,
    rows: Sequence[Mapping[str, torch.Tensor]],
) -> torch.Tensor:
    """Capture live target inputs and stop before executing the target linear."""

    captured = []

    def pre_hook(_module, args):
        value = args[0]
        if not isinstance(value, torch.Tensor) or value.shape[-1] <= 0:
            raise TypeError("P4 target input hook requires a nonempty tensor")
        captured.append(value.detach().reshape(-1, value.shape[-1]).to(torch.float32))
        raise _ForwardCaptured

    handle = target.register_forward_pre_hook(pre_hook)
    try:
        for row in rows:
            try:
                model(**row)
            except _ForwardCaptured:
                continue
            raise RuntimeError("P4 target module was not reached")
    finally:
        handle.remove()
    if len(captured) != len(rows):
        raise RuntimeError("P4 target input capture did not preserve one entry per row")
    return torch.cat(captured, dim=0)


@torch.inference_mode()
def _teacher_logits(
    model: torch.nn.Module,
    rows: Sequence[Mapping[str, torch.Tensor]],
) -> tuple[torch.Tensor, ...]:
    return tuple(model(**row).logits[:, :-1].detach().float().cpu() for row in rows)


@torch.inference_mode()
def _compare_logits(
    model: torch.nn.Module,
    rows: Sequence[Mapping[str, torch.Tensor]],
    teacher: Sequence[torch.Tensor],
) -> dict[str, object]:
    if len(rows) != len(teacher):
        raise ValueError("P4 rows and teacher logits must have equal lengths")
    accumulator = _WeightedMetricAccumulator()
    for row, dense_logits in zip(rows, teacher, strict=True):
        student_logits = model(**row).logits[:, :-1].detach().float().cpu()
        flattened_dense = dense_logits.reshape(-1, dense_logits.shape[-1])
        flattened_student = student_logits.reshape(-1, student_logits.shape[-1])
        accumulator.add(
            tensor_metrics(flattened_dense, flattened_student, normalize_distribution=False, include_top10=True),
            rows=flattened_dense.shape[0],
        )
    return accumulator.result()


def _passes_confirmation(
    baseline: Mapping[str, object],
    proposal: Mapping[str, object],
    *,
    topn_regression_limit: float,
    minimum_relative_kl_improvement: float = 0.001,
) -> bool:
    if not 0 <= minimum_relative_kl_improvement < 1:
        raise ValueError("minimum relative KL improvement must be in [0, 1)")
    if not bool(proposal["finite"]):
        return False
    baseline_kl = float(baseline["kl_forward"]["mean"])
    proposal_kl = float(proposal["kl_forward"]["mean"])
    if not math.isfinite(baseline_kl) or not math.isfinite(proposal_kl) or baseline_kl < 0:
        return False
    required_improvement = baseline_kl * minimum_relative_kl_improvement
    if baseline_kl - proposal_kl < required_improvement:
        return False
    for key in ("top1_agreement", "top5_overlap", "top10_overlap"):
        baseline_value = baseline[key] if key == "top1_agreement" else baseline[key]["mean"]
        proposal_value = proposal[key] if key == "top1_agreement" else proposal[key]["mean"]
        if proposal_value < baseline_value - topn_regression_limit:
            return False
    return True


def _minimax_relative_replay_score(
    candidate_fold_kl: Sequence[float],
    baseline_fold_kl: Sequence[float],
) -> float:
    """Return the worst fold-relative KL; values below one improve every fold."""

    if len(candidate_fold_kl) != len(baseline_fold_kl) or not candidate_fold_kl:
        raise ValueError("replay fold scores must be non-empty and aligned")
    ratios = []
    for candidate, baseline in zip(candidate_fold_kl, baseline_fold_kl, strict=True):
        if not math.isfinite(candidate) or not math.isfinite(baseline) or candidate < 0 or baseline <= 0:
            return math.inf
        ratios.append(candidate / baseline)
    return max(ratios)


def _replace_target_with_result(
    model: torch.nn.Module,
    *,
    target_name: str,
    result: QVQLinearQuantizationResult,
    bits: float,
) -> QVQLinear:
    current = model.get_submodule(target_name)
    if not isinstance(current, torch.nn.Linear):
        raise TypeError("P4 selected target must replace a dense torch.nn.Linear")
    replacement = QVQLinear(
        bits=bits,
        in_features=current.in_features,
        out_features=current.out_features,
        name=target_name,
        tensors={name: value.to(current.weight.device) for name, value in result.serialized_tensors().items()},
        dtype=current.weight.dtype,
        out_dtype=current.weight.dtype,
        vector_size=2,
        trellis_window=16,
        bank_count=2,
        v2b2_p32=True,
    ).eval()
    replacement.post_init()
    parent_name, _, child_name = target_name.rpartition(".")
    setattr(model.get_submodule(parent_name), child_name, replacement)
    return replacement


def _install_prefix_artifacts(
    model: torch.nn.Module,
    *,
    paths: Sequence[Path],
    model_path: Path,
    excluded_module: str | None = None,
) -> tuple[tuple[dict[str, object], ...], dict[str, QVQLinear]]:
    """Validate and atomically install a compatible set of packed prefixes."""

    if not paths:
        raise ValueError("P4 requires at least one packed prefix artifact")
    manifests = []
    combined_tensors = {}
    combined_modules = {}
    fixed_contract = None
    resolved_model = model_path.resolve()
    for path in paths:
        manifest, tensors = _load_qvq_prefix_artifact(path)
        provenance = manifest.get("provenance")
        if not isinstance(provenance, Mapping):
            raise TypeError(f"packed prefix provenance must be an object: {path}")
        source_snapshot = provenance.get("source_snapshot")
        provenance_model = provenance.get("model")
        if source_snapshot is not None and resolved_model.name != source_snapshot:
            raise ValueError(f"packed prefix source snapshot does not match --model: {path}")
        if source_snapshot is None and provenance_model is not None and Path(str(provenance_model)).resolve() != resolved_model:
            raise ValueError(f"packed prefix model does not match --model: {path}")
        contract = tuple(manifest.get(name) for name in ("format", "bits", "vector_size", "trellis_window", "bank_count", "codebook_version"))
        if fixed_contract is None:
            fixed_contract = contract
        elif contract != fixed_contract:
            raise ValueError("packed prefix artifacts must share one exact codec contract")
        modules = manifest["modules"]
        overlap = set(combined_modules).intersection(modules)
        if overlap:
            raise ValueError(f"packed prefix artifacts contain duplicate modules: {sorted(overlap)}")
        if excluded_module is not None and excluded_module in modules:
            raise ValueError(f"P4 target `{excluded_module}` must remain dense before refinement")
        combined_modules.update(modules)
        combined_tensors.update(tensors)
        manifests.append(manifest)

    combined_manifest = dict(manifests[0])
    combined_manifest["modules"] = combined_modules
    combined_manifest["provenance"] = {"component_artifacts": [str(path.resolve()) for path in paths]}
    replacements = _install_qvq_prefix_artifact(
        model,
        manifest=combined_manifest,
        module_tensors=combined_tensors,
    )
    return tuple(manifests), replacements


def _localized_summary(
    result: QVQLinearQuantizationResult,
    callback_report: Mapping[str, object],
) -> dict[str, object]:
    """Summarize only diagnostics exported by the stable result/callback contracts."""

    candidates = result.yaqa_spectral_candidates or {}
    selected_candidates = [name for name, record in candidates.items() if record.get("selected") is True]
    return {
        "proposed": "baseline" in callback_report and "proposal" in callback_report,
        "accepted": bool(callback_report.get("accepted", False)),
        "selector_churn": result.yaqa_spectral_selector_churn,
        "family_changed": result.yaqa_spectral_family_changed,
        "selected_changes": len(selected_candidates),
        "selected_candidates": selected_candidates,
    }


def main() -> None:
    args = _parser().parse_args()
    splits = {
        "search": (args.search_row_offset, args.search_rows),
        "confirmation": (args.confirmation_row_offset, args.confirmation_rows),
        "evaluation": (args.evaluation_row_offset, args.evaluation_rows),
    }
    _validate_disjoint_splits(splits)
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if args.layers < 2:
        raise ValueError("P4 live-prefix validation requires at least two decoder layers")
    if args.replay_candidates < 0:
        raise ValueError("full-horizon replay candidate count must be nonnegative")
    if not 0 <= args.direct_replay_candidates <= args.replay_candidates:
        raise ValueError("direct replay candidate count must be between zero and replay-candidates")
    if args.replay_folds < 1 or args.replay_folds > args.search_rows:
        raise ValueError("replay fold count must be between one and the search row count")
    if not 0 <= args.minimum_relative_kl_improvement < 1:
        raise ValueError("minimum relative KL improvement must be in [0, 1)")
    if args.baseline_only and args.replay_candidates:
        raise ValueError("baseline-only mode cannot enable full-horizon candidate replay")
    torch.manual_seed(args.seed)

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    source_layers = int(config.num_hidden_layers)
    config.num_hidden_layers = args.layers
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Loading dense teacher and live-prefix student", flush=True)
    dense_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval().to(device)
    student_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval().to(device)

    prefix_manifests, prefix_modules = _install_prefix_artifacts(
        student_model,
        paths=args.prefix_artifact,
        model_path=args.model,
        excluded_module=args.target,
    )
    target = student_model.get_submodule(args.target)
    dense_target = dense_model.get_submodule(args.target)
    if not isinstance(target, torch.nn.Linear) or not isinstance(dense_target, torch.nn.Linear):
        raise TypeError("P4 target must remain dense before localized quantization")

    metadata_document = json.loads(args.yaqa_metadata.read_text(encoding="utf-8"))
    factor_metadata = metadata_document.get("metadata")
    if not isinstance(factor_metadata, dict):
        raise TypeError("YAQA metadata document has no metadata object")
    input_hessians, output_hessians, factor_stats = _load_yaqa_factor_cache(
        args.yaqa_factor_cache,
        expected_metadata=factor_metadata,
    )
    if args.target not in input_hessians or args.target not in output_hessians:
        raise ValueError("YAQA factor cache does not contain the P4 target")

    row_sets = {}
    row_stats = {}
    for name, (offset, count) in splits.items():
        row_sets[name], row_stats[name] = _load_rows(
            tokenizer,
            dataset=args.dataset,
            offset=offset,
            rows=count,
            max_length=args.max_length,
            device=device,
        )
    source_weight = dense_target.weight.detach().to(device=device, dtype=torch.float32)
    source_bias = None if dense_target.bias is None else dense_target.bias.detach().to(device=device, dtype=torch.float32)
    propagated_inputs = None
    propagated_target = None
    search_teacher = None
    confirmation_teacher = None
    if not args.baseline_only:
        print("Capturing live-prefix search inputs with target-boundary early stop", flush=True)
        propagated_inputs = _capture_target_inputs(student_model, target, row_sets["search"])
        propagated_target = F.linear(propagated_inputs, source_weight, source_bias)
        if args.replay_candidates:
            print("Caching dense teacher logits for full-horizon search reranking", flush=True)
            search_teacher = _teacher_logits(dense_model, row_sets["search"])
        print("Caching dense teacher logits for confirmation and untouched evaluation", flush=True)
        confirmation_teacher = _teacher_logits(dense_model, row_sets["confirmation"])
    else:
        print("Skipping localized search for the ordinary YAQA baseline oracle", flush=True)
    evaluation_teacher = _teacher_logits(dense_model, row_sets["evaluation"])
    callback_report: dict[str, object] = {}
    callback_weights: dict[str, torch.Tensor] = {}
    replay_scores: list[dict[str, object]] = []
    replay_baseline_fold_kl: list[float] | None = None
    replay_row_folds = tuple(
        tuple(row_sets["search"][fold_index :: args.replay_folds])
        for fold_index in range(args.replay_folds)
    )

    def full_horizon_candidate_score(candidate: torch.Tensor) -> float:
        nonlocal replay_baseline_fold_kl
        assert search_teacher is not None
        try:
            with torch.no_grad():
                target.weight.copy_(candidate.to(device=device, dtype=target.weight.dtype))
            fold_metrics = []
            for fold_index, fold_rows in enumerate(replay_row_folds):
                fold_teacher = tuple(search_teacher[fold_index :: args.replay_folds])
                fold_metrics.append(_compare_logits(student_model, fold_rows, fold_teacher))
            fold_kl = [float(metrics["kl_forward"]["mean"]) for metrics in fold_metrics]
            if replay_baseline_fold_kl is None:
                replay_baseline_fold_kl = fold_kl
                score = 1.0
            else:
                score = _minimax_relative_replay_score(fold_kl, replay_baseline_fold_kl)
            fold_tokens = [int(metrics["shape"][0]) for metrics in fold_metrics]
            total_tokens = sum(fold_tokens)

            def weighted(name: str) -> float:
                values = [
                    metrics[name] if name == "top1_agreement" else metrics[name]["mean"]
                    for metrics in fold_metrics
                ]
                return sum(float(value) * rows for value, rows in zip(values, fold_tokens, strict=True)) / total_tokens

            replay_scores.append(
                {
                    "score": score,
                    "fold_kl_forward": fold_kl,
                    "kl_forward": sum(value * rows for value, rows in zip(fold_kl, fold_tokens, strict=True))
                    / total_tokens,
                    "top1": weighted("top1_agreement"),
                    "top5": weighted("top5_overlap"),
                    "top10": weighted("top10_overlap"),
                }
            )
            return score
        finally:
            with torch.no_grad():
                target.weight.copy_(source_weight.to(device=device, dtype=target.weight.dtype))

    def confirmation_callback(proposal: torch.Tensor, rollback: torch.Tensor) -> bool:
        assert confirmation_teacher is not None
        callback_weights["proposal"] = proposal.detach().clone()
        callback_weights["rollback"] = rollback.detach().clone()
        with torch.no_grad():
            target.weight.copy_(rollback.to(device=device, dtype=target.weight.dtype))
        baseline_metrics = _compare_logits(student_model, row_sets["confirmation"], confirmation_teacher)
        with torch.no_grad():
            target.weight.copy_(proposal.to(device=device, dtype=target.weight.dtype))
        proposal_metrics = _compare_logits(student_model, row_sets["confirmation"], confirmation_teacher)
        accepted = _passes_confirmation(
            baseline_metrics,
            proposal_metrics,
            topn_regression_limit=args.topn_regression_limit,
            minimum_relative_kl_improvement=args.minimum_relative_kl_improvement,
        )
        callback_report.update({"baseline": baseline_metrics, "proposal": proposal_metrics, "accepted": accepted})
        with torch.no_grad():
            target.weight.copy_(rollback.to(device=device, dtype=target.weight.dtype))
        return accepted

    mode = "ordinary YAQA baseline" if args.baseline_only else "localized P4"
    print(f"Starting {mode} quantization for {args.target}", flush=True)
    started = time.perf_counter()
    quantization_kwargs = {
        "output_hessian": output_hessians[args.target].to(device),
        "seed": args.seed,
        "trellis_batch_size": args.trellis_batch_size,
        "rounding": "yaqa",
        "bank_count": 2,
        "v2b2_p32": True,
        "yaqa_v2b2_family_mode": "reselect",
    }
    if not args.baseline_only:
        quantization_kwargs.update(
            yaqa_spectral_localized=True,
            yaqa_spectral_ranks=tuple(args.ranks),
            yaqa_spectral_localized_alphas=tuple(args.alphas),
            yaqa_spectral_localized_max_segments=args.max_segments,
            yaqa_spectral_localized_max_changes=args.max_changes,
            yaqa_spectral_localized_replay_candidates=args.replay_candidates,
            yaqa_spectral_localized_direct_replay_candidates=args.direct_replay_candidates,
            propagated_inputs=propagated_inputs,
            propagated_target_output=propagated_target,
            propagated_acceptance=confirmation_callback,
            propagated_candidate_score=(full_horizon_candidate_score if args.replay_candidates else None),
        )
    result = quantize_qvq_linear(
        source_weight,
        input_hessians[args.target].to(device),
        bits=args.bits,
        **quantization_kwargs,
    )
    quantization_seconds = time.perf_counter() - started
    if args.baseline_only:
        callback_report = {"accepted": False, "reason": "baseline-only ordinary YAQA oracle"}
    elif not callback_report:
        callback_report = {"accepted": False, "reason": "localized proposal unchanged; callback not invoked"}

    rollback_weight = callback_weights.get("rollback", result.weight).detach().float().clone()
    with torch.no_grad():
        target.weight.copy_(rollback_weight.to(device=device, dtype=target.weight.dtype))
    rollback_evaluation = _compare_logits(student_model, row_sets["evaluation"], evaluation_teacher)
    with torch.no_grad():
        target.weight.copy_(result.weight.to(device=device, dtype=target.weight.dtype))
    selected_dense_evaluation = _compare_logits(student_model, row_sets["evaluation"], evaluation_teacher)
    with torch.no_grad():
        target.weight.copy_(rollback_weight.to(dtype=target.weight.dtype))
    _replace_target_with_result(student_model, target_name=args.target, result=result, bits=args.bits)
    selected_packed_evaluation = _compare_logits(student_model, row_sets["evaluation"], evaluation_teacher)

    provenance = {
        "model": str(args.model.resolve()),
        "prefix_artifacts": [str(path.resolve()) for path in args.prefix_artifact],
        "prefix_modules": sorted(prefix_modules),
        "target": args.target,
        "bits": args.bits,
        "seed": args.seed,
        "splits": splits,
        "yaqa_factor_cache": str(args.yaqa_factor_cache.resolve()),
    }
    if args.selected_artifact is not None:
        _save_qvq_prefix_artifact(
            args.selected_artifact,
            module_results={args.target: result},
            bits=args.bits,
            provenance=provenance,
        )
    report = {
        "settings": {
            **provenance,
            "source_layers": source_layers,
            "tested_layers": args.layers,
            "prefix_manifest_count": len(prefix_manifests),
            "device": str(device),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "rows": row_stats,
            "factor_stats": factor_stats,
            "ranks": list(args.ranks),
            "alphas": list(args.alphas),
            "max_segments": args.max_segments,
            "max_changes": args.max_changes,
            "replay_candidates": args.replay_candidates,
            "direct_replay_candidates": args.direct_replay_candidates,
            "replay_folds": args.replay_folds,
            "topn_regression_limit": args.topn_regression_limit,
            "minimum_relative_kl_improvement": args.minimum_relative_kl_improvement,
            "baseline_only": args.baseline_only,
        },
        "quantization_seconds": quantization_seconds,
        "search_valid_tokens": 0 if propagated_inputs is None else int(propagated_inputs.shape[0]),
        "localized": _localized_summary(result, callback_report),
        "full_horizon_search": {
            "enabled": bool(args.replay_candidates),
            "evaluations": replay_scores,
        },
        "confirmation": callback_report,
        "evaluation": {
            "rollback_dense": rollback_evaluation,
            "selected_dense": selected_dense_evaluation,
            "selected_packed": selected_packed_evaluation,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logits = selected_packed_evaluation
    print(
        f"P4 complete in {quantization_seconds:.2f}s: accepted={report['localized']['accepted']} "
        f"finalKL={logits['kl_forward']['mean']:.8f} top1={logits['top1_agreement']:.4f} "
        f"top5={logits['top5_overlap']['mean']:.4f} top10={logits['top10_overlap']['mean']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
