#!/usr/bin/env python3
"""Validate one localized P4 proposal behind a real packed QVQ live prefix."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import gptqmodel.quantization.qvq as qvq_module
from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.quantization.qvq import (
    QVQLinearQuantizationResult,
    quantize_qvq_linear,
    rht_reconstruct_weight,
)
from gptqmodel.quantization.qvq_spectral import (
    realized_propagation_product,
    select_crossfit_propagation_shaped_svd,
    select_propagation_shaped_svd,
)
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
    parser.add_argument(
        "--gradient-row-offset",
        type=int,
        help="Optional disjoint rows used only to generate the P9 teacher-KL gradient.",
    )
    parser.add_argument(
        "--gradient-rows",
        type=int,
        help="Number of optional disjoint P9 gradient-generation rows (defaults to search rows).",
    )
    parser.add_argument(
        "--gradient-folds",
        type=int,
        default=1,
        help="Interleaved independent folds used for robust propagation-gradient consensus.",
    )
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
    parser.add_argument(
        "--gradient-ranked-direct",
        action="store_true",
        help="Rank localized candidates by one full-horizon teacher-KL gradient at the serialized baseline.",
    )
    parser.add_argument(
        "--gradient-shaped-spectral",
        action="store_true",
        help="Generate localized candidates from complete signed spectral atoms favored by a disjoint teacher-KL gradient.",
    )
    parser.add_argument(
        "--require-favorable-realized-gradient",
        action="store_true",
        help="Skip full replay when the actual serialized candidate delta has a non-favorable gradient product.",
    )
    parser.add_argument("--replay-folds", type=int, default=1)
    parser.add_argument("--topn-regression-limit", type=float, default=0.0025)
    parser.add_argument("--minimum-relative-kl-improvement", type=float, default=0.001)
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Serialize ordinary V2B2-P32+YAQA without localized P4 candidate generation.",
    )
    parser.add_argument(
        "--complete-family-selection",
        action="store_true",
        help=(
            "Independently quantize canonical V2+YAQA and all three fixed B2-P32 families, then select only by "
            "cross-fitted full-horizon replay plus disjoint confirmation."
        ),
    )
    return parser


def _fixed_v2b2_yaqa_family(
    original,
    family_id: int,
):
    """Force one independently encoded YAQA family; family zero is exact canonical V2."""

    if isinstance(family_id, bool) or not isinstance(family_id, int) or family_id not in range(4):
        raise ValueError("complete-family YAQA family ID must be 0, 1, 2, or 3")

    def fixed_family(*args, **kwargs):
        if family_id:
            kwargs["family_mode"] = "fixed_block_ldlq"
            kwargs["sample_strategy"] = "full"
            kwargs["block_family_id"] = family_id
            return original(*args, **kwargs)

        inner_weight, input_hessian, output_hessian, codebook_library = args[:4]
        allowed = {
            name: kwargs[name]
            for name in (
                "bits",
                "tile_rows",
                "tile_cols",
                "trellis_batch_size",
                "tail_biting_candidates",
                "factorization",
                "telemetry",
                "_incremental_cuda_feedback",
                "_trusted_inputs",
            )
            if name in kwargs
        }
        canonical_weight, canonical_states = qvq_module.yaqa_inner(
            inner_weight,
            input_hessian,
            output_hessian,
            codebook_library[0],
            **allowed,
        )
        selectors = torch.zeros(
            canonical_states.shape[0] * 8,
            dtype=torch.uint8,
            device=canonical_states.device,
        )
        inactive_family = torch.ones((1,), dtype=torch.uint8, device=canonical_states.device)
        return canonical_weight, canonical_states, selectors, inactive_family

    return fixed_family


def _serialized_v2b2_weight(result: QVQLinearQuantizationResult, *, bits: float) -> torch.Tensor:
    """Reconstruct the exact B2-P32 checkpoint payload instead of trusting its staging tensor."""

    module = QVQLinear(
        bits=bits,
        in_features=result.SU.numel(),
        out_features=result.SV.numel(),
        tensors={name: value.detach().clone() for name, value in result.serialized_tensors().items()},
        dtype=result.weight.dtype,
        out_dtype=result.weight.dtype,
        vector_size=2,
        trellis_window=16,
        bank_count=2,
        v2b2_p32=True,
    ).eval()
    module.post_init()
    inner = module.get_inner_weight_tensor(dtype=torch.float32)
    reconstructed = rht_reconstruct_weight(inner, module.SU, module.SV).to(result.weight.dtype)
    maximum_error = float((reconstructed - result.weight).abs().max().item())
    if maximum_error > 1e-6:
        raise RuntimeError(
            f"serialized B2-P32 reconstruction differs from the quantizer result by {maximum_error:.9g}"
        )
    return reconstructed


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


def _teacher_kl_weight_gradient(
    model: torch.nn.Module,
    target: torch.nn.Linear,
    rows: Sequence[Mapping[str, torch.Tensor]],
    teacher: Sequence[torch.Tensor],
    candidate_weight: torch.Tensor,
) -> torch.Tensor:
    """Differentiate token-mean teacher KL through the live quantized suffix.

    A forward hook substitutes an FP32 leaf for the target weight. This avoids
    accumulating the propagation gradient in an FP16 model parameter while
    preserving the model's native forward dtype and complete downstream graph.
    Rows are differentiated one at a time so full-sequence activations are
    released between backward passes.
    """

    if len(rows) != len(teacher) or not rows:
        raise ValueError("P9 gradient rows and teacher logits must be non-empty and aligned")
    if candidate_weight.shape != target.weight.shape or not candidate_weight.is_floating_point():
        raise ValueError("P9 candidate weight must match the target linear")
    device = target.weight.device
    candidate = candidate_weight.detach().to(device=device, dtype=torch.float32).requires_grad_(True)
    total_tokens = sum(int(logits.shape[0] * logits.shape[1]) for logits in teacher)
    if total_tokens < 1:
        raise ValueError("P9 teacher logits must contain at least one next-token target")
    requires_grad = tuple((parameter, parameter.requires_grad) for parameter in model.parameters())
    for parameter, _ in requires_grad:
        parameter.requires_grad_(False)

    def substitute_weight(module, args, _output):
        hidden = args[0]
        return F.linear(hidden, candidate.to(hidden.dtype), module.bias)

    handle = target.register_forward_hook(substitute_weight)
    accumulated = torch.zeros_like(candidate)
    try:
        for row, dense_logits in zip(rows, teacher, strict=True):
            student_logits = model(**row).logits[:, :-1].to(torch.float32)
            teacher_log_probs = F.log_softmax(dense_logits.to(device=device, dtype=torch.float32), dim=-1)
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            loss = F.kl_div(
                student_log_probs,
                teacher_log_probs,
                reduction="sum",
                log_target=True,
            ) / total_tokens
            (row_gradient,) = torch.autograd.grad(loss, candidate, create_graph=False)
            accumulated.add_(row_gradient)
    finally:
        handle.remove()
        for parameter, enabled in requires_grad:
            parameter.requires_grad_(enabled)
    return accumulated.detach()


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


def _qvq_module_from_result(
    current: torch.nn.Linear,
    *,
    target_name: str,
    result: QVQLinearQuantizationResult,
    bits: float,
) -> QVQLinear:
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
    return replacement


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
    replacement = _qvq_module_from_result(current, target_name=target_name, result=result, bits=bits)
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
    replayed_candidates = [
        {
            "name": name,
            "generator": record.get("generator", "spectral_push"),
            "rank": record.get("rank"),
            "alpha": record.get("alpha"),
            "tile": record.get("tile"),
            "segment": record.get("segment"),
            **(
                {
                    "original_yaqa_loss": record["loss"],
                    "original_yaqa_relative_improvement": record.get("relative_improvement"),
                }
                if "loss" in record
                else {}
            ),
            **(
                {
                    "module_search_loss": record["search_loss"],
                    "module_search_relative_improvement": record.get("search_relative_improvement"),
                }
                if "search_loss" in record
                else {}
            ),
            "propagated_first_order": record.get("propagated_first_order"),
            "replay_score": record["replay_score"],
            "selected": record.get("selected") is True,
        }
        for name, record in candidates.items()
        if "replay_score" in record
    ]
    return {
        "proposed": "baseline" in callback_report and "proposal" in callback_report,
        "accepted": bool(callback_report.get("accepted", False)),
        "selector_churn": result.yaqa_spectral_selector_churn,
        "family_changed": result.yaqa_spectral_family_changed,
        "selected_changes": len(selected_candidates),
        "selected_candidates": selected_candidates,
        "replayed_candidates": replayed_candidates,
    }


def main() -> None:
    args = _parser().parse_args()
    if args.complete_family_selection and args.baseline_only:
        raise ValueError("complete-family selection and baseline-only mode are mutually exclusive")
    if args.complete_family_selection and (args.gradient_ranked_direct or args.gradient_shaped_spectral):
        raise ValueError("complete-family selection cannot be combined with localized gradient searches")
    if args.gradient_ranked_direct and args.gradient_shaped_spectral:
        raise ValueError("gradient-ranked and gradient-shaped spectral searches are mutually exclusive")
    gradient_enabled = args.gradient_ranked_direct or args.gradient_shaped_spectral
    splits = {
        "search": (args.search_row_offset, args.search_rows),
        "confirmation": (args.confirmation_row_offset, args.confirmation_rows),
        "evaluation": (args.evaluation_row_offset, args.evaluation_rows),
    }
    if args.gradient_row_offset is not None or args.gradient_rows is not None:
        if args.gradient_row_offset is None or args.gradient_rows is None:
            raise ValueError("--gradient-row-offset and --gradient-rows must be provided together")
        splits["gradient"] = (args.gradient_row_offset, args.gradient_rows)
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
    if args.gradient_ranked_direct and args.direct_replay_candidates < 1:
        raise ValueError("gradient-ranked direct search requires at least one direct replay candidate")
    if args.gradient_shaped_spectral and args.replay_candidates < 1:
        raise ValueError("gradient-shaped spectral search requires at least one replay candidate")
    if args.gradient_shaped_spectral and "gradient" not in splits:
        raise ValueError("gradient-shaped spectral search requires an explicit disjoint gradient split")
    if args.gradient_folds < 1 or (
        args.gradient_rows is not None and args.gradient_folds > args.gradient_rows
    ):
        raise ValueError("gradient folds must be between one and the gradient row count")
    if args.require_favorable_realized_gradient and not gradient_enabled:
        raise ValueError("realized-gradient gating requires a propagated-gradient strategy")
    if args.replay_folds < 1 or args.replay_folds > args.search_rows:
        raise ValueError("replay fold count must be between one and the search row count")
    if not 0 <= args.minimum_relative_kl_improvement < 1:
        raise ValueError("minimum relative KL improvement must be in [0, 1)")
    if args.baseline_only and args.replay_candidates:
        raise ValueError("baseline-only mode cannot enable full-horizon candidate replay")
    if args.complete_family_selection and args.replay_candidates:
        raise ValueError("complete-family selection enumerates its four arms and does not use --replay-candidates")
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
    gradient_teacher = None
    confirmation_teacher = None
    if not args.baseline_only:
        print("Capturing live-prefix search inputs with target-boundary early stop", flush=True)
        propagated_inputs = _capture_target_inputs(student_model, target, row_sets["search"])
        propagated_target = F.linear(propagated_inputs, source_weight, source_bias)
        if args.replay_candidates or args.complete_family_selection:
            print("Caching dense teacher logits for full-horizon search reranking", flush=True)
            search_teacher = _teacher_logits(dense_model, row_sets["search"])
            if gradient_enabled:
                gradient_rows = row_sets.get("gradient", row_sets["search"])
                print("Caching dense teacher logits for disjoint propagation-gradient generation", flush=True)
                gradient_teacher = _teacher_logits(dense_model, gradient_rows)
        print("Caching dense teacher logits for confirmation and untouched evaluation", flush=True)
        confirmation_teacher = _teacher_logits(dense_model, row_sets["confirmation"])
    else:
        print("Skipping localized search for the ordinary YAQA baseline oracle", flush=True)
    evaluation_teacher = _teacher_logits(dense_model, row_sets["evaluation"])
    callback_report: dict[str, object] = {}
    callback_weights: dict[str, torch.Tensor] = {}
    replay_scores: list[dict[str, object]] = []
    gradient_report: dict[str, object] = {
        "enabled": gradient_enabled,
        "strategy": (
            "shaped_spectral"
            if args.gradient_shaped_spectral
            else "ranked_direct"
            if args.gradient_ranked_direct
            else "disabled"
        ),
    }
    full_horizon_gradient_weight: torch.Tensor | None = None
    gradient_baseline_weight: torch.Tensor | None = None
    full_horizon_gradient_folds: tuple[torch.Tensor, ...] = ()
    replay_baseline_fold_kl: list[float] | None = None
    replay_row_folds = tuple(
        tuple(row_sets["search"][fold_index :: args.replay_folds])
        for fold_index in range(args.replay_folds)
    )

    def full_horizon_candidate_score(candidate: torch.Tensor) -> float:
        nonlocal replay_baseline_fold_kl
        assert search_teacher is not None
        if args.require_favorable_realized_gradient:
            if full_horizon_gradient_weight is None or gradient_baseline_weight is None:
                raise RuntimeError("realized-gradient gate was invoked before gradient generation")
            if not torch.equal(candidate, gradient_baseline_weight):
                fold_gradients = full_horizon_gradient_folds or (full_horizon_gradient_weight,)
                products = [
                    float(realized_propagation_product(gradient, candidate, gradient_baseline_weight).item())
                    for gradient in fold_gradients
                ]
                if any(product >= 0 for product in products):
                    replay_scores.append(
                        {
                            "score": None,
                            "gated": "non-favorable realized gradient",
                            "realized_propagated_first_order_folds": products,
                        }
                    )
                    return math.inf
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

    target_parent_name, _, target_child_name = args.target.rpartition(".")
    target_parent = student_model.get_submodule(target_parent_name)

    def install_target(module: torch.nn.Module) -> None:
        setattr(target_parent, target_child_name, module)

    def full_horizon_serialized_score(result: QVQLinearQuantizationResult) -> float:
        """Replay one actual packed QVQLinear and restore the original dense target atomically."""

        nonlocal replay_baseline_fold_kl
        assert search_teacher is not None
        replacement = _qvq_module_from_result(target, target_name=args.target, result=result, bits=args.bits)
        install_target(replacement)
        try:
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
                    "runtime": "packed_qvqlinear",
                }
            )
            return score
        finally:
            install_target(target)

    def full_horizon_candidate_gradient(candidate: torch.Tensor) -> torch.Tensor:
        nonlocal full_horizon_gradient_folds, full_horizon_gradient_weight, gradient_baseline_weight
        assert search_teacher is not None
        gradient_rows = row_sets.get("gradient", row_sets["search"])
        gradient_targets = gradient_teacher if gradient_teacher is not None else search_teacher
        started_gradient = time.perf_counter()
        try:
            gradients = []
            fold_tokens = []
            for fold_index in range(args.gradient_folds):
                fold_rows = tuple(gradient_rows[fold_index :: args.gradient_folds])
                fold_targets = tuple(gradient_targets[fold_index :: args.gradient_folds])
                gradients.append(
                    _teacher_kl_weight_gradient(
                        student_model,
                        target,
                        fold_rows,
                        fold_targets,
                        candidate,
                    )
                )
                fold_tokens.append(sum(int(logits.shape[0] * logits.shape[1]) for logits in fold_targets))
            total_tokens = sum(fold_tokens)
            gradient = sum(
                fold_gradient * (tokens / total_tokens)
                for fold_gradient, tokens in zip(gradients, fold_tokens, strict=True)
            )
            gradient_report.update(
                {
                    "seconds": time.perf_counter() - started_gradient,
                    "valid_tokens": sum(int(logits.shape[0] * logits.shape[1]) for logits in gradient_targets),
                    "l2_norm": float(torch.linalg.vector_norm(gradient.to(torch.float32)).item()),
                    "max_abs": float(gradient.abs().max().item()),
                    "finite": bool(torch.isfinite(gradient).all()),
                    "folds": [
                        {
                            "valid_tokens": tokens,
                            "l2_norm": float(torch.linalg.vector_norm(fold_gradient.to(torch.float32)).item()),
                            "max_abs": float(fold_gradient.abs().max().item()),
                        }
                        for fold_gradient, tokens in zip(gradients, fold_tokens, strict=True)
                    ],
                }
            )
            full_horizon_gradient_weight = gradient.detach().clone()
            gradient_baseline_weight = candidate.detach().clone()
            full_horizon_gradient_folds = tuple(fold_gradient.detach().clone() for fold_gradient in gradients)
            return gradient
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

    def serialized_confirmation_callback(
        proposal: QVQLinearQuantizationResult,
        rollback: QVQLinearQuantizationResult,
    ) -> bool:
        assert confirmation_teacher is not None

        def metrics_for(result: QVQLinearQuantizationResult) -> dict[str, object]:
            install_target(_qvq_module_from_result(target, target_name=args.target, result=result, bits=args.bits))
            try:
                return _compare_logits(student_model, row_sets["confirmation"], confirmation_teacher)
            finally:
                install_target(target)

        baseline_metrics = metrics_for(rollback)
        proposal_metrics = metrics_for(proposal)
        accepted = _passes_confirmation(
            baseline_metrics,
            proposal_metrics,
            topn_regression_limit=args.topn_regression_limit,
            minimum_relative_kl_improvement=args.minimum_relative_kl_improvement,
        )
        callback_report.update(
            {
                "baseline": baseline_metrics,
                "proposal": proposal_metrics,
                "accepted": accepted,
                "runtime": "packed_qvqlinear",
            }
        )
        return accepted

    mode = (
        "ordinary YAQA baseline"
        if args.baseline_only
        else "complete serialized YAQA family selection"
        if args.complete_family_selection
        else "localized P4"
    )
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
    if not args.baseline_only and not args.complete_family_selection:
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
            propagated_candidate_gradient=(
                full_horizon_candidate_gradient if gradient_enabled else None
            ),
        )
    original_localized_refiner = qvq_module.yaqa_localized_spectral_refine_v2b2_p32

    def gradient_shaped_localized_refiner(*refiner_args, **refiner_kwargs):
        replay_gradient = refiner_kwargs.get("replay_gradient")
        if replay_gradient is None:
            gradient_report["shaping_status"] = "gradient unavailable; exact localized baseline retained"
            return refiner_args[4]
        input_hessian = refiner_args[1].to(torch.float32)
        output_hessian = refiner_args[2].to(torch.float32)
        input_root = torch.linalg.cholesky(input_hessian)
        output_root = torch.linalg.cholesky(output_hessian)

        from gptqmodel.eora import eora as eora_module

        original_svd = eora_module._eora_compute_svd

        def propagation_shaped_svd(matrix: torch.Tensor, rank: int, algo: str = "lowrank"):
            left, singular, right_h = original_svd(matrix, rank, algo=algo)
            spectral_device = input_root.device
            spectral_args = (
                input_root,
                output_root,
                left.to(device=spectral_device, dtype=torch.float32),
                singular.to(device=spectral_device, dtype=torch.float32),
                right_h.to(device=spectral_device, dtype=torch.float32),
            )
            if len(full_horizon_gradient_folds) > 1:
                (
                    selected_left,
                    selected_singular,
                    selected_right_h,
                    fold_products,
                    products,
                    indices,
                ) = select_crossfit_propagation_shaped_svd(
                    *spectral_args,
                    torch.stack(full_horizon_gradient_folds).to(device=spectral_device, dtype=torch.float32),
                    maximum_modes=rank,
                )
                gradient_report["spectral_mode_product_folds"] = fold_products.detach().cpu().tolist()
            else:
                selected_left, selected_singular, selected_right_h, products, indices = select_propagation_shaped_svd(
                    *spectral_args,
                    replay_gradient.to(device=spectral_device, dtype=torch.float32),
                    maximum_modes=rank,
                )
            gradient_report.update(
                {
                    "shaping_status": "applied" if indices.numel() else "no favorable signed modes",
                    "spectral_mode_products": products.detach().cpu().tolist(),
                    "selected_original_mode_indices": indices.detach().cpu().tolist(),
                    "favorable_mode_count": int(indices.numel()),
                }
            )
            return (
                selected_left.to(device=matrix.device, dtype=left.dtype),
                selected_singular.to(device=matrix.device, dtype=singular.dtype),
                selected_right_h.to(device=matrix.device, dtype=right_h.dtype),
            )

        with patch.object(eora_module, "_eora_compute_svd", propagation_shaped_svd):
            return original_localized_refiner(*refiner_args, **refiner_kwargs)

    localized_context = (
        patch.object(
            qvq_module,
            "yaqa_localized_spectral_refine_v2b2_p32",
            gradient_shaped_localized_refiner,
        )
        if args.gradient_shaped_spectral
        else nullcontext()
    )
    complete_family_report: list[dict[str, object]] = []
    selected_family: int | None = None
    if args.complete_family_selection:
        assert search_teacher is not None and confirmation_teacher is not None
        original_family_quantizer = qvq_module.yaqa_inner_v2b2_p32
        family_results: list[QVQLinearQuantizationResult] = []
        for family_id in range(4):
            family_started = time.perf_counter()
            with patch.object(
                qvq_module,
                "yaqa_inner_v2b2_p32",
                _fixed_v2b2_yaqa_family(original_family_quantizer, family_id),
            ):
                family_result = quantize_qvq_linear(
                    source_weight,
                    input_hessians[args.target].to(device),
                    bits=args.bits,
                    **quantization_kwargs,
                )
            serialized_weight = _serialized_v2b2_weight(family_result, bits=args.bits).to(device)
            replay_index = len(replay_scores)
            replay_score = full_horizon_serialized_score(family_result)
            family_results.append(family_result)
            selectors = family_result.bank_ids
            nonzero = 0.0 if selectors is None else float((selectors != 0).float().mean().item())
            complete_family_report.append(
                {
                    "family_id": family_id,
                    "seconds": time.perf_counter() - family_started,
                    "serialized_max_abs_error": float((serialized_weight - family_result.weight).abs().max().item()),
                    "proxy_loss": float(family_result.proxy_loss.item()),
                    "kronecker_proxy_loss": (
                        None
                        if family_result.kronecker_proxy_loss is None
                        else float(family_result.kronecker_proxy_loss.item())
                    ),
                    "selector_nonzero_fraction": nonzero,
                    "fallback_to_v2": family_result.yaqa_bank_fallback_to_v2,
                    "replay": replay_scores[replay_index],
                    "replay_score": replay_score,
                }
            )
            print(
                f"Family {family_id}: score={replay_score:.8f} selectors={nonzero:.4f} "
                f"time={complete_family_report[-1]['seconds']:.2f}s",
                flush=True,
            )

        result = family_results[0]
        eligible = [
            index
            for index in range(1, 4)
            if complete_family_report[index]["replay_score"] < 1 - args.minimum_relative_kl_improvement
        ]
        selected_family = (
            min(eligible, key=lambda index: complete_family_report[index]["replay_score"])
            if eligible
            else 0
        )
        if selected_family and serialized_confirmation_callback(
            family_results[selected_family],
            family_results[0],
        ):
            result = family_results[selected_family]
        else:
            selected_family = 0
            if not callback_report:
                callback_report = {
                    "accepted": False,
                    "reason": "no complete family improved every search fold by the required margin",
                }
        for record in complete_family_report:
            record["selected"] = record["family_id"] == selected_family
    else:
        with localized_context:
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

    rollback_result = family_results[0] if args.complete_family_selection else None
    rollback_weight = (
        rollback_result.weight
        if rollback_result is not None
        else callback_weights.get("rollback", result.weight)
    ).detach().float().clone()
    with torch.no_grad():
        target.weight.copy_(rollback_weight.to(device=device, dtype=target.weight.dtype))
    rollback_evaluation = _compare_logits(student_model, row_sets["evaluation"], evaluation_teacher)
    with torch.no_grad():
        target.weight.copy_(result.weight.to(device=device, dtype=target.weight.dtype))
    selected_dense_evaluation = (
        rollback_evaluation
        if args.complete_family_selection and selected_family == 0
        else _compare_logits(student_model, row_sets["evaluation"], evaluation_teacher)
    )
    with torch.no_grad():
        target.weight.copy_(rollback_weight.to(dtype=target.weight.dtype))
    rollback_packed_evaluation = None
    if rollback_result is not None:
        install_target(_qvq_module_from_result(target, target_name=args.target, result=rollback_result, bits=args.bits))
        rollback_packed_evaluation = _compare_logits(student_model, row_sets["evaluation"], evaluation_teacher)
        install_target(target)
    _replace_target_with_result(student_model, target_name=args.target, result=result, bits=args.bits)
    selected_packed_evaluation = (
        rollback_packed_evaluation
        if args.complete_family_selection and selected_family == 0
        else _compare_logits(student_model, row_sets["evaluation"], evaluation_teacher)
    )

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
            "gradient_ranked_direct": args.gradient_ranked_direct,
            "gradient_shaped_spectral": args.gradient_shaped_spectral,
            "gradient_folds": args.gradient_folds,
            "require_favorable_realized_gradient": args.require_favorable_realized_gradient,
            "replay_folds": args.replay_folds,
            "topn_regression_limit": args.topn_regression_limit,
            "minimum_relative_kl_improvement": args.minimum_relative_kl_improvement,
            "baseline_only": args.baseline_only,
            "complete_family_selection": args.complete_family_selection,
        },
        "quantization_seconds": quantization_seconds,
        "search_valid_tokens": 0 if propagated_inputs is None else int(propagated_inputs.shape[0]),
        "localized": _localized_summary(result, callback_report),
        "full_horizon_search": {
            "enabled": bool(args.replay_candidates or args.complete_family_selection),
            "evaluations": replay_scores,
        },
        "complete_family_selection": complete_family_report,
        "full_horizon_gradient": gradient_report,
        "confirmation": callback_report,
        "evaluation": {
            "rollback_dense": rollback_evaluation,
            "rollback_packed": rollback_packed_evaluation,
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
