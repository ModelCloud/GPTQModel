#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Measure fixed rank8 corrections on a C4 validation subset; never fit on evaluation inputs."""

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_idle_preflight import add_gpu_idle_preflight_args, bootstrap_gpu_idle_preflight


def paired_summary(rows, *, samples=10000, seed=137):
    """Token-weighted metrics with paired resampling of complete documents."""
    if not rows or samples < 100:
        raise ValueError("summary requires documents and at least 100 bootstrap samples")
    tokens = sum(row["teacher"]["tokens"] for row in rows)
    if any(row["teacher"]["tokens"] <= 0 for row in rows):
        raise ValueError("each document must contain prediction tokens")
    if any(row[mode]["tokens"] != row["teacher"]["tokens"] for row in rows for mode in ("fast", "quality")):
        raise ValueError("paired modes must use identical prediction tokens")
    result = {"documents": len(rows), "tokens": tokens, "modes": {},
              "bootstrap": {"samples": samples, "seed": seed, "unit": "document"},
              "quality_minus_fast": {}}
    optional_mode_metrics = (
        ("kl_low_temp_sum", "low_temperature_kl_per_token", 1),
        ("margin_sum", "candidate_margin", 1),
        ("teacher_margin_sum", "teacher_margin", 1),
        ("margin_abs_error_sum", "margin_absolute_error", 1),
        ("margin_sign_matches", "margin_sign_agreement", 1),
        ("top5_overlap_sum", "top5_agreement", 5),
        ("top10_overlap_sum", "top10_agreement", 10),
        ("top32_overlap_sum", "top32_agreement", 32),
    )
    for mode in ("teacher", "fast", "quality"):
        nll = sum(row[mode]["nll_sum"] for row in rows) / tokens
        values = {"nll_per_token": nll, "perplexity": math.exp(nll)}
        if mode != "teacher":
            values.update({
                "kl_per_token": sum(row[mode]["kl_sum"] for row in rows) / tokens,
                "top1_agreement": sum(row[mode]["top1_matches"] for row in rows) / tokens,
            })
            for source, name, denominator in optional_mode_metrics:
                if all(source in row[mode] for row in rows):
                    values[name] = sum(row[mode][source] for row in rows) / (tokens * denominator)
        result["modes"][mode] = values
    rng = random.Random(seed)
    delta_metrics = ["nll_sum", "kl_sum", "top1_matches"]
    for metric in (
        "kl_low_temp_sum",
        "margin_sum",
        "margin_abs_error_sum",
        "margin_sign_matches",
        "top5_overlap_sum",
        "top10_overlap_sum",
        "top32_overlap_sum",
    ):
        if all(metric in row["fast"] and metric in row["quality"] for row in rows):
            delta_metrics.append(metric)
    for metric in delta_metrics:
        deltas = [row["quality"][metric] - row["fast"][metric] for row in rows]
        boot = []
        for _ in range(samples):
            indices = [rng.randrange(len(rows)) for _ in rows]
            boot.append(sum(deltas[i] for i in indices) /
                        sum(rows[i]["teacher"]["tokens"] for i in indices))
        boot.sort()
        result["quality_minus_fast"][metric] = {
            "per_token": sum(deltas) / tokens,
            "ci95": [boot[int(samples * 0.025) - 1], boot[int(samples * 0.975) - 1]],
            "positive_documents": sum(value > 0 for value in deltas),
        }
    return result


def logit_metrics(logits, target, reference=None):
    """Return teacher CE and quality-sensitive logit agreement statistics.

    The returned sums are over predicted tokens.  Keeping sums at the row
    level lets :func:`paired_summary` perform document-level bootstrap samples
    without giving long documents or repeated tokens an incorrect weight.
    """
    import torch
    import torch.nn.functional as F

    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    z = logits[0, :-1].float()
    labels = target[0, 1:]
    if z.shape[0] != labels.shape[0]:
        raise ValueError("logits and target sequence lengths do not match")
    if not torch.isfinite(z).all():
        raise ValueError("non-finite model logits")
    result = {
        "tokens": labels.numel(),
        "nll_sum": float(F.cross_entropy(z, labels, reduction="sum")),
    }
    if reference is None:
        return result
    r = reference[0, :-1].float()
    if r.shape != z.shape or not torch.isfinite(r).all():
        raise ValueError("reference logits do not match candidate logits")
    result["kl_sum"] = float((r.log_softmax(-1).exp() * (r.log_softmax(-1) - z.log_softmax(-1))).sum())
    low_temperature = 0.5
    reference_low = (r / low_temperature).log_softmax(-1)
    candidate_low = (z / low_temperature).log_softmax(-1)
    result["kl_low_temp_sum"] = float(
        (reference_low.exp() * (reference_low - candidate_low)).sum()
    )
    teacher_top2 = r.topk(2, dim=-1).values
    candidate_top2 = z.topk(2, dim=-1).values
    teacher_margin = teacher_top2[:, 0] - teacher_top2[:, 1]
    candidate_margin = candidate_top2[:, 0] - candidate_top2[:, 1]
    result["teacher_margin_sum"] = float(teacher_margin.sum())
    result["margin_sum"] = float(candidate_margin.sum())
    result["margin_abs_error_sum"] = float((candidate_margin - teacher_margin).abs().sum())
    result["margin_sign_matches"] = int((candidate_margin.sign() == teacher_margin.sign()).sum())
    result["top1_matches"] = int((r.argmax(-1) == z.argmax(-1)).sum())
    vocabulary = z.shape[-1]
    for k in (5, 10, 32):
        if vocabulary < k:
            raise ValueError(f"vocabulary must contain at least {k} logits")
        teacher_topk = r.topk(k, dim=-1).indices
        candidate_topk = z.topk(k, dim=-1).indices
        overlap = (
            candidate_topk.unsqueeze(-1) == teacher_topk.unsqueeze(-2)
        ).any(-1).sum()
        result[f"top{k}_overlap_sum"] = int(overlap)
    return result


def main():
    idle = bootstrap_gpu_idle_preflight()
    parser = argparse.ArgumentParser(description=__doc__)
    add_gpu_idle_preflight_args(parser)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fit-directory", type=Path, required=True)
    parser.add_argument("--validation-jsonl-gz", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--documents", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fused", action="store_true", help="Use padded Tensor Core projection and fused epilogue")
    parser.add_argument("--projection", choices=["separate_reference", "tensor_core"], default="tensor_core")
    parser.add_argument("--verify-off", action="store_true", help="Require fused-off logits to equal original-off logits")
    parser.add_argument("--verify-graphs", action="store_true", help="Check separate quality graphs against eager logits")
    parser.add_argument(
        "--max-graph-resident",
        type=int,
        default=8,
        help="maximum captured input signatures retained while verifying graphs",
    )
    args = parser.parse_args()
    if args.verify_off and not args.fused:
        parser.error("--verify-off requires --fused")
    if args.documents < 1 or args.max_tokens < 2:
        parser.error("positive documents and at least two tokens required")
    if args.max_graph_resident < 1:
        parser.error("--max-graph-resident must be positive")
    import hashlib
    import json

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gptqmodel import GPTQModel
    from gptqmodel.quantization.qvq_rank8 import (
        P32WindowConfig,
        _base,
        _digest,
        load_window_package,
        prepare_rank8,
    )
    from gptqmodel.utils.backend import BACKEND

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    teacher_path = args.teacher
    checkpoint = args.checkpoint
    archive = args.fit_directory
    fitted = json.loads((archive / "report.json").read_text())
    import gzip
    import itertools

    source = args.validation_jsonl_gz
    with gzip.open(source, "rt") as handle:
        rows = [json.loads(line) for line in itertools.islice(handle, args.documents)]
    for row in rows:
        row["normalized_user_sha256"] = hashlib.sha256(row["text"].encode()).hexdigest()
    tok = AutoTokenizer.from_pretrained(teacher_path)
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
    ).eval()
    quantized = GPTQModel.load(
        checkpoint,
        backend=BACKEND.QVQ,
        dtype=torch.float16,
        device_map={"": "cuda:0"},
        attn_implementation="eager",
    ).model.eval()
    from gptqmodel.quantization.qvq_window_graphs import P32WindowGraphs

    graph_owner = (
        P32WindowGraphs(quantized, max_graphs=args.max_graph_resident)
        if args.verify_graphs
        else None
    )
    children = []
    for name in fitted["modules"]:
        loaded = load_window_package(
            torch.load(archive / (name + ".pt"), weights_only=True), device="cuda"
        )
        child = quantized.get_submodule(name)
        assert _digest(*_base(child)) == _digest(*_base(loaded))
        assert (
            _digest(dict(teacher.get_submodule(name).named_parameters()), {})
            == fitted["modules"][name]["fit"]["teacher_hash"]
        )
        child.rank8_A = loaded.rank8_A
        child.rank8_B = loaded.rank8_B
        child.rank8_metadata = loaded.rank8_metadata
        children.append(child)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(source, "rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    report = {
        "scope": "Whole-model propagation with fixed fitted corrections; C4 validation subset, first requested documents, per-document BOS and capped tokens, no chat template. Not full C4 validation.",
        "source": str(source),
        "checkpoint": checkpoint,
        "teacher": teacher_path,
        "modules": list(fitted["modules"]),
        "preflight": None if idle is None else idle.as_dict(),
        "dataset_revision": args.dataset_revision,
        "documents_requested": args.documents,
        "max_tokens": args.max_tokens,
        "correction_implementation": f"{args.projection}/fused_epilogue" if args.fused else "separate_reference",
        "graph_residency_policy": {
            "max_graph_resident": args.max_graph_resident,
            "retirement": "synchronize_latest_replay_event_before_lru_release",
        } if graph_owner is not None else None,
        "source_sha256": source_hash,
        "rows": [],
    }

    with torch.no_grad():
        for index, row in enumerate(rows):
            ids = tok(row["text"], add_special_tokens=True)["input_ids"][
                : args.max_tokens
            ]
            if len(ids) < 2:
                raise ValueError("validation document has fewer than two tokens")
            x = torch.tensor([ids], device="cuda")
            reference = teacher(x, use_cache=False).logits
            entry = {
                "document_id": row["normalized_user_sha256"],
                "input_hash": _digest({"input_ids": x}, {}),
                "teacher": logit_metrics(reference, x),
            }
            if graph_owner is not None:
                graph_configs = {
                    mode: {
                        name: P32WindowConfig(
                            recovery_kernel="fused_epilogue" if args.fused else "separate_reference",
                            recovery_projection=args.projection if args.fused else "separate_reference",
                        ) for name in fitted["modules"]
                    } for mode in ("fast", "balanced", "quality")
                }
                # The eager Transformers mask builder creates an unpinned CPU
                # scalar during forward. Supply its equivalent additive causal
                # mask as an explicit graph input; leave eager comparison intact.
                graph_inputs = {
                    "input_ids": x,
                    "attention_mask": torch.full(
                        (1, 1, x.shape[1], x.shape[1]), torch.finfo(torch.float16).min,
                        dtype=torch.float16, device=x.device,
                    ).triu(1),
                }
                graph_owner.capture(
                    index, graph_inputs, configs=graph_configs,
                    static_kwargs={"use_cache": False, "return_dict": False},
                )
                entry["graph_logits_exact"] = {}
            original_off = None
            if args.verify_off:
                for child in children:
                    prepare_rank8(child, P32WindowConfig())
                original_off = quantized(x, use_cache=False).logits
            for mode in (("fast", "balanced", "quality") if graph_owner is not None else ("fast", "quality")):
                for child in children:
                    prepare_rank8(
                        child, P32WindowConfig(
                            recovery_mode="auto", quality_mode=mode,
                            recovery_kernel="fused_epilogue" if args.fused else "separate_reference",
                            recovery_projection=args.projection if args.fused else "separate_reference",
                        )
                    )
                logits = quantized(x, use_cache=False).logits
                if mode == "fast" and original_off is not None:
                    entry["off_logits_exact"] = bool(torch.equal(original_off, logits))
                    if not entry["off_logits_exact"]:
                        raise ValueError("fused-off model logits differ from original-off logits")
                if graph_owner is not None:
                    captured_logits = graph_owner.replay(index, mode, **graph_inputs)[0]
                    exact = bool(torch.equal(captured_logits, logits))
                    entry["graph_logits_exact"][mode] = exact
                    if not exact:
                        raise ValueError(f"{mode} captured logits differ from eager logits")
                entry[mode] = logit_metrics(logits, x, reference)
            if graph_owner is not None:
                entry["graph_residency"] = graph_owner.residency_stats()
            report["rows"].append(entry)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(index, entry, flush=True)
    if graph_owner is not None:
        report["graph_residency"] = graph_owner.residency_stats()
        graph_owner.close()
    report["summary"] = paired_summary(report["rows"])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["summary"], flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
