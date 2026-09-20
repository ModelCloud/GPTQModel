#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the complete Qwen3.8-Flash-Next P32 projection mix on H100."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import benchmark_qvq_a41_phase4_production as common
from scripts.benchmark_qvq_p32_qwen38_flash_next_experts_h100 import (
    _metrics,
    _payload,
)

RATES = (2.0, 2.5, 3.0, 3.5)
M_VALUES = (1, 2, 4, 8, 16)
HIDDEN_SIZE = 2560
EXPERT_WIDTH = 640
FULL_LAYERS = 12
LINEAR_LAYERS = 36
ACTIVE_EXPERTS = 11  # ten routed experts plus one shared expert
FULL_QKV_WIDTHS = (12288, 512, 512)
LINEAR_INPUT_WIDTHS = (10240, 6144)
OUTPUT_SPLITS = {2.0: 12, 2.5: 6, 3.0: 12, 3.5: 24}
SOURCE_PATHS = (
    Path("gptqmodel/nn_modules/qlinear/qvq.py"),
    Path("gptqmodel/nn_modules/qvq_grouped_runtime.py"),
    Path("gptqmodel/utils/qvq_ampere_cuda.py"),
    Path("gptqmodel/utils/qvq_wgmma_cuda.py"),
    Path("gptqmodel_ext/qvq/qvq_ampere_cuda.cu"),
    Path("gptqmodel_ext/qvq/qvq_wgmma_cuda.cu"),
    Path("scripts/benchmark_qvq_p32_qwen38_flash_next_model_h100.py"),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", nargs="+", type=float, default=RATES)
    parser.add_argument("--m-values", nargs="+", type=int, default=M_VALUES)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--replays-per-sample", type=int, default=50)
    parser.add_argument("--fp64-columns", type=int, default=64)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-interval", type=float, default=0.2)
    parser.add_argument("--idle-memory-mib", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qwen38_flash_next_h100/model_scorecard.json"),
    )
    args = parser.parse_args()
    if any(rate not in RATES for rate in args.rates):
        parser.error("rates must be W2, W2.5, W3, or W3.5")
    if any(value not in M_VALUES for value in args.m_values):
        parser.error("M must be one of 1, 2, 4, 8, or 16")
    if min(args.warmup, args.samples, args.replays_per_sample) <= 0:
        parser.error("timing counts must be positive")
    if not 0 < args.fp64_columns <= EXPERT_WIDTH:
        parser.error("fp64-columns must be in [1, 640]")
    return args


def _fingerprint() -> str:
    digest = hashlib.sha256()
    for path in SOURCE_PATHS:
        digest.update(str(path).encode())
        digest.update((REPO_ROOT / path).read_bytes())
    return digest.hexdigest()


def _timed_sandwich(
    torch,
    baseline: Callable[[], Sequence],
    candidate: Callable[[], Sequence],
    args: argparse.Namespace,
):
    candidate_a, candidate_outputs_a = common._graph_timing(
        torch,
        candidate,
        warmup=args.warmup,
        samples=args.samples,
        replays_per_sample=args.replays_per_sample,
    )
    baseline_timing, baseline_outputs = common._graph_timing(
        torch,
        baseline,
        warmup=args.warmup,
        samples=args.samples,
        replays_per_sample=args.replays_per_sample,
    )
    candidate_b, candidate_outputs_b = common._graph_timing(
        torch,
        candidate,
        warmup=args.warmup,
        samples=args.samples,
        replays_per_sample=args.replays_per_sample,
    )
    if len(candidate_outputs_a) != len(candidate_outputs_b) or not all(
        torch.equal(left, right)
        for left, right in zip(candidate_outputs_a, candidate_outputs_b, strict=True)
    ):
        raise RuntimeError("candidate CUDA Graph replay is not bitwise repeatable")
    candidate_median = math.sqrt(
        candidate_a["median_ms"] * candidate_b["median_ms"]
    )
    return (
        {
            "baseline": baseline_timing,
            "candidate_first": candidate_a,
            "candidate_second": candidate_b,
            "candidate_sandwich_median_ms": candidate_median,
            "speedup": baseline_timing["median_ms"] / candidate_median,
        },
        baseline_outputs,
        candidate_outputs_b,
    )


def _sample_columns(torch, width: int, count: int):
    return torch.linspace(
        0,
        width - 1,
        min(width, count),
        dtype=torch.int64,
        device="cuda",
    )


def _oracle_metrics(
    torch,
    *,
    x,
    dense_weights,
    baseline_outputs,
    candidate_outputs,
    logical_rows: int,
    columns: int,
):
    local_rows = []
    for dense, baseline, candidate in zip(
        dense_weights,
        baseline_outputs,
        candidate_outputs,
        strict=True,
    ):
        selected = _sample_columns(torch, dense.shape[1], columns)
        fp64 = x.double() @ dense[:, selected].double()
        fp32 = x.float() @ dense[:, selected].float()
        baseline = baseline[:logical_rows]
        candidate = candidate[:logical_rows]
        local = _metrics(candidate, baseline)
        if local["mean_abs"] > 4e-3 or local["max_abs"] > 0.046875:
            raise RuntimeError(f"local accuracy gate failed: {local}")
        local_rows.append(
            {
                "candidate_vs_baseline": local,
                "baseline_vs_fp64": _metrics(baseline[:, selected], fp64),
                "candidate_vs_fp64": _metrics(candidate[:, selected], fp64),
                "fp32_vs_fp64": _metrics(fp32, fp64),
            }
        )
    return local_rows


def _planar_call(qvq_cuda_gemv, x, payloads, bits, widths):
    return tuple(
        qvq_cuda_gemv(
            x,
            payload[0],
            bits,
            out_features=width,
            output_fp32=True,
            vector_size=2,
            bank_ids=payload[2],
            v2b2_p32=True,
            bank_alt_id=3,
            _bank_ids_validated=True,
        )
        for payload, width in zip(payloads, widths, strict=True)
    )


def _weighted_total(
    site_times: dict[str, tuple[float, float]],
    index: int,
    *,
    full_layers: int,
    linear_layers: int,
) -> float:
    attention = (
        full_layers * site_times["full_qkv"][index]
        + linear_layers * site_times["linear_qkv_z"][index]
        + (full_layers + linear_layers) * site_times["attention_output"][index]
    )
    experts = (
        (full_layers + linear_layers)
        * ACTIVE_EXPERTS
        * (
            site_times["expert_gate_up"][index]
            + site_times["expert_down"][index]
        )
    )
    return attention + experts


def _run(args: argparse.Namespace) -> None:
    os.environ["QVQ_AMPERE_AUTOTUNE"] = "0"

    import torch

    from gptqmodel.quantization.qvq_codecs import (
        PGC16_CODEBOOK_VERSION,
        pgc16_levels_for_version,
    )
    from gptqmodel.quantization.qvq_rates import qvq_transition_bits
    from gptqmodel.utils.qvq_ampere_cuda import (
        qvq_p32_window_ampere,
        qvq_p32_window_ampere_group_plan,
        qvq_p32_window_ampere_grouped_packed,
        qvq_pack_p32_window_ampere_group,
    )
    from gptqmodel.utils.qvq_cuda import qvq_cuda_gemv
    from gptqmodel.utils.qvq_wgmma_cuda import (
        qvq_h100_grouped_ordered_split_counts,
        qvq_p32_window_wgmma_group_plan,
        qvq_p32_window_wgmma_grouped_ordered_packed,
        qvq_p32_window_wgmma_grouped_packed,
        qvq_p32_window_wgmma_m16_tma_ordered_split,
        qvq_pack_p32_window_hopper_group,
    )

    device_info = common._assert_h100(torch)
    properties = torch.cuda.get_device_properties(0)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().cuda()
    fingerprint = _fingerprint()
    rows = []
    scorecards = []

    for bits in args.rates:
        transition_bits = qvq_transition_bits(bits, vector_size=2)
        full_payloads = tuple(
            _payload(
                torch,
                bits=bits,
                k=HIDDEN_SIZE,
                n=width,
                seed=20263000 + int(bits * 10) * 100 + index,
            )
            for index, width in enumerate(FULL_QKV_WIDTHS)
        )
        linear_payloads = tuple(
            _payload(
                torch,
                bits=bits,
                k=HIDDEN_SIZE,
                n=width,
                seed=20264000 + int(bits * 10) * 100 + index,
            )
            for index, width in enumerate(LINEAR_INPUT_WIDTHS)
        )
        output_payload = _payload(
            torch,
            bits=bits,
            k=6144,
            n=HIDDEN_SIZE,
            seed=20265000 + int(bits * 10),
        )
        expert_payloads = tuple(
            _payload(
                torch,
                bits=bits,
                k=HIDDEN_SIZE,
                n=EXPERT_WIDTH,
                seed=20266000 + int(bits * 10) * 10 + index,
            )
            for index in range(2)
        )
        down_payload = _payload(
            torch,
            bits=bits,
            k=EXPERT_WIDTH,
            n=HIDDEN_SIZE,
            seed=20267000 + int(bits * 10),
        )

        for logical_rows in args.m_values:
            generator = torch.Generator(device="cuda").manual_seed(
                20268000 + int(bits * 10) * 100 + logical_rows
            )
            hidden = (
                torch.randn(
                    (logical_rows, HIDDEN_SIZE),
                    generator=generator,
                    device="cuda",
                )
                * 0.02
            ).half()
            padded_hidden = torch.zeros(
                (16, HIDDEN_SIZE), dtype=torch.float16, device="cuda"
            )
            padded_hidden[:logical_rows].copy_(hidden)
            output_input = (
                torch.randn(
                    (logical_rows, 6144),
                    generator=generator,
                    device="cuda",
                )
                * 0.02
            ).half()
            down_input = (
                torch.randn(
                    (logical_rows, EXPERT_WIDTH),
                    generator=generator,
                    device="cuda",
                )
                * 0.02
            ).half()

            site_times = {}
            for name, payloads, widths in (
                ("full_qkv", full_payloads, FULL_QKV_WIDTHS),
                ("linear_qkv_z", linear_payloads, LINEAR_INPUT_WIDTHS),
            ):
                splits = qvq_h100_grouped_ordered_split_counts(
                    device_name=properties.name,
                    compute_capability=(properties.major, properties.minor),
                    in_features=HIDDEN_SIZE,
                    out_features=widths,
                    transition_bits=transition_bits,
                )
                if splits is None:
                    raise RuntimeError(f"missing production split schedule for {name}")
                plan = qvq_p32_window_wgmma_group_plan(
                    padded_hidden,
                    tuple(payload[1] for payload in payloads),
                    levels,
                    tuple(payload[2] for payload in payloads),
                    bits,
                    out_features=widths,
                    bank_alt_ids=(3,) * len(widths),
                    split_counts=splits,
                )
                packed = qvq_pack_p32_window_hopper_group(
                    tuple(payload[1] for payload in payloads),
                    tuple(payload[2] for payload in payloads),
                    plan,
                )
                grouped = (
                    qvq_p32_window_wgmma_grouped_ordered_packed
                    if any(split != 1 for split in splits)
                    else qvq_p32_window_wgmma_grouped_packed
                )

                def baseline(
                    payloads=payloads,
                    widths=widths,
                    hidden=hidden,
                    bits=bits,
                ):
                    return _planar_call(
                        qvq_cuda_gemv, hidden, payloads, bits, widths
                    )

                def candidate(
                    grouped=grouped,
                    packed=packed,
                    padded_hidden=padded_hidden,
                ):
                    return grouped(padded_hidden, packed, levels)

                timing, baseline_outputs, candidate_outputs = _timed_sandwich(
                    torch, baseline, candidate, args
                )
                oracle = _oracle_metrics(
                    torch,
                    x=hidden,
                    dense_weights=tuple(payload[3] for payload in payloads),
                    baseline_outputs=baseline_outputs,
                    candidate_outputs=candidate_outputs,
                    logical_rows=logical_rows,
                    columns=args.fp64_columns,
                )
                row = {
                    "site": name,
                    "bits": bits,
                    "m": logical_rows,
                    "k": HIDDEN_SIZE,
                    "widths": widths,
                    "splits": splits,
                    **timing,
                    "oracles": oracle,
                    "repeatable": True,
                }
                rows.append(row)
                site_times[name] = (
                    timing["baseline"]["median_ms"],
                    timing["candidate_sandwich_median_ms"],
                )

            planar, window, selectors, output_dense = output_payload

            def output_baseline(
                output_input=output_input,
                planar=planar,
                bits=bits,
                selectors=selectors,
            ):
                return (
                    qvq_cuda_gemv(
                        output_input,
                        planar,
                        bits,
                        out_features=HIDDEN_SIZE,
                        output_fp32=True,
                        vector_size=2,
                        bank_ids=selectors,
                        v2b2_p32=True,
                        bank_alt_id=3,
                        _bank_ids_validated=True,
                    ),
                )

            def output_candidate(
                output_input=output_input,
                window=window,
                selectors=selectors,
                bits=bits,
            ):
                return (
                    qvq_p32_window_wgmma_m16_tma_ordered_split(
                        output_input,
                        window,
                        levels,
                        selectors,
                        bits,
                        out_features=HIDDEN_SIZE,
                        bank_alt_id=3,
                        split_count=OUTPUT_SPLITS[bits],
                    ),
                )

            timing, baseline_outputs, candidate_outputs = _timed_sandwich(
                torch, output_baseline, output_candidate, args
            )
            oracle = _oracle_metrics(
                torch,
                x=output_input,
                dense_weights=(output_dense,),
                baseline_outputs=baseline_outputs,
                candidate_outputs=candidate_outputs,
                logical_rows=logical_rows,
                columns=args.fp64_columns,
            )
            rows.append(
                {
                    "site": "attention_output",
                    "bits": bits,
                    "m": logical_rows,
                    "k": 6144,
                    "widths": (HIDDEN_SIZE,),
                    "splits": (OUTPUT_SPLITS[bits],),
                    **timing,
                    "oracles": oracle,
                    "repeatable": True,
                }
            )
            site_times["attention_output"] = (
                timing["baseline"]["median_ms"],
                timing["candidate_sandwich_median_ms"],
            )

            expert_plan = qvq_p32_window_ampere_group_plan(
                hidden,
                tuple(payload[1] for payload in expert_payloads),
                levels,
                tuple(payload[2] for payload in expert_payloads),
                bits,
                out_features=(EXPERT_WIDTH, EXPERT_WIDTH),
                bank_alt_ids=(3, 3),
                split_counts=(32, 32),
            )
            expert_packed = qvq_pack_p32_window_ampere_group(
                tuple(payload[1] for payload in expert_payloads),
                tuple(payload[2] for payload in expert_payloads),
                expert_plan,
            )

            def expert_baseline(
                hidden=hidden,
                payloads=expert_payloads,
                bits=bits,
            ):
                return _planar_call(
                    qvq_cuda_gemv,
                    hidden,
                    payloads,
                    bits,
                    (EXPERT_WIDTH, EXPERT_WIDTH),
                )

            def expert_candidate(
                hidden=hidden,
                expert_packed=expert_packed,
            ):
                return qvq_p32_window_ampere_grouped_packed(
                    hidden, expert_packed, levels
                )

            timing, baseline_outputs, candidate_outputs = _timed_sandwich(
                torch, expert_baseline, expert_candidate, args
            )
            oracle = _oracle_metrics(
                torch,
                x=hidden,
                dense_weights=tuple(payload[3] for payload in expert_payloads),
                baseline_outputs=baseline_outputs,
                candidate_outputs=candidate_outputs,
                logical_rows=logical_rows,
                columns=args.fp64_columns,
            )
            rows.append(
                {
                    "site": "expert_gate_up",
                    "bits": bits,
                    "m": logical_rows,
                    "k": HIDDEN_SIZE,
                    "widths": (EXPERT_WIDTH, EXPERT_WIDTH),
                    "splits": (32, 32),
                    **timing,
                    "oracles": oracle,
                    "repeatable": True,
                }
            )
            site_times["expert_gate_up"] = (
                timing["baseline"]["median_ms"],
                timing["candidate_sandwich_median_ms"],
            )

            down_planar, down_window, down_selectors, down_dense = down_payload

            def down_baseline(
                down_input=down_input,
                down_planar=down_planar,
                bits=bits,
                down_selectors=down_selectors,
            ):
                return (
                    qvq_cuda_gemv(
                        down_input,
                        down_planar,
                        bits,
                        out_features=HIDDEN_SIZE,
                        output_fp32=True,
                        vector_size=2,
                        bank_ids=down_selectors,
                        v2b2_p32=True,
                        bank_alt_id=3,
                        _bank_ids_validated=True,
                    ),
                )

            def down_candidate(
                down_input=down_input,
                down_window=down_window,
                down_selectors=down_selectors,
                bits=bits,
            ):
                return (
                    qvq_p32_window_ampere(
                        down_input,
                        down_window,
                        levels,
                        down_selectors,
                        bits,
                        out_features=HIDDEN_SIZE,
                        bank_alt_id=3,
                        split_count=0,
                    ),
                )

            timing, baseline_outputs, candidate_outputs = _timed_sandwich(
                torch, down_baseline, down_candidate, args
            )
            oracle = _oracle_metrics(
                torch,
                x=down_input,
                dense_weights=(down_dense,),
                baseline_outputs=baseline_outputs,
                candidate_outputs=candidate_outputs,
                logical_rows=logical_rows,
                columns=args.fp64_columns,
            )
            rows.append(
                {
                    "site": "expert_down",
                    "bits": bits,
                    "m": logical_rows,
                    "k": EXPERT_WIDTH,
                    "widths": (HIDDEN_SIZE,),
                    **timing,
                    "oracles": oracle,
                    "repeatable": True,
                }
            )
            site_times["expert_down"] = (
                timing["baseline"]["median_ms"],
                timing["candidate_sandwich_median_ms"],
            )

            full_baseline = _weighted_total(
                site_times, 0, full_layers=1, linear_layers=0
            )
            full_candidate = _weighted_total(
                site_times, 1, full_layers=1, linear_layers=0
            )
            linear_baseline = _weighted_total(
                site_times, 0, full_layers=0, linear_layers=1
            )
            linear_candidate = _weighted_total(
                site_times, 1, full_layers=0, linear_layers=1
            )
            model_baseline = _weighted_total(
                site_times,
                0,
                full_layers=FULL_LAYERS,
                linear_layers=LINEAR_LAYERS,
            )
            model_candidate = _weighted_total(
                site_times,
                1,
                full_layers=FULL_LAYERS,
                linear_layers=LINEAR_LAYERS,
            )
            attention_baseline = (
                FULL_LAYERS * site_times["full_qkv"][0]
                + LINEAR_LAYERS * site_times["linear_qkv_z"][0]
                + (FULL_LAYERS + LINEAR_LAYERS)
                * site_times["attention_output"][0]
            )
            attention_candidate = (
                FULL_LAYERS * site_times["full_qkv"][1]
                + LINEAR_LAYERS * site_times["linear_qkv_z"][1]
                + (FULL_LAYERS + LINEAR_LAYERS)
                * site_times["attention_output"][1]
            )
            scorecard = {
                "bits": bits,
                "m": logical_rows,
                "full_layer": {
                    "baseline_ms": full_baseline,
                    "candidate_ms": full_candidate,
                    "speedup": full_baseline / full_candidate,
                },
                "linear_layer": {
                    "baseline_ms": linear_baseline,
                    "candidate_ms": linear_candidate,
                    "speedup": linear_baseline / linear_candidate,
                },
                "attention_stack": {
                    "baseline_ms": attention_baseline,
                    "candidate_ms": attention_candidate,
                    "speedup": attention_baseline / attention_candidate,
                },
                "text_model_projection_stack": {
                    "baseline_ms": model_baseline,
                    "candidate_ms": model_candidate,
                    "speedup": model_baseline / model_candidate,
                },
            }
            scorecards.append(scorecard)
            print(
                f"W{bits:g} M{logical_rows}: full={scorecard['full_layer']['speedup']:.3f}x "
                f"linear={scorecard['linear_layer']['speedup']:.3f}x "
                f"attention={scorecard['attention_stack']['speedup']:.3f}x "
                f"model={scorecard['text_model_projection_stack']['speedup']:.3f}x",
                flush=True,
            )

        del (
            full_payloads,
            linear_payloads,
            output_payload,
            expert_payloads,
            down_payload,
        )
        torch.cuda.empty_cache()

    if fingerprint != _fingerprint():
        raise RuntimeError("benchmark sources changed during execution")
    if any(
        score["text_model_projection_stack"]["speedup"] < 2.0
        for score in scorecards
    ):
        raise RuntimeError("the complete projection stack did not reach 2x")
    result = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "source_fingerprint": fingerprint,
        "checkpoint": "/monster/data/model/Qwen3.8-Flash-Next",
        "device": device_info,
        "baseline": "planar qvq_cuda V2B2-P32 projection kernels",
        "candidate": "production exact contiguous-window projection kernels",
        "scope": {
            "text_layers": FULL_LAYERS + LINEAR_LAYERS,
            "full_attention_layers": FULL_LAYERS,
            "linear_attention_layers": LINEAR_LAYERS,
            "active_experts_per_layer": ACTIVE_EXPERTS,
            "excluded": "attention core, routing, activations, norms, collectives, vision, and MTP",
        },
        "math_policy": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "kernel_input": "FP16",
            "kernel_accumulator": "FP32",
            "output": "FP32",
        },
        "timing": {
            "method": "candidate/control/candidate CUDA Graph replay timed by CUDA events",
            "warmup": args.warmup,
            "samples": args.samples,
            "replays_per_sample": args.replays_per_sample,
            "aggregation": "sum of measured site medians weighted by 12/36 layers and 11 experts",
        },
        "oracle": {
            "kind": "sampled FP32 and FP64 accumulation over full K",
            "columns_per_child": args.fp64_columns,
            "local_mean_abs_gate": 4e-3,
            "local_max_abs_gate": 0.046875,
        },
        "rows": rows,
        "scorecards": scorecards,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"result: {args.output}", flush=True)


if __name__ == "__main__":
    parsed_args = _args()
    common._idle_h100_preflight(parsed_args)
    _run(parsed_args)
