#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for import_root in (SCRIPT_DIR, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from gpu_idle_preflight import (  # noqa: E402
    add_gpu_idle_preflight_args,
    bootstrap_gpu_idle_preflight,
    recheck_gpu_exclusivity,
)

_GPU_IDLE_PREFLIGHT = bootstrap_gpu_idle_preflight() if __name__ == "__main__" else None

import torch  # noqa: E402
from tabulate import tabulate  # noqa: E402

from gptqmodel import extension  # noqa: E402
from gptqmodel.utils import amplin  # noqa: E402
from scripts.benchmark_amplin_vs_marlin import (  # noqa: E402
    BITS,
    GROUP_SIZE,
    _build_marlin,
    _dequantized_weight,
    _dtype_name,
    _git_revision,
    _make_case,
    _measure,
    _nvidia_smi_inventory,
    _raw_marlin_call,
    _resolve_dtypes,
)


@dataclass(frozen=True)
class ShapeSpec:
    model: str
    role: str
    size_k: int
    size_n: int


SHAPES = (
    # Laguna S 2.1
    ShapeSpec("laguna-s-2.1", "expert-down", 1024, 3072),
    ShapeSpec("laguna-s-2.1", "attn-g48", 3072, 48),
    ShapeSpec("laguna-s-2.1", "attn-g72", 3072, 72),
    ShapeSpec("laguna-s-2.1", "kv/expert-up", 3072, 1024),
    ShapeSpec("laguna-s-2.1", "q-proj-6144", 3072, 6144),
    ShapeSpec("laguna-s-2.1", "q-proj-9216", 3072, 9216),
    ShapeSpec("laguna-s-2.1", "dense-up", 3072, 12288),
    ShapeSpec("laguna-s-2.1", "o-proj-6144", 6144, 3072),
    ShapeSpec("laguna-s-2.1", "o-proj-9216", 9216, 3072),
    ShapeSpec("laguna-s-2.1", "dense-down", 12288, 3072),
    ShapeSpec("laguna-s-2.1", "router-gate", 3072, 256),
    # GLM 5.2
    ShapeSpec("glm-5.2", "q-a-proj", 6144, 2048),
    ShapeSpec("glm-5.2", "q-b-proj", 2048, 4096),
    ShapeSpec("glm-5.2", "q-b-proj-large", 2048, 16384),
    ShapeSpec("glm-5.2", "kv-a-proj", 6144, 128),
    ShapeSpec("glm-5.2", "kv-a-proj-mqa", 6144, 576),
    ShapeSpec("glm-5.2", "kv-b-proj", 512, 28672),
    ShapeSpec("glm-5.2", "o-proj", 16384, 6144),
    ShapeSpec("glm-5.2", "dense-up", 6144, 12288),
    ShapeSpec("glm-5.2", "dense-down", 12288, 6144),
    ShapeSpec("glm-5.2", "moe-up", 6144, 2048),
    ShapeSpec("glm-5.2", "moe-down", 2048, 6144),
    ShapeSpec("glm-5.2", "indexer-wq-b", 2048, 4096),
    ShapeSpec("glm-5.2", "lm-head", 6144, 154880),
    # Kimi K2.5 / K2.6 (K2.6 is a strict subset)
    ShapeSpec("kimi-k2.5", "q-a-proj", 7168, 1536),
    ShapeSpec("kimi-k2.5", "q-b-proj", 1536, 12288),
    ShapeSpec("kimi-k2.5", "kv-a-proj-mqa", 7168, 576),
    ShapeSpec("kimi-k2.5", "kv-b-proj", 512, 16384),
    ShapeSpec("kimi-k2.5", "o-proj", 8192, 7168),
    ShapeSpec("kimi-k2.5", "dense-up", 7168, 18432),
    ShapeSpec("kimi-k2.5", "dense-down", 18432, 7168),
    ShapeSpec("kimi-k2.5", "shared-up", 7168, 2048),
    ShapeSpec("kimi-k2.5", "shared-down", 2048, 7168),
    ShapeSpec("kimi-k2.5", "router-gate", 7168, 384),
    ShapeSpec("kimi-k2.5", "lm-head", 7168, 163840),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark raw Amplin over exact Laguna S 2.1, GLM 5.2, and Kimi K2.5 GPTQ W4 linear shapes."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", choices=("all", "laguna-s-2.1", "glm-5.2", "kimi-k2.5"), default="all")
    parser.add_argument("--shape", action="append", help="Optional repeated KxN filter, for example 4096x12288.")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "both"), default="both")
    parser.add_argument(
        "--m-values",
        default="1,2,4,8,16",
        help="Comma-separated flattened input-row counts; defaults to common decode batches 1,2,4,8 plus 16.",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--hmma",
        action="store_true",
        help="Also time selected and retained-V0 W4-N64-K128 HMMA paths where M is divisible by 16 and N by 64.",
    )
    parser.add_argument(
        "--k12288-wide",
        action="store_true",
        help="Also time the barrier-free 16-warp M1 K12288 research path where N is divisible by 16.",
    )
    parser.add_argument(
        "--multirow",
        action="store_true",
        help="Also time the 2/4-row weight-reuse research path on large MLP shapes at M=2,4,8,16.",
    )
    parser.add_argument(
        "--padded-m16",
        action="store_true",
        help="Also time the packed N16 MMA path that pads M=2,4,8 to M16 in registers.",
    )
    parser.add_argument(
        "--splitk4-m16",
        action="store_true",
        help="Also time the K12288 four-warp intra-CTA split-K padded-M16 path.",
    )
    parser.add_argument(
        "--splitk8-m16",
        action="store_true",
        help="Also time the K12288 eight-warp intra-CTA split-K padded-M16 path.",
    )
    parser.add_argument(
        "--splitk12-m16",
        action="store_true",
        help="Also time the K12288 12-warp intra-CTA split-K padded-M16 path.",
    )
    parser.add_argument(
        "--splitk12-n32",
        action="store_true",
        help="Also time the K12288 12-warp split-K path that reuses each A fragment across N32.",
    )
    parser.add_argument(
        "--splitk16-n32",
        action="store_true",
        help="Also time the K12288 16-warp split-K path that reuses each A fragment across N32.",
    )
    parser.add_argument(
        "--splitk-n32-pipe2",
        action="store_true",
        help="Also time the N32 split-K8/K12/K16 controls with two-stage register prefetching.",
    )
    parser.add_argument(
        "--splitk12-n32-interleaved",
        action="store_true",
        help="Also time split-K12 pipe2 with lane-local N32 int32 word pairs loaded as aligned uint2.",
    )
    parser.add_argument(
        "--splitk24-n64-interleaved",
        action="store_true",
        help="Also time split-K24/N64 pipe2 with lane-local int32 word quads loaded as aligned uint4.",
    )
    parser.add_argument(
        "--splitk12x2-n64-coop",
        action="store_true",
        help="Also time two cooperative K12 CTAs per interleaved N64 tile with FP32 cross-CTA reduction.",
    )
    parser.add_argument(
        "--splitk16-m16",
        action="store_true",
        help="Also time the K12288 16-warp intra-CTA split-K padded-M16 path.",
    )
    parser.add_argument(
        "--m32-sweep",
        action="store_true",
        help=(
            "With --hmma, time M32 direct-A, legal M64 direct-A, and Marlin paths; "
            "M<32 retains canonical Amplin for a legal small-batch comparison."
        ),
    )
    parser.add_argument(
        "--m32-splitk12x2-n64-coop",
        action="store_true",
        help="Also time two cooperative K12 CTAs per interleaved N64 tile for M<=32.",
    )
    parser.add_argument(
        "--m32-splitk24-n64",
        action="store_true",
        help="Also time single-CTA K24 split per interleaved N64 tile for M=17..32.",
    )
    parser.add_argument(
        "--dense-ceiling",
        action="store_true",
        help="Also time a resident dense weight matmul; this uses 4x the W4 weight bytes and is not a GPTQ path.",
    )
    parser.add_argument("--json-out", type=Path)
    add_gpu_idle_preflight_args(parser)
    return parser.parse_args()


def _parse_m_values(value: str) -> tuple[int, ...]:
    try:
        values = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    except ValueError as exc:
        raise ValueError("--m-values must be a comma-separated list of integers") from exc
    if not values or any(item <= 0 for item in values):
        raise ValueError("--m-values must contain positive integers")
    return values


def _parse_shape_filters(values: list[str] | None) -> set[tuple[int, int]] | None:
    if not values:
        return None
    result = set()
    for value in values:
        try:
            size_k_text, size_n_text = value.lower().split("x", maxsplit=1)
            size_k, size_n = int(size_k_text), int(size_n_text)
        except ValueError as exc:
            raise ValueError(f"invalid --shape {value!r}; expected KxN") from exc
        result.add((size_k, size_n))
    return result


def _select_shapes(model: str, filters: set[tuple[int, int]] | None) -> tuple[ShapeSpec, ...]:
    selected = tuple(
        spec
        for spec in SHAPES
        if (model == "all" or spec.model == model)
        and (filters is None or (spec.size_k, spec.size_n) in filters)
    )
    if not selected:
        raise ValueError("the model and --shape filters selected no Amplin shape")
    return selected


def _benchmark_shape_dtype(
    *,
    spec: ShapeSpec,
    dtype: torch.dtype,
    m_values: tuple[int, ...],
    device: torch.device,
    seed: int,
    warmup: int,
    iters: int,
    rounds: int,
    dense_ceiling: bool,
    amplin_op,
    amplin_k12288_wide_op,
    amplin_multirow_op,
    amplin_hmma_op,
    amplin_hmma_v0_op,
    amplin_hmma_m64_v1_op,
    amplin_hmma_m64_v2_op,
    amplin_hmma_m64_v2_sync_a128_op,
    amplin_hmma_m64_v3_op,
    amplin_mma_lane_m64_op,
    amplin_mma_lane_m64_global_a_op,
    amplin_mma_lane_m32_global_a_op,
    amplin_mma_lane_m32_n32_global_a_op,
    amplin_mma_lane_m32_n64_splitk12x2_coop_interleaved_op,
    amplin_mma_lane_m32_n64_splitk24_pipe2_interleaved_op,
    amplin_mma_lane_m16_n16_padded_op,
    amplin_mma_lane_m16_n16_splitk4_op,
    amplin_mma_lane_m16_n16_splitk8_op,
    amplin_mma_lane_m16_n16_splitk12_op,
    amplin_mma_lane_m16_n32_splitk12_op,
    amplin_mma_lane_m16_n32_splitk16_op,
    amplin_mma_lane_m16_n32_splitk8_pipe2_op,
    amplin_mma_lane_m16_n32_splitk12_pipe2_op,
    amplin_mma_lane_m16_n32_splitk12_pipe2_interleaved_op,
    amplin_mma_lane_m16_n64_splitk24_pipe2_interleaved_op,
    amplin_mma_lane_m16_n64_splitk12x2_coop_interleaved_op,
    amplin_mma_lane_m16_n32_splitk16_pipe2_op,
    amplin_mma_lane_m16_n16_splitk16_op,
    marlin_op,
    m32_sweep: bool,
    pre_timing_check: Callable[[], None] | None,
) -> list[dict]:
    max_m = max(m_values)
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    full_input, canonical_qweight, canonical_scales = _make_case(
        device=device,
        dtype=dtype,
        size_m=max_m,
        size_k=spec.size_k,
        size_n=spec.size_n,
        seed=seed,
    )
    dense_weight_fp32 = _dequantized_weight(canonical_qweight, canonical_scales)
    dense_weight = dense_weight_fp32.to(dtype) if dense_ceiling else None
    marlin_legal = spec.size_n % 64 == 0
    hmma_shape_legal = amplin_hmma_op is not None and spec.size_n % amplin.HMMA_N_TILE == 0
    mma_lane_n32_shape_legal = (
        amplin_mma_lane_m32_n32_global_a_op is not None and spec.size_n % 8 == 0
    )
    padded_m16_shape_legal = (
        amplin_mma_lane_m16_n16_padded_op is not None
        and spec.size_k % 128 == 0
        and spec.size_n % 16 == 0
    )
    splitk4_m16_shape_legal = (
        amplin_mma_lane_m16_n16_splitk4_op is not None
        and spec.size_k % 128 == 0
        and (spec.size_k // 128) % 4 == 0
        and spec.size_n % 16 == 0
    )
    splitk8_m16_shape_legal = (
        amplin_mma_lane_m16_n16_splitk8_op is not None
        and spec.size_k % 128 == 0
        and (spec.size_k // 128) % 8 == 0
        and spec.size_n % 16 == 0
    )
    splitk12_m16_shape_legal = (
        amplin_mma_lane_m16_n16_splitk12_op is not None
        and spec.size_k % 128 == 0
        and (spec.size_k // 128) % 12 == 0
        and spec.size_n % 16 == 0
    )
    splitk12_n32_shape_legal = (
        amplin_mma_lane_m16_n32_splitk12_op is not None
        and spec.size_k % 128 == 0
        and (spec.size_k // 128) % 12 == 0
        and spec.size_n % 32 == 0
    )
    splitk16_n32_shape_legal = (
        amplin_mma_lane_m16_n32_splitk16_op is not None
        and spec.size_k % 128 == 0
        and (spec.size_k // 128) % 16 == 0
        and spec.size_n % 32 == 0
    )
    splitk8_n32_pipe2_shape_legal = (
        amplin_mma_lane_m16_n32_splitk8_pipe2_op is not None
        and spec.size_k % 128 == 0
        and (spec.size_k // 128) % 8 == 0
        and spec.size_n % 32 == 0
    )
    splitk12_n32_pipe2_shape_legal = (
        amplin_mma_lane_m16_n32_splitk12_pipe2_op is not None
        and spec.size_k % 128 == 0
        and (spec.size_k // 128) % 12 == 0
        and spec.size_n % 32 == 0
    )
    splitk12_n32_interleaved_shape_legal = (
        amplin_mma_lane_m16_n32_splitk12_pipe2_interleaved_op is not None
        and spec.size_k % 128 == 0
        and (spec.size_k // 128) % 12 == 0
        and spec.size_n % 32 == 0
    )
    splitk24_n64_interleaved_shape_legal = (
        amplin_mma_lane_m16_n64_splitk24_pipe2_interleaved_op is not None
        and spec.size_k % 128 == 0
        and spec.size_n % 64 == 0
    )
    splitk12x2_n64_coop_shape_legal = (
        amplin_mma_lane_m16_n64_splitk12x2_coop_interleaved_op is not None
        and spec.size_k % 128 == 0
        and spec.size_n % 64 == 0
        and (spec.size_n // 64) * 2 <= sm_count * 2
    )
    m32_splitk12x2_n64_coop_shape_legal = (
        amplin_mma_lane_m32_n64_splitk12x2_coop_interleaved_op is not None
        and spec.size_k % 128 == 0
        and spec.size_n % 64 == 0
    )
    m32_splitk24_n64_shape_legal = (
        amplin_mma_lane_m32_n64_splitk24_pipe2_interleaved_op is not None
        and spec.size_k % 128 == 0
        and spec.size_n % 64 == 0
    )
    splitk16_n32_pipe2_shape_legal = (
        amplin_mma_lane_m16_n32_splitk16_pipe2_op is not None
        and spec.size_k % 128 == 0
        and (spec.size_k // 128) % 16 == 0
        and spec.size_n % 32 == 0
    )
    splitk16_m16_shape_legal = (
        amplin_mma_lane_m16_n16_splitk16_op is not None
        and spec.size_k % 128 == 0
        and (spec.size_k // 128) % 16 == 0
        and spec.size_n % 16 == 0
    )
    packed_hmma_qweight = amplin.pack_hmma_qweight(canonical_qweight) if hmma_shape_legal else None
    packed_hmma_scales = (
        amplin.pack_hmma_scales(canonical_scales)
        if (
            hmma_shape_legal
            or mma_lane_n32_shape_legal
            or padded_m16_shape_legal
            or splitk4_m16_shape_legal
            or splitk8_m16_shape_legal
            or splitk12_m16_shape_legal
            or splitk12_n32_shape_legal
            or splitk16_n32_shape_legal
            or splitk8_n32_pipe2_shape_legal
            or splitk12_n32_pipe2_shape_legal
            or splitk12_n32_interleaved_shape_legal
            or splitk24_n64_interleaved_shape_legal
            or splitk12x2_n64_coop_shape_legal
            or splitk16_n32_pipe2_shape_legal
            or splitk16_m16_shape_legal
        )
        else None
    )
    packed_mma_lane_qweight = (
        amplin.pack_mma_lane_qweight(canonical_qweight)
        if (
            hmma_shape_legal
            or mma_lane_n32_shape_legal
            or padded_m16_shape_legal
            or splitk4_m16_shape_legal
            or splitk8_m16_shape_legal
            or splitk12_m16_shape_legal
            or splitk12_n32_shape_legal
            or splitk16_n32_shape_legal
            or splitk8_n32_pipe2_shape_legal
            or splitk12_n32_pipe2_shape_legal
            or splitk12_n32_interleaved_shape_legal
            or splitk24_n64_interleaved_shape_legal
            or splitk12x2_n64_coop_shape_legal
            or splitk16_n32_pipe2_shape_legal
            or splitk16_m16_shape_legal
        )
        else None
    )
    packed_mma_lane_n32_qweight = (
        amplin.pack_mma_lane_n32_qweight(canonical_qweight)
        if splitk12_n32_interleaved_shape_legal
        else None
    )
    packed_mma_lane_n64_qweight = (
        amplin.pack_mma_lane_n64_qweight(canonical_qweight)
        if (
            splitk24_n64_interleaved_shape_legal
            or splitk12x2_n64_coop_shape_legal
            or m32_splitk12x2_n64_coop_shape_legal
            or m32_splitk24_n64_shape_legal
        )
        else None
    )
    marlin_module = (
        _build_marlin(
            device=device,
            dtype=dtype,
            qweight=canonical_qweight,
            scales=canonical_scales,
        )
        if marlin_legal
        else None
    )

    canonical_weight_bytes = canonical_qweight.numel() * canonical_qweight.element_size()
    canonical_scale_bytes = canonical_scales.numel() * canonical_scales.element_size()
    dense_weight_bytes = dense_weight_fp32.numel() * torch.tensor([], dtype=dtype).element_size()
    rows = []

    with torch.inference_mode():
        for size_m in m_values:
            input = full_input[:size_m].contiguous()
            reference = input.to(torch.float32) @ dense_weight_fp32
            functions = {
                "amplin_raw": lambda: amplin_op(input, canonical_qweight, canonical_scales),
            }
            k12288_wide_legal = (
                amplin_k12288_wide_op is not None
                and size_m == 1
                and spec.size_k == 12288
                and spec.size_n % 16 == 0
            )
            if k12288_wide_legal:
                functions["amplin_k12288_wide_raw"] = lambda: amplin_k12288_wide_op(
                    input,
                    canonical_qweight,
                    canonical_scales,
                )
            multirow_legal = (
                amplin_multirow_op is not None
                and size_m in (2, 4, 8, 16)
                and spec.size_k in (3072, 4096, 12288)
                and spec.size_n % 16 == 0
            )
            if multirow_legal:
                functions["amplin_multirow_raw"] = lambda: amplin_multirow_op(
                    input,
                    canonical_qweight,
                    canonical_scales,
                )
            padded_m16_legal = padded_m16_shape_legal and 1 <= size_m <= 16
            if padded_m16_legal:
                functions["amplin_mma_lane_m16_n16_padded_raw"] = (
                    lambda: amplin_mma_lane_m16_n16_padded_op(
                        input,
                        packed_mma_lane_qweight,
                        packed_hmma_scales,
                        spec.size_n,
                    )
                )
            splitk4_m16_legal = splitk4_m16_shape_legal and 1 <= size_m <= 16
            if splitk4_m16_legal:
                functions["amplin_mma_lane_m16_n16_splitk4_raw"] = (
                    lambda: amplin_mma_lane_m16_n16_splitk4_op(
                        input,
                        packed_mma_lane_qweight,
                        packed_hmma_scales,
                        spec.size_n,
                    )
                )
            splitk8_m16_legal = splitk8_m16_shape_legal and 1 <= size_m <= 16
            if splitk8_m16_legal:
                functions["amplin_mma_lane_m16_n16_splitk8_raw"] = (
                    lambda: amplin_mma_lane_m16_n16_splitk8_op(
                        input,
                        packed_mma_lane_qweight,
                        packed_hmma_scales,
                        spec.size_n,
                    )
                )
            splitk12_m16_legal = splitk12_m16_shape_legal and 1 <= size_m <= 16
            if splitk12_m16_legal:
                functions["amplin_mma_lane_m16_n16_splitk12_raw"] = (
                    lambda: amplin_mma_lane_m16_n16_splitk12_op(
                        input,
                        packed_mma_lane_qweight,
                        packed_hmma_scales,
                        spec.size_n,
                    )
                )
            splitk12_n32_legal = splitk12_n32_shape_legal and 1 <= size_m <= 16
            if splitk12_n32_legal:
                functions["amplin_mma_lane_m16_n32_splitk12_raw"] = (
                    lambda: amplin_mma_lane_m16_n32_splitk12_op(
                        input,
                        packed_mma_lane_qweight,
                        packed_hmma_scales,
                        spec.size_n,
                    )
                )
            splitk16_n32_legal = splitk16_n32_shape_legal and 1 <= size_m <= 16
            if splitk16_n32_legal:
                functions["amplin_mma_lane_m16_n32_splitk16_raw"] = (
                    lambda: amplin_mma_lane_m16_n32_splitk16_op(
                        input,
                        packed_mma_lane_qweight,
                        packed_hmma_scales,
                        spec.size_n,
                    )
                )
            splitk12_n32_pipe2_legal = (
                splitk12_n32_pipe2_shape_legal and 1 <= size_m <= 16
            )
            if splitk12_n32_pipe2_legal:
                functions["amplin_mma_lane_m16_n32_splitk12_pipe2_raw"] = (
                    lambda: amplin_mma_lane_m16_n32_splitk12_pipe2_op(
                        input,
                        packed_mma_lane_qweight,
                        packed_hmma_scales,
                        spec.size_n,
                    )
                )
            splitk8_n32_pipe2_legal = (
                splitk8_n32_pipe2_shape_legal and 1 <= size_m <= 16
            )
            if splitk8_n32_pipe2_legal:
                functions["amplin_mma_lane_m16_n32_splitk8_pipe2_raw"] = (
                    lambda: amplin_mma_lane_m16_n32_splitk8_pipe2_op(
                        input,
                        packed_mma_lane_qweight,
                        packed_hmma_scales,
                        spec.size_n,
                    )
                )
            splitk12_n32_interleaved_legal = (
                splitk12_n32_interleaved_shape_legal and 1 <= size_m <= 16
            )
            if splitk12_n32_interleaved_legal:
                functions["amplin_mma_lane_m16_n32_splitk12_pipe2_interleaved_raw"] = (
                    lambda: amplin_mma_lane_m16_n32_splitk12_pipe2_interleaved_op(
                        input,
                        packed_mma_lane_n32_qweight,
                        packed_hmma_scales,
                        spec.size_n,
                    )
                )
            splitk24_n64_interleaved_legal = (
                splitk24_n64_interleaved_shape_legal and 1 <= size_m <= 16
            )
            if splitk24_n64_interleaved_legal:
                functions["amplin_mma_lane_m16_n64_splitk24_pipe2_interleaved_raw"] = (
                    lambda: amplin_mma_lane_m16_n64_splitk24_pipe2_interleaved_op(
                        input,
                        packed_mma_lane_n64_qweight,
                        packed_hmma_scales,
                        spec.size_n,
                    )
                )
            splitk12x2_n64_coop_legal = (
                splitk12x2_n64_coop_shape_legal and 1 <= size_m <= 16
            )
            if splitk12x2_n64_coop_legal:
                functions["amplin_mma_lane_m16_n64_splitk12x2_coop_interleaved_raw"] = (
                    lambda: amplin_mma_lane_m16_n64_splitk12x2_coop_interleaved_op(
                        input,
                        packed_mma_lane_n64_qweight,
                        packed_hmma_scales,
                        spec.size_n,
                    )
                )
            m32_splitk12x2_n64_coop_legal = (
                m32_splitk12x2_n64_coop_shape_legal and 17 <= size_m <= 32
            )
            if m32_splitk12x2_n64_coop_legal:
                functions["amplin_mma_lane_m32_n64_splitk12x2_coop_interleaved_raw"] = (
                    lambda: amplin_mma_lane_m32_n64_splitk12x2_coop_interleaved_op(
                        input,
                        packed_mma_lane_n64_qweight,
                        packed_hmma_scales,
                        spec.size_n,
                    )
                )
            m32_splitk24_n64_legal = (
                m32_splitk24_n64_shape_legal and 17 <= size_m <= 32
            )
            if m32_splitk24_n64_legal:
                functions["amplin_mma_lane_m32_n64_splitk24_pipe2_interleaved_raw"] = (
                    lambda: amplin_mma_lane_m32_n64_splitk24_pipe2_interleaved_op(
                        input,
                        packed_mma_lane_n64_qweight,
                        packed_hmma_scales,
                        spec.size_n,
                    )
                )
            splitk16_n32_pipe2_legal = (
                splitk16_n32_pipe2_shape_legal and 1 <= size_m <= 16
            )
            if splitk16_n32_pipe2_legal:
                functions["amplin_mma_lane_m16_n32_splitk16_pipe2_raw"] = (
                    lambda: amplin_mma_lane_m16_n32_splitk16_pipe2_op(
                        input,
                        packed_mma_lane_qweight,
                        packed_hmma_scales,
                        spec.size_n,
                    )
                )
            splitk16_m16_legal = splitk16_m16_shape_legal and 1 <= size_m <= 16
            if splitk16_m16_legal:
                functions["amplin_mma_lane_m16_n16_splitk16_raw"] = (
                    lambda: amplin_mma_lane_m16_n16_splitk16_op(
                        input,
                        packed_mma_lane_qweight,
                        packed_hmma_scales,
                        spec.size_n,
                    )
                )
            hmma_legal = hmma_shape_legal and size_m % 16 == 0
            m64_grid_ctas = (
                (spec.size_n // amplin.HMMA_N_TILE) * (size_m // 64)
                if hmma_legal and size_m % 64 == 0
                else 0
            )
            use_m64_reuse = m64_grid_ctas >= sm_count
            hmma_selected_schedule = "m64-reuse-v3-async-a" if use_m64_reuse else "m16-v0"
            if hmma_legal:
                functions["amplin_hmma_v0_raw"] = lambda: amplin_hmma_v0_op(
                    input,
                    packed_hmma_qweight,
                    packed_hmma_scales,
                    spec.size_n,
                )
                functions["amplin_hmma_raw"] = lambda: amplin_hmma_op(
                    input,
                    packed_hmma_qweight,
                    packed_hmma_scales,
                    spec.size_n,
                )
                if size_m % 64 == 0:
                    functions["amplin_hmma_m64_v1_raw"] = lambda: amplin_hmma_m64_v1_op(
                        input,
                        packed_hmma_qweight,
                        packed_hmma_scales,
                        spec.size_n,
                    )
                    functions["amplin_hmma_m64_v2_raw"] = lambda: amplin_hmma_m64_v2_op(
                        input,
                        packed_hmma_qweight,
                        packed_hmma_scales,
                        spec.size_n,
                    )
                    functions["amplin_hmma_m64_v2_sync_a128_raw"] = (
                        lambda: amplin_hmma_m64_v2_sync_a128_op(
                            input,
                            packed_hmma_qweight,
                            packed_hmma_scales,
                            spec.size_n,
                        )
                    )
                    functions["amplin_hmma_m64_v3_raw"] = lambda: amplin_hmma_m64_v3_op(
                        input,
                        packed_hmma_qweight,
                        packed_hmma_scales,
                        spec.size_n,
                    )
                    functions["amplin_mma_lane_m64_raw"] = lambda: amplin_mma_lane_m64_op(
                        input,
                        packed_mma_lane_qweight,
                        packed_hmma_scales,
                        spec.size_n,
                    )
                    functions["amplin_mma_lane_m64_global_a_raw"] = (
                        lambda: amplin_mma_lane_m64_global_a_op(
                            input,
                            packed_mma_lane_qweight,
                            packed_hmma_scales,
                            spec.size_n,
                        )
                    )
                if size_m % 32 == 0:
                    functions["amplin_mma_lane_m32_global_a_raw"] = (
                        lambda: amplin_mma_lane_m32_global_a_op(
                            input,
                            packed_mma_lane_qweight,
                            packed_hmma_scales,
                            spec.size_n,
                        )
                    )
            if mma_lane_n32_shape_legal and size_m % 32 == 0:
                functions["amplin_mma_lane_m32_n32_global_a_raw"] = (
                    lambda: amplin_mma_lane_m32_n32_global_a_op(
                        input,
                        packed_mma_lane_qweight,
                        packed_hmma_scales,
                        spec.size_n,
                    )
                )
            if marlin_module is not None:
                functions["marlin_raw"] = lambda: _raw_marlin_call(
                    op=marlin_op,
                    input=input,
                    module=marlin_module,
                )
            if dense_weight is not None:
                functions["dense_resident"] = lambda: input @ dense_weight
            if m32_sweep:
                retained_names = {
                    "amplin_mma_lane_m64_global_a_raw",
                    "amplin_mma_lane_m32_global_a_raw",
                    "amplin_mma_lane_m32_n32_global_a_raw",
                    "marlin_raw",
                }
                if size_m < 32:
                    retained_names.add("amplin_raw")
                if m32_splitk12x2_n64_coop_legal:
                    retained_names.add(
                        "amplin_mma_lane_m32_n64_splitk12x2_coop_interleaved_raw"
                    )
                if m32_splitk24_n64_legal:
                    retained_names.add(
                        "amplin_mma_lane_m32_n64_splitk24_pipe2_interleaved_raw"
                    )
                functions = {
                    name: function
                    for name, function in functions.items()
                    if name in retained_names
                }

            errors = {}
            error_limit = 2e-3 if dtype == torch.float16 else 2e-2
            for name, function in functions.items():
                output = function()
                torch.cuda.synchronize(device)
                error = (output.to(torch.float32) - reference).abs()
                max_error = error.max().item()
                mean_error = error.mean().item()
                if not torch.isfinite(output).all() or max_error > error_limit:
                    raise AssertionError(
                        f"{name} failed for {spec.model} {spec.role}, M={size_m}, "
                        f"K={spec.size_k}, N={spec.size_n}, dtype={dtype}: "
                        f"finite={bool(torch.isfinite(output).all())}, "
                        f"max_abs={max_error}, limit={error_limit}"
                    )
                errors[name] = {"max_abs": max_error, "mean_abs": mean_error}

            timing = _measure(
                functions,
                device=device,
                warmup=warmup,
                iters=iters,
                rounds=rounds,
                pre_timing_check=pre_timing_check,
            )
            activation_bytes = input.numel() * input.element_size()
            output_bytes = size_m * spec.size_n * input.element_size()
            minimum_gptq_bytes = (
                canonical_weight_bytes + canonical_scale_bytes + activation_bytes + output_bytes
            )
            row_independent_bytes = (
                size_m * (canonical_weight_bytes + canonical_scale_bytes)
                + activation_bytes
                + output_bytes
            )
            multirow_rows = 2 if size_m == 2 else 4
            multirow_requested_bytes = (
                (size_m // multirow_rows) * (canonical_weight_bytes + canonical_scale_bytes)
                + activation_bytes
                + output_bytes
            )
            mma_lane_m16_n16_padded_requested_bytes = (
                canonical_weight_bytes
                + 4 * canonical_scale_bytes
                + (spec.size_n // 16) * activation_bytes
                + output_bytes
            )
            mma_lane_m16_n32_splitk12_requested_bytes = (
                canonical_weight_bytes
                + 4 * canonical_scale_bytes
                + (spec.size_n // 32) * activation_bytes
                + output_bytes
            )
            mma_lane_m16_n64_splitk24_requested_bytes = (
                canonical_weight_bytes
                + 4 * canonical_scale_bytes
                + (spec.size_n // 64) * activation_bytes
                + output_bytes
            )
            mma_lane_m32_n64_splitk24_requested_bytes = (
                canonical_weight_bytes
                + 4 * canonical_scale_bytes
                + (spec.size_n // 64) * activation_bytes
                + output_bytes
            )
            hmma_v0_requested_bytes = (
                (size_m // 16) * (canonical_weight_bytes + canonical_scale_bytes)
                + (spec.size_n // amplin.HMMA_N_TILE) * activation_bytes
                + output_bytes
            )
            hmma_m64_requested_bytes = (
                (size_m // 64) * (canonical_weight_bytes + canonical_scale_bytes)
                + (spec.size_n // amplin.HMMA_N_TILE) * activation_bytes
                + output_bytes
            )
            mma_lane_m64_requested_bytes = (
                (size_m // 64) * (2 * canonical_weight_bytes + 8 * canonical_scale_bytes)
                + (spec.size_n // amplin.HMMA_N_TILE) * activation_bytes
                + output_bytes
            )
            mma_lane_m64_global_a_requested_bytes = (
                (size_m // 64) * (2 * canonical_weight_bytes + 8 * canonical_scale_bytes)
                + 4 * (spec.size_n // amplin.HMMA_N_TILE) * activation_bytes
                + output_bytes
            )
            mma_lane_m32_global_a_requested_bytes = (
                (size_m // 32) * (2 * canonical_weight_bytes + 8 * canonical_scale_bytes)
                + 2 * (spec.size_n // amplin.HMMA_N_TILE) * activation_bytes
                + output_bytes
            )
            n32_tiles = (spec.size_n + 31) // 32
            padded_n32 = n32_tiles * 32
            mma_lane_n32_weight_bytes = (spec.size_k // 8) * padded_n32 * torch.int32.itemsize
            mma_lane_n32_scale_bytes = (
                (spec.size_k // GROUP_SIZE) * padded_n32 * canonical_scales.element_size()
            )
            mma_lane_m32_n32_global_a_requested_bytes = (
                (size_m // 32) * (2 * mma_lane_n32_weight_bytes + 8 * mma_lane_n32_scale_bytes)
                + n32_tiles * activation_bytes
                + output_bytes
            )
            selected_hmma_block_m = 64 if use_m64_reuse else 16
            hmma_requested_bytes = (
                (size_m // selected_hmma_block_m) * (canonical_weight_bytes + canonical_scale_bytes)
                + (spec.size_n // amplin.HMMA_N_TILE) * activation_bytes
                + output_bytes
            )
            dense_bytes = dense_weight_bytes + activation_bytes + output_bytes
            marlin_time = timing["marlin_raw"].batch_event_median_us if "marlin_raw" in timing else None
            scalar_amplin_time = (
                timing["amplin_raw"].batch_event_median_us
                if "amplin_raw" in timing
                else None
            )
            hmma_v0_time = (
                timing["amplin_hmma_v0_raw"].batch_event_median_us
                if "amplin_hmma_v0_raw" in timing
                else None
            )
            hmma_m64_v1_time = (
                timing["amplin_hmma_m64_v1_raw"].batch_event_median_us
                if "amplin_hmma_m64_v1_raw" in timing
                else None
            )
            hmma_m64_v2_time = (
                timing["amplin_hmma_m64_v2_raw"].batch_event_median_us
                if "amplin_hmma_m64_v2_raw" in timing
                else None
            )
            hmma_m64_v2_sync_a128_time = (
                timing["amplin_hmma_m64_v2_sync_a128_raw"].batch_event_median_us
                if "amplin_hmma_m64_v2_sync_a128_raw" in timing
                else None
            )
            hmma_m64_v3_time = (
                timing["amplin_hmma_m64_v3_raw"].batch_event_median_us
                if "amplin_hmma_m64_v3_raw" in timing
                else None
            )
            mma_lane_m64_time = (
                timing["amplin_mma_lane_m64_raw"].batch_event_median_us
                if "amplin_mma_lane_m64_raw" in timing
                else None
            )
            mma_lane_m64_global_a_time = (
                timing["amplin_mma_lane_m64_global_a_raw"].batch_event_median_us
                if "amplin_mma_lane_m64_global_a_raw" in timing
                else None
            )
            mma_lane_m32_global_a_time = (
                timing["amplin_mma_lane_m32_global_a_raw"].batch_event_median_us
                if "amplin_mma_lane_m32_global_a_raw" in timing
                else None
            )

            for name, stats in timing.items():
                if name == "amplin_raw":
                    requested_bytes = row_independent_bytes
                elif name == "amplin_k12288_wide_raw":
                    requested_bytes = minimum_gptq_bytes
                elif name == "amplin_multirow_raw":
                    requested_bytes = multirow_requested_bytes
                elif name in (
                    "amplin_mma_lane_m16_n16_padded_raw",
                    "amplin_mma_lane_m16_n16_splitk4_raw",
                    "amplin_mma_lane_m16_n16_splitk8_raw",
                    "amplin_mma_lane_m16_n16_splitk12_raw",
                    "amplin_mma_lane_m16_n16_splitk16_raw",
                ):
                    requested_bytes = mma_lane_m16_n16_padded_requested_bytes
                elif name in (
                    "amplin_mma_lane_m16_n32_splitk12_raw",
                    "amplin_mma_lane_m16_n32_splitk16_raw",
                    "amplin_mma_lane_m16_n32_splitk12_pipe2_raw",
                    "amplin_mma_lane_m16_n32_splitk12_pipe2_interleaved_raw",
                    "amplin_mma_lane_m16_n32_splitk16_pipe2_raw",
                ):
                    requested_bytes = mma_lane_m16_n32_splitk12_requested_bytes
                elif name in (
                    "amplin_mma_lane_m16_n64_splitk24_pipe2_interleaved_raw",
                    "amplin_mma_lane_m16_n64_splitk12x2_coop_interleaved_raw",
                ):
                    requested_bytes = mma_lane_m16_n64_splitk24_requested_bytes
                elif name == "amplin_mma_lane_m32_n64_splitk24_pipe2_interleaved_raw":
                    requested_bytes = mma_lane_m32_n64_splitk24_requested_bytes
                elif name == "amplin_hmma_raw":
                    requested_bytes = hmma_requested_bytes
                elif name == "amplin_hmma_v0_raw":
                    requested_bytes = hmma_v0_requested_bytes
                elif name in (
                    "amplin_hmma_m64_v1_raw",
                    "amplin_hmma_m64_v2_raw",
                    "amplin_hmma_m64_v2_sync_a128_raw",
                    "amplin_hmma_m64_v3_raw",
                ):
                    requested_bytes = hmma_m64_requested_bytes
                elif name == "amplin_mma_lane_m64_raw":
                    requested_bytes = mma_lane_m64_requested_bytes
                elif name == "amplin_mma_lane_m64_global_a_raw":
                    requested_bytes = mma_lane_m64_global_a_requested_bytes
                elif name == "amplin_mma_lane_m32_global_a_raw":
                    requested_bytes = mma_lane_m32_global_a_requested_bytes
                elif name == "amplin_mma_lane_m32_n32_global_a_raw":
                    requested_bytes = mma_lane_m32_n32_global_a_requested_bytes
                elif name == "dense_resident":
                    requested_bytes = dense_bytes
                else:
                    requested_bytes = minimum_gptq_bytes
                rows.append(
                    {
                        "model": spec.model,
                        "role": spec.role,
                        "dtype": _dtype_name(dtype),
                        "m": size_m,
                        "k": spec.size_k,
                        "n": spec.size_n,
                        "path": name,
                        "marlin_legal": marlin_legal,
                        "hmma_legal": hmma_legal,
                        "hmma_selected_schedule": hmma_selected_schedule if hmma_legal else None,
                        "hmma_m64_grid_ctas": m64_grid_ctas if hmma_legal else None,
                        "selected_device_sm_count": sm_count,
                        **asdict(stats),
                        "minimum_gptq_bytes": minimum_gptq_bytes,
                        "row_independent_requested_bytes": row_independent_bytes,
                        "mma_lane_m16_n16_padded_requested_bytes": (
                            mma_lane_m16_n16_padded_requested_bytes
                        ),
                        "mma_lane_m16_n32_splitk12_requested_bytes": (
                            mma_lane_m16_n32_splitk12_requested_bytes
                        ),
                        "mma_lane_m16_n64_splitk24_requested_bytes": (
                            mma_lane_m16_n64_splitk24_requested_bytes
                        ),
                        "mma_lane_m32_n64_splitk24_requested_bytes": (
                            mma_lane_m32_n64_splitk24_requested_bytes
                        ),
                        "hmma_v0_requested_bytes": hmma_v0_requested_bytes,
                        "hmma_m64_requested_bytes": hmma_m64_requested_bytes,
                        "mma_lane_m64_requested_bytes": mma_lane_m64_requested_bytes,
                        "mma_lane_m64_global_a_requested_bytes": mma_lane_m64_global_a_requested_bytes,
                        "mma_lane_m32_global_a_requested_bytes": mma_lane_m32_global_a_requested_bytes,
                        "mma_lane_m32_n32_global_a_requested_bytes": (
                            mma_lane_m32_n32_global_a_requested_bytes
                        ),
                        "hmma_requested_bytes": hmma_requested_bytes,
                        "dense_resident_bytes": dense_bytes,
                        "effective_requested_gbs": requested_bytes / (stats.batch_event_median_us * 1000.0),
                        "speedup_vs_marlin_raw": (
                            marlin_time / stats.batch_event_median_us if marlin_time is not None else None
                        ),
                        "speedup_vs_amplin_raw": (
                            scalar_amplin_time / stats.batch_event_median_us
                            if scalar_amplin_time is not None
                            else None
                        ),
                        "speedup_vs_hmma_v0_raw": (
                            hmma_v0_time / stats.batch_event_median_us if hmma_v0_time is not None else None
                        ),
                        "speedup_vs_hmma_m64_v1_raw": (
                            hmma_m64_v1_time / stats.batch_event_median_us
                            if hmma_m64_v1_time is not None
                            else None
                        ),
                        "speedup_vs_hmma_m64_v2_raw": (
                            hmma_m64_v2_time / stats.batch_event_median_us
                            if hmma_m64_v2_time is not None
                            else None
                        ),
                        "speedup_vs_hmma_m64_v2_sync_a128_raw": (
                            hmma_m64_v2_sync_a128_time / stats.batch_event_median_us
                            if hmma_m64_v2_sync_a128_time is not None
                            else None
                        ),
                        "speedup_vs_hmma_m64_v3_raw": (
                            hmma_m64_v3_time / stats.batch_event_median_us
                            if hmma_m64_v3_time is not None
                            else None
                        ),
                        "speedup_vs_mma_lane_m64_raw": (
                            mma_lane_m64_time / stats.batch_event_median_us
                            if mma_lane_m64_time is not None
                            else None
                        ),
                        "speedup_vs_mma_lane_m64_global_a_raw": (
                            mma_lane_m64_global_a_time / stats.batch_event_median_us
                            if mma_lane_m64_global_a_time is not None
                            else None
                        ),
                        "speedup_vs_mma_lane_m32_global_a_raw": (
                            mma_lane_m32_global_a_time / stats.batch_event_median_us
                            if mma_lane_m32_global_a_time is not None
                            else None
                        ),
                        "error_vs_fp32_dequant": errors[name],
                    }
                )
    return rows


def _print_results(rows: list[dict]) -> None:
    table = []
    for row in rows:
        marlin_speedup = row["speedup_vs_marlin_raw"]
        amplin_speedup = row["speedup_vs_amplin_raw"]
        hmma_v0_speedup = row["speedup_vs_hmma_v0_raw"]
        hmma_m64_v1_speedup = row["speedup_vs_hmma_m64_v1_raw"]
        hmma_m64_v2_speedup = row["speedup_vs_hmma_m64_v2_raw"]
        hmma_m64_v2_sync_a128_speedup = row["speedup_vs_hmma_m64_v2_sync_a128_raw"]
        hmma_m64_v3_speedup = row["speedup_vs_hmma_m64_v3_raw"]
        mma_lane_m64_speedup = row["speedup_vs_mma_lane_m64_raw"]
        mma_lane_m64_global_a_speedup = row["speedup_vs_mma_lane_m64_global_a_raw"]
        mma_lane_m32_global_a_speedup = row["speedup_vs_mma_lane_m32_global_a_raw"]
        table.append(
            [
                row["model"],
                row["role"],
                row["dtype"],
                row["m"],
                row["k"],
                row["n"],
                row["path"],
                row["hmma_selected_schedule"] or "-",
                "yes" if row["marlin_legal"] else "no",
                f"{row['median_us']:.3f}",
                f"{row['p95_us']:.3f}",
                f"{row['batch_event_median_us']:.3f}",
                f"{row['batch_event_mean_us']:.3f}",
                f"{row['batch_event_min_us']:.3f}-{row['batch_event_max_us']:.3f}",
                f"{row['wall_median_us']:.3f}",
                f"{row['wall_mean_us']:.3f}",
                f"{row['effective_requested_gbs']:.2f}",
                f"{amplin_speedup:.3f}x" if amplin_speedup is not None else "-",
                f"{hmma_v0_speedup:.3f}x" if hmma_v0_speedup is not None else "-",
                f"{hmma_m64_v1_speedup:.3f}x" if hmma_m64_v1_speedup is not None else "-",
                f"{hmma_m64_v2_speedup:.3f}x" if hmma_m64_v2_speedup is not None else "-",
                (
                    f"{hmma_m64_v2_sync_a128_speedup:.3f}x"
                    if hmma_m64_v2_sync_a128_speedup is not None
                    else "-"
                ),
                f"{hmma_m64_v3_speedup:.3f}x" if hmma_m64_v3_speedup is not None else "-",
                f"{mma_lane_m64_speedup:.3f}x" if mma_lane_m64_speedup is not None else "-",
                (
                    f"{mma_lane_m64_global_a_speedup:.3f}x"
                    if mma_lane_m64_global_a_speedup is not None
                    else "-"
                ),
                (
                    f"{mma_lane_m32_global_a_speedup:.3f}x"
                    if mma_lane_m32_global_a_speedup is not None
                    else "-"
                ),
                f"{marlin_speedup:.3f}x" if marlin_speedup is not None else "-",
                f"{row['error_vs_fp32_dequant']['max_abs']:.7f}",
            ]
        )
    print(
        tabulate(
            table,
            headers=(
                "model",
                "role",
                "dtype",
                "M",
                "K",
                "N",
                "path",
                "HMMA schedule",
                "Marlin legal",
                "p50 us",
                "p95 us",
                "batch median us",
                "batch mean us",
                "batch range us",
                "wall median us",
                "wall mean us",
                "requested GB/s",
                "vs scalar",
                "vs HMMA V0",
                "vs M64 V1",
                "vs M64 V2",
                "vs sync A128",
                "vs M64 V3",
                "vs lane M64",
                "vs direct-A",
                "vs M32 N64",
                "vs Marlin",
                "max abs",
            ),
            tablefmt="grid",
        )
    )


def main() -> None:
    args = _parse_args()
    m_values = _parse_m_values(args.m_values)
    shape_filters = _parse_shape_filters(args.shape)
    shapes = _select_shapes(args.model, shape_filters)
    if args.warmup < 1 or args.iters < 2 or args.rounds < 1:
        raise ValueError("--warmup must be positive, --iters at least 2, and --rounds positive")
    if args.m32_sweep and not args.hmma:
        raise ValueError("--m32-sweep requires --hmma")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Amplin model-shape benchmarking requires a CUDA device")
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    if (properties.major, properties.minor) != (8, 0):
        raise RuntimeError(
            f"Amplin model-shape benchmark requires compute capability 8.0, "
            f"got {properties.major}.{properties.minor}"
        )
    dtypes = _resolve_dtypes(args.dtype)
    if torch.bfloat16 in dtypes and not torch.cuda.is_bf16_supported():
        raise RuntimeError("requested BF16 benchmark but the selected CUDA device does not support BF16")

    hardware = {
        "device_argument": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "sm_count": properties.multi_processor_count,
        "total_memory_bytes": properties.total_memory,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "git_revision": _git_revision(),
        "nvidia_smi_inventory": _nvidia_smi_inventory(),
        "gpu_idle_preflight": (
            _GPU_IDLE_PREFLIGHT.as_dict() if _GPU_IDLE_PREFLIGHT is not None else None
        ),
    }
    print(json.dumps(hardware, indent=2))

    all_rows = []
    for dtype_index, dtype in enumerate(dtypes):
        amplin_op = extension.op("amplin", "gemv")
        amplin_k12288_wide_op = (
            extension.op("amplin", "gemv_k12288_wide")
            if args.k12288_wide
            else None
        )
        amplin_multirow_op = (
            extension.op("amplin", "gemv_multirow")
            if args.multirow
            else None
        )
        amplin_hmma_op = extension.op("amplin", "gemm_hmma") if args.hmma else None
        amplin_hmma_v0_op = extension.op("amplin", "gemm_hmma_v0") if args.hmma else None
        amplin_hmma_m64_v1_op = extension.op("amplin", "gemm_hmma_m64_v1") if args.hmma else None
        amplin_hmma_m64_v2_op = extension.op("amplin", "gemm_hmma_m64_v2") if args.hmma else None
        amplin_hmma_m64_v2_sync_a128_op = (
            extension.op("amplin", "gemm_hmma_m64_v2_sync_a128")
            if args.hmma
            else None
        )
        amplin_hmma_m64_v3_op = extension.op("amplin", "gemm_hmma_m64_v3") if args.hmma else None
        amplin_mma_lane_m64_op = extension.op("amplin", "mma_lane_m64") if args.hmma else None
        amplin_mma_lane_m64_global_a_op = (
            extension.op("amplin", "mma_lane_m64_global_a")
            if args.hmma
            else None
        )
        amplin_mma_lane_m32_global_a_op = (
            extension.op("amplin", "mma_lane_m32_global_a")
            if args.hmma
            else None
        )
        amplin_mma_lane_m32_n32_global_a_op = (
            extension.op("amplin", "mma_lane_m32_n32_global_a")
            if args.hmma
            else None
        )
        amplin_mma_lane_m32_n64_splitk12x2_coop_interleaved_op = (
            extension.op("amplin", "mma_lane_m32_n64_splitk12x2_coop_interleaved")
            if args.m32_splitk12x2_n64_coop
            else None
        )
        amplin_mma_lane_m32_n64_splitk24_pipe2_interleaved_op = (
            extension.op("amplin", "mma_lane_m32_n64_splitk24_pipe2_interleaved")
            if args.m32_splitk24_n64
            else None
        )
        amplin_mma_lane_m16_n16_padded_op = (
            extension.op("amplin", "mma_lane_m16_n16_padded")
            if args.padded_m16
            else None
        )
        amplin_mma_lane_m16_n16_splitk4_op = (
            extension.op("amplin", "mma_lane_m16_n16_splitk4")
            if args.splitk4_m16
            else None
        )
        amplin_mma_lane_m16_n16_splitk8_op = (
            extension.op("amplin", "mma_lane_m16_n16_splitk8")
            if args.splitk8_m16
            else None
        )
        amplin_mma_lane_m16_n16_splitk12_op = (
            extension.op("amplin", "mma_lane_m16_n16_splitk12")
            if args.splitk12_m16
            else None
        )
        amplin_mma_lane_m16_n32_splitk12_op = (
            extension.op("amplin", "mma_lane_m16_n32_splitk12")
            if args.splitk12_n32
            else None
        )
        amplin_mma_lane_m16_n32_splitk16_op = (
            extension.op("amplin", "mma_lane_m16_n32_splitk16")
            if args.splitk16_n32
            else None
        )
        amplin_mma_lane_m16_n32_splitk8_pipe2_op = (
            extension.op("amplin", "mma_lane_m16_n32_splitk8_pipe2")
            if args.splitk_n32_pipe2
            else None
        )
        amplin_mma_lane_m16_n32_splitk12_pipe2_op = (
            extension.op("amplin", "mma_lane_m16_n32_splitk12_pipe2")
            if args.splitk_n32_pipe2
            else None
        )
        amplin_mma_lane_m16_n32_splitk12_pipe2_interleaved_op = (
            extension.op("amplin", "mma_lane_m16_n32_splitk12_pipe2_interleaved")
            if args.splitk12_n32_interleaved
            else None
        )
        amplin_mma_lane_m16_n64_splitk24_pipe2_interleaved_op = (
            extension.op("amplin", "mma_lane_m16_n64_splitk24_pipe2_interleaved")
            if args.splitk24_n64_interleaved
            else None
        )
        amplin_mma_lane_m16_n64_splitk12x2_coop_interleaved_op = (
            extension.op("amplin", "mma_lane_m16_n64_splitk12x2_coop_interleaved")
            if args.splitk12x2_n64_coop
            else None
        )
        amplin_mma_lane_m16_n32_splitk16_pipe2_op = (
            extension.op("amplin", "mma_lane_m16_n32_splitk16_pipe2")
            if args.splitk_n32_pipe2
            else None
        )
        amplin_mma_lane_m16_n16_splitk16_op = (
            extension.op("amplin", "mma_lane_m16_n16_splitk16")
            if args.splitk16_m16
            else None
        )
        marlin_extension = "marlin_fp16" if dtype == torch.float16 else "marlin_bf16"
        marlin_op_name = "gptq_marlin_gemm_fp16" if dtype == torch.float16 else "gptq_marlin_gemm_bf16"
        marlin_op = extension.op(marlin_extension, marlin_op_name)
        for shape_index, spec in enumerate(shapes):
            all_rows.extend(
                _benchmark_shape_dtype(
                    spec=spec,
                    dtype=dtype,
                    m_values=m_values,
                    device=device,
                    seed=args.seed + dtype_index * 1000 + shape_index,
                    warmup=args.warmup,
                    iters=args.iters,
                    rounds=args.rounds,
                    dense_ceiling=args.dense_ceiling,
                    amplin_op=amplin_op,
                    amplin_k12288_wide_op=amplin_k12288_wide_op,
                    amplin_multirow_op=amplin_multirow_op,
                    amplin_hmma_op=amplin_hmma_op,
                    amplin_hmma_v0_op=amplin_hmma_v0_op,
                    amplin_hmma_m64_v1_op=amplin_hmma_m64_v1_op,
                    amplin_hmma_m64_v2_op=amplin_hmma_m64_v2_op,
                    amplin_hmma_m64_v2_sync_a128_op=amplin_hmma_m64_v2_sync_a128_op,
                    amplin_hmma_m64_v3_op=amplin_hmma_m64_v3_op,
                    amplin_mma_lane_m64_op=amplin_mma_lane_m64_op,
                    amplin_mma_lane_m64_global_a_op=amplin_mma_lane_m64_global_a_op,
                    amplin_mma_lane_m32_global_a_op=amplin_mma_lane_m32_global_a_op,
                    amplin_mma_lane_m32_n32_global_a_op=amplin_mma_lane_m32_n32_global_a_op,
                    amplin_mma_lane_m32_n64_splitk12x2_coop_interleaved_op=amplin_mma_lane_m32_n64_splitk12x2_coop_interleaved_op,
                    amplin_mma_lane_m32_n64_splitk24_pipe2_interleaved_op=amplin_mma_lane_m32_n64_splitk24_pipe2_interleaved_op,
                    amplin_mma_lane_m16_n16_padded_op=amplin_mma_lane_m16_n16_padded_op,
                    amplin_mma_lane_m16_n16_splitk4_op=amplin_mma_lane_m16_n16_splitk4_op,
                    amplin_mma_lane_m16_n16_splitk8_op=amplin_mma_lane_m16_n16_splitk8_op,
                    amplin_mma_lane_m16_n16_splitk12_op=amplin_mma_lane_m16_n16_splitk12_op,
                    amplin_mma_lane_m16_n32_splitk12_op=amplin_mma_lane_m16_n32_splitk12_op,
                    amplin_mma_lane_m16_n32_splitk16_op=amplin_mma_lane_m16_n32_splitk16_op,
                    amplin_mma_lane_m16_n32_splitk12_pipe2_op=(
                        amplin_mma_lane_m16_n32_splitk12_pipe2_op
                    ),
                    amplin_mma_lane_m16_n32_splitk8_pipe2_op=(
                        amplin_mma_lane_m16_n32_splitk8_pipe2_op
                    ),
                    amplin_mma_lane_m16_n32_splitk12_pipe2_interleaved_op=(
                        amplin_mma_lane_m16_n32_splitk12_pipe2_interleaved_op
                    ),
                    amplin_mma_lane_m16_n64_splitk24_pipe2_interleaved_op=(
                        amplin_mma_lane_m16_n64_splitk24_pipe2_interleaved_op
                    ),
                    amplin_mma_lane_m16_n64_splitk12x2_coop_interleaved_op=(
                        amplin_mma_lane_m16_n64_splitk12x2_coop_interleaved_op
                    ),
                    amplin_mma_lane_m16_n32_splitk16_pipe2_op=(
                        amplin_mma_lane_m16_n32_splitk16_pipe2_op
                    ),
                    amplin_mma_lane_m16_n16_splitk16_op=amplin_mma_lane_m16_n16_splitk16_op,
                    marlin_op=marlin_op,
                    m32_sweep=args.m32_sweep,
                    pre_timing_check=(
                        (lambda: recheck_gpu_exclusivity(_GPU_IDLE_PREFLIGHT))
                        if _GPU_IDLE_PREFLIGHT is not None
                        else None
                    ),
                )
            )

    _print_results(all_rows)
    payload = {
        "hardware": hardware,
        "benchmark": {
            "models": args.model,
            "shape_filters": sorted(shape_filters) if shape_filters is not None else None,
            "m_values": m_values,
            "warmup": args.warmup,
            "iters_per_round": args.iters,
            "rounds": args.rounds,
            "gate_quality_round_count": args.rounds >= 5,
            "primary_statistic": "median batched CUDA-event time across rounds",
            "dense_ceiling": args.dense_ceiling,
            "k12288_wide": args.k12288_wide,
            "multirow": args.multirow,
            "padded_m16": args.padded_m16,
            "splitk4_m16": args.splitk4_m16,
            "splitk8_m16": args.splitk8_m16,
            "splitk12_m16": args.splitk12_m16,
            "splitk12_n32": args.splitk12_n32,
            "splitk16_n32": args.splitk16_n32,
            "splitk_n32_pipe2": args.splitk_n32_pipe2,
            "splitk12_n32_interleaved": args.splitk12_n32_interleaved,
            "splitk24_n64_interleaved": args.splitk24_n64_interleaved,
            "splitk12x2_n64_coop": args.splitk12x2_n64_coop,
            "m32_splitk24_n64": args.m32_splitk24_n64,
            "splitk16_m16": args.splitk16_m16,
            "hmma": args.hmma,
            "m32_sweep": args.m32_sweep,
            "quantization": {
                "bits": BITS,
                "group_size": GROUP_SIZE,
                "sym": True,
                "desc_act": False,
                "pack_dtype": "torch.int32",
            },
            "marlin_constraint": "N must be divisible by 64; N=48 and N=72 are reported as not legal",
            "k12288_wide_constraint": "prototype requires M=1, K=12288, and N divisible by 16",
            "multirow_constraint": "prototype requires M=2,4,8,16, K=3072,4096,12288, and N divisible by 16",
            "padded_m16_constraint": "prototype requires M=2,4,8,16 and N divisible by 16",
            "splitk4_m16_constraint": "prototype requires M=2,4,8,16, K=12288, and N divisible by 16",
            "splitk8_m16_constraint": "prototype requires M=2,4,8,16, K=12288, and N divisible by 16",
            "splitk12_m16_constraint": "prototype requires M=2,4,8,16, K=12288, and N divisible by 16",
            "splitk12_n32_constraint": "prototype requires M=2,4,8,16, K=12288, and N divisible by 32",
            "splitk16_n32_constraint": "prototype requires M=2,4,8,16, K=12288, and N divisible by 32",
            "splitk_n32_pipe2_constraint": (
                "prototype requires M=2,4,8,16, K divisible by 1024, and N divisible by 32"
            ),
            "splitk12_n32_interleaved_constraint": (
                "prototype requires M=2,4,8,16, K=12288, N divisible by 32, "
                "and [N32,K128,K16,lane,word-pair] int32 weights"
            ),
            "splitk24_n64_interleaved_constraint": (
                "prototype requires M=2,4,8,16, K=12288, N divisible by 64, "
                "and [N64,K128,K16,lane,word-quad] int32 weights"
            ),
            "splitk12x2_n64_coop_constraint": (
                "prototype requires M=2,4,8,16, K=12288, N divisible by 64, "
                "sm_80 cooperative launch, no CUDA graph capture, and resident grid capacity"
            ),
            "m32_splitk24_n64_constraint": (
                "prototype requires M=17..32, K divisible by 128, N divisible by 64, "
                "sm_80, and 96 KB opt-in shared memory"
            ),
            "splitk16_m16_constraint": "prototype requires M=2,4,8,16, K=12288, and N divisible by 16",
            "mma_lane_n32_constraint": "M must be divisible by 32 and N by 8; N tails use padded lane weights",
        },
        "results": all_rows,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\njson_out={args.json_out}")


if __name__ == "__main__":
    main()
