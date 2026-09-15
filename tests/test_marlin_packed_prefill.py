# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

import pytest
import torch

from gptqmodel.nn_modules.qlinear.marlin import (
    _PACKED_PREFILL_ROUTES,
    _PackedPrefillDecision,
    MarlinLinear,
    _packed_prefill_enabled,
    _record_packed_prefill_route,
    _select_packed_prefill_config,
    get_marlin_packed_prefill_route_stats,
    reset_marlin_packed_prefill_route_stats,
)
from gptqmodel.utils import marlin as marlin_utils
from scripts.analyze_marlin_packed_prefill_promotion import (
    PromotionObservation,
    evaluate_candidate,
)


class _DecisionInput:
    def __init__(self, *, shape: tuple[int, ...], dtype: torch.dtype, device: str):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = torch.device(device)


def _packed_prefill_test_module() -> MarlinLinear:
    module = MarlinLinear.__new__(MarlinLinear)
    torch.nn.Module.__init__(module)
    module.packed_prefill = True
    module.packed_prefill_stats = False
    module.packed_prefill_min_rows = 1024
    module.packed_prefill_config = 0
    module.weight_type = MarlinLinear.TYPE_MAP[(4, True)]
    module.group_size = 128
    module.desc_act = False
    module.is_k_full = True
    module.qzeros = torch.empty(0, dtype=torch.int32)
    module.in_features = 4096
    module.out_features = 14336
    module.padded_in_features = 4096
    module.padded_out_features = 14336
    module._packed_prefill_hardware = (8, 0, 124)
    return module


def test_packed_prefill_auto_routing_is_on_by_default_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv("GPTQMODEL_MARLIN_PACKED_PREFILL", raising=False)
    assert _packed_prefill_enabled()

    monkeypatch.setenv("GPTQMODEL_MARLIN_PACKED_PREFILL", "0")
    assert not _packed_prefill_enabled()


@pytest.mark.parametrize(
    ("shape", "rows", "reason"),
    (
        ((1, 4096), 1, "decode"),
        ((32, 1, 4096), 32, "decode"),
        ((1, 1023, 4096), 1023, "below_min_rows"),
    ),
)
def test_packed_prefill_decision_keeps_decode_and_small_prefill_on_marlin(shape, rows, reason):
    module = _packed_prefill_test_module()
    inputs = _DecisionInput(shape=shape, dtype=torch.bfloat16, device="cuda:0")

    assert module._packed_prefill_decision(inputs, rows=rows) == _PackedPrefillDecision(0, reason)


def test_packed_prefill_has_a_separate_generated_kernel_family():
    root = marlin_utils._ensure_generated_marlin_kernels()
    generator = (root / "generate_kernels.py").read_text(encoding="utf-8")
    kernel_h = (root / "kernel.h").read_text(encoding="utf-8")
    template_h = (root / "marlin_template.h").read_text(encoding="utf-8")
    gemm_cu = (root / "gptq_marlin.cu").read_text(encoding="utf-8")
    bf16_prefill = (root / "kernel_bf16_prefill_ku4b8.cu").read_text(encoding="utf-8")

    assert "PREFILL_CONFIGS" in generator
    assert "__global__ void MarlinPrefill" in kernel_h
    assert "#if MARLIN_DIRECT_PREFILL" in template_h
    assert "select_marlin_packed_prefill_config" in gemm_cu
    assert "q_type == vllm::kU4B8" in gemm_cu
    assert "group_size == 128" in gemm_cu
    assert bf16_prefill.count("template __global__ void MarlinPrefill<") == 4


@pytest.mark.parametrize(
    "route",
    _PACKED_PREFILL_ROUTES,
    ids=lambda route: f"{route.dtype}-k{route.size_k}-n{route.size_n}-m{route.min_m}-{route.max_m}",
)
def test_packed_prefill_profiled_interval_boundaries_hit(route):
    dtype = torch.float16 if route.dtype == "fp16" else torch.bfloat16
    for rows in {route.min_m, route.max_m}:
        config, reason = _select_packed_prefill_config(
            major=8,
            minor=0,
            sms=124,
            dtype=dtype,
            rows=rows,
            size_k=route.size_k,
            size_n=route.size_n,
        )

        assert config == route.config
        assert reason == "hit"


@pytest.mark.parametrize(
    ("major", "minor", "sms"),
    (
        (8, 0, 108),
        (8, 6, 84),
        (9, 0, 120),
    ),
)
def test_packed_prefill_unprofiled_hardware_misses(major, minor, sms):
    config, reason = _select_packed_prefill_config(
        major=major,
        minor=minor,
        sms=sms,
        dtype=torch.bfloat16,
        rows=4096,
        size_k=4096,
        size_n=14336,
    )

    assert config == 0
    assert reason == "hardware_miss"


def test_packed_prefill_dtype_shape_and_m_misses_are_distinct():
    common = {
        "major": 8,
        "minor": 0,
        "sms": 124,
        "rows": 4096,
        "size_k": 4096,
        "size_n": 14336,
    }
    assert _select_packed_prefill_config(dtype=torch.float32, **common) == (0, "dtype_miss")
    assert _select_packed_prefill_config(
        dtype=torch.bfloat16,
        **{**common, "size_n": 1024},
    ) == (0, "shape_miss")
    assert _select_packed_prefill_config(
        dtype=torch.bfloat16,
        **{**common, "rows": 1024},
    ) == (0, "m_miss")


@pytest.mark.parametrize("rows", (2049, 3072, 8192))
def test_packed_prefill_withdraws_entire_llama70_bf16_gate_up_interval(rows):
    assert _select_packed_prefill_config(
        major=8,
        minor=0,
        sms=124,
        dtype=torch.bfloat16,
        rows=rows,
        size_k=8192,
        size_n=28672,
    ) == (0, "shape_miss")


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_packed_prefill_withdraws_unprofitable_2048_square_projection(dtype):
    assert _select_packed_prefill_config(
        major=8,
        minor=0,
        sms=124,
        dtype=dtype,
        rows=2048,
        size_k=2048,
        size_n=2048,
    ) == (0, "shape_miss")


def test_marlin_module_auto_route_uses_logical_shape_and_rejects_padding():
    module = _packed_prefill_test_module()
    inputs = _DecisionInput(shape=(1025, 4096), dtype=torch.bfloat16, device="cuda:0")

    assert module._packed_prefill_decision(inputs, rows=1025) == _PackedPrefillDecision(2, "hit")

    module.padded_out_features += 64
    assert module._packed_prefill_decision(inputs, rows=1025) == _PackedPrefillDecision(0, "contract_miss")


def test_packed_prefill_route_table_has_no_overlapping_dtype_shape_intervals():
    for index, route in enumerate(_PACKED_PREFILL_ROUTES):
        assert route.config in (1, 2)
        assert 1 <= route.min_m <= route.max_m
        for other in _PACKED_PREFILL_ROUTES[index + 1:]:
            if (route.dtype, route.size_k, route.size_n) != (other.dtype, other.size_k, other.size_n):
                continue
            assert route.max_m < other.min_m or other.max_m < route.min_m


def test_packed_prefill_python_and_native_route_tables_match():
    gemm_cu = (
        marlin_utils._ensure_generated_marlin_kernels() / "gptq_marlin.cu"
    ).read_text(encoding="utf-8")
    native_routes = set()
    for dtype in ("fp16", "bf16"):
        section = re.search(
            rf"packed_prefill_{dtype}_routes\[\] = \{{(?P<body>.*?)\n\}};",
            gemm_cu,
            flags=re.DOTALL,
        )
        assert section is not None
        for size_k, size_n, min_m, max_m, config in re.findall(
            r"\{(\d+), (\d+), (\d+), (\d+), (\d+)\}",
            section.group("body"),
        ):
            native_routes.add((dtype, *(int(value) for value in (size_k, size_n, min_m, max_m, config))))

    python_routes = {
        (route.dtype, route.size_k, route.size_n, route.min_m, route.max_m, route.config)
        for route in _PACKED_PREFILL_ROUTES
    }
    assert native_routes == python_routes
    assert "sms != 124" in gemm_cu
    assert "select_marlin_packed_prefill_config<scalar_t>" in gemm_cu


def test_packed_prefill_route_stats_report_hits_and_miss_reasons():
    reset_marlin_packed_prefill_route_stats()
    _record_packed_prefill_route(
        decision=_PackedPrefillDecision(2, "hit"),
        hardware=(8, 0, 124),
        dtype=torch.bfloat16,
        rows=4096,
        size_k=4096,
        size_n=14336,
    )
    _record_packed_prefill_route(
        decision=_PackedPrefillDecision(0, "shape_miss"),
        hardware=(8, 0, 124),
        dtype=torch.bfloat16,
        rows=4096,
        size_k=4096,
        size_n=1024,
    )
    _record_packed_prefill_route(
        decision=_PackedPrefillDecision(1, "manual_config"),
        hardware=(8, 0, 108),
        dtype=torch.float16,
        rows=4096,
        size_k=4096,
        size_n=14336,
    )

    snapshot = get_marlin_packed_prefill_route_stats(reset=True)
    assert snapshot["total"] == 3
    assert snapshot["auto_hits"] == 1
    assert snapshot["auto_misses"] == 1
    assert snapshot["auto_hit_rate"] == 0.5
    assert snapshot["manual_attempts"] == 1
    assert snapshot["by_reason"] == {"hit": 1, "manual_config": 1, "shape_miss": 1}
    assert get_marlin_packed_prefill_route_stats()["total"] == 0


def _promotion_observation(
    gpu: str,
    rounds: tuple[float, ...],
    *,
    raw: float,
    finite: bool = True,
) -> PromotionObservation:
    return PromotionObservation(
        gpu_uuid=f"GPU-{gpu}",
        paired_round_speedups=rounds,
        raw_speedup=raw,
        finite=finite,
        source=f"{gpu}.json",
    )


def test_packed_prefill_promotion_accepts_stable_cross_gpu_confidence_bound():
    result = evaluate_candidate(
        (
            _promotion_observation("a", (1.061, 1.062, 1.061), raw=1.061),
            _promotion_observation("b", (1.062, 1.061, 1.062), raw=1.062),
        )
    )

    assert result["paired_log_speedup_lcb95"] >= 1.05
    assert result["confidence_pass"]
    assert not result["raw_margin_pass"]
    assert result["eligible"]


def test_packed_prefill_promotion_accepts_raw_margin_on_every_gpu():
    result = evaluate_candidate(
        (
            _promotion_observation("a", (0.96, 1.20, 1.11), raw=1.071),
            _promotion_observation("b", (0.98, 1.18, 1.09), raw=1.075),
        )
    )

    assert result["raw_margin_pass"]
    assert result["eligible"]


def test_packed_prefill_promotion_rejects_noisy_result_without_margin():
    result = evaluate_candidate(
        (
            _promotion_observation("a", (0.90, 1.30, 1.15), raw=1.06),
            _promotion_observation("b", (0.92, 1.28, 1.14), raw=1.06),
        )
    )

    assert not result["confidence_pass"]
    assert not result["raw_margin_pass"]
    assert not result["eligible"]


def test_packed_prefill_promotion_requires_multiple_physical_gpus():
    result = evaluate_candidate((_promotion_observation("a", (1.20, 1.20, 1.20), raw=1.20),))

    assert not result["resource_pass"]
    assert not result["eligible"]


def test_packed_prefill_promotion_rejects_nonfinite_correctness():
    result = evaluate_candidate(
        (
            _promotion_observation("a", (1.20, 1.20, 1.20), raw=1.20),
            _promotion_observation("b", (1.20, 1.20, 1.20), raw=1.20, finite=False),
        )
    )

    assert not result["correctness_pass"]
    assert not result["eligible"]


def test_packed_prefill_source_does_not_define_a_dense_weight_cache():
    source = (
        marlin_utils._marlin_root().parents[1]
        / "gptqmodel"
        / "nn_modules"
        / "qlinear"
        / "marlin.py"
    ).read_text(encoding="utf-8")

    assert "dense_prefill" not in source.lower()
    assert "_dense_prefill_weight" not in source
