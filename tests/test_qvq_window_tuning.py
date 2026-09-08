# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Selection/cache tests; fabricated timings here make no performance claim."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from test_qvq_window_recovery import _kernel_rank8, fixture

from gptqmodel.quantization.qvq_rank8 import (
    _HOPPER_LARGE_M_THRESHOLD,
    P32WindowConfig,
    export_window_package,
    grouped_window_kernel_shape_score,
    load_window_artifact,
    load_window_package,
    prepare_rank8,
    save_window_artifact,
    window_kernel_candidates,
    window_kernel_candidates_for_shape,
    window_kernel_shape_score,
)
from gptqmodel.quantization.qvq_window_tuning import (
    measure_rank8_overhead,
    tune_grouped_window_kernel,
    tune_window_kernel,
)


def _recorded_zml_reports():
    """Yield committed ZML reports used to audit the historical budget gate."""
    results = Path(__file__).parents[1] / "docs/kernels/results"
    addmm = json.loads((results / "p32_window_native_zml_addmm.json").read_text())
    yield "addmm-unbudgeted", addmm["verifier_report"]
    policy = json.loads(
        (results / "p32_window_native_zml_m8192_policy.json").read_text()
    )
    yield "m8192-unbudgeted", policy["reports"]["unbudgeted"]
    yield "m8192-budget5", policy["reports"]["budget5"]


def test_shape_scores_are_public_ordering_hints_only():
    layer = type("Layer", (), {"out_features": 2048})()
    small_m16 = P32WindowConfig(algorithm="hopper_m16")
    direct_bm32 = P32WindowConfig(
        algorithm="hopper_direct_decode_mma", block_m=32, block_n=64, warp_groups=1
    )
    large_m16 = window_kernel_shape_score(layer, small_m16, m=512)
    assert window_kernel_shape_score(layer, small_m16, m=96) == 0
    assert window_kernel_shape_score(layer, direct_bm32, m=96) == 0
    assert large_m16 == 200
    assert grouped_window_kernel_shape_score(
        (layer, layer), (direct_bm32, direct_bm32), m=96
    ) == 0


def test_hopper_auto_policy_selects_direct_consumer_with_small_m_cutover(monkeypatch):
    """SM90 auto policy uses large-M reuse kernels without changing ABI state."""

    class Layer:
        v2b2_p32 = True
        training = False
        window_only = True
        activation = None
        bits = 3
        in_features = 5120
        out_features = 2048

        @staticmethod
        def runtime_device():
            return torch.device("cuda")

        @staticmethod
        def _prepare_amd_p32_metadata(_device):
            return None

        @staticmethod
        def _prepare_hopper_p32_window(_device):
            return None

    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: type(
            "Properties", (), {"major": 9, "minor": 0, "name": "NVIDIA H100"}
        )(),
    )
    monkeypatch.setattr(
        "gptqmodel.utils.qvq_cuda.prewarm_qvq_cuda", lambda: None
    )

    layer = Layer()
    prepare_rank8(layer, P32WindowConfig())

    assert layer._p32_window_config.algorithm == "hopper_direct_decode_mma"
    assert _HOPPER_LARGE_M_THRESHOLD == 32


def test_hopper_qwen_down_fast_policy_selects_tensor_core_projection(monkeypatch):
    """SM90 fast auto policy uses the measured Qwen-down recovery path."""

    class Layer:
        v2b2_p32 = True
        training = False
        window_only = True
        activation = None
        bits = 3
        in_features = 17408
        out_features = 5120
        input_hadamard = False
        output_hadamard = False
        rank8_metadata = None

        @staticmethod
        def runtime_device():
            return torch.device("cuda")

        @staticmethod
        def _prepare_amd_p32_metadata(_device):
            return None

        @staticmethod
        def _prepare_hopper_p32_window(_device):
            return None

    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: type(
            "Properties", (), {"major": 9, "minor": 0, "name": "NVIDIA H100"}
        )(),
    )
    monkeypatch.setattr(
        "gptqmodel.utils.qvq_cuda.prewarm_qvq_cuda", lambda: None
    )

    layer = Layer()
    prepare_rank8(
        layer,
        P32WindowConfig(
            recovery_mode="auto",
            recovery_kernel="fully_fused",
            recovery_projection="project_output_fused",
            arithmetic_signature="unverified_project_output_fused",
        ),
    )

    assert layer._p32_window_config.algorithm == "hopper_direct_decode_mma"
    assert layer._p32_window_config.recovery_kernel == "fused_epilogue"
    assert layer._p32_window_config.recovery_projection == "tensor_core"
    assert layer._p32_window_config.arithmetic_signature == "unverified_tensor_core"

    balanced = Layer()
    prepare_rank8(
        balanced,
        P32WindowConfig(
            recovery_mode="auto",
            recovery_kernel="fully_fused",
            recovery_projection="project_output_fused",
            arithmetic_signature="unverified_project_output_fused",
            quality_mode="balanced",
        ),
    )
    assert balanced._p32_window_config.recovery_kernel == "fully_fused"
    assert balanced._p32_window_config.recovery_projection == "project_output_fused"


def test_shape_ordering_keeps_every_single_projection_candidate(monkeypatch):
    """The public shape helper orders candidates without filtering winners."""
    import gptqmodel.quantization.qvq_rank8 as rank8

    layer = type("Layer", (), {"out_features": 2048})()
    candidates = (
        P32WindowConfig(algorithm="production_window"),
        P32WindowConfig(
            algorithm="hopper_direct_decode_mma",
            block_m=128,
            block_n=128,
            warp_groups=2,
        ),
        P32WindowConfig(algorithm="hopper_m16"),
    )
    monkeypatch.setattr(
        rank8,
        "window_kernel_candidates",
        lambda ignored, *, m: candidates,
    )
    ordered = window_kernel_candidates_for_shape(layer, m=96)
    assert ordered == (candidates[0], candidates[2], candidates[1])
    assert set(ordered) == set(candidates)


def test_recorded_overhead_audit_keeps_uncapped_improvements_visible():
    """Historical >5% rows stay eligible unless a caller requested a cap."""
    reports = dict(_recorded_zml_reports())

    for name in ("addmm-unbudgeted", "m8192-unbudgeted"):
        report = reports[name]
        rows = [
            row
            for row in report["entries"]
            if row.get("rank8_enabled") and row.get("accepted")
        ]
        assert rows
        over_target = [
            row
            for row in rows
            if row["recovery_pair"]["overhead_percent"] > 5.0
        ]
        assert over_target, name
        # With no explicit budget, selection is latency based after numerical
        # and arithmetic eligibility; the aspirational 3--6% target cannot
        # remove an otherwise valid row.
        fastest = min(rows, key=lambda row: row["median_ns"])
        assert report["selected_on"] == fastest["candidate_index"]

    budgeted = reports["m8192-budget5"]
    assert budgeted["max_recovery_overhead_percent"] == 5
    # The budgeted run intentionally excludes over-budget rows from the
    # winner while retaining them in the report for review.
    assert any(
        row["recovery_pair"]["overhead_percent"] > 5.0
        for row in budgeted["entries"]
        if row.get("rank8_enabled")
    )
    selected = next(
        row
        for row in budgeted["entries"]
        if row.get("rank8_enabled")
        and row["candidate_index"] == budgeted["selected_on"]
    )
    assert selected["recovery_pair"]["overhead_percent"] <= 5.0


def test_cpu_cache_revalidates_and_binds_inputs(tmp_path):
    layer, _, x, _ = fixture()
    original = P32WindowConfig()
    prepare_rank8(layer, original)
    calls = []

    def sample(fn, inputs):
        calls.append(fn(inputs))
        return [2, 1, 3]

    result = tune_window_kernel(
        layer, x, benchmark=sample, build_id="test", cache_dir=tmp_path, apply=False
    )
    assert not result.cache_hit and len(calls) == 1
    assert layer._p32_window_config == original
    cached = tune_window_kernel(
        layer, x, benchmark=sample, build_id="test", cache_dir=tmp_path, apply=False
    )
    assert cached.cache_hit and len(calls) == 1
    changed = tune_window_kernel(
        layer, x + 0.1, benchmark=sample, build_id="test", cache_dir=tmp_path
    )
    assert not changed.cache_hit and len(calls) == 2
    assert layer._p32_window_config == changed.config
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_grouped_tuner_selects_one_complete_child_tuple_and_caches_it(monkeypatch, tmp_path):
    first, _, x, _ = fixture()
    second, _, _, _ = fixture()
    original = (P32WindowConfig(), P32WindowConfig())
    from gptqmodel.quantization import qvq_window_tuning

    for layer, config in zip((first, second), original, strict=True):
        prepare_rank8(layer, config)
    production = tuple(
        replace(config, algorithm="production_window", min_m=1, max_m=1)
        for config in original
    )
    alternate = tuple(
        replace(config, algorithm="auto", min_m=1, max_m=1)
        for config in original
    )
    monkeypatch.setattr(
        qvq_window_tuning,
        "grouped_window_kernel_candidates_for_shape",
        lambda layers, *, m: (production, alternate),
    )

    def compile_candidate(configs):
        cost = 2.0 if configs[0]["algorithm"] == "production_window" else 1.0

        def forward(value):
            return tuple(
                torch.ones((*value.shape[:-1], layer.out_features), dtype=value.dtype)
                for layer in (first, second)
            )

        forward.cost = cost
        return forward

    result = tune_grouped_window_kernel(
        (first, second),
        x,
        benchmark=lambda fn, value: [fn.cost],
        compile_candidate=compile_candidate,
        build_id="grouped-test",
        cache_dir=tmp_path,
        apply=False,
    )
    assert result.configs == alternate
    assert result.report["selected"] == [config.to_backend_config() for config in alternate]
    assert not result.cache_hit
    assert len(result.report["identity"]["candidate_shape_scores"]) == 2
    assert all("shape_score" in row for row in result.report["rows"])

    cached = tune_grouped_window_kernel(
        (first, second),
        x,
        benchmark=lambda fn, value: [fn.cost],
        compile_candidate=compile_candidate,
        build_id="grouped-test",
        cache_dir=tmp_path,
        apply=False,
    )
    assert cached.cache_hit
    assert cached.configs == alternate
    assert first._p32_window_config == original[0]
    assert second._p32_window_config == original[1]


def test_failure_restores_policy_and_cache_does_not_bypass_validation(tmp_path):
    layer, _, x, _ = fixture()
    original = P32WindowConfig()
    prepare_rank8(layer, original)
    with pytest.raises(ValueError, match="latency"):
        tune_window_kernel(
            layer, x, benchmark=lambda fn, x: [float("nan")], build_id="test"
        )
    assert layer._p32_window_config == original
    tune_window_kernel(
        layer,
        x,
        benchmark=lambda fn, x: [1],
        build_id="test",
        cache_dir=tmp_path,
        apply=False,
    )
    with pytest.raises(ValueError, match="no kernel"):
        tune_window_kernel(
            layer,
            x,
            benchmark=lambda fn, x: [1],
            build_id="test",
            cache_dir=tmp_path,
            compile_candidate=lambda config: lambda inputs: layer(inputs) + 1,
        )
    assert layer._p32_window_config == original


def test_rank8_overhead_measures_matched_states_and_restores_policy():
    layer, _, x, _ = fixture()
    _kernel_rank8(layer)
    original = P32WindowConfig(recovery_mode="on", quality_mode="fast")
    prepare_rank8(layer, original)
    seen = []

    def sample(fn, inputs):
        seen.append(bool(layer._p32_rank8_enabled))
        fn(inputs)
        return [1.0 if not layer._p32_rank8_enabled else 1.04]

    report = measure_rank8_overhead(
        layer,
        x,
        benchmark=sample,
    )
    assert seen == [False, True]
    assert report["m"] == x.shape[0]
    assert report["off"]["median_us"] == 1.0
    assert report["on"]["median_us"] == 1.04
    assert report["overhead_us"] == pytest.approx(0.04)
    assert report["overhead_percent"] == pytest.approx(4.0)
    assert layer._p32_window_config == original
    assert layer._p32_rank8_enabled


def test_rank8_overhead_rejects_invalid_samples_and_restores_policy():
    layer, _, x, _ = fixture()
    _kernel_rank8(layer)
    original = P32WindowConfig(recovery_mode="on")
    prepare_rank8(layer, original)
    with pytest.raises(ValueError, match="finite positive"):
        measure_rank8_overhead(layer, x, benchmark=lambda _fn, _x: [0])
    assert layer._p32_window_config == original
    assert layer._p32_rank8_enabled


def test_window_tuner_can_attach_matched_rank8_overhead(tmp_path):
    layer, _, x, _ = fixture()
    _kernel_rank8(layer)
    original = P32WindowConfig(recovery_mode="on")
    prepare_rank8(layer, original)

    def sample(fn, inputs):
        fn(inputs)
        return [1.05 if layer._p32_rank8_enabled else 1.0]

    result = tune_window_kernel(
        layer,
        x,
        benchmark=sample,
        build_id="overhead",
        cache_dir=tmp_path,
        measure_recovery=True,
    )
    assert result.report["recovery_overhead"]["overhead_percent"] == pytest.approx(5.0)
    assert len(result.report["identity"]["candidate_shape_scores"]) == len(
        result.report["rows"]
    )
    assert "shape_score" in result.report["rows"][0]
    cached = next(tmp_path.glob("*.json"))
    assert json.loads(cached.read_text())["recovery_overhead"]["overhead_percent"] == pytest.approx(5.0)
    assert layer._p32_window_config == result.config


def test_window_tuner_keeps_improvement_without_default_overhead_cap(tmp_path):
    layer, _, x, _ = fixture()
    _kernel_rank8(layer)
    original = P32WindowConfig(recovery_mode="on", quality_mode="fast")
    prepare_rank8(layer, original)

    def sample(fn, inputs):
        fn(inputs)
        # This deliberately exceeds the aspirational 3--5% scorecard target.
        # With no explicit cap, a numerically accepted improvement remains a
        # valid tuning result and its measured cost is retained in the report.
        return [1.20 if layer._p32_rank8_enabled else 1.0]

    result = tune_window_kernel(
        layer,
        x,
        benchmark=sample,
        build_id="uncapped-improvement",
        cache_dir=tmp_path,
        measure_recovery=True,
    )
    assert result.report["recovery_overhead"]["overhead_percent"] == pytest.approx(20.0)
    assert result.report["identity"]["max_recovery_overhead_percent"] is None
    assert layer._p32_window_config == result.config


def test_window_tuner_can_measure_recovery_pair_for_each_candidate(tmp_path):
    layer, _, x, _ = fixture()
    _kernel_rank8(layer)
    prepare_rank8(layer, P32WindowConfig(recovery_mode="on", quality_mode="fast"))

    def sample(fn, inputs):
        fn(inputs)
        return [1.05 if layer._p32_rank8_enabled else 1.0]

    result = tune_window_kernel(
        layer,
        x,
        benchmark=sample,
        build_id="per-candidate-overhead",
        cache_dir=tmp_path,
        measure_recovery_candidates=True,
        max_recovery_overhead_percent=5.0,
    )
    assert len(result.report["rows"]) == 1  # CPU fixture has one production candidate.
    pair = result.report["rows"][0]["recovery_overhead"]
    assert pair["off"]["median_us"] == pytest.approx(1.0)
    assert pair["on"]["median_us"] == pytest.approx(1.05)
    assert pair["overhead_percent"] == pytest.approx(5.0)
    assert result.report["rows"][0]["recovery_overhead_eligible"] is True
    assert len(layer._p32_window_tuning["candidate_recovery_overhead"]) == 1
    assert layer._p32_window_tuning["candidate_recovery_overhead"][0]["recovery_overhead"]["overhead_percent"] == pytest.approx(5.0)
    package = export_window_package(layer)
    restored = load_window_package(package)
    assert restored._p32_window_tuning["candidate_recovery_overhead"] == layer._p32_window_tuning["candidate_recovery_overhead"]
    cached = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert cached["identity"]["measure_recovery_candidates"] is True


def test_window_tuner_overhead_gate_rejects_slow_recovery(tmp_path):
    layer, _, x, _ = fixture()
    _kernel_rank8(layer)
    original = P32WindowConfig(recovery_mode="on", quality_mode="fast")
    prepare_rank8(layer, original)

    def sample(fn, inputs):
        fn(inputs)
        return [1.05 if layer._p32_rank8_enabled else 1.0]

    with pytest.raises(ValueError, match="no kernel candidate"):
        tune_window_kernel(
            layer,
            x,
            benchmark=sample,
            build_id="overhead-gate",
            cache_dir=tmp_path,
            measure_recovery_candidates=True,
            max_recovery_overhead_percent=4.0,
        )
    assert layer._p32_window_config == original
    assert layer._p32_rank8_enabled


def test_applied_tuning_metadata_roundtrips_with_unified_package(tmp_path):
    layer, _, x, _ = fixture()
    prepare_rank8(layer, P32WindowConfig())
    result = tune_window_kernel(
        layer,
        x,
        benchmark=lambda fn, inputs: [1],
        build_id="artifact-test",
        cache_dir=tmp_path,
    )
    assert layer._p32_window_tuning["version"] == 1
    assert layer._p32_window_tuning["selected"] == result.config.to_backend_config()
    package = export_window_package(layer)
    assert package["kernel_tuning"]["identity"]["state_hash"]
    restored = load_window_package(package)
    assert restored._p32_window_tuning == package["kernel_tuning"]
    artifact = tmp_path / "artifact"
    save_window_artifact(layer, artifact)
    assert (artifact / "manifest.json").exists()
    loaded_artifact = load_window_artifact(artifact)
    assert loaded_artifact._p32_window_tuning == package["kernel_tuning"]


def test_stale_tuning_metadata_cannot_be_exported_or_loaded():
    layer, _, x, _ = fixture()
    prepare_rank8(layer, P32WindowConfig())
    tune_window_kernel(layer, x, benchmark=lambda fn, inputs: [1], build_id="stale")
    package = export_window_package(layer)
    layer.SU.add_(1)
    with pytest.raises(ValueError, match="state hash"):
        export_window_package(layer)
    package["metadata"]["output_hadamard"] = False
    with pytest.raises(ValueError, match="state hash"):
        load_window_package(package)


def test_tuning_metadata_rejects_unmeasured_selected_kernel():
    layer, _, x, _ = fixture()
    prepare_rank8(layer, P32WindowConfig())
    tune_window_kernel(layer, x, benchmark=lambda fn, inputs: [1], build_id="candidate")
    package = export_window_package(layer)
    package["kernel_tuning"]["selected"].update(
        algorithm="hopper_direct_decode_mma", block_m=32, block_n=64, warp_groups=1
    )
    with pytest.raises(ValueError, match="absent from measured candidates"):
        load_window_package(package)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_quality_rank8_candidates_are_reference_arithmetic_only():
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    from test_qvq_grouped_runtime import _child

    layer = _child("q_proj", device="cuda").eval()
    _kernel_rank8(layer)
    prepare_rank8(
        layer,
        P32WindowConfig(recovery_mode="on", quality_mode="quality"),
    )
    candidates = window_kernel_candidates(layer, m=33)
    assert candidates
    assert all(c.arithmetic_signature == "reference_fp32_v1" for c in candidates)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hopper_external_candidate_gates_and_fixed_quality(tmp_path):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 required")
    from test_qvq_grouped_runtime import _child

    layer = _child("q_proj", device="cuda").eval()
    _kernel_rank8(layer)
    original = P32WindowConfig(recovery_mode="on")
    prepare_rank8(layer, original)
    x = torch.randn(33, 256, device="cuda", dtype=torch.float16) * 0.01
    candidates = window_kernel_candidates(layer, m=33)
    chosen = [
        c
        for c in candidates
        if c.recovery_kernel == "separate_reference"
        and c.recovery_projection == "separate_reference"
        and c.block_n == 64
        and c.block_m in (32, 64)
    ]
    compiled = []

    def compile_candidate(config):
        compiled.append(config)
        offset = 10 if config["block_m"] == 32 else 0
        return lambda inputs: layer(inputs) + offset

    def sample(fn, inputs):
        return [0.1 if layer._p32_window_config.block_m == 32 else 1]

    result = tune_window_kernel(
        layer,
        (x, -x),
        candidates=chosen,
        compile_candidate=compile_candidate,
        benchmark=sample,
        build_id="test",
        cache_dir=tmp_path,
    )
    assert result.config.block_m == 64
    assert result.config.recovery_mode == "on" and layer._p32_rank8_enabled
    assert len(compiled) == 2
    assert result.report["rows"][0]["exception_review_required"]
    assert not result.report["rows"][0]["accepted"]
    with pytest.raises(ValueError, match="prepared quality"):
        tune_window_kernel(
            layer,
            x,
            candidates=[replace(result.config, recovery_mode="off")],
            benchmark=sample,
            build_id="test",
        )
