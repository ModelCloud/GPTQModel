# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Selection/cache tests; fabricated timings here make no performance claim."""

import json
from dataclasses import replace

import pytest
import torch
from test_qvq_window_recovery import _kernel_rank8, fixture

from gptqmodel.quantization.qvq_rank8 import (
    P32WindowConfig,
    export_window_package,
    load_window_artifact,
    load_window_package,
    prepare_rank8,
    save_window_artifact,
    window_kernel_candidates,
)
from gptqmodel.quantization.qvq_window_tuning import (
    measure_rank8_overhead,
    tune_window_kernel,
)


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
    cached = next(tmp_path.glob("*.json"))
    assert json.loads(cached.read_text())["recovery_overhead"]["overhead_percent"] == pytest.approx(5.0)
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
    )
    assert len(result.report["rows"]) == 1  # CPU fixture has one production candidate.
    pair = result.report["rows"][0]["recovery_overhead"]
    assert pair["off"]["median_us"] == pytest.approx(1.0)
    assert pair["on"]["median_us"] == pytest.approx(1.05)
    assert pair["overhead_percent"] == pytest.approx(5.0)
    assert len(layer._p32_window_tuning["candidate_recovery_overhead"]) == 1
    assert layer._p32_window_tuning["candidate_recovery_overhead"][0]["recovery_overhead"]["overhead_percent"] == pytest.approx(5.0)
    cached = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert cached["identity"]["measure_recovery_candidates"] is True


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
