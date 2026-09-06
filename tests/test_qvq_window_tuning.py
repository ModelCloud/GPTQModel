# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Selection/cache tests; fabricated timings here make no performance claim."""

from dataclasses import replace

import pytest
import torch
from test_qvq_window_recovery import _kernel_rank8, fixture

from gptqmodel.quantization.qvq_rank8 import (
    P32WindowConfig,
    prepare_rank8,
    window_kernel_candidates,
)
from gptqmodel.quantization.qvq_window_tuning import tune_window_kernel


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
