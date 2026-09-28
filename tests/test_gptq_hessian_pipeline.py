# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Focused Hessian pipeline tests for bounded recovery and stream-safe reuse."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from gptqmodel.quantization.config import HessianConfig, QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ, SharedHessianArtifactCache


def _task(*, chunk_size=3, staging_dtype=torch.float32, qcfg=None, out_features=4):
    module = torch.nn.Linear(8, out_features, bias=False)
    config = qcfg or QuantizeConfig(
        hessian=HessianConfig(
            chunk_size=chunk_size,
            staging_dtype=staging_dtype,
        )
    )
    return GPTQ(module, config)


def test_hessian_accumulation_matches_float64_oracle():
    torch.manual_seed(4)
    task = _task(chunk_size=3)
    samples = torch.randn(11, 8, dtype=torch.float16)
    task.add_batch(samples, torch.empty(0))
    actual = task.finalize_hessian()

    samples64 = samples.to(torch.float64)
    oracle = 2.0 * (samples64.T @ samples64) / samples64.shape[0]
    normalized_error = torch.linalg.vector_norm(actual.to(torch.float64) - oracle) / torch.linalg.vector_norm(oracle)

    assert torch.isfinite(actual).all()
    assert float(normalized_error) <= 1e-6


def test_hessian_inverse_uses_at_most_twelve_cpu_probes(monkeypatch):
    task = _task(chunk_size=None)
    hessian = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    original = torch.linalg.cholesky_ex
    probes = 0

    def counted(matrix, *args, **kwargs):
        nonlocal probes
        probes += 1
        return original(matrix, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", counted)
    inverse, damp = task.hessian_inverse(hessian)

    assert inverse is None
    assert damp == 1.0
    assert probes <= 12
    assert torch.allclose(hessian.diagonal(), torch.full((2,), 0.1))


def test_hessian_inverse_binary_searches_beyond_four_damping_steps():
    task = _task(chunk_size=None)
    task.qcfg.damp_percent = 0.01
    task.qcfg.damp_auto_increment = 0.1
    # The first successful grid point is 0.51 (the sixth additive value).
    hessian = torch.tensor([[1.0, 1.5], [1.5, 1.0]])

    factor, damp = task.hessian_inverse(hessian)

    assert factor is not None
    assert damp == pytest.approx(0.51)


def test_hessian_inverse_rejects_nonfinite_and_asymmetric_input():
    task = _task(chunk_size=None)
    nonfinite = torch.tensor([[1.0, 0.0], [0.0, float("nan")]])
    nonfinite_original = nonfinite.clone()
    assert task.hessian_inverse(nonfinite) == (None, 1.0)
    torch.testing.assert_close(nonfinite, nonfinite_original, equal_nan=True)

    asymmetric = torch.tensor([[2.0, 1.0], [0.0, 2.0]])
    asymmetric_original = asymmetric.clone()
    assert task.hessian_inverse(asymmetric) == (None, 1.0)
    assert torch.equal(asymmetric, asymmetric_original)


def test_shared_artifact_cache_is_single_flight_on_cpu():
    cache = SharedHessianArtifactCache(generation=3, cohort="attn")
    calls = 0

    def producer():
        nonlocal calls
        calls += 1
        return torch.arange(4, dtype=torch.float32)

    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda _: cache.get_or_compute(("factor", 8), producer), range(8)))

    assert calls == 1
    assert all(torch.equal(values[0], value) for value in values[1:])


def test_cuda_producer_event_is_synchronized_at_cpu_consumption_boundary():
    task = _task(chunk_size=None)

    class _FakeEvent:
        synchronized = False

        def synchronize(self):
            self.synchronized = True

    event = _FakeEvent()
    task._hessian_ready_event = event

    task._wait_hessian_ready(torch.device("cpu"))

    assert event.synchronized


def test_shared_cached_quantization_matches_uncached_and_factors_once(monkeypatch):
    torch.manual_seed(9)
    config = QuantizeConfig(bits=4, group_size=4, desc_act=True)
    weights = [torch.nn.Linear(8, 6, bias=False) for _ in range(2)]
    cached = [_task(chunk_size=None, qcfg=config, out_features=6) for _ in range(2)]
    uncached = [_task(chunk_size=None, qcfg=config, out_features=6) for _ in range(2)]
    for source, target in zip(weights, cached):
        target.module.weight.data.copy_(source.weight.data)
    for source, target in zip(weights, uncached):
        target.module.weight.data.copy_(source.weight.data)
    for task in cached + uncached:
        task.quantizer.configure(perchannel=True)

    samples = torch.randn(24, 8)
    for task in cached + uncached:
        task.add_batch(samples, torch.empty(0))
    cache = SharedHessianArtifactCache(generation=11, cohort="qkv")
    for task in cached:
        task.attach_shared_hessian_cache(cache)

    original = torch.linalg.cholesky_ex
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", counted)
    cached_results = [task.quantize() for task in cached]
    cached_factor_calls = calls
    uncached_results = [task.quantize() for task in uncached]

    assert cached_factor_calls == 1
    for cached_result, uncached_result in zip(cached_results, uncached_results):
        for cached_value, uncached_value in zip(cached_result[:4], uncached_result[:4]):
            if isinstance(cached_value, torch.Tensor):
                assert torch.equal(cached_value, uncached_value)
            else:
                assert cached_value == uncached_value
        assert cached_result[6] == uncached_result[6]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_hessian_chunk_does_not_call_device_synchronize(monkeypatch):
    device = torch.device("cuda")
    task = _task(chunk_size=2, staging_dtype=torch.float32)
    task.module.to(device)
    samples = torch.randn(7, 8, device=device, dtype=torch.float16)

    def forbidden(*args, **kwargs):
        raise AssertionError("Hessian chunk path must use events, not device synchronize")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    _, hessian, _ = task.process_batch(samples)
    assert hessian is not None
    torch.cuda.current_stream(device).synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_workspace_reuse_across_streams():
    device = torch.device("cuda")
    task = _task(chunk_size=2, staging_dtype=torch.float32)
    task.module.to(device)
    stream_a = torch.cuda.Stream(device=device)
    stream_b = torch.cuda.Stream(device=device)
    samples = [torch.randn(7, 8, device=device, dtype=torch.float16) for _ in range(2)]

    with torch.cuda.stream(stream_a):
        _, first, _ = task.process_batch(samples[0])
    with torch.cuda.stream(stream_b):
        _, second, _ = task.process_batch(samples[1])
    stream_a.synchronize()
    stream_b.synchronize()

    assert first is not None and second is not None
    for actual, sample in zip((first, second), samples):
        oracle_input = sample.reshape(-1, 8).to(torch.float32)
        oracle = oracle_input.T @ oracle_input
        torch.testing.assert_close(actual, oracle, rtol=1e-5, atol=1e-5)
