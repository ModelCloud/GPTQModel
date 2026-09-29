# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Focused Hessian pipeline tests for stream-safe reuse and factor caching."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from gptqmodel.quantization.config import HessianConfig, QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ, SharedHessianFactorCache


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


def test_shared_factor_cache_is_single_flight_on_cpu():
    cache = SharedHessianFactorCache(generation=3, cohort="attn")
    calls = 0

    def producer():
        nonlocal calls
        calls += 1
        return torch.arange(4, dtype=torch.float32), 0.01, torch.ones(4)

    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda _: cache.get_or_compute(("factor", 8), producer), range(8)))

    assert calls == 1
    assert all(torch.equal(values[0][0], value[0]) for value in values[1:])


def test_shared_factor_cache_does_not_retain_failed_build():
    cache = SharedHessianFactorCache(generation=3, cohort="attn")
    calls = 0

    def producer():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("factor build failed")
        return torch.eye(2), 0.01, torch.ones(2)

    with pytest.raises(RuntimeError, match="factor build failed"):
        cache.get_or_compute(("factor", 2), producer)
    factor, _, _ = cache.get_or_compute(("factor", 2), producer)

    assert calls == 2
    assert torch.equal(factor, torch.eye(2))


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


@pytest.mark.parametrize("ordering", ["natural", "desc_act", "act_group_aware"])
def test_shared_cached_quantization_matches_uncached_and_factors_once(monkeypatch, ordering):
    torch.manual_seed(9)
    config = QuantizeConfig(
        bits=4,
        group_size=3,
        desc_act=ordering == "desc_act",
        act_group_aware=ordering == "act_group_aware",
    )
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
    cache = SharedHessianFactorCache(generation=11, cohort="qkv")
    for task in cached:
        task.attach_shared_hessian_cache(cache)

    original_cholesky = torch.linalg.cholesky
    calls = 0

    def counted_cholesky(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_cholesky(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky", counted_cholesky)
    cached_results = [task.quantize() for task in cached]
    cached_factor_calls = calls
    uncached_results = [task.quantize() for task in uncached]
    uncached_factor_calls = calls - cached_factor_calls

    # One probe Cholesky plus the final Cholesky of the inverse factor.
    assert cached_factor_calls == 2
    assert uncached_factor_calls == 4
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
