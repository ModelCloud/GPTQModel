# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Validation CLI backend selection contracts."""

import sys

import pytest


def test_explicit_cuda_backend_reports_unavailable_native_kernel(monkeypatch):
    from gptqmodel.utils import nf12_ad
    from scripts import validate_nf12_ad

    monkeypatch.setattr(nf12_ad, "load", lambda: False)
    monkeypatch.setattr(nf12_ad, "last_error", lambda: "nvcc unavailable")

    with pytest.raises(RuntimeError, match="nvcc unavailable"):
        validate_nf12_ad.backend_module("cuda", "cuda:0")


def test_auto_backend_uses_torch_for_cpu_even_when_cuda_extension_loads(monkeypatch):
    from gptqmodel.utils import nf12_ad
    from scripts import validate_nf12_ad

    def unexpected_probe():
        raise AssertionError("CPU validation must not probe the CUDA extension")

    monkeypatch.setattr(nf12_ad, "load", unexpected_probe)

    assert validate_nf12_ad.backend_module("auto", "cpu") == (None, "torch")


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_explicit_gpu_backend_rejects_cpu_device(backend):
    from scripts import validate_nf12_ad

    with pytest.raises(ValueError, match="requires a CUDA/HIP device"):
        validate_nf12_ad.backend_module(backend, "cpu")


def test_perf_helpers_synchronize_requested_gpu(monkeypatch):
    from scripts import validate_nf12_ad

    observed = []
    tiny = validate_nf12_ad.torch.ones(1)
    monkeypatch.setattr(validate_nf12_ad.torch.cuda, "synchronize",
                        lambda device=None: observed.append(device))
    monkeypatch.setattr(validate_nf12_ad.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(validate_nf12_ad.torch, "empty", lambda *args, **kwargs: tiny)
    monkeypatch.setattr(validate_nf12_ad.torch, "randn", lambda *args, **kwargs: tiny)

    validate_nf12_ad.timed(lambda: None, "cuda:1", warmup=0, iters=1)
    validate_nf12_ad.measure_bandwidth("cuda:1")
    validate_nf12_ad.measure_cache_size("cuda:1", repetitions=1)

    assert observed and set(observed) == {"cuda:1"}


def test_perf_only_failure_returns_nonzero_status(monkeypatch):
    from scripts import validate_nf12_ad

    def fail_perf(*args):
        raise RuntimeError("native backend unavailable")

    monkeypatch.setattr(validate_nf12_ad, "run_perf", fail_perf)
    monkeypatch.setattr(validate_nf12_ad, "print_reports", lambda results: None)
    monkeypatch.setattr(sys, "argv", ["validate_nf12_ad", "--backend", "cuda", "--perf-only"])

    assert validate_nf12_ad.main() == 1
