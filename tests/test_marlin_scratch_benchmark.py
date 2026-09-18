# SPDX-License-Identifier: Apache-2.0
"""CPU integration checks for benchmark orchestration, without GPU timing claims."""
from types import SimpleNamespace

import torch

from gptqmodel.utils.marlin_scratch import active_marlin_scratch_context
from scripts import benchmark_marlin_scratch as benchmark
from scripts import marlin_scratch_fixture as fixture


def test_paired_case_uses_public_context_module(monkeypatch, tmp_path):
    # Run the real A/B orchestration with CPU arithmetic and stub only GPU
    # measurements. This also catches imports hidden behind CUDA availability.
    dense = torch.eye(8, 4, dtype=torch.float16)
    calls = []

    def layer(x):
        calls.append(active_marlin_scratch_context() is not None)
        return x @ dense

    monkeypatch.setattr(fixture, "make_layer", lambda *a, **kw: (layer, dense, None))
    monkeypatch.setattr(fixture, "gemm_arguments", lambda layer, x: {
        "a": x, "size_m": x.shape[0], "size_k": 8, "size_n": 4})
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)

    def measure(fn, torch, device, iters):
        fn()
        return dict(cpu_submit_us=1., forward_us=2., gpu_span_us=3.)

    def audit(fn, torch, device, path):
        fn()
        return {}

    monkeypatch.setattr(benchmark, "measure", measure)
    monkeypatch.setattr(benchmark, "allocation_audit", audit)
    # The Marlin operation utility intentionally does not re-export the context.
    utils = SimpleNamespace(marlin_scratch_sizes=lambda *args: (16, 0))
    args = SimpleNamespace(device="cpu", seed=17, cache_bytes=1024,
                           warmup=1, rounds=2, iters=1, graph=False)
    case = dict(shape="test", k=8, n=4, m=2, dtype="fp16", method="gptq",
                act_order=False, fp32=True)
    result = benchmark.run_case(case, torch, utils, args, tmp_path)

    assert result["status"] == "complete"
    assert all(len(samples) == 2 for samples in result["samples"].values())
    assert any(calls) and not all(calls)
    assert active_marlin_scratch_context() is None
    assert result["scratch_requirements"]["executed_k"] == 8
    assert result["scratch_requirements"]["c_tmp_bytes"] == 64
