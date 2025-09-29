# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import time

import pytest
import torch
import torch.nn as nn

from gptqmodel.utils.threadx import DeviceThreadPool


mps_available = hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(not mps_available, reason="MPS not available"),
]

def test_mps_worker_basic():
    d_mps = torch.device("mps")
    p = DeviceThreadPool(devices=[d_mps], inference_mode=True, empty_cache_every_n=3)
    try:
        def add(a, b): return a + b
        a = torch.randn(256, 256, device=d_mps)
        b = torch.randn(256, 256, device=d_mps)
        out = p.do(d_mps, add, a, b)
        assert out.device.type == "mps"
        torch.testing.assert_close(out, a + b)
    finally:
        p.shutdown()

def test_mps_linear_forward_and_counters(monkeypatch):
    d_mps = torch.device("mps")
    calls = []

    # Spy on torch.mps.empty_cache to confirm janitor invocation
    if hasattr(torch, "mps"):
        orig = torch.mps.empty_cache
        def spy():
            calls.append("ec")
            # tiny delay to ensure pass runs
            time.sleep(0.01)
        monkeypatch.setattr(torch.mps, "empty_cache", spy)

    p = DeviceThreadPool(devices=[d_mps], inference_mode=True, empty_cache_every_n=2)
    try:
        m = nn.Linear(64, 32).to(d_mps)
        x = torch.randn(16, 64, device=d_mps)

        def fwd(mod, inp): return mod(inp)

        # two tasks -> threshold 2 -> janitor should run once
        p.do(d_mps, fwd, m, x)
        y = p.do(d_mps, fwd, m, x)
        assert y.shape == (16, 32)

        # allow janitor to run
        time.sleep(0.1)

        st = p.stats()
        assert st["per_device"]["mps"] >= 2
        assert st["total"] >= 2
        if hasattr(torch, "mps"):
            assert len(calls) >= 1
    finally:
        p.shutdown()
        if hasattr(torch, "mps"):
            monkeypatch.setattr(torch.mps, "empty_cache", orig)
