# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for #5072: the ROCm default must not pin max_split_size_mb.

``max_split_size_mb:256`` caps the largest block the caching allocator will
split, so a request below the cap cannot be served from a free block at or
above it. MACaQ's per-layer flip-sweep churn on a wide (14336x4096) projection
therefore maps new segments instead of reusing the freed ones: *reserved*
memory grows while *active* memory stays flat, and the inflated reservation is
still held after ``quantize()`` returns. On one 14336x4096 MACaQ layer the old
default peaked at 5258 MiB reserved against 4224 MiB allocated; the cap-free
default peaked at 4466 MiB with identical tensors.

The policy test is device-free; the reuse test is a ROCm device test and is
skipped elsewhere. The allocator-level reuse check is deliberate: a ratio
threshold on one wide layer (1.25x here) would not catch the regression, while
the freed-segment reuse rule is exactly what the cap changes. See
``tests/test_awq_rocm_layer_cache_4313.py`` for the AWQ clip-scratch contract
the old default was kept for.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch

from gptqmodel import _alloc_conf


IS_ROCM = torch.version.hip is not None
MIB = 1 << 20


def test_rocm_default_keeps_only_the_vmm_exclusion():
    conf = _alloc_conf.default_alloc_conf("rocm")

    assert conf == _alloc_conf.ROCM_DEFAULT_ALLOC_CONF
    assert conf == "expandable_segments:False"
    assert _alloc_conf.max_split_bytes_from_conf(conf) is None
    # ``garbage_collection_threshold`` is a ratio of the whole device and never
    # fires without a memory fraction, so it is not a mitigation.
    assert "garbage_collection_threshold" not in conf


def test_rocm_default_does_not_clamp_awq_clip_budget(monkeypatch):
    """No configured cap means the clip workspace keeps its resolved size."""

    monkeypatch.setattr(_alloc_conf, "allocator_max_split_bytes", lambda: None)

    assert _alloc_conf.clamp_budget_below_allocator_split(256 * MIB) == 256 * MIB


@pytest.mark.skipif(not IS_ROCM, reason="ROCm allocator policy")
def test_default_alloc_conf_reuses_a_freed_segment_for_a_smaller_request():
    """A sub-cap request must be served from the freed cap-sized block.

    This is the allocator-level trigger behind #5072: allocate a 255 MiB
    buffer, free it, then request 200 MiB. With ``max_split_size_mb:256`` the
    second request may not use the freed 256 MiB segment, so reserved memory
    grows by a fresh segment on top of the one already held.
    """

    script = textwrap.dedent(
        """
        import json, os
        import torch
        import gptqmodel  # noqa: F401  installs the default before CUDA init

        torch.zeros(1, device="cuda")
        big = torch.empty(255 * 2**20, dtype=torch.uint8, device="cuda")
        del big
        torch.cuda.synchronize()
        before = torch.cuda.memory_reserved()
        small = torch.empty(200 * 2**20, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        print(json.dumps({
            "conf": os.environ.get("PYTORCH_ALLOC_CONF"),
            "before": before,
            "live": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
        }))
        """
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in _alloc_conf.alloc_conf_env_vars()
    }
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    payload = json.loads(proc.stdout.strip().splitlines()[-1])

    assert payload["conf"] == _alloc_conf.ROCM_DEFAULT_ALLOC_CONF
    assert payload["live"] == 200 * MIB
    # The freed segment is reused instead of a second 256 MiB one being mapped.
    assert payload["reserved"] < 2 * payload["live"], (
        f"reserved {payload['reserved'] / MIB:.0f} MiB for "
        f"{payload['live'] / MIB:.0f} MiB live"
    )
