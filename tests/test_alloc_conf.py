# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
"""The default CUDA allocator configuration must be installed before CUDA starts.

Torch parses ``PYTORCH_ALLOC_CONF`` exactly once, when the caching allocator is
created. Assigning it from ``gptqmodel/models/auto.py`` happens after
``gptqmodel.utils`` has already touched CUDA, so the value was silently ignored
and ``expandable_segments`` never took effect. That is what these tests pin.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

from gptqmodel._alloc_conf import (
    ALLOC_CONF_ENV_VARS,
    DEFAULT_ALLOC_CONF,
    alloc_conf_configured,
    allocator_max_split_bytes,
    apply_default_alloc_conf,
    clamp_budget_below_allocator_split,
    default_alloc_conf,
    max_split_bytes_from_conf,
)


@pytest.fixture()
def clean_alloc_env(monkeypatch):
    for name in ALLOC_CONF_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_default_is_installed_when_unset(clean_alloc_env):
    assert not alloc_conf_configured()

    installed = apply_default_alloc_conf()
    expected = default_alloc_conf()

    assert installed == expected
    for name in ALLOC_CONF_ENV_VARS:
        assert os.environ[name] == expected
    # `expandable_segments` is the part that actually prevents the
    # fragmentation cliff on CUDA; ROCm omits it because the VMM path
    # corrupts results across ``empty_cache()``.
    if expected == DEFAULT_ALLOC_CONF:
        assert "expandable_segments:True" in DEFAULT_ALLOC_CONF
    else:
        assert "expandable_segments" not in expected


def test_second_call_is_a_no_op(clean_alloc_env):
    assert apply_default_alloc_conf() == default_alloc_conf()
    assert apply_default_alloc_conf() is None


@pytest.mark.parametrize(
    ("conf", "expected"),
    [
        (None, None),
        ("expandable_segments:True", None),
        ("max_split_size_mb:256,garbage_collection_threshold:0.7", 256 << 20),
        (" garbage_collection_threshold:0.7 , max_split_size_mb:128 ", 128 << 20),
        ("max_split_size_mb:0", None),
        ("max_split_size_mb:not-a-number", None),
    ],
)
def test_max_split_bytes_from_conf(conf, expected):
    assert max_split_bytes_from_conf(conf) == expected


def test_clamp_budget_below_allocator_split(monkeypatch):
    # No cap (the CUDA default with expandable segments): untouched.
    monkeypatch.setattr(
        "gptqmodel._alloc_conf.allocator_max_split_bytes", lambda: None
    )
    assert clamp_budget_below_allocator_split(256 << 20) == 256 << 20

    # A 256 MiB split cap: the default workspace lands in the unsplittable
    # last 2 MiB below the cap, so it is clamped with headroom.
    monkeypatch.setattr(
        "gptqmodel._alloc_conf.allocator_max_split_bytes", lambda: 256 << 20
    )
    assert clamp_budget_below_allocator_split(256 << 20) == 254 << 20
    # Smaller explicit budgets are preserved.
    assert clamp_budget_below_allocator_split(64 << 20) == 64 << 20
    # The floor keeps a pathological cap from producing a zero budget.
    monkeypatch.setattr(
        "gptqmodel._alloc_conf.allocator_max_split_bytes", lambda: 1 << 20
    )
    assert clamp_budget_below_allocator_split(256 << 20) == 1 << 20


def test_allocator_max_split_reads_effective_settings(monkeypatch):
    monkeypatch.setattr(
        "gptqmodel._alloc_conf._effective_alloc_conf",
        lambda: "max_split_size_mb:256,garbage_collection_threshold:0.7",
    )
    assert allocator_max_split_bytes() == 256 << 20


@pytest.mark.parametrize("name", ALLOC_CONF_ENV_VARS)
def test_caller_configuration_is_preserved(clean_alloc_env, name):
    clean_alloc_env.setenv(name, "expandable_segments:False")

    assert alloc_conf_configured()
    assert apply_default_alloc_conf() is None
    assert os.environ[name] == "expandable_segments:False"
    # The other spelling must not be populated behind the caller's back.
    for other in ALLOC_CONF_ENV_VARS:
        if other != name:
            assert other not in os.environ


def test_importing_gptqmodel_installs_the_default():
    """Importing the package is enough; no explicit call is required."""

    script = textwrap.dedent(
        """
        import json, os
        import gptqmodel  # noqa: F401
        print(json.dumps({k: os.environ.get(k) for k in
                          ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")}))
        """
    )
    env = {k: v for k, v in os.environ.items() if k not in ALLOC_CONF_ENV_VARS}
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    payload = json.loads(out.stdout.strip().splitlines()[-1])
    for name in ALLOC_CONF_ENV_VARS:
        assert payload[name] == default_alloc_conf()


def test_runtime_hook_applies_the_default_on_the_native_backend(clean_alloc_env, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    calls = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(
        torch.cuda.memory, "_set_allocator_settings", lambda value: calls.append(value)
    )

    assert apply_default_alloc_conf() == default_alloc_conf()
    assert calls == [default_alloc_conf()]


def test_runtime_hook_leaves_a_non_native_backend_alone(clean_alloc_env, monkeypatch):
    """Torch rejects an allocator-settings change after the backend is parsed."""

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    calls = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "cudaMallocAsync")
    monkeypatch.setattr(
        torch.cuda.memory, "_set_allocator_settings", lambda value: calls.append(value)
    )

    assert apply_default_alloc_conf() == default_alloc_conf()
    assert calls == []


def test_expandable_segments_are_active_on_cuda():
    """End-to-end guard: the allocator really is configured, not just the env."""

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.version.hip is not None:
        pytest.skip("ROCm deliberately does not enable expandable_segments")

    script = textwrap.dedent(
        """
        import json
        import torch
        import gptqmodel  # noqa: F401  (installs the default before CUDA init)
        torch.zeros(1, device="cuda:0")
        print(json.dumps([s.get("is_expandable") for s in torch.cuda.memory_snapshot()]))
        """
    )
    env = {k: v for k, v in os.environ.items() if k not in ALLOC_CONF_ENV_VARS}
    env["CUDA_VISIBLE_DEVICES"] = "0"
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    expandable = json.loads(out.stdout.strip().splitlines()[-1])
    assert expandable, "no allocator segments reported"
    assert all(expandable), f"expandable_segments is not active: {expandable}"
