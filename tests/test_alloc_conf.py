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
    apply_default_alloc_conf,
)


@pytest.fixture()
def clean_alloc_env(monkeypatch):
    for name in ALLOC_CONF_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_default_is_installed_when_unset(clean_alloc_env):
    assert not alloc_conf_configured()

    installed = apply_default_alloc_conf()

    assert installed == DEFAULT_ALLOC_CONF
    for name in ALLOC_CONF_ENV_VARS:
        assert os.environ[name] == DEFAULT_ALLOC_CONF
    # `expandable_segments` is the part that actually prevents the
    # fragmentation cliff; keep the assertion narrow and meaningful.
    assert "expandable_segments:True" in DEFAULT_ALLOC_CONF


def test_second_call_is_a_no_op(clean_alloc_env):
    assert apply_default_alloc_conf() == DEFAULT_ALLOC_CONF
    assert apply_default_alloc_conf() is None


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
        assert payload[name] == DEFAULT_ALLOC_CONF


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

    assert apply_default_alloc_conf() == DEFAULT_ALLOC_CONF
    assert calls == [DEFAULT_ALLOC_CONF]


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

    assert apply_default_alloc_conf() == DEFAULT_ALLOC_CONF
    assert calls == []


def test_expandable_segments_are_active_on_cuda():
    """End-to-end guard: the allocator really is configured, not just the env."""

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

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
