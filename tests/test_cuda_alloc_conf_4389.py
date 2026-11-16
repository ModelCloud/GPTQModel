# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for #4389: the CUDA default must not pin max_split_size_mb.

``max_split_size_mb:256`` on top of ``expandable_segments:True`` caps the
largest block the allocator is willing to split, so a freed 256 MiB block
cannot serve the next layer's slightly smaller request. GPTQ's per-layer
Hessian/workspace churn then maps new pages instead of reusing the cache:
*reserved* VRAM grows per layer while *active* memory stays flat (Llama-3.2-1B:
1.23 GiB allocated, 5.40 GiB reserved with the old default, 1.67 GiB with
expandable segments alone).

The ROCm default is intentionally unchanged here: #4313's tests pin
``max_split_size_mb:256,garbage_collection_threshold:0.7``.
"""

from __future__ import annotations

import os

import pytest

from gptqmodel import _alloc_conf


@pytest.fixture()
def clean_alloc_env(monkeypatch):
    for name in _alloc_conf.ALLOC_CONF_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_cuda_default_keeps_expandable_segments_without_split_cap():
    conf = _alloc_conf.default_alloc_conf("cuda")

    assert conf == _alloc_conf.DEFAULT_ALLOC_CONF
    # expandable_segments is the knob that actually removes fragmentation.
    assert "expandable_segments:True" in conf
    # The 256 MiB split cap is what fragmented the run in #4389.
    assert "max_split_size_mb" not in conf
    # No memory fraction is set, so a GC threshold would only fire above a
    # fraction of the whole device; drop it rather than pretend it helps.
    assert "garbage_collection_threshold" not in conf


def test_rocm_default_still_matches_the_4313_contract():
    conf = _alloc_conf.default_alloc_conf("rocm")

    assert conf == _alloc_conf.ROCM_DEFAULT_ALLOC_CONF
    assert "expandable_segments" not in conf
    assert "max_split_size_mb:256" in conf
    assert "garbage_collection_threshold:0.7" in conf


def test_cuda_install_drops_split_cap_and_keeps_expandable_segments(clean_alloc_env, monkeypatch):
    monkeypatch.setattr(_alloc_conf, "running_on_rocm", lambda: False)
    monkeypatch.setattr(_alloc_conf, "_apply_runtime_alloc_conf", lambda conf: None)

    installed = _alloc_conf.apply_default_alloc_conf()

    assert installed == _alloc_conf.DEFAULT_ALLOC_CONF
    assert "expandable_segments:True" in installed
    assert "max_split_size_mb" not in installed
    for name in _alloc_conf.ALLOC_CONF_ENV_VARS:
        assert os.environ[name] == installed


@pytest.mark.parametrize("name", _alloc_conf.ALLOC_CONF_ENV_VARS)
def test_user_provided_alloc_conf_is_never_overwritten(clean_alloc_env, name):
    user_value = "expandable_segments:True,max_split_size_mb:512"
    clean_alloc_env.setenv(name, user_value)
    clean_alloc_env.setattr(_alloc_conf, "running_on_rocm", lambda: False)
    clean_alloc_env.setattr(_alloc_conf, "_apply_runtime_alloc_conf", lambda conf: None)

    assert _alloc_conf.alloc_conf_configured()
    assert _alloc_conf.apply_default_alloc_conf() is None
    assert os.environ[name] == user_value
    # The other spelling must not be populated behind the caller's back.
    for other in _alloc_conf.ALLOC_CONF_ENV_VARS:
        if other != name:
            assert other not in os.environ
