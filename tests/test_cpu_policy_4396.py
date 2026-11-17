# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for #4396 (CPU worker policy, BF16x3 policy, retired envs)."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

import gptqmodel
from gptqmodel.models import auto
from gptqmodel.utils import internal_gguf
from gptqmodel.utils.backend_config import RuntimeConfig
from gptqmodel.utils.kernel import KernelConfig, configure_kernel, reset_kernel_config
from gptqmodel.utils.p2p import P2PConfig


def _fake_quantize_config():
    return SimpleNamespace(kernel=None, runtime=None, p2p=None, dynamic=False, preprocessors=None)


def _install_fake_dense_load(monkeypatch, captured, pool_sizes):
    fake_config = SimpleNamespace(model_type="llama", quantization_config=None)
    monkeypatch.setattr(
        auto.AutoConfig,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: fake_config),
    )
    monkeypatch.setattr(auto, "native_floatx_source_format", lambda *args, **kwargs: None)

    class _Definition:
        @staticmethod
        def from_pretrained(**kwargs):
            captured.append(kwargs)
            pool_sizes.append(len(gptqmodel.get_device_thread_pool()._worker_groups["cpu"]))
            return SimpleNamespace(loaded=True)

    monkeypatch.setattr(auto, "check_and_get_model_definition", lambda *args, **kwargs: _Definition)


@pytest.fixture
def isolated_pool(monkeypatch):
    """Detach the process singleton so a test never inherits another's workers."""

    previous = gptqmodel._DEVICE_THREAD_POOL
    gptqmodel._DEVICE_THREAD_POOL = None
    # Keep the pool prebuild deterministic even if the runner sets the flag.
    monkeypatch.setattr(auto, "env_flag", lambda *args, **kwargs: True)
    try:
        yield
    finally:
        pool = gptqmodel._DEVICE_THREAD_POOL
        gptqmodel._DEVICE_THREAD_POOL = previous
        reset_kernel_config()
        if pool is not None and pool is not previous:
            pool.shutdown(wait=True)


def test_from_pretrained_installs_runtime_cpu_workers_before_the_pool(isolated_pool, monkeypatch):
    captured, pool_sizes = [], []
    _install_fake_dense_load(monkeypatch, captured, pool_sizes)

    auto.GPTQModel.from_pretrained(
        "fake/model",
        quantize_config=_fake_quantize_config(),
        runtime=RuntimeConfig(cpu_workers=2),
        device="cpu",
    )

    assert len(gptqmodel.get_device_thread_pool()._worker_groups["cpu"]) == 2
    # The pool already had the requested size when the model definition ran:
    # the policy was installed before the prebuild.
    assert pool_sizes == [2]
    assert captured[0]["runtime"].cpu_workers == 2


def test_second_load_resizes_the_existing_singleton_pool(isolated_pool, monkeypatch):
    captured, pool_sizes = [], []
    _install_fake_dense_load(monkeypatch, captured, pool_sizes)

    auto.GPTQModel.from_pretrained(
        "fake/model",
        quantize_config=_fake_quantize_config(),
        runtime=RuntimeConfig(cpu_workers=2),
        device="cpu",
    )
    auto.GPTQModel.from_pretrained(
        "fake/model",
        quantize_config=_fake_quantize_config(),
        runtime=RuntimeConfig(cpu_workers=5),
        device="cpu",
    )

    assert pool_sizes == [2, 5]
    assert len(gptqmodel.get_device_thread_pool()._worker_groups["cpu"]) == 5


def test_from_pretrained_forwards_kernel_and_p2p_to_the_definition(isolated_pool, monkeypatch):
    captured, pool_sizes = [], []
    _install_fake_dense_load(monkeypatch, captured, pool_sizes)
    policy = KernelConfig(gguf_cpp_native_matmul=False, gguf_cpp_threads=3)
    p2p = P2PConfig(mode="off")

    auto.GPTQModel.from_pretrained(
        "fake/model",
        quantize_config=_fake_quantize_config(),
        kernel=policy,
        p2p=p2p,
        device="cpu",
    )

    assert captured[0]["kernel"] is policy
    assert captured[0]["p2p"] is p2p


def test_cpu_bf16x3_policy_toggle_is_observed_by_consumers(monkeypatch):
    import gptqmodel.nn_modules.exllamav3_torch as exllamav3_torch
    import gptqmodel.utils.cpu_features as cpu_features

    monkeypatch.setattr(cpu_features, "has_avx512_bf16", lambda: True)
    monkeypatch.setattr(cpu_features, "has_cpu_bf16x3_precision", lambda: True)
    monkeypatch.setattr(exllamav3_torch, "has_avx512_bf16", lambda: True)
    monkeypatch.setattr(exllamav3_torch, "has_cpu_bf16x3_precision", lambda: True)
    exllamav3_torch._cpu_bf16x3_support.cache_clear()

    try:
        for enabled in (True, False, True):
            configure_kernel(KernelConfig(cpu_bf16x3_matmul=enabled))
            assert cpu_features.has_cpu_bf16x3_matmul() is enabled
            assert exllamav3_torch._cpu_bf16x3_support() is enabled
    finally:
        reset_kernel_config()


def test_each_retired_env_var_warns_exactly_once(monkeypatch):
    calls = []

    class _Recorder:
        def warn(self, message):
            calls.append(message)

    monkeypatch.setattr(gptqmodel, "log", _Recorder())
    monkeypatch.setattr(gptqmodel, "_WARNED_RETIRED_KERNEL_ENV_VARS", set())
    monkeypatch.setenv("GPTQ_TORCH_STREAMING", "1")
    monkeypatch.setenv("GPTQMODEL_CPU_WORKERS", "2")

    assert set(gptqmodel.warn_retired_kernel_env_vars()) == {
        "GPTQ_TORCH_STREAMING",
        "GPTQMODEL_CPU_WORKERS",
    }
    assert gptqmodel.warn_retired_kernel_env_vars() == []
    assert len(calls) == 2
    text = "\n".join(calls)
    assert "KernelConfig.torch_streaming" in text
    assert "RuntimeConfig.cpu_workers" in text


def test_install_runtime_keeps_an_imported_real_gguf(monkeypatch):
    real = types.ModuleType("gguf")
    monkeypatch.setitem(sys.modules, "gguf", real)

    assert internal_gguf.install_runtime() is real
    assert sys.modules["gguf"] is real


def test_install_runtime_does_not_shadow_an_installed_gguf(monkeypatch):
    monkeypatch.delitem(sys.modules, "gguf", raising=False)
    monkeypatch.setattr(
        internal_gguf.importlib.util,
        "find_spec",
        lambda name: object() if name == "gguf" else None,
    )

    assert internal_gguf.install_runtime() is None
    assert "gguf" not in sys.modules


def test_install_runtime_installs_the_shim_when_gguf_is_absent(monkeypatch):
    monkeypatch.delitem(sys.modules, "gguf", raising=False)
    monkeypatch.setattr(internal_gguf.importlib.util, "find_spec", lambda name: None)

    assert internal_gguf.install_runtime() is internal_gguf
    assert sys.modules["gguf"] is internal_gguf
