# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""#4448: later loads apply every pool setting; typed API reach and validation."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

import gptqmodel
from gptqmodel.backend import BitsAndBytesConfig, GGUFCUDAConfig, RuntimeConfig
from gptqmodel.models import auto
from gptqmodel.quantization.config import AWQConfig, GPTQConfig, QuantizeConfig
from gptqmodel.utils.backend_config import resolve_backend_config
from gptqmodel.utils.kernel import KernelConfig, configure_kernel, reset_kernel_config


class _FakePool:
    def __init__(self, *, workers, intraop_threads_per_worker, **kwargs):
        self.workers = workers
        self.intraop = intraop_threads_per_worker
        self._worker_groups = {"cpu": [None] * workers["cpu"]}
        self.shutdowns = 0

    def shutdown(self, *, wait):
        self.shutdowns += 1


@pytest.fixture
def fake_pool(monkeypatch):
    monkeypatch.setattr(gptqmodel, "DeviceThreadPool", _FakePool)
    monkeypatch.setattr(gptqmodel.atexit, "register", lambda *args: None)
    monkeypatch.setattr(gptqmodel, "_DEVICE_THREAD_POOL", None)
    try:
        yield
    finally:
        reset_kernel_config()


def _resize_for(**settings):
    configure_kernel(KernelConfig(**settings))
    return gptqmodel.resize_device_thread_pool()


@pytest.mark.parametrize("change", [
    {"cuda_workers": 2},
    {"pool_intraop_threads": 2},
    {"cpu_workers": 3},
])
def test_any_changed_pool_setting_rebuilds_the_pool(fake_pool, change):
    first = _resize_for(cpu_workers=4, cuda_workers=3, pool_intraop_threads=1)
    assert first.workers["cuda:per"] == 3 and first.workers["cpu"] == 4 and first.intraop == 1

    settings = {"cpu_workers": 4, "cuda_workers": 3, "pool_intraop_threads": 1, **change}
    second = _resize_for(**settings)

    assert second is not first
    assert first.shutdowns == 1
    assert second.workers["cpu"] == settings["cpu_workers"]
    assert second.workers["cuda:per"] == settings["cuda_workers"]
    assert second.intraop == settings["pool_intraop_threads"]
    assert gptqmodel.get_device_thread_pool() is second


def test_unchanged_pool_settings_keep_the_pool(fake_pool):
    first = _resize_for(cpu_workers=4, cuda_workers=3)
    assert _resize_for(cpu_workers=4, cuda_workers=3) is first
    assert first.shutdowns == 0


def test_post_quantized_retune_applies_cuda_workers(fake_pool):
    """`create=False` (post-quantized loads) also retunes a stale pool."""

    first = _resize_for(cuda_workers=1)
    configure_kernel(KernelConfig(cuda_workers=2))
    second = gptqmodel.resize_device_thread_pool(create=False)
    assert second is not first and second.workers["cuda:per"] == 2


def test_typed_api_reaches_pool_and_bitsandbytes_triton():
    assert resolve_backend_config("auto", runtime=RuntimeConfig(pool_intraop_threads=2))[1].pool_intraop_threads == 2
    _, policy = resolve_backend_config(BitsAndBytesConfig(triton=True))
    assert policy.bitsandbytes_triton is True
    with pytest.raises(ValueError):
        RuntimeConfig(pool_intraop_threads=-1)


def test_gguf_cuda_config_has_no_unused_options():
    """ggml-cuda never reads the Torch fused-forward gates."""

    assert dataclasses.fields(GGUFCUDAConfig) == ()
    with pytest.raises(TypeError):
        GGUFCUDAConfig(fused_cuda_max_rows=8)


@pytest.mark.parametrize("cls,overrides", [
    (GPTQConfig, {"true_sequential": False, "damp_percent": 0.07, "offload_to_disk": False, "pack_impl": "cpu"}),
    (AWQConfig, {"true_sequential": False, "offload_to_disk": False, "pack_impl": "cpu"}),
])
def test_concrete_class_round_trip_keeps_meta_held_fields(cls, overrides):
    cfg = cls(bits=4, **overrides)
    payload = cfg.to_dict()
    concrete = cls(**payload)
    factory = QuantizeConfig(**payload)
    for name, value in overrides.items():
        assert getattr(concrete, name) == value
        assert getattr(concrete, name) == getattr(factory, name)


def test_retired_env_var_set_after_import_warns_at_load(monkeypatch):
    """A retired variable exported after `import gptqmodel` is reported at the next load."""

    warned = []
    monkeypatch.setattr(gptqmodel, "_WARNED_RETIRED_KERNEL_ENV_VARS", set())
    monkeypatch.setattr(gptqmodel, "log", SimpleNamespace(warn=warned.append, info=lambda *a, **k: None))
    monkeypatch.setenv("GPTQMODEL_CUDA_WORKERS", "2")

    def stop(*args, **kwargs):
        raise RuntimeError("stop")

    # Stop right after the policy is installed; no checkpoint is needed.
    monkeypatch.setattr(auto, "normalize_model_id_or_path_for_hf_gguf", stop)
    try:
        with pytest.raises(RuntimeError, match="stop"):
            auto.GPTQModel.from_quantized("does-not-exist")
    finally:
        reset_kernel_config()
    assert any("GPTQMODEL_CUDA_WORKERS" in message for message in warned)
