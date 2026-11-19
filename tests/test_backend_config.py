# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Typed backend selection, API forwarding and execution-policy ownership."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

import gptqmodel.backend as configs
from gptqmodel import GPTQModel, MarlinConfig
from gptqmodel.backend import BACKEND, BackendConfig, RuntimeConfig, TorchConfig
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.utils.backend import normalize_backend
from gptqmodel.utils.backend_config import resolve_backend_config, resolve_quantize_backend
from gptqmodel.utils.kernel import KernelConfig, configure_kernel, get_kernel_config, resolve_kernel_config


CONFIG_CLASSES = [
    getattr(configs, name) for name in configs.__all__
    if name not in {"BACKEND", "BackendConfig", "RuntimeConfig"}
]


@pytest.fixture(autouse=True)
def restore_policy():
    previous = get_kernel_config()
    yield
    configure_kernel(previous)


def test_backend_identity_is_fixed_and_base_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        BackendConfig()
    config = MarlinConfig(fp32=False)
    assert config.backend is BACKEND.MARLIN
    with pytest.raises(TypeError):
        MarlinConfig(backend=BACKEND.GPTQ_TORCH)
    with pytest.raises(FrozenInstanceError):
        config.backend = BACKEND.GPTQ_TORCH
    with pytest.raises(FrozenInstanceError):
        config.fp32 = True
    with pytest.raises(TypeError):
        MarlinConfig(streaming=True)


def test_existing_gguf_quantization_export_keeps_its_identity():
    from gptqmodel import GGUFConfig
    from gptqmodel.quantization import GGUFConfig as QuantizeGGUFConfig

    assert GGUFConfig is QuantizeGGUFConfig
    assert configs.GGUFTorchConfig().backend is BACKEND.GGUF_TORCH


@pytest.mark.parametrize("config_cls", CONFIG_CLASSES)
def test_typed_defaults_preserve_existing_execution_defaults(config_cls):
    config = config_cls()
    assert isinstance(config, BackendConfig)
    selected, policy = resolve_backend_config(config)
    assert selected is config.backend
    assert policy.to_dict() == KernelConfig().to_dict()
    assert normalize_backend(config) is config.backend


@pytest.mark.parametrize("selector", [BACKEND.MARLIN, "marlin", "MARLIN", "gptq_marlin", "awq_marlin"])
def test_enum_and_string_compatibility(selector):
    selected, policy = resolve_backend_config(selector)
    assert selected is BACKEND.MARLIN
    assert policy is None


@pytest.mark.parametrize("config_cls,options", [
    (MarlinConfig, {"fp32": 1}),
    (TorchConfig, {"stream_tile": 0}),
    (TorchConfig, {"streaming": "yes"}),
    (configs.AwqTritonConfig, {"schedule": "typo"}),
    (configs.SwordfishConfig, {"dense_m": -1}),
    (configs.GGUFCPPConfig, {"threads": False}),
    (configs.ParoTritonConfig, {"decode_max_rows": 0}),
    (RuntimeConfig, {"max_packers": 0}),
])
def test_invalid_options_fail_at_construction(config_cls, options):
    with pytest.raises((TypeError, ValueError)):
        config_cls(**options)


def test_shared_runtime_and_backend_options_are_composed_without_global_mutation():
    previous = get_kernel_config()
    _, policy = resolve_backend_config(MarlinConfig(fp32=False), runtime=RuntimeConfig(max_packers=2))
    assert policy.marlin_fp32 is False
    assert policy.max_packers == 2
    assert get_kernel_config() is previous


@pytest.mark.parametrize("entry", [GPTQModel.load, GPTQModel.from_pretrained, GPTQModel.from_quantized])
def test_conflicting_policies_fail_before_checkpoint_access(entry):
    kwargs = {"quantize_config": QuantizeConfig()} if entry.__name__ == "from_pretrained" else {}
    with pytest.raises(ValueError, match="cannot be supplied together"):
        entry("does-not-exist", backend=MarlinConfig(), kernel=KernelConfig(), **kwargs)


def test_legacy_policy_and_shared_runtime_reject_ambiguity():
    with pytest.raises(ValueError, match="cannot be supplied together"):
        resolve_backend_config(BACKEND.AUTO, KernelConfig(), RuntimeConfig())
    with pytest.raises(ValueError, match="cannot be supplied together"):
        QuantizeConfig(runtime=RuntimeConfig(), kernel=KernelConfig())


def test_shared_runtime_is_not_checkpoint_metadata():
    qcfg = QuantizeConfig(runtime=RuntimeConfig(max_packers=2))
    assert resolve_kernel_config(qcfg).max_packers == 2
    payload = qcfg.to_dict()
    assert "runtime" not in payload
    assert "runtime" not in payload.get("meta", {})
    assert "kernel" not in payload


def test_quantization_retains_and_can_replace_a_typed_load_choice():
    qcfg = QuantizeConfig(runtime=RuntimeConfig(max_packers=2))
    resolve_quantize_backend(MarlinConfig(fp32=False), qcfg)
    selected, policy = resolve_quantize_backend("auto", qcfg, inherit=True)
    assert selected is BACKEND.MARLIN
    assert policy.marlin_fp32 is False
    assert policy.max_packers == 2
    selected, policy = resolve_quantize_backend(TorchConfig(streaming=True), qcfg, inherit=True)
    assert selected is BACKEND.GPTQ_TORCH
    assert policy.torch_streaming is True
    assert policy.marlin_fp32 is True
    selected, _ = resolve_quantize_backend(BACKEND.MARLIN, qcfg, inherit=True)
    assert selected is BACKEND.MARLIN


def test_quantization_rejects_a_caller_legacy_policy_even_after_a_typed_load():
    qcfg = QuantizeConfig(kernel=KernelConfig())
    with pytest.raises(ValueError, match="cannot be supplied together"):
        resolve_quantize_backend(MarlinConfig(), qcfg, inherit=True)
    qcfg.kernel = None
    resolve_quantize_backend(MarlinConfig(), qcfg)
    qcfg.kernel = KernelConfig()
    with pytest.raises(ValueError, match="cannot be supplied together"):
        resolve_quantize_backend(BACKEND.AUTO, qcfg, inherit=True)


@pytest.mark.parametrize("entry", ["load", "from_pretrained", "from_quantized"])
def test_public_entry_points_forward_typed_selection(monkeypatch, tmp_path, entry):
    from gptqmodel.models import auto

    captured = {}
    # Current GPTQModel.load logs the resource budget on the returned model.
    result = SimpleNamespace(log_budget=lambda: None)

    def receive(**kwargs):
        captured.update(kwargs)
        return result

    definition = SimpleNamespace(from_quantized=receive, from_pretrained=receive)
    monkeypatch.setattr(auto, "check_and_get_model_definition", lambda *args, **kwargs: definition)
    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda *args, **kwargs: False)
    monkeypatch.setattr(auto, "patch_remote_code_before_config_load", lambda *args, **kwargs: None)
    monkeypatch.setattr(auto, "native_floatx_source_format", lambda *args, **kwargs: None)
    monkeypatch.setattr(auto, "_kick_quantize_prewarms", lambda *args, **kwargs: None)
    monkeypatch.setattr(auto, "_prebuild_device_thread_pool", lambda *args, **kwargs: None)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(auto, "_is_supported_quantization_config", lambda *args: entry != "from_pretrained")
    kwargs = {"backend": MarlinConfig(fp32=False)}
    if entry == "from_pretrained":
        kwargs["quantize_config"] = QuantizeConfig(runtime=RuntimeConfig(max_packers=2))
    assert getattr(GPTQModel, entry)(str(tmp_path), **kwargs) is result
    if entry == "from_pretrained":
        # The model definition re-resolves the caller's own arguments against
        # the same config, so it receives them unchanged.
        qcfg = captured["quantize_config"]
        assert captured["backend"] is kwargs["backend"]
        assert captured["kernel"] is None
        assert qcfg.kernel is None
        assert qcfg._backend_config == kwargs["backend"]
        policy = resolve_kernel_config(qcfg)
        assert policy.max_packers == 2
    else:
        assert captured["backend"] is BACKEND.MARLIN
        policy = captured["kernel"]
    assert policy.marlin_fp32 is False


def _torch_layer(config):
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear

    layer = TorchLinear(
        bits=4, group_size=32, sym=True, desc_act=False,
        in_features=32, out_features=32, bias=False,
        backend=config, register_buffers=True, dtype=torch.float16,
    ).eval()
    layer.qweight.fill_(0x11111111)
    layer.qzeros.zero_()
    layer.scales.fill_(1)
    layer.g_idx.zero_()
    return layer


def test_two_models_keep_settings_and_outputs_after_policy_replacement():
    first = _torch_layer(TorchConfig(streaming=True, stream_tile=16, triton_dequant=False))
    second = _torch_layer(TorchConfig(streaming=False, stream_tile=32, triton_dequant=False))
    configure_kernel(KernelConfig(torch_streaming=False, torch_stream_tile=128))
    assert first.kernel is not second.kernel
    assert first._streaming_enabled is True
    assert second._streaming_enabled is False
    assert first._stream_tile_cols == 16
    assert second._stream_tile_cols == 32
    x = torch.arange(64, dtype=torch.float16).reshape(2, 32) / 64
    # Every packed nibble is 1 with zero point 0 and scale 1.
    expected = x.float().sum(dim=-1, keepdim=True).expand(2, 32).half()
    with torch.no_grad():
        for layer in (first, second, first):
            torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)


def test_explicit_kernel_object_is_resolved_and_snapshotted():
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear

    policy = KernelConfig(torch_streaming=True, torch_stream_tile=16)
    assert resolve_kernel_config(policy) is policy
    layer = TorchLinear(
        bits=4, group_size=32, sym=True, desc_act=False,
        in_features=32, out_features=32, bias=False, kernel=policy,
    )
    policy.torch_stream_tile = 128
    assert layer.kernel.torch_stream_tile == 16
    assert layer._streaming_enabled is True


def test_awq_native_options_are_passed_to_the_op_without_global_lookup(monkeypatch):
    from gptqmodel.nn_modules.qlinear import gemm_awq

    captured = []
    monkeypatch.setattr(gemm_awq, "awq_gemm_forward", lambda *args: captured.append(args))
    policy = configs.AwqGemmConfig(fp32_accum=False, fused_splitk_reduce=False).to_kernel_config()
    configure_kernel(KernelConfig(fp32_accum=True, awq_fused_splitk_reduce=True))
    gemm_awq._awq_cuda_gemm_forward(None, None, None, None, 8, kernel=policy)
    assert captured[0][-2:] == (False, False)
