# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

import gptqmodel.models.base as base_module
from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.looper.named_module import NamedModule
import gptqmodel.models.loader as loader_module
from gptqmodel.models.loader import (
    configure_native_floatx_source_quantization,
    native_floatx_source_format,
)
from gptqmodel.nn_modules.qlinear.fp4 import TorchFP4Linear
from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from gptqmodel.quantization import AutoModuleDecoderConfig, QuantizeConfig
from gptqmodel.quantization.dtype import (
    dequantize_f4_e2m1,
    dequantize_fp8,
)
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.utils.device_telemetry import (
    clear_device_telemetry_records,
    device_telemetry_scope,
    get_device_telemetry_records,
)
from gptqmodel.utils.structure import LazyTurtle


try:
    from torchao.prototype.mx_formats.nvfp4_tensor import nvfp4_quantize
except Exception:
    nvfp4_quantize = None


class _LinearWrapper(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)


def _write_index(path: Path, shard_name: str, keys: list[str]) -> None:
    weight_map = dict.fromkeys(keys, shard_name)
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )


def test_native_floatx_source_configuration_injects_decoder_and_preserves_lazy_weights(monkeypatch):
    source_config = SimpleNamespace(
        quantization_config={
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
        }
    )
    qcfg = QuantizeConfig(bits=4, offload_to_disk=False)

    monkeypatch.setattr(loader_module, "device_supports_dtype", lambda *args, **kwargs: True)

    source_format = configure_native_floatx_source_quantization(
        source_config,
        qcfg,
        device=torch.device("cuda:0"),
    )

    assert source_format == "nvfp4"
    assert qcfg.offload_to_disk is True
    assert qcfg._native_floatx_forward_plan["source_format"] == "nvfp4"
    assert qcfg._native_floatx_forward_plan["bf16_validated"] is True
    decoder = next(item for item in qcfg.preprocessors if isinstance(item, AutoModuleDecoderConfig))
    assert decoder.target_dtype is torch.bfloat16


def test_native_floatx_source_configuration_fails_before_model_load_without_bf16(monkeypatch):
    source_config = SimpleNamespace(quantization_config={"quant_algo": "FP8"})
    qcfg = QuantizeConfig(bits=4, offload_to_disk=False)

    monkeypatch.setattr(loader_module, "device_supports_dtype", lambda *args, **kwargs: False)

    with pytest.raises(EnvironmentError, match="BF16 linear support"):
        configure_native_floatx_source_quantization(
            source_config,
            qcfg,
            device=torch.device("cuda:0"),
        )

    assert qcfg.offload_to_disk is False
    assert not any(isinstance(item, AutoModuleDecoderConfig) for item in qcfg.preprocessors)


def test_native_floatx_source_configuration_preserves_fp32_hessian(monkeypatch):
    source_config = SimpleNamespace(quantization_config={"quant_algo": "FP8"})
    qcfg = QuantizeConfig(bits=4)
    qcfg.hessian.staging_dtype = torch.bfloat16

    monkeypatch.setattr(loader_module, "device_supports_dtype", lambda *args, **kwargs: True)

    configure_native_floatx_source_quantization(
        source_config,
        qcfg,
        device=torch.device("cuda:0"),
    )

    assert qcfg.hessian.staging_dtype is torch.float32
    assert qcfg._native_floatx_forward_plan["hessian_accumulation_dtype"] == "float32"


def test_native_floatx_source_format_reads_modelopt_sidecar(tmp_path):
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quant_algo": "FP8"}}),
        encoding="utf-8",
    )

    assert native_floatx_source_format(
        SimpleNamespace(quantization_config=None),
        model_local_path=str(tmp_path),
    ) == "fp8"


def test_auto_module_decoder_config_exposes_validated_policies():
    config = AutoModuleDecoderConfig(
        passthrough_forward_policy="NATIVE",
        passthrough_save_policy="decode",
    )

    assert config.passthrough_forward_policy == "native"
    assert config.to_dict()["passthrough_save_policy"] == "decode"

    with pytest.raises(ValueError, match="passthrough_forward_policy"):
        AutoModuleDecoderConfig(passthrough_forward_policy="invalid")


def test_fp4_decoder_has_torch_only_fallback(monkeypatch):
    import gptqmodel.quantization.dtype as dtype_module

    monkeypatch.setattr(dtype_module, "unpack_uint4", None)
    monkeypatch.setattr(dtype_module, "f4_unpacked_to_f32", None)
    packed = torch.tensor([[0x10, 0x98]], dtype=torch.uint8)

    decoded = dequantize_f4_e2m1(packed, axis=None, target_dtype=torch.bfloat16)

    expected = torch.tensor([[0.0, 0.5, 0.0, -0.5]], dtype=torch.bfloat16)
    torch.testing.assert_close(decoded, expected)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_shell_materialize_forward_defers_fp8_quant_source_until_quantization(tmp_path, monkeypatch):
    source_model = _LinearWrapper(16, 8).eval()
    model_dir = tmp_path / "fp8_source"
    model_dir.mkdir()

    weight_fp8 = source_model.linear.weight.detach().to(torch.float8_e4m3fn).cpu()
    scale_inv = torch.linspace(2.0, 3.75, steps=source_model.linear.out_features, dtype=torch.float32)
    bias = source_model.linear.bias.detach().cpu()
    shard_name = "model.safetensors"
    save_file(
        {
            "linear.weight": weight_fp8,
            "linear.weight_scale_inv": scale_inv,
            "linear.bias": bias,
        },
        str(model_dir / shard_name),
    )
    _write_index(model_dir, shard_name, ["linear.weight", "linear.weight_scale_inv", "linear.bias"])

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    shell_model = _LinearWrapper(16, 8).eval()
    shell_model.linear.weight = nn.Parameter(
        torch.empty_like(shell_model.linear.weight, device="meta"),
        requires_grad=False,
    )
    shell_model.linear.bias = nn.Parameter(
        torch.empty_like(shell_model.linear.bias, device="meta"),
        requires_grad=False,
    )

    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = shell_model
    harness.turtle_model = turtle
    harness._turtle_lock = threading.RLock()
    harness.auto_module_decoder_events = []

    named = NamedModule(shell_model.linear, name="linear", full_name="linear", layer_index=0)
    named.state["auto_module_decoder"] = {
        "code": "auto_module_decoder",
        "source_dtype": "auto",
        "target_dtype": torch.bfloat16,
        "quant_source_decode_policy": "defer_until_quantization",
    }

    monkeypatch.setattr(base_module, "device_supports_dtype", lambda *args, **kwargs: True)

    clear_device_telemetry_records()
    with device_telemetry_scope(True):
        prepared = base_module.BaseQModel.shell_module_materialize(
            harness,
            target_submodule=shell_model.linear,
            device=torch.device("cpu"),
            role="forward",
            named_module=named,
        )

    assert isinstance(prepared, TorchFP8Linear)
    assert isinstance(shell_model.linear, TorchFP8Linear)
    assert named.state["auto_module_decoder_forward_mode"] == "native"
    assert "quant_source_module" not in named.state
    expected = dequantize_fp8(
        weight_fp8,
        scale_inv=scale_inv,
        axis=None,
        target_dtype=torch.bfloat16,
    )
    with device_telemetry_scope(True):
        quant_source = base_module.BaseQModel.shell_module_materialize(
            harness,
            target_submodule=prepared,
            device=torch.device("cpu"),
            role="quant_source",
            named_module=named,
        )
    assert isinstance(quant_source, nn.Linear)
    assert quant_source.weight.device.type == "cpu"
    assert quant_source.weight.dtype == torch.bfloat16
    torch.testing.assert_close(quant_source.weight, expected)
    assert harness.auto_module_decoder_events[0]["forward_mode"] == "native"
    assert harness.auto_module_decoder_events[0]["deferred_quant_source_decode"] is True
    assert {record["event"] for record in get_device_telemetry_records()} == {
        "floatx_forward_plan",
        "floatx_native_forward_used",
        "floatx_deferred_quant_source_decode",
    }


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_shell_materialize_forward_decodes_when_passthrough_forward_policy_is_decode(tmp_path, monkeypatch):
    source_model = _LinearWrapper(16, 8).eval()
    model_dir = tmp_path / "fp8_source_decode"
    model_dir.mkdir()

    weight_fp8 = source_model.linear.weight.detach().to(torch.float8_e4m3fn).cpu()
    scale_inv = torch.linspace(2.0, 3.75, steps=source_model.linear.out_features, dtype=torch.float32)
    bias = source_model.linear.bias.detach().cpu()
    shard_name = "model.safetensors"
    save_file(
        {
            "linear.weight": weight_fp8,
            "linear.weight_scale_inv": scale_inv,
            "linear.bias": bias,
        },
        str(model_dir / shard_name),
    )
    _write_index(model_dir, shard_name, ["linear.weight", "linear.weight_scale_inv", "linear.bias"])

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    shell_model = _LinearWrapper(16, 8).eval()
    shell_model.linear.weight = nn.Parameter(
        torch.empty_like(shell_model.linear.weight, device="meta"),
        requires_grad=False,
    )
    shell_model.linear.bias = nn.Parameter(
        torch.empty_like(shell_model.linear.bias, device="meta"),
        requires_grad=False,
    )

    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = shell_model
    harness.turtle_model = turtle
    harness._turtle_lock = threading.RLock()
    harness.auto_module_decoder_events = []

    named = NamedModule(shell_model.linear, name="linear", full_name="linear", layer_index=0)
    named.state["auto_module_decoder"] = {
        "code": "auto_module_decoder",
        "source_dtype": "auto",
        "target_dtype": torch.bfloat16,
        "passthrough_forward_policy": "decode",
        "passthrough_save_policy": "decode",
    }

    monkeypatch.setattr(base_module, "device_supports_dtype", lambda *args, **kwargs: True)

    prepared = base_module.BaseQModel.shell_module_materialize(
        harness,
        target_submodule=shell_model.linear,
        device=torch.device("cpu"),
        role="forward",
        named_module=named,
    )

    expected = dequantize_fp8(
        weight_fp8,
        scale_inv=scale_inv,
        axis=None,
        target_dtype=torch.bfloat16,
    )
    assert isinstance(prepared, nn.Linear)
    assert not isinstance(prepared, TorchFP8Linear)
    assert named.state["auto_module_decoder_forward_mode"] == "decode"
    torch.testing.assert_close(prepared.weight, expected)
    assert harness.auto_module_decoder_events[0]["forward_mode"] == "decode"


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_shell_materialize_quant_source_swaps_back_to_dense_module(tmp_path, monkeypatch):
    source_model = _LinearWrapper(8, 4).eval()
    model_dir = tmp_path / "fp8_source"
    model_dir.mkdir()

    shard_name = "model.safetensors"
    save_file(
        {
            "linear.weight": source_model.linear.weight.detach().to(torch.float8_e4m3fn).cpu(),
            "linear.weight_scale_inv": torch.ones(4, dtype=torch.float32),
            "linear.bias": source_model.linear.bias.detach().cpu(),
        },
        str(model_dir / shard_name),
    )
    _write_index(model_dir, shard_name, ["linear.weight", "linear.weight_scale_inv", "linear.bias"])

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    shell_model = _LinearWrapper(8, 4).eval()
    shell_model.linear.weight = nn.Parameter(
        torch.empty_like(shell_model.linear.weight, device="meta"),
        requires_grad=False,
    )
    shell_model.linear.bias = nn.Parameter(
        torch.empty_like(shell_model.linear.bias, device="meta"),
        requires_grad=False,
    )

    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = shell_model
    harness.turtle_model = turtle
    harness._turtle_lock = threading.RLock()
    harness.auto_module_decoder_events = []

    named = NamedModule(shell_model.linear, name="linear", full_name="linear", layer_index=0)
    named.state["auto_module_decoder"] = {
        "code": "auto_module_decoder",
        "source_dtype": "auto",
        "target_dtype": torch.bfloat16,
    }

    monkeypatch.setattr(base_module, "device_supports_dtype", lambda *args, **kwargs: True)

    forward_module = base_module.BaseQModel.shell_module_materialize(
        harness,
        target_submodule=shell_model.linear,
        device=torch.device("cpu"),
        role="forward",
        named_module=named,
    )
    named.module = forward_module

    quant_source = base_module.BaseQModel.shell_module_materialize(
        harness,
        target_submodule=forward_module,
        device=torch.device("cpu"),
        role="quant_source",
        named_module=named,
    )

    assert isinstance(quant_source, nn.Linear)
    assert isinstance(shell_model.linear, nn.Linear)
    torch.testing.assert_close(
        quant_source.weight,
        named.state["quant_source_module"].weight,
    )


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_shell_materialize_forward_builds_fp8_wrapper_from_weight_scale_metadata(tmp_path, monkeypatch):
    source_model = _LinearWrapper(16, 8).eval()
    model_dir = tmp_path / "fp8_source_scale"
    model_dir.mkdir()

    weight_fp8 = source_model.linear.weight.detach().to(torch.float8_e4m3fn).cpu()
    scale = torch.full((), 0.5, dtype=torch.float32)
    bias = source_model.linear.bias.detach().cpu()
    shard_name = "model.safetensors"
    save_file(
        {
            "linear.weight": weight_fp8,
            "linear.weight_scale": scale,
            "linear.bias": bias,
        },
        str(model_dir / shard_name),
    )
    _write_index(model_dir, shard_name, ["linear.weight", "linear.weight_scale", "linear.bias"])

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    shell_model = _LinearWrapper(16, 8).eval()
    shell_model.linear.weight = nn.Parameter(
        torch.empty_like(shell_model.linear.weight, device="meta"),
        requires_grad=False,
    )
    shell_model.linear.bias = nn.Parameter(
        torch.empty_like(shell_model.linear.bias, device="meta"),
        requires_grad=False,
    )

    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = shell_model
    harness.turtle_model = turtle
    harness._turtle_lock = threading.RLock()
    harness.auto_module_decoder_events = []

    named = NamedModule(shell_model.linear, name="linear", full_name="linear", layer_index=0)
    named.state["auto_module_decoder"] = {
        "code": "auto_module_decoder",
        "source_dtype": "auto",
        "target_dtype": torch.bfloat16,
    }

    monkeypatch.setattr(base_module, "device_supports_dtype", lambda *args, **kwargs: True)

    prepared = base_module.BaseQModel.shell_module_materialize(
        harness,
        target_submodule=shell_model.linear,
        device=torch.device("cpu"),
        role="forward",
        named_module=named,
    )

    assert isinstance(prepared, TorchFP8Linear)
    assert named.state["auto_module_decoder_forward_mode"] == "native"
    assert "quant_source_module" not in named.state
    torch.testing.assert_close(prepared.weight_scale_inv, torch.reciprocal(scale).to(torch.float32))
    assert harness.auto_module_decoder_events[0]["forward_mode"] == "native"


@pytest.mark.skipif(nvfp4_quantize is None, reason="torchao NVFP4 support required")
@pytest.mark.skipif(not hasattr(torch, "float4_e2m1fn_x2"), reason="float4 packed dtype not available")
def test_shell_materialize_forward_decodes_fp4_source_to_dense_module(tmp_path):
    source_model = _LinearWrapper(16, 4).eval()
    model_dir = tmp_path / "fp4_source"
    model_dir.mkdir()

    scales, packed = nvfp4_quantize(source_model.linear.weight.detach().to(torch.float32), block_size=16)
    packed_float4 = packed.view(torch.float4_e2m1fn_x2)
    bias = source_model.linear.bias.detach().cpu()
    shard_name = "model.safetensors"
    save_file(
        {
            "linear.weight": packed_float4.cpu(),
            "linear.weight_scale": scales.cpu(),
            "linear.bias": bias,
        },
        str(model_dir / shard_name),
    )
    _write_index(model_dir, shard_name, ["linear.weight", "linear.weight_scale", "linear.bias"])

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    shell_model = _LinearWrapper(16, 4).eval()
    shell_model.linear.weight = nn.Parameter(
        torch.empty_like(shell_model.linear.weight, device="meta"),
        requires_grad=False,
    )
    shell_model.linear.bias = nn.Parameter(
        torch.empty_like(shell_model.linear.bias, device="meta"),
        requires_grad=False,
    )

    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = shell_model
    harness.turtle_model = turtle
    harness._turtle_lock = threading.RLock()
    harness.auto_module_decoder_events = []

    named = NamedModule(shell_model.linear, name="linear", full_name="linear", layer_index=0)
    named.state["auto_module_decoder"] = {
        "code": "auto_module_decoder",
        "source_dtype": "auto",
        "target_dtype": torch.bfloat16,
    }

    prepared = base_module.BaseQModel.shell_module_materialize(
        harness,
        target_submodule=shell_model.linear,
        device=torch.device("cpu"),
        role="forward",
        named_module=named,
    )

    expected = dequantize_f4_e2m1(
        packed_float4.cpu(),
        scale=scales.cpu(),
        axis=None,
        target_dtype=torch.bfloat16,
    )
    assert isinstance(prepared, nn.Linear)
    assert not isinstance(prepared, TorchFP8Linear)
    assert named.state["auto_module_decoder_forward_mode"] == "decode"
    torch.testing.assert_close(prepared.weight, expected, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(named.state["quant_source_module"].weight, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(nvfp4_quantize is None, reason="torchao NVFP4 support required")
@pytest.mark.skipif(not hasattr(torch, "float4_e2m1fn_x2"), reason="float4 packed dtype not available")
def test_shell_materialize_forward_builds_fp4_wrapper_when_native_supported(tmp_path, monkeypatch):
    source_model = _LinearWrapper(16, 4).eval()
    model_dir = tmp_path / "fp4_native_source"
    model_dir.mkdir()

    scales, packed = nvfp4_quantize(source_model.linear.weight.detach().to(torch.float32), block_size=16)
    packed_float4 = packed.view(torch.float4_e2m1fn_x2)
    bias = source_model.linear.bias.detach().cpu()
    shard_name = "model.safetensors"
    save_file(
        {
            "linear.weight": packed_float4.cpu(),
            "linear.weight_scale": scales.cpu(),
            "linear.bias": bias,
        },
        str(model_dir / shard_name),
    )
    _write_index(model_dir, shard_name, ["linear.weight", "linear.weight_scale", "linear.bias"])

    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(
            _experts_implementation=None,
            quantization_config={"format": "nvfp4"},
        ),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None

    shell_model = _LinearWrapper(16, 4).eval()
    shell_model.config = SimpleNamespace(quantization_config={"format": "nvfp4"})
    shell_model.linear.weight = nn.Parameter(
        torch.empty_like(shell_model.linear.weight, device="meta"),
        requires_grad=False,
    )
    shell_model.linear.bias = nn.Parameter(
        torch.empty_like(shell_model.linear.bias, device="meta"),
        requires_grad=False,
    )

    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = shell_model
    harness.turtle_model = turtle
    harness._turtle_lock = threading.RLock()
    harness.auto_module_decoder_events = []

    named = NamedModule(shell_model.linear, name="linear", full_name="linear", layer_index=0)
    named.state["auto_module_decoder"] = {
        "code": "auto_module_decoder",
        "source_dtype": "auto",
        "target_dtype": torch.bfloat16,
    }

    monkeypatch.setattr(base_module, "device_supports_native_fp4", lambda *args, **kwargs: True)

    prepared = base_module.BaseQModel.shell_module_materialize(
        harness,
        target_submodule=shell_model.linear,
        device=torch.device("cpu"),
        role="forward",
        named_module=named,
    )

    expected = dequantize_f4_e2m1(
        packed_float4.cpu(),
        scale=scales.cpu(),
        axis=None,
        target_dtype=torch.bfloat16,
    )
    assert isinstance(prepared, TorchFP4Linear)
    assert named.state["auto_module_decoder_forward_mode"] == "native"
    assert "quant_source_module" not in named.state


def test_configure_modelopt_runtime_rejects_modelopt_input_activation_quantization():
    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = SimpleNamespace(
        config=SimpleNamespace(
            quantization_config={
                "quant_method": "modelopt",
                "config_groups": {
                    "group_0": {
                        "input_activations": {"num_bits": 4, "type": "float", "dynamic": False},
                    }
                },
            },
        ),
        state_dict=lambda: {},
    )
    harness.turtle_model = None

    with pytest.raises(ValueError, match="activation quantization"):
        base_module.BaseQModel._configure_modelopt_runtime(harness)


def test_configure_modelopt_runtime_allows_native_source_requantization():
    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = SimpleNamespace(
        config=SimpleNamespace(
            quantization_config={
                "quant_method": "modelopt",
                "config_groups": {
                    "group_0": {
                        "input_activations": {"num_bits": 4, "type": "float", "dynamic": False},
                    }
                },
            }
        ),
        state_dict=lambda: {},
    )
    harness.turtle_model = None
    harness.quantize_config = SimpleNamespace(preprocessors=[AutoModuleDecoderConfig()])

    base_module.BaseQModel._configure_modelopt_runtime(harness)


def test_configure_modelopt_runtime_rejects_checkpoint_activation_scale_metadata():
    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = SimpleNamespace(
        config=SimpleNamespace(quantization_config={"quant_method": "modelopt"}),
        state_dict=lambda: {},
    )
    harness.turtle_model = base_module.LazyTurtle.__new__(base_module.LazyTurtle)
    harness.turtle_model._weight_map = {"model.layers.0.self_attn.q_proj.input_scale": "model.safetensors"}

    with pytest.raises(ValueError, match="activation quantization"):
        base_module.BaseQModel._configure_modelopt_runtime(harness)


def test_configure_modelopt_runtime_allows_weight_only_modelopt_metadata():
    harness = base_module.BaseQModel.__new__(base_module.BaseQModel)
    nn.Module.__init__(harness)
    harness.model = SimpleNamespace(
        config=SimpleNamespace(
            quantization_config={
                "quant_method": "modelopt",
                "config_groups": {
                    "group_0": {
                        "weights": {"num_bits": 4, "type": "float", "dynamic": False},
                    }
                },
            },
        ),
        state_dict=lambda: {},
    )
    harness.turtle_model = base_module.LazyTurtle.__new__(base_module.LazyTurtle)
    harness.turtle_model._weight_map = {"model.layers.0.self_attn.q_proj.weight_scale": "model.safetensors"}

    base_module.BaseQModel._configure_modelopt_runtime(harness)


def test_gptq_prefers_quant_source_module_when_present():
    forward_module = nn.Linear(8, 4, bias=False)
    quant_source = nn.Linear(8, 4, bias=False)
    named = NamedModule(forward_module, name="linear", full_name="linear", layer_index=0)
    named.state["quant_source_module"] = quant_source

    task = GPTQ(named)

    assert task.module is quant_source


def test_module_looper_materializes_deferred_native_source_before_gptq():
    """The native Hessian wrapper must never become the GPTQ source module."""

    forward_module = nn.Linear(8, 4, bias=False)
    quant_source = nn.Linear(8, 4, bias=False)
    named = NamedModule(forward_module, name="linear", full_name="linear", layer_index=0)
    named.state.update(
        {
            "auto_module_decoder": {"code": "auto_module_decoder"},
            "auto_module_decoder_forward_mode": "native",
        }
    )

    calls = []

    class _Model:
        def shell_module_materialize(self, **kwargs):
            calls.append(kwargs["role"])
            kwargs["named_module"].state["quant_source_module"] = quant_source
            return quant_source

    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = _Model()
    looper._assign_quant_device_for_module = lambda named_module, fallback_device: fallback_device

    ModuleLooper._prepare_named_module_for_quantization(
        looper,
        SimpleNamespace(tasks={}),
        named,
        torch.device("cpu"),
    )

    assert calls == ["quant_source"]
    assert named.module is quant_source


def test_awq_resolve_quant_source_module_prefers_dense_source():
    forward_module = nn.Linear(8, 4, bias=False)
    quant_source = nn.Linear(8, 4, bias=False)
    named = NamedModule(forward_module, name="linear", full_name="linear", layer_index=0)
    named.state["quant_source_module"] = quant_source

    resolved = AWQProcessor.resolve_quant_source_module(named)

    assert resolved is quant_source
