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
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.base import BaseQModel
from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from gptqmodel.quantization.dtype import dequantize_f4_e2m1, dequantize_fp8
from gptqmodel.quantization.gptq import GPTQ
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


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8 dtype not available")
def test_shell_materialize_forward_builds_fp8_wrapper_and_quant_source(tmp_path, monkeypatch):
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

    harness = BaseQModel.__new__(BaseQModel)
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

    prepared = BaseQModel.shell_module_materialize(
        harness,
        target_submodule=shell_model.linear,
        device=torch.device("cpu"),
        role="forward",
        named_module=named,
    )

    assert isinstance(prepared, TorchFP8Linear)
    assert isinstance(shell_model.linear, TorchFP8Linear)
    assert named.state["auto_module_decoder_forward_mode"] == "native"
    assert isinstance(named.state["quant_source_module"], nn.Linear)
    assert named.state["quant_source_module"].weight.device.type == "cpu"
    assert named.state["quant_source_module"].weight.dtype == torch.bfloat16
    expected = dequantize_fp8(
        weight_fp8,
        scale_inv=scale_inv,
        axis=None,
        target_dtype=torch.bfloat16,
    )
    torch.testing.assert_close(named.state["quant_source_module"].weight, expected)
    assert harness.auto_module_decoder_events[0]["forward_mode"] == "native"


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

    harness = BaseQModel.__new__(BaseQModel)
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

    forward_module = BaseQModel.shell_module_materialize(
        harness,
        target_submodule=shell_model.linear,
        device=torch.device("cpu"),
        role="forward",
        named_module=named,
    )
    named.module = forward_module

    quant_source = BaseQModel.shell_module_materialize(
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

    harness = BaseQModel.__new__(BaseQModel)
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

    prepared = BaseQModel.shell_module_materialize(
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


def test_gptq_prefers_quant_source_module_when_present():
    forward_module = nn.Linear(8, 4, bias=False)
    quant_source = nn.Linear(8, 4, bias=False)
    named = NamedModule(forward_module, name="linear", full_name="linear", layer_index=0)
    named.state["quant_source_module"] = quant_source

    task = GPTQ(named)

    assert task.module is quant_source


def test_awq_resolve_quant_source_module_prefers_dense_source():
    forward_module = nn.Linear(8, 4, bias=False)
    quant_source = nn.Linear(8, 4, bias=False)
    named = NamedModule(forward_module, name="linear", full_name="linear", layer_index=0)
    named.state["quant_source_module"] = quant_source

    resolved = AWQProcessor.resolve_quant_source_module(named)

    assert resolved is quant_source
