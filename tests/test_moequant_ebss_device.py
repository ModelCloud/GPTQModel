# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=1

"""MoEQuant EBSS must run its self-sampling search on the quantization device."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import gptqmodel.utils.ebss as ebss
from gptqmodel.models.base import BaseQModel
from gptqmodel.models.ports.native.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from gptqmodel.models.ports.native.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
from gptqmodel.quantization.config import CalibrationConfig, EBSSConfig, MoEQuantConfig


pytestmark = [pytest.mark.gpu]


def _tiny_model() -> Qwen3MoeForCausalLM:
    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        head_dim=8,
    )
    return Qwen3MoeForCausalLM(config).eval()


def _quantizing_wrapper(model: Qwen3MoeForCausalLM, ebss_config: EBSSConfig) -> BaseQModel:
    """Build the small part of ``GPTQModel`` that ``_maybe_generate_ebss_calibration`` reads."""

    wrapper = BaseQModel.__new__(BaseQModel)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = model
    wrapper.turtle_model = None
    wrapper.quantize_config = SimpleNamespace(
        moequant=MoEQuantConfig(agq=False, ebss=ebss_config),
        device="cuda:0",
    )
    wrapper.extract_layers_node = lambda: ["model.layers"]
    return wrapper


def _capture_input_devices(model: Qwen3MoeForCausalLM):
    devices = []

    def hook(_module, _args, kwargs):
        devices.append(kwargs["input_ids"].device)

    return devices, model.register_forward_pre_hook(hook, with_kwargs=True)


def _assert_cpu_placement_restored(model: Qwen3MoeForCausalLM) -> None:
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert all(buffer.device.type == "cpu" for buffer in model.buffers())
    assert not any(
        getattr(module, "_hf_hook", None) is not None for module in model.modules()
    )
    with torch.inference_mode():
        output = model(input_ids=torch.tensor([[1, 2]]))
    assert output.logits.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_quantize_ebss_search_uses_quantize_device_and_restores_placement():
    model = _tiny_model()
    wrapper = _quantizing_wrapper(
        model, EBSSConfig(num_sentences=1, seq_len=3, width=2)
    )
    target = wrapper._pre_quantize_device()
    assert target.type == "cuda"

    devices, handle = _capture_input_devices(model)
    try:
        calibration = wrapper._maybe_generate_ebss_calibration(CalibrationConfig())
    finally:
        handle.remove()

    assert len(calibration.dataset) == 1
    assert devices and all(device == target for device in devices)
    _assert_cpu_placement_restored(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_quantize_ebss_search_layerwise_fallback_uses_device_and_restores_placement(
    monkeypatch,
):
    model = _tiny_model()
    wrapper = _quantizing_wrapper(
        model, EBSSConfig(num_sentences=1, seq_len=3, width=2)
    )
    target = wrapper._pre_quantize_device()
    assert target.type == "cuda"

    monkeypatch.setattr(ebss, "_model_fits_on_device", lambda *_args, **_kwargs: False)

    layerwise_used = []
    original_layerwise = ebss._layerwise_model_execution

    @contextmanager
    def record_layerwise(*args, **kwargs):
        layerwise_used.append(True)
        with original_layerwise(*args, **kwargs):
            yield

    monkeypatch.setattr(ebss, "_layerwise_model_execution", record_layerwise)

    devices, handle = _capture_input_devices(model)
    layer_devices = []
    layer_handle = model.model.layers[0].register_forward_hook(
        lambda module, *_args, **_kwargs: layer_devices.append(
            next(module.parameters()).device
        )
    )
    try:
        calibration = wrapper._maybe_generate_ebss_calibration(CalibrationConfig())
    finally:
        handle.remove()
        layer_handle.remove()

    assert len(calibration.dataset) == 1
    assert layerwise_used
    assert devices and all(device == target for device in devices)
    assert layer_devices and all(device == target for device in layer_devices)
    _assert_cpu_placement_restored(model)
