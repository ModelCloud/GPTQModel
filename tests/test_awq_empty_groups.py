# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ must quantize a layer even when no scaling group can be built.

Phi3/Apertus/CodeGen/MPT used to produce an empty scaling map; the layer was
then finalized without quantizing, and packing failed later with
``KeyError: 'q_zeros'``. Both empty-group paths must instead fall back to plain
quantization for the layer's supported modules.
"""

from types import SimpleNamespace

import torch
from torch import nn

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization.config import AWQConfig, CalibrationConfig


def _processor(model):
    processor = AWQProcessor(
        tokenizer=None,
        qcfg=AWQConfig(device="cpu"),
        calibration=CalibrationConfig(),
        gptq_model=SimpleNamespace(
            model=model,
            qlinear_kernel=None,
            rotary_embedding=None,
            lm_head="lm_head",
        ),
        model=model,
        calculate_w_wq_diff=False,
    )
    processor.draw_progress = lambda *args, **kwargs: None
    processor.log_new_row = lambda *args, **kwargs: None
    return processor


def _layer_and_processor(monkeypatch):
    layer = nn.Module()
    layer.linear = nn.Linear(16, 8, bias=False)
    layer.other = nn.Linear(16, 8, bias=False)
    model = nn.Module()
    model.layers = nn.ModuleList([layer])
    processor = _processor(model)
    state = processor._get_layer_state(0)
    state.layer_module = layer
    state.modules = {
        "0.linear": NamedModule(layer.linear, "linear", "0.linear", 0),
        "0.other": NamedModule(layer.other, "other", "0.other", 0),
    }
    processor._feature_task_names = set()
    processor._awq_feature_kwargs = {}
    processor.apply_clip = False
    features = {
        "0.linear": torch.randn(4, 16),
        "0.other": torch.randn(4, 16),
    }
    monkeypatch.setattr(processor, "_layer_input_features", lambda state: features)
    calls = []
    monkeypatch.setattr(
        processor,
        "apply_quant",
        lambda named, scales_list: calls.append((list(named), list(scales_list))),
    )
    return layer, processor, state, calls


def test_no_module_configuration_falls_back_to_plain_quantization(monkeypatch):
    _, processor, state, calls = _layer_and_processor(monkeypatch)
    monkeypatch.setattr(
        processor.gptq_model,
        "awq_get_modules_for_scaling",
        lambda *args, **kwargs: [],
        raising=False,
    )

    processor._quantize_layer(0, state)

    assert state.quantized is True
    assert len(calls) == 1, "the empty-group path must quantize, not just finalize"
    quantized_names, scales_list = calls[0]
    assert sorted(quantized_names) == ["0.linear", "0.other"]
    assert scales_list == []


def test_no_surviving_group_falls_back_to_plain_quantization(monkeypatch):
    layer, processor, state, calls = _layer_and_processor(monkeypatch)
    # A group whose prev_op output width does not match the consumer's input is
    # dropped by the sanitizer, leaving no valid group.
    bad_prev = nn.Linear(16, 32, bias=False)
    monkeypatch.setattr(
        processor.gptq_model,
        "awq_get_modules_for_scaling",
        lambda *args, **kwargs: [
            {"prev_op": bad_prev, "layers": [layer.linear], "inp": torch.randn(4, 16), "kwargs": {}}
        ],
        raising=False,
    )

    processor._quantize_layer(0, state)

    assert state.quantized is True
    assert len(calls) == 1
    assert calls[0][1] == []
