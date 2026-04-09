# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch
import torch.nn.functional as F

from gptqmodel.looper.module_preprocessor import ModulePreProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.quantization.config import QuantizeConfig, TensorParallelPadderConfig
from gptqmodel.quantization.gptq import GPTQ


@pytest.fixture(autouse=True)
def _disable_device_smi(monkeypatch):
    monkeypatch.setattr(ModulePreProcessor, "_init_device_smi_handles", lambda self: {})
    monkeypatch.setattr(ModulePreProcessor, "_init_cpu_device_handle", lambda self: None)


def _build_preprocessor(qcfg: QuantizeConfig) -> ModulePreProcessor:
    return ModulePreProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=[],
        prepare_dataset_func=None,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
    )


def test_tensor_parallel_padder_applies_zero_pad_metadata():
    linear = torch.nn.Linear(10, 7, bias=False)
    named = NamedModule(linear, name="proj", full_name="layer.0.proj", layer_index=0)

    qcfg = QuantizeConfig(
        bits=4,
        mock_quantization=True,
        preprocessors=[TensorParallelPadderConfig()],
    )
    qcfg.group_size = -1
    qcfg.desc_act = False
    qcfg.act_group_aware = False

    _build_preprocessor(qcfg).preprocess(named)

    pad_info = named.state["tp_pad_info"]
    assert pad_info["pad_cols"] == 6
    assert pad_info["original_columns"] == 10
    assert named.state["preprocessor_pipeline"][0]["code"] == "tensor_parallel_padder"

    gptq = GPTQ(named, qcfg)
    gptq.quantizer.configure(perchannel=True)

    assert gptq._tp_pad_cols == 6
    assert gptq.columns == 16

    inputs = torch.randn(32, 10)
    outputs = linear(inputs)
    gptq.add_batch(inputs, outputs)

    Q, scales, zeros, g_idx, *_ = gptq.quantize(blocksize=16)

    assert Q.shape == linear.weight.shape
    assert scales.shape[-1] <= linear.weight.shape[1]
    assert zeros.shape[-1] <= linear.weight.shape[1]
    assert g_idx.numel() == linear.weight.shape[1]

    gptq.free()
    assert "tp_pad_info" not in named.state


@pytest.mark.parametrize(
    ("group_size", "expected_target_multiple"),
    [
        (-1, 8),
        (32, 32),
        (64, 64),
        (12, 24),
    ],
)
def test_tensor_parallel_padder_uses_group_size_lcm(group_size: int, expected_target_multiple: int):
    linear = torch.nn.Linear(10, 7, bias=False)
    named = NamedModule(linear, name="proj", full_name="layer.0.proj", layer_index=0)

    qcfg = QuantizeConfig(
        bits=4,
        mock_quantization=True,
        preprocessors=[TensorParallelPadderConfig()],
    )
    qcfg.group_size = group_size
    qcfg.desc_act = False
    qcfg.act_group_aware = False

    _build_preprocessor(qcfg).preprocess(named)

    assert named.state["tp_pad_info"]["target_multiple"] == expected_target_multiple


def test_tensor_parallel_padder_does_not_change_quantized_matmul_output():
    torch.manual_seed(17)

    linear = torch.nn.Linear(10, 7, bias=False, dtype=torch.float32).eval()
    calibration_inputs = torch.randn(64, 10, dtype=torch.float32)
    calibration_outputs = linear(calibration_inputs)
    eval_inputs = torch.randn(19, 10, dtype=torch.float32)

    baseline_named = NamedModule(
        copy.deepcopy(linear),
        name="proj",
        full_name="layer.0.proj",
        layer_index=0,
    )
    baseline_qcfg = QuantizeConfig(bits=4, group_size=12)
    baseline_qcfg.desc_act = False
    baseline_qcfg.act_group_aware = False

    baseline_gptq = GPTQ(baseline_named, baseline_qcfg)
    baseline_gptq.quantizer.configure(perchannel=True)
    baseline_gptq.add_batch(calibration_inputs, calibration_outputs)
    baseline_weight, *_ = baseline_gptq.quantize(blocksize=16)
    baseline_gptq.free()

    padded_named = NamedModule(
        copy.deepcopy(linear),
        name="proj",
        full_name="layer.0.proj",
        layer_index=0,
    )
    padded_qcfg = QuantizeConfig(
        bits=4,
        group_size=12,
        preprocessors=[TensorParallelPadderConfig()],
    )
    padded_qcfg.desc_act = False
    padded_qcfg.act_group_aware = False

    _build_preprocessor(padded_qcfg).preprocess(padded_named)

    padded_gptq = GPTQ(padded_named, padded_qcfg)
    padded_gptq.quantizer.configure(perchannel=True)
    padded_gptq.add_batch(calibration_inputs, calibration_outputs)
    padded_weight, *_ = padded_gptq.quantize(blocksize=16)
    padded_gptq.free()

    baseline_output = F.linear(eval_inputs, baseline_weight)
    padded_output = F.linear(eval_inputs, padded_weight)

    torch.testing.assert_close(padded_output, baseline_output, rtol=0.0, atol=0.0)
