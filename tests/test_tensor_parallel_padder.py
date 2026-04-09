# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

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
