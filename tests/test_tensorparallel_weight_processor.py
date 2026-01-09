# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import math

import torch

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.tensorparallel_weight_processor import TensorParallelWeightProcessor
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


def _noop_prepare_dataset(*, calibration_dataset, **_):
    return calibration_dataset


def test_tensorparallel_pre_padding_applies_zero_pad_metadata():
    linear = torch.nn.Linear(10, 7, bias=False)
    named = NamedModule(linear, name="proj", full_name="layer.0.proj", layer_index=0)

    qcfg = QuantizeConfig(bits=4, mock_quantization=True, process={"gptq": {"act_group_aware": False}})
    qcfg.group_size = -1
    qcfg.desc_act = False

    calibration_stub = [{"input_ids": torch.ones((1, 1), dtype=torch.long)}]

    preprocessor = TensorParallelWeightProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=calibration_stub,
        prepare_dataset_func=_noop_prepare_dataset,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
    )

    preprocessor.process(named)

    pad_info = named.state["tp_pad_info"]
    assert pad_info["pad_cols"] == 6
    assert pad_info["original_columns"] == 10

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


def test_tensorparallel_weight_processor_with_positive_group_size():
    """Test that _target_multiple is correctly calculated when group_size > 0."""
    linear = torch.nn.Linear(10, 7, bias=False)
    NamedModule(linear, name="proj", full_name="layer.0.proj", layer_index=0)

    qcfg = QuantizeConfig(bits=4, mock_quantization=True, process={"gptq": {"act_group_aware": False}})
    qcfg.group_size = 128  # Positive group_size
    qcfg.desc_act = False

    calibration_stub = [{"input_ids": torch.ones((1, 1), dtype=torch.long)}]

    preprocessor = TensorParallelWeightProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=calibration_stub,
        prepare_dataset_func=_noop_prepare_dataset,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
    )

    # Verify that _target_multiple includes group_size in LCM calculation
    # Default TP_TARGETS = (2, 4, 8), so math.lcm(2, 4, 8) = 8
    # With group_size = 128, math.lcm(8, 128) = 128
    expected_target_multiple = math.lcm(8, 128)
    assert preprocessor._target_multiple == expected_target_multiple
    assert preprocessor._target_multiple == 128


def test_tensorparallel_weight_processor_with_negative_group_size():
    """Test that _target_multiple uses default value when group_size < 0."""
    linear = torch.nn.Linear(10, 7, bias=False)
    NamedModule(linear, name="proj", full_name="layer.0.proj", layer_index=0)

    qcfg = QuantizeConfig(bits=4, mock_quantization=True, process={"gptq": {"act_group_aware": False}})
    qcfg.group_size = -1  # Negative group_size
    qcfg.desc_act = False

    calibration_stub = [{"input_ids": torch.ones((1, 1), dtype=torch.long)}]

    preprocessor = TensorParallelWeightProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=calibration_stub,
        prepare_dataset_func=_noop_prepare_dataset,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
    )

    # Verify that _target_multiple only uses TP_TARGETS when group_size < 0
    # Default TP_TARGETS = (2, 4, 8), so math.lcm(2, 4, 8) = 8
    expected_target_multiple = math.lcm(2, 4, 8)
    assert preprocessor._target_multiple == expected_target_multiple
    assert preprocessor._target_multiple == 8


def test_tensorparallel_weight_processor_group_size_lcm_calculation():
    """Test LCM calculation with various group_size values."""
    linear = torch.nn.Linear(10, 7, bias=False)
    NamedModule(linear, name="proj", full_name="layer.0.proj", layer_index=0)

    calibration_stub = [{"input_ids": torch.ones((1, 1), dtype=torch.long)}]

    # Test with group_size = 32
    qcfg = QuantizeConfig(bits=4, mock_quantization=True, process={"gptq": {"act_group_aware": False}})
    qcfg.group_size = 32
    qcfg.desc_act = False

    preprocessor = TensorParallelWeightProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=calibration_stub,
        prepare_dataset_func=_noop_prepare_dataset,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
    )

    # math.lcm(8, 32) = 32
    assert preprocessor._target_multiple == 32

    # Test with group_size = 64
    qcfg.group_size = 64
    preprocessor = TensorParallelWeightProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=calibration_stub,
        prepare_dataset_func=_noop_prepare_dataset,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
    )

    # math.lcm(8, 64) = 64
    assert preprocessor._target_multiple == 64

    # Test with group_size = 12 (not a power of 2)
    qcfg.group_size = 12
    preprocessor = TensorParallelWeightProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=calibration_stub,
        prepare_dataset_func=_noop_prepare_dataset,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
    )

    # math.lcm(8, 12) = 24
    expected_lcm = math.lcm(8, 12)
    assert preprocessor._target_multiple == expected_lcm
    assert preprocessor._target_multiple == 24
