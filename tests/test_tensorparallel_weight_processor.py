# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

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

    qcfg = QuantizeConfig(bits=4, mock_quantization=True)
    qcfg.group_size = -1
    qcfg.desc_act = False
    qcfg.act_group_aware = False

    calibration_stub = [{"input_ids": torch.ones((1, 1), dtype=torch.long)}]

    preprocessor = TensorParallelWeightProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=calibration_stub,
        prepare_dataset_func=_noop_prepare_dataset,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
        logger_board="",
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
