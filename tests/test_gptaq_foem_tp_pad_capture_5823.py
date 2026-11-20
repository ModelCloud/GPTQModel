# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GPTAQ/FOEM must pad the feature axis during Hessian capture (#5823).

With ``preprocessors=["tensor_parallel_padder"]`` the GPTQ base initializer
expands ``self.columns`` to the padded width, but GPTAQ/FOEM transpose the
capture to ``[features, tokens]`` and handed the accumulator the unpadded
feature count. The accumulator then tried to add a 96x96 product into a
128x128 Hessian. These tests pin the padded capture, its zero padding block
and an exactly representable 4-bit reference weight for GPTQ, GPTAQ and FOEM.
"""

import pytest
import torch

from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.looper.module_preprocessor import ModulePreProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.native_processor import NATIVE_INPUTS_STATE_KEY
from gptqmodel.quantization.config import (
    CalibrationConfig,
    FOEMConfig,
    GPTAQConfig,
    QuantizeConfig,
)


IN_FEATURES = 96
OUT_FEATURES = 32
GROUP_SIZE = 64
PAD_COLS = 32


@pytest.fixture(autouse=True)
def _disable_device_smi(monkeypatch):
    monkeypatch.setattr(ModulePreProcessor, "_init_device_smi_handles", lambda self: {})
    monkeypatch.setattr(ModulePreProcessor, "_init_cpu_device_handle", lambda self: None)


def _reference_weight() -> torch.Tensor:
    """A 32x96 weight that is exactly representable as asymmetric INT4.

    Every group spans all 16 codes on a 0.25 grid from -1.0 to 2.75, so the
    zero-padded TP columns (weight 0) cannot change a group's range and the
    reconstructed weight must be bit-identical to the source.
    """

    codes = torch.arange(16, dtype=torch.float32)
    row = torch.stack([codes] * OUT_FEATURES, dim=0)
    group0 = row.repeat(1, 4)
    group1 = row.repeat(1, 2)
    return torch.cat((group0, group1), dim=1) * 0.25 - 1.0


def _variant(kind: str) -> dict:
    if kind == "gptq":
        return {}
    if kind == "gptaq":
        return {"gptaq": GPTAQConfig(alpha=0.25)}
    return {"foem": FOEMConfig(alpha=0.25, beta=0.25)}


@pytest.mark.parametrize("kind", ["gptq", "gptaq", "foem"])
def test_tp_padded_capture_reconstructs_reference_weight(kind):
    torch.manual_seed(5823)
    weight = _reference_weight()
    calibration = torch.eye(IN_FEATURES)

    layer = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    with torch.no_grad():
        layer.weight.copy_(weight)
    named = NamedModule(layer, name="proj", full_name="model.layers.0.proj", layer_index=0)
    named.state[NATIVE_INPUTS_STATE_KEY] = [calibration * 1.125]
    qcfg = QuantizeConfig(
        bits=4,
        group_size=GROUP_SIZE,
        sym=False,
        desc_act=False,
        act_group_aware=False,
        preprocessors=["tensor_parallel_padder"],
        **_variant(kind),
    )
    ModulePreProcessor(qcfg=qcfg).preprocess(named)
    assert named.state["tp_pad_info"]["pad_cols"] == PAD_COLS

    processor = GPTQProcessor(tokenizer=None, qcfg=qcfg, calibration=CalibrationConfig())
    processor.preprocess(named)
    processor.pre_process_fwd_hook(named.name)(layer, (calibration,), None)

    task = processor.tasks[named.name]
    assert task.columns == IN_FEATURES + PAD_COLS
    assert task.nsamples == IN_FEATURES

    hessian = task.finalize_hessian() if kind == "gptq" else task.H
    assert hessian.shape == (task.columns, task.columns)
    assert torch.count_nonzero(hessian[IN_FEATURES:, :]) == 0
    assert torch.count_nonzero(hessian[:, IN_FEATURES:]) == 0

    if kind != "gptq":
        # The native delta is accumulated in the same padded space.
        assert task.dXXT is not None
        assert task.dXXT.shape == (task.columns, task.columns)
        assert torch.count_nonzero(task.dXXT[IN_FEATURES:, :]) == 0
        assert torch.count_nonzero(task.dXXT[:, IN_FEATURES:]) == 0

    quantized, _scale, _zero, g_idx, *_ = task.quantize(blocksize=GROUP_SIZE)

    assert quantized.shape == weight.shape
    assert g_idx.numel() == IN_FEATURES
    torch.testing.assert_close(quantized, weight, rtol=0.0, atol=0.0)
