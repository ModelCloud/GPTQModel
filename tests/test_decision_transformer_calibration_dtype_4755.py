# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""Tensor calibration rows must arrive in the model's dtype.

#4755 hands the user's own tensors to ``DecisionTransformerModel.forward``. A
model loaded at the default bfloat16 fed ``torch.randn`` rows (float32, the
natural spelling) failed inside the first matmul with

    RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16

which is a dtype error reported by torch's kernel, not by the loader. These
tests pin the normalization that turns it into a working run, and confirm the
token path is untouched.
"""

import types

import pytest
import torch
from transformers import DecisionTransformerConfig, DecisionTransformerModel

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel, QuantizeConfig
from gptqmodel.models.base import BaseQModel


_STATE_DIM = 8
_ACT_DIM = 3


def _config():
    return DecisionTransformerConfig(
        state_dim=_STATE_DIM, act_dim=_ACT_DIM, hidden_size=64, n_layer=2, n_head=2, max_ep_len=8,
    )


def _samples(count=4, dtype=torch.float32):
    rows = []
    for seed in range(count):
        generator = torch.Generator().manual_seed(seed)
        rows.append({
            "states": torch.randn(1, 8, _STATE_DIM, generator=generator).to(dtype),
            "actions": torch.randn(1, 8, _ACT_DIM, generator=generator).to(dtype),
            "rewards": torch.randn(1, 8, 1, generator=generator).to(dtype),
            "returns_to_go": torch.randn(1, 8, 1, generator=generator).to(dtype),
            "timesteps": torch.randint(0, 8, (1, 8), generator=generator),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
        })
    return rows


class _Stub:
    """Object good enough for the input-capture helpers."""

    def __init__(self, dtype, *, tensor_kwargs=True):
        parameters = [torch.zeros(1, dtype=torch.int64), torch.zeros(1, dtype=dtype)]
        self.model = types.SimpleNamespace(parameters=lambda: iter(parameters))
        self.ATTENTION_MASKS_DTYPE = None
        if tensor_kwargs:
            self.calibration_inputs = "tensor_kwargs"

    def finalize_input_capture_example(self, example):
        return example


def test_floating_rows_are_cast_to_the_model_dtype():
    stub = _Stub(torch.bfloat16)
    example = {"states": torch.randn(1, 8, _STATE_DIM), "attention_mask": torch.ones(1, 8, dtype=torch.long)}

    out = BaseQModel.move_input_capture_example(stub, example, torch.device("cpu"))

    assert out["states"].dtype == torch.bfloat16
    # Integer rows keep their dtype: timesteps and masks must not become floats.
    assert out["attention_mask"].dtype == torch.long


def test_an_fp32_model_leaves_rows_alone():
    stub = _Stub(torch.float32)
    example = {"states": torch.randn(1, 8, _STATE_DIM)}
    out = BaseQModel.move_input_capture_example(stub, example, torch.device("cpu"))
    assert out["states"].dtype == torch.float32


def test_the_token_path_is_not_cast():
    """A definition without the tensor contract keeps whatever it was given."""

    stub = _Stub(torch.bfloat16, tensor_kwargs=False)
    example = {"pixel_values": torch.randn(1, 3, 4, 4)}
    out = BaseQModel.move_input_capture_example(stub, example, torch.device("cpu"))
    assert out["pixel_values"].dtype == torch.float32


def test_list_rows_are_cast_elementwise():
    stub = _Stub(torch.bfloat16)
    example = {"values": [torch.randn(2), torch.ones(2, dtype=torch.long)]}
    out = BaseQModel.move_input_capture_example(stub, example, torch.device("cpu"))
    assert out["values"][0].dtype == torch.bfloat16
    assert out["values"][1].dtype == torch.long


def _tiny_checkpoint(tmp_path):
    source = tmp_path / "dt"
    torch.manual_seed(0)
    DecisionTransformerModel(_config()).eval().save_pretrained(source)
    return source


def test_bf16_model_quantizes_with_fp32_calibration_rows(tmp_path):
    """The reported failure: default bf16 load, float32 rows."""

    source = _tiny_checkpoint(tmp_path)
    model = GPTQModel.load(str(source), quantize_config=QuantizeConfig(bits=4, group_size=32, desc_act=False,
                                                                 device="cpu"), dtype=torch.bfloat16)
    model.quantize(CalibrationConfig(dataset=_samples(dtype=torch.float32), batch_size=1, min_length=1),
                   backend=BACKEND.TORCH)

    batch = _samples(count=1, dtype=torch.bfloat16)[0]
    with torch.no_grad():
        out = model.model(**batch).action_preds

    assert out.shape == (1, 8, _ACT_DIM)
    assert bool(torch.isfinite(out).all())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cast_is_a_no_op_when_the_dtypes_agree(tmp_path, dtype):
    stub = _Stub(dtype)
    example = {"states": torch.randn(1, 8, _STATE_DIM, dtype=dtype)}
    out = BaseQModel.move_input_capture_example(stub, example, torch.device("cpu"))
    assert out["states"].dtype == dtype
