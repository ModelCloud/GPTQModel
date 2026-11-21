# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Storage parameters can opt out of the model-dtype cast at load time."""

from __future__ import annotations

import torch
import torch.nn as nn
from safetensors.torch import save_file

import gptqmodel.utils.model as model_utils


class _Packed(nn.Module):
    # Packed payload and its decoding factors are kernel storage, not
    # activation operands: they must keep the checkpoint dtype.
    CHECKPOINT_EXACT_DTYPE_PARAMETERS = ("codes", "scales", "amax")

    def __init__(self):
        super().__init__()
        self.codes = nn.Parameter(torch.zeros(4, dtype=torch.uint8), requires_grad=False)
        self.scales = nn.Parameter(
            torch.zeros(4, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        self.amax = nn.Parameter(torch.zeros(4, dtype=torch.float32), requires_grad=False)
        self.dense = nn.Parameter(torch.zeros(4, dtype=torch.float32), requires_grad=False)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _Packed()

    def tie_weights(self):
        pass


def _write_checkpoint(path):
    save_file(
        {
            "experts.codes": torch.tensor([1, 2, 3, 4], dtype=torch.uint8),
            "experts.scales": torch.tensor([1.0, 2.0, 3.0, 4.0]).to(torch.float8_e4m3fn),
            "experts.amax": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
            "experts.dense": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
        },
        str(path),
    )


def test_declared_storage_parameters_keep_the_checkpoint_dtype(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    _write_checkpoint(checkpoint)
    model = _Model()

    model_utils.load_checkpoint_in_model_then_tie_weights(
        model,
        checkpoint=str(checkpoint),
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
    )

    assert model.experts.codes.dtype == torch.uint8
    assert model.experts.scales.dtype == torch.float8_e4m3fn
    assert model.experts.amax.dtype == torch.float32
    # An undeclared parameter still follows the requested model dtype.
    assert model.experts.dense.dtype == torch.bfloat16
    assert model.experts.scales.float().tolist() == [1.0, 2.0, 3.0, 4.0]
