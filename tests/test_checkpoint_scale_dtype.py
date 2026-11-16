# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A checkpoint's scale dtype is part of the quantization contract."""

from __future__ import annotations

import torch
import torch.nn as nn

import gptqmodel.utils.model as model_utils
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = TorchLinear(
            bits=4,
            group_size=32,
            sym=True,
            desc_act=False,
            in_features=64,
            out_features=32,
            bias=False,
            pack_dtype=torch.int32,
            register_buffers=True,
            format=FORMAT.GPTQ_V2,
        )


def test_gptq_scales_are_restored_from_the_checkpoint(monkeypatch, tmp_path):
    model = _Model()
    captured = {}

    monkeypatch.setattr(
        model_utils.accelerate, "load_checkpoint_in_model", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        model_utils, "_tie_weights_after_checkpoint_load", lambda *args, **kwargs: None
    )

    def fake_restore(model, checkpoint, dtypes, **kwargs):
        captured.update(dtypes)

    monkeypatch.setattr(model_utils, "_restore_checkpoint_buffers", fake_restore)

    model_utils.load_checkpoint_in_model_then_tie_weights(
        model,
        checkpoint=str(tmp_path),
        dtype="torch.bfloat16",
    )

    # ``None`` means "use the source dtype" in _restore_checkpoint_buffers.
    assert captured.get("proj.scales") is None
