# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A reload must keep a packed projection's FP32 checkpoint bias (#5650).

``build_blockfloat_model`` installed each block-float projection at the model's
weight dtype, so a checkpoint that stores a float32 bias next to bfloat16
weights had that bias narrowed by ``load_state_dict``: ``-257`` became ``-256``
and a cancelling projection returned ``0`` instead of ``-1``. The bias is dense
state, so its buffer has to follow the checkpoint's dtype; the activation dtype
must stay the model's own, and the bias is added once before the output rounds.
"""

from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel.nn_modules.qlinear.tt_blockfloat import TTBlockfloatLinear
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.tt_blockfloat_quant import (
    build_blockfloat_model,
    load_hf_state_dict,
    save_blockfloat_model,
)


PROJ = "model.layers.0.self_attn.q_proj"
BIAS = -257.0
INPUT = 256.0
# 256 * 1 - 257, exactly representable in bfloat16; the buggy reload returned 0
# because the bias had already rounded to -256 before the subtraction.
EXPECTED = -1.0


def _packed_attention():
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        attention_bias=True,
        dtype=torch.bfloat16,
    )
    model = LlamaForCausalLM._from_config(config).eval()
    source = torch.nn.Linear(16, 16)
    with torch.no_grad():
        source.weight.zero_()
        source.weight[:, 0] = 1.0
        source.bias.fill_(BIAS)
    packed = TTBlockfloatLinear(
        bits=4,
        in_features=16,
        out_features=16,
        bias=True,
        backend=BACKEND.TT_BLOCKFLOAT,
        dtype=torch.bfloat16,
    ).pack(source)
    model.model.layers[0].self_attn.q_proj = packed
    x = torch.zeros(1, 16, dtype=torch.bfloat16)
    x[0, 0] = INPUT
    return model, x


def _assert_bias_and_projection(model, x):
    proj = model.get_submodule(PROJ)
    assert proj.bias is not None
    # -257 is not representable in bfloat16 (spacing 2 at that magnitude), so
    # any narrowing shows up both in the dtype and in the value.
    assert proj.bias.dtype is torch.float32
    assert proj.bias[0].item() == BIAS
    out = proj(x)
    assert out.dtype is torch.bfloat16
    assert out[0, 0].item() == EXPECTED


def _assert_saved_bias(directory):
    saved = load_hf_state_dict(str(directory))[f"{PROJ}.bias"]
    assert saved.dtype is torch.float32
    assert saved[0].item() == BIAS


def test_save_load_resave_keeps_fp32_bias_and_projection(tmp_path):
    model, x = _packed_attention()
    # Control: the direct pack already preserves the source dtype (the case
    # #3821 fixed), so only the reload is under test.
    _assert_bias_and_projection(model, x)

    save_blockfloat_model(model, str(tmp_path))
    _assert_saved_bias(tmp_path)

    restored, _restored_layers, _info = build_blockfloat_model(str(tmp_path))
    # Only the bias buffer follows the checkpoint. The model itself must stay
    # bfloat16, or every activation through it would be upcast.
    assert next(restored.parameters()).dtype is torch.bfloat16
    _assert_bias_and_projection(restored, x)

    resaved = tmp_path / "resaved"
    save_blockfloat_model(restored, str(resaved))
    _assert_saved_bias(resaved)
    again, _again_layers, _again_info = build_blockfloat_model(str(resaved))
    _assert_bias_and_projection(again, x)
