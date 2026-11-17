# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ must not fold `attention.dense` into `input_layernorm` on Ling 2.0 (#4677).

`attention.dense` consumes the attention output, not the normed hidden state.
#4429 split the remote `attention` block into `query_key_value/q_a/kv_a/g` (0),
`q_b/kv_b` (1) and `dense` (2). A Ling 2.0 layer owns no MLA pair, so block 1
resolved to nothing; the `dense` block then fell back to `last_module`, which
still pointed at `input_layernorm`, and the two norm-sourced groups merged into
`input_layernorm -> [query_key_value, dense]`. Folding that group rescales
`dense`'s input columns with no compensation (13-14% logit change).

A block whose declared modules are all absent on this layer no longer feeds the
`last_module` fallback.
"""

import os


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch
from torch import nn

from gptqmodel.models.definitions.bailing_moe import BailingMoeQModel
from gptqmodel.quantization.awq.quantize.scale import apply_scale
from gptqmodel.quantization.awq.utils.module import get_op_name


pytestmark = [pytest.mark.cpu]

WIDTH = 8


class _RemoteAttention(nn.Module):
    """Ling 2.0 remote-code attention: `query_key_value` + `dense` only."""

    def __init__(self, *, with_mla: bool):
        super().__init__()
        self.query_key_value = nn.Linear(WIDTH, 3 * WIDTH, bias=False)
        self.dense = nn.Linear(WIDTH, WIDTH, bias=False)
        if with_mla:
            self.q_b_proj = nn.Linear(WIDTH, WIDTH, bias=False)
            self.kv_b_proj = nn.Linear(WIDTH, WIDTH, bias=False)

    def forward(self, hidden_states):
        qkv = self.query_key_value(hidden_states)
        # `dense` reads the attention output, not the normed hidden state.
        return self.dense(qkv[..., :WIDTH] * qkv[..., WIDTH : 2 * WIDTH])


class _RemoteLayer(nn.Module):
    def __init__(self, *, with_mla: bool):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(WIDTH)
        self.attention = _RemoteAttention(with_mla=with_mla)
        self.post_attention_layernorm = nn.LayerNorm(WIDTH)

    def forward(self, hidden_states):
        attn = self.attention(self.input_layernorm(hidden_states))
        return self.post_attention_layernorm(hidden_states + attn)


class _RemoteModel(nn.Module):
    def __init__(self, *, with_mla: bool):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_RemoteLayer(with_mla=with_mla)])
        self.model.norm = nn.LayerNorm(WIDTH)
        self.lm_head = nn.Linear(WIDTH, WIDTH, bias=False)
        self.config = type("Config", (), {"model_type": "bailing_moe"})()

    def forward(self, hidden_states):
        hidden_states = self.model.norm(self.model.layers[0](hidden_states))
        return self.lm_head(hidden_states)


class _LingDefinition(BailingMoeQModel):
    """The Ling 2.0 subset split, without building a real checkpoint."""

    def full_layer_modules(self, *args, **kwargs):
        return [
            ["input_layernorm:!"],
            ["attention.query_key_value", "attention.q_a_proj", "attention.kv_a_proj_with_mqa", "attention.g_proj"],
            ["attention.q_b_proj", "attention.kv_b_proj"],
            ["attention.dense"],
            ["post_attention_layernorm:!"],
        ]


def _scaling_groups(*, with_mla: bool):
    model = _RemoteModel(with_mla=with_mla)
    layer = model.model.layers[0]

    qmodel = _LingDefinition.__new__(_LingDefinition)
    nn.Module.__init__(qmodel)
    qmodel.__dict__["model"] = model
    qmodel.__dict__["quantize_config"] = type("QC", (), {"dynamic": None})()
    qmodel.__dict__["layer_modules_strict"] = False
    qmodel.__dict__["dynamic_expert_index"] = None
    qmodel.__dict__["awq_scale_optimize_shape_dependent_modules"] = None
    qmodel.__dict__["awq_post_norm_modules"] = lambda _module: []

    features = {
        "attention.query_key_value": torch.randn(4, WIDTH),
        "attention.dense": torch.randn(4, WIDTH),
    }
    return layer, qmodel.awq_get_modules_for_scaling(layer, features, {})


def _group_names(layer, node):
    return [get_op_name(layer, module) for module in node.get("layers") or []]


@pytest.mark.parametrize("with_mla", [False, True], ids=["ling-2.0", "ling-2.6-mla"])
def test_attention_dense_is_never_folded_into_the_input_norm(with_mla):
    layer, nodes = _scaling_groups(with_mla=with_mla)

    for node in nodes:
        if node.get("prev_op") is not layer.input_layernorm:
            continue
        assert "attention.dense" not in _group_names(layer, node)


def test_query_key_value_still_folds_into_the_input_norm():
    layer, nodes = _scaling_groups(with_mla=False)

    normed = [node for node in nodes if node.get("prev_op") is layer.input_layernorm]
    assert len(normed) == 1
    assert _group_names(layer, normed[0]) == ["attention.query_key_value"]


def test_folding_the_reported_groups_preserves_the_layer_output():
    layer, nodes = _scaling_groups(with_mla=False)

    generator = torch.Generator().manual_seed(4677)
    hidden = torch.randn(2, 3, WIDTH, generator=generator)
    layer.eval()
    with torch.no_grad():
        before = layer(hidden).clone()

    for node in nodes:
        prev_op = node.get("prev_op")
        consumers = node.get("layers") or []
        if prev_op is None or not consumers:
            continue
        scales = torch.rand(prev_op.weight.numel(), generator=generator) + 0.5
        group = (
            get_op_name(layer, prev_op),
            [get_op_name(layer, module) for module in consumers],
            scales,
            None,
        )
        with torch.no_grad():
            apply_scale(layer, [group])
            after = layer(hidden)
            apply_scale(
                layer,
                [(group[0], group[1], 1.0 / scales, None)],
            )

    torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-6)
