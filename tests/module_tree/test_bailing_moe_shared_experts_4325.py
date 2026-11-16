# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""Issue #4325: BailingMoe shared-expert `down_proj` is its own stage.

`down_proj` consumes the gate/up product, so keeping it in the gate/up subset
calibrated it on unquantized gate/up activations (GPTQ) and, when the hidden and
MoE intermediate widths coincide, let AWQ fold the post-attention norm into it.
The module tree now declares `("gate_proj:0", "up_proj:0", "down_proj:1")`.

The neutral-fold tests reuse the #4175 methodology: build a real tiny
BailingHybrid layer, apply the emitted AWQ folds without quantization, and
require the full layer output to stay unchanged. Before the fix the equal-width
case folded `post_attention_layernorm` into shared `down_proj` and changed the
layer output by ~1.7e-4.
"""

import pytest
import torch
import transformers

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.bailing_moe import BailingMoeQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from tests.module_tree.test_awq_fold_safety_4175 import _assert_neutral


TOLERANCE = 1e-5


def _layer_modules():
    return BailingMoeQModel.simple_layer_modules(
        model_config=type("Cfg", (), {"num_experts": 2, "num_local_experts": 2, "num_shared_experts": 1})(),
        quantize_config=type("QC", (), {"dynamic": None})(),
    )


def _plain_blocks():
    return [
        [entry.split(":", 1)[0] for entry in block]
        for block in _layer_modules()
    ]


def test_shared_expert_down_proj_is_its_own_subset():
    blocks = _plain_blocks()

    assert ["mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj"] in blocks
    assert ["mlp.shared_experts.down_proj"] in blocks
    # Regression guard for the rest of the declaration.
    assert [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.g_proj",
        "self_attn.kv_a_proj_with_mqa",
        "self_attn.forget_gate.f_proj",
        "self_attn.b_proj",
    ] in blocks
    assert ["self_attn.o_proj", "self_attn.kv_b_proj"] in blocks
    assert ["attention.query_key_value"] in blocks
    assert ["attention.dense"] in blocks
    assert ["mlp.gate_proj", "mlp.up_proj"] in blocks
    assert ["mlp.down_proj"] in blocks
    assert [
        "mlp.experts.0.gate_proj",
        "mlp.experts.1.gate_proj",
        "mlp.experts.0.up_proj",
        "mlp.experts.1.up_proj",
    ] in blocks
    assert ["mlp.experts.0.down_proj", "mlp.experts.1.down_proj"] in blocks


def _tiny_bailing_hybrid(*, moe_intermediate_size):
    config = transformers.BailingHybridConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=moe_intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        num_local_experts=2,
        num_experts_per_tok=1,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=0,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        layer_types=["full_attention"],
        mlp_layer_types=["sparse"],
    )
    config._attn_implementation = "eager"
    torch.manual_seed(7)
    model = transformers.BailingHybridForCausalLM(config).eval()
    _convert_model_with_defuser(MODEL_MAP["bailing_hybrid"], model, cleanup_original=False)
    return model


@pytest.mark.parametrize("moe_intermediate_size", [16, 32])
def test_bailing_moe_shared_expert_folds_are_neutral(moe_intermediate_size):
    """Apply every emitted fold to a real tiny layer; the output must not move.

    ``moe_intermediate_size == hidden_size`` is the case where the pre-fix tree
    folded the post-attention norm into shared `down_proj` and changed the
    layer output by ~1.7e-4. With unequal widths the pre-fix tree simply left
    `down_proj` out of every AWQ group; the fix emits its own gate/up -> down
    fold in both cases.
    """

    model = _tiny_bailing_hybrid(moe_intermediate_size=moe_intermediate_size)
    definition = MODEL_MAP["bailing_hybrid"]
    layer = model.model.layers[0]

    hidden = torch.randn(1, 8, 32)
    position_ids = torch.arange(8).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(hidden, position_ids)

    def forward():
        return layer(
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=None,
            position_ids=position_ids,
        )

    checked = _assert_neutral(definition, model, layer, forward)
    flat = {(source, tuple(targets)) for source, targets, _consumers in checked}

    # `down_proj` now has its own fold from the gate/up stage.
    assert (
        "mlp.shared_experts.up_proj",
        ("mlp.shared_experts.down_proj",),
    ) in flat
    # The post-attention norm still folds into the shared gate/up pair, but
    # never into `down_proj`.
    norm_folds = [
        targets
        for source, targets, _consumers in checked
        if source == "post_attention_layernorm"
    ]
    assert norm_folds, "post_attention_layernorm did not fold"
    for targets in norm_folds:
        assert "mlp.shared_experts.down_proj" not in targets
    assert any(
        "mlp.shared_experts.gate_proj" in targets
        and "mlp.shared_experts.up_proj" in targets
        for targets in norm_folds
    )
    # Dense/attention declarations in the same file still fold neutrally.
    assert ("input_layernorm", ("self_attn.q_proj", "self_attn.g_proj", "self_attn.kv_a_proj_with_mqa")) in flat
    assert (
        "mlp.experts.0.up_proj",
        ("mlp.experts.0.down_proj",),
    ) in flat
