# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Defuser GLM-4.x MoE blocks must accept the Transformers 5.22 router tuple.

Transformers 5.22 routers return ``(router_logits, topk_weights,
topk_indices)``. Defuser 0.0.30's replacement block passed that tuple straight
into ``route_tokens_to_experts`` and crashed at the first MoE layer with
``AttributeError: 'tuple' object has no attribute 'sigmoid'``.
"""

import pytest
import torch

from gptqmodel.utils.defuser_compat import apply_glm4_moe_router_compat


def _tiny_config(model_type: str):
    transformers = pytest.importorskip("transformers")
    common = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 48,
        "moe_intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 8,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "n_group": 1,
        "topk_group": 1,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    if model_type == "glm4_moe":
        return transformers.AutoConfig.for_model(
            model_type,
            first_k_dense_replace=1,
            layer_types=["full_attention", "full_attention"],
            **common,
        )
    return transformers.AutoConfig.for_model(
        model_type,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        mlp_layer_types=["dense", "sparse"],
        **common,
    )


def _build_defused_model(model_type: str):
    defuser = pytest.importorskip("defuser")
    transformers = pytest.importorskip("transformers")

    config = _tiny_config(model_type)
    # The loader patches the HF MoE class before constructing the model.
    assert defuser.replace_fused_blocks(model_type)
    model = transformers.AutoModelForCausalLM.from_config(config).eval()
    return model, config


def _moe_block(model):
    for module in model.modules():
        if type(module).__name__ in ("LinearGlm4MoeMoE", "LinearGlm4MoeLiteMoE"):
            return module
    return None


@pytest.mark.parametrize("model_type", ["glm4_moe", "glm4_moe_lite"])
def test_glm4_moe_forward_accepts_router_tuple(model_type):
    model, _ = _build_defused_model(model_type)
    block = _moe_block(model)
    assert block is not None, f"{model_type}: Defuser block was not installed"

    apply_glm4_moe_router_compat()

    input_ids = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        logits = model(input_ids).logits
    assert logits.shape == (1, 8, 64)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("model_type", ["glm4_moe", "glm4_moe_lite"])
def test_router_tuple_and_tensor_paths_are_equivalent(model_type):
    """The shim must only unwrap; routing math must stay byte-identical."""

    model, config = _build_defused_model(model_type)
    block = _moe_block(model)
    assert block is not None, f"{model_type}: Defuser block was not installed"

    apply_glm4_moe_router_compat()

    hidden = torch.randn(1, 5, config.hidden_size)
    with torch.no_grad():
        tuple_output = block(hidden)

    native_router = block.gate
    native_forward = native_router.forward
    native_router.forward = lambda states: native_forward(states)[0]
    try:
        with torch.no_grad():
            tensor_output = block(hidden)
    finally:
        native_router.forward = native_forward

    torch.testing.assert_close(tuple_output, tensor_output, rtol=0, atol=0)


def test_router_compat_is_idempotent():
    _build_defused_model("glm4_moe")
    apply_glm4_moe_router_compat()
    assert apply_glm4_moe_router_compat() == ()
