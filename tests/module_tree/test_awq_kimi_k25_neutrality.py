# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Kimi-K2.5 AWQ scale folds must leave the full multimodal model output unchanged."""

import pytest
import torch
from awq_neutrality import assert_awq_folds_neutral

from gptqmodel.models.definitions.kimi_k25 import KimiK25QModel
from gptqmodel.models.loader import _convert_model_with_defuser
from tests.test_kimi_k25_local_port import tiny_config


EXPERTS = 4


@pytest.mark.parametrize("q_lora_rank", [None, 8], ids=["dense_q", "split_q"])
def test_kimi_k25_awq_folds_preserve_dense_and_moe_layer_output(q_lora_rank):
    torch.manual_seed(5)
    model = KimiK25QModel.loader.from_config(tiny_config(q_lora_rank, first_k_dense_replace=1)).eval().double()
    _convert_model_with_defuser(KimiK25QModel, model, cleanup_original=True)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            # Non-unit gains and a non-zero routing bias so a wrong fold shows.
            if parameter.ndim == 1 and "norm" in name:
                parameter.copy_(torch.linspace(0.7, 1.3, parameter.numel()))
            if name.endswith("e_score_correction_bias"):
                parameter.copy_(torch.linspace(-0.05, 0.05, parameter.numel()))
    ids = torch.tensor([[1, 4, 6, 9, 12, 7, 5, 3]])

    def forward():
        return model(input_ids=ids, use_cache=False).logits

    dense, moe = model.model.language_model.layers
    attention = ["self_attn.q_proj" if q_lora_rank is None else "self_attn.q_a_proj", "self_attn.kv_a_proj_with_mqa"]

    checked = assert_awq_folds_neutral(KimiK25QModel, model, dense, forward)
    assert checked == [
        ("input_layernorm", attention, []),
        ("post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"], []),
        ("mlp.up_proj", ["mlp.down_proj"], []),
    ]

    checked = assert_awq_folds_neutral(KimiK25QModel, model, moe, forward)
    gate_up = [f"mlp.experts.{e}.{leaf}" for e in range(EXPERTS) for leaf in ("gate_proj", "up_proj")]
    gate_up += ["mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj"]
    assert checked == [
        ("input_layernorm", attention, []),
        # The router reads the same normed hidden state and is compensated.
        ("post_attention_layernorm", gate_up, ["mlp.gate"]),
        *[(f"mlp.experts.{e}.up_proj", [f"mlp.experts.{e}.down_proj"], []) for e in range(EXPERTS)],
        ("mlp.shared_experts.up_proj", ["mlp.shared_experts.down_proj"], []),
    ]
