# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""Expert-Balanced Self-Sampling: balance term and beam-search smoke coverage."""

from __future__ import annotations

import math

import pytest
import torch

from gptqmodel.models.ports.native.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from gptqmodel.models.ports.native.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
from gptqmodel.quantization.config import EBSSConfig
from gptqmodel.utils.ebss import (
    _expert_balance,
    _extract_router_indices,
    _router_key,
    _router_num_experts,
    generate_ebss_calibration,
)


pytestmark = [pytest.mark.cpu]


def test_expert_balance_is_zero_for_uniform_usage():
    usage = {"layers.0": torch.tensor([5, 5, 5, 5], dtype=torch.int64)}
    assert _expert_balance(usage, {"layers.0": 4}) == 0.0


def test_expert_balance_matches_sample_std_formula():
    usage = {"layers.0": torch.tensor([10, 0, 0, 0], dtype=torch.int64)}
    expected = math.sqrt(4.0 / 3.0 * ((1.0 - 0.25) ** 2 + 3 * (0.25**2)))
    assert _expert_balance(usage, {"layers.0": 4}) == pytest.approx(expected)


def test_expert_balance_averages_layers_and_skips_single_expert():
    usage = {
        "layers.0": torch.tensor([10, 0, 0, 0], dtype=torch.int64),
        "layers.1": torch.tensor([3], dtype=torch.int64),
    }
    expected = math.sqrt(4.0 / 3.0 * ((1.0 - 0.25) ** 2 + 3 * (0.25**2))) / 2.0
    assert _expert_balance(usage, {"layers.0": 4, "layers.1": 1}) == pytest.approx(expected)


def test_extract_router_indices_prefers_integer_tuple_member():
    class _Router(torch.nn.Module):
        top_k = 2

    logits = torch.randn(5, 4)
    indices = torch.tensor([[0, 1], [2, 3], [1, 1], [0, 0], [3, 2]])
    found = _extract_router_indices((logits, torch.softmax(logits, -1), indices), _Router())
    torch.testing.assert_close(found, indices)


def test_extract_router_indices_derives_topk_from_logits():
    class _Router(torch.nn.Module):
        top_k = 2

    logits = torch.tensor([[0.0, 3.0, 1.0, 2.0]])
    found = _extract_router_indices(logits, _Router())
    assert found.tolist() == [[1, 3]]


def test_router_key_keeps_multiple_routers_per_layer_independent():
    assert _router_key("model.layers.0.mlp.gate") == "model.layers.0.mlp"
    assert _router_key("model.layers.0.mlp2.gate") == "model.layers.0.mlp2"
    assert _router_key("router") == "router"


def test_router_num_experts_handles_v_experts_module_list():
    class _Router(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.top_k = 2
            self.v_experts = torch.nn.ModuleList(
                [torch.nn.Linear(2, 2) for _ in range(3)]
            )

    assert _router_num_experts(_Router(), None) == 3


def test_extract_router_indices_squeezes_singleton_expert_axis():
    class _Router(torch.nn.Module):
        top_k = 2

    indices = torch.tensor([[[0], [1]], [[2], [0]]])
    assert _extract_router_indices(indices, _Router()).tolist() == [[0, 1], [2, 0]]
    leading = indices.permute(0, 2, 1)
    assert _extract_router_indices(leading, _Router()).tolist() == [[0, 1], [2, 0]]
    with pytest.raises(ValueError, match="unsupported"):
        _extract_router_indices(torch.zeros(2, 2, 2, dtype=torch.long), _Router())


def _tiny_model() -> Qwen3MoeForCausalLM:
    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        head_dim=8,
    )
    return Qwen3MoeForCausalLM(config).eval()


def test_ebss_generates_deterministic_valid_sequences():
    model = _tiny_model()
    config = EBSSConfig(num_sentences=2, seq_len=4, width=2, seed=7)
    first = generate_ebss_calibration(model, config)
    second = generate_ebss_calibration(model, config)
    assert first == second
    assert len(first) == 2
    assert all(len(sentence) == 4 for sentence in first)
    assert all(0 <= token < 64 for sentence in first for token in sentence)
    assert model.training is False


def test_ebss_rejects_seq_len_beyond_model_context():
    model = _tiny_model()
    with pytest.raises(ValueError, match="max_position_embeddings"):
        generate_ebss_calibration(
            model, EBSSConfig(num_sentences=1, seq_len=128, width=1)
        )


def test_ebss_greedy_width_one_runs():
    model = _tiny_model()
    config = EBSSConfig(num_sentences=1, seq_len=3, width=1, seed=1)
    sentences = generate_ebss_calibration(model, config)
    assert len(sentences) == 1
    assert len(sentences[0]) == 3
