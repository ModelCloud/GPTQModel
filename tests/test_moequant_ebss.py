# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""Expert-Balanced Self-Sampling: balance term and beam-search smoke coverage."""

from __future__ import annotations

from statistics import stdev
from types import SimpleNamespace

import pytest
import torch

import gptqmodel.utils.ebss as ebss_module
from gptqmodel.models.base import BaseQModel
from gptqmodel.models.ports.native.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from gptqmodel.models.ports.native.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
from gptqmodel.quantization.config import CalibrationConfig, EBSSConfig, MoEQuantConfig
from gptqmodel.utils.ebss import (
    _expert_balance,
    _extract_router_indices,
    _router_num_experts,
    _RoutingObserver,
    generate_ebss_calibration,
)


pytestmark = [pytest.mark.cpu]


def test_expert_balance_is_zero_for_uniform_usage():
    usage = {"layers.0": torch.tensor([5, 5, 5, 5], dtype=torch.int64)}
    assert _expert_balance(usage, {"layers.0": 4}) == 0.0


def test_expert_balance_matches_sample_std_formula():
    usage = {"layers.0": torch.tensor([10, 0, 0, 0], dtype=torch.int64)}
    expected = stdev([1.0, 0.0, 0.0, 0.0])
    assert _expert_balance(usage, {"layers.0": 4}) == pytest.approx(expected)


def test_expert_balance_averages_layers_and_skips_single_expert():
    usage = {
        "layers.0": torch.tensor([10, 0, 0, 0], dtype=torch.int64),
        "layers.1": torch.tensor([3], dtype=torch.int64),
    }
    expected = stdev([1.0, 0.0, 0.0, 0.0]) / 2.0
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


class _FixedRouter(torch.nn.Module):
    num_experts = 2
    top_k = 1

    def forward(self, input_ids):
        return (input_ids == 1).long()


class _FixedLogitMoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=3)
        self.embedding = torch.nn.Embedding(3, 1)
        self.router = _FixedRouter()

    def forward(self, input_ids, use_cache, past_key_values=None):
        assert use_cache
        # The uniform first token always uses expert 0. Tokens 0 and 1
        # thereafter have similar likelihoods but route to different experts.
        self.router(torch.zeros_like(input_ids) if past_key_values is None else input_ids)
        return SimpleNamespace(
            logits=torch.tensor([[[4.0, 3.95, -5.0]]]),
            past_key_values=SimpleNamespace(),
        )


@pytest.mark.parametrize(
    ("width", "seq_len", "temperature", "expected_tail"),
    [
        (1, 2, 1.2, [0]),
        (2, 2, 1.2, [1]),
        (2, 2, 100.0, [0]),
        (2, 3, 1.2, [1, 0]),
    ],
    ids=["prune-low-likelihood", "final-balance", "final-likelihood", "prune-imbalanced-prefix"],
)
def test_ebss_minimizes_likelihood_and_balance_cost(width, seq_len, temperature, expected_tail):
    model = _FixedLogitMoE()
    config = EBSSConfig(num_sentences=1, seq_len=seq_len, width=width, temperature=temperature)
    [sentence] = generate_ebss_calibration(model, config)
    assert sentence[1:] == expected_tail


@pytest.mark.parametrize("token_ids", [[3], [1, 3]])
def test_observer_counts_only_dispatched_qwen_experts(token_ids):
    model = _tiny_model()
    tokens = torch.tensor([token_ids])
    with torch.inference_mode(), _RoutingObserver(model) as observer:
        output = model(input_ids=tokens, output_router_logits=True)
        indices = torch.softmax(output.router_logits[0], -1).topk(2, dim=-1).indices
        expected = torch.bincount(indices.flatten(), minlength=4)
        usage = observer.end_forward()
    assert len(usage) == 1
    torch.testing.assert_close(next(iter(usage.values())), expected)


def test_sibling_routers_have_independent_balance_terms():
    block = torch.nn.ModuleDict({"first": _FixedRouter(), "second": _FixedRouter()})
    model = torch.nn.ModuleDict({"block": block})
    with _RoutingObserver(model) as observer:
        block.first(torch.tensor([[0]]))
        block.second(torch.tensor([[1]]))
        usage = observer.end_forward()
        assert _expert_balance(usage, observer.num_experts) == pytest.approx(stdev([0.0, 1.0]))


@pytest.mark.parametrize("interrupt", [False, True])
def test_ebss_restores_mixed_training_modes(interrupt):
    model = _FixedLogitMoE().train()
    model.router.eval()
    original = [module.training for module in model.modules()]
    config = EBSSConfig(num_sentences=1, seq_len=2, width=1)

    def stop(*_):
        raise RuntimeError("stop sampling")

    if interrupt:
        with pytest.raises(RuntimeError, match="stop sampling"):
            generate_ebss_calibration(model, config, progress=stop)
    else:
        generate_ebss_calibration(model, config)
    assert [module.training for module in model.modules()] == original


def test_ebss_rejects_logits_without_an_expert_axis():
    class FeatureProjection(torch.nn.Module):
        num_experts = 2
        top_k = 1

        def forward(self, inputs):
            return torch.ones(inputs.numel(), 8)

    model = _FixedLogitMoE()
    model.router = FeatureProjection()
    with pytest.raises(ValueError, match="expert axis"):
        generate_ebss_calibration(model, EBSSConfig(num_sentences=1, seq_len=2))


def test_ebss_requires_routes_during_sampling():
    model = _FixedLogitMoE()
    model.unused_router = model.router
    model.router = torch.nn.Identity()
    with pytest.raises(ValueError, match="no routing assignments"):
        generate_ebss_calibration(model, EBSSConfig(num_sentences=1, seq_len=2))


def test_maybe_generate_ebss_passes_quantize_device_and_layer_prefixes(monkeypatch):
    model = _tiny_model()
    wrapper = BaseQModel.__new__(BaseQModel)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = model
    wrapper.turtle_model = None
    wrapper.quantize_config = SimpleNamespace(
        moequant=MoEQuantConfig(
            agq=False,
            ebss=EBSSConfig(num_sentences=1, seq_len=2),
        ),
        device="cpu",
    )
    wrapper._pre_quantize_device = lambda: torch.device("cpu")
    wrapper.extract_layers_node = lambda: ["model.layers"]

    calls = {}

    def fake_generate(inner_model, config, **kwargs):
        calls["model"] = inner_model
        calls["config"] = config
        calls.update(kwargs)
        return [[1, 2]]

    monkeypatch.setattr(ebss_module, "generate_ebss_calibration", fake_generate)

    calibration = wrapper._maybe_generate_ebss_calibration(CalibrationConfig())

    assert calls["model"] is model
    assert calls["device"] == torch.device("cpu")
    assert calls["layer_prefixes"] == ["model.layers"]
    assert calibration.dataset == [[1, 2]]
