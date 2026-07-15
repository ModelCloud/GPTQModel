# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from torch import nn

from gptqmodel.models import auto
from gptqmodel.models.definitions.nemotron_h_puzzle import NemotronHPuzzleQModel


def test_nemotron_h_puzzle_model_type_selects_definition(monkeypatch):
    fake_config = SimpleNamespace(model_type="nemotron_h_puzzle")

    monkeypatch.setattr(auto, "resolve_trust_remote_code", lambda path, trust_remote_code=False: trust_remote_code)
    monkeypatch.setattr(auto.AutoConfig, "from_pretrained", lambda *args, **kwargs: fake_config)

    assert auto.check_and_get_model_definition("/tmp/nemotron_h_puzzle") is NemotronHPuzzleQModel


def test_nemotron_h_puzzle_module_tree_covers_hybrid_mixers_and_moe():
    layer_modules = NemotronHPuzzleQModel.simple_layer_modules(
        model_config=SimpleNamespace(n_routed_experts=3),
        quantize_config=SimpleNamespace(dynamic=None),
    )
    flat_modules = {name for block in layer_modules for name in block}

    assert NemotronHPuzzleQModel.require_trust_remote_code is True
    assert NemotronHPuzzleQModel.layer_modules_strict is False
    assert NemotronHPuzzleQModel.pre_lm_head_norm_module == "model.norm_f"
    assert NemotronHPuzzleQModel.extract_layers_node() == ["model.layers"]

    assert "mixer.q_proj" in flat_modules
    assert "mixer.k_proj" in flat_modules
    assert "mixer.v_proj" in flat_modules
    assert "mixer.o_proj" in flat_modules
    assert "mixer.in_proj" in flat_modules
    assert "mixer.out_proj" in flat_modules
    assert "mixer.fc1_latent_proj" in flat_modules
    assert "mixer.fc2_latent_proj" in flat_modules
    assert "mixer.experts.0.up_proj" in flat_modules
    assert "mixer.experts.1.down_proj" in flat_modules
    assert "mixer.experts.2.up_proj" in flat_modules
    assert "mixer.shared_experts.up_proj" in flat_modules
    assert "mixer.shared_experts.down_proj" in flat_modules
    assert "mixer.gate" not in flat_modules


def test_nemotron_h_puzzle_module_tree_matches_checkpoint_layout():
    assert NemotronHPuzzleQModel.module_tree[:3] == ["model", "layers", "#"]
    mixer_tree = NemotronHPuzzleQModel.module_tree[-1]["mixer:moe"]

    assert "experts" in mixer_tree
    assert "#" in mixer_tree["experts"]
    assert "shared_experts" in mixer_tree


def test_nemotron_h_puzzle_moe_projection_order_matches_forward():
    layer_modules = NemotronHPuzzleQModel.simple_layer_modules(
        model_config=SimpleNamespace(n_routed_experts=2),
        quantize_config=SimpleNamespace(dynamic=None),
    )

    fc1_index = next(i for i, block in enumerate(layer_modules) if "mixer.fc1_latent_proj" in block)
    expert_index = next(i for i, block in enumerate(layer_modules) if "mixer.experts.0.up_proj" in block)
    fc2_index = next(i for i, block in enumerate(layer_modules) if "mixer.fc2_latent_proj" in block)
    shared_index = next(i for i, block in enumerate(layer_modules) if "mixer.shared_experts.up_proj" in block)

    assert fc1_index < expert_index < fc2_index < shared_index


def test_nemotron_h_puzzle_replay_drops_unsupported_use_cache():
    model_def = object.__new__(NemotronHPuzzleQModel)
    attention_mask = torch.ones((1, 4), dtype=torch.long)
    cache_position = torch.arange(4)
    replay_kwargs = model_def.prepare_layer_replay_kwargs(
        layer=SimpleNamespace(),
        layer_input=[torch.zeros((1, 4, 8))],
        additional_inputs={
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "output_attentions": False,
            "use_cache": False,
        },
        target_device=torch.device("cpu"),
    )

    assert "use_cache" not in replay_kwargs
    assert replay_kwargs["attention_mask"] is attention_mask
    assert replay_kwargs["cache_position"] is cache_position
    assert replay_kwargs["output_attentions"] is False


def test_nemotron_h_puzzle_replay_builds_attention_causal_mask():
    model_def = object.__new__(NemotronHPuzzleQModel)
    mixer_config = SimpleNamespace(_attn_implementation="flash_attention_2")
    layer = SimpleNamespace(block_type="attention", mixer=SimpleNamespace(config=mixer_config))
    hidden_states = torch.zeros((2, 4, 8), dtype=torch.bfloat16)
    padding_mask = torch.tensor(
        [
            [0, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    replay_kwargs = model_def.prepare_layer_replay_kwargs(
        layer=layer,
        layer_input=[hidden_states],
        additional_inputs={
            "attention_mask": padding_mask,
            "cache_position": torch.arange(4),
            "use_cache": False,
        },
        target_device=torch.device("cpu"),
    )

    causal_mask = replay_kwargs["attention_mask"]
    min_dtype = torch.finfo(torch.bfloat16).min

    assert mixer_config._attn_implementation == "eager"
    assert causal_mask.shape == (2, 1, 4, 4)
    assert causal_mask.dtype == torch.bfloat16
    assert torch.all(causal_mask[0, 0, :, 0] == min_dtype)
    assert causal_mask[1, 0, 3, 3] == 0
    assert causal_mask[1, 0, 0, 1] == min_dtype


def test_nemotron_h_puzzle_replay_keeps_mamba_padding_mask():
    model_def = object.__new__(NemotronHPuzzleQModel)
    padding_mask = torch.tensor([[0, 1, 1, 1]], dtype=torch.long)
    layer = SimpleNamespace(block_type="mamba", mixer=SimpleNamespace())

    replay_kwargs = model_def.prepare_layer_replay_kwargs(
        layer=layer,
        layer_input=[torch.zeros((1, 4, 8), dtype=torch.bfloat16)],
        additional_inputs={"attention_mask": padding_mask, "use_cache": False},
        target_device=torch.device("cpu"),
    )

    assert replay_kwargs["attention_mask"] is padding_mask


class _WeightlessLinear(nn.Module):
    """QuantLinear-shaped test double which intentionally has no dense weight."""

    def forward(self, hidden_states):
        return hidden_states


class _WeightlessExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = _WeightlessLinear()
        self.down_proj = _WeightlessLinear()

    def forward(self, hidden_states):
        return self.down_proj(self.up_proj(hidden_states))


class NemotronHMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([_WeightlessExpert(), _WeightlessExpert()])


def test_nemotron_h_puzzle_empty_expert_does_not_require_dense_weight():
    moe = NemotronHMoE()
    model_def = SimpleNamespace(
        model=SimpleNamespace(
            model=SimpleNamespace(
                layers=[SimpleNamespace(mixer=moe)],
            )
        )
    )
    NemotronHPuzzleQModel.monkey_patch(model_def)

    hidden_states = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    # Only expert 0 is routed; expert 1 exercises the empty-expert branch.
    topk_indices = torch.zeros((2, 1), dtype=torch.long)
    topk_weights = torch.ones((2, 1), dtype=torch.bfloat16)

    output = moe.moe(hidden_states, topk_indices, topk_weights)

    assert torch.equal(output, hidden_states)
    assert not hasattr(moe.experts[1].down_proj, "weight")
