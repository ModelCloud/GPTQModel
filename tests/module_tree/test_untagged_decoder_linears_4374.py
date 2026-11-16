# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Decoder Linears that were silently left in the source dtype (issue #4374).

``module_tree`` is the oracle for checkpoint-owned weights. A Linear that the
tree neither declares nor covers with an explicit ``:!``/``:?`` tag stays in
BF16 without an error. These tests walk ``state_dict()`` keys, resolve the
deepest owning module, and require a plan entry (or a tagged ancestor).

DeepSeek-VL2's remote implementation needs ``timm``, so its non-MLA layout is
covered with a local architecture fixture whose module names match the
checkpoint (``language.model.layers.N.self_attn.{q,k,v,o}_proj``). Nemotron-H
and Qwen4-Exp are built from their native Transformers configs on ``meta``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models._const import SUPPORTS_MODULE_TYPES
from gptqmodel.models.definitions.deepseek_vl_v2 import DeepSeekVLV2QModel
from gptqmodel.models.definitions.nemotron_h import NemotronHQModel
from gptqmodel.models.definitions.qwen4_exp import Qwen4ExpQModel, Qwen4ExpTextQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.utils.model import get_layers_with_prefixes


QC = SimpleNamespace(dynamic=None)


def _covered(plan: ModulePlan, path: str) -> bool:
    if plan.find(path) is not None:
        return True
    parts = path.split(".")
    for end in range(len(parts) - 1, 0, -1):
        spec = plan.find(".".join(parts[:end]))
        if spec is not None and any(flag in {"!", "?"} for flag in spec.flags):
            return True
    return False


def _state_dict_weight_owners(layer: nn.Module):
    """Yield the deepest owning module for every checkpoint-style tensor."""

    for weight_name in layer.state_dict():
        owner = weight_name.rsplit(".", 1)[0]
        module = layer.get_submodule(owner)
        if isinstance(module, tuple(SUPPORTS_MODULE_TYPES)):
            yield owner


def _assert_state_dict_leaves_declared(definition, layers):
    plan = ModulePlan.from_tree(definition.module_tree)
    undeclared = {}
    for layer_index, layer in enumerate(layers):
        for owner in _state_dict_weight_owners(layer):
            if not _covered(plan, owner):
                undeclared.setdefault(owner, []).append(layer_index)
    assert not undeclared, (
        f"{definition.__name__}: checkpoint-owned decoder leaves are neither declared "
        f"nor tagged :!/:? and stay in the source dtype: "
        + ", ".join(f"{path} (layer(s) {indexes})" for path, indexes in sorted(undeclared.items()))
    )
    return plan


# --- DeepSeek-VL2: non-MLA checkpoints use LlamaAttention ---------------------


class _NonMLAttention(nn.Module):
    """``LlamaAttention`` layout used when ``language_config.use_mla`` is false."""

    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class _MLAttention(nn.Module):
    """MLA layout kept by the other DeepSeek-VL2 checkpoints."""

    def __init__(self, hidden_size: int = 8, rank: int = 4):
        super().__init__()
        self.q_a_proj = nn.Linear(hidden_size, rank, bias=False)
        self.q_b_proj = nn.Linear(rank, hidden_size, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(hidden_size, rank, bias=False)
        self.kv_b_proj = nn.Linear(rank, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class _DenseMLP(nn.Module):
    def __init__(self, hidden_size: int = 8, intermediate_size: int = 16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)


class _Expert(nn.Module):
    def __init__(self, hidden_size: int = 8, intermediate_size: int = 16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)


class _Router(nn.Module):
    """Top-k router owning a checkpoint weight but no quantizable Linear."""

    def __init__(self, hidden_size: int = 8, num_experts: int = 2):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_experts, hidden_size))


class _MoE(nn.Module):
    def __init__(self, hidden_size: int = 8, intermediate_size: int = 16):
        super().__init__()
        self.gate = _Router(hidden_size)
        self.experts = nn.ModuleList([_Expert(hidden_size, intermediate_size)])
        self.shared_experts = _DenseMLP(hidden_size, intermediate_size)


class _DecoderLayer(nn.Module):
    def __init__(self, attention: nn.Module, mlp: nn.Module):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(8)
        self.self_attn = attention
        self.post_attention_layernorm = nn.LayerNorm(8)
        self.mlp = mlp


def _deepseek_vl_v2_fixture() -> nn.Module:
    """Mirror the checkpoint's decoder names for both attention layouts."""

    layers = nn.ModuleList(
        [
            _DecoderLayer(_NonMLAttention(), _DenseMLP()),
            _DecoderLayer(_MLAttention(), _MoE()),
        ]
    )
    shell = nn.Module()
    shell.language = nn.Module()
    shell.language.model = nn.Module()
    shell.language.model.layers = layers
    return shell


def test_deepseek_vl_v2_non_mla_k_and_v_projections_are_quantized():
    shell = _deepseek_vl_v2_fixture()
    layers = shell.language.model.layers

    # The non-MLA checkpoint tensors are the ones the old tree skipped.
    assert "self_attn.k_proj.weight" in layers[0].state_dict()
    assert "self_attn.v_proj.weight" in layers[0].state_dict()

    plan = _assert_state_dict_leaves_declared(DeepSeekVLV2QModel, layers)
    for path in (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "mlp.experts.0.gate_proj",
        "mlp.experts.0.up_proj",
        "mlp.experts.0.down_proj",
        "mlp.shared_experts.gate_proj",
        "mlp.shared_experts.up_proj",
        "mlp.shared_experts.down_proj",
    ):
        spec = plan.find(path)
        assert spec is not None, f"{path} is not declared"
        assert spec.quantize, (path, spec.flags)

    # The MLA branch stays declared for the other checkpoints.
    for path in (
        "self_attn.q_a_proj",
        "self_attn.q_b_proj",
        "self_attn.kv_a_proj_with_mqa",
        "self_attn.kv_b_proj",
    ):
        spec = plan.find(path)
        assert spec is not None, f"{path} is not declared"
        assert spec.quantize, (path, spec.flags)


# --- Nemotron-H: latent-MoE projections --------------------------------------


def _tiny_nemotron_h_latent_config():
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

    return NemotronHConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_n_groups=1,
        mamba_d_head=16,
        mamba_n_heads=4,
        layers_block_type=["mamba", "moe"],
        moe_intermediate_size=128,
        moe_shared_expert_intermediate_size=64,
        moe_latent_size=64,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def _build_nemotron_h_latent_model():
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM

    config = _tiny_nemotron_h_latent_config()
    with torch.device("meta"):
        model = NemotronHForCausalLM(config)
    _convert_model_with_defuser(NemotronHQModel, model, cleanup_original=False)
    return config, model


def test_nemotron_h_latent_moe_projections_are_declared_and_ordered():
    config, model = _build_nemotron_h_latent_model()
    layers = model.model.layers
    moe_layer = next(layer for layer in layers if type(layer.mixer).__name__ == "NemotronHMoE")
    assert "mixer.fc1_latent_proj.weight" in moe_layer.state_dict()
    assert "mixer.fc2_latent_proj.weight" in moe_layer.state_dict()

    plan = _assert_state_dict_leaves_declared(NemotronHQModel, layers)
    for path in ("mixer.fc1_latent_proj", "mixer.fc2_latent_proj"):
        spec = plan.find(path)
        assert spec is not None, f"{path} is not declared"
        assert spec.quantize, (path, spec.flags)

    blocks = NemotronHQModel.simple_layer_modules(model_config=config, quantize_config=QC)
    fc1 = next(i for i, block in enumerate(blocks) if "mixer.fc1_latent_proj" in block)
    experts = next(i for i, block in enumerate(blocks) if "mixer.experts.0.up_proj" in block)
    fc2 = next(i for i, block in enumerate(blocks) if "mixer.fc2_latent_proj" in block)
    assert fc1 < experts < fc2, blocks


# --- Qwen4-Exp: PLE and hyper-connection Linears -----------------------------


def _tiny_qwen4_exp_text_config():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    return Qwen4ExpTextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=3,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        layer_types=["linear_attention", "indexed_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=2,
        hc_count=2,
        hc_lowrank=4,
        ple_layer_ids=[1],
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=4,
        indexer_budget=8,
        indexer_compress_ratio=2,
    )


def _tiny_qwen4_exp_config():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import (
        Qwen4ExpConfig,
        Qwen4ExpVisionConfig,
    )

    return Qwen4ExpConfig(
        text_config=_tiny_qwen4_exp_text_config(),
        vision_config=Qwen4ExpVisionConfig(
            depth=1,
            hidden_size=16,
            num_heads=2,
            intermediate_size=16,
        ),
    )


@pytest.mark.parametrize(
    ("definition", "config_factory"),
    [
        (Qwen4ExpTextQModel, _tiny_qwen4_exp_text_config),
        (Qwen4ExpQModel, _tiny_qwen4_exp_config),
    ],
    ids=["qwen4_exp_text", "qwen4_exp"],
)
def test_qwen4_exp_ple_and_hyper_connection_leaves_are_excluded(definition, config_factory):
    config = config_factory()
    with torch.device("meta"):
        model = definition.loader.from_config(config)
    _convert_model_with_defuser(definition, model, cleanup_original=False)
    layers, _ = get_layers_with_prefixes(model, definition.extract_layers_node())
    assert layers

    plan = _assert_state_dict_leaves_declared(definition, layers)
    for path in ("ple", "attn_hyper_connection", "mlp_hyper_connection"):
        spec = plan.find(path)
        assert spec is not None, f"{path} is not declared"
        assert not spec.quantize, (path, spec.flags)

    # PLE is optional per layer; it is present because ``ple_layer_ids=[1]``.
    assert "ple.key_proj.weight" in layers[0].state_dict()
    assert "ple.value_proj.weight" in layers[0].state_dict()
    assert "ple.conv1d.weight" in layers[0].state_dict()
    assert "ple.ple_embedding.ngram_embedding.weight" in layers[0].state_dict()
    for path in (
        "attn_hyper_connection.input_mix_weight_down",
        "attn_hyper_connection.input_mix_weight_up",
        "attn_hyper_connection.block_inject_weight",
        "mlp_hyper_connection.input_mix_weight_down",
        "mlp_hyper_connection.input_mix_weight_up",
        "mlp_hyper_connection.block_inject_weight",
    ):
        assert _covered(plan, path), f"{path} is not declared or covered by a tagged ancestor"
