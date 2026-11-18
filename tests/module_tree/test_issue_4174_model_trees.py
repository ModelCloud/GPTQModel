# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Per-model module-plan regressions for issue #4174.

Each case builds the real Transformers shell on the ``meta`` device where the
config is available locally, runs the loader's Defuser conversion, and checks
that the definition's declared leaves resolve on the actual decoder layers.
Remote-code-only layouts (``bailing_moe`` / ``grinmoe``) use a minimal local
architecture fixture with the same module names; the registry-wide test skips
those configs because constructing them needs network access.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import transformers
from torch import nn

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.bailing_moe import BailingMoeQModel
from gptqmodel.models.definitions.gemma4 import Gemma4TextQModel
from gptqmodel.models.definitions.grinmoe import GrinMoeQModel
from gptqmodel.models.definitions.kimi_linear import KimiLinearNativeQModel
from gptqmodel.models.definitions.llama4 import Llama4QModel, Llama4TextQModel
from gptqmodel.models.definitions.phi3 import Phi3QModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.native.auto import AutoConfig as NativeAutoConfig
from gptqmodel.utils.model import get_layers_with_prefixes


def _auto_config(model_type, **kwargs):
    """Build the default config with the class the definition's loader expects."""
    loader = getattr(MODEL_MAP.get(model_type), "loader", None)
    if isinstance(loader, LocalModelLoader) and loader.config_class.startswith("gptqmodel.models.ports.native."):
        return NativeAutoConfig.for_model(model_type, **kwargs)
    return transformers.AutoConfig.for_model(model_type, **kwargs)


QC = SimpleNamespace(dynamic=None)


def _build_meta(model_type, config=None):
    definition = MODEL_MAP[model_type]
    if config is None:
        config = _auto_config(model_type)
    with torch.device("meta"):
        model = definition.loader.from_config(config)
    _convert_model_with_defuser(definition, model, cleanup_original=False)
    layers, _ = get_layers_with_prefixes(model, definition.extract_layers_node())
    return definition, config, model, layers or []


def _assert_plan_resolves(plan, paths, *, quantize=None):
    for path in paths:
        spec = plan.find(path)
        assert spec is not None, f"{path} is not declared"
        if quantize is not None:
            assert spec.quantize is quantize, (path, spec.flags)


def _covered(plan, path):
    if plan.find(path) is not None:
        return True
    parts = path.split(".")
    for end in range(len(parts) - 1, 0, -1):
        spec = plan.find(".".join(parts[:end]))
        if spec is not None and any(flag in {"!", "?"} for flag in spec.flags):
            return True
    return False


def test_voxtral_tree_uses_the_5_22_language_model_root():
    definition, _, _, layers = _build_meta("voxtral")

    assert definition.extract_layers_node() == ["model.language_model.layers"]
    assert layers, "Voxtral must resolve decoder layers from model.language_model"
    plan = ModulePlan.from_tree(definition.module_tree)
    _assert_plan_resolves(
        plan,
        (
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ),
        quantize=True,
    )


def _nemotron_moe_config():
    config = _auto_config("nemotron_h")
    config.num_hidden_layers = 4
    config.layers_block_type = ["linear_attention", "full_attention", "mlp", "moe"]
    return config


def test_nemotron_h_moe_declares_routed_shared_and_router_leaves():
    definition, config, _, layers = _build_meta("nemotron_h", _nemotron_moe_config())

    assert definition.dynamic_expert_index == "n_routed_experts"
    moe_layer = next(layer for layer in layers if getattr(layer, "mixer", None) is not None and hasattr(layer.mixer, "experts"))
    assert moe_layer.mixer.experts.__class__.__name__ == "NemotronHExperts"

    plan = ModulePlan.from_tree(definition.module_tree)
    _assert_plan_resolves(
        plan,
        (
            "mixer.experts.0.up_proj",
            "mixer.experts.0.down_proj",
            "mixer.shared_experts.up_proj",
            "mixer.shared_experts.down_proj",
        ),
        quantize=True,
    )
    _assert_plan_resolves(plan, ("mixer.conv1d", "mixer.gate"), quantize=False)

    specs = {
        name.split(":", 1)[0]
        for block in definition.simple_layer_modules(model_config=config, quantize_config=QC)
        for name in block
    }
    assert "mixer.experts.0.up_proj" in specs
    assert "mixer.shared_experts.up_proj" in specs


def test_bailing_moe_keeps_the_remote_attention_layout():
    """`bailing_moe` is remote-code only and still uses attention.qkv/dense."""

    plan = ModulePlan.from_tree(BailingMoeQModel.module_tree)
    _assert_plan_resolves(plan, ("attention.query_key_value", "attention.dense"), quantize=True)
    _assert_plan_resolves(plan, ("self_attn.q_proj", "self_attn.conv1d"), quantize=None)

    # The same definition also serves native `bailing_hybrid`, which keeps
    # `self_attn`. Both branches must be emitted with strict resolution off.
    specs = {
        name.split(":", 1)[0]
        for block in BailingMoeQModel.simple_layer_modules(
            model_config=SimpleNamespace(num_experts=2, num_local_experts=2, num_shared_experts=1),
            quantize_config=QC,
        )
        for name in block
    }
    assert "attention.query_key_value" in specs
    assert "attention.dense" in specs
    assert "self_attn.q_proj" in specs

    declared = {
        name.split(":", 1)[0]
        for block in BailingMoeQModel.build_layer_modules(
            BailingMoeQModel.module_tree, include_capture_only=True
        )
        for name in block
    }
    assert "self_attn.conv1d" in declared


def _llama4_mixed_config():
    config = _auto_config("llama4_text")
    config.num_hidden_layers = 4
    config.interleave_moe_layer_step = 2
    config.moe_layers = [0, 2]
    return config


def test_llama4_dense_layers_are_declared_next_to_moe():
    definition, _, model, layers = _build_meta("llama4_text", _llama4_mixed_config())

    assert [type(layer.feed_forward).__name__ for layer in layers] == [
        "Llama4TextMoe",
        "Llama4TextMLP",
        "Llama4TextMoe",
        "Llama4TextMLP",
    ]
    plan = ModulePlan.from_tree(definition.module_tree)
    _assert_plan_resolves(
        plan,
        (
            "feed_forward.gate_proj",
            "feed_forward.up_proj",
            "feed_forward.down_proj",
            "feed_forward.experts.0.gate_proj",
        ),
        quantize=True,
    )

    # The text-only causal LM owns its head at the top level.
    assert Llama4TextQModel.lm_head == "lm_head"
    assert model.get_submodule(Llama4TextQModel.lm_head) is model.lm_head
    assert Llama4QModel.lm_head == "language_model.lm_head"


def _llama4_scout_config():
    config = _auto_config("llama4_text")
    config.num_hidden_layers = 2
    config.interleave_moe_layer_step = 1
    config.moe_layers = [0, 1]
    return config


def test_llama4_scout_declares_no_dense_feed_forward_branch():
    """Scout is all-MoE, so #4245's dense fallback must not be required (#4689)."""

    definition, config, model, layers = _build_meta("llama4_text", _llama4_scout_config())

    assert [type(layer.feed_forward).__name__ for layer in layers] == [
        "Llama4TextMoe",
        "Llama4TextMoe",
    ]

    # The class tree still declares the dense branch for Maverick.
    assert ModulePlan.from_tree(Llama4TextQModel.module_tree).find("feed_forward.gate_proj") is not None

    # A Scout checkpoint resolves an effective tree without it, so a strict
    # `create_named_modules` never asks for `feed_forward.gate_proj`.
    tree = definition._resolve_effective_module_tree(model, QC)
    plan = ModulePlan.from_tree(tree)
    assert plan.find("feed_forward.gate_proj") is None
    _assert_plan_resolves(
        plan,
        (
            "feed_forward.experts.0.gate_proj",
            "feed_forward.shared_expert.gate_proj",
        ),
        quantize=True,
    )

    with definition._module_tree_context(tree):
        modules = definition.simple_layer_modules(config, QC)
    names = {name.split(":", 1)[0] for block in modules for name in block}
    assert "feed_forward.experts.0.gate_proj" in names
    assert "feed_forward.gate_proj" not in names


def _gemma4_moe_config():
    config = _auto_config("gemma4_text")
    config.num_hidden_layers = 2
    config.enable_moe_block = True
    config.num_experts = 8
    config.top_k_experts = 2
    config.moe_intermediate_size = 256
    return config


def test_gemma4_moe_splits_and_declares_3d_expert_parameters():
    definition = Gemma4TextQModel
    assert definition.defuser_generic_experts is True
    assert definition.dynamic_expert_index == "num_experts"

    config = _gemma4_moe_config()
    with torch.device("meta"):
        model = definition.loader.from_config(config)
    layers, _ = get_layers_with_prefixes(model, definition.extract_layers_node())
    experts = layers[0].experts
    assert experts.__class__.__name__ == "Gemma4TextExperts"
    assert experts.gate_up_proj.dim() == 3
    assert experts.down_proj.dim() == 3

    _convert_model_with_defuser(definition, model, cleanup_original=False)
    assert isinstance(layers[0].experts[0].gate_proj, nn.Linear)

    plan = ModulePlan.from_tree(definition.module_tree)
    _assert_plan_resolves(
        plan,
        (
            "experts.0.gate_proj",
            "experts.0.up_proj",
            "experts.0.down_proj",
            "router.proj",
        ),
        quantize=None,
    )
    assert plan.find("experts.0.gate_proj").quantize
    assert not plan.find("router.proj").quantize


def _phi3_longrope_config():
    factors = [1.0] * 48
    return transformers.Phi3Config(
        max_position_embeddings=131072,
        rope_parameters={
            "rope_type": "longrope",
            "short_factor": factors,
            "long_factor": [2.0] * 48,
            "original_max_position_embeddings": 4096,
        },
    )


def test_phi3_longrope_uses_the_same_gate_up_split_as_native():
    definition, _, _, layers = _build_meta("phi3", _phi3_longrope_config())
    plan = ModulePlan.from_tree(definition.module_tree)
    assert layers
    _assert_plan_resolves(
        plan,
        ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
        quantize=True,
    )
    assert plan.find("mlp.gate_up_proj") is None


class Phi3MLP(nn.Module):
    """Minimal stand-in for the remote-code `Phi3MLP` class."""

    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Linear(8, 16, bias=False)
        self.down_proj = nn.Linear(8, 8, bias=False)
        self.activation_fn = nn.Identity()


class _RemotePhi3Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = Phi3MLP()


class _RemotePhi3Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_RemotePhi3Layer()])


def test_phi3_remote_fused_mlp_is_split_before_strict_resolution():
    model = _RemotePhi3Root()
    plan = ModulePlan.from_tree(Phi3QModel.module_tree)

    # The remote layout exposes only `gate_up_proj`; the declared gate/up
    # specs miss and strict resolution raises during quantization.
    assert plan.find("mlp.gate_up_proj") is None

    Phi3QModel.after_defuser_conversion(model)
    mlp = model.model.layers[0].mlp
    assert not hasattr(mlp, "gate_up_proj")
    assert mlp.gate_proj.weight.shape == mlp.up_proj.weight.shape == (8, 8)
    _assert_plan_resolves(
        plan,
        ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"),
        quantize=True,
    )


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("phimoe", "model.norm"),
        ("falcon_h1", "model.final_layernorm"),
        ("nemotron_h", "model.norm_f"),
        ("hrm_text", "model.H_module.final_norm"),
        ("zamba", "model.final_layernorm"),
        ("zamba2", "model.final_layernorm"),
    ],
)
def test_pre_lm_head_norm_resolves_on_the_real_meta_model(model_type, expected):
    definition = MODEL_MAP[model_type]
    assert definition.pre_lm_head_norm_module == expected

    config = _auto_config(model_type)
    with torch.device("meta"):
        model = definition.loader.from_config(config)

    instance = definition.__new__(definition)
    nn.Module.__init__(instance)
    instance.model = model
    resolved = instance._resolved_pre_lm_head_norm_candidates()
    assert [name for _, name in resolved] == [expected]


def test_zamba_family_declares_conv_and_shared_subtree():
    for model_type, shared in (("zamba", "shared_transf"), ("zamba2", "shared_transformer")):
        definition = MODEL_MAP[model_type]
        plan = ModulePlan.from_tree(definition.module_tree)
        _assert_plan_resolves(plan, ("mamba.conv1d", "mamba_decoder.mamba.conv1d"), quantize=False)
        assert _covered(plan, f"{shared}.self_attn.q_proj")
        assert _covered(plan, f"{shared}.feed_forward.down_proj")
        assert not plan.find("mamba.conv1d").quantize
        assert not plan.find(shared).quantize


def test_jamba_declares_dt_proj_and_conv():
    plan = ModulePlan.from_tree(MODEL_MAP["jamba"].module_tree)
    _assert_plan_resolves(plan, ("mamba.conv1d", "mamba.dt_proj"), quantize=False)


class _GrinMoeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(8, 4, bias=False)


def test_grin_moe_declares_the_block_sparse_gate():
    plan = ModulePlan.from_tree(GrinMoeQModel.module_tree)
    spec = plan.find("block_sparse_moe.gate")
    assert spec is not None
    assert not spec.quantize
    assert isinstance(_GrinMoeBlock().gate, nn.Linear)


def test_kimi_linear_native_has_no_phantom_rotary_path():
    definition = KimiLinearNativeQModel
    assert definition.rotary_embedding is None

    config = _auto_config("kimi_linear")
    with torch.device("meta"):
        model = definition.loader.from_config(config)
    assert not hasattr(model.model, "rotary_emb")

    _convert_model_with_defuser(definition, model, cleanup_original=False)
    plan = ModulePlan.from_tree(definition.module_tree)
    _assert_plan_resolves(
        plan,
        (
            "self_attn.conv1d",
            "mlp.experts.0.gate_proj",
            "mlp.experts.0.down_proj",
        ),
        quantize=None,
    )
    assert not plan.find("self_attn.conv1d").quantize
