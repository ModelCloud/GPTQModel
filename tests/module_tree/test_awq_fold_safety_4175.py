# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""Issue #4175: an emitted AWQ fold must leave the full layer output unchanged.

``AWQProcessor._validate_scale_group`` checks widths, bindings, replay
coverage and execution, but it cannot see whether the selected producer's
output really is the consumer's input. Every definition below passed that
validation while folding a scale across a lerp, a nonlinear gate, a
hyper-connection mix, or a scale-free normalization.

Each test builds a real tiny CPU layer, asks the definition for its scaling
groups, applies one non-unit non-uniform fold **without quantization**, and
compares the full output with a fresh native baseline. A correct fold is
neutral to about 1e-7; the parent commit changes these outputs by 1e-3..2e-1.
The `bailing_moe` case builds the real port config: #4620 added an
empty-group override keyed on `isinstance(config, BailingConfig)`, so a
duck-typed fixture passed while every production model emitted no folds.
"""

from types import SimpleNamespace

import pytest
import torch
from awq_neutrality import (  # noqa: F401
    TOLERANCE,
    _assert_neutral,
    _capture,
    _quantized_features,
    _wrapper,
    assert_awq_folds_neutral,
)
from torch import nn

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.base import BaseQModel


class _LanguageModelContainer(nn.Module):
    """Multimodal wrapper shape used by the Qwen3.5-derived definitions."""

    def __init__(self, text, config):
        super().__init__()
        self.config = config
        self.model = nn.Module()
        self.model.language_model = text


def _tiny_qwen35_text(*, hidden_size=32, layer_types=("linear_attention", "full_attention")):
    import transformers

    config = transformers.Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=hidden_size,
        intermediate_size=64,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=64,
        layer_types=list(layer_types),
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=2,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(7)
    return transformers.Qwen3_5TextModel(config).eval()


def test_minicpmv_4_6_linear_attn_tags_preserve_layer_output():
    text = _tiny_qwen35_text()
    model = _LanguageModelContainer(text, SimpleNamespace(text_config=text.config))
    definition = MODEL_MAP["minicpmv4_6"]
    ids = torch.randint(2, 60, (1, 8))

    def forward():
        return text(input_ids=ids).last_hidden_state

    checked = []
    for layer in text.layers:
        checked.extend(_assert_neutral(definition, model, layer, forward))

    linear_groups = [
        (targets, consumers)
        for source, targets, consumers in checked
        if source == "input_layernorm" and "linear_attn.in_proj_qkv" in targets
    ]
    assert linear_groups, "the linear-attention input norm did not fold"
    targets, consumers = linear_groups[0]
    # Both DeltaNet input branches share the norm output, and the unquantized
    # gate projections read the same tensor, so they must be compensated.
    assert targets == ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"]
    assert {"linear_attn.in_proj_b", "linear_attn.in_proj_a"} <= set(consumers)


@pytest.mark.parametrize("model_type", ["minicpmv", "minicpmo"])
def test_minicpm_attention_and_mlp_folds_preserve_layer_output(model_type):
    from tests.test_minicpm_o_local_port import tiny_config as minicpmo_config
    from tests.test_minicpmv_local_port import tiny_config as minicpmv_config

    config = minicpmv_config() if model_type == "minicpmv" else minicpmo_config()
    config._attn_implementation = "eager"
    torch.manual_seed(7)
    definition = MODEL_MAP[model_type]
    model = definition.loader.from_config(config).eval()
    ids = torch.randint(2, 60, (1, 8))

    def forward():
        return model.llm(input_ids=ids, use_cache=False).logits

    checked = []
    for layer in model.llm.model.layers:
        checked.extend(_assert_neutral(definition, model, layer, forward))

    flat = {(source, tuple(targets)) for source, targets, _consumers in checked}
    assert ("input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")) in flat
    assert ("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj")) in flat
    assert ("mlp.up_proj", ("mlp.down_proj",)) in flat

    blocks = [
        [entry.split(":", 1)[0] for entry in block]
        for block in definition.simple_layer_modules(config, SimpleNamespace(dynamic=None))
    ]
    assert ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] in blocks
    assert ["self_attn.o_proj"] in blocks
    assert ["mlp.gate_proj", "mlp.up_proj"] in blocks
    assert ["mlp.down_proj"] in blocks


def _tiny_diffusion_gemma_model():
    import transformers
    from transformers import AutoModelForMultimodalLM

    from gptqmodel.models.loader import _convert_model_with_defuser

    config = transformers.DiffusionGemmaConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 48,
            "intermediate_size": 64,
            "moe_intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "num_global_key_value_heads": 4,
            "head_dim": 8,
            "global_head_dim": 8,
            "num_experts": 2,
            "top_k_experts": 1,
            "max_position_embeddings": 64,
            "sliding_window": 8,
            "layer_types": ["sliding_attention", "full_attention"],
            "use_bidirectional_attention": "vision",
        },
        vision_config={
            "model_type": "gemma4_vision",
            "hidden_size": 16,
            "intermediate_size": 24,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "patch_size": 2,
            "pooling_kernel_size": 1,
            "position_embedding_size": 16,
        },
        canvas_length=4,
        image_token_id=60,
        boi_token_id=61,
        eoi_token_id=62,
    )
    config.text_config._attn_implementation = "eager"
    torch.manual_seed(7)
    model = AutoModelForMultimodalLM.from_config(config, dtype=torch.float32).eval()
    _convert_model_with_defuser(MODEL_MAP["diffusion_gemma"], model, cleanup_original=False)
    return model


def test_diffusion_gemma_router_and_value_folds_are_neutral():
    model = _tiny_diffusion_gemma_model()
    definition = MODEL_MAP["diffusion_gemma"]
    text_model = model.model.encoder.language_model
    layers = text_model.layers
    hidden = torch.randn(1, 8, 48)
    position_ids = torch.arange(8).unsqueeze(0)

    checked = []
    for index, layer in enumerate(layers):
        layer_type = model.config.text_config.layer_types[index]
        cos, sin = text_model.rotary_emb(hidden, position_ids, layer_type)

        def forward(layer=layer, cos=cos, sin=sin):
            output = layer(
                hidden,
                position_embeddings=(cos, sin),
                attention_mask=None,
                position_ids=position_ids,
            )
            return output[0] if isinstance(output, tuple) else output

        checked.extend(_assert_neutral(definition, model, layer, forward))

    assert checked
    # The scale-free `v_norm` sits between `v_proj` and `o_proj`.
    assert "self_attn.v_proj" not in {source for source, _targets, _consumers in checked}
    # The router applies its own scale-free norm to the raw residual; it never
    # receives `pre_feedforward_layernorm_2`'s scale.
    for source, _targets, consumers in checked:
        assert "router.proj" not in consumers, source


def test_ricursive_folds_preserve_layer_output():
    from ricursive_test_utils import make_model

    # Released shape: value residual, gated attention, GQA.
    model = make_model()
    definition = MODEL_MAP["ricursive"]
    ids = torch.randint(2, 60, (1, 8))

    def forward():
        return model(input_ids=ids, use_cache=False).logits

    checked = []
    for layer in model.model.layers:
        checked.extend(_assert_neutral(definition, model, layer, forward))

    # The value-residual lerp keeps `input_layernorm` out; GQA keeps
    # `v_proj -> o_proj` out by width.
    assert {source for source, _targets, _consumers in checked} == {"pre_feedforward_layernorm", "mlp.up_proj"}


def test_qwen4_exp_shared_expert_folds_are_neutral():
    from test_qwen4_exp_support import _tiny_qwen4_exp_text_model

    model = _tiny_qwen4_exp_text_model()
    definition = MODEL_MAP["qwen4_exp_text"]
    ids = torch.randint(2, 60, (1, 8))

    def forward():
        output = model(input_ids=ids)
        return output.logits if hasattr(output, "logits") else output.last_hidden_state

    checked = []
    for layer in model.model.layers:
        checked.extend(_assert_neutral(definition, model, layer, forward))

    assert checked
    sources = {source for source, _targets, _consumers in checked}
    # The hyper-connection mix sits between the attention output and the MLP.
    assert "self_attn.o_proj" not in sources
    assert "linear_attn.out_proj" not in sources
    assert "linear_attn.in_proj_a" not in sources


# ---------------------------------------------------------------------------
# Issue #4759: an absent *sibling branch* must not take the norm away.
#
# #4751 exempted a block from `refuse_last_module_fallback` only when the block
# and the preceding absent block shared one `:in=` tag. Hybrid trees whose
# alternative mixer branches carry no tag (`mamba`/`self_attn` on Granite-4.0-H,
# `conv`/`self_attn` on LFM2, the `mixer` alternatives of Nemotron-H,
# `self_attn`/`attention` on Ling 2.0) still lost the norm -> projection group,
# so the layer silently fell back to RTN. These cases assert the group is
# emitted *and* that folding it leaves the layer output unchanged.
# ---------------------------------------------------------------------------


def _tiny_granitemoehybrid_attention():
    import transformers

    config = transformers.GraniteMoeHybridConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        # Granite-4.0-H interleaves Mamba and attention layers. An attention
        # layer owns no `mamba` mixer, so its declared `mamba.*` blocks resolve
        # to nothing.
        layer_types=["full_attention", "full_attention"],
        mamba_n_heads=4,
        mamba_d_head=16,
        mamba_d_state=16,
        mamba_n_groups=1,
        mamba_d_conv=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        shared_intermediate_size=16,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(7)
    return transformers.GraniteMoeHybridForCausalLM(config).eval()


def test_granitemoehybrid_attention_norm_still_folds():
    model = _tiny_granitemoehybrid_attention()
    definition = MODEL_MAP["granitemoehybrid"]
    ids = torch.randint(2, 60, (1, 8))

    def forward():
        return model(input_ids=ids).logits

    checked = []
    for layer in model.model.layers:
        checked.extend(_assert_neutral(definition, model, layer, forward))

    norm_groups = [
        targets for source, targets, _consumers in checked if source == "input_layernorm"
    ]
    assert ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] in norm_groups, (
        "the attention input norm did not fold after the absent `mamba` sibling branch"
    )


def _tiny_lfm2_attention():
    import transformers

    config = transformers.Lfm2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        # LFM2 mixes short-convolution and full-attention layers. A
        # full-attention layer owns no `conv` mixer.
        layer_types=["full_attention", "full_attention"],
    )
    config._attn_implementation = "eager"
    torch.manual_seed(7)
    return transformers.Lfm2ForCausalLM(config).eval()


def test_lfm2_attention_norm_still_folds():
    model = _tiny_lfm2_attention()
    definition = MODEL_MAP["lfm2"]
    ids = torch.randint(2, 60, (1, 8))

    def forward():
        return model(input_ids=ids).logits

    checked = []
    for layer in model.model.layers:
        checked.extend(_assert_neutral(definition, model, layer, forward))

    norm_groups = [
        targets for source, targets, _consumers in checked if source == "operator_norm"
    ]
    assert ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"] in norm_groups, (
        "the attention input norm did not fold after the absent `conv` sibling branch"
    )


def _tiny_nemotron_h_mixers():
    import transformers

    config = transformers.NemotronHConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        # One layer of each mixer the issue names: Mamba (`M`), dense MLP (`-`)
        # and attention (`*`). `moe` layers are covered elsewhere.
        layers_block_type=["linear_attention", "mlp", "full_attention"],
        mamba_num_heads=4,
        mamba_head_dim=8,
        ssm_state_size=8,
        n_groups=1,
        conv_kernel=2,
        expand=2,
        n_routed_experts=0,
        n_shared_experts=1,
        moe_intermediate_size=16,
        moe_shared_expert_intermediate_size=16,
        num_experts_per_tok=2,
        use_mamba_kernels=False,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(7)
    return transformers.NemotronHForCausalLM(config).eval()


def test_generic_planner_keeps_nemotron_h_mixer_norm_folds():
    from test_awq_scaling_groups import _mamba_ssm_cpu_unavailable_reason

    reason = _mamba_ssm_cpu_unavailable_reason()
    if reason is not None:
        pytest.skip(reason)

    model = _tiny_nemotron_h_mixers()
    definition = MODEL_MAP["nemotron_h"]
    ids = torch.randint(2, 60, (1, 8))

    def forward():
        return model(input_ids=ids).logits

    checked = []
    for layer in model.model.layers:
        # The registered Nemotron-H port uses plain AWQ. Exercise the generic
        # planner here to retain the absent-sibling regression on this layout.
        checked.extend(_assert_neutral(
            definition, model, layer, forward, scaling_provider=BaseQModel.awq_get_modules_for_scaling,
        ))

    norm_groups = [targets for source, targets, _consumers in checked if source == "norm"]
    # Mamba (`M`) layer: the SSM input projection reads the block norm.
    assert ["mixer.in_proj"] in norm_groups, (
        "the Mamba mixer norm group was dropped after the absent `mixer.q_proj` siblings"
    )
    # Dense MLP (`-`) layer: `NemotronHMLP` has no `gate_proj`, and `out_proj`
    # belongs to the other mixers.
    assert ["mixer.up_proj"] in norm_groups, (
        "the MLP mixer norm group was dropped after the absent `mixer.in_proj` siblings"
    )
    # Attention (`*`) layer keeps the group it already had.
    assert ["mixer.q_proj", "mixer.k_proj", "mixer.v_proj"] in norm_groups


# ---------------------------------------------------------------------------
# Ling 2.0/2.5 (`bailing_moe`): the fold plan must be built from the real
# `BailingConfig`. #4620 keyed an empty-group override on
# `isinstance(model.config, BailingConfig)`; the previous fixture used
# `SimpleNamespace(model_type="bailing_moe")`, bypassed that guard and passed
# while every production checkpoint silently fell back to plain quantization.
# ---------------------------------------------------------------------------


def _bailing_attention_inputs(layout):
    """Exact `input_layernorm` fold targets for the two-layer tiny stack."""
    if layout == "hybrid":
        # Ling 2.5 alternates lightning attention (layer 0, qkv + sigmoid
        # gate) with MLA (layer 1, low-rank q/kv inputs).
        return [
            ["attention.query_key_value", "attention.g_proj"],
            ["attention.q_a_proj", "attention.kv_a_proj_with_mqa"],
        ]
    if layout == "mla":
        return [["attention.q_a_proj", "attention.kv_a_proj_with_mqa"]] * 2
    # `dense` and `legacy_gqa` both use `DenseAttention`.
    return [["attention.query_key_value"]] * 2


@pytest.mark.parametrize("layout", ["dense", "mla", "hybrid", "legacy_gqa"])
def test_bailing_moe_awq_folds_are_neutral_on_production_config(layout):
    from gptqmodel.models.definitions.bailing_moe import BailingMoeQModel
    from gptqmodel.models.ports.bailing_moe.configuration_bailing import BailingConfig
    from tests.test_bailing_moe_local_port import tiny_config

    torch.manual_seed(7)
    model = BailingMoeQModel.loader.from_config(tiny_config(layout)).eval()
    # The removed override was keyed on this exact type, so a duck-typed
    # `SimpleNamespace(model_type="bailing_moe")` cannot reproduce the
    # regression this test guards.
    assert isinstance(model.config, BailingConfig)

    ids = torch.randint(2, model.config.vocab_size, (1, 8))

    def forward():
        return model(input_ids=ids, use_cache=False).logits

    emitted = []
    for layer in model.model.layers:
        checked = assert_awq_folds_neutral(BailingMoeQModel, model, layer, forward)
        emitted.append({
            (source, tuple(targets), tuple(consumers))
            for source, targets, consumers in checked
        })
    assert len(emitted) == 2

    for layer_index, targets in enumerate(_bailing_attention_inputs(layout)):
        assert ("input_layernorm", tuple(targets), ()) in emitted[layer_index], (
            f"{layout}: input_layernorm did not fold {targets}; got {sorted(emitted[layer_index])}"
        )

    # `first_k_dense_replace=1`: layer 0 is the dense prefix, layer 1 is MoE.
    assert ("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj"), ()) in emitted[0]
    assert ("mlp.up_proj", ("mlp.down_proj",), ()) in emitted[0]

    # The MoE block folds the norm into the shared expert and every routed
    # expert, and compensates the unquantized router that reads the same norm.
    moe_targets = ("mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj") + tuple(
        f"mlp.experts.{expert}.{suffix}"
        for expert in range(4)
        for suffix in ("gate_proj", "up_proj")
    )
    assert ("post_attention_layernorm", moe_targets, ("mlp.gate",)) in emitted[1], (
        f"{layout}: MoE norm group did not cover shared and routed experts; "
        f"got {sorted(emitted[1])}"
    )
    assert ("mlp.shared_experts.up_proj", ("mlp.shared_experts.down_proj",), ()) in emitted[1]
    for expert in range(4):
        assert (
            f"mlp.experts.{expert}.up_proj",
            (f"mlp.experts.{expert}.down_proj",),
            (),
        ) in emitted[1]
