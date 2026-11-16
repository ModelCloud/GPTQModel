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
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_awq_scaling_groups import _foldable_groups
from torch import nn

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.quantization.awq.quantize.scale import apply_scale
from gptqmodel.quantization.awq.utils.module import get_op_name


TOLERANCE = 1e-5
_INTERN_REVISION = "4f57cab513689b089019fce4ad24e26520df183c"


def _wrapper(definition, model, tree):
    wrapper = definition.__new__(definition)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    wrapper.__dict__["effective_module_tree"] = tree
    wrapper.__dict__["module_tree"] = tree
    return wrapper


def _capture(layer, forward):
    features = {}
    handles = []

    def record(name):
        def hook(module, args, kwargs):
            value = args[0] if args else kwargs.get("hidden_states")
            if isinstance(value, torch.Tensor):
                features[name] = value.detach().clone()
        return hook

    try:
        for name, module in layer.named_modules():
            handles.append(module.register_forward_pre_hook(record(name), with_kwargs=True))
        with torch.inference_mode():
            reference = forward().clone()
    finally:
        for handle in handles:
            handle.remove()
    return reference, features


def _quantized_features(layer, features, tree):
    """Keep only the captured inputs of quantized 2-D weight modules.

    The AWQ processor's ``input_feat`` only contains the modules it will
    quantize. Capturing every module would let a non-quantized norm inside a
    stage win ``_select_feature_name`` and change the measured input width.
    """

    plan = ModulePlan.from_tree(tree)
    kept = {}
    for name, value in features.items():
        spec = plan.find(name)
        if spec is None or not spec.quantize:
            continue
        weight = getattr(layer.get_submodule(name), "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            kept[name] = value
    return kept


def _assert_neutral(definition, model, layer, forward, tree=None):
    """Apply every emitted fold once and require the full output to be unchanged."""

    tree = tree if tree is not None else definition.module_tree
    wrapper = _wrapper(definition, model, tree)
    reference, features = _capture(layer, forward)
    features = _quantized_features(layer, features, tree)
    nodes = wrapper.awq_get_modules_for_scaling(layer, features, {})
    groups = _foldable_groups(layer, nodes)
    assert groups, f"{type(definition).__name__}: no foldable AWQ group was emitted"

    snapshot = {name: value.clone() for name, value in layer.state_dict().items()}
    checked = []
    try:
        for node in groups:
            source = get_op_name(layer, node["prev_op"])
            targets = [get_op_name(layer, module) for module in node["layers"]]
            if isinstance(node["prev_op"], nn.Linear):
                width = node["prev_op"].weight.shape[0]
            else:
                width = node["prev_op"].weight.numel()
            scales = torch.linspace(0.55, 1.45, width)
            consumers = list(node.get("_scale_consumers") or [])
            apply_scale(layer, [(source, targets, scales, None)], consumer_map={source: consumers})
            with torch.inference_mode():
                actual = forward()
            relative = float((actual - reference).norm() / reference.norm())
            assert relative < TOLERANCE, (
                f"{type(definition).__name__}: {source} -> {targets} changed the layer output "
                f"by {relative:.3e}"
            )
            checked.append((source, targets, consumers))
            layer.load_state_dict(snapshot)
    finally:
        layer.load_state_dict(snapshot)
    return checked


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


@pytest.fixture(scope="module")
def intern_s2_module():
    try:
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                "internlm/Intern-S2-Preview",
                revision=_INTERN_REVISION,
                allow_patterns=[
                    "configuration_interns2_preview.py",
                    "modeling_interns2_preview.py",
                ],
            )
        )
    except Exception as exc:  # noqa: BLE001 - optional remote reference code
        pytest.skip(f"Intern-S2 reference code unavailable: {exc}")

    package = types.ModuleType("intern_s2_reference")
    package.__path__ = [str(root)]
    sys.modules[package.__name__] = package
    for name in ("configuration_interns2_preview", "modeling_interns2_preview"):
        spec = importlib.util.spec_from_file_location(f"{package.__name__}.{name}", root / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{package.__name__}.modeling_interns2_preview"]


def test_intern_s2_preview_linear_attn_tags_preserve_layer_output(intern_s2_module):
    config = intern_s2_module.InternS2PreviewTextConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=2,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        layer_types=["linear_attention", "full_attention"],
    )
    config._attn_implementation = "eager"
    torch.manual_seed(7)
    # Build the real text model so `post_init`/`_init_weights` initialize the
    # DeltaNet `dt_bias`/`A_log` and the experts. Its `forward` targets a newer
    # Transformers mask signature, so the tests drive the layers directly.
    text = intern_s2_module.InternS2PreviewTextModel(config).eval()
    model = nn.Module()
    model.config = config
    model.model = nn.Module()
    model.model.language_model = text
    # The shipped checkpoint stores packed 3-D experts; the loader defuses them
    # into per-expert projections before quantization. Mirror that so the
    # routed-expert consumers share the norm's single scale group.
    from gptqmodel.models.loader import _convert_model_with_defuser

    _convert_model_with_defuser(MODEL_MAP["intern_s2_preview"], model, cleanup_original=False)
    layers = list(text.layers)

    hidden = torch.randn(1, 8, 32)
    position_ids = torch.arange(8).unsqueeze(0)
    cos = torch.randn(1, 8, 8)
    sin = torch.randn(1, 8, 8)

    checked = []
    for layer in layers:
        def forward(layer=layer):
            return layer(
                hidden,
                position_embeddings=(cos, sin),
                attention_mask=None,
                position_ids=position_ids,
            )

        checked.extend(_assert_neutral(MODEL_MAP["intern_s2_preview"], model, layer, forward))

    linear_groups = [
        (targets, consumers)
        for source, targets, consumers in checked
        if source == "input_layernorm" and "linear_attn.in_proj_qkv" in targets
    ]
    assert linear_groups, "the linear-attention input norm did not fold"
    targets, consumers = linear_groups[0]
    assert targets == ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"]
    assert {"linear_attn.in_proj_b", "linear_attn.in_proj_a"} <= set(consumers)


@pytest.mark.parametrize("model_type", ["minicpmv", "minicpmo"])
def test_minicpm_attention_and_mlp_stages_preserve_layer_output(model_type):
    import transformers

    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(7)
    llm = transformers.Qwen3ForCausalLM(config).eval()
    model = nn.Module()
    model.config = config
    model.llm = llm

    ids = torch.randint(2, 60, (1, 8))

    def forward():
        return llm(input_ids=ids).logits

    checked = []
    for layer in llm.model.layers:
        checked.extend(_assert_neutral(MODEL_MAP[model_type], model, layer, forward))

    flat = {(source, tuple(targets)) for source, targets, _consumers in checked}
    assert ("input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")) in flat
    assert ("self_attn.v_proj", ("self_attn.o_proj",)) in flat
    assert ("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj")) in flat
    assert ("mlp.up_proj", ("mlp.down_proj",)) in flat

    blocks = [
        [entry.split(":", 1)[0] for entry in block]
        for block in MODEL_MAP[model_type].simple_layer_modules(config, SimpleNamespace(dynamic=None))
    ]
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


def test_ricursive_value_residual_folds_are_neutral():
    try:
        from ricursive_test_utils import make_model
    except Exception as exc:  # noqa: BLE001 - optional reference custom code
        pytest.skip(f"Ripple reference code unavailable: {exc}")

    try:
        model = make_model()
    except Exception as exc:  # noqa: BLE001 - optional reference custom code
        pytest.skip(f"Ripple reference code unavailable: {exc}")

    definition = MODEL_MAP["ricursive"]
    ids = torch.randint(2, 60, (1, 8))

    def forward():
        return model(input_ids=ids).logits

    checked = []
    for layer in model.model.layers:
        checked.extend(_assert_neutral(definition, model, layer, forward))

    assert checked
    sources = {source for source, _targets, _consumers in checked}
    # `v_proj` reads a lerp of the value residual and the norm output, so the
    # input norm cannot be folded at all with `use_valres`.
    assert "input_layernorm" not in sources
    # The gated attention output and the lerped value are not fold sources.
    assert "self_attn.g_proj" not in sources
    assert "self_attn.v_proj" not in sources


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
