# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Real-model module-tree contract for Inkling 2 Mini (``inkling2_loop``).

The checkpoint stores routed experts as 3-D ``gate_up_proj`` / ``down_proj``
parameters, attention projections inside wrappers (``q_proj.linear`` plus
L2-clip children, a gated ``o_proj`` pair, short convolutions) and four decoder
layer classes: sliding/full attention, each with a looped and a non-looped
variant. The looped variants add q/k/v/mlp message gates.

The architecture is implemented in ``gptqmodel/models/ports/inkling2_loop``, so every
case here builds the real model class from a tiny config: nothing depends on a
Transformers release shipping an ``inkling2_loop`` implementation.
"""

from __future__ import annotations

import copy
import json
import math
import re
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from awq_neutrality import assert_awq_folds_neutral
from transformers.cache_utils import Cache, LinearAttentionLayer

from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.models._const import EXPERT_INDEX_PLACEHOLDER
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.base import BaseQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.looped_capture import LoopedLayerCaptureMixin
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.utils.model import find_modules, get_layers_with_prefixes


QC = SimpleNamespace(dynamic=None)

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
EXPECTED_LAYERS = "model.language_model.layers"

# Every decoder-layer variant this model instantiates. The tiny fixture is sized
# so that one layer of each class is present.
LAYER_CLASSES = (
    "Inkling2LoopSlidingDecoderLayer",
    "Inkling2LoopFullDecoderLayer",
    "Inkling2LoopSlidingLoopDecoderLayer",
    "Inkling2LoopFullLoopDecoderLayer",
)

ATTENTION_LEAVES = (
    ("self_attn.q_proj.linear", ("q",)),
    ("self_attn.k_proj.linear", ("k",)),
    ("self_attn.v_proj.linear", ("v",)),
    ("self_attn.o_proj.linear", ("o",)),
    ("self_attn.o_proj.gate_proj", ("gate",)),
)


def _inkling2_modules():
    """Return the clean-room Inkling2Loop configuration/modeling modules."""

    from gptqmodel.models.ports.inkling2_loop import configuration, modeling

    return configuration, modeling


def _tiny_config(configuration, **overrides):
    hidden = 32
    text_config = {
        "hidden_size": hidden,
        # 12 layers with the default `layer_types` schedule (full attention
        # every 6th layer) and a 6..12 loop core, so one layer of each of
        # the four decoder classes exists.
        "num_hidden_layers": 12,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "intermediate_size": 16,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 8,
        "vocab_size": 64,
        "unpadded_vocab_size": 64,
        "n_layers_per_attn_res_block": 3,
        "d_rel": 2,
        "rel_extent": 4,
        "sliding_window_size": 4,
        "loop_config": {"core_start": 6, "core_end": 12, "num_passes": 2, "gate_rank": 4},
    }
    text_config.update(overrides)
    return configuration.Inkling2LoopConfig(
        text_config=text_config,
        vision_config={
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "projection_hidden_dim": 16,
            "text_hidden_size": hidden,
        },
    )


def test_inkling2_loop_public_loader_uses_local_code_and_checkpoint_keys(tmp_path, monkeypatch):
    pytest.importorskip("torchvision")
    from PIL import Image
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping

    from gptqmodel import GPTQModel

    configuration, modeling = _inkling2_modules()
    config = _tiny_config(configuration)
    config.image_token_id = 4
    config.text_config.eos_token_id = 3
    config.vision_config.patch_size = 2
    config.auto_map = {"AutoConfig": "missing.BadConfig", "AutoModelForMultimodalLM": "missing.BadModel"}
    config.save_pretrained(tmp_path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "[BOS]": 2, "[EOS]": 3, "<|unused_200054|>": 4},
            unk_token="[UNK]",
        )),
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        additional_special_tokens=["<|unused_200054|>"],
    )
    tokenizer.save_pretrained(tmp_path)
    (tmp_path / "processor_config.json").write_text(json.dumps({
        "processor_class": "Inkling2Processor",
        "image_processor": {"image_processor_type": "Inkling2ImageProcessor", "size": {"height": 4, "width": 4}},
        "auto_map": {"AutoProcessor": "missing.BadProcessor"},
    }))
    source = modeling.Inkling2LoopForConditionalGeneration(config)
    checkpoint = {key: tensor.clone().contiguous() for key, tensor in source.state_dict().items()}
    for runtime, released in (
        ("model.language_model.embed_tokens.weight", "model.llm.embed.weight"),
        ("model.language_model.layers.1.attention_input.query", "model.llm.attn_res.attn_queries.0"),
    ):
        checkpoint[released] = checkpoint.pop(runtime)
    save_file(checkpoint, tmp_path / "model.safetensors", metadata={"format": "pt"})

    monkeypatch.setenv("GPTQ_BYPASS_SAFETY_CHECKS", "1")
    wrapper = GPTQModel.load(str(tmp_path), device="cpu", dtype=torch.float32, trust_remote_code=True)
    assert isinstance(wrapper.model, modeling.Inkling2LoopForConditionalGeneration)
    assert get_checkpoint_conversion_mapping("inkling2_loop") is None
    for name in ("model.language_model.embed_tokens.weight", "model.language_model.layers.1.attention_input.query"):
        torch.testing.assert_close(wrapper.model.state_dict()[name], source.state_dict()[name], rtol=0, atol=0)
    assert type(wrapper).awq_get_modules_for_scaling is not BaseQModel.awq_get_modules_for_scaling
    image_inputs = wrapper.processor.image_processor(Image.new("RGB", (7, 3)), return_tensors="pt")
    assert image_inputs["image_grid_thw"].tolist() == [[1, 1, 2]]
    assert wrapper.processor.replace_image_token(image_inputs, 0) == "<|unused_200054|>" * 2
    with torch.no_grad():
        output = wrapper.model(
            input_ids=torch.tensor([[2, 4, 4, 7]]), use_cache=False, **image_inputs,
        )
    assert output.logits.shape == (1, 4, config.text_config.unpadded_vocab_size)
    assert torch.isfinite(output.logits).all()


def test_inkling2_config_preserves_unused_kwargs_with_expert_manifest(tmp_path):
    configuration, _ = _inkling2_modules()
    config_dir = tmp_path / "submodel"
    _tiny_config(configuration).save_pretrained(config_dir)
    manifest = {
        "model": {
            "parameter_groups": [
                {
                    "output_dtype": "uint8",
                    "tensors": ["model.llm.layers.0.mlp.experts.w13_weight"],
                }
            ]
        },
    }
    (config_dir / configuration.QUANT_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")

    config_class = configuration.Inkling2LoopConfig
    loaded, unused = config_class.from_pretrained(
        tmp_path,
        subfolder="submodel",
        return_unused_kwargs=True,
        test_argument="retained",
    )
    plain = config_class.from_pretrained(tmp_path, subfolder="submodel")
    assert unused["test_argument"] == "retained"
    assert isinstance(loaded, config_class)
    assert loaded.text_config.experts_formats == ["nvfp4"] + [None] * 11
    assert plain.text_config.experts_formats == loaded.text_config.experts_formats


def test_inkling2_concrete_model_loads_without_explicit_config(tmp_path):
    configuration, modeling = _inkling2_modules()
    model_class = modeling.Inkling2LoopForConditionalGeneration
    model = model_class(_tiny_config(configuration))
    model.save_pretrained(tmp_path)

    loaded = model_class.from_pretrained(tmp_path)
    assert loaded.config.text_config.hidden_size == model.config.text_config.hidden_size
    torch.testing.assert_close(loaded.state_dict(), model.state_dict(), rtol=0, atol=0)


@torch.no_grad()
def test_inkling2_defused_dense_experts_keep_bf16_routing_order():
    configuration, modeling = _inkling2_modules()
    model = modeling.Inkling2LoopForConditionalGeneration(_tiny_config(configuration))
    experts = model.model.language_model.layers[0].mlp.experts.to(torch.bfloat16)
    half = experts.gate_up_proj.shape[1] // 2
    experts.gate_up_proj.zero_()
    experts.gate_up_proj[0, [0, 1, half, half + 1], 0] = torch.tensor([1, -1, 2, 4], dtype=torch.bfloat16)
    experts.down_proj.zero_()
    experts.down_proj[0, 0, :2] = torch.tensor([1, 0.5], dtype=torch.bfloat16)

    x = torch.zeros((1, model.config.text_config.hidden_size), dtype=torch.bfloat16)
    x[0, 0] = 1.25
    indices = torch.tensor([[0]])
    routing = torch.tensor([[0.4731]])
    gate = torch.tensor([1.25, -1.25])
    up = torch.tensor([2.5, 5.0])
    activation = (4 * (gate / 4).tanh() * gate.sigmoid() * (25 * (up / 25).tanh()) * routing).bfloat16()
    expected = torch.zeros_like(x)
    expected[0, 0] = activation[0, 0].float() + 0.5 * activation[0, 1].float()

    torch.testing.assert_close(experts(x, indices, routing), expected, rtol=0, atol=0)
    assert _convert_model_with_defuser(MODEL_MAP["inkling2_loop"], model, cleanup_original=True)
    assert isinstance(experts.get_submodule("0.down_proj"), torch.nn.Linear)
    torch.testing.assert_close(experts(x, indices, routing), expected, rtol=0, atol=0)


@pytest.fixture(scope="module")
def inkling2_case():
    configuration, modeling = _inkling2_modules()
    config = _tiny_config(configuration)
    definition = MODEL_MAP["inkling2_loop"]

    with torch.device("meta"):
        model = modeling.Inkling2LoopForConditionalGeneration(config)

    assert _convert_model_with_defuser(definition, model, cleanup_original=False), (
        "inkling2_loop: Defuser did not expand the routed experts; the installed "
        "Defuser registry has no `inkling2_loop` entry."
    )
    layers, layer_names = get_layers_with_prefixes(model, definition.extract_layers_node())
    assert layers, "inkling2_loop: no decoder layers resolved"

    # The real quantizer consumes `model.module_plan()` from a constructed
    # wrapper; the processor load is a loader concern and needs a checkpoint.
    previous = definition.require_load_processor
    definition.require_load_processor = False
    try:
        wrapper = definition(model=model, quantized=False, quantize_config=None)
    finally:
        definition.require_load_processor = previous

    return SimpleNamespace(
        config=config,
        definition=definition,
        model=model,
        wrapper=wrapper,
        layers=layers,
        layer_names=layer_names,
        plan=wrapper.module_plan(),
    )


def _quantizable_specs(definition, config) -> tuple[str, ...]:
    blocks = definition.simple_layer_modules(model_config=config, quantize_config=QC)
    return tuple(sorted({name.split(":", 1)[0] for block in blocks for name in block}))


def _matches(template: str, name: str) -> bool:
    pattern = "^" + re.escape(template).replace(re.escape(EXPERT_INDEX_PLACEHOLDER), r"\d+") + "$"
    return re.match(pattern, name) is not None


def _weight_owners(layer) -> set[str]:
    """Return modules that own checkpoint state entries (weights/buffers)."""

    modules = {name for name, _ in layer.named_modules() if name}
    owners = set()
    for weight_path in layer.state_dict():
        owner = max(
            (
                name
                for name in modules
                if weight_path.startswith(f"{name}.")
                and "." not in weight_path[len(name) + 1:]
            ),
            key=len,
            default="",
        )
        assert owner, f"state entry {weight_path!r} is not owned by a module"
        owners.add(owner)
    return owners


def _plan_looper(plan: ModulePlan) -> ModuleLooper:
    """A looper whose only model capability is the compiled plan."""

    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = SimpleNamespace(module_plan=lambda: plan)
    return looper


def _looper(wrapper) -> ModuleLooper:
    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = wrapper
    looper.layer_module_coverage = SimpleNamespace(record_hit=lambda *a, **k: None, record_miss=lambda *a, **k: None)
    looper.input_embeddings_name = None
    looper.output_embeddings_name = None
    return looper


def test_inkling2_loop_tree_declares_attention_and_routed_experts():
    definition = MODEL_MAP["inkling2_loop"]
    plan = ModulePlan.from_tree(definition.module_tree)

    assert definition.extract_layers_node() == [EXPECTED_LAYERS]
    assert definition.layer_modules_strict is False
    assert definition.dynamic_expert_index == "n_routed_experts"
    assert definition.pre_lm_head_norm_module == "model.language_model.norm"

    for path, roles in ATTENTION_LEAVES:
        spec = plan.find(path)
        assert spec is not None, path
        assert spec.quantize, path
        assert spec.roles == roles, path

    # The gated output projection must not share the attention-output subset.
    assert plan.find("self_attn.o_proj.linear").flags == ("1", "o")
    assert plan.find("self_attn.o_proj.gate_proj").flags == ("2", "gate")

    assert tuple(spec.template for spec in plan.moe_roots()) == ("mlp",)
    assert plan.expert_container_leaf("mlp") == "experts"
    assert plan.shared_container_leaf("mlp") is None

    router = plan.find("mlp.gate")
    assert router is not None and not router.quantize

    for projection, role in zip(PROJECTIONS, ("gate", "up", "down")):
        spec = plan.find(f"mlp.experts.0.{projection}")
        assert spec is not None and spec.quantize
        assert spec.roles == (role,)

    assert plan.is_expert("mlp.experts.1.down_proj")
    assert plan.expert_root("mlp.experts.1.down_proj") == "mlp.experts.1"
    assert plan.same_moe_family("mlp.experts.1.gate_proj", "mlp.experts.1.down_proj")
    assert not plan.same_moe_family("mlp.experts.0.gate_proj", "mlp.experts.1.down_proj")

    # Attention plumbing stays dense but declared.
    for path in (
        "self_attn.q_proj.norm",
        "self_attn.k_proj.sconv",
        "self_attn.k_proj.sconv.conv1d",
        "self_attn.k_proj.sconv.weight_proj",
        "self_attn.v_proj.sconv",
        "self_attn.sink_proj",
        "self_attn.relative_position_bias",
        "self_attn.relative_position_bias.linear",
        "self_attn.relative_position_bias.norm",
        "post_attention_convolution.conv1d",
        "post_mlp_convolution.conv1d",
        "q_message_gate.down_proj",
        "k_message_gate.forget_gate_proj",
        "v_message_gate.message_gate_proj",
        "mlp_message_gate.down_proj",
    ):
        spec = plan.find(path)
        assert spec is not None, path
        assert not spec.quantize, path


def test_inkling2_loop_blocks_group_the_projection_subsets():
    definition = MODEL_MAP["inkling2_loop"]
    blocks = definition.build_layer_modules(definition.module_tree, include_capture_only=True)

    assert ["self_attn.q_proj.linear", "self_attn.k_proj.linear", "self_attn.v_proj.linear"] in [
        block[:3] for block in blocks
    ]
    assert ["self_attn.o_proj.linear"] in blocks
    assert ["self_attn.o_proj.gate_proj"] in blocks
    assert ["mlp.experts.{expert_index}.gate_proj", "mlp.experts.{expert_index}.up_proj"] in blocks
    assert ["mlp.experts.{expert_index}.down_proj"] in blocks
    # No duplicated or nested path may leak into a block.
    assert not [name for block in blocks for name in block if name.count("gate_proj") > 1]


def test_inkling2_loop_defuses_every_routed_expert(inkling2_case):
    experts = inkling2_case.config.text_config.n_routed_experts
    for layer in inkling2_case.layers:
        block = layer.mlp.experts
        assert [name for name, _ in block.named_parameters() if "." not in name] == [], (
            "fused 3-D expert parameters survived Defuser"
        )
        for index in range(experts):
            container = getattr(block, str(index))
            assert [name for name, _ in container.named_parameters()] == [
                "down_proj.weight",
                "gate_proj.weight",
                "up_proj.weight",
            ]
            for projection in PROJECTIONS:
                spec = inkling2_case.plan.find(f"mlp.experts.{index}.{projection}")
                assert spec is not None and spec.quantize


def test_inkling2_loop_rejects_an_unconverted_model(inkling2_case):
    """A Defuser without the `inkling2_loop` entry must fail loudly, not silently."""

    _, modeling = _inkling2_modules()
    with torch.device("meta"):
        raw = modeling.Inkling2LoopForConditionalGeneration(inkling2_case.config)

    with pytest.raises(RuntimeError, match="not defused into per-expert Linear modules"):
        inkling2_case.definition.after_defuser_conversion(raw)


def test_inkling2_loop_quantizable_specs_resolve_on_every_layer_variant(inkling2_case):
    specs = _quantizable_specs(inkling2_case.definition, inkling2_case.config)
    assert specs, "inkling2_loop: definition emits no quantizable specs"

    seen = {}
    for layer in inkling2_case.layers:
        seen[type(layer).__name__] = layer
        resolved = set(find_modules(layer))
        missing = [spec for spec in specs if not any(_matches(spec, name) for name in resolved)]
        assert not missing, f"{type(layer).__name__}: unresolved quantizable specs {missing}"

    assert set(seen) == set(LAYER_CLASSES), f"tiny fixture must cover every layer class, got {sorted(seen)}"

    # Loop layers add message gates; the plan keeps them dense on every variant.
    loop_layer = seen["Inkling2LoopSlidingLoopDecoderLayer"]
    for gate in ("q_message_gate", "k_message_gate", "v_message_gate", "mlp_message_gate"):
        module = getattr(loop_layer, gate)
        assert module.down_proj is not None
        assert inkling2_case.plan.find(f"{gate}.down_proj").quantize is False


def test_inkling2_loop_checkpoint_owners_carry_plan_tags(inkling2_case):
    specs = _quantizable_specs(inkling2_case.definition, inkling2_case.config)
    patterns = [
        re.compile("^" + re.escape(spec).replace(re.escape(EXPERT_INDEX_PLACEHOLDER), r"\d+") + "$")
        for spec in specs
    ]

    owners = set()
    for layer in inkling2_case.layers:
        owners.update(_weight_owners(layer))
    assert owners

    for owner in owners:
        spec = inkling2_case.plan.find(owner)
        assert spec is not None, f"checkpoint weights owned by undeclared module {owner!r}"
        if not any(pattern.match(owner) for pattern in patterns):
            assert not spec.quantize, f"non-quantized owner {owner!r} must carry an explicit `:!` tag"

    # Representative checkpoint-only tensors the definition must keep dense.
    for owner in (
        "self_attn.relative_position_bias",
        "self_attn.q_proj.norm",
        "attention_input",
        "mlp.gate",
        "input_layernorm",
    ):
        assert owner in owners
        assert inkling2_case.plan.find(owner).quantize is False


def test_inkling2_loop_looper_and_moe_hooks_use_the_plan(inkling2_case):
    definition = inkling2_case.definition
    plan = inkling2_case.plan
    looper = _looper(inkling2_case.wrapper)

    assert looper._is_attention_module_name("self_attn.q_proj.linear")
    assert looper._is_attention_module_name("self_attn.v_proj.linear")
    assert not looper._is_attention_module_name("self_attn.o_proj.linear")
    assert looper._extract_moe_group_key("mlp.experts.1.gate_proj") == "mlp.experts.1"
    assert looper._extract_moe_group_key("mlp.gate") is None

    blocks = definition.simple_layer_modules(model_config=inkling2_case.config, quantize_config=QC)
    expert_blocks = [block for block in blocks if any(".experts." in name for name in block)]
    assert expert_blocks

    layer = inkling2_case.layers[0]
    full = find_modules(layer)
    selected = set()
    for block in expert_blocks:
        subset = looper.create_named_modules(
            layer,
            full,
            False,
            0,
            inkling2_case.layer_names[0],
            block,
            processor=SimpleNamespace(preprocess=lambda *a, **k: None, is_skipped=lambda *a, **k: False),
            fallback=None,
        )
        selected.update(subset)
    assert {name for name in _weight_owners(layer) if ".experts." in name} <= selected

    hooks = copy.copy(definition.moe_lifecycle_hooks)
    assert hooks is not None
    hooks.module_plan = plan

    moe_block = hooks.get_moe_block(layer, definition)
    assert moe_block is not None
    assert hooks.get_experts_module_name(moe_block) == "experts"
    assert hooks.get_experts_module(moe_block, definition) is not None
    assert hooks.get_shared_experts_module(moe_block, definition) is None


def test_inkling2_loop_rejects_undeclared_and_similar_paths():
    definition = MODEL_MAP["inkling2_loop"]
    plan = ModulePlan.from_tree(definition.module_tree)
    looper = _plan_looper(plan)

    for path in (
        "self_attn.q_proj",
        "self_attn.q_proj.linear.undeclared",
        "self_attn.o_proj.undeclared",
        "self_attn.sink_proj.undeclared",
        "mlp.experts.0.gate_proj.undeclared",
        "mlp.gate.undeclared",
        "mlp.experts.undeclared",
        "post_attention_convolution.undeclared",
        "q_message_gate.undeclared",
    ):
        assert plan.find(path) is None, path
        assert not plan.is_expert(path), path
        assert looper._extract_moe_group_key(path) is None, path

    assert not plan.is_moe("self_attn.q_proj.linear")
    assert plan.is_moe("mlp.experts.1.gate_proj")


def test_inkling2_loop_wrapper_compiles_the_same_plan(inkling2_case):
    definition = inkling2_case.definition
    plan = inkling2_case.plan

    assert {spec.template for spec in plan.specs} == {
        spec.template for spec in ModulePlan.from_tree(definition.module_tree).specs
    }
    assert plan.roles("self_attn.q_proj.linear") == ("q",)
    assert plan.summary()["moe_roots"] == ["mlp"]
    assert plan.summary()["expert_containers"] == ["mlp.experts"]

    language_model = inkling2_case.model.model.language_model
    # The pre-lm-head norm and every module the generate hooks materialize must
    # exist on the real model.
    assert getattr(language_model, definition.pre_lm_head_norm_module.rsplit(".", 1)[-1], None) is not None
    for module in ("embed_tokens", "final_input", "message_input"):
        assert getattr(language_model, module, None) is not None
    assert getattr(inkling2_case.model.model, "vision_tower", None) is not None


@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_inkling2_short_convolution_is_causal_and_chunk_independent(dynamic, dtype):
    _, modeling = _inkling2_modules()
    x = torch.arange(1, 17, dtype=dtype).view(1, 8, 2) / 16
    weights = torch.tensor([[[1.0, 0.5, -0.25, 2.0]], [[-0.5, 1.0, 3.0, -2.0]]])
    if dynamic:
        conv = modeling.Inkling2LoopDynamicShortConvolution(2, 2, 1, 4, 0, 0)
        conv.static_weight.data.copy_(weights.flip(-1).squeeze(1).T)
        conv.weight_proj.to(dtype=dtype)
        conv.weight_proj.weight.data.copy_(torch.tensor([[.125, 0], [0, .25], [0, -.125], [.25, 0]]))
    else:
        conv = modeling.Inkling2LoopShortConvolution(2, 4, 0, 0)
        conv.conv1d.weight.data.copy_(weights)

    with torch.no_grad():
        full = conv(x, dynamic_input=x)
        prefix = conv(x[:, :3], dynamic_input=x[:, :3])
        assert full.dtype == x.dtype
        torch.testing.assert_close(prefix, full[:, :3], atol=1e-6, rtol=1e-6)

        if not dynamic:
            # Unpadded depthwise correlation on explicitly left-padded history.
            oracle = F.conv1d(F.pad(x.float().transpose(1, 2), (3, 0)), weights, groups=2)
            expected = ((oracle.transpose(1, 2) + x.float()) / math.sqrt(5)).to(dtype)
            torch.testing.assert_close(full, expected, atol=1e-6, rtol=1e-6)

        for record_past in (False, True):
            cache = Cache(layers=[LinearAttentionLayer()])
            if record_past:
                cache.activate_past_recording()
            chunks = [
                conv(x[:, start:end], dynamic_input=x[:, start:end], cache=cache)
                for start, end in ((0, 3), (3, 5), (5, 6), (6, 8))
            ]
            torch.testing.assert_close(torch.cat(chunks, dim=1), full, atol=1e-6, rtol=1e-6)


def test_inkling2_loop_logits_are_prefix_invariant():
    configuration, modeling = _inkling2_modules()
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(7)
        model = modeling.Inkling2LoopForConditionalGeneration(_tiny_config(configuration)).eval()
        input_ids = torch.tensor([[7, 11, 13, 17, 19, 23, 29]])
        full = model(input_ids=input_ids, use_cache=False).logits
        prefix = model(input_ids=input_ids[:, :3], use_cache=False).logits
    assert torch.isfinite(full).all()
    torch.testing.assert_close(full[:, :3], prefix, atol=1e-5, rtol=2e-5)


def test_inkling2_loop_replay_rebuilds_block_states_and_masks():
    """Decoder replay must reproduce the model's attention-residual bank.

    Every ``Inkling2Loop`` decoder layer takes ``block_states`` (the bank of
    block-boundary inputs) as its second positional argument, and the generic
    capture keeps only ``hidden_states``. The definition's replay hooks must
    rebuild that bank, refresh the per-layer sliding/full mask, and produce the
    same layer output as the model's own forward.
    """

    configuration, modeling = _inkling2_modules()
    definition = MODEL_MAP["inkling2_loop"]
    torch.manual_seed(0)
    model = modeling.Inkling2LoopForConditionalGeneration(_tiny_config(configuration)).float().eval()
    assert _convert_model_with_defuser(definition, model, cleanup_original=False)

    layers, _ = get_layers_with_prefixes(model, definition.extract_layers_node())
    assert layers

    # Reference pass: record each layer's exact inputs and first-pass output.
    records = {}

    def capture(module, args, kwargs):
        records.setdefault(
            module,
            {
                "args": [arg.detach().clone() for arg in args if torch.is_tensor(arg)],
                "kwargs": {
                    key: value.detach().clone() if torch.is_tensor(value) else value
                    for key, value in kwargs.items()
                },
                "output": None,
            },
        )

    def capture_output(module, args, kwargs, output):
        if records[module]["output"] is None:
            records[module]["output"] = output.detach().clone()

    handles = [layer.register_forward_pre_hook(capture, with_kwargs=True) for layer in layers]
    handles += [layer.register_forward_hook(capture_output, with_kwargs=True) for layer in layers]
    input_ids = torch.randint(0, 64, (1, 24))
    try:
        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    assert set(records) == set(layers)

    previous = definition.require_load_processor
    definition.require_load_processor = False
    try:
        wrapper = definition(model=model, quantized=False, quantize_config=None)
    finally:
        definition.require_load_processor = previous

    first = records[layers[0]]
    layer_input_kwargs = {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in first["kwargs"].items()
        if key not in ("hidden_states", "attention_mask", "position_ids")
    }
    layer_input_kwargs = wrapper.capture_first_layer_input_kwargs(
        args=tuple(first["args"]),
        kwargs=first["kwargs"],
        batch_device=torch.device("cpu"),
        layer_input_kwargs=layer_input_kwargs,
    )

    for position, layer in enumerate(layers):
        record = records[layer]
        additional_inputs = dict(layer_input_kwargs)
        additional_inputs["attention_mask"] = first["kwargs"]["attention_mask"]
        additional_inputs["position_ids"] = first["kwargs"]["position_ids"]
        additional_inputs["use_cache"] = False
        additional_inputs = wrapper.prepare_layer_replay_kwargs(
            layer=layer,
            layer_input=[record["args"][0]],
            additional_inputs=additional_inputs,
            target_device=torch.device("cpu"),
        )

        block_states = additional_inputs.pop("block_states")
        reference_block_states = record["args"][1]
        assert block_states.shape == reference_block_states.shape, (position, block_states.shape)
        if block_states.numel():
            assert torch.equal(block_states, reference_block_states), position
        else:
            assert reference_block_states.numel() == 0, position

        rebuilt_mask = additional_inputs["attention_mask"]
        assert torch.equal(rebuilt_mask, record["kwargs"]["attention_mask"]), position

        with torch.no_grad():
            output = layer(record["args"][0], block_states=block_states, **additional_inputs)
        assert torch.allclose(output, record["output"], atol=1e-6, rtol=0), position


def test_inkling2_loop_replay_serves_every_loop_invocation():
    """#6106: post-loop layers must replay on the final-pass output.

    The model runs the loop core ``loop_num_passes`` times, resets to the
    pre-loop state before passes 2..N, injects the message gates, and only the
    final pass feeds layers after ``core_end``. The default first-layer capture
    replays one 0..N chain, so a post-loop layer would see the pass-1 output and
    a loop-core layer only one of its three input distributions. The looped
    full-forward capture must record every invocation with its own residual
    bank and messages.
    """

    configuration, modeling = _inkling2_modules()
    definition = MODEL_MAP["inkling2_loop"]
    assert issubclass(definition, LoopedLayerCaptureMixin), (
        "inkling2_loop must opt into LoopedLayerCaptureMixin; the default "
        "first-layer capture can only replay the pass-1 chain"
    )
    config = _tiny_config(
        configuration,
        num_hidden_layers=15,
        loop_config={"core_start": 6, "core_end": 12, "num_passes": 3, "gate_rank": 4},
    )
    torch.manual_seed(0)
    model = modeling.Inkling2LoopForConditionalGeneration(config).float().eval()
    assert _convert_model_with_defuser(definition, model, cleanup_original=False)
    layers, layer_names = get_layers_with_prefixes(model, definition.extract_layers_node())

    records = {id(layer): [] for layer in layers}

    def capture(module, args, kwargs):
        records[id(module)].append(
            {
                "hidden": args[0].detach().clone(),
                "bank": args[1].detach().clone()
                if len(args) > 1 and torch.is_tensor(args[1])
                else None,
                "messages": args[2].detach().clone()
                if len(args) > 2 and torch.is_tensor(args[2])
                else None,
                "output": None,
            }
        )

    def capture_output(module, args, kwargs, output):
        records[id(module)][-1]["output"] = output.detach().clone()

    handles = [layer.register_forward_pre_hook(capture, with_kwargs=True) for layer in layers]
    handles += [layer.register_forward_hook(capture_output, with_kwargs=True) for layer in layers]
    example = {
        "input_ids": torch.randint(0, 64, (1, 12)),
        "attention_mask": torch.ones(1, 12, dtype=torch.long),
    }
    try:
        with torch.no_grad():
            model(**example, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    expected_invocations = [1] * 6 + [3] * 6 + [1] * 3
    assert [len(records[id(layer)]) for layer in layers] == expected_invocations

    previous = definition.require_load_processor
    definition.require_load_processor = False
    try:
        wrapper = definition(model=model, quantized=False, quantize_config=None)
    finally:
        definition.require_load_processor = previous
    wrapper.quantize_config = SimpleNamespace(
        device=torch.device("cpu"),
        activation_mode=None,
        activation_recipe=None,
        offload_to_disk=False,
        compute_device_filter=None,
    )

    cache = wrapper.build_quantization_input_cache(
        layers=layers,
        calibration_data=[example],
        use_cache=False,
        layer_names=layer_names,
    )
    per_layer = wrapper._looped_per_layer
    assert [len(per_layer[id(layer)]["layer_inputs"]) for layer in layers] == expected_invocations
    assert cache.captured_batch_count == 3

    # The first post-loop layer consumes the pass-3 output of layer 11, not the
    # pass-1 output the old single-chain replay fed it.
    first_post_loop = 12
    final_input = per_layer[id(layers[first_post_loop])]["layer_inputs"][0][0]
    torch.testing.assert_close(final_input, records[id(layers[first_post_loop])][0]["hidden"])
    pass1_output = records[id(layers[first_post_loop - 1])][0]["output"]
    relative = (final_input - pass1_output).norm() / pass1_output.norm()
    assert relative > 1e-3, "the post-loop input must differ from the pass-1 chain"

    for layer in layers:
        bucket = per_layer[id(layer)]
        for invocation, (layer_input, replay_kwargs, attention_mask, position_ids) in enumerate(
            zip(
                bucket["layer_inputs"],
                bucket["layer_input_kwargs"],
                bucket["attention_masks"],
                bucket["position_ids"],
            )
        ):
            record = records[id(layer)][invocation]
            additional_inputs = dict(replay_kwargs)
            additional_inputs["attention_mask"] = attention_mask
            additional_inputs["position_ids"] = position_ids
            additional_inputs["use_cache"] = False
            additional_inputs = wrapper.prepare_layer_replay_kwargs(
                layer=layer,
                layer_input=list(layer_input),
                additional_inputs=additional_inputs,
                target_device=torch.device("cpu"),
            )
            torch.testing.assert_close(additional_inputs["block_states"], record["bank"])
            if record["messages"] is None:
                assert "messages" not in additional_inputs
            else:
                torch.testing.assert_close(
                    additional_inputs["messages"], record["messages"]
                )
            with torch.no_grad():
                output = layer(layer_input[0], **additional_inputs)
            torch.testing.assert_close(output, record["output"], atol=1e-6, rtol=0)


def test_inkling2_loop_looped_capture_populates_lm_head_banks():
    """#6106 must not regress the #6107 head path.

    The looped full-forward capture supplies each layer invocation's own
    ``block_states``. ``prepare_layer_replay_kwargs`` must still record the
    final post-layer bank so ``lm_head_pre_quantize_generate_hook`` can apply
    ``final_input`` before the norm instead of falling back to ``norm(h_last)``.
    """

    configuration, modeling = _inkling2_modules()
    definition = MODEL_MAP["inkling2_loop"]
    config = _tiny_config(
        configuration,
        num_hidden_layers=15,
        loop_config={"core_start": 6, "core_end": 12, "num_passes": 3, "gate_rank": 4},
    )
    torch.manual_seed(0)
    model = modeling.Inkling2LoopForConditionalGeneration(config).float().eval()
    assert _convert_model_with_defuser(definition, model, cleanup_original=False)
    layers, layer_names = get_layers_with_prefixes(model, definition.extract_layers_node())
    language_model = model.model.language_model

    records = {id(layer): [] for layer in layers}
    final_args = []

    def capture(module, args, kwargs):
        records[id(module)].append(
            {
                "hidden": args[0].detach().clone(),
                "bank": args[1].detach().clone()
                if len(args) > 1 and torch.is_tensor(args[1])
                else None,
                "output": None,
            }
        )

    def capture_output(module, args, kwargs, output):
        records[id(module)][-1]["output"] = output.detach().clone()

    def capture_final_input(module, args):
        final_args.append(
            {
                "bank": args[0].detach().clone(),
                "hidden": args[1].detach().clone(),
            }
        )

    handles = [layer.register_forward_pre_hook(capture, with_kwargs=True) for layer in layers]
    handles += [layer.register_forward_hook(capture_output, with_kwargs=True) for layer in layers]
    handles.append(language_model.final_input.register_forward_pre_hook(capture_final_input))
    examples = [
        {
            "input_ids": torch.randint(0, 64, (1, 12)),
            "attention_mask": torch.ones(1, 12, dtype=torch.long),
        },
        {
            "input_ids": torch.randint(0, 64, (1, 9)),
            "attention_mask": torch.ones(1, 9, dtype=torch.long),
        },
    ]
    try:
        with torch.no_grad():
            for example in examples:
                model(**example, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    previous = definition.require_load_processor
    definition.require_load_processor = False
    try:
        wrapper = definition(model=model, quantized=False, quantize_config=None)
    finally:
        definition.require_load_processor = previous
    wrapper.quantize_config = SimpleNamespace(
        device=torch.device("cpu"),
        activation_mode=None,
        activation_recipe=None,
        offload_to_disk=False,
        compute_device_filter=None,
    )

    wrapper.build_quantization_input_cache(
        layers=layers,
        calibration_data=examples,
        use_cache=False,
        layer_names=layer_names,
    )
    final_layer = layers[-1]
    bucket = wrapper._looped_per_layer[id(final_layer)]
    assert len(bucket["layer_inputs"]) == len(examples)
    for batch_index in range(len(examples)):
        additional_inputs = dict(bucket["layer_input_kwargs"][batch_index])
        additional_inputs["attention_mask"] = bucket["attention_masks"][batch_index]
        additional_inputs["position_ids"] = bucket["position_ids"][batch_index]
        additional_inputs["use_cache"] = False
        wrapper.prepare_layer_replay_kwargs(
            layer=final_layer,
            layer_input=list(bucket["layer_inputs"][batch_index]),
            additional_inputs=additional_inputs,
            target_device=torch.device("cpu"),
        )

    banks = wrapper._inkling2_loop_final_block_banks
    assert sorted(banks) == list(range(len(examples)))
    for batch_index, batch_final in enumerate(final_args):
        torch.testing.assert_close(banks[batch_index], batch_final["bank"])

    replayed = wrapper.lm_head_pre_quantize_generate_hook(
        [
            [records[id(final_layer)][batch_index]["output"]]
            for batch_index in range(len(examples))
        ]
    )
    for batch_index, batch_final in enumerate(final_args):
        expected = language_model.norm(
            language_model.final_input(batch_final["bank"], batch_final["hidden"])[0]
        )
        torch.testing.assert_close(
            replayed[batch_index][0], expected, atol=1e-6, rtol=0
        )


@pytest.fixture(scope="module")
def inkling2_awq_case():
    """A float32 tiny model plus each layer's replay inputs from one forward."""

    configuration, modeling = _inkling2_modules()
    definition = MODEL_MAP["inkling2_loop"]
    torch.manual_seed(0)
    model = modeling.Inkling2LoopForConditionalGeneration(_tiny_config(configuration)).float().eval()
    assert _convert_model_with_defuser(definition, model, cleanup_original=False)
    layers, _ = get_layers_with_prefixes(model, definition.extract_layers_node())

    captured = {}

    def capture(module, args, kwargs):
        captured[module] = (args, dict(kwargs))

    handles = [layer.register_forward_pre_hook(capture, with_kwargs=True) for layer in layers]
    input_ids = torch.randint(0, 64, (1, 24))
    try:
        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    assert set(captured) == set(layers)
    return SimpleNamespace(
        definition=definition,
        model=model,
        layers=layers,
        captured=captured,
    )


EXPERT_UP_DOWN_FOLDS = [
    ("mlp.experts.0.up_proj", ["mlp.experts.0.down_proj"], []),
    ("mlp.experts.1.up_proj", ["mlp.experts.1.down_proj"], []),
]


@pytest.mark.parametrize(
    "layer_index",
    [0, 2, 7, 8],
    ids=["sliding", "full", "sliding-loop", "full-loop"],
)
def test_inkling2_loop_awq_folds_only_per_expert_up_down(inkling2_awq_case, layer_index):
    """Tree order is not data flow: only the routed activation fold survives.

    The generic planner folds ``post_attention_layernorm`` into q/k/v (which
    read the *input* norm) and ``o_proj.linear`` into the attention-input
    ``o_proj.gate_proj``. Both change the layer output. The per-expert
    ``up_proj -> down_proj`` fold multiplies ``up_proj``'s output straight
    into ``down_proj`` (apart from the bounded ``up`` softcap), so it is
    output-preserving at this scale.
    """

    layer = inkling2_awq_case.layers[layer_index]
    args, kwargs = inkling2_awq_case.captured[layer]
    checked = assert_awq_folds_neutral(
        inkling2_awq_case.definition,
        inkling2_awq_case.model,
        layer,
        lambda: layer(*args, **kwargs),
    )
    assert checked == EXPERT_UP_DOWN_FOLDS


@pytest.mark.parametrize(
    "source,expected",
    [
        (
            "post_attention_layernorm",
            r"post_attention_layernorm -> \[.*\] changed the layer output by",
        ),
        (
            "o_proj",
            r"self_attn\.o_proj\.linear -> \['self_attn\.o_proj\.gate_proj'\] changed the layer output by",
        ),
    ],
    ids=["attention-input-norm", "attention-output"],
)
def test_inkling2_loop_awq_dropped_attention_folds_are_not_neutral(
    inkling2_awq_case, source, expected
):
    """Pin why the definition drops the two generic attention groups."""

    layer = inkling2_awq_case.layers[0]
    args, kwargs = inkling2_awq_case.captured[layer]

    def generic(wrapper, module, input_feat, module_kwargs):
        nodes = BaseQModel.awq_get_modules_for_scaling(wrapper, module, input_feat, module_kwargs)
        if source == "post_attention_layernorm":
            return [node for node in nodes if node["prev_op"] is module.post_attention_layernorm]
        return [node for node in nodes if node["prev_op"] is module.self_attn.o_proj.linear]

    with pytest.raises(AssertionError, match=expected):
        assert_awq_folds_neutral(
            inkling2_awq_case.definition,
            inkling2_awq_case.model,
            layer,
            lambda: layer(*args, **kwargs),
            scaling_provider=generic,
        )


def _capture_inkling2_forward(model, layers, input_ids):
    """Capture one calibration batch's layer inputs and the real lm_head input."""

    records = {}
    head_input = {}

    def capture(module, args, kwargs):
        records.setdefault(
            module,
            {
                "args": [arg.detach().clone() for arg in args if torch.is_tensor(arg)],
                "kwargs": {
                    key: value.detach().clone() if torch.is_tensor(value) else value
                    for key, value in kwargs.items()
                },
            },
        )

    def capture_output(module, args, kwargs, output):
        # Looped layers run more than once. Keep the first pass, which is the
        # pass the quantizer replays, instead of overwriting it.
        records[module].setdefault("output", output.detach().clone())

    def capture_head(module, args):
        head_input["input"] = args[0].detach().clone()

    handles = [layer.register_forward_pre_hook(capture, with_kwargs=True) for layer in layers]
    handles += [layer.register_forward_hook(capture_output, with_kwargs=True) for layer in layers]
    handles.append(model.lm_head.register_forward_pre_hook(capture_head))
    try:
        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    assert set(records) == set(layers)
    return records, head_input["input"]


@pytest.fixture(scope="module")
def inkling2_lm_head_case():
    configuration, modeling = _inkling2_modules()
    definition = MODEL_MAP["inkling2_loop"]
    torch.manual_seed(0)
    model = modeling.Inkling2LoopForConditionalGeneration(
        _tiny_config(configuration)
    ).float().eval()
    assert _convert_model_with_defuser(definition, model, cleanup_original=False)
    layers, _ = get_layers_with_prefixes(model, definition.extract_layers_node())

    previous = definition.require_load_processor
    definition.require_load_processor = False
    try:
        wrapper = definition(model=model, quantized=False, quantize_config=None)
    finally:
        definition.require_load_processor = previous
    wrapper.quantize_config = SimpleNamespace(
        device=torch.device("cpu"),
        activation_mode=None,
        activation_recipe=None,
        offload_to_disk=False,
        compute_device_filter=None,
    )
    return SimpleNamespace(model=model, wrapper=wrapper, layers=layers)


def test_inkling2_loop_lm_head_replay_uses_final_input_and_per_batch_bank(
    inkling2_lm_head_case,
):
    """The head must be calibrated on ``norm(final_input(bank, h_last))``.

    The generic hook applies only ``norm`` to the last decoder output. Inkling 2
    first retrieves the final hidden state from the attention-residual bank, so
    the buggy path feeds the head a different tensor (measured cosine ~0.44 on
    this fixture). Two batches with different sequence lengths also prove the
    bank is selected per batch rather than shared.
    """

    model = inkling2_lm_head_case.model
    wrapper = inkling2_lm_head_case.wrapper
    layers = inkling2_lm_head_case.layers
    wrapper.pre_quantize_generate_hook_start()

    batches = []
    for input_ids in (
        torch.tensor([[7, 11, 13, 17, 19, 23, 29, 31]]),
        torch.tensor([[2, 3, 5, 37, 41]]),
    ):
        records, head_input = _capture_inkling2_forward(model, layers, input_ids)
        first = records[layers[0]]
        replay_kwargs = {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in first["kwargs"].items()
            if key not in ("hidden_states", "attention_mask", "position_ids")
        }
        replay_kwargs = wrapper.capture_first_layer_input_kwargs(
            args=tuple(first["args"]),
            kwargs=first["kwargs"],
            batch_device=torch.device("cpu"),
            layer_input_kwargs=replay_kwargs,
        )
        batches.append(
            {
                "records": records,
                "head_input": head_input,
                "replay_kwargs": replay_kwargs,
            }
        )

    # The layer stage replays every layer for every batch before advancing, so
    # build the banks layer-major and batch-minor.
    for layer in layers:
        for batch in batches:
            records = batch["records"]
            first = records[layers[0]]
            additional_inputs = dict(batch["replay_kwargs"])
            additional_inputs["attention_mask"] = first["kwargs"]["attention_mask"]
            additional_inputs["position_ids"] = first["kwargs"]["position_ids"]
            additional_inputs["use_cache"] = False
            wrapper.prepare_layer_replay_kwargs(
                layer=layer,
                layer_input=[records[layer]["args"][0]],
                additional_inputs=additional_inputs,
                target_device=torch.device("cpu"),
            )

    language_model = model.model.language_model
    head_inputs = []
    expected = []
    for batch in batches:
        h_last = batch["records"][layers[-1]]["output"]
        bank = batch["replay_kwargs"]["_inkling2_loop_block_states_carry"].states
        expected.append(
            language_model.norm(language_model.final_input(bank, h_last)[0]).to(
                h_last.dtype
            )
        )
        head_inputs.append(h_last.clone())

    replayed = wrapper.lm_head_pre_quantize_generate_hook(
        [[value] for value in head_inputs]
    )

    assert len(replayed) == len(expected)
    for batch_index, (output, want) in enumerate(zip(replayed, expected)):
        torch.testing.assert_close(
            output[0],
            want,
            atol=1e-6,
            rtol=0,
            msg=f"lm_head calibration batch {batch_index}",
        )

    # The regression is the missing ``final_input`` hop, not the exact values.
    for batch, output in zip(batches, replayed):
        norm_only = language_model.norm(batch["records"][layers[-1]]["output"])
        assert not torch.allclose(output[0], norm_only, atol=1e-5), (
            "lm_head calibration input skipped final_input"
        )
        # The replay bank is the first loop pass, while a real forward's head
        # input comes from the last pass. The head path must still match.
        cosine = F.cosine_similarity(
            output[0].flatten(), batch["head_input"].flatten(), dim=0
        )
        assert cosine > 0.999, cosine


def test_inkling2_loop_lm_head_objective_falls_back_to_block_mse(
    inkling2_lm_head_case,
):
    """G2PTQ must not score ``norm -> lm_head`` for this model."""

    assert inkling2_lm_head_case.wrapper.last_block_objective_modules() is None
