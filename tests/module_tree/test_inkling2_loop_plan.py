# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Real-model module-tree contract for Inkling 2 Mini (``inkling2_loop``).

The checkpoint stores routed experts as 3-D ``gate_up_proj`` / ``down_proj``
parameters, attention projections inside wrappers (``q_proj.linear`` plus
L2-clip children, a gated ``o_proj`` pair, short convolutions) and four decoder
layer classes: sliding/full attention, each with a looped and a non-looped
variant. The looped variants add q/k/v/mlp message gates.

The architecture itself is vendored in ``gptqmodel/hf_inkling2_loop``, so
every case here builds the real model class from a tiny config: nothing skips
on a Transformers release that has no ``inkling2_loop`` implementation.
"""

from __future__ import annotations

import copy
import re
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.models._const import EXPERT_INDEX_PLACEHOLDER
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.loader import _convert_model_with_defuser
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
    """Return the vendored Inkling2Loop configuration/modeling modules.

    GPT-QModel ships its own copy of the architecture (see
    ``gptqmodel/hf_inkling2_loop``), so the contract runs on every supported
    Transformers release instead of waiting for the modeling code to land
    upstream.
    """

    from gptqmodel.hf_inkling2_loop import configuration_inkling2_loop, modeling_inkling2_loop

    return configuration_inkling2_loop, modeling_inkling2_loop


def _tiny_config(configuration):
    hidden = 32
    return configuration.Inkling2LoopConfig(
        text_config={
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
        },
        vision_config={
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "projection_hidden_dim": 16,
            "text_hidden_size": hidden,
        },
    )


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

    with pytest.raises(RuntimeError, match="no `inkling2_loop` registry entry"):
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
