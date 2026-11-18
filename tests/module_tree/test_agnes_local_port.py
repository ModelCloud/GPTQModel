# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Pinned checkpoint ownership, AWQ fold neutrality and real quantization for Agnes layouts."""

import hashlib

import pytest
import torch
from awq_neutrality import assert_awq_folds_neutral
from torch import nn

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.loop_processor import CalibrationConfig
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.agnes import AgnesQModel
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.agnes.configuration_agnes import AgnesConfig
from gptqmodel.models.ports.agnes.modeling_agnes import AgnesForConditionalGeneration
from gptqmodel.quantization.config import AWQConfig
from tests.test_agnes_local_port import tiny_config


# Agnes-AI/Agnes-3.0-Flash @ 891ce4f9ffb89b22888aa7fcc2bb2f3618867684 (config.json geometry).
RELEASED_CONFIG = {
    "text_config": {
        "hidden_size": 5120, "num_hidden_layers": 72, "intermediate_size": 17408,
        "parallel_ffn_intermediate_size": 2048, "global_attention_interval": 4, "num_attention_heads": 24,
        "num_key_value_heads": 4, "head_dim": 256, "linear_num_key_heads": 16, "linear_num_value_heads": 48,
    },
    "vision_config": {"out_hidden_size": 5120},
}
# sha256 of the sorted non-MTP tensor names of the released safetensors index.
RELEASED_KEYS_SHA256 = "21ba3094422ebda2670d892c502a7778aff4b8650c67c8f1c47b2b5ed22340d1"
LAYER_KEYS = {
    "delta_attn": {f"delta_attn.{name}" for name in (
        "A_log", "conv1d.weight", "dt_bias", "in_proj_a.weight", "in_proj_b.weight", "in_proj_qkv.weight",
        "in_proj_z.weight", "norm.weight", "out_proj.weight")},
    "global_attn": {f"global_attn.{name}.weight" for name in ("k_norm", "k_proj", "o_proj", "q_norm", "q_proj", "v_proj")},
    "common": {"input_layernorm.weight", "post_attention_layernorm.weight",
               *(f"mlp.{p}{name}.weight" for p in ("", "parallel_ffn.") for name in ("gate_proj", "up_proj", "down_proj"))},
}


def test_released_checkpoint_exact_meta_ownership():
    with torch.device("meta"):
        model = AgnesForConditionalGeneration(AgnesConfig(**RELEASED_CONFIG))
    keys = set(model.state_dict())
    assert hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest() == RELEASED_KEYS_SHA256
    assert len(keys) == 1506
    assert {"lm_head.weight", "model.visual.pos_embed.weight", "model.visual.merger.linear_fc2.weight",
            "model.language_model.layers.3.global_attn.q_norm.weight",
            "model.language_model.layers.70.delta_attn.A_log"} <= keys
    assert len(model.model.language_model.layers) == 72
    assert len(model.model.visual.blocks) == 27
    assert AgnesQModel.out_of_model_tensors == {"prefixes": ["mtp"]}
    assert {name for name, definition in MODEL_MAP.items() if definition is AgnesQModel} == {"agnes"}


@pytest.mark.parametrize("parallel,bias", [(0, False), (12, True)])
def test_all_live_decoder_weight_owners_and_quantization_consumers(parallel, bias):
    model = AgnesForConditionalGeneration(tiny_config(parallel, bias=bias)).eval()
    _convert_model_with_defuser(AgnesQModel, model, cleanup_original=False)
    wrapper = AgnesQModel.__new__(AgnesQModel)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    wrapper.__dict__["_module_plan"] = ModulePlan.from_tree(AgnesQModel.module_tree)
    plan = wrapper.module_plan()
    assert wrapper.extract_layers_node() == ["model.language_model.layers"]
    declared = {name for block in wrapper.simple_layer_modules(model.config, None) for name in block}
    for index, layer in enumerate(model.model.language_model.layers):
        # Compare with the two released layer layouts, allowing only the
        # explicitly configured absence of parallel FFN and presence of bias.
        kind = "delta_attn" if index == 0 else "global_attn"
        expected = LAYER_KEYS[kind] | LAYER_KEYS["common"]
        if not parallel:
            expected = {k for k in expected if ".parallel_ffn." not in k}
        actual = set(layer.state_dict())
        assert {k for k in actual if not k.endswith(".bias")} == expected
        if bias and index == 1:
            assert {k for k in actual if k.endswith(".bias")} == {
                f"{kind}.{projection}.bias" for projection in ("q_proj", "k_proj", "v_proj", "o_proj")}
        for name in actual:
            owner = name.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, name
            if spec.quantize:
                assert owner in declared
                assert isinstance(layer.get_submodule(owner), nn.Linear)
            else:
                assert set(spec.flags) & {"!", "?"}
        live = {name for name in declared if isinstance(_get_submodule(layer, name), nn.Linear)}
        assert len(live) == ((6 if index == 0 else 7) + (3 if parallel else 0))
        assert not plan.find("delta_attn.undeclared")
        assert not plan.find("global_attn.q_proj_extra")
    assert plan.find("delta_attn").flags == ("!",)
    for path, role in [("global_attn.q_proj", "q"), ("global_attn.o_proj", "o"),
                       ("mlp.gate_proj", "gate"), ("mlp.parallel_ffn.down_proj", "down")]:
        assert role in plan.find(path).roles


def _get_submodule(layer, path):
    try:
        return layer.get_submodule(path)
    except AttributeError:
        return None


def _scaled_tiny_model(parallel, kv_heads=1):
    config = tiny_config(parallel)
    config.text_config.num_key_value_heads = kv_heads
    config.text_config._attn_implementation = "eager"
    torch.manual_seed(23)
    model = AgnesForConditionalGeneration(config).eval()
    # Agnes norms scale by `1 + weight` and initialise to zero; perturb every
    # 1-D gain so a fold into the wrong place cannot hide behind identities.
    with torch.no_grad():
        for name, parameter in model.model.language_model.named_parameters():
            if parameter.ndim == 1 and name.endswith("norm.weight"):
                parameter.normal_(0.0, 0.3)
    return model


EXPECTED_GROUPS = {
    "agnes_delta_attention": [
        ("input_layernorm", ["delta_attn.in_proj_qkv", "delta_attn.in_proj_z"],
         ["delta_attn.in_proj_b", "delta_attn.in_proj_a"]),
    ],
    "agnes_global_attention": [
        ("input_layernorm", ["global_attn.q_proj", "global_attn.k_proj", "global_attn.v_proj"], []),
    ],
}


@pytest.mark.parametrize("parallel", [0, 12])
@pytest.mark.parametrize("kv_heads", [1, 2])
def test_awq_folds_preserve_delta_and_global_layer_output(parallel, kv_heads):
    model = _scaled_tiny_model(parallel, kv_heads)
    tree = AgnesQModel._resolve_effective_module_tree(model, None)
    ids = torch.randint(2, 60, (1, 9))

    def forward():
        return model(input_ids=ids, use_cache=False).logits

    mlp = [("post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"]
            + (["mlp.parallel_ffn.gate_proj", "mlp.parallel_ffn.up_proj"] if parallel else []), [])]
    mlp.append(("mlp.up_proj", ["mlp.down_proj"], []))
    if parallel:
        mlp.append(("mlp.parallel_ffn.up_proj", ["mlp.parallel_ffn.down_proj"], []))
    for layer in model.model.language_model.layers:
        checked = assert_awq_folds_neutral(AgnesQModel, model, layer, forward, tree=tree)
        expected = list(EXPECTED_GROUPS[layer.layer_type])
        if layer.layer_type == "agnes_global_attention" and kv_heads == 2:
            # Without GQA `v_proj` feeds `o_proj` through softmax(QK)V and the
            # per-channel sigmoid output gate, both linear in V's channels.
            expected.append(("global_attn.v_proj", ["global_attn.o_proj"], []))
        expected += mlp
        assert [(s, t, list(c)) for s, t, c in checked] == expected


@pytest.mark.parametrize("parallel", [0, 12])
@torch.no_grad()
def test_awq_scaled_quantization_covers_every_declared_linear(parallel, monkeypatch):
    model = _scaled_tiny_model(parallel)
    wrapper = AgnesQModel.__new__(AgnesQModel)
    nn.Module.__init__(wrapper)
    wrapper.__dict__["model"] = model
    declared = {name for block in AgnesQModel.simple_layer_modules(model.config, None) for name in block}
    layers = model.model.language_model.layers
    ids = torch.randint(2, 60, (2, 9))

    captured, handles = {}, []

    def record_layer(index):
        def hook(module, args, kwargs):
            captured[index] = {k: v for k, v in kwargs.items() if k != "past_key_values"}
        return hook

    def record(index, name):
        def hook(module, args):
            captured.setdefault((index, "features"), {})[name] = args[0].detach().clone()
        return hook

    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(record_layer(index), with_kwargs=True))
        for name in declared:
            target = _get_submodule(layer, name)
            if isinstance(target, nn.Linear):
                handles.append(target.register_forward_pre_hook(record(index, name)))
    try:
        model(input_ids=ids, use_cache=False).logits
    finally:
        for handle in handles:
            handle.remove()

    qcfg = AWQConfig(device="cpu", group_size=4, fallback=None)
    processor = AWQProcessor(
        tokenizer=None, qcfg=qcfg, calibration=CalibrationConfig(), gptq_model=wrapper,
        model=model, calculate_w_wq_diff=False,
    )
    processor.draw_progress = lambda *args, **kwargs: None
    processor.log_new_row = lambda *args, **kwargs: None
    processor.apply_clip = False
    processor._feature_task_names = set()
    processor._awq_feature_kwargs = {}
    for index, layer in enumerate(layers):
        state = processor._get_layer_state(index)
        state.layer_module = layer
        # The processor keys a layer's modules by their layer-relative name.
        state.modules = {
            name: NamedModule(
                layer.get_submodule(name), name, f"model.language_model.layers.{index}.{name}", index,
            ) for name in sorted(declared) if isinstance(_get_submodule(layer, name), nn.Linear)
        }
        original = {name: value.clone() for name, value in layer.state_dict().items()}
        named_linears = list(state.modules.values())
        assert len(named_linears) == ((6 if index == 0 else 7) + (3 if parallel else 0))
        features = captured[(index, "features")]
        assert set(features) == {named.name for named in named_linears}
        processor._module_forward_kwargs = dict(captured[index])
        monkeypatch.setattr(processor, "_layer_input_features", lambda state, features=features: features)
        assert wrapper.awq_get_modules_for_scaling(layer, features, {})
        processor._quantize_layer(index, state)
        assert state.quantized
        for named in named_linears:
            assert "q_scales" in named.state and "q_zeros" in named.state
            assert not torch.equal(named.module.weight, original[named.name + ".weight"])
        changed = {name for name, value in layer.state_dict().items() if not torch.equal(value, original[name])}
        # Scale search folded into both RMS norms (plus the delta gate inputs).
        assert {"input_layernorm.weight", "post_attention_layernorm.weight"} <= changed
        allowed = {named.name + ".weight" for named in named_linears} | {
            "input_layernorm.weight", "post_attention_layernorm.weight"}
        assert changed <= allowed | {"delta_attn.in_proj_b.weight", "delta_attn.in_proj_a.weight"}
    output = model(input_ids=ids, use_cache=False).logits
    assert torch.isfinite(output).all()
