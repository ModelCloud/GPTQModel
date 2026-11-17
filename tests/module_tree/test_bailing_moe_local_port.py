# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Real-model ownership, aliases, MoE consumers and conservative AWQ policy."""

import copy
from types import SimpleNamespace

import pytest
import torch

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.bailing_moe import BailingMoeQModel
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.bailing_moe.modeling_bailing import BailingForCausalLM
from tests.test_bailing_moe_local_port import tiny_config


def test_mtp_envelope_and_mla_variants_have_explicit_ordered_subsets():
    blocks = BailingMoeQModel.build_layer_modules(BailingMoeQModel.module_tree, include_capture_only=True)
    assert blocks[:5] == [
        ["input_layernorm:!"], ["enorm:!"], ["hnorm:!"], ["eh_proj"], ["final_layernorm:!"],
    ]
    assert blocks[8] == [
        "attention.q_proj", "attention.query_key_value", "attention.q_a_proj", "attention.kv_a_proj_with_mqa",
        "attention.g_proj", "attention.query_layernorm:!", "attention.key_layernorm:!", "attention.g_norm:!",
        "attention.q_a_layernorm:!", "attention.kv_a_layernorm:!",
    ]
    assert blocks[9:11] == [["attention.q_b_proj", "attention.kv_b_proj"], ["attention.dense"]]
    groups = BailingMoeQModel.simple_layer_modules(tiny_config("hybrid"), SimpleNamespace(dynamic=None))
    assert groups[0] == ["eh_proj"]
    assert "self_attn.kv_a_proj_with_mqa" in groups[1]
    assert groups[2:4] == [["self_attn.kv_b_proj"], ["self_attn.o_proj"]]


@pytest.mark.parametrize("layout", ["dense", "mla", "mla_dense_q", "hybrid", "legacy_gqa", "mla_alias"])
@pytest.mark.parametrize("model_type", ["bailing_moe", "bailing_hybrid"])
def test_real_checkpoint_owners_and_moe_consumers(layout, model_type):
    definition = MODEL_MAP[model_type]
    assert definition is BailingMoeQModel
    config = tiny_config(layout, num_nextn_predict_layers=1)
    with torch.device("meta"):
        model = definition.loader.from_config(config)
    wrapper = definition(model=model, quantized=False, quantize_config=None)
    plan = wrapper.module_plan()
    declared = {p for block in definition.simple_layer_modules(config, SimpleNamespace(dynamic=None)) for p in block}
    for layer in model.model.layers:
        for key in layer.state_dict():
            owner = key.rsplit(".", 1)[0]
            spec = plan.find(owner)
            assert spec is not None, key
            assert spec.quantize == (owner in declared), (owner, spec.flags)
            if not spec.quantize:
                assert "!" in spec.flags or spec.capture_only
    assert plan.find("attention.query_key_value_typo") is None
    assert plan.find("attention.q_a_proj.undeclared_child") is None
    assert plan.roles("mlp.experts.0.gate_proj") == ("gate",)
    assert plan.roles("mlp.shared_experts.down_proj") == ("down",)
    assert plan.same_moe_family("mlp.experts.0.up_proj", "mlp.experts.0.down_proj")
    assert not plan.same_moe_family("mlp.experts.0.up_proj", "mlp.experts.1.down_proj")
    hooks = copy.copy(definition.moe_lifecycle_hooks)
    hooks.module_plan = plan
    for layer in model.model.layers[1:]:
        block = hooks.get_moe_block(layer, wrapper)
        assert block is layer.mlp
        assert hooks.get_experts_module(block, wrapper) is layer.mlp.experts
        assert hooks.get_shared_experts_module(block, wrapper) is layer.mlp.shared_experts


@pytest.mark.parametrize("layout", ["dense", "mla", "mla_dense_q", "hybrid", "legacy_gqa", "mla_alias"])
def test_awq_empty_policy_runs_real_plain_quantization_and_packing(layout, monkeypatch):
    from gptqmodel.looper.named_module import NamedModule
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization import AWQConfig
    from tests.test_awq_empty_groups import _processor

    torch.manual_seed(19)
    config = tiny_config(layout, intermediate_size=32, moe_intermediate_size=16, v_head_dim=8)
    model = BailingForCausalLM(config).to(torch.bfloat16).eval()
    definition = BailingMoeQModel(model=model, quantized=False, quantize_config=None)
    processor = _processor(model)
    processor.gptq_model = definition
    processor.qcfg = AWQConfig(group_size=-1, device="cpu")
    processor.apply_clip = False
    processor._feature_task_names = set()
    processor._awq_feature_kwargs = {}
    monkeypatch.setattr(processor, "_layer_input_features", lambda state: {})
    plan = ModulePlan.from_tree(BailingMoeQModel.module_tree)
    protected = {k: v.clone() for k, v in model.state_dict().items() if "norm" in k or ".gate." in k}
    for idx, layer in enumerate(model.model.layers):
        assert definition.awq_get_modules_for_scaling(layer, {}, {}) == []
        state = processor._get_layer_state(idx)
        state.layer_module = layer
        linears = {name: mod for name, mod in layer.named_modules() if isinstance(mod, torch.nn.Linear) and plan.find(name).quantize}
        state.modules = {name: NamedModule(mod, name, f"model.layers.{idx}.{name}", idx) for name, mod in linears.items()}
        named = dict(state.modules)
        processor._quantize_layer(idx, state)
        assert state.quantized
        assert named and all("q_scales" in mod.state and "q_zeros" in mod.state for mod in named.values())
        for name, entry in named.items():
            linear = linears[name]
            packed = AwqTorchLinear(4, -1, False, False, linear.in_features, linear.out_features, bias=linear.bias is not None, dtype=torch.bfloat16)
            packed.pack(linear, entry.state["q_scales"], entry.state["q_zeros"])
            assert packed.qweight.dtype == torch.int32
            assert "weight" not in dict(packed.named_parameters())
            layer.set_submodule(name, packed)
    for key, original in protected.items():
        torch.testing.assert_close(model.state_dict()[key], original, rtol=0, atol=0)
    with torch.no_grad():
        output = model(torch.tensor([[3, 8, 9]]), use_cache=False).logits
    assert torch.isfinite(output).all()


def _pinned_keys(config):
    """Index-owned templates, transcribed from the three pinned safetensors indexes."""
    keys = {"model.word_embeddings.weight", "model.norm.weight", "lm_head.weight"}
    for idx in range(config.num_hidden_layers + config.num_nextn_predict_layers):
        names = {"input_layernorm.weight", "post_attention_layernorm.weight", "attention.dense.weight"}
        linear = config.model_type == "bailing_hybrid" and idx < 32 and (idx + 1) % 8 != 0
        if config.model_type == "bailing_moe" or linear:
            names.add("attention.query_key_value.weight")
            if config.use_qk_norm:
                names |= {f"attention.{p}.weight" for p in ("query_layernorm", "key_layernorm")}
            if linear:
                names |= {"attention.g_proj.weight", "attention.g_norm.weight"}
        else:
            names |= {f"attention.{p}.weight" for p in ("q_a_proj", "q_a_layernorm", "q_b_proj", "kv_a_proj_with_mqa", "kv_a_layernorm", "kv_b_proj")}
        if idx < config.first_k_dense_replace:
            names |= {f"mlp.{p}_proj.weight" for p in ("gate", "up", "down")}
        else:
            names.add("mlp.gate.weight")
            if not config.legacy_gqa:
                names.add("mlp.gate.expert_bias")
            names |= {f"mlp.shared_experts.{p}_proj.weight" for p in ("gate", "up", "down")}
            names |= {f"mlp.experts.{expert}.{p}_proj.weight" for expert in range(config.num_experts) for p in ("gate", "up", "down")}
        if idx == 32:
            names |= {f"{p}.weight" for p in ("eh_proj", "enorm", "hnorm", "final_layernorm")}
        keys |= {f"model.layers.{idx}.{name}" for name in names}
    return keys


@pytest.mark.parametrize("layout,expected_count", [("legacy_gqa", 5603), ("dense", 14813), ("hybrid", 25015)])
def test_pinned_full_checkpoint_index_ownership_on_meta(layout, expected_count):
    # Exact published layer/expert counts; no trained weights or network access.
    hybrid, legacy = layout == "hybrid", layout == "legacy_gqa"
    config = tiny_config(
        layout, num_hidden_layers=28 if legacy else 32 if hybrid else 20,
        num_nextn_predict_layers=1 if hybrid else 0, num_experts=256, n_group=8,
        num_experts_per_tok=8, topk_group=4, layer_group_size=8,
    )
    if legacy:
        config.num_experts = 64
        config.first_k_dense_replace = 0
        config.num_shared_experts = 2
    with torch.device("meta"):
        model = BailingForCausalLM(config)
    expected = _pinned_keys(config)
    assert len(expected) == expected_count
    assert set(model.state_dict()) == expected


@pytest.mark.parametrize("layout", ["dense", "mla", "mla_dense_q", "hybrid", "legacy_gqa", "mla_alias"])
def test_calibration_replay_matches_every_native_layer_including_mtp(layout):
    torch.manual_seed(81)
    model = BailingForCausalLM(tiny_config(layout, num_nextn_predict_layers=2)).eval()
    wrapper = BailingMoeQModel(model=model, quantized=False, quantize_config=None)
    ids = torch.tensor([[0, 3, 8, 9, 4], [6, 8, 7, 5, 9]])
    mask = (ids != 0).long()
    positions = torch.arange(ids.shape[1])[None].expand(ids.shape[0], -1)
    source = {"input_ids": ids, "attention_mask": mask, "position_ids": positions}
    device = torch.device("cpu")
    wrapper.begin_input_capture_example(source, device)
    captured = {}

    def capture(module, args, kwargs):
        captured["input"] = wrapper.capture_first_layer_positional_inputs(args, kwargs, device)
        captured["kwargs"] = wrapper.capture_first_layer_input_kwargs(args, kwargs, device, {})

    expected = []
    handle = model.model.layers[0].register_forward_pre_hook(capture, with_kwargs=True)
    handles = [layer.register_forward_hook(lambda module, args, output: expected.append(output[0].clone())) for layer in model.model.layers]
    try:
        with torch.no_grad():
            model(**source, use_cache=False)
    finally:
        handle.remove()
        for hook in handles:
            hook.remove()
        wrapper.end_input_capture_example()
    hidden = captured["input"][0]
    with torch.no_grad():
        for layer, reference in zip(model.model.layers, expected, strict=True):
            inputs = [hidden]
            kwargs = wrapper.prepare_layer_replay_kwargs(layer, inputs, dict(captured["kwargs"]), device)
            hidden = layer(*inputs, **kwargs)[0]
            torch.testing.assert_close(hidden, reference, rtol=0, atol=0)
