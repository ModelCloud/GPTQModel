# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""Real tiny CPU models qualify inferred AWQ folds, not just path resolution."""

from types import SimpleNamespace

import pytest
import torch
from test_awq_scaling_groups import (
    MAMBA_SSM_KERNEL_FAMILIES,
    _foldable_groups,
    _mamba_ssm_cpu_unavailable_reason,
    _tiny_config,
)
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.cohere_compass.configuration_cohere_compass import CohereCompassTextConfig
from transformers.models.cohere_compass.modeling_cohere_compass import CohereCompassDecoderLayer
from transformers.models.gemma3n.modeling_gemma3n import Gemma3nTextDecoderLayer
from transformers.models.inkling.modeling_inkling import InklingTextModel

from gptqmodel.looper.awq_processor import AWQProcessor, _AWQScaleGroupError
from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.quantization.awq.quantize.scale import apply_scale
from gptqmodel.quantization.awq.utils.module import get_op_name


class _InklingTextFixture(nn.Module):
    """Native text backbone under its real multimodal checkpoint prefix."""

    def __init__(self, text_config):
        super().__init__()
        self.config = AutoConfig.for_model("inkling_mm_model", text_config=text_config)
        self.model = nn.Module()
        self.model.language_model = InklingTextModel(text_config)

    def forward(self, ids, use_cache=False):
        return self.model.language_model(ids, use_cache=use_cache)


def _text_output(model, ids):
    output = model(ids, use_cache=False)
    return output.logits if hasattr(output, "logits") else output.last_hidden_state


@pytest.fixture(scope="module", params=[
    pytest.param(("granitemoe_swa", {"shared_intermediate_size": 16}), id="granite-shared"),
    pytest.param(("granitemoe_swa", {"shared_intermediate_size": 0}), id="granite-no-shared"),
    pytest.param(("hunyuan_v1_moe", {"expert_hidden_dim": 64}), id="hunyuan-nested-router"),
    pytest.param(("falcon_mamba", {"hidden_size": 8, "intermediate_size": 16, "state_size": 4, "time_step_rank": 8}), id="falcon-ssm-width-collision"),
    pytest.param(("axk2", {}), id="axk2-concatenated-query"),
    pytest.param(("glm_moe_dsa", {}), id="glm-dsa-indexer"),
    pytest.param(("inkling_mm_model", {}), id="inkling-native-shared-experts"),
    pytest.param(("longcat_flash", {"n_routed_experts": 4, "zero_expert_num": 0, "expert_ffn_hidden_size": 8, "num_experts_per_tok": 2}), id="longcat-dual-residual"),
])
def case(request):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        family, overrides = request.param
        config = _tiny_config(family) if family != "inkling_mm_model" else AutoConfig.for_model(
            "inkling_text", vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=4, head_dim=8, max_position_embeddings=64,
            swa_num_attention_heads=4, swa_num_key_value_heads=4, swa_head_dim=8, d_rel=2, rel_extent=16,
            n_routed_experts=4, n_shared_experts=2, moe_intermediate_size=16, num_experts_per_tok=2,
            mlp_layer_types=["dense", "sparse"], layer_types=["full_attention", "hybrid_sliding"],
        )
        for name, value in overrides.items():
            setattr(config, name, value)
        config._attn_implementation = "eager"
        definition = MODEL_MAP[family]
        model = (_InklingTextFixture(config) if family == "inkling_mm_model" else AutoModelForCausalLM.from_config(config, trust_remote_code=False)).eval()
        _convert_model_with_defuser(definition, model, cleanup_original=False)
        wrapper = definition.__new__(definition)
        nn.Module.__init__(wrapper)
        wrapper.__dict__["model"] = model
        tree = definition._resolve_effective_module_tree(model, None)
        wrapper.__dict__["effective_module_tree"] = tree
        wrapper.__dict__["module_tree"] = tree
        layers = model.get_submodule(definition.extract_layers_node()[0])
        try:
            yield SimpleNamespace(family=family, config=config, model=model, wrapper=wrapper, layers=layers, plan=ModulePlan.from_tree(tree))
        finally:
            torch.set_num_threads(previous_threads)


def _capture(model, layer, ids):
    return _capture_forward(layer, lambda: _text_output(model, ids))


def _capture_forward(layer, forward):
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


def test_real_model_weight_owners_are_declared(case):
    declared = {name.split(":", 1)[0] for block in case.wrapper.simple_layer_modules(case.model.config, None) for name in block}
    for layer in case.layers:
        for name in layer.state_dict():
            owner = name.rsplit(".", 1)[0]
            spec = case.plan.find(owner)
            assert spec is not None, (case.family, name)
            if spec.quantize:
                assert owner in declared or spec.template in declared, (case.family, name)
            else:
                assert set(spec.flags) & {"!", "?"}, (case.family, name)


@pytest.mark.parametrize("family,paths", [
    ("moss", ["ln_1", "attn.out_proj", "mlp.fc_in"]),
    ("gemma3n", ["input_layernorm", "self_attn.v_proj", "post_feedforward_layernorm", "per_layer_input_gate"]),
    ("gemma3n_text", ["input_layernorm", "self_attn.v_proj", "post_feedforward_layernorm", "per_layer_input_gate"]),
    ("axk2", ["self_attn.indexer.weights_proj", "self_attn.q_a_proj", "self_attn.kv_b_proj"]),
    ("falcon_mamba", ["mixer.in_proj", "mixer.x_proj"]),
    ("longcat_flash", ["self_attn.0.o_proj", "mlps.0.down_proj"]),
    ("glm_moe_dsa", ["input_layernorm", "self_attn.indexer.wk", "self_attn.kv_b_proj"]),
    ("qwen", ["mlp.w2"]),
    ("inkling_mm_model", ["post_attention_layernorm"]),
])
def test_nonlinear_and_residual_boundaries_are_not_fold_sources(family, paths):
    plan = ModulePlan.from_tree(MODEL_MAP[family].module_tree)
    for path in paths:
        assert "nosrc" in plan.find(path).flags


@pytest.mark.parametrize("family,expected", [
    ("internlm2", [["attention.wqkv"], ["attention.wo"]]),
    ("minicpm", [["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], ["self_attn.o_proj"], ["mlp.gate_proj", "mlp.up_proj"], ["mlp.down_proj"]]),
])
def test_parallel_projections_are_not_declared_as_sequential(family, expected):
    blocks = MODEL_MAP[family].simple_layer_modules(None, None)
    for block in expected:
        assert block in blocks


def test_every_inferred_fold_preserves_full_text_outputs(case):
    if case.family in MAMBA_SSM_KERNEL_FAMILIES:
        # The invariant runs a real CPU forward; `mamba_ssm` would route it to
        # CUDA-only kernels and fail on the device, not on the fold math.
        reason = _mamba_ssm_cpu_unavailable_reason()
        if reason is not None:
            pytest.skip(reason)
    ids = torch.randint(2, case.config.vocab_size, (1, 8))
    checked = 0
    for layer in case.layers:
        reference, features = _capture(case.model, layer, ids)
        nodes = case.wrapper.awq_get_modules_for_scaling(layer, features, {})
        AWQProcessor._validate_scale_group_conflicts(layer, nodes)
        if case.family == "axk2":
            sources = {get_op_name(layer, node["prev_op"]) for node in _foldable_groups(layer, nodes)}
            assert not sources & {
                "input_layernorm.norm", "post_attention_layernorm.norm", "self_attn.indexer.weights_proj",
            }
        checked += _assert_fold_preservation(layer, nodes, reference, lambda: _text_output(case.model, ids))
    assert checked


def _assert_fold_preservation(layer, nodes, reference, forward):
    snapshot = {name: value.clone() for name, value in layer.state_dict().items()}
    checked = 0
    for node in _foldable_groups(layer, nodes):
        source = get_op_name(layer, node["prev_op"])
        targets = [get_op_name(layer, module) for module in node["layers"]]
        consumers = node.get("_scale_consumers") or []
        scales = torch.linspace(0.55, 1.45, node["inp"].shape[-1])
        try:
            apply_scale(layer, [(source, targets, scales, None)], consumer_map={source: consumers})
            with torch.inference_mode():
                actual = forward()
            relative_error = (actual - reference).norm() / reference.norm()
            assert relative_error < 1e-5, (source, targets, relative_error.item())
            checked += 1
        finally:
            layer.load_state_dict(snapshot)
    return checked


@pytest.mark.parametrize("family,kv_heads", [("cohere_compass", 4), ("cohere_compass", 2), ("gemma3n_text", 2)])
def test_native_branched_decoders(family, kv_heads):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        config = _tiny_config("gemma3n_text") if family == "gemma3n_text" else CohereCompassTextConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
            num_key_value_heads=kv_heads, vocab_size=128, eos_token_id=2, layer_types=["full_attention"],
        )
        config._attn_implementation = "eager"
        positions = (torch.ones(1, 8, 8), torch.zeros(1, 8, 8))
        if family == "gemma3n_text":
            config.num_kv_shared_layers = 0
            layer = Gemma3nTextDecoderLayer(config, 0).eval()
            hidden = torch.randn(config.altup_num_inputs, 1, 8, 32)
            per_layer = torch.randn(1, 8, config.hidden_size_per_layer_input)

            def forward():
                return layer(hidden, position_embeddings=positions, per_layer_input=per_layer, shared_kv_states={})
        else:
            layer = CohereCompassDecoderLayer(config, 0).eval()
            hidden = torch.randn(1, 8, 32)

            def forward():
                return layer(hidden, position_embeddings=positions)

        definition = MODEL_MAP[family]
        wrapper = definition.__new__(definition)
        nn.Module.__init__(wrapper)
        wrapper.__dict__["model"] = SimpleNamespace(config=config)
        wrapper.__dict__["effective_module_tree"] = definition.module_tree
        wrapper.__dict__["module_tree"] = definition.module_tree
        plan = ModulePlan.from_tree(definition.module_tree)
        for name in layer.state_dict():
            assert plan.find(name.rsplit(".", 1)[0]) is not None, name
        reference, features = _capture_forward(layer, forward)
        nodes = wrapper.awq_get_modules_for_scaling(layer, features, {})
        AWQProcessor._validate_scale_group_conflicts(layer, nodes)
        assert _assert_fold_preservation(layer, nodes, reference, forward)
        if family == "cohere_compass":
            node = next(node for node in nodes if node["prev_op"] is layer.input_layernorm)
            assert layer.mlp.gate_proj in node["layers"]
            with pytest.raises(_AWQScaleGroupError, match="replay boundary"):
                AWQProcessor._validate_scale_group(layer, node["prev_op"], node["layers"], node["inp"], node["module2inspect"], node.get("_scale_consumers"))


def test_cross_branch_groups_reject_incomplete_replay(case):
    if case.family != "granitemoe_swa" or not case.model.config.shared_intermediate_size:
        return
    layer = case.layers[0]
    _, features = _capture(case.model, layer, torch.ones((1, 8), dtype=torch.long))
    nodes = case.wrapper.awq_get_modules_for_scaling(layer, features, {})
    node = next(node for node in nodes if node["prev_op"] is layer.post_attention_layernorm)
    assert layer.shared_mlp.input_linear in node["layers"]
    with pytest.raises(_AWQScaleGroupError, match="shared_mlp.input_linear"):
        AWQProcessor._validate_scale_group(
            layer, node["prev_op"], node["layers"], node["inp"], node["module2inspect"], node.get("_scale_consumers"),
        )
