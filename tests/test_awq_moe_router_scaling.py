# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ MoE folds must also scale the router and ``shared_expert_gate``.

`apply_scale` divides a norm by ``s`` and multiplies the quantized consumers'
input columns by ``s``. The MoE router and Qwen2's ``shared_expert_gate`` read
the same normed hidden state but are not quantized, so before this regression
test they kept reading ``x / s``: router logits changed, different experts were
selected, and the block output moved by ~2e-2 relative.

The oracle is the pre-fold block forward evaluated in float64 on an independent
copy of the layer, plus the closed-form ``w * s`` weight ratio.
"""

import copy
from types import SimpleNamespace

import defuser
import pytest
import torch
from torch import nn
from transformers import (
    Qwen2MoeConfig,
    Qwen2MoeForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.models.base import NON_QUANTIZE_FLAGS
from gptqmodel.models.definitions.qwen2_moe import Qwen2MoeQModel
from gptqmodel.models.definitions.qwen3_moe import Qwen3MoeQModel
from gptqmodel.quantization.awq.quantize.scale import apply_scale
from gptqmodel.quantization.awq.utils.module import get_op_name
from gptqmodel.quantization.config import AWQConfig, CalibrationConfig


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.double() - expected.double()).norm() / expected.double().norm().clamp_min(1e-12))


def _tiny_qwen2_moe() -> Qwen2MoeForCausalLM:
    torch.manual_seed(0)
    config = Qwen2MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    model = Qwen2MoeForCausalLM(config).eval()
    # transformers 5 ships fused 3-D expert tensors; the loader splits them with
    # defuser before quantization, so the test does the same.
    defuser.convert_model(model, cleanup_original=True)
    return model


def _tiny_qwen3_moe() -> Qwen3MoeForCausalLM:
    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    model = Qwen3MoeForCausalLM(config).eval()
    defuser.convert_model(model, cleanup_original=True)
    return model


def _bare_qmodel(definition, hf_model):
    qmodel = definition.__new__(definition)
    nn.Module.__init__(qmodel)
    qmodel.__dict__["model"] = hf_model
    return qmodel


def _features(definition, hf_model, config, batch: int = 1, seq: int = 6):
    """Capture-shaped activations for every declared quantized module."""

    layer = hf_model.model.layers[0]
    modules = dict(layer.named_modules())
    features = {}
    for block in definition.full_layer_modules(config, is_awq_quantize=True, include_capture_only=True):
        for raw_name in block:
            if any(flag in raw_name for flag in NON_QUANTIZE_FLAGS):
                continue
            name = raw_name
            for flag in NON_QUANTIZE_FLAGS:
                name = name.replace(flag, "")
            module = modules.get(name)
            if not isinstance(module, nn.Linear):
                continue
            features[name] = torch.randn(batch, seq, module.in_features)
    # The AWQ policy captures the MoE block input under its root path.
    features["mlp"] = torch.randn(batch, seq, config.hidden_size)
    return features


def _scales_for(nodes, layer):
    torch.manual_seed(1)
    return [
        (
            get_op_name(layer, node["prev_op"]),
            [get_op_name(layer, module) for module in node["layers"]],
            torch.rand(node["prev_op"].weight.shape[0]) + 0.5,
            None,
        )
        for node in nodes
    ]


def _consumer_map(nodes, layer):
    consumer_map = {}
    for node in nodes:
        if node.get("_scale_consumers"):
            consumer_map[get_op_name(layer, node["prev_op"])] = node["_scale_consumers"]
    return consumer_map


def test_qwen2_moe_fold_declares_router_and_shared_gate_consumers():
    hf_model = _tiny_qwen2_moe()
    layer = hf_model.model.layers[0]
    qmodel = _bare_qmodel(Qwen2MoeQModel, hf_model)

    nodes = qmodel.awq_get_modules_for_scaling(layer, _features(Qwen2MoeQModel, hf_model, hf_model.config), {})
    consumer_map = _consumer_map(nodes, layer)

    assert consumer_map == {
        "post_attention_layernorm": ["mlp.gate", "mlp.shared_expert_gate"],
    }, consumer_map


def test_qwen3_moe_fold_declares_router_consumer():
    hf_model = _tiny_qwen3_moe()
    layer = hf_model.model.layers[0]
    qmodel = _bare_qmodel(Qwen3MoeQModel, hf_model)

    nodes = qmodel.awq_get_modules_for_scaling(layer, _features(Qwen3MoeQModel, hf_model, hf_model.config), {})
    consumer_map = _consumer_map(nodes, layer)

    assert consumer_map == {"post_attention_layernorm": ["mlp.gate"]}, consumer_map


@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize(
    "definition,builder",
    [(Qwen2MoeQModel, _tiny_qwen2_moe), (Qwen3MoeQModel, _tiny_qwen3_moe)],
    ids=["qwen2_moe", "qwen3_moe"],
)
@torch.inference_mode()
def test_moe_identity_search_preserves_router_logits(definition, builder, chunked, monkeypatch):
    hf_model = builder()
    layer = hf_model.model.layers[0]
    qmodel = _bare_qmodel(definition, hf_model)
    nodes = qmodel.awq_get_modules_for_scaling(layer, _features(definition, hf_model, hf_model.config), {})
    node = next(node for node in nodes if node["prev_op"] is layer.post_attention_layernorm)
    inp = layer.post_attention_layernorm(torch.randn(1, 64, hf_model.config.hidden_size))
    processor = AWQProcessor(
        tokenizer=None,
        qcfg=AWQConfig(device="cpu", group_size=16, scale_search_chunked_activations=chunked, fallback=None),
        calibration=CalibrationConfig(), gptq_model=SimpleNamespace(rotary_embedding=None),
        model=hf_model, calculate_w_wq_diff=False,
    )
    processor._quant_batch_size = 0
    baseline = torch.nn.functional.linear(inp.reshape(-1, inp.shape[-1]), layer.mlp.gate.weight)
    weights = {name: value.clone() for name, value in layer.state_dict().items()}
    logits = []
    hook = layer.mlp.gate.register_forward_pre_hook(
        lambda gate, args: logits.append(torch.nn.functional.linear(args[0], gate.weight).clone())
    )
    monkeypatch.setattr(processor, "_pseudo_quantize_tensor_into", lambda src, dst, **_kwargs: dst.copy_(src))
    try:
        processor._search_best_scale(
            layer, node["prev_op"], node["layers"], inp, module2inspect=node["module2inspect"],
            scale_consumers=node.get("_scale_consumers"),
        )
    finally:
        hook.remove()
    assert len(logits) == 21
    for candidate in logits:
        assert torch.equal(candidate, baseline)
        assert torch.equal(candidate.topk(2).indices, baseline.topk(2).indices)
    for name, value in layer.state_dict().items():
        assert torch.equal(value, weights[name]), name


@pytest.mark.parametrize(
    "definition,builder",
    [(Qwen2MoeQModel, _tiny_qwen2_moe), (Qwen3MoeQModel, _tiny_qwen3_moe)],
    ids=["qwen2_moe", "qwen3_moe"],
)
def test_moe_fold_preserves_block_output_and_routing(definition, builder):
    hf_model = builder()
    config = hf_model.config
    layer = hf_model.model.layers[0]
    qmodel = _bare_qmodel(definition, hf_model)

    features = _features(definition, hf_model, config)
    nodes = qmodel.awq_get_modules_for_scaling(layer, features, {})
    scales_list = _scales_for(nodes, layer)
    consumer_map = _consumer_map(nodes, layer)
    assert consumer_map, "the MoE fold must declare its non-quantized consumers"

    reference_layer = copy.deepcopy(layer).double()
    router = layer.mlp.gate
    router_before = router.weight.detach().clone()
    shared_gate_before = (
        layer.mlp.shared_expert_gate.weight.detach().clone()
        if hasattr(layer.mlp, "shared_expert_gate")
        else None
    )

    x = torch.randn(1, 6, config.hidden_size)
    with torch.no_grad():
        normed = layer.post_attention_layernorm(x)
        reference_out = reference_layer.mlp(reference_layer.post_attention_layernorm(x.double()))
        reference_logits = torch.nn.functional.linear(
            reference_layer.post_attention_layernorm(x.double()),
            reference_layer.mlp.gate.weight.double(),
        )
        reference_shared = (
            reference_layer.mlp.shared_expert_gate(reference_layer.post_attention_layernorm(x.double()))
            if shared_gate_before is not None
            else None
        )

        apply_scale(layer, scales_list, input_feat_dict=features, consumer_map=consumer_map)

        folded_normed = layer.post_attention_layernorm(x)
        folded_out = layer.mlp(folded_normed).double()
        folded_logits = torch.nn.functional.linear(folded_normed.double(), router.weight.double())
        folded_shared = (
            layer.mlp.shared_expert_gate(folded_normed).double() if shared_gate_before is not None else None
        )

    # The norm is divided by the accumulated scale; the consumers must receive it.
    assert _relative_error(normed, folded_normed) > 0.1
    assert _relative_error(folded_out, reference_out) < 1e-5
    assert _relative_error(folded_logits, reference_logits) < 1e-5
    if shared_gate_before is not None:
        assert _relative_error(folded_shared, reference_shared) < 1e-5

    expected_scale = torch.ones_like(router_before[0])
    for prev_op_name, _, scales, _ in scales_list:
        if prev_op_name == "post_attention_layernorm":
            expected_scale = expected_scale * scales
    torch.testing.assert_close(
        router.weight.detach(),
        router_before * expected_scale.view(1, -1),
        rtol=1e-6,
        atol=1e-7,
    )
    if shared_gate_before is not None:
        torch.testing.assert_close(
            layer.mlp.shared_expert_gate.weight.detach(),
            shared_gate_before * expected_scale.view(1, -1),
            rtol=1e-6,
            atol=1e-7,
        )

    assert torch.equal(
        torch.topk(reference_logits, 2, dim=-1).indices,
        torch.topk(folded_logits, 2, dim=-1).indices,
    )


@pytest.mark.parametrize(
    "definition,builder",
    [(Qwen2MoeQModel, _tiny_qwen2_moe), (Qwen3MoeQModel, _tiny_qwen3_moe)],
    ids=["qwen2_moe", "qwen3_moe"],
)
def test_moe_fold_without_consumers_changes_the_block(definition, builder):
    """Control: the pre-fix fold (no ``consumer_map``) is not output-preserving."""

    hf_model = builder()
    config = hf_model.config
    layer = hf_model.model.layers[0]
    qmodel = _bare_qmodel(definition, hf_model)

    features = _features(definition, hf_model, config)
    nodes = qmodel.awq_get_modules_for_scaling(layer, features, {})
    scales_list = _scales_for(nodes, layer)

    reference_layer = copy.deepcopy(layer).double()
    x = torch.randn(1, 6, config.hidden_size)
    with torch.no_grad():
        reference_out = reference_layer.mlp(reference_layer.post_attention_layernorm(x.double()))
        apply_scale(layer, scales_list, input_feat_dict=features)
        folded_out = layer.mlp(layer.post_attention_layernorm(x)).double()

    # This is the measured pre-fix error (~2e-2); the fix brings it to <1e-5.
    assert _relative_error(folded_out, reference_out) > 1e-3
