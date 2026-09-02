# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import copy
from dataclasses import replace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel.quantization.qvq_transform_llama import LlamaQVQTransformImplementor
from gptqmodel.quantization.qvq_transform_planner import (
    ProjectionRole,
    QVQTransformPlanner,
    TransformKind,
    TransformPlacement,
)


def _tiny_llama():
    torch.manual_seed(20260831)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return LlamaForCausalLM(config).float().eval()


def test_qvq_transform_plan_counts_and_role_descriptors():
    planner = QVQTransformPlanner(_tiny_llama(), LlamaQVQTransformImplementor())
    semantics = planner.analyze_model_graph()

    assert len(semantics) == 14
    assert {semantic.role for semantic in semantics[:7]} == set(ProjectionRole)
    assert planner.build_transform_plan("A0").online_hadamards_per_block == 14
    assert planner.build_transform_plan("A1").online_hadamards_per_block == 5
    assert planner.build_transform_plan("A3").online_hadamards_per_block == 3
    assert planner.build_transform_plan("A4").online_hadamards_per_block == 1
    assert planner.build_transform_plan("A6").online_hadamards_per_block == 0
    assert planner.build_transform_plan("A20").online_hadamards_per_block == 3
    assert planner.build_transform_plan("A21").online_hadamards_per_block == 3
    assert planner.build_transform_plan("A22").online_hadamards_per_block == 3
    assert planner.build_transform_plan("A23").online_hadamards_per_block == 2
    assert planner.build_transform_plan("A24").online_hadamards_per_block == 12
    assert planner.build_transform_plan("A25").online_hadamards_per_block == 12
    assert planner.build_transform_plan("A26").online_hadamards_per_block == 10
    assert planner.build_transform_plan("A27").online_hadamards_per_block == 12
    assert planner.build_transform_plan("A28").online_hadamards_per_block == 10
    assert planner.build_transform_plan("A29").online_hadamards_per_block == 12
    assert planner.build_transform_plan("A30").online_hadamards_per_block == 10
    assert planner.build_transform_plan("A31").online_hadamards_per_block == 9
    assert planner.build_transform_plan("A33").online_hadamards_per_block == 5
    assert planner.build_transform_plan("A34").online_hadamards_per_block == 5
    assert planner.build_transform_plan("A41").online_hadamards_per_block == 9

    a31 = planner.build_transform_plan("A31")
    layer_zero = [item for item in a31.modules if ".layers.0." in item.module_name]
    attention_inputs = {
        item.input_transform.basis_id
        for item in layer_zero
        if item.role
        in {
            ProjectionRole.ATTENTION_Q,
            ProjectionRole.ATTENTION_K,
            ProjectionRole.ATTENTION_V,
        }
    }
    mlp_inputs = {
        item.input_transform.basis_id
        for item in layer_zero
        if item.role in {ProjectionRole.MLP_GATE, ProjectionRole.MLP_UP}
    }
    assert attention_inputs == {"sibling.attn.l0"}
    assert mlp_inputs == {"sibling.mlp.l0"}
    assert all(
        item.input_transform.placement == TransformPlacement.SHARED
        for item in layer_zero
        if item.input_transform.basis_id in attention_inputs | mlp_inputs
    )

    a4 = planner.build_transform_plan("A4")
    gate = next(item for item in a4.modules if item.role == ProjectionRole.MLP_GATE)
    up = next(item for item in a4.modules if item.role == ProjectionRole.MLP_UP)
    down = next(item for item in a4.modules if item.role == ProjectionRole.MLP_DOWN)
    assert (gate.output_transform.kind, gate.output_transform.placement) == (
        TransformKind.PERMUTATION,
        TransformPlacement.FOLDED,
    )
    assert up.output_transform.kind == TransformKind.DIAGONAL
    assert down.input_transform.kind == TransformKind.HADAMARD


@pytest.mark.parametrize(
    "arm",
    (
        "A1", "A3", "A4", "A6", "A20", "A21", "A22", "A23", "A24", "A25", "A26",
        "A27", "A28", "A29", "A30", "A31", "A41",
    ),
)
def test_qvq_llama_dense_rewrite_preserves_final_logits(arm):
    original = _tiny_llama()
    rewritten = copy.deepcopy(original)
    input_ids = torch.tensor([[1, 17, 9, 41, 3], [1, 5, 6, 7, 8]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        expected = original(input_ids=input_ids, attention_mask=attention_mask).logits

    planner = QVQTransformPlanner(rewritten, LlamaQVQTransformImplementor())
    metadata = planner.rewrite_dense_weights(planner.build_transform_plan(arm), seed=29)
    with torch.inference_mode():
        actual = rewritten(input_ids=input_ids, attention_mask=attention_mask).logits

    delta = actual - expected
    assert metadata["rewritten"] is True
    assert delta.abs().max().item() < 2e-5
    assert torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected) < 2e-5
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))


def test_qvq_a0_rewrite_is_byte_preserving():
    model = _tiny_llama()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    planner = QVQTransformPlanner(model, LlamaQVQTransformImplementor())
    metadata = planner.rewrite_dense_weights(planner.build_transform_plan("A0"), seed=13)

    assert metadata == {"arm": "A0", "rewritten": False, "blockers": []}
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_qvq_residual_rewrite_materializes_ordinary_untied_lm_head_parameter():
    model = _tiny_llama()
    assert model.model.embed_tokens.weight is model.lm_head.weight
    planner = QVQTransformPlanner(model, LlamaQVQTransformImplementor())

    planner.rewrite_dense_weights(planner.build_transform_plan("A1"), seed=13)

    assert model.model.embed_tokens.weight is not model.lm_head.weight
    assert not torch.is_inference(model.lm_head.weight)
    model.half()
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([[1, 2, 3]])).logits
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("arm", ("A2", "A8", "A9"))
def test_qvq_learned_folded_arms_fail_closed_without_a_fitted_basis(arm):
    model = _tiny_llama()
    planner = QVQTransformPlanner(model, LlamaQVQTransformImplementor())

    with pytest.raises(NotImplementedError, match="refuses to substitute"):
        planner.rewrite_dense_weights(planner.build_transform_plan(arm), seed=9)


def test_qvq_a33_requires_fitted_basis_metadata():
    model = _tiny_llama()
    planner = QVQTransformPlanner(model, LlamaQVQTransformImplementor())

    with pytest.raises(NotImplementedError, match="fitted_residual_basis"):
        planner.rewrite_dense_weights(planner.build_transform_plan("A33"), seed=9)


def test_qvq_a33_fitted_structured_basis_preserves_final_logits():
    original = _tiny_llama()
    rewritten = copy.deepcopy(original)
    input_ids = torch.tensor([[1, 17, 9, 41, 3], [1, 5, 6, 7, 8]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        expected = original(input_ids=input_ids, attention_mask=attention_mask).logits

    planner = QVQTransformPlanner(rewritten, LlamaQVQTransformImplementor())
    plan = planner.build_transform_plan("A33")
    plan_metadata = copy.deepcopy(plan.metadata)
    plan_metadata["fitted_residual_basis"] = {
        "schema": "qvq.signed-permutation.v1",
        "core_seed": 29,
        "adaptation_seed": 101,
    }
    metadata = planner.rewrite_dense_weights(
        replace(plan, metadata=plan_metadata), seed=7
    )
    with torch.inference_mode():
        actual = rewritten(input_ids=input_ids, attention_mask=attention_mask).logits

    delta = actual - expected
    assert metadata["fitted_residual_basis"] == plan_metadata[
        "fitted_residual_basis"
    ]
    assert metadata["residual_condition_number"] == 1.0
    assert delta.abs().max().item() < 2e-5
    assert torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected) < 2e-5
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))


def test_qvq_a34_requires_one_fitted_adaptation_per_layer():
    model = _tiny_llama()
    planner = QVQTransformPlanner(model, LlamaQVQTransformImplementor())

    with pytest.raises(NotImplementedError, match="fitted_layer_bases"):
        planner.rewrite_dense_weights(planner.build_transform_plan("A34"), seed=9)


def test_qvq_a34_layer_bridges_are_dense_exact():
    original = _tiny_llama()
    rewritten = copy.deepcopy(original)
    input_ids = torch.tensor([[1, 17, 9, 41, 3], [1, 5, 6, 7, 8]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        expected = original(input_ids=input_ids, attention_mask=attention_mask).logits

    planner = QVQTransformPlanner(rewritten, LlamaQVQTransformImplementor())
    plan = planner.build_transform_plan("A34")
    plan_metadata = copy.deepcopy(plan.metadata)
    plan_metadata["fitted_layer_bases"] = {
        "schema": "qvq.layered-signed-permutation.v1",
        "core_seed": 29,
        "adaptation_seeds": [None, 71],
    }
    metadata = planner.rewrite_dense_weights(
        replace(plan, metadata=plan_metadata), seed=7
    )
    with torch.inference_mode():
        actual = rewritten(input_ids=input_ids, attention_mask=attention_mask).logits

    delta = actual - expected
    assert metadata["online_residual_bridges"] == 1
    assert metadata["residual_bridge_kind"] == "signed_permutation"
    assert len(rewritten._qvq_residual_bridge_handles) == 1
    assert delta.abs().max().item() < 2e-5
    assert torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected) < 2e-5
    assert torch.equal(actual.argmax(dim=-1), expected.argmax(dim=-1))


def test_qvq_a34_skips_identity_layer_bridges():
    model = _tiny_llama()
    planner = QVQTransformPlanner(model, LlamaQVQTransformImplementor())
    plan = planner.build_transform_plan("A34")
    plan_metadata = copy.deepcopy(plan.metadata)
    plan_metadata["fitted_layer_bases"] = {
        "schema": "qvq.layered-signed-permutation.v1",
        "core_seed": 29,
        "adaptation_seeds": [71, 71],
    }

    metadata = planner.rewrite_dense_weights(
        replace(plan, metadata=plan_metadata), seed=7
    )

    assert metadata["online_residual_bridges"] == 0
    assert model._qvq_residual_bridge_handles == ()


def test_qvq_qk_fold_fails_closed_with_intervening_norm():
    model = _tiny_llama()
    model.model.layers[0].self_attn.q_norm = torch.nn.Identity()
    planner = QVQTransformPlanner(model, LlamaQVQTransformImplementor())

    with pytest.raises(ValueError, match="q_norm/k_norm"):
        planner.rewrite_dense_weights(planner.build_transform_plan("A3"), seed=0)
