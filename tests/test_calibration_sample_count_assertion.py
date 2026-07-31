# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for calibration sample count assertions."""

from unittest.mock import MagicMock

import pytest

from gptqmodel.looper.loop_processor import LoopProcessor
from gptqmodel.models.definitions.laguna import LagunaQModel
from gptqmodel.quantization.config import (
    BaseMoERouting,
    ExpertsRoutingBypass,
    MoEConfig,
    QuantizeConfig,
)


@pytest.fixture
def processor():
    """A minimal LoopProcessor wired with a real MoE model definition."""

    qcfg = QuantizeConfig()
    p = LoopProcessor(None, qcfg, None)
    # Use LagunaQModel because its module_tree carries the :moe flag, so
    # _module_is_moe_related derives membership from module_tree flags rather
    # than from hard-coded name patterns.
    p.gptq_model = object.__new__(LagunaQModel)
    p.gptq_model.moe_lifecycle_hooks = MagicMock()
    p.gptq_model.moe_lifecycle_hooks.expert_block_names = ["experts"]
    p.gptq_model.moe_lifecycle_hooks.shared_expert_block_names = ["shared_expert"]
    p._global_max_padded_nsamples = 1024
    return p


def test_non_moe_modules_require_constant_sample_count(processor):
    """Dense modules must all report the same number of calibration samples."""

    processor._assert_calibration_sample_count("model.layers.0.self_attn.q_proj", 1024)
    processor._assert_calibration_sample_count("model.layers.0.self_attn.k_proj", 1024)
    assert processor._global_reference_nsamples == 1024

    with pytest.raises(AssertionError):
        processor._assert_calibration_sample_count("model.layers.0.self_attn.v_proj", 512)


def test_partial_sample_count_after_reference_is_a_regression(processor):
    """A dense module with a partial non-zero count indicates a sample leak."""

    processor._assert_calibration_sample_count("model.layers.0.self_attn.q_proj", 1024)
    with pytest.raises(AssertionError):
        processor._assert_calibration_sample_count("model.layers.0.self_attn.k_proj", 512)


def test_unactivated_dense_module_allowed(processor):
    """A dense module that is never activated can legitimately report 0 samples."""

    processor._assert_calibration_sample_count("model.layers.0.self_attn.q_proj", 1024)
    processor._assert_calibration_sample_count("model.layers.0.self_attn.k_proj", 0)


def test_moe_bypass_experts_require_constant_sample_count(processor):
    """When routing=bypass every expert must see the same token count."""

    processor.qcfg.moe = MoEConfig(routing=ExpertsRoutingBypass())
    processor._assert_calibration_sample_count("model.layers.0.mlp.experts.0.gate_proj", 1024)
    processor._assert_calibration_sample_count("model.layers.0.mlp.experts.12.down_proj", 1024)
    assert processor._global_reference_nsamples == 1024
    # An unactivated bypass expert can legitimately report 0 samples when the
    # all-experts replay falls back to native routed forward.
    processor._assert_calibration_sample_count("model.layers.0.mlp.experts.198.down_proj", 0)

    with pytest.raises(AssertionError):
        processor._assert_calibration_sample_count("model.layers.0.mlp.experts.1.gate_proj", 512)


def test_normal_moe_routing_allows_expert_counts_below_reference(processor):
    """Routed (non-bypass) experts may see fewer tokens than the dense reference."""

    processor.qcfg.moe = MoEConfig(routing=BaseMoERouting())
    processor._assert_calibration_sample_count("model.layers.0.mlp.gate", 1024)
    processor._assert_calibration_sample_count("model.layers.0.mlp.experts.0.gate_proj", 256)
    processor._assert_calibration_sample_count("model.layers.0.mlp.experts.1.gate_proj", 0)


def test_normal_moe_routing_rejects_expert_counts_above_reference(processor):
    """No expert may exceed the global reference; that would imply double counting."""

    processor.qcfg.moe = MoEConfig(routing=BaseMoERouting())
    processor._assert_calibration_sample_count("model.layers.0.mlp.gate", 1024)
    with pytest.raises(AssertionError):
        processor._assert_calibration_sample_count("model.layers.0.mlp.experts.0.gate_proj", 1500)


def test_normal_moe_routing_allows_padded_token_counts(processor):
    """Flattened MoE activations may include padded positions up to the padded upper bound."""

    processor.qcfg.moe = MoEConfig(routing=BaseMoERouting())
    processor._assert_calibration_sample_count("model.layers.0.self_attn.q_proj", 1024)
    processor._assert_calibration_sample_count("model.layers.0.mlp.gate", 1024)
    processor._assert_calibration_sample_count("model.layers.0.mlp.experts.0.gate_proj", 1024)


def test_normal_moe_routing_rejects_counts_above_padded_bound(processor):
    """MoE modules may not exceed the total padded token count for the dataset."""

    processor.qcfg.moe = MoEConfig(routing=BaseMoERouting())
    processor._assert_calibration_sample_count("model.layers.0.self_attn.q_proj", 1024)
    with pytest.raises(AssertionError):
        processor._assert_calibration_sample_count("model.layers.0.mlp.gate", 2048)


def test_native_moe_routing_allows_increase_below_padded_bound(processor):
    """Native-routed MoE counts may exceed the dense reference up to the padded bound."""

    processor.qcfg.moe = MoEConfig(routing=BaseMoERouting())
    processor._global_max_padded_nsamples = 2048
    processor._assert_calibration_sample_count("model.layers.0.self_attn.q_proj", 1024)
    processor._assert_calibration_sample_count("model.layers.0.mlp.experts.0.gate_proj", 1500)
    with pytest.raises(AssertionError):
        processor._assert_calibration_sample_count("model.layers.0.mlp.experts.0.gate_proj", 2500)


def test_embedding_module_uses_padded_token_count(processor):
    """Input embeddings count all token positions including padding."""

    processor.total_calibration_tokens = 512
    processor._global_max_padded_nsamples = 1024
    processor._input_embeddings_name = "model.embed_tokens"
    processor._assert_calibration_sample_count("model.embed_tokens", 1024)

    with pytest.raises(AssertionError):
        processor._assert_calibration_sample_count("model.embed_tokens", 512)


def test_weight_only_processor_skips_sample_validation():
    """Weight-only processors never see calibration forwards, so they skip validation."""

    from gptqmodel.looper.weight_only_processor import WeightOnlyProcessor
    from gptqmodel.quantization.config import RTNConfig

    p = WeightOnlyProcessor(None, RTNConfig())
    p._assert_calibration_sample_count("model.layers.0.mlp.gate_proj", 0)
    p._assert_calibration_sample_count("model.layers.0.mlp.down_proj", 1024)
    assert p._global_reference_nsamples is None
