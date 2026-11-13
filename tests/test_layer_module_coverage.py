# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from types import SimpleNamespace

import pytest
import torch.nn as nn

from gptqmodel.looper import module_coverage
from gptqmodel.looper.module_coverage import (
    LayerModuleCoverageTracker,
    normalize_layer_module_name,
)
from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.looper.weight_only_looper import WeightOnlyLooper


class _RecordingLogger:
    def __init__(self):
        self.warnings = []
        self.infos = []
        self.debugs = []

    def warn(self, message):
        self.warnings.append(message)

    def info(self, message):
        self.infos.append(message)

    def debug(self, message):
        self.debugs.append(message)


def test_normalize_layer_module_name_folds_expanded_experts():
    assert (
        normalize_layer_module_name("mlp.experts.17.gate_proj")
        == "mlp.experts.#.gate_proj"
    )
    assert (
        normalize_layer_module_name("block_sparse_moe.experts.255.down_proj")
        == "block_sparse_moe.experts.#.down_proj"
    )
    # Non-numeric names are untouched.
    assert normalize_layer_module_name("self_attn.q_proj") == "self_attn.q_proj"
    assert normalize_layer_module_name("input_linear.0.linear") == "input_linear.#.linear"


def test_tracker_warns_once_per_never_matched_spec(monkeypatch):
    logger = _RecordingLogger()
    monkeypatch.setattr(module_coverage, "log", logger)

    tracker = LayerModuleCoverageTracker()
    tracker.record_hit("self_attn.q_proj", 0)
    tracker.record_hit("self_attn.q_proj", 1)
    # 256 expanded experts missing in 2 layers -> one warning entry.
    for layer_index in (0, 1):
        for expert_index in range(256):
            tracker.record_miss(f"mlp.experts.{expert_index}.gate_proj", layer_index)
    tracker.record_hit("mlp.shared_experts.gate_proj", 0)
    tracker.record_miss("mlp.shared_experts.gate_proj", 1)

    tracker.report("DummyQModel")

    assert len(logger.warnings) == 2  # summary + one stale spec
    assert "matched no layer" in logger.warnings[0]
    assert "mlp.experts.0.gate_proj" in logger.warnings[1]
    assert "0 layer matches, 2 miss(es)" in logger.warnings[1]
    # Partial matches are summarized, not warned per spec.
    assert any("matched only some layers" in message for message in logger.infos)


def test_tracker_reports_all_matched(monkeypatch):
    logger = _RecordingLogger()
    monkeypatch.setattr(module_coverage, "log", logger)

    tracker = LayerModuleCoverageTracker()
    tracker.record_hit("self_attn.q_proj", 0)
    tracker.report()

    assert logger.warnings == []
    assert any("matched at least one layer" in message for message in logger.infos)


def test_weight_only_looper_records_missing_spec():
    tracker = LayerModuleCoverageTracker()
    looper = WeightOnlyLooper.__new__(WeightOnlyLooper)
    looper.gptq_model = SimpleNamespace(layer_modules_strict=False)
    looper.layer_module_coverage = tracker

    resolved = looper._resolve_named_module(
        layer_module=nn.Module(),
        full={},
        layer_index=3,
        layer_path="model.layers.3",
        module_name="mlp.experts.7.gate_proj",
        is_lm_head_module=False,
    )

    assert resolved is None
    snapshot = tracker.snapshot()
    assert snapshot["mlp.experts.#.gate_proj"]["hit_layers"] == 0
    assert snapshot["mlp.experts.#.gate_proj"]["miss_layers"] == 1


def test_weight_only_looper_strict_still_raises():
    looper = WeightOnlyLooper.__new__(WeightOnlyLooper)
    looper.gptq_model = SimpleNamespace(layer_modules_strict=True)
    looper.layer_module_coverage = LayerModuleCoverageTracker()

    with pytest.raises(ValueError, match="not found in model"):
        looper._resolve_named_module(
            layer_module=nn.Module(),
            full={},
            layer_index=0,
            layer_path="model.layers.0",
            module_name="mlp.missing",
            is_lm_head_module=False,
        )


def test_module_looper_create_named_modules_records_missing_spec():
    tracker = LayerModuleCoverageTracker()
    looper = ModuleLooper.__new__(ModuleLooper)
    looper.gptq_model = SimpleNamespace(layer_modules_strict=False)
    looper.layer_module_coverage = tracker
    looper.input_embeddings_name = None
    looper.output_embeddings_name = None

    subset = looper.create_named_modules(
        module=nn.Module(),
        full={},
        is_lm_head_module=False,
        layer_index=5,
        layers_prefix="model.layers.5",
        names=["mlp.experts.0.gate_proj"],
        processor=None,
        fallback=None,
    )

    assert subset == {}
    snapshot = tracker.snapshot()
    assert snapshot["mlp.experts.#.gate_proj"]["miss_layers"] == 1


def test_mixed_expert_block_does_not_duplicate_non_expert_specs():
    """Non-placeholder specs must be emitted once, not once per routed expert.

    ``build_moe_modules_if_need`` expands a block as soon as one name carries
    ``{expert_index}``, but the same block also holds shared-expert and dense
    MLP entries. Expanding those with the expert count inflated the planned
    spec list 3x for `afmoe` at 192 experts.
    """
    from collections import Counter

    from gptqmodel.models.definitions.afmoe import AfMoeQModel

    config = SimpleNamespace(num_experts=8, num_local_experts=8, num_shared_experts=1)
    blocks = AfMoeQModel.simple_layer_modules(
        model_config=config,
        quantize_config=SimpleNamespace(dynamic=None),
    )

    for index, block in enumerate(blocks):
        duplicates = {name: count for name, count in Counter(block).items() if count > 1}
        assert not duplicates, f"block {index} repeats specs: {duplicates}"

    flat = {name for block in blocks for name in block}
    # The non-expert entries still have to survive the expansion.
    assert "mlp.shared_experts.gate_proj" in flat
    assert "mlp.gate_proj" in flat
    assert "mlp.down_proj" in flat
    # ... and the placeholder still expands for every expert.
    assert {f"mlp.experts.{expert}.gate_proj" for expert in range(8)} <= flat
