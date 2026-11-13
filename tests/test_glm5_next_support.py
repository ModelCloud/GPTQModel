# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models.definitions.glm5_next import (
    GLM5_NEXT_CAPTURE_INPUT_IDS,
    GLM5_NEXT_ROUTED_EXPERT_PATTERN,
    Glm5NextQModel,
    _prune_glm5_next_replay_frontier,
)


def test_routed_expert_pattern_includes_target_and_checkpoint_mtp_only():
    import re

    pattern = re.compile(GLM5_NEXT_ROUTED_EXPERT_PATTERN)
    assert pattern.fullmatch(
        "model.language_model.layers.3.mlp.experts.0.gate_proj"
    )
    assert pattern.fullmatch(
        "model.language_model.layers.45.mlp.experts.287.down_proj"
    )
    assert not pattern.fullmatch(
        "model.language_model.layers.2.mlp.experts.0.gate_proj"
    )
    assert not pattern.fullmatch(
        "model.language_model.layers.46.mlp.experts.0.gate_proj"
    )
    assert not pattern.fullmatch(
        "model.language_model.layers.3.mlp.shared_experts.gate_proj"
    )


def test_route_recovery_resolves_nested_mtp_as_conventional_replay():
    layers = [nn.Identity() for _ in range(46)]
    adapter = Glm5NextQModel.__new__(Glm5NextQModel)
    adapter.model = SimpleNamespace(
        model=SimpleNamespace(
            language_model=SimpleNamespace(layers=nn.ModuleList(layers))
        )
    )

    assert adapter.zero_route_recovery_block_identity(layers[3]) == (
        "base",
        3,
        "model.language_model.layers.3.mlp.experts.",
    )
    assert adapter.zero_route_recovery_block_identity(layers[45]) == (
        "base",
        45,
        "model.language_model.layers.45.mlp.experts.",
    )
    with pytest.raises(RuntimeError, match="canonical decoder block"):
        adapter.zero_route_recovery_block_identity(nn.Identity())


def test_replay_frontier_prunes_only_older_exact_layer_directories(tmp_path):
    old_complete = tmp_path / "layer-000003"
    old_partial = tmp_path / ".layer-000004.partial"
    current = tmp_path / "layer-000005"
    future = tmp_path / "layer-000006"
    unrelated = tmp_path / "layer-not-a-frontier"
    for path in (old_complete, old_partial, current, future, unrelated):
        path.mkdir()

    _prune_glm5_next_replay_frontier(tmp_path, before_layer=5)

    assert not old_complete.exists()
    assert not old_partial.exists()
    assert current.is_dir()
    assert future.is_dir()
    assert unrelated.is_dir()


def test_capture_preserves_original_token_ids_for_mtp_replay():
    adapter = Glm5NextQModel.__new__(Glm5NextQModel)
    input_ids = torch.tensor([[11, 12, 13, 14]])

    adapter.begin_input_capture_example(
        {"input_ids": input_ids}, batch_device=torch.device("cpu")
    )
    captured = adapter.capture_first_layer_input_kwargs(
        args=(),
        kwargs={},
        batch_device=torch.device("cpu"),
        layer_input_kwargs={"ordinary": True},
    )
    adapter.end_input_capture_example()

    assert captured["ordinary"] is True
    assert torch.equal(captured[GLM5_NEXT_CAPTURE_INPUT_IDS], input_ids)
    assert adapter._glm5_next_capture_input_ids is None


def test_prepare_replay_exposes_shifted_tokens_only_to_mtp():
    adapter = Glm5NextQModel.__new__(Glm5NextQModel)
    input_ids = torch.tensor([[11, 12, 13, 14]])
    common = {
        GLM5_NEXT_CAPTURE_INPUT_IDS: input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "position_ids": torch.arange(4).unsqueeze(0),
        "position_embeddings": None,
        "prev_topk_indices": torch.tensor([[2, 3]]),
        "use_cache": False,
    }

    ordinary = adapter.prepare_layer_replay_kwargs(
        layer=SimpleNamespace(layer_idx=44),
        layer_input=[torch.empty(1, 4, 2, 8)],
        additional_inputs=dict(common),
        target_device=torch.device("cpu"),
    )
    assert GLM5_NEXT_CAPTURE_INPUT_IDS not in ordinary
    assert "input_ids" not in ordinary
    assert ordinary["attention_mask"].shape == (1, 4)

    mtp = adapter.prepare_layer_replay_kwargs(
        layer=SimpleNamespace(layer_idx=45),
        layer_input=[torch.empty(1, 4, 2, 8)],
        additional_inputs=dict(common),
        target_device=torch.device("cpu"),
    )
    assert torch.equal(mtp["input_ids"], torch.tensor([[12, 13, 14]]))
    assert torch.equal(mtp["attention_mask"], torch.ones(1, 3, dtype=torch.long))
    assert torch.equal(mtp["position_ids"], torch.tensor([[1, 2, 3]]))
    assert "prev_topk_indices" not in mtp
    assert "position_embeddings" not in mtp
    assert GLM5_NEXT_CAPTURE_INPUT_IDS not in mtp


def test_target_terminal_layer_discards_router_state_without_slicing_tokens():
    adapter = Glm5NextQModel.__new__(Glm5NextQModel)
    input_ids = torch.tensor([[11, 12, 13, 14]])
    kwargs = {
        GLM5_NEXT_CAPTURE_INPUT_IDS: input_ids,
        "prev_topk_indices": torch.tensor([[2, 3]]),
    }

    result = adapter.update_layer_replay_kwargs_from_output(
        layer=SimpleNamespace(layer_idx=44),
        layer_output=(torch.empty(1, 4, 2, 8), None),
        layer_input_kwargs=kwargs,
        target_device=torch.device("cpu"),
    )

    assert "prev_topk_indices" not in result
    assert torch.equal(result[GLM5_NEXT_CAPTURE_INPUT_IDS], input_ids)


def test_prepare_replay_slices_route_capture_mask_only_for_mtp():
    adapter = Glm5NextQModel.__new__(Glm5NextQModel)
    keep_mask = torch.tensor([[False, True, True, False]])
    hidden = [torch.empty(1, 4, 2, 8)]

    ordinary = adapter.prepare_layer_replay_keep_mask(
        layer=SimpleNamespace(layer_idx=44),
        layer_input=hidden,
        additional_inputs={},
        keep_mask=keep_mask,
        target_device=torch.device("cpu"),
    )
    assert ordinary is keep_mask

    mtp = adapter.prepare_layer_replay_keep_mask(
        layer=SimpleNamespace(layer_idx=45),
        layer_input=hidden,
        additional_inputs={},
        keep_mask=keep_mask,
        target_device=torch.device("cpu"),
    )
    assert torch.equal(mtp, torch.tensor([[True, True, False]]))
