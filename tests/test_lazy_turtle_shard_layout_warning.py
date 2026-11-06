# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the LazyTurtle non-layer-wise shard warning."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import gptqmodel.utils.structure as structure


LLAMA_MODULE_TREE = [
    "model",
    "layers",
    "#",
    {
        "input_layernorm": ("input_layernorm:!:norm",),
        "self_attn": ("q_proj:0:q:in=x", "k_proj:0:k:in=x", "v_proj:0:v:in=x", "o_proj:1:o"),
        "post_attention_layernorm": ("post_attention_layernorm:!:norm",),
        "mlp": ("gate_proj:0:gate:in=x", "up_proj:0:up:in=x", "down_proj:1:down"),
    },
]


def _layer_weight(layer: int) -> str:
    return f"model.layers.{layer}.self_attn.q_proj.weight"


@pytest.fixture(autouse=True)
def _reset_warning_state():
    structure.reset_non_layerwise_shard_warning()
    yield
    structure.reset_non_layerwise_shard_warning()


@pytest.fixture
def warnings(monkeypatch):
    messages: list[str] = []

    def _capture(message, *args, **_kwargs):
        messages.append(message % args if args else message)

    monkeypatch.setattr(structure.log, "warning", _capture)
    return messages


def _write_checkpoint(
    model_dir: Path,
    shards: dict[str, list[str]],
    *,
    write_index: bool = True,
) -> Path:
    """Write a sharded safetensors checkpoint from shard name -> tensor names."""

    model_dir.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    for shard_name, tensor_names in shards.items():
        tensors = {name: torch.zeros(2, 2, dtype=torch.float32) for name in tensor_names}
        save_file(tensors, str(model_dir / shard_name))
        for name in tensor_names:
            weight_map[name] = shard_name
    if write_index:
        (model_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map}),
            encoding="utf-8",
        )
    return model_dir


def _build_turtle(model_dir: Path, *, module_tree=LLAMA_MODULE_TREE) -> structure.LazyTurtle:
    turtle = structure.LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        module_tree=module_tree,
    )
    assert turtle is not None
    return turtle


def test_monolithic_shard_warns_once(tmp_path, warnings):
    checkpoint = _write_checkpoint(
        tmp_path / "ckpt",
        {
            "model.safetensors": [
                _layer_weight(0),
                _layer_weight(1),
                _layer_weight(2),
                "model.embed_tokens.weight",
                "lm_head.weight",
            ]
        },
        write_index=False,
    )

    assert not (checkpoint / "model.safetensors.index.json").exists()

    turtle = _build_turtle(checkpoint)
    turtle.warn_if_checkpoint_shards_are_not_layerwise()

    assert len(warnings) == 1
    # The message must state the quantization-only context, the repeated-I/O cause,
    # and the remedy.
    assert "for quantization" in warnings[0]
    assert "shards are not layer-wise" in warnings[0]
    assert "`model.safetensors` holds 3 model layers" in warnings[0]
    assert "opened and re-read once per layer" in warnings[0]
    assert "avoidable I/O" in warnings[0]
    assert "re-shard so each layer has its own file" in warnings[0]
    # A single-file checkpoint should not report an awkward "1 of 1" count.
    assert "of 1 safetensors" not in warnings[0]

    # The warning is process-global: a second load must not repeat it.
    second = _build_turtle(checkpoint)
    second.warn_if_checkpoint_shards_are_not_layerwise()
    assert len(warnings) == 1


def test_per_layer_shards_do_not_warn(tmp_path, warnings):
    checkpoint = _write_checkpoint(
        tmp_path / "ckpt",
        {
            "model-layer-0.safetensors": [_layer_weight(0)],
            "model-layer-1.safetensors": [_layer_weight(1)],
            "model-shared.safetensors": ["model.embed_tokens.weight", "lm_head.weight"],
        },
    )

    turtle = _build_turtle(checkpoint)
    turtle.warn_if_checkpoint_shards_are_not_layerwise()

    assert warnings == []


def test_shard_spanning_two_layers_warns(tmp_path, warnings):
    checkpoint = _write_checkpoint(
        tmp_path / "ckpt",
        {
            "model-0.safetensors": [_layer_weight(0), _layer_weight(1)],
            "model-1.safetensors": [_layer_weight(2), _layer_weight(3)],
        },
    )

    turtle = _build_turtle(checkpoint)
    turtle.warn_if_checkpoint_shards_are_not_layerwise()

    assert len(warnings) == 1
    assert "2 of 2 safetensors files hold multiple model layers" in warnings[0]


def test_single_offending_shard_uses_singular_verb(tmp_path, warnings):
    checkpoint = _write_checkpoint(
        tmp_path / "ckpt",
        {
            "model-0.safetensors": [_layer_weight(0), _layer_weight(1)],
            "model-1.safetensors": [_layer_weight(2)],
            "model-2.safetensors": [_layer_weight(3)],
        },
    )

    turtle = _build_turtle(checkpoint)
    turtle.warn_if_checkpoint_shards_are_not_layerwise()

    assert len(warnings) == 1
    assert "1 of 3 safetensors files holds multiple model layers" in warnings[0]


def test_shared_tensors_in_a_layer_shard_do_not_count_as_a_second_layer(tmp_path, warnings):
    checkpoint = _write_checkpoint(
        tmp_path / "ckpt",
        {
            "model-0.safetensors": [
                _layer_weight(0),
                "model.embed_tokens.weight",
                "lm_head.weight",
            ],
            "model-1.safetensors": [_layer_weight(1)],
        },
    )

    turtle = _build_turtle(checkpoint)
    turtle.warn_if_checkpoint_shards_are_not_layerwise()

    assert warnings == []


def test_fallback_detects_common_layer_containers_without_module_tree(tmp_path, warnings):
    checkpoint = _write_checkpoint(
        tmp_path / "ckpt",
        {
            "model.safetensors": [
                "transformer.h.0.attn.c_attn.weight",
                "transformer.h.1.attn.c_attn.weight",
            ]
        },
    )

    turtle = _build_turtle(checkpoint, module_tree=None)
    assert turtle.checkpoint_shard_layer_groups() == {
        "model.safetensors": frozenset({"transformer.h.0", "transformer.h.1"})
    }

    turtle.warn_if_checkpoint_shards_are_not_layerwise()
    assert len(warnings) == 1


def test_encoder_layers_use_model_layer_wording(tmp_path, warnings):
    checkpoint = _write_checkpoint(
        tmp_path / "ckpt",
        {
            "model.safetensors": [
                "transformer.encoder.layers.0.self_attention.query_key_value.weight",
                "transformer.encoder.layers.1.self_attention.query_key_value.weight",
            ]
        },
    )

    turtle = _build_turtle(
        checkpoint,
        module_tree=["transformer", "encoder", "layers", "#", {}],
    )
    turtle.warn_if_checkpoint_shards_are_not_layerwise()

    assert len(warnings) == 1
    assert "holds 2 model layers" in warnings[0]
    assert "decoder layers" not in warnings[0]


def test_moe_expert_indices_are_not_treated_as_layers(tmp_path):
    checkpoint = _write_checkpoint(
        tmp_path / "ckpt",
        {
            "model-0.safetensors": [
                "model.layers.0.mlp.experts.0.w1.weight",
                "model.layers.0.mlp.experts.1.w1.weight",
                _layer_weight(1),
            ],
        },
    )

    turtle = _build_turtle(checkpoint)
    groups = turtle.checkpoint_shard_layer_groups()

    assert groups["model-0.safetensors"] == frozenset({"model.layers.0", "model.layers.1"})


def test_reset_allows_the_warning_to_fire_again(tmp_path, warnings):
    checkpoint = _write_checkpoint(
        tmp_path / "ckpt",
        {"model.safetensors": [_layer_weight(0), _layer_weight(1)]},
    )

    turtle = _build_turtle(checkpoint)
    turtle.warn_if_checkpoint_shards_are_not_layerwise()
    assert len(warnings) == 1

    structure.reset_non_layerwise_shard_warning()
    turtle.warn_if_checkpoint_shards_are_not_layerwise()
    assert len(warnings) == 2
