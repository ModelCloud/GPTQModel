# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from gptqmodel.models.base import BaseQModel
from gptqmodel.models.writer import _save_embedding_replacement_safetensors


def test_embedding_replacement_rewrites_only_affected_shard(tmp_path):
    source = tmp_path / "source"
    saved = tmp_path / "saved"
    source.mkdir()
    saved.mkdir()
    save_file(
        {"embed_tokens.weight": torch.ones(4, 3), "final_norm.weight": torch.ones(3)},
        str(source / "model-00001-of-00002.safetensors"),
    )
    save_file(
        {"layers.0.linear.weight": torch.full((3, 3), 2.0)},
        str(source / "model-00002-of-00002.safetensors"),
    )
    model = nn.Module()
    model.embed_tokens = nn.Module()
    model.embed_tokens.register_buffer("qweight", torch.ones(2, 2, dtype=torch.int32))
    turtle = SimpleNamespace(
        model_local_path=str(source),
        _weight_map={
            "embed_tokens.weight": "model-00001-of-00002.safetensors",
            "final_norm.weight": "model-00001-of-00002.safetensors",
            "layers.0.linear.weight": "model-00002-of-00002.safetensors",
        },
    )

    rewritten, weight_map, _size, removed = _save_embedding_replacement_safetensors(
        model,
        turtle,
        ["embed_tokens"],
        save_dir=str(saved),
        metadata={"format": "pt"},
    )

    assert rewritten == ["model-00001-of-00002.safetensors"]
    assert removed == ["embed_tokens.weight"]
    assert weight_map["embed_tokens.qweight"] == "model-00001-of-00002.safetensors"
    assert not (saved / "model-00002-of-00002.safetensors").exists()
    with safe_open(str(saved / rewritten[0]), framework="pt", device="cpu") as handler:
        assert set(handler.keys()) == {"embed_tokens.qweight", "final_norm.weight"}
    with safe_open(str(source / rewritten[0]), framework="pt", device="cpu") as handler:
        assert set(handler.keys()) == {"embed_tokens.weight", "final_norm.weight"}


def test_model_save_routes_embedding_only_lifecycle_with_metadata(tmp_path):
    model = BaseQModel.__new__(BaseQModel)
    nn.Module.__init__(model)
    model.quantized = True
    model._model_free_weight_only_embeddings_only = True
    model.quant_override_files = {}
    model.quant_region_timer = None
    captured = {}
    model.save_quantized_embeddings = lambda **kwargs: captured.update(kwargs)
    model.save_quantized = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("embedding-only lifecycle must not use the full save path")
    )

    model.save(str(tmp_path / "saved"), meta_quantizer="test-quantizer:1.0")

    assert captured == {
        "save_dir": str(tmp_path / "saved"),
        "safetensors_metadata": None,
        "max_shard_size": "4GB",
        "meta_quantizer": "test-quantizer:1.0",
    }
