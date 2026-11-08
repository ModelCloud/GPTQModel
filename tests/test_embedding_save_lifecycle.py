# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from gptqmodel.models import writer
from gptqmodel.models.base import BaseQModel
from gptqmodel.models.writer import _save_module_replacement_safetensors


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

    rewritten, weight_map, _size, removed = _save_module_replacement_safetensors(
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


def test_embedding_replacement_write_failure_keeps_existing_shard(tmp_path, monkeypatch):
    source = tmp_path / "source"
    saved = tmp_path / "saved"
    source.mkdir()
    saved.mkdir()
    shard_name = "model.safetensors"
    tensors = {"embed_tokens.weight": torch.ones(4, 3), "final_norm.weight": torch.ones(3)}
    save_file(tensors, str(source / shard_name))
    save_file(tensors, str(saved / shard_name))
    original = (saved / shard_name).read_bytes()

    model = nn.Module()
    model.embed_tokens = nn.Module()
    model.embed_tokens.register_buffer("qweight", torch.ones(2, 2, dtype=torch.int32))
    turtle = SimpleNamespace(
        model_local_path=str(source),
        _weight_map={"embed_tokens.weight": shard_name, "final_norm.weight": shard_name},
    )

    def fail_after_open(_tensors, path, metadata):
        with open(path, "wb") as handle:
            handle.write(b"partial shard")
        raise OSError("injected write failure")

    monkeypatch.setattr(writer, "save_file", fail_after_open)
    with pytest.raises(OSError, match="injected write failure"):
        _save_module_replacement_safetensors(
            model, turtle, ["embed_tokens"], save_dir=str(saved), metadata={"format": "pt"}
        )

    assert (saved / shard_name).read_bytes() == original
    assert sorted(path.name for path in saved.iterdir()) == [shard_name]


def test_embedding_replacement_removes_old_keys_from_another_indexed_shard(tmp_path):
    source = tmp_path / "source"
    saved = tmp_path / "saved"
    source.mkdir()
    saved.mkdir()
    first = "model-00001-of-00002.safetensors"
    second = "model-00002-of-00002.safetensors"
    index_name = "model.safetensors.index.json"
    save_file({"embed_tokens.weight": torch.ones(4, 3)}, source / first)
    save_file({"final_norm.weight": torch.ones(3)}, source / second)
    save_file({"embed_tokens.qweight": torch.ones(2, 2, dtype=torch.int32)}, saved / first)
    save_file(
        {"embed_tokens.old_scale": torch.ones(2), "final_norm.weight": torch.ones(3)},
        saved / second,
    )
    (saved / index_name).write_text(json.dumps({"weight_map": {
        "embed_tokens.qweight": first,
        "embed_tokens.old_scale": second,
        "final_norm.weight": second,
    }}))

    model = nn.Module()
    model.embed_tokens = nn.Module()
    model.embed_tokens.register_buffer("qweight", torch.zeros(2, 2, dtype=torch.int32))
    turtle = SimpleNamespace(
        model_local_path=str(source),
        _weight_map={"embed_tokens.weight": first, "final_norm.weight": second},
        _index_filename=index_name,
    )

    rewritten, weight_map, _size, removed = _save_module_replacement_safetensors(
        model, turtle, ["embed_tokens"], save_dir=str(saved), metadata={"format": "pt"}
    )

    assert set(rewritten) == {first, second}
    assert "embed_tokens.old_scale" in removed
    assert "embed_tokens.old_scale" not in weight_map
    with safe_open(saved / first, framework="pt", device="cpu") as handler:
        assert set(handler.keys()) == {"embed_tokens.qweight"}
        torch.testing.assert_close(handler.get_tensor("embed_tokens.qweight"), model.embed_tokens.qweight)
    with safe_open(saved / second, framework="pt", device="cpu") as handler:
        assert set(handler.keys()) == {"final_norm.weight"}


@pytest.mark.parametrize(
    ("prefixes", "expected_removed", "expected_weight_map", "expected_keys"),
    [
        (
            ["embed_tokens", "lm_head"],
            ["embed_tokens.weight"],
            {
                "embed_tokens.qweight": "model.safetensors",
                "lm_head.weight": "model.safetensors",
            },
            {"embed_tokens.qweight", "lm_head.weight"},
        ),
        (
            ["embed_tokens"],
            ["embed_tokens.weight"],
            {
                "embed_tokens.qweight": "model.safetensors",
                "lm_head.weight": "model.safetensors",
            },
            {"embed_tokens.qweight", "lm_head.weight"},
        ),
        (
            ["lm_head"],
            [],
            {
                "embed_tokens.weight": "model.safetensors",
                "lm_head.weight": "model.safetensors",
            },
            {"embed_tokens.weight", "lm_head.weight"},
        ),
    ],
    ids=["both", "input-only", "output-only"],
)
def test_embedding_replacement_saves_tied_lm_head_omitted_from_checkpoint(
    tmp_path, prefixes, expected_removed, expected_weight_map, expected_keys
):
    source = tmp_path / "source"
    saved = tmp_path / "saved"
    source.mkdir()
    saved.mkdir()
    shard_name = "model.safetensors"
    save_file({"embed_tokens.weight": torch.ones(4, 3)}, str(source / shard_name))
    (source / "config.json").write_text('{"tie_word_embeddings": true}\n', encoding="utf-8")

    class TiedModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Requantization unties the runtime modules, while the source
            # checkpoint still records that the omitted output weight was tied.
            self.config = SimpleNamespace(tie_word_embeddings=False)
            self.embed_tokens = nn.Module()
            self.embed_tokens.register_buffer("qweight", torch.ones(2, 2, dtype=torch.int32))
            self.lm_head = nn.Linear(3, 4, bias=False)

        def get_input_embeddings(self):
            return self.embed_tokens

        def get_output_embeddings(self):
            return self.lm_head

    model = TiedModel()
    turtle = SimpleNamespace(
        model_local_path=str(source),
        _weight_map={"embed_tokens.weight": shard_name},
    )

    rewritten, weight_map, _size, removed = _save_module_replacement_safetensors(
        model,
        turtle,
        prefixes,
        save_dir=str(saved),
        metadata={"format": "pt"},
    )

    assert rewritten == [shard_name]
    assert removed == expected_removed
    assert weight_map == expected_weight_map
    with safe_open(str(saved / shard_name), framework="pt", device="cpu") as handler:
        assert set(handler.keys()) == expected_keys


def test_model_save_routes_embedding_only_lifecycle_with_metadata(tmp_path):
    model = BaseQModel.__new__(BaseQModel)
    nn.Module.__init__(model)
    model.quantized = True
    model._model_free_weight_only_replacements_only = True
    model.quant_override_files = {}
    model.quant_region_timer = None
    model.quantize_config = SimpleNamespace(dynamic=None)
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
