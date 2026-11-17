# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""#4395: untouched base modules must not be re-serialized into the offload dir.

Quantization offloaded ``embed_tokens``/the final norm by writing a byte
identical copy of the source checkpoint tensor. The offload index now points
at the already-authenticated source safetensors payload when every module
tensor still matches it; anything else falls back to a serialized bundle.
"""

from __future__ import annotations

import json
import os

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import CalibrationConfig
from gptqmodel.utils.offload import (
    offload_to_disk,
    release_private_offload_root,
    undo_offload_to_disk,
)


class _Holder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(128, 64)
        self.model.norm = torch.nn.LayerNorm(64)

    def tie_weights(self) -> None:  # pragma: no cover - interface match
        return None


def _holder() -> _Holder:
    torch.manual_seed(4395)
    return _Holder().eval()


def _index(path: str) -> dict:
    with open(os.path.join(path, "index.json"), encoding="utf-8") as handle:
        return json.load(handle)


def test_reference_offload_writes_only_index_and_restores_bitwise(tmp_path):
    model = _holder()
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    source = tmp_path / "model.safetensors"
    save_file(state, str(source))
    expected = model.model.embed_tokens.weight.detach().clone()

    offload_to_disk(
        module=["model.embed_tokens"],
        model=model,
        disk_path=str(tmp_path),
        source_root=str(tmp_path),
        source_checkpoint=str(source),
    )

    module_dir = tmp_path / "model.embed_tokens"
    assert sorted(os.listdir(module_dir)) == ["index.json"]
    entry = _index(str(module_dir))["weight"]
    assert entry["safetensors_file"] == str(source)
    assert entry["weight_name"] == "model.embed_tokens.weight"
    assert sum(os.path.getsize(module_dir / name) for name in os.listdir(module_dir)) < 4096
    assert model.model.embed_tokens.weight.is_meta

    undo_offload_to_disk(model.model.embed_tokens, device=torch.device("cpu"))
    assert torch.equal(model.model.embed_tokens.weight, expected)


def test_reference_offload_supports_sharded_sources(tmp_path):
    model = _holder()
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    save_file(
        {"model.embed_tokens.weight": state["model.embed_tokens.weight"]},
        str(tmp_path / "shard-0.safetensors"),
    )
    save_file(
        {name: value for name, value in state.items() if name != "model.embed_tokens.weight"},
        str(tmp_path / "shard-1.safetensors"),
    )
    weight_map = {
        name: ("shard-0.safetensors" if name == "model.embed_tokens.weight" else "shard-1.safetensors")
        for name in state
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    expected = model.model.embed_tokens.weight.detach().clone()

    offload_to_disk(
        module=["model.embed_tokens"],
        model=model,
        disk_path=str(tmp_path),
        source_root=str(tmp_path),
    )

    module_dir = tmp_path / "model.embed_tokens"
    assert sorted(os.listdir(module_dir)) == ["index.json"]
    assert _index(str(module_dir))["weight"]["safetensors_file"].endswith("shard-0.safetensors")
    undo_offload_to_disk(model.model.embed_tokens, device=torch.device("cpu"))
    assert torch.equal(model.model.embed_tokens.weight, expected)


def test_reference_offload_falls_back_when_content_differs(tmp_path):
    model = _holder()
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    source = tmp_path / "model.safetensors"
    save_file(state, str(source))
    model.model.embed_tokens.weight.data.add_(1.0)
    expected = model.model.embed_tokens.weight.detach().clone()

    offload_to_disk(
        module=["model.embed_tokens"],
        model=model,
        disk_path=str(tmp_path),
        source_root=str(tmp_path),
        source_checkpoint=str(source),
    )

    module_dir = tmp_path / "model.embed_tokens"
    assert "module.safetensors" in os.listdir(module_dir)
    undo_offload_to_disk(model.model.embed_tokens, device=torch.device("cpu"))
    assert torch.equal(model.model.embed_tokens.weight, expected)


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    words = (
        "[UNK]", "[PAD]", "<s>", "</s>", "A", "short", "calibration",
        "sentence", "about", "fox", "Another", "moon", ".", "runs",
    )
    inner = Tokenizer(
        models.WordLevel(vocab={word: index for index, word in enumerate(words)}, unk_token="[UNK]")
    )
    inner.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=inner,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="<s>",
        eos_token="</s>",
    )


def test_quantize_references_source_embedding_in_offload_dir(tmp_path):
    from gptqmodel import BACKEND, GPTQModel
    from gptqmodel.quantization.config import QuantizeConfig

    torch.manual_seed(4395)
    source = tmp_path / "source"
    saved = tmp_path / "saved"
    source.mkdir()
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=1,
    )
    LlamaForCausalLM(config).save_pretrained(source)
    _tiny_tokenizer().save_pretrained(source)
    from safetensors import safe_open

    with safe_open(str(source / "model.safetensors"), framework="pt", device="cpu") as handle:
        expected_embedding = handle.get_tensor("model.embed_tokens.weight").clone()

    offload_root = tmp_path / "offload"
    model = GPTQModel.load(
        str(source),
        QuantizeConfig(bits=4, group_size=32, offload_to_disk=True, offload_to_disk_path=str(offload_root)),
        device="cpu",
        dtype=torch.float32,
    )
    model.quantize(
        CalibrationConfig(
            dataset=["A short calibration sentence about a fox. A short calibration sentence about a fox."],
            batch_size=1,
        ),
        backend=BACKEND.GPTQ_TORCH,
    )
    try:
        run_root = model._private_offload_run_root
        assert run_root is not None
        module_dir = os.path.join(run_root, "model.embed_tokens")
        assert sorted(os.listdir(module_dir)) == ["index.json"]
        entry = _index(module_dir)["weight"]
        assert entry["safetensors_file"] == str(source / "model.safetensors")

        model.save(str(saved))
        with safe_open(str(saved / "model.safetensors"), framework="pt", device="cpu") as handle:
            saved_embedding = handle.get_tensor("model.embed_tokens.weight")
        assert torch.equal(saved_embedding, expected_embedding)
    finally:
        release_private_offload_root(model)
