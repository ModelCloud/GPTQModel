# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""A legacy-layout checkpoint must load without copying its shards (#4959).

``GPTQModel.load`` used to rewrite every safetensors header into a temporary
file, copying the whole data section, just to rename legacy
``language_model.model.*`` keys to the runtime ``model.language_model.*``
layout. For a multi-GiB VL checkpoint that was ~80% of the load time.

The lazy path reads the source shards with ``safe_open`` and places the
renamed tensors with Accelerate's ``set_module_tensor_to_device``. These cases
pin the contract:

* no shard data is copied,
* the loaded tensors are bit-identical to the source file,
* logits match the copy-based fallback,
* a re-save still round-trips.
"""

import os

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText

import gptqmodel.utils.model as gm
from gptqmodel import BACKEND, CalibrationConfig, GPTQModel, QuantizeConfig
from gptqmodel.models.auto import MODEL_MAP_CASEFOLD
from gptqmodel.models.loader import _checkpoint_key_converter


_TEXT_CONFIG = {
    "vocab_size": 16,
    "hidden_size": 8,
    "intermediate_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "max_position_embeddings": 16,
}

_VISION_CONFIG = {
    "hidden_size": 8,
    "intermediate_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "image_size": 4,
    "patch_size": 2,
}

RUNTIME_PREFIX = "model.language_model.layers.0.mlp.down_proj"
LEGACY_PREFIX = "language_model.model.layers.0.mlp.down_proj"


def _gemma3_config():
    from transformers import Gemma3Config

    return Gemma3Config(
        text_config=dict(_TEXT_CONFIG, head_dim=4),
        vision_config=dict(_VISION_CONFIG),
        mm_tokens_per_image=4,
    )


def _tiny_model():
    torch.manual_seed(0)
    return AutoModelForImageTextToText.from_config(_gemma3_config())


def _quantize_and_save(tmp_path, monkeypatch):
    definition = MODEL_MAP_CASEFOLD["gemma3"]
    monkeypatch.setattr(definition, "require_load_processor", False, raising=False)

    dense_dir = tmp_path / "dense"
    quantized_dir = tmp_path / "quantized"
    model = _tiny_model()
    model.save_pretrained(dense_dir)

    wrapped = GPTQModel.load(
        str(dense_dir),
        quantize_config=QuantizeConfig(
            bits=4,
            group_size=-1,
            desc_act=False,
            sym=True,
            device="cpu",
        ),
        backend=BACKEND.GPTQ_TORCH,
    )
    wrapped.quantize(
        CalibrationConfig(
            dataset=[
                {
                    "input_ids": torch.tensor(
                        [[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long
                    ),
                    "attention_mask": torch.ones((1, 8), dtype=torch.long),
                }
            ],
            batch_size=1,
            min_length=1,
        ),
        backend=BACKEND.GPTQ_TORCH,
    )
    wrapped.save(quantized_dir)
    return quantized_dir, wrapped


def _rewrite_to_legacy_layout(checkpoint_dir, wrapped):
    definition = MODEL_MAP_CASEFOLD["gemma3"]
    to_checkpoint = _checkpoint_key_converter(definition, wrapped.model, to_runtime=False)
    source = checkpoint_dir / "model.safetensors"
    tensors = load_file(source)
    legacy = {to_checkpoint(name): tensor for name, tensor in tensors.items()}
    assert len(legacy) == len(tensors)
    save_file(legacy, source)
    return legacy


def _load(path):
    return GPTQModel.load(str(path), backend=BACKEND.GPTQ_TORCH, device="cpu")


def _runtime_state(checkpoint_dir, wrapped):
    definition = MODEL_MAP_CASEFOLD["gemma3"]
    to_runtime = _checkpoint_key_converter(definition, wrapped.model, to_runtime=True)
    return {
        to_runtime(name): tensor
        for name, tensor in load_file(checkpoint_dir / "model.safetensors").items()
    }


def _assert_same_tensors(left, right):
    assert set(left) == set(right)
    for name, tensor in left.items():
        other = right[name]
        assert tensor.shape == other.shape, name
        assert tensor.dtype == other.dtype, name
        assert torch.equal(tensor, other), name


def _force_copy_path(monkeypatch):
    """Disable the lazy path so the same checkpoint loads through the old copy."""

    lazy = getattr(gm, "_load_key_converted_safetensors", None)
    if lazy is not None:
        monkeypatch.setattr(gm, "_load_key_converted_safetensors", lambda *a, **k: False)


def test_legacy_layout_load_does_not_copy_shards(tmp_path, monkeypatch):
    quantized_dir, wrapped = _quantize_and_save(tmp_path, monkeypatch)
    _rewrite_to_legacy_layout(quantized_dir, wrapped)

    copied = {"calls": 0, "bytes": 0}
    original = gm._write_safetensors_with_converted_header

    def _record(source_path, target_path, key_converter, exclude_names=frozenset()):
        copied["calls"] += 1
        copied["bytes"] += os.path.getsize(source_path)
        return original(source_path, target_path, key_converter, exclude_names)

    monkeypatch.setattr(gm, "_write_safetensors_with_converted_header", _record)

    loaded = _load(quantized_dir)

    assert copied["calls"] == 0, (
        f"legacy-layout load copied {copied['calls']} shard(s), "
        f"{copied['bytes']} bytes"
    )
    assert any(
        module.__class__.__name__ == "TorchLinear"
        for module in loaded.model.modules()
    )

    # The checkpoint view must be bit-identical to the source shards.
    _assert_same_tensors(
        _runtime_state(quantized_dir, wrapped),
        {
            name: tensor.cpu()
            for name, tensor in loaded.model.state_dict().items()
            if name in _runtime_state(quantized_dir, wrapped)
        },
    )


def test_legacy_layout_load_matches_copy_reference(tmp_path, monkeypatch):
    quantized_dir, wrapped = _quantize_and_save(tmp_path, monkeypatch)
    _rewrite_to_legacy_layout(quantized_dir, wrapped)

    lazy_model = _load(quantized_dir)

    _force_copy_path(monkeypatch)
    copied_model = _load(quantized_dir)

    input_ids = torch.tensor([[1, 3, 4, 5]])
    with torch.inference_mode():
        lazy_logits = lazy_model.model(input_ids=input_ids, use_cache=False).logits.float()
        copied_logits = copied_model.model(input_ids=input_ids, use_cache=False).logits.float()
    assert torch.equal(lazy_logits, copied_logits)

    _assert_same_tensors(
        {
            name: tensor.cpu()
            for name, tensor in lazy_model.model.state_dict().items()
        },
        {
            name: tensor.cpu()
            for name, tensor in copied_model.model.state_dict().items()
        },
    )


def test_legacy_layout_load_resave_round_trips(tmp_path, monkeypatch):
    quantized_dir, wrapped = _quantize_and_save(tmp_path, monkeypatch)
    _rewrite_to_legacy_layout(quantized_dir, wrapped)

    loaded = _load(quantized_dir)
    resaved_dir = tmp_path / "resaved"
    loaded.save(resaved_dir)

    reloaded = _load(resaved_dir)
    _assert_same_tensors(
        {
            name: tensor.cpu()
            for name, tensor in loaded.model.state_dict().items()
        },
        {
            name: tensor.cpu()
            for name, tensor in reloaded.model.state_dict().items()
        },
    )

    input_ids = torch.tensor([[1, 3, 4, 5]])
    with torch.inference_mode():
        before = loaded.model(input_ids=input_ids, use_cache=False).logits.float()
        after = reloaded.model(input_ids=input_ids, use_cache=False).logits.float()
    assert torch.equal(before, after)


def test_legacy_layout_load_keeps_a_distinct_tied_head(tmp_path, monkeypatch):
    """A checkpoint that stores both copies with different values stays untied.

    ``tie_weights(missing_keys=...)`` compares the two loaded tensors and skips
    tying when they differ, so a config that advertises tying must not hide a
    deliberately distinct ``lm_head``.
    """

    quantized_dir, wrapped = _quantize_and_save(tmp_path, monkeypatch)
    definition = MODEL_MAP_CASEFOLD["gemma3"]
    to_checkpoint = _checkpoint_key_converter(definition, wrapped.model, to_runtime=False)

    # The saved checkpoint already uses the legacy layout; add a second head
    # copy whose values differ from the embedding it is supposed to tie to.
    source = quantized_dir / "model.safetensors"
    tensors = load_file(source)
    embed_checkpoint = to_checkpoint("model.language_model.embed_tokens.weight")
    head_checkpoint = to_checkpoint("lm_head.weight")
    embed = tensors[embed_checkpoint]
    tensors[head_checkpoint] = (embed.clone().float() + 0.5).to(embed.dtype)
    save_file(tensors, source)

    copied = {"calls": 0}
    original = gm._write_safetensors_with_converted_header

    def _record(*args, **kwargs):
        copied["calls"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(gm, "_write_safetensors_with_converted_header", _record)

    loaded = _load(quantized_dir)

    assert copied["calls"] == 0, "the distinct-head case must stay on the lazy path"
    state = loaded.model.state_dict()
    embed_runtime = "model.language_model.embed_tokens.weight"
    assert not torch.equal(state["lm_head.weight"], state[embed_runtime])
    assert torch.equal(state["lm_head.weight"].cpu(), tensors[head_checkpoint].cpu())
    assert torch.equal(state[embed_runtime].cpu(), tensors[embed_checkpoint].cpu())
