# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Tied checkpoints that store both copies must load with a CPU-offload map (#4943).

#4890 (#4866) makes the explicit CPU-offload maps place a really-tied group on
one device, but it runs in ``simple_dispatch_model``, after the checkpoint load.
``load_checkpoint_in_model_then_tie_weights`` calls
``tie_weights(missing_keys=...)`` first, and when the checkpoint stores *both*
``lm_head.weight`` and ``model.embed_tokens.weight`` (AutoAWQ, GPTQModel <= 1.x)
transformers compares the two with ``torch.equal``. A map that puts one on CPU
and the other on the accelerator made that comparison raise before any hook was
installed.

These tests load a real quantized checkpoint in each of the shapes a tied model
is published in -- only ``embed_tokens`` (what GPTQModel saves now) and both
copies with equal values -- plus an untied control, through the four maps from
#4866.
"""

from __future__ import annotations

import shutil

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen2Config, Qwen2ForCausalLM

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.quantization.config import GPTQConfig


_VOCAB = 128
_HIDDEN = 64
_INTERMEDIATE = 128
_LAYERS = 2
_GROUP_SIZE = 32

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="an explicit CPU-offload map needs one CUDA device plus the CPU",
)


def _build_quantized_tied_checkpoint(root):
    """Quantize one tiny tied Qwen2; GPTQModel stores only ``embed_tokens``."""

    source = root / "source"
    checkpoint = root / "tied-embed"
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_hidden_layers=_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    Qwen2ForCausalLM(config).to(torch.float16).save_pretrained(source)

    generator = torch.Generator().manual_seed(1)
    calibration = [
        {"input_ids": torch.randint(0, _VOCAB, (1, 32), generator=generator)}
        for _ in range(4)
    ]
    model = GPTQModel.load(
        str(source),
        GPTQConfig(bits=4, group_size=_GROUP_SIZE, offload_to_disk=False),
        backend=BACKEND.TORCH,
        device="cpu",
        dtype=torch.float16,
    )
    model.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    model.save(str(checkpoint))
    return checkpoint


def _copy_with_head(source_dir, target_dir, *, offset: float):
    """Copy a checkpoint and store a second ``lm_head.weight`` in its shard."""

    shutil.copytree(source_dir, target_dir)
    shard = target_dir / "model.safetensors"
    assert shard.exists(), "the tiny checkpoint must be a single safetensors shard"
    tensors = load_file(str(shard))
    assert "lm_head.weight" not in tensors
    head = tensors["model.embed_tokens.weight"]
    tensors["lm_head.weight"] = head.clone() if offset == 0.0 else (head.clone() + offset)
    save_file(tensors, str(shard))
    return target_dir


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    root = tmp_path_factory.mktemp("tied_both_4943")
    tied_embed = _build_quantized_tied_checkpoint(root)
    return {
        # Only the embedding is stored (the shape GPTQModel writes today).
        "tied_embed": tied_embed,
        # Both copies, equal: AutoAWQ and GPTQModel <= 1.x tied checkpoints.
        "tied_both": _copy_with_head(tied_embed, root / "tied-both", offset=0.0),
        # Both copies, different: a real separate head behind a tied config.
        "untied_both": _copy_with_head(tied_embed, root / "untied-both", offset=0.5),
    }


def _split(first: str, second: str) -> dict:
    device_map = {
        "model.embed_tokens": first,
        "model.rotary_emb": first,
        "model.norm": second,
        "lm_head": second,
    }
    device_map.update(
        {f"model.layers.{index}": (first if index < _LAYERS // 2 else second) for index in range(_LAYERS)}
    )
    return device_map


_MAPS = {
    "head_cpu": {**_split("cuda:0", "cuda:0"), "lm_head": "cpu"},
    "1_then_cpu": _split("cuda:0", "cpu"),
    "cpu_then1": _split("cpu", "cuda:0"),
    "rotary_cpu": {**_split("cuda:0", "cuda:0"), "model.rotary_emb": "cpu"},
}


def _load(checkpoint, device_map):
    return GPTQModel.load(
        str(checkpoint),
        backend=BACKEND.AUTO,
        device_map=dict(device_map),
        dtype=torch.float16,
    )


def _forward(model):
    inner = model.model
    embedding = inner.get_input_embeddings().weight
    ids = torch.randint(0, _VOCAB, (1, 8), generator=torch.Generator().manual_seed(11))
    with torch.inference_mode():
        logits = inner(ids.to(embedding.device)).logits
    return logits.float()


@requires_cuda
@pytest.mark.parametrize("checkpoint_name", ["tied_embed", "tied_both"])
@pytest.mark.parametrize("map_name", sorted(_MAPS))
def test_tied_checkpoint_loads_and_keeps_its_tie(checkpoints, checkpoint_name, map_name):
    """Both published tied shapes must load, run, and keep one shared head."""

    model = _load(checkpoints[checkpoint_name], _MAPS[map_name])
    inner = model.model

    logits = _forward(model)
    assert logits.shape == (1, 8, _VOCAB)
    assert torch.isfinite(logits).all(), f"{checkpoint_name}/{map_name}: non-finite logits"

    head = inner.lm_head.weight
    embedding = inner.get_input_embeddings().weight
    assert head is embedding, (
        f"{checkpoint_name}/{map_name}: lm_head.weight and embed_tokens.weight are no "
        "longer the same tensor"
    )
    assert head.device == embedding.device
    assert inner.hf_device_map["lm_head"] == inner.hf_device_map["model.embed_tokens"]


@requires_cuda
@pytest.mark.parametrize("map_name", sorted(_MAPS))
def test_untied_checkpoint_keeps_its_own_head(checkpoints, map_name):
    """A distinct head is not tied, so its placement is left alone (#4799)."""

    model = _load(checkpoints["untied_both"], _MAPS[map_name])
    inner = model.model

    assert torch.isfinite(_forward(model)).all()
    assert inner.lm_head.weight is not inner.get_input_embeddings().weight


@requires_cuda
def test_both_copy_shape_matches_the_embed_only_shape(checkpoints):
    """The duplicate checkpoint copy must not change what the model computes."""

    embedding_only = _load(checkpoints["tied_embed"], _MAPS["head_cpu"])
    both_copies = _load(checkpoints["tied_both"], _MAPS["head_cpu"])

    expected = _forward(embedding_only)
    actual = _forward(both_copies)

    assert torch.equal(actual, expected), (
        f"a checkpoint that stores both tied copies produced different logits, "
        f"max|delta|={(actual - expected).abs().max().item():.3e}"
    )
