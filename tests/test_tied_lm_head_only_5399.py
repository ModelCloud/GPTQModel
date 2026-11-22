# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #5399: a tied checkpoint may store the shared tensor under the head.

Transformers loads a tied Llama checkpoint that keeps only ``lm_head.weight``
and reconstructs the embedding from it. GPT-QModel dropped the declared pair
whenever the declared *source* (``model.embed_tokens.weight``) was absent, so
the lazy path left the embedding on meta and calibration failed with
``Tensor.item() cannot be called on meta tensors``.
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.quantization.config import CalibrationConfig
from gptqmodel.utils.tied_weights import (
    lm_head_is_tied,
    resolve_tied_weights_keys,
    snapshot_tied_weights_keys,
)
from tests.test_adapter_targets import _write_tiny_llama_tokenizer


pytestmark = [pytest.mark.cpu]

_EMBED = "model.embed_tokens.weight"
_HEAD = "lm_head.weight"
_CALIBRATION_TEXT = [
    "tiny llama calibration sample one with enough tokens to survive filtering"
] * 4


class _Config:
    tie_word_embeddings = True


class _Model:
    """The minimum a resolver reads: a config and a declared tie map."""

    def __init__(self):
        self.config = _Config()
        self.all_tied_weights_keys = {_HEAD: _EMBED}
        self._tied_weights_keys = {_HEAD: _EMBED}


def _lookup(*present, equal=True):
    """A checkpoint that stores only ``present``, with equal shared values."""

    present = set(present)

    def has_tensor(name):
        return name in present

    def tensors_equal(left, right):
        return equal if {left, right} <= present else None

    return has_tensor, tensors_equal


def test_embedding_only_checkpoint_keeps_the_declared_tie():
    has_tensor, tensors_equal = _lookup(_EMBED)

    resolved = resolve_tied_weights_keys(
        _Model(),
        checkpoint_has_tensor=has_tensor,
        checkpoint_tensors_equal=tensors_equal,
    )

    assert resolved == {_HEAD: _EMBED}


def test_head_only_checkpoint_keeps_the_declared_tie():
    """The regression: the head is the only stored member of the tie."""

    has_tensor, tensors_equal = _lookup(_HEAD)

    resolved = resolve_tied_weights_keys(
        _Model(),
        checkpoint_has_tensor=has_tensor,
        checkpoint_tensors_equal=tensors_equal,
    )

    assert resolved == {_HEAD: _EMBED}, "the tie must be kept, not dropped"


def test_head_only_checkpoint_still_reports_the_head_as_tied():
    model = _Model()
    has_tensor, tensors_equal = _lookup(_HEAD)

    snapshot_tied_weights_keys(
        model,
        checkpoint_has_tensor=has_tensor,
        checkpoint_tensors_equal=tensors_equal,
    )

    assert lm_head_is_tied(model) is True


def test_neither_member_present_drops_the_tie():
    has_tensor, tensors_equal = _lookup()

    resolved = resolve_tied_weights_keys(
        _Model(),
        checkpoint_has_tensor=has_tensor,
        checkpoint_tensors_equal=tensors_equal,
    )

    assert resolved == {}


def test_unknown_presence_keeps_the_declared_tie():
    resolved = resolve_tied_weights_keys(
        _Model(),
        checkpoint_has_tensor=lambda name: None,
    )

    assert resolved == {_HEAD: _EMBED}


def test_two_different_tensors_are_not_tied():
    has_tensor, tensors_equal = _lookup(_EMBED, _HEAD, equal=False)

    resolved = resolve_tied_weights_keys(
        _Model(),
        checkpoint_has_tensor=has_tensor,
        checkpoint_tensors_equal=tensors_equal,
    )

    assert resolved == {}


def _write_tied_llama(root, *, keep: str):
    """Save a tiny tied Llama storing only the requested member."""

    root.mkdir(parents=True)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    LlamaForCausalLM(config).save_pretrained(root)
    tensors = load_file(str(root / "model.safetensors"))
    if keep == "head":
        tensors[_HEAD] = tensors.pop(_EMBED)
    else:
        tensors.pop(_HEAD, None)
    save_file(
        {name: tensor.contiguous() for name, tensor in tensors.items()},
        str(root / "model.safetensors"),
    )
    _write_tiny_llama_tokenizer(root)
    return root


@pytest.mark.parametrize("keep", ["embed", "head"])
def test_lazy_calibration_materializes_either_stored_member(tmp_path, keep):
    """The issue's repro: each alias as the only saved tensor must quantize."""

    source = _write_tied_llama(tmp_path / keep, keep=keep)
    stored = load_file(str(source / "model.safetensors"))
    assert (_EMBED in stored) is (keep == "embed")
    assert (_HEAD in stored) is (keep == "head")

    model = GPTQModel.load(
        str(source), QuantizeConfig(bits=4, group_size=32, offload_to_disk=True)
    )
    model.quantize(CalibrationConfig(dataset=_CALIBRATION_TEXT, batch_size=1))

    embedding = model.model.model.embed_tokens.weight
    head = model.model.lm_head.weight
    assert embedding.device.type != "meta"
    assert head.device.type != "meta"
    # The tie is restored, so both names expose the same loaded values.
    torch.testing.assert_close(head, embedding)
