# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""InternVL ties its head from the wrapper flag, not the text config (#4973).

``InternVLConfig(text_config=<qwen2 dict>)`` leaves the wrapper
``tie_word_embeddings`` at the ``PretrainedConfig`` default ``True`` while the
nested ``text_config.tie_word_embeddings`` defaults to ``False``. Transformers
ties ``lm_head.weight`` to ``model.language_model.embed_tokens.weight`` from the
*wrapper* flag and its declared ``_tied_weights_keys``, so ``save_pretrained``
writes only the embedding.

#4852 required *both* levels to declare a tie before it would honour the
declared pair, so the load-time snapshot came out empty. The lazy loader never
materialized the head and quantization ended with::

    RuntimeError: Offload: modules remain on the meta device after
    quantization (lm_head).

The resolution is evidence-driven: the wrapper flag and its declared mapping
decide which pairs may be tied, and the checkpoint can only refute a declared
pair (a distinct stored head, #4799). A decoder ``text_config`` flag never
untied a head the wrapper ties, and a wrapper that says ``False`` still ties
nothing (#4369).
"""

from __future__ import annotations

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import (
    InternVLConfig,
    InternVLForConditionalGeneration,
    InternVLVisionConfig,
    Qwen2Config,
)

from gptqmodel import GPTQModel
from gptqmodel.models.definitions.internvl import InternVLQModel
from gptqmodel.quantization.config import CalibrationConfig, GPTQConfig
from gptqmodel.utils.tied_weights import (
    resolve_tied_weights_keys,
    word_embeddings_tied,
)


_VOCAB = 128
_HIDDEN = 64
_INTERMEDIATE = 128
_LAYERS = 2
_EMBED_KEY = "language_model.model.embed_tokens.weight"
_HEAD_KEY = "lm_head.weight"


@pytest.fixture(autouse=True)
def _no_processor(monkeypatch):
    """The tiny checkpoint has no image processor; text-only calibration does not need one."""

    monkeypatch.setattr(InternVLQModel, "require_load_processor", False)


def _internvl_config(*, text_tied: bool, top_tied: bool) -> InternVLConfig:
    text = Qwen2Config(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_hidden_layers=_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=text_tied,
    )
    vision = InternVLVisionConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        image_size=(32, 32),
        patch_size=(8, 8),
    )
    config = InternVLConfig(text_config=text, vision_config=vision)
    config.tie_word_embeddings = top_tied
    return config


def _internvl(*, text_tied: bool, top_tied: bool) -> InternVLForConditionalGeneration:
    torch.manual_seed(0)
    return InternVLForConditionalGeneration(_internvl_config(text_tied=text_tied, top_tied=top_tied))


def _rewrite_head(root, mode: str) -> None:
    """Force the saved shard into the requested head layout."""

    shards = sorted(root.glob("*.safetensors"))
    assert len(shards) == 1, f"expected one shard, got {shards}"
    tensors = load_file(str(shards[0]))
    assert _EMBED_KEY in tensors, sorted(tensors)[:8]
    if mode == "absent":
        tensors.pop(_HEAD_KEY, None)
    elif mode == "equal":
        tensors[_HEAD_KEY] = tensors[_EMBED_KEY].clone()
    elif mode == "distinct":
        tensors[_HEAD_KEY] = tensors[_EMBED_KEY].clone() + 0.01
    else:  # pragma: no cover - test helper guard
        raise AssertionError(mode)
    save_file(tensors, str(shards[0]))


def _save_internvl(root, *, text_tied: bool, top_tied: bool, head: str) -> None:
    model = _internvl(text_tied=text_tied, top_tied=top_tied).to(torch.float16)
    model.save_pretrained(root)
    _rewrite_head(root, head)


def _saved_heads(root):
    heads = {}
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(str(path), "pt") as handler:
            for key in handler.keys():
                if key.endswith("lm_head.weight"):
                    heads[key] = handler.get_tensor(key).clone()
    return heads


def _embedding_copies(root) -> int:
    """Count stored input-embedding tensors across the shards.

    The HF source spells the key ``language_model.model.embed_tokens.weight``
    while the quantized save uses the runtime name
    ``model.language_model.embed_tokens.weight``; only the suffix is stable.
    """

    copies = 0
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(str(path), "pt") as handler:
            copies += sum(1 for key in handler.keys() if key.endswith("embed_tokens.weight"))
    return copies


def _calibration():
    generator = torch.Generator().manual_seed(0)
    return [
        {"input_ids": torch.randint(0, _VOCAB, (1, 16), generator=generator)}
        for _ in range(2)
    ]


def _load(src):
    return GPTQModel.load(
        str(src),
        GPTQConfig(bits=4, group_size=32, offload_to_disk=True),
        device="cpu",
        dtype=torch.float16,
    )


def _quantize(src, out):
    model = _load(src)
    model.quantize(CalibrationConfig(dataset=_calibration(), batch_size=1, min_length=1))
    model.save(str(out))
    return model


@pytest.mark.parametrize("head", ["absent", "equal", "distinct"])
@pytest.mark.parametrize("text_tied", [True, False])
@pytest.mark.parametrize("top_tied", [True, False])
def test_resolution_matrix(top_tied, text_tied, head):
    """The wrapper flag and checkpoint decide the tie; the text flag is inert."""

    model = _internvl(text_tied=text_tied, top_tied=top_tied)
    has = {
        _HEAD_KEY: head != "absent",
        "model.language_model.embed_tokens.weight": True,
    }
    resolved = resolve_tied_weights_keys(
        model,
        checkpoint_has_tensor=has.get,
        checkpoint_tensors_equal=lambda _target, _source: head != "distinct",
    )

    expected_tied = top_tied and head != "distinct"
    assert bool(resolved) is expected_tied
    if expected_tied:
        assert resolved == {_HEAD_KEY: "model.language_model.embed_tokens.weight"}
    else:
        assert resolved == {}


def test_text_flag_alone_does_not_untie_the_declared_pair():
    """The regression guard: #4852's both-levels gate must not decide the tie."""

    model = _internvl(text_tied=False, top_tied=True)
    # The config-only fallback still reports the conservative both-levels view.
    assert word_embeddings_tied(model.config) is False
    # The declared pair plus a tied-save checkpoint (no stored head) is a tie.
    resolved = resolve_tied_weights_keys(
        model,
        checkpoint_has_tensor={_HEAD_KEY: False, "model.language_model.embed_tokens.weight": True}.get,
    )
    assert resolved == {_HEAD_KEY: "model.language_model.embed_tokens.weight"}


def test_text_untied_head_absent_survives_quantize_save_reload(tmp_path):
    """Issue #4973: the exact failing shape, end to end."""

    source = tmp_path / "source"
    saved = tmp_path / "saved"
    _save_internvl(source, text_tied=False, top_tied=True, head="absent")
    assert _saved_heads(source) == {}
    assert _embedding_copies(source) == 1

    model = _load(source)
    assert getattr(model.model, "_gptqmodel_tied_weights_keys", None) == {
        _HEAD_KEY: "model.language_model.embed_tokens.weight"
    }

    model.quantize(CalibrationConfig(dataset=_calibration(), batch_size=1, min_length=1))
    assert model.model.lm_head.weight is model.model.get_input_embeddings().weight

    model.save(str(saved))
    assert _saved_heads(saved) == {}, "a tied head must not be stored twice"
    assert _embedding_copies(saved) == 1

    reloaded = GPTQModel.load(str(saved), device="cpu", dtype=torch.float16)
    assert reloaded.model.lm_head.weight is reloaded.model.get_input_embeddings().weight


def test_text_untied_head_equal_stays_tied(tmp_path):
    """A checkpoint that stores both copies with equal values is still a tie."""

    source = tmp_path / "source"
    saved = tmp_path / "saved"
    _save_internvl(source, text_tied=False, top_tied=True, head="equal")
    assert len(_saved_heads(source)) == 1

    model = _load(source)
    assert getattr(model.model, "_gptqmodel_tied_weights_keys", None) == {
        _HEAD_KEY: "model.language_model.embed_tokens.weight"
    }

    model.quantize(CalibrationConfig(dataset=_calibration(), batch_size=1, min_length=1))
    assert model.model.lm_head.weight is model.model.get_input_embeddings().weight

    model.save(str(saved))
    assert _saved_heads(saved) == {}
    assert _embedding_copies(saved) == 1


def test_text_tied_head_absent_stays_tied(tmp_path):
    """Control: both flags tied and a tied-save checkpoint stay tied."""

    source = tmp_path / "source"
    saved = tmp_path / "saved"
    _save_internvl(source, text_tied=True, top_tied=True, head="absent")
    assert _saved_heads(source) == {}

    model = _load(source)
    assert getattr(model.model, "_gptqmodel_tied_weights_keys", None) == {
        _HEAD_KEY: "model.language_model.embed_tokens.weight"
    }
    model.quantize(CalibrationConfig(dataset=_calibration(), batch_size=1, min_length=1))
    model.save(str(saved))
    assert _saved_heads(saved) == {}
    assert _embedding_copies(saved) == 1


def test_wrapper_untied_head_distinct_stays_untied(tmp_path):
    """Control (#4369 shape): a wrapper that says False never ties the head."""

    source = tmp_path / "source"
    saved = tmp_path / "saved"
    # A consistent untied config: with `text_tied=True` Transformers also writes
    # a nested `language_model.lm_head.weight` alias, which is a separate save
    # quirk unrelated to this fix (the resolver matrix above covers that shape).
    _save_internvl(source, text_tied=False, top_tied=False, head="distinct")
    # Transformers also writes a nested `language_model.lm_head.weight` alias;
    # the loader materializes the runtime head from the top-level key.
    source_head = _saved_heads(source)[_HEAD_KEY]

    model = _load(source)
    assert getattr(model.model, "_gptqmodel_tied_weights_keys", None) == {}
    assert model.model.lm_head.weight is not model.model.get_input_embeddings().weight

    model.quantize(CalibrationConfig(dataset=_calibration(), batch_size=1, min_length=1))
    assert model.model.lm_head.weight is not model.model.get_input_embeddings().weight
    assert torch.equal(model.model.lm_head.weight, source_head)

    model.save(str(saved))
    saved_heads = _saved_heads(saved)
    assert len(saved_heads) == 1
    assert torch.equal(next(iter(saved_heads.values())), source_head)

    reloaded = GPTQModel.load(str(saved), device="cpu", dtype=torch.float16)
    assert torch.equal(reloaded.model.lm_head.weight, source_head)
