# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""InternVL3-hf keeps its untied `lm_head` through quantize() and save() (#4799).

`InternVLConfig` leaves the composite `tie_word_embeddings` at the
`PretrainedConfig` default `True` while `text_config.tie_word_embeddings` is
`False` and the checkpoint stores a distinct `language_model.lm_head.weight`.
The lazy loader materializes a meta shell piece by piece and used to call
`model.tie_weights()` after every step; that recomputes the mapping from the
wrapper flag, replaces the real head with the input embedding, and the writer
then omits `lm_head.weight` as a duplicate. The quantized model silently gets a
different output layer.

The regression drives the same path: a tiny composite checkpoint with a
distinct head is quantized with `offload_to_disk=True` (the lazy-turtle path),
saved, and reloaded. A genuinely tied control must still be tied and must still
omit the head so its save size does not change.
"""

from __future__ import annotations

import pytest
import torch
from safetensors import safe_open
from transformers import (
    InternVLConfig,
    InternVLForConditionalGeneration,
    InternVLVisionConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
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


def _save_internvl(root, *, text_tied: bool, top_tied: bool) -> torch.Tensor:
    """Save a tiny composite checkpoint; return its head when it stores one."""

    torch.manual_seed(0)
    model = InternVLForConditionalGeneration(_internvl_config(text_tied=text_tied, top_tied=top_tied))
    model = model.to(torch.float16)
    if text_tied:
        model.lm_head.weight = model.model.language_model.embed_tokens.weight
        head = None
    else:
        # The source checkpoint's head is a real, separate tensor.
        head = model.lm_head.weight.detach().clone() + 0.01
        model.lm_head.weight = torch.nn.Parameter(head)
    model.save_pretrained(root)
    return head


def _save_qwen(root, *, tied: bool) -> torch.Tensor:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_hidden_layers=_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=tied,
    )
    model = Qwen2ForCausalLM(config).to(torch.float16)
    head = None
    if tied:
        model.lm_head.weight = model.model.embed_tokens.weight
    else:
        head = model.lm_head.weight.detach().clone() + 0.01
        model.lm_head.weight = torch.nn.Parameter(head)
    model.save_pretrained(root)
    return head


def _calibration():
    generator = torch.Generator().manual_seed(0)
    return [
        {"input_ids": torch.randint(0, _VOCAB, (1, 16), generator=generator)}
        for _ in range(2)
    ]


def _quantize(src, out):
    model = GPTQModel.load(
        str(src),
        GPTQConfig(bits=4, group_size=32, offload_to_disk=True),
        device="cpu",
        dtype=torch.float16,
    )
    model.quantize(CalibrationConfig(dataset=_calibration(), batch_size=1, min_length=1))
    model.save(str(out))
    return model


def _saved_heads(root):
    heads = {}
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(str(path), "pt") as handler:
            for key in handler.keys():
                if key.endswith("lm_head.weight"):
                    heads[key] = handler.get_tensor(key).clone()
    return heads


def test_untied_internvl_head_survives_quantize_and_save(tmp_path):
    source = tmp_path / "source"
    saved = tmp_path / "saved"
    source_head = _save_internvl(source, text_tied=False, top_tied=True)
    assert source_head is not None
    assert len(_saved_heads(source)) == 1

    model = GPTQModel.load(
        str(source),
        GPTQConfig(bits=4, group_size=32, offload_to_disk=True),
        device="cpu",
        dtype=torch.float16,
    )

    # Meta-shell check: the load-time snapshot resolved the untied head before
    # any layer was materialized, and the shell's head is not the embedding.
    assert getattr(model.model, "_gptqmodel_tied_weights_keys", None) == {}
    assert model.model.lm_head.weight is not model.model.get_input_embeddings().weight

    model.quantize(CalibrationConfig(dataset=_calibration(), batch_size=1, min_length=1))

    head = model.model.lm_head.weight
    assert head is not model.model.get_input_embeddings().weight
    assert torch.equal(head, source_head)

    model.save(str(saved))

    saved_heads = _saved_heads(saved)
    assert len(saved_heads) == 1, f"lm_head.weight was dropped: {sorted(saved_heads)}"
    assert torch.equal(next(iter(saved_heads.values())), source_head)

    reloaded = GPTQModel.load(str(saved), device="cpu", dtype=torch.float16)
    assert torch.equal(reloaded.model.lm_head.weight, source_head)
    assert reloaded.model.lm_head.weight is not reloaded.model.get_input_embeddings().weight


def test_tied_internvl_control_stays_tied_and_omits_the_head(tmp_path):
    source = tmp_path / "source"
    saved = tmp_path / "saved"
    assert _save_internvl(source, text_tied=True, top_tied=True) is None
    assert _saved_heads(source) == {}

    model = _quantize(source, saved)

    assert model.model.lm_head.weight is model.model.get_input_embeddings().weight
    assert _saved_heads(saved) == {}, "a tied checkpoint must not start storing the head"


def test_tied_qwen_control_stays_tied_and_omits_the_head(tmp_path):
    source = tmp_path / "source"
    saved = tmp_path / "saved"
    assert _save_qwen(source, tied=True) is None
    assert _saved_heads(source) == {}

    model = _quantize(source, saved)

    assert model.model.lm_head.weight is model.model.get_input_embeddings().weight
    assert _saved_heads(saved) == {}


def test_untied_qwen_head_survives_quantize_and_save(tmp_path):
    """A plain untied decoder keeps its head too."""

    source = tmp_path / "source"
    saved = tmp_path / "saved"
    source_head = _save_qwen(source, tied=False)
    assert source_head is not None

    model = _quantize(source, saved)

    assert model.model.lm_head.weight is not model.model.get_input_embeddings().weight
    assert torch.equal(model.model.lm_head.weight, source_head)
    saved_heads = _saved_heads(saved)
    assert len(saved_heads) == 1
    assert torch.equal(next(iter(saved_heads.values())), source_head)


def test_word_embeddings_tied_requires_both_levels():
    untied_decoder = InternVLConfig(
        text_config=Qwen2Config(vocab_size=16, hidden_size=8, num_hidden_layers=1, tie_word_embeddings=False),
        vision_config=InternVLVisionConfig(hidden_size=8, num_hidden_layers=1, num_attention_heads=1),
    )
    untied_decoder.tie_word_embeddings = True
    assert word_embeddings_tied(untied_decoder) is False

    untied_wrapper = InternVLConfig(
        text_config=Qwen2Config(vocab_size=16, hidden_size=8, num_hidden_layers=1, tie_word_embeddings=True),
        vision_config=InternVLVisionConfig(hidden_size=8, num_hidden_layers=1, num_attention_heads=1),
    )
    untied_wrapper.tie_word_embeddings = False
    assert word_embeddings_tied(untied_wrapper) is False

    tied = Qwen2Config(vocab_size=16, hidden_size=8, num_hidden_layers=1, tie_word_embeddings=True)
    assert word_embeddings_tied(tied) is True
    assert word_embeddings_tied(Qwen2Config(vocab_size=16, hidden_size=8, num_hidden_layers=1, tie_word_embeddings=False)) is False


def test_resolve_tied_weights_keys_trusts_a_distinct_checkpoint_head():
    class _Config:
        tie_word_embeddings = True

    class _Model:
        config = _Config()
        all_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    model = _Model()

    # The wrapper flag and its declared mapping decide which pairs may be tied.
    # A decoder text flag is inert for the wrapper's head (#4973).
    model.config.get_text_config = lambda: type("Text", (), {"tie_word_embeddings": False})()
    assert resolve_tied_weights_keys(model) == {"lm_head.weight": "model.embed_tokens.weight"}

    # A wrapper that says False ties nothing (#4369).
    model.config.tie_word_embeddings = False
    assert resolve_tied_weights_keys(model) == {}
    model.config.tie_word_embeddings = True

    # The checkpoint stores both sides with different values: a real head.
    distinct = {
        "lm_head.weight": True,
        "model.embed_tokens.weight": True,
    }
    assert (
        resolve_tied_weights_keys(
            model,
            checkpoint_has_tensor=distinct.get,
            checkpoint_tensors_equal=lambda _target, _source: False,
        )
        == {}
    )

    # The checkpoint omits the head (a tied checkpoint): the pair stays tied.
    absent = {"lm_head.weight": False, "model.embed_tokens.weight": True}
    assert resolve_tied_weights_keys(model, checkpoint_has_tensor=absent.get) == {
        "lm_head.weight": "model.embed_tokens.weight"
    }

    # The checkpoint has no source tensor: nothing to tie to.
    missing_source = {"lm_head.weight": True, "model.embed_tokens.weight": False}
    assert resolve_tied_weights_keys(model, checkpoint_has_tensor=missing_source.get) == {}
