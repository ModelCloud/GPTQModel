# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for #4861: padded calibration capture must pass position_ids.

`prepare_calibration_dataset` honours `tokenizer.padding_side == "left"` when it
collates `batch_size > 1` rows.  The input-capture forward then ran the model
with only `input_ids` + `attention_mask`, so models with learned absolute
position embeddings (GPT-2, GPT-Neo) defaulted to `arange(seq_len)` and embedded
every real token of a left-padded row at `pad_len + i` instead of `i`.  The
capture must derive `position_ids` from the mask, exactly as `generate()` does.
"""

import types
from functools import lru_cache

import torch
import torch.nn as nn
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    OPTConfig,
    OPTForCausalLM,
    PreTrainedTokenizerFast,
)

from gptqmodel.models.base import BaseQModel
from gptqmodel.utils.calibration import prepare_calibration_dataset
from gptqmodel.utils.inspect import bound_forward_argument


TEXTS = [
    "the quick brown fox jumps over the lazy dog and then runs into the forest to hide",
    "hello world this is a short calibration row of text",
]
TOL = 1e-5


def _capture_qmodel(model):
    """Minimal stand-in that exposes the production capture forward."""

    class _CaptureQModel:
        INPUT_EMBEDDING_EXTRA_ARGS = None
        _sanitize_input_ids_for_embeddings = BaseQModel._sanitize_input_ids_for_embeddings
        run_input_capture = BaseQModel.run_input_capture

        def __init__(self, wrapped):
            self.model = wrapped

    return _CaptureQModel(model)


@lru_cache(maxsize=1)
def _tokenizer():
    corpus = [
        "the quick brown fox jumps over the lazy dog and then runs into the forest to hide",
        "hello world this is a short calibration row of text",
    ]
    specials = ["<unk>", "<pad>", "<eos>"]
    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(specials)
    tokenizer.train_from_iterator(
        corpus,
        trainer=trainers.WordLevelTrainer(special_tokens=specials, vocab_size=64),
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<eos>",
    )


@lru_cache(maxsize=1)
def _gpt2():
    """Tiny absolute-position model: `position_ids` defaults to `arange(seq_len)`."""

    tokenizer = _tokenizer()
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=128,
        n_embd=32,
        n_layer=2,
        n_head=2,
        pad_token_id=tokenizer.pad_token_id,
    )
    torch.manual_seed(0)
    return tokenizer, GPT2LMHeadModel(config).eval()


def _tiny_opt():
    tokenizer = _tokenizer()
    config = OPTConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        ffn_dim=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=128,
        pad_token_id=tokenizer.pad_token_id,
    )
    torch.manual_seed(0)
    return OPTForCausalLM(config).eval()


def _prepare(tokenizer, model, *, batch_size, padding_side, dataset=None, concat_size=None):
    tokenizer.padding_side = padding_side
    qmodel = types.SimpleNamespace(
        tokenizer=tokenizer,
        support_batch_quantize=True,
        model=types.SimpleNamespace(config=model.config),
    )
    return prepare_calibration_dataset(
        qmodel,
        TEXTS if dataset is None else dataset,
        calibration_dataset_concat_size=concat_size,
        batch_size=batch_size,
        calibration_data_min_length=1,
    )


def _capture(model, layer, example):
    """Run the production capture forward and return the first layer's inputs."""

    layer_inputs = []
    positions = []

    def hook(module, args, kwargs):
        layer_inputs.append(args[0].detach().clone())
        pos_ids = bound_forward_argument(module, args, kwargs, "position_ids")
        positions.append(None if pos_ids is None else pos_ids.detach().clone())

    handle = layer.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        _capture_qmodel(model).run_input_capture(
            example, use_cache=False, data_device=torch.device("cpu")
        )
    finally:
        handle.remove()

    assert len(layer_inputs) == 1, "capture forward never reached the first decoder layer"
    return layer_inputs[0], positions[0]


def _real_token_rows(batch_hidden, batch_mask):
    return [batch_hidden[row][batch_mask[row].bool()] for row in range(batch_hidden.shape[0])]


def _max_abs_diff(reference_hidden, batch_hidden, batch_mask):
    diffs = []
    for ref, got in zip(reference_hidden, _real_token_rows(batch_hidden, batch_mask)):
        assert got.shape == ref.shape
        diffs.append((got - ref).abs().max().item())
    return diffs


def _unpadded_reference(model, layer, tokenizer):
    return [
        _capture(model, layer, example)[0][0]
        for example in _prepare(tokenizer, model, batch_size=1, padding_side="right")
    ]


def test_gpt2_left_padding_matches_unpadded_reference():
    tokenizer, model = _gpt2()
    layer = model.transformer.h[0]
    references = _unpadded_reference(model, layer, tokenizer)

    batch = _prepare(tokenizer, model, batch_size=2, padding_side="left")[0]
    assert batch["attention_mask"].shape == (2, max(len(r) for r in batch["input_ids"]))
    assert int((batch["attention_mask"][1] == 0).sum()) > 0, "row 1 must be left padded"

    hidden, positions = _capture(model, layer, batch)
    diffs = _max_abs_diff(references, hidden, batch["attention_mask"])
    assert max(diffs) < TOL, f"left padding changed layer-0 inputs: {diffs}"

    # Real tokens must count from their first non-pad slot.  Pad slots are
    # masked out of attention, so their placeholder value is not pinned here.
    assert positions is not None
    running = batch["attention_mask"].long().cumsum(-1) - 1
    real = batch["attention_mask"].bool()
    torch.testing.assert_close(positions[real], running[real])


def test_gpt2_right_padding_matches_unpadded_reference():
    tokenizer, model = _gpt2()
    layer = model.transformer.h[0]
    references = _unpadded_reference(model, layer, tokenizer)

    batch = _prepare(tokenizer, model, batch_size=2, padding_side="right")[0]
    assert int((batch["attention_mask"][1] == 0).sum()) > 0, "row 1 must be right padded"

    hidden, _ = _capture(model, layer, batch)
    diffs = _max_abs_diff(references, hidden, batch["attention_mask"])
    assert max(diffs) < TOL, f"right padding changed layer-0 inputs: {diffs}"


def test_gpt2_concat_tail_left_padding_matches_unpadded_reference():
    tokenizer, model = _gpt2()
    layer = model.transformer.h[0]

    row = tokenizer(TEXTS[0])["input_ids"]
    concat_size = 24
    assert 0 < len(row) < concat_size

    batch = _prepare(
        tokenizer,
        model,
        batch_size=1,
        padding_side="left",
        dataset=[{"input_ids": [row], "attention_mask": [[1] * len(row)]}],
        concat_size=concat_size,
    )[0]
    assert batch["input_ids"].shape == (1, concat_size)
    assert int((batch["attention_mask"][0] == 0).sum()) == concat_size - len(row)

    reference = _capture(
        model,
        layer,
        {
            "input_ids": torch.tensor([row], dtype=torch.long),
            "attention_mask": torch.ones(1, len(row), dtype=torch.long),
        },
    )[0][0]
    hidden, _ = _capture(model, layer, batch)
    got = _real_token_rows(hidden, batch["attention_mask"])[0]
    diff = (got - reference).abs().max().item()
    assert diff < TOL, f"concat tail left padding changed layer-0 inputs: {diff}"


def test_opt_left_padding_matches_unpadded_reference():
    tokenizer = _tokenizer()
    model = _tiny_opt()
    layer = model.model.decoder.layers[0]
    references = _unpadded_reference(model, layer, tokenizer)

    batch = _prepare(tokenizer, model, batch_size=2, padding_side="left")[0]
    assert int((batch["attention_mask"][1] == 0).sum()) > 0

    hidden, _ = _capture(model, layer, batch)
    diffs = _max_abs_diff(references, hidden, batch["attention_mask"])
    assert max(diffs) < TOL, f"OPT left padding changed layer-0 inputs: {diffs}"


class _RecordingModel(nn.Module):
    """Forward that declares `position_ids` and records what it received."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 8)
        self.received = "unset"

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        self.received = position_ids
        return self.embed(input_ids)


class _NoPositionIdsModel(nn.Module):
    """Forward that does not declare `position_ids` and must never receive it."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 8)
        self.received = "unset"

    def forward(self, input_ids=None, attention_mask=None, use_cache=None):
        self.received = "unset"
        return self.embed(input_ids)


def test_left_padded_mask_adds_mask_derived_position_ids():
    model = _RecordingModel()
    example = {
        "input_ids": torch.tensor([[7, 7, 5, 6]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1]]),
    }

    _capture_qmodel(model).run_input_capture(
        example, use_cache=False, data_device=torch.device("cpu")
    )

    assert model.received is not None
    torch.testing.assert_close(model.received[0, 2:], torch.tensor([0, 1]))
    assert "position_ids" not in example, "capture must not mutate the caller's batch"


def test_interior_zero_mask_keeps_running_position():
    model = _RecordingModel()
    example = {
        "input_ids": torch.tensor([[7, 5, 7, 6]]),
        "attention_mask": torch.tensor([[1, 1, 0, 1]]),
    }

    _capture_qmodel(model).run_input_capture(
        example, use_cache=False, data_device=torch.device("cpu")
    )

    torch.testing.assert_close(model.received, torch.tensor([[0, 1, 1, 2]]))


def test_unpadded_mask_positions_equal_the_model_default():
    """An unpadded batch must keep the model's `arange` positions."""

    model = _RecordingModel()
    example = {
        "input_ids": torch.tensor([[5, 6, 7]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }

    _capture_qmodel(model).run_input_capture(
        example, use_cache=False, data_device=torch.device("cpu")
    )

    if model.received is not None:
        torch.testing.assert_close(model.received, torch.arange(3).unsqueeze(0))
    assert "position_ids" not in example


def test_forward_without_position_ids_is_not_given_one():
    model = _NoPositionIdsModel()
    example = {
        "input_ids": torch.tensor([[7, 7, 5, 6]]),
        "attention_mask": torch.tensor([[0, 0, 1, 1]]),
    }

    _capture_qmodel(model).run_input_capture(
        example, use_cache=False, data_device=torch.device("cpu")
    )

    assert model.received == "unset"
    assert "position_ids" not in example


def test_unpadded_single_row_is_bit_identical():
    tokenizer, model = _gpt2()
    layer = model.transformer.h[0]

    example = _prepare(tokenizer, model, batch_size=1, padding_side="right")[0]
    assert example["input_ids"].shape[0] == 1
    assert int((example["attention_mask"] == 0).sum()) == 0

    without_mask = _capture(model, layer, {"input_ids": example["input_ids"]})[0]
    with_mask = _capture(model, layer, example)[0]

    assert torch.equal(without_mask, with_mask)
