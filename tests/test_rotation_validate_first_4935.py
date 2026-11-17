# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A refused rotation run must not mutate the model (#4935).

``quantize()`` fused the layer norms and rotated the weights before it
validated the calibration data. Every calibration-data error therefore left
the in-session model rotated but without the online Hadamard that a rotated
``down_proj`` needs, and a retry on the same object rotated (and fused) it a
second time, silently quantizing from that broken state.

The calibration input is now validated before ``fuse_layer_norms`` /
``rotate_model``. These tests pin both halves of that contract:

* a failed call leaves every weight and the logits bitwise unchanged, and
* a failed call followed by a valid retry produces exactly the checkpoint of
  a first-time valid run.
"""

from __future__ import annotations

import io
from functools import partial
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, CalibrationConfig
from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.quantization.config import GPTQConfig


# ``hidden_size`` and ``intermediate_size`` are powers of two, so the model has
# an exact Hadamard transform and the rotation itself is not what fails.
_HIDDEN = 32
_INTERMEDIATE = 64
_VOCAB = 64

_FIXED_IDS = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]])

_VALID_ROWS = [
    {"input_ids": list(range(1, 17)), "attention_mask": [1] * 16},
    {"input_ids": list(range(17, 33)), "attention_mask": [1] * 16},
    {"input_ids": list(range(2, 18)), "attention_mask": [1] * 16},
    {"input_ids": list(range(18, 34)), "attention_mask": [1] * 16},
]

_TOO_SHORT_ROWS = [
    # Eight tokens, at or under the default ``min_length`` of ten.
    {"input_ids": [1, 3, 5, 7, 9, 11, 13, 15], "attention_mask": [1] * 8},
]

# A row that ``prepare_calibration_dataset`` cannot turn into token ids.
_UNSUPPORTED_ROWS = [object()]


def _quantize_config() -> GPTQConfig:
    return GPTQConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        device="cpu",
        rotation="hadamard",
    )


def _build_qmodel(tmp_path: Path, quantize_config) -> LlamaQModel:
    source = tmp_path / "source"
    config = LlamaConfig(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=_INTERMEDIATE,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(1234)
    model = LlamaForCausalLM(config).eval()
    model.save_pretrained(source)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel({"[PAD]": 0, "[UNK]": 1, "a": 2}, unk_token="[UNK]")
        ),
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    return LlamaQModel(
        model=model,
        quantized=False,
        tokenizer=tokenizer,
        quantize_config=quantize_config,
        model_local_path=str(source),
    )


def _snapshot(qmodel: LlamaQModel):
    """Return the embedding, one decoder weight and the logits, cloned."""

    embedding = qmodel.model.get_input_embeddings().weight.detach().clone()
    decoder = qmodel.model.model.layers[0].mlp.down_proj.weight.detach().clone()
    with torch.no_grad():
        logits = qmodel.model(_FIXED_IDS).logits.clone()
    return embedding, decoder, logits


def _assert_snapshot_unchanged(qmodel: LlamaQModel, snapshot) -> None:
    embedding, decoder, logits = snapshot
    assert torch.equal(
        embedding, qmodel.model.get_input_embeddings().weight
    ), "the failed call rotated the input embedding"
    assert torch.equal(
        decoder, qmodel.model.model.layers[0].mlp.down_proj.weight
    ), "the failed call rotated a decoder weight"
    with torch.no_grad():
        after = qmodel.model(_FIXED_IDS).logits
    assert torch.equal(
        logits, after
    ), "the failed call changed the logits of the in-session model"


_FAILURES = {
    # (zero-argument callable returning the config, exception message fragment)
    "missing_dataset": (
        CalibrationConfig,
        "CalibrationConfig.dataset is required",
    ),
    "empty_dataset": (
        partial(CalibrationConfig, dataset=[]),
        "Calibration dataset must not be empty",
    ),
    "all_rows_too_short": (
        partial(CalibrationConfig, dataset=_TOO_SHORT_ROWS),
        "empty after CalibrationConfig.prepare_dataset",
    ),
    "unsupported_row_type": (
        partial(CalibrationConfig, dataset=_UNSUPPORTED_ROWS),
        "unsupported calibration example type",
    ),
}


@pytest.mark.parametrize("failure", sorted(_FAILURES), ids=sorted(_FAILURES))
def test_failed_calibration_leaves_the_model_bitwise_unchanged(tmp_path, failure):
    make_calibration, message = _FAILURES[failure]
    qmodel = _build_qmodel(tmp_path, _quantize_config())
    before = _snapshot(qmodel)

    with pytest.raises(ValueError, match=message):
        qmodel.quantize(make_calibration(), backend=BACKEND.GPTQ_TORCH)

    _assert_snapshot_unchanged(qmodel, before)


def _quantized_state_bytes(qmodel: LlamaQModel) -> bytes:
    buffer = io.BytesIO()
    torch.save(qmodel.model.state_dict(), buffer)
    return buffer.getvalue()


def _quantize_valid(qmodel: LlamaQModel) -> bytes:
    qmodel.quantize(
        CalibrationConfig(dataset=_VALID_ROWS, batch_size=2, min_length=4),
        backend=BACKEND.GPTQ_TORCH,
    )
    return _quantized_state_bytes(qmodel)


@pytest.mark.parametrize("failure", sorted(_FAILURES), ids=sorted(_FAILURES))
def test_retry_after_a_failed_call_matches_a_first_time_run(tmp_path, failure):
    """A failed call must not leave a rotation behind for the retry to redo."""

    make_calibration, message = _FAILURES[failure]

    reference = _quantize_valid(_build_qmodel(tmp_path / "reference", _quantize_config()))

    retried = _build_qmodel(tmp_path / "retried", _quantize_config())
    with pytest.raises(ValueError, match=message):
        retried.quantize(make_calibration(), backend=BACKEND.GPTQ_TORCH)
    retried_bytes = _quantize_valid(retried)

    assert retried_bytes == reference, (
        "a failed call followed by a valid retry quantized a different "
        "(double-rotated) checkpoint than a first-time valid run"
    )
