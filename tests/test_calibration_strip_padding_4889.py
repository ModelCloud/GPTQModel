# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Pre-padded calibration rows must be stripped before sorting and batching (#4889).

``tokenizer(texts, padding=True)`` / ``padding="max_length"`` produces rows
that are padded to a common width. Those pad positions used to survive into
``prepare_calibration_dataset``: every capture and per-layer replay forward ran
at the padded width, and the default ``sort="desc"`` sorted by padded width, so
it became a no-op and ``batch_size > 1`` batches were no longer length-grouped.

The tests below tokenize one text set twice (padded vs unpadded), prepare both,
and assert that the pad positions are gone and the real-token content, sample
count, and mask-filtered Hessian are unchanged. Interior mask zeros are kept:
they describe a caller-built mask rather than tokenizer padding.
"""

from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from gptqmodel.utils.calibration import prepare_calibration_dataset


TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox",
    "a b c d e f g",
    "one two three four five six seven eight nine ten eleven",
]


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizerFast:
    specials = ["<s>", "</s>", "<unk>", "<pad>"]
    word_level = Tokenizer(models.WordLevel(unk_token="<unk>"))
    word_level.pre_tokenizer = pre_tokenizers.Whitespace()
    word_level.add_special_tokens(specials)
    word_level.train_from_iterator(
        TEXTS,
        trainer=trainers.WordLevelTrainer(special_tokens=specials, vocab_size=64),
    )
    return PreTrainedTokenizerFast(tokenizer_object=word_level, pad_token="<pad>")


def _qmodel(tokenizer) -> SimpleNamespace:
    return SimpleNamespace(
        tokenizer=tokenizer,
        support_batch_quantize=True,
        model=SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=128),
        ),
    )


def _prepare(tokenizer, dataset, **kwargs):
    kwargs.setdefault("batch_size", 1)
    kwargs.setdefault("calibration_data_min_length", 0)
    kwargs.setdefault("calibration_dataset_sort", "desc")
    return prepare_calibration_dataset(_qmodel(tokenizer), dataset, **kwargs)


def _widths(prepared) -> list[int]:
    return [len(batch["input_ids"][0]) for batch in prepared]


def _real_rows(prepared) -> list[tuple[int, ...]]:
    rows = []
    for batch in prepared:
        ids = batch["input_ids"][0]
        mask = batch["attention_mask"][0].to(torch.bool)
        rows.append(tuple(int(token) for token in ids[mask]))
    return rows


def _masked_hessian(prepared, *, vocab_size: int = 64, hidden: int = 8):
    """Hessian proxy over the mask-filtered rows, in canonical token order.

    The GPTQ processor drops pad positions before ``add_batch``
    (``gptq_processor.py``), so the Hessian only depends on the multiset of real
    tokens. Sorting the rows first makes the accumulation order independent of
    the calibration sort, which is exactly the property the issue verified.
    """

    generator = torch.Generator().manual_seed(0)
    embedding = torch.randn(vocab_size, hidden, generator=generator, dtype=torch.float64)
    ids = torch.cat(
        [
            batch["input_ids"][batch["attention_mask"].to(torch.bool)]
            for batch in prepared
        ]
    )
    ids = ids.sort().values
    features = embedding[ids]
    return ids.numel(), features.T @ features


def test_padded_rows_are_stripped_before_sorting_and_batching(tokenizer):
    padded = tokenizer(TEXTS, padding=True, return_tensors="pt")
    unpadded = [tokenizer(text, return_tensors="pt") for text in TEXTS]

    prepared_padded = _prepare(tokenizer, padded)
    prepared_unpadded = _prepare(tokenizer, unpadded)

    # Same real-token content, so the batches must have the same total width.
    assert sum(_widths(prepared_padded)) == sum(_widths(prepared_unpadded))
    assert sum(_widths(prepared_padded)) == sum(len(ids) for ids in _real_rows(prepared_padded))

    # `sort="desc"` now orders by real length instead of the padded width.
    padded_widths = _widths(prepared_padded)
    assert padded_widths == sorted(padded_widths, reverse=True)
    assert padded_widths == _widths(prepared_unpadded)


def test_padding_max_length_is_stripped_too(tokenizer):
    padded = tokenizer(TEXTS, padding="max_length", max_length=32, return_tensors="pt")
    unpadded = [tokenizer(text, return_tensors="pt") for text in TEXTS]

    prepared_padded = _prepare(tokenizer, padded)
    prepared_unpadded = _prepare(tokenizer, unpadded)

    assert sum(_widths(prepared_padded)) == sum(_widths(prepared_unpadded))
    assert _widths(prepared_padded) == _widths(prepared_unpadded)
    assert all(len(batch["input_ids"][0]) < 32 for batch in prepared_padded)


def test_padded_and_unpadded_keep_the_same_real_tokens_and_nsamples(tokenizer):
    padded = tokenizer(TEXTS, padding=True, return_tensors="pt")
    unpadded = [tokenizer(text, return_tensors="pt") for text in TEXTS]

    prepared_padded = _prepare(tokenizer, padded)
    prepared_unpadded = _prepare(tokenizer, unpadded)

    assert sorted(_real_rows(prepared_padded)) == sorted(_real_rows(prepared_unpadded))

    nsamples_padded, hessian_padded = _masked_hessian(prepared_padded)
    nsamples_unpadded, hessian_unpadded = _masked_hessian(prepared_unpadded)

    assert nsamples_padded == nsamples_unpadded
    assert torch.equal(hessian_padded, hessian_unpadded)


def test_interior_mask_zeros_are_preserved(tokenizer):
    """Only tokenizer padding is stripped; a caller-built interior mask stays."""

    prepared = _prepare(
        tokenizer,
        [{"input_ids": [[5, 6, 7, 8, 9]], "attention_mask": [[1, 1, 0, 0, 1]]}],
    )

    assert prepared[0]["input_ids"].tolist() == [[5, 6, 7, 8, 9]]
    assert prepared[0]["attention_mask"].tolist() == [[1, 1, 0, 0, 1]]


def test_batches_are_padded_to_their_own_widest_row(tokenizer):
    """Stripping lets ``collate_data`` re-pad per batch, not to the dataset max."""

    padded = tokenizer(TEXTS, padding=True, return_tensors="pt")

    prepared = prepare_calibration_dataset(
        _qmodel(tokenizer),
        padded,
        batch_size=2,
        calibration_data_min_length=0,
        calibration_dataset_sort="desc",
    )

    # Real lengths 11, 9, 7, 4 -> batches of 2 pad to 11 and 7, not to the
    # dataset-wide padded width of 11 (which would give [11, 11]).
    assert [batch["input_ids"].shape[1] for batch in prepared] == [11, 7]
    assert sum(int(batch["attention_mask"].sum()) for batch in prepared) == 31
