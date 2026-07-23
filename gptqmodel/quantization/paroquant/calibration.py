# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Paper-compatible ParoQuant calibration dataset construction.

The implementation follows the official ``z-lab/paroquant`` legacy branch:
documents are shuffled per source, over-length documents are skipped, the
remaining token streams are concatenated into full fixed-length sequences, and
the evenly mixed training sequences are shuffled once more.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import torch


PAROQUANT_PAPER_TRAIN_SOURCES = ("wikitext2", "c4", "redpajama")
PAROQUANT_PAPER_VALIDATION_SOURCE = "pileval"


@dataclass(frozen=True)
class ParoQuantCalibrationDatasets:
    """Independent tokenized streams consumed by ParoQuant quantization."""

    train: list[dict[str, torch.Tensor]]
    validation: list[dict[str, torch.Tensor]]
    sequence_length: int
    seed: int


def _default_dataset_loader(*args, **kwargs):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on optional installation
        raise ImportError(
            "Building the ParoQuant paper calibration set requires the `datasets` package."
        ) from exc
    return load_dataset(*args, **kwargs)


def _shuffle_dataset(dataset: Any, *, seed: int):
    shuffle = getattr(dataset, "shuffle", None)
    if callable(shuffle):
        return shuffle(seed=seed)
    rows = list(dataset)
    random.Random(seed).shuffle(rows)
    return rows


def _select_dataset(dataset: Any, indices: range):
    select = getattr(dataset, "select", None)
    if callable(select):
        return select(indices)
    rows = list(dataset)
    return [rows[index] for index in indices]


def _load_calibration_source(
    source: str,
    *,
    split: str,
    seed: int,
    dataset_loader: Callable[..., Any],
):
    if source == "pileval":
        dataset = dataset_loader("mit-han-lab/pile-val-backup", split="validation")
        return _shuffle_dataset(dataset, seed=seed)

    if source == "wikitext2":
        dataset = dataset_loader("wikitext", "wikitext-2-raw-v1", split=split)
        return _shuffle_dataset(dataset, seed=seed)

    if source == "c4":
        if split == "train":
            data_files = {"train": "en/c4-train.00000-of-01024.json.gz"}
        elif split == "validation":
            data_files = {"validation": "en/c4-validation.00001-of-00008.json.gz"}
        else:
            raise ValueError(f"ParoQuant C4 calibration does not define split `{split}`.")
        dataset = dataset_loader("allenai/c4", data_files=data_files, split=split)
        return _shuffle_dataset(dataset, seed=seed)

    if source == "redpajama":
        dataset = dataset_loader(
            "liang2kl/RedPajama-Data-1T-Sample-Backup",
            split="train",
            trust_remote_code=True,
        )
        dataset = _shuffle_dataset(dataset, seed=seed)
        test_size = int(len(dataset) * 0.2)
        validation_size = int(len(dataset) * 0.1)
        train_size = len(dataset) - test_size - validation_size
        if split == "train":
            return _select_dataset(dataset, range(0, train_size))
        if split == "validation":
            return _select_dataset(dataset, range(train_size, train_size + validation_size))
        if split == "test":
            return _select_dataset(dataset, range(len(dataset) - test_size, len(dataset)))
        raise ValueError(f"ParoQuant RedPajama calibration does not define split `{split}`.")

    raise ValueError(f"Unsupported ParoQuant calibration source `{source}`.")


def _encode_text(tokenizer, text: str) -> list[int]:
    encoded = tokenizer.encode(text)
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.detach().cpu().reshape(-1).tolist()
    return [int(token) for token in encoded]


def _sample_full_sequences(
    dataset: Iterable[dict[str, Any]],
    *,
    tokenizer,
    sample_count: int,
    sequence_length: int,
) -> list[torch.Tensor]:
    required_tokens = sample_count * sequence_length
    tokens: list[int] = []
    for row in dataset:
        text = str(row["text"]).strip()
        encoded = _encode_text(tokenizer, text)
        if not encoded or len(encoded) > sequence_length:
            continue
        tokens.extend(encoded)
        if len(tokens) >= required_tokens:
            break

    available_samples = len(tokens) // sequence_length
    if available_samples < sample_count:
        raise ValueError(
            "ParoQuant calibration source did not contain enough eligible tokens: "
            f"requested {sample_count}x{sequence_length}, available {available_samples} full sequences."
        )

    return [
        torch.tensor(
            tokens[index * sequence_length : (index + 1) * sequence_length],
            dtype=torch.long,
        )
        for index in range(sample_count)
    ]


def _as_calibration_examples(sequences: Sequence[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    return [{"input_ids": sequence} for sequence in sequences]


def build_paroquant_calibration_datasets(
    tokenizer,
    *,
    train_samples: int = 2048,
    validation_samples: int = 64,
    sequence_length: int = 2048,
    seed: int = 0,
    train_sources: Sequence[str] = PAROQUANT_PAPER_TRAIN_SOURCES,
    validation_source: str = PAROQUANT_PAPER_VALIDATION_SOURCE,
    dataset_loader: Callable[..., Any] | None = None,
) -> ParoQuantCalibrationDatasets:
    """Build the paper's mixed training set and independent Pile validation set.

    Dataset loading is explicit: importing ParoQuant or calling ``quantize()``
    never downloads these corpora. ``dataset_loader`` exists for offline tests
    and compatible dataset providers.
    """
    train_samples = int(train_samples)
    validation_samples = int(validation_samples)
    sequence_length = int(sequence_length)
    seed = int(seed)
    train_sources = tuple(train_sources)
    if train_samples <= 0 or validation_samples <= 0 or sequence_length <= 0:
        raise ValueError("ParoQuant calibration sample counts and sequence length must be positive.")
    if not train_sources:
        raise ValueError("ParoQuant calibration requires at least one training source.")

    load = dataset_loader or _default_dataset_loader
    per_source = train_samples // len(train_sources)
    source_counts = [per_source for _source in train_sources]
    source_counts[-1] += train_samples - sum(source_counts)

    train_sequences: list[torch.Tensor] = []
    for source, source_count in zip(train_sources, source_counts):
        dataset = _load_calibration_source(
            source,
            split="train",
            seed=seed,
            dataset_loader=load,
        )
        train_sequences.extend(
            _sample_full_sequences(
                dataset,
                tokenizer=tokenizer,
                sample_count=source_count,
                sequence_length=sequence_length,
            )
        )
    random.Random(seed).shuffle(train_sequences)

    validation_dataset = _load_calibration_source(
        validation_source,
        split="validation",
        seed=seed,
        dataset_loader=load,
    )
    validation_sequences = _sample_full_sequences(
        validation_dataset,
        tokenizer=tokenizer,
        sample_count=validation_samples,
        sequence_length=sequence_length,
    )

    return ParoQuantCalibrationDatasets(
        train=_as_calibration_examples(train_sequences),
        validation=_as_calibration_examples(validation_sequences),
        sequence_length=sequence_length,
        seed=seed,
    )


__all__ = [
    "PAROQUANT_PAPER_TRAIN_SOURCES",
    "PAROQUANT_PAPER_VALIDATION_SOURCE",
    "ParoQuantCalibrationDatasets",
    "build_paroquant_calibration_datasets",
]
