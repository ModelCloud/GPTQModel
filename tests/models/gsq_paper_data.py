# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Paper-sized GSQ token streams from the local nm calibration dataset.

The source has fewer unique tokens than the paper's FineWeb-Edu recipe. Source
rows stay disjoint across splits; each split repeats its own rows as needed.
"""

import hashlib
import random
from itertools import cycle
from pathlib import Path

import pyarrow.parquet as pq

NM_DATASET = Path("/monster/data/model/dataset/nm-calibration/llm.parquet")
SOURCE_ROWS = 10_000
SOURCE_SHA256 = "26122fd822e64d2fc704b0fe84af7a2df8a24a4455d347e16a6b5a7484f5cbef"
SEQUENCE_LENGTH = 4096
TRAIN_SEQUENCES = 4096
VALIDATION_SEQUENCES = 128
GPTQ_SEQUENCES = 512


def _complete_documents(texts, tokenizer, *, count, sequence_length):
    tokenized = [tokenizer(text, return_tensors=None)["input_ids"] for text in texts]
    tokenized = [tokens for tokens in tokenized if tokens]
    if not tokenized:
        raise ValueError("GSQ calibration split contains no tokens")
    documents = []
    buffer = []
    for tokens in cycle(tokenized):
        buffer.extend(tokens)
        while len(buffer) >= sequence_length:
            documents.append({"input_ids": buffer[:sequence_length]})
            del buffer[:sequence_length]
            if len(documents) == count:
                return documents


def split_documents(texts, tokenizer, *, sequence_length=SEQUENCE_LENGTH,
                    train_sequences=TRAIN_SEQUENCES,
                    validation_sequences=VALIDATION_SEQUENCES,
                    gptq_sequences=GPTQ_SEQUENCES):
    """Shuffle source rows, partition them, then fill each split independently."""
    total = train_sequences+validation_sequences+gptq_sequences
    if sequence_length < 1 or min(train_sequences, validation_sequences, gptq_sequences) < 1:
        raise ValueError("GSQ paper budget requires positive sequence and split sizes")
    texts = list(texts)
    if len(texts) < 3 or any(not isinstance(text, str) for text in texts):
        raise ValueError("GSQ calibration requires at least three text rows")
    random.Random(42).shuffle(texts)
    train_rows = round(len(texts)*train_sequences/total)
    validation_rows = round(len(texts)*validation_sequences/total)
    if min(train_rows, validation_rows, len(texts)-train_rows-validation_rows) < 1:
        raise ValueError("GSQ calibration cannot allocate disjoint source rows")
    source_splits = (texts[:train_rows],
                     texts[train_rows:train_rows+validation_rows],
                     texts[train_rows+validation_rows:])
    return tuple(_complete_documents(rows, tokenizer, count=count, sequence_length=sequence_length)
                 for rows, count in zip(source_splits,
                                        (train_sequences, validation_sequences, gptq_sequences)))


def nm_calibration_documents(tokenizer):
    if hashlib.sha256(NM_DATASET.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("GSQ nm calibration parquet differs from the pinned source")
    table = pq.read_table(NM_DATASET, columns=["text"])
    if table.num_rows != SOURCE_ROWS:
        raise ValueError("GSQ nm calibration parquet row count changed")
    return split_documents(table.column("text").to_pylist(), tokenizer)
