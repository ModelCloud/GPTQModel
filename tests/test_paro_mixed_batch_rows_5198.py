# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""Regression for #5198: ParoQuant must keep every calibration batch.

``ParoProcessor._layer_input_features`` used to join per-batch captures with
``torch.cat(tensors, dim=0)`` and silently fall back to ``tensors[0]`` on the
resulting ``RuntimeError``. Routed MoE experts always capture a different
number of rows per batch, so every expert (and every dense layer when one
calibration row is shorter than the per-batch quota) was optimized on the
first batch only, with no warning.

The captures are now flattened to rows and concatenated on the row axis.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import torch

from gptqmodel.looper.paro_processor import ParoProcessor


def _processor(train_rows: int = 1000, val_rows: int = 100) -> ParoProcessor:
    processor = object.__new__(ParoProcessor)
    processor.lock = threading.Lock()
    processor._batch_tls = threading.local()
    processor.tasks = {}
    processor.qcfg = SimpleNamespace(
        opt_train_samples=train_rows,
        opt_validation_samples=val_rows,
    )
    processor.inputs_cache = None
    processor.calibration = None
    processor._ensure_task_bucket("proj", 0)
    return processor


def test_mixed_row_counts_keep_every_batch_in_batch_order():
    processor = _processor()

    # Out of order completion, different row counts: both must survive and be
    # replayed in calibration order.
    processor._set_current_batch_index(1)
    processor._record_input_feature("proj", torch.full((1, 7, 3), 2.0))
    processor._set_current_batch_index(0)
    processor._record_input_feature("proj", torch.ones(1, 5, 3))

    features = processor._layer_input_features(SimpleNamespace(modules={"proj": None}))["proj"]

    assert features.shape == (1, 12, 3)
    assert features[0, :5, 0].tolist() == [1.0] * 5
    assert features[0, 5:, 0].tolist() == [2.0] * 7


def test_rows_are_still_capped_after_joining():
    processor = _processor(train_rows=4, val_rows=1)

    processor._set_current_batch_index(0)
    processor._record_input_feature("proj", torch.arange(10.0).view(1, 10, 1))
    processor._set_current_batch_index(1)
    processor._record_input_feature("proj", torch.arange(10.0, 20.0).view(1, 10, 1))

    features = processor._layer_input_features(SimpleNamespace(modules={"proj": None}))["proj"]

    assert features.shape == (1, 5, 1)
    # The bounded replay is sampled from the joined rows, not from batch 0.
    assert features.max().item() > 10.0


def test_empty_captures_are_skipped():
    processor = _processor()

    processor._set_current_batch_index(0)
    processor._record_input_feature("proj", torch.empty(1, 0, 3))
    processor._set_current_batch_index(1)
    processor._record_input_feature("proj", torch.ones(1, 4, 3))

    features = processor._layer_input_features(SimpleNamespace(modules={"proj": None}))["proj"]

    assert features.shape == (1, 4, 3)
