# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Sink exclusion must drop the BOS rows, not the whole batch.

``attention_sink_positions_for_batch`` returns a ``[B, S]`` boolean mask of BOS
positions. ``_extract_cache_batch_lengths`` subtracted ``numel()`` from the
batch's token count, which removed every token of the batch and zeroed every
planned length whenever ``exclude_attention_sinks=True`` (#5275). A zero budget
then sent every dense module through the hard-coded 906-row reservoir instead
of the documented ``max(largest_batch, batches)`` plan.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import torch

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.quantization.config import CalibrationConfig


SEQ_LEN = 8


def _processor(*, exclude_sinks: bool, sinks=None) -> AWQProcessor:
    """A processor holding only the state ``_extract_cache_batch_lengths`` reads."""
    cache = SimpleNamespace()
    cache.layer_inputs = [[torch.randn(1, SEQ_LEN, 4)]]
    cache.attention_masks = [None]
    cache.position_ids = [torch.arange(SEQ_LEN).unsqueeze(0)]
    cache.layer_input_kwargs = [{}]

    processor = AWQProcessor.__new__(AWQProcessor)
    processor.lock = threading.Lock()
    processor._batch_tls = threading.local()
    processor._mask_tls = threading.local()
    processor.tasks = {}
    processor._capture_host_cache = {}
    processor._capture_sampled_cache = {}
    processor._token_row_plans = {}
    processor._capture_host_batch = None
    processor._module_forward_kwargs = {}
    processor._feature_task_names = set()
    processor._feature_stats = {}
    processor._awq_feature_kwargs = {}
    processor._scale_feature_by_module = {}
    processor._nsamples_total = 1
    processor.inputs_cache = cache
    processor.calibration = CalibrationConfig(exclude_attention_sinks=exclude_sinks)
    if sinks is not None:
        processor.attention_sink_positions = sinks
    processor.gptq_model = SimpleNamespace()
    return processor


def _sink_mask(*positions: int) -> torch.Tensor:
    mask = torch.zeros(1, SEQ_LEN, dtype=torch.bool)
    for position in positions:
        mask[0, position] = True
    return mask


def test_one_sink_drops_one_row():
    processor = _processor(exclude_sinks=True, sinks=[_sink_mask(0)])

    assert processor._extract_cache_batch_lengths() == [SEQ_LEN - 1]


def test_every_sink_position_is_dropped():
    processor = _processor(exclude_sinks=True, sinks=[_sink_mask(0, 3, SEQ_LEN - 1)])

    assert processor._extract_cache_batch_lengths() == [SEQ_LEN - 3]


def test_without_sinks_the_length_is_unchanged():
    # No captured BOS positions at all.
    assert _processor(exclude_sinks=True)._extract_cache_batch_lengths() == [SEQ_LEN]
    # Sinks captured but exclusion is off.
    assert _processor(
        exclude_sinks=False, sinks=[_sink_mask(0)]
    )._extract_cache_batch_lengths() == [SEQ_LEN]


def test_masked_and_unmasked_batches_plan_the_same_length():
    """The mask path already subtracted the sinks; the unmasked path must agree."""
    sinks = [_sink_mask(0, SEQ_LEN - 1)]
    unmasked = _processor(exclude_sinks=True, sinks=sinks)
    masked = _processor(exclude_sinks=True, sinks=sinks)
    masked.inputs_cache.attention_masks = [torch.ones(1, SEQ_LEN, dtype=torch.bool)]

    assert unmasked._extract_cache_batch_lengths() == [SEQ_LEN - 2]
    assert masked._extract_cache_batch_lengths() == [SEQ_LEN - 2]
