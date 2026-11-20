# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""FOEM must ignore a calibration batch with no usable routed rows (#5768).

When a capture's first batch routes only padding rows to an expert, the
processor filters the activation to ``[0, K]``. FOEM still entered
``_accumulate_device_batch``, where the per-device normalization divided by a
zero token count and raised ``ZeroDivisionError``. Plain GPTQ accepts the empty
contribution and later falls back to RTN with zero counted rows.

FOEM now returns before allocating an accumulator or scaling the batch. These
tests pin an empty first contribution, an empty contribution after a real
batch (Hessian and native correction unchanged), and the full capture path
that stays empty and reaches RTN fallback.
"""

from __future__ import annotations

import threading

import pytest
import torch
import torch.nn as nn

from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.native_processor import NATIVE_INPUTS_STATE_KEY
from gptqmodel.quantization.config import (
    CalibrationConfig,
    FOEMConfig,
    QuantizeConfig,
)
from gptqmodel.quantization.foem import FOEM


COLUMNS = 16


def _qcfg() -> QuantizeConfig:
    return QuantizeConfig(bits=4, group_size=8, foem=FOEMConfig(alpha=0.25))


def _foem_task(*, routed: bool, columns: int = COLUMNS) -> FOEM:
    named = NamedModule(
        nn.Linear(columns, 8, bias=False),
        name="proj",
        full_name="model.layers.0.proj",
        layer_index=0,
    )
    named.state["is_routed_expert"] = routed
    named.state[NATIVE_INPUTS_STATE_KEY] = []
    task = FOEM(named, _qcfg())
    task.quantizer.configure(perchannel=True)
    return task


def _empty_batch(columns: int = COLUMNS) -> torch.Tensor:
    return torch.zeros(0, columns)


def test_empty_first_batch_allocates_no_accumulator():
    task = _foem_task(routed=True)
    token_count, xtx, device = task.process_batch(_empty_batch())

    assert token_count == 0
    assert xtx is None
    assert device.type == "cpu"
    assert task.H is None
    assert task.nsamples == 0


def test_empty_first_batch_keeps_native_correction_unallocated():
    task = _foem_task(routed=False)
    task.native_inps = [(0, _empty_batch())]

    token_count, xtx, _ = task.process_batch(_empty_batch(), batch_index=0)

    assert token_count == 0 and xtx is None
    assert task.H is None
    assert task.dXXT is None
    assert task.nsamples == 0


def test_empty_batch_after_real_batch_keeps_the_routed_hessian():
    torch.manual_seed(5768)
    rows = torch.randn(3, COLUMNS)

    reference = _foem_task(routed=True)
    reference.add_batch(rows, None)

    with_empty = _foem_task(routed=True)
    with_empty.add_batch(rows, None)
    with_empty.add_batch(_empty_batch(), None)

    assert with_empty.nsamples == reference.nsamples == 3
    torch.testing.assert_close(with_empty.H, reference.H, rtol=0, atol=0)


def test_empty_batch_after_native_batch_keeps_hessian_and_correction():
    torch.manual_seed(5769)
    rows = torch.randn(3, COLUMNS)
    native_rows = torch.randn(3, COLUMNS)

    reference = _foem_task(routed=False)
    reference.native_inps = [(0, native_rows)]
    reference.add_batch(rows, None, batch_index=0)

    with_empty = _foem_task(routed=False)
    with_empty.native_inps = [(0, native_rows), (1, _empty_batch())]
    with_empty.add_batch(rows, None, batch_index=0)
    with_empty.add_batch(_empty_batch(), None, batch_index=1)

    assert reference.gptaq and with_empty.gptaq
    assert with_empty.nsamples == reference.nsamples == 3
    torch.testing.assert_close(with_empty.H, reference.H, rtol=0, atol=0)
    assert reference.dXXT is not None and with_empty.dXXT is not None
    torch.testing.assert_close(with_empty.dXXT, reference.dXXT, rtol=0, atol=0)


@pytest.mark.parametrize("use_foem", [False, True], ids=["gptq", "foem"])
def test_padding_only_capture_reaches_rtn_fallback(use_foem):
    config = QuantizeConfig(
        bits=4,
        group_size=8,
        offload_to_disk=False,
        foem=FOEMConfig(alpha=0.25) if use_foem else None,
    )
    processor = GPTQProcessor(
        tokenizer=None, qcfg=config, calibration=CalibrationConfig()
    )
    layer = nn.Linear(8, 4, bias=False)
    named = NamedModule(
        layer, name="expert", full_name="model.layers.0.expert", layer_index=0
    )
    named.state["is_routed_expert"] = True
    named.state[NATIVE_INPUTS_STATE_KEY] = []
    processor.preprocess(named)

    # The root capture hook zeroes padding before the expert sees it, so this
    # expert receives the two padding rows only.
    processor._mask_tls = threading.local()
    processor._mask_tls.value = torch.tensor([[True, True, False, False]])
    processor._mask_tls.mask_has_padding = True
    rows = torch.zeros(2, 8)
    hook = processor.pre_process_fwd_hook(named.name)
    hook(layer, (rows,), layer(rows))

    task = processor.tasks[named.name]
    loss = task.quantize()[5]
    assert task.nsamples == 0
    assert loss.startswith("fallback(rtn)")
