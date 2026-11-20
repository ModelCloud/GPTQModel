# GPU=-1
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""EoRA calibration rows for flattened captures and MoE padding (#5719).

``ModuleLooper._masked_hook_wrapper`` masks ``[B, S, H]`` captures, so EoRA's
forward hook used to accumulate the raw input. A flattened ``[B*S, H]`` full
grid -- an all-expert MoE bypass or a shared expert fed
``hidden_states.view(-1, H)`` -- therefore kept its padding rows and any
excluded attention sinks in both the covariance and the token count, although
QQQ, MACaQ and EXL3 already filtered that layout (#5385). Natively routed
expert rows cannot be mapped back to a sequence position, so the MoE block
input is instead zeroed at the padded rows, which contributes nothing to any
expert's ``X^T X``.

These tests pin:

* the flat and grid captures of one batch accumulate the same retained rows;
* those rows are exactly the padding-only or sink-excluded mask, and the
  float64 oracle over them;
* a capture with no bound mask is unchanged;
* the MoE padding hook zeroes only padded rows, for positional and keyword
  ``hidden_states``, and GPTQ keeps its ``preserve_batch_keep_mask`` gate.
"""

import threading
from types import SimpleNamespace

import torch
import torch.nn as nn

from gptqmodel.eora.eora import eora_process_input
from gptqmodel.looper.eora_processor import EoraProcessor
from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.quantization.config import CalibrationConfig


BOS = 7
HIDDEN = 8
NAME = "layer.proj"


class _Module(nn.Module):
    """A module EoRA can identify the device of and that returns its input."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(0))

    def forward(self, hidden_states):
        return hidden_states


def _batch():
    """BOS + right pad, left pad + BOS, and a window with a mid-row BOS."""

    input_ids = torch.tensor(
        [
            [BOS, 11, 12, 13, 0],
            [0, 0, BOS, 21, 22],
            [31, 32, BOS, 34, 35],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    return input_ids, attention_mask


def _expected_keep() -> torch.Tensor:
    keep = _batch()[1].bool().clone()
    keep[0, 0] = False  # leading BOS of a right-padded row
    keep[1, 2] = False  # leading BOS of a left-padded row
    keep[2, 2] = False  # document-boundary BOS inside the window
    return keep


def _inputs() -> torch.Tensor:
    torch.manual_seed(0)
    x = torch.randn(3, 5, HIDDEN, dtype=torch.float32)
    # Make the sinks dominate, as they do in a real model.
    x[0, 0] *= 300.0
    x[1, 2] *= 300.0
    x[2, 2] *= 300.0
    return x


def _processor(*, exclude_sinks: bool) -> EoraProcessor:
    processor = object.__new__(EoraProcessor)
    processor.lock = threading.Lock()
    processor.num_batches = 16
    processor._segment_accumulators = {NAME: {}}
    processor._module_target_devices = {NAME: torch.device("cpu")}
    processor._batch_tls = SimpleNamespace(index=0)
    processor.eora_process_input = eora_process_input
    processor.calibration = CalibrationConfig(exclude_attention_sinks=exclude_sinks)
    processor.tokenizer = SimpleNamespace(bos_token_id=BOS)
    input_ids, attention_mask = _batch()
    processor.attention_sink_positions = (
        processor._compute_attention_sink_positions(
            [{"input_ids": input_ids, "attention_mask": attention_mask}], BOS
        )[0]
        if exclude_sinks
        else None
    )
    return processor


def _bind_masks(processor, hidden: torch.Tensor):
    """Bind the masks exactly as ``ForwardExecutor`` does for one batch."""

    _, attention_mask = _batch()
    padding = attention_mask.bool()
    keep = processor.calibration_keep_mask(padding, batch_index=0, layer_input=[hidden])
    ModuleLooper.__new__(ModuleLooper)._set_processor_mask(processor, keep, padding_mask=padding)
    return keep, padding


def _accumulate(processor, x: torch.Tensor):
    """Run EoRA's forward hook once and return the accumulated record."""

    processor.pre_process_fwd_hook(NAME)(_Module(), (x,), None)
    records = processor._segment_accumulators[NAME]
    assert len(records) == 1
    return next(iter(records.values()))


def _oracle(rows: torch.Tensor) -> torch.Tensor:
    rows64 = rows.reshape(-1, rows.shape[-1]).to(torch.float64)
    return rows64.T @ rows64


def test_flat_capture_accumulates_only_the_calibration_rows():
    x = _inputs()
    processor = _processor(exclude_sinks=True)
    keep, _ = _bind_masks(processor, x)
    assert torch.equal(keep, _expected_keep())
    assert int(keep.sum()) == 9

    # An all-expert MoE bypass / shared expert reaches the hook flattened.
    record = _accumulate(processor, x.reshape(-1, HIDDEN))

    assert record["token_count"] == 9
    torch.testing.assert_close(
        record["total"].to(torch.float64), _oracle(x[keep]), rtol=1e-5, atol=1e-5
    )


def test_flat_and_grid_captures_accumulate_the_same_statistics():
    x = _inputs()
    grid_processor = _processor(exclude_sinks=True)
    keep, _ = _bind_masks(grid_processor, x)
    # What the looper's mask wrapper passes for a `[B, S, H]` capture.
    grid_record = _accumulate(grid_processor, x[keep])

    flat_processor = _processor(exclude_sinks=True)
    _bind_masks(flat_processor, x)
    flat_record = _accumulate(flat_processor, x.reshape(-1, HIDDEN))

    assert flat_record["token_count"] == grid_record["token_count"] == 9
    torch.testing.assert_close(flat_record["total"], grid_record["total"], rtol=1e-6, atol=1e-6)


def test_flat_capture_drops_padding_even_without_sinks():
    x = _inputs()
    processor = _processor(exclude_sinks=False)
    keep, padding = _bind_masks(processor, x)
    assert keep is padding
    assert int(padding.sum()) == 12

    record = _accumulate(processor, x.reshape(-1, HIDDEN))

    assert record["token_count"] == 12
    torch.testing.assert_close(
        record["total"].to(torch.float64), _oracle(x[padding]), rtol=1e-5, atol=1e-5
    )


def test_capture_without_a_bound_mask_is_unchanged():
    # The historical path: no looper mask, e.g. a hook called outside a batch.
    x = _inputs()
    record = _accumulate(_processor(exclude_sinks=True), x.reshape(-1, HIDDEN))

    assert record["token_count"] == 15
    torch.testing.assert_close(
        record["total"].to(torch.float64), _oracle(x), rtol=1e-5, atol=1e-5
    )


def test_eora_moe_hook_zeroes_padded_rows():
    x = _inputs()
    processor = _processor(exclude_sinks=True)
    _, padding = _bind_masks(processor, x)
    module = _Module()
    handles = []

    assert processor.register_moe_root_capture_hook(module, "layers.0.mlp", handles) is True
    assert len(handles) == 1

    out = module(x)
    torch.testing.assert_close(out, x * padding.unsqueeze(-1).to(x.dtype))
    # Real rows are untouched, including an attention sink.
    torch.testing.assert_close(out[padding], x[padding])
    assert torch.equal(out[0, 0], x[0, 0])
    assert out[~padding].abs().sum() == 0


def test_eora_moe_hook_zeroes_padded_rows_in_keyword_form():
    x = _inputs()
    processor = _processor(exclude_sinks=True)
    _, padding = _bind_masks(processor, x)
    module = _Module()
    handles = []

    assert processor.register_moe_root_capture_hook(module, "layers.0.mlp", handles) is True
    out = module(hidden_states=x)

    torch.testing.assert_close(out[padding], x[padding])
    assert out[~padding].abs().sum() == 0


def test_eora_moe_hook_ignores_a_mask_that_does_not_fit():
    processor = _processor(exclude_sinks=True)
    ModuleLooper.__new__(ModuleLooper)._set_processor_mask(
        processor, torch.ones(2, 3, dtype=torch.bool)
    )
    module = _Module()
    handles = []
    assert processor.register_moe_root_capture_hook(module, "layers.0.mlp", handles) is True

    x = _inputs()
    # A [B, S, H] shape that does not match the mask is forwarded unchanged.
    assert module(x) is x


def test_gptq_moe_hook_stays_gated_on_preserve_batch_keep_mask():
    x = _inputs()
    padding = _batch()[1].bool()

    gated = object.__new__(GPTQProcessor)
    gated.preserve_batch_keep_mask = False
    handles = []
    assert gated.register_moe_root_capture_hook(_Module(), "layers.0.mlp", handles) is False
    assert handles == []

    enabled = object.__new__(GPTQProcessor)
    enabled.preserve_batch_keep_mask = True
    ModuleLooper.__new__(ModuleLooper)._set_processor_mask(enabled, padding, padding_mask=padding)
    handles = []
    module = _Module()
    assert enabled.register_moe_root_capture_hook(module, "layers.0.mlp", handles) is True
    assert len(handles) == 1
    torch.testing.assert_close(module(x), x * padding.unsqueeze(-1).to(x.dtype))
