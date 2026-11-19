# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""AWQ routed-expert captures must not keep calibration padding rows (#5337).

#4035 fixed this for GPTQ only: the MoE-root hook zeroes padded positions
before routing, and the expert hook drops the all-zero rows. AWQ had neither,
so every padded position the router sent to an expert counted toward that
expert's `raw_tokens` and its scale/clip sample. Padding is not spread evenly
(the router sends it to a few experts), so those experts could be dominated by
pad-token activations at `batch_size > 1`.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import torch
from torch import nn

from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.quantization.config import CalibrationConfig


def _processor():
    processor = AWQProcessor.__new__(AWQProcessor)
    processor.lock = threading.Lock()
    processor._batch_tls = threading.local()
    processor._batch_tls.index = 0
    processor._mask_tls = threading.local()
    processor.tasks = {}
    processor._capture_host_cache = {}
    processor._capture_sampled_cache = {}
    processor._token_row_plans = {}
    processor._capture_host_batch = None
    processor.lifecycle = SimpleNamespace(
        forward=SimpleNamespace(activation_capture=True)
    )
    processor.calibration = CalibrationConfig(exclude_attention_sinks=False)
    model = SimpleNamespace()
    model.awq_input_feature_aggregation = lambda name: None
    processor.gptq_model = model
    return processor


def _bind_mask(processor, mask):
    tls = processor._mask_tls
    tls.value = mask
    tls.padding_value = mask
    tls.mask_has_padding = not bool(mask.all())


def test_routed_expert_capture_drops_zeroed_padding_rows():
    processor = _processor()
    # A 2 x 3 batch with three padding positions.
    _bind_mask(processor, torch.tensor([[True, True, False], [True, False, False]]))

    inp = torch.randn(5, 8)
    inp[2] = 0
    inp[4] = 0
    module_name = "mlp.experts.1.gate_proj"
    processor.tasks[module_name] = {"inputs": [], "batch_indices": []}

    processor._record_input_feature(module_name, inp)

    entry = processor.tasks[module_name]
    assert len(entry["inputs"]) == 1
    kept = entry["inputs"][0]
    assert tuple(kept.shape) == (1, 3, 8)
    torch.testing.assert_close(kept[0], inp[[0, 1, 3]])


def test_routed_expert_capture_keeps_rows_without_padding():
    processor = _processor()
    _bind_mask(processor, torch.ones((2, 3), dtype=torch.bool))

    inp = torch.randn(5, 8)
    inp[2] = 0
    module_name = "mlp.experts.1.gate_proj"
    processor.tasks[module_name] = {"inputs": [], "batch_indices": []}

    processor._record_input_feature(module_name, inp)

    kept = processor.tasks[module_name]["inputs"][0]
    assert tuple(kept.shape) == (1, 5, 8)
    torch.testing.assert_close(kept[0], inp)


def test_moe_root_hook_zeroes_padded_positions():
    processor = _processor()
    _bind_mask(processor, torch.tensor([[True, True, False]]))

    seen = {}

    class _Block(nn.Module):
        def forward(self, x):
            seen["x"] = x
            return x

    block = _Block()
    handles = []
    # The root capture is opt-in, but the padding zeroing is not.
    assert processor.register_moe_root_capture_hook(block, "mlp", handles) is False

    block(torch.ones(1, 3, 4))

    torch.testing.assert_close(seen["x"][0, :2], torch.ones(2, 4))
    torch.testing.assert_close(seen["x"][0, 2], torch.zeros(4))


def test_moe_root_hook_is_a_no_op_without_padding():
    processor = _processor()
    _bind_mask(processor, torch.ones((1, 3), dtype=torch.bool))

    seen = {}

    class _Block(nn.Module):
        def forward(self, x):
            seen["x"] = x
            return x

    block = _Block()
    handles = []
    processor.register_moe_root_capture_hook(block, "mlp", handles)

    hidden = torch.randn(1, 3, 4)
    hidden[0, 1] = 0
    block(hidden)

    torch.testing.assert_close(seen["x"], hidden)
