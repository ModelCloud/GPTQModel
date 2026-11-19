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


def test_routed_capture_drops_pads_when_its_row_count_equals_the_kept_count():
    """#5360: the routed row count can coincide with `keep_mask.sum()`.

    A two-by-four batch with keep mask ``[[1, 1, 1, 1], [1, 1, 0, 0]]`` has six
    attended positions. If the router sends positions ``[0, 1, 2, 3, 6, 7]`` to
    one expert, that capture has six rows -- four real and two padding -- so a
    row-count test against ``keep_mask.sum()`` cannot tell it apart from an
    already-filtered grid. Only the full `[B*S, H]` grid may skip the drop.
    """

    processor = _processor()
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    _bind_mask(processor, mask)
    assert int(mask.sum()) == 6 and int(mask.numel()) == 8

    real = torch.arange(1, 9, dtype=torch.float32).reshape(4, 2)
    # The MoE-root hook zeroes the padded positions before routing (#4035), and
    # the router still sends them: four real rows plus two pad rows here.
    routed = torch.cat([real, torch.zeros(2, 2)], dim=0)
    module_name = "mlp.experts.2.gate_proj"
    processor.tasks[module_name] = {"inputs": [], "batch_indices": []}

    processor._record_input_feature(module_name, routed)

    kept = processor.tasks[module_name]["inputs"][0]
    assert tuple(kept.shape) == (1, 4, 2)
    torch.testing.assert_close(kept[0], real)


def test_a_full_grid_capture_is_left_to_the_mask_path():
    """`[B*S, H]` rows are filtered by the mask, not by the zero check.

    Dropping zeros here would also drop a real all-zero activation, so the full
    grid keeps its rows and `_apply_active_keep_mask_to_feature` selects them.
    """

    processor = _processor()
    mask = torch.tensor([[True, True, False], [True, False, False]])
    _bind_mask(processor, mask)

    grid = torch.randn(6, 4)
    grid[3] = 0
    module_name = "mlp.shared_expert.gate_proj"
    processor.tasks[module_name] = {"inputs": [], "batch_indices": []}

    processor._record_input_feature(module_name, grid)

    kept = processor.tasks[module_name]["inputs"][0]
    # Three attended rows, the real all-zero one included.
    assert tuple(kept.shape) == (1, 3, 4)
    torch.testing.assert_close(kept[0], grid[mask.reshape(-1)])
