# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Calibration padding must not reach routed-expert or shared-expert Hessians.

`GPTQProcessor.pre_process_fwd_hook` dropped masked rows only for 3-D
activations. Routed experts reached through the MoE bypass and HF qwen2_moe's
shared expert are called on flattened `[B*S, H]` rows, and natively routed
experts receive a `[routed_rows, H]` tensor whose row count is unrelated to the
keep mask. Both paths used to accumulate padding into `X^T X`, so `batch_size=1`
and `batch_size>1` produced different expert weights.
"""

import threading
from types import SimpleNamespace

import torch
from torch import nn

from gptqmodel.looper.gptq_processor import GPTQProcessor


class _RecordingTask:
    """Minimal GPTQ task: records the rows the hook hands to `add_batch`."""

    def __init__(self, module=None):
        self.module = module
        self.batches = []

    def add_batch(self, inp, out, batch_index=None):
        self.batches.append((inp.clone(), out, batch_index))


def _bare_processor(mask):
    processor = GPTQProcessor.__new__(GPTQProcessor)
    processor.qcfg = SimpleNamespace(activation_mode=None)
    processor._mask_tls = SimpleNamespace(value=mask)
    processor._batch_tls = threading.local()
    processor._batch_tls.index = 0
    processor.tasks = {}
    processor.preserve_batch_keep_mask = True
    processor._activation_headroom_probe = False
    return processor


def test_2d_activations_drop_masked_rows():
    """MoE bypass / 2-D shared expert: `[B*S, H]` rows must be filtered."""

    mask = torch.tensor([[True, True, False], [True, False, False]])
    processor = _bare_processor(mask)
    task = _RecordingTask(module=nn.Linear(4, 4))
    processor.tasks["mlp.experts.0.gate_proj"] = task

    inp = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
    out = inp * 2
    hook = processor.pre_process_fwd_hook("mlp.experts.0.gate_proj")
    hook(nn.Linear(4, 4), (inp,), out)

    # One `add_batch` per sample keeps the accumulation order identical to
    # `batch_size=1`; only the three real token rows survive.
    assert len(task.batches) == 2
    kept_inp = torch.cat([batch[0] for batch in task.batches])
    kept_out = torch.cat([batch[1] for batch in task.batches])
    assert kept_inp.shape == (3, 4)
    torch.testing.assert_close(kept_inp, inp[mask.reshape(-1)])
    torch.testing.assert_close(kept_out, out[mask.reshape(-1)])


def test_2d_activations_without_a_matching_mask_are_untouched():
    """A routed-expert `[routed_rows, H]` tensor is not `B*S`; leave it alone."""

    mask = torch.tensor([[True, True, False], [True, False, False]])
    processor = _bare_processor(mask)
    task = _RecordingTask(module=nn.Linear(4, 4))
    processor.tasks["mlp.experts.1.gate_proj"] = task

    inp = torch.randn(7, 4)
    hook = processor.pre_process_fwd_hook("mlp.experts.1.gate_proj")
    hook(nn.Linear(4, 4), (inp,), inp * 2)

    kept_inp, _, _ = task.batches[0]
    torch.testing.assert_close(kept_inp, inp)


def test_routed_expert_drops_zeroed_padding_rows():
    """A natively routed expert sees zeroed padding rows; they must not count.

    The MoE-root hook zeroes padded positions before routing, but the router
    still routes them. An expert that only wins padding rows would otherwise
    build an all-zero Hessian and have every weight column neutralized.
    """

    mask = torch.tensor([[True, True, False], [True, False, False]])
    processor = _bare_processor(mask)
    task = _RecordingTask(module=nn.Linear(4, 4))
    processor.tasks["mlp.experts.1.gate_proj"] = task

    # Five routed rows for a six-token batch: four real rows and one zeroed
    # padding row. The row count is unrelated to `B*S`, as for native routing.
    inp = torch.randn(5, 4)
    inp[2] = 0
    out = inp * 2
    hook = processor.pre_process_fwd_hook("mlp.experts.1.gate_proj")
    hook(nn.Linear(4, 4), (inp,), out)

    assert len(task.batches) == 1
    kept_inp, kept_out, _ = task.batches[0]
    assert kept_inp.shape == (4, 4)
    torch.testing.assert_close(kept_inp, inp[[0, 1, 3, 4]])
    torch.testing.assert_close(kept_out, out[[0, 1, 3, 4]])


def test_routed_expert_without_batch_padding_is_untouched():
    """A full batch has no padding rows to drop, even if one row is zero."""

    mask = torch.ones((2, 3), dtype=torch.bool)
    processor = _bare_processor(mask)
    task = _RecordingTask(module=nn.Linear(4, 4))
    processor.tasks["mlp.experts.1.gate_proj"] = task

    inp = torch.randn(5, 4)
    inp[0] = 0
    out = inp * 2
    hook = processor.pre_process_fwd_hook("mlp.experts.1.gate_proj")
    hook(nn.Linear(4, 4), (inp,), out)

    kept_inp, _, _ = task.batches[0]
    torch.testing.assert_close(kept_inp, inp)


class _MoeBlock(nn.Module):
    """MoE block that records the input each routed expert receives."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(4, 2, bias=False)
        self.experts = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(2)])
        self.seen = None

    def forward(self, hidden_states):
        self.seen = hidden_states.detach().clone()
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        return self.experts[0](flat).reshape(hidden_states.shape)


def test_moe_root_hook_zeroes_padded_rows_only():
    mask = torch.tensor([[True, True, False], [True, False, False]])
    processor = _bare_processor(mask)
    block = _MoeBlock()
    handles = []

    assert processor.register_moe_root_capture_hook(block, "mlp", handles) is True
    assert len(handles) == 1

    hidden = torch.ones(2, 3, 4)
    with torch.no_grad():
        block(hidden)

    expected = hidden * mask.unsqueeze(-1).to(hidden.dtype)
    torch.testing.assert_close(block.seen, expected)
    # The caller's tensor is not mutated in place.
    torch.testing.assert_close(hidden, torch.ones(2, 3, 4))

    for handle in handles:
        handle.remove()


def test_moe_root_hook_is_skipped_without_a_mask():
    processor = _bare_processor(None)
    # Registration happens unconditionally; the hook itself is a no-op without a
    # mask, so a replay forward keeps the real activations.
    block = _MoeBlock()
    handles = []
    assert processor.register_moe_root_capture_hook(block, "mlp", handles) is True
    hidden = torch.ones(2, 3, 4)
    with torch.no_grad():
        block(hidden)
    torch.testing.assert_close(block.seen, hidden)
    for handle in handles:
        handle.remove()
