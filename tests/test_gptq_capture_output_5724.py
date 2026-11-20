# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""#5724: the GPTQ capture hook must not build a forward output it never reads.

GPTQ, GPTAQ and FOEM accumulate only ``X^T X`` from the module input. The hook
still sliced, boolean-gathered and ``.contiguous()``-copied the module output
once per calibration sample per hooked module and handed it to ``add_batch``,
which dropped it. On an accelerator each of those boolean gathers also costs a
``nonzero`` device-to-host sync, so a wide projection paid a full
``kept_tokens x out_features`` copy plus a sync per sample for nothing.

The hook now asks the task whether it reads the output (``needs_module_output``)
and passes ``None`` when it does not. These tests pin both halves: the default
path never touches the output tensor, and a task that opts in still receives
the kept rows. The Hessian must not depend on the choice.
"""

from types import SimpleNamespace

import torch

from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


# Three tokens per sample; the middle sample keeps only its last token.
MASK = torch.tensor(
    [
        [True, True, False],
        [False, False, True],
        [True, False, False],
    ]
)
KEPT_ROWS = [(0, [0, 1]), (1, [2]), (2, [0])]


class _GatherForbidden(torch.Tensor):
    """A tensor that fails the test as soon as the hook indexes it."""

    @staticmethod
    def __new__(cls, data: torch.Tensor) -> "_GatherForbidden":
        return torch.Tensor._make_subclass(cls, data, False)

    def __getitem__(self, index):
        raise AssertionError("the capture hook sliced the module output (#5724)")


def _processor(task) -> GPTQProcessor:
    processor = GPTQProcessor.__new__(GPTQProcessor)
    processor.tasks = {"linear": task}
    processor._batch_tls = SimpleNamespace(index=3)
    processor._mask_tls = SimpleNamespace(value=MASK)
    processor.qcfg = None
    processor._activation_headroom_probe = False
    processor.preserve_batch_keep_mask = True
    return processor


def _capture(*, needs_output: bool):
    """Run one capture forward; return the Hessian and what add_batch received."""

    layer = torch.nn.Linear(4, 6, bias=False)
    task = GPTQ(layer, qcfg=QuantizeConfig(bits=4, group_size=2))
    task.needs_module_output = needs_output
    seen = []
    original = task.add_batch

    def spy(inp, out, batch_index=None):
        seen.append((inp, out, batch_index))
        return original(inp, out, batch_index=batch_index)

    task.add_batch = spy

    torch.manual_seed(0)
    inputs = torch.randn(3, 3, 4)
    outputs = layer(inputs)
    _processor(task).pre_process_fwd_hook("linear")(layer, (inputs,), outputs)
    return task.finalize_hessian(), task, inputs, outputs, seen


def test_the_hook_never_indexes_an_output_the_task_does_not_read():
    """Red on main: the hook gathered ``out`` before ``add_batch`` dropped it."""

    layer = torch.nn.Linear(4, 6, bias=False)
    task = GPTQ(layer, qcfg=QuantizeConfig(bits=4, group_size=2))
    seen = []
    original = task.add_batch

    def spy(inp, out, batch_index=None):
        seen.append(out)
        return original(inp, out, batch_index=batch_index)

    task.add_batch = spy

    inputs = torch.randn(3, 3, 4)
    outputs = _GatherForbidden(layer(inputs))
    _processor(task).pre_process_fwd_hook("linear")(layer, (inputs,), outputs)

    assert len(seen) == len(KEPT_ROWS)
    assert all(out is None for out in seen), seen


def test_a_task_that_reads_the_output_still_receives_the_kept_rows():
    """Control: opting in keeps the per-sample row selection the hook had."""

    _, _, inputs, outputs, seen = _capture(needs_output=True)

    assert len(seen) == len(KEPT_ROWS)
    for (inp, out, batch_index), (row, positions) in zip(seen, KEPT_ROWS):
        assert batch_index == 3
        assert torch.is_tensor(out)
        torch.testing.assert_close(inp, inputs[row : row + 1, positions, :])
        torch.testing.assert_close(out, outputs[row : row + 1, positions, :])
        assert out.is_contiguous()


def test_the_hessian_is_identical_whether_or_not_the_output_is_gathered():
    """The dropped output must not have contributed anything to the statistics."""

    without, _, _, _, _ = _capture(needs_output=False)
    with_output, _, _, _, _ = _capture(needs_output=True)

    torch.testing.assert_close(without, with_output, rtol=0, atol=0)
