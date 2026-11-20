# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Parallel calibration workers must bind the derived mask flags (#5777).

The serial replay path resolves the per-sample emptiness and padding-presence
flags once per batch in ``ModuleLooper._set_processor_mask``. The parallel
worker used to publish only the two masks, so every projection hook re-read the
flags to the host and routed experts fell back to the calibration mask, which
also excludes real BOS sink tokens.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.looper.module_looper import ModuleLooper
from gptqmodel.quantization.config import CalibrationConfig, GPTQConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.utils.looper_helpers import bind_processor_masks, forward_batch_worker


WIDTH = 16
TOKENS = 4


class ProjectionBlock(nn.Module):
    def __init__(self, projections, routed):
        super().__init__()
        self.proj = nn.ModuleList(nn.Linear(WIDTH, WIDTH, bias=False) for _ in range(projections))
        self.routed = routed

    def forward(self, x, **kwargs):
        # A routed expert receives only the attended rows at positions 1 and 2
        # of the first example; position 0 is BOS and routes elsewhere.
        rows = x[0, 1:3] if self.routed else x
        for proj in self.proj:
            proj(rows)
        return x


def _identity_prepare(calibration_dataset, **kwargs):
    return calibration_dataset


def _run(x, padding_mask, *, worker, routed=False):
    qcfg = GPTQConfig(bits=4, group_size=16, offload_to_disk=False, device="cpu")
    ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    calibration = CalibrationConfig(
        dataset=[{"input_ids": ids, "attention_mask": padding_mask}],
        min_length=1,
        exclude_attention_sinks=routed,
    ).bind(_identity_prepare)
    proc = GPTQProcessor(
        tokenizer=SimpleNamespace(bos_token_id=0),
        qcfg=qcfg,
        calibration=calibration,
    )
    block = ProjectionBlock(1 if routed else 8, routed)
    looper = ModuleLooper.__new__(ModuleLooper)
    looper._processor_mask_tls(proc)

    for index, proj in enumerate(block.proj):
        name = str(index)
        proj.target_device = x.device
        proc.tasks[name] = GPTQ(proj, qcfg=qcfg)
        proj.register_forward_hook(proc.pre_process_fwd_hook(name))

    original_tolist = torch.Tensor.tolist
    reads = 0

    def count_mask_flags(tensor):
        nonlocal reads
        if tensor.dtype == torch.bool and tuple(tensor.shape) == (x.shape[0],):
            reads += 1
        return original_tolist(tensor)

    with patch.object(torch.Tensor, "tolist", count_mask_flags):
        if worker:
            forward_batch_worker(
                block,
                proc,
                0,
                [x],
                {},
                padding_mask,
                None,
                support_batch_quantize=True,
                is_lm_head_module=False,
                need_output=False,
                reuse_kv=False,
                prev_kv=None,
            )
        else:
            keep = proc.calibration_keep_mask(padding_mask, batch_index=0, layer_input=[x])
            looper._set_processor_mask(proc, keep, padding_mask=padding_mask)
            proc._set_current_batch_index(0)
            try:
                block(x)
            finally:
                looper._set_processor_mask(proc, None)

    return reads, list(proc.tasks.values())


def _reference(rows):
    rows = rows.double()
    return (2.0 / len(rows)) * (rows.T @ rows)


def test_worker_binds_the_sample_flags_once_per_batch():
    torch.set_num_threads(1)
    x = torch.arange(2 * TOKENS * WIDTH, dtype=torch.float32).reshape(2, TOKENS, WIDTH) / WIDTH
    mask = torch.tensor([[True, True, False, False], [False, False, False, False]])

    serial_reads, serial_tasks = _run(x, mask, worker=False)
    worker_reads, worker_tasks = _run(x, mask, worker=True)

    expected = _reference(x[mask])
    assert serial_reads == 1
    assert worker_reads == 1, f"worker performed {worker_reads} per-projection mask reads"
    for task in (*serial_tasks, *worker_tasks):
        assert task.nsamples == int(mask.sum())
        torch.testing.assert_close(task.finalize_hessian().double(), expected)


def test_worker_keeps_routed_rows_when_only_sinks_are_excluded():
    torch.set_num_threads(1)
    x = torch.arange(2 * TOKENS * WIDTH, dtype=torch.float32).reshape(2, TOKENS, WIDTH) / WIDTH
    x[0, 1].zero_()
    padding_mask = torch.ones(2, TOKENS, dtype=torch.bool)

    _, [serial] = _run(x, padding_mask, worker=False, routed=True)
    worker_reads, [worker] = _run(x, padding_mask, worker=True, routed=True)

    expected = _reference(x[0, 1:3])
    assert (serial.nsamples, worker.nsamples) == (2, 2)
    assert worker_reads == 1
    torch.testing.assert_close(serial.finalize_hessian().double(), expected)
    torch.testing.assert_close(worker.finalize_hessian().double(), expected)


def test_binding_derives_padding_presence_from_the_padding_mask():
    """AWQ's routed filter reads the same flag, so pin its source mask."""

    processor = SimpleNamespace()
    padding = torch.ones(2, TOKENS, dtype=torch.bool)
    calibration = padding.clone()
    calibration[0, 0] = False  # an excluded BOS attention sink

    bind_processor_masks(processor, calibration, padding_mask=padding)
    tls = processor._mask_tls
    assert tls.value is calibration
    assert tls.padding_value is padding
    assert tls.sample_has_tokens == [True, True]
    assert tls.mask_has_padding is False

    bind_processor_masks(processor, None)
    assert tls.value is None
    assert tls.padding_value is None
    assert tls.sample_has_tokens is None
    assert tls.mask_has_padding is None
