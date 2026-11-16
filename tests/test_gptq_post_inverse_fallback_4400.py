# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #4400 item 4: GPTQ's CPU fallback must cover the block update.

The Hessian-inverse fallback moved ``W``/``H`` to the host, but the W-sized
``Losses``/``Q`` allocations and the ``W[:, i2:] -= Err1.matmul(...)`` block
update that follow still ran on the failed device and aborted the run. The
block pass now retries on CPU under the same handler.
"""

import pytest
import torch

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the fallback only engages for accelerator allocations",
)

ROWS, COLUMNS = 32, 64


def _task(device):
    # A dedicated generator keeps the CPU and CUDA fixtures identical: the
    # layer's own init draws from the global RNG, which differs per device.
    generator = torch.Generator().manual_seed(4400)
    weight = torch.randn(ROWS, COLUMNS, generator=generator) * 0.05
    layer = torch.nn.Linear(COLUMNS, ROWS, bias=False, dtype=torch.float32, device=device)
    with torch.no_grad():
        layer.weight.copy_(weight.to(device))

    qcfg = QuantizeConfig(
        bits=4,
        group_size=16,
        desc_act=False,
        sym=True,
        damp_percent=0.125,
        device=str(device),
    )
    task = GPTQ(layer, qcfg=qcfg)
    task.quantizer.configure(perchannel=True)
    # Diagonal Hessian: CPU and CUDA Cholesky/inverse are bitwise identical, so
    # the only difference under test is where the block pass runs.
    task.H = torch.eye(COLUMNS) * float(COLUMNS)
    task._hessian_dirty = False
    task._hessian_total_samples = 64
    task.nsamples = 64
    return task


def _fail_first_w_shaped_zeros_like(monkeypatch, error):
    real_zeros_like = torch.zeros_like
    state = {"failed": False}

    def flaky(tensor, *args, **kwargs):
        if (
            not state["failed"]
            and isinstance(tensor, torch.Tensor)
            and tensor.device.type == "cuda"
            and tensor.dim() == 2
            and tuple(tensor.shape) == (ROWS, COLUMNS)
        ):
            state["failed"] = True
            raise error
        return real_zeros_like(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "zeros_like", flaky)
    return state


def test_post_inverse_allocation_failure_retries_on_cpu(monkeypatch):
    reference, *_ = _task("cpu").quantize(blocksize=16)

    task = _task("cuda")
    state = _fail_first_w_shaped_zeros_like(
        monkeypatch,
        torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 32.00 MiB"),
    )
    messages = []
    monkeypatch.setattr(
        "gptqmodel.quantization.gptq.log.warn",
        lambda message, *args, **kwargs: messages.append(
            message % args if args else message
        ),
    )

    retried, *_ = task.quantize(blocksize=16)

    assert state["failed"] is True
    assert "block update" in "\n".join(messages)
    assert retried.device.type == "cuda"
    assert torch.equal(reference.cpu(), retried.cpu())


def test_post_inverse_non_memory_failure_is_not_retried(monkeypatch):
    task = _task("cuda")
    state = _fail_first_w_shaped_zeros_like(
        monkeypatch, RuntimeError("unrelated block-pass failure")
    )

    with pytest.raises(RuntimeError, match="unrelated block-pass failure"):
        task.quantize(blocksize=16)

    assert state["failed"] is True
