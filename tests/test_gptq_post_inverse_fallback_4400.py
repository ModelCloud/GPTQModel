# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #4400 item 4: GPTQ's CPU fallback must cover the block update.

The Hessian-inverse fallback moved ``W``/``H`` to the host, but the W-sized
``Losses``/``Q`` allocations and the ``W[:, i2:] -= Err1.matmul(...)`` block
update that follow still ran on the failed device and aborted the run. The
block pass now retries on CPU under the same handler.

Issue #4464: the block-pass retry must restart from the pre-pass weight (the
failed pass already error-propagated finished blocks into ``W`` in place), and
every CPU fallback must carry ``static_groups`` quantizers along with ``W``.

Issue #4463 item 2: QQQ must not keep the input Hessian alive through its
inverse and block pass.
"""

import pytest
import torch

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.qqq import QQQ


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


# --- issue #4464 ---------------------------------------------------------

DENSE_ROWS, DENSE_COLUMNS, DENSE_BLOCK = 32, 128, 16
DEAD_COLUMNS = (5, 77)


def _dense_hessian(columns, generator, dead_columns=()):
    """Dense, correlated Hessian, optionally with calibration-dead channels.

    A diagonal Hessian has no off-block entries, so ``W[:, i2:] -= ...``
    never changes ``W`` and a retry that reuses the partially updated weight
    looks correct. Correlated inputs make every block update dense.
    """

    inputs = torch.randn(4 * columns, columns, generator=generator)
    half = columns // 2
    inputs[:, :half] += 0.8 * inputs[:, half:]
    for column in dead_columns:
        inputs[:, column] = 0
    return inputs.T @ inputs / inputs.shape[0]


def _dense_task(device, dead_columns=DEAD_COLUMNS, **qcfg_overrides):
    generator = torch.Generator().manual_seed(4464)
    weight = torch.randn(DENSE_ROWS, DENSE_COLUMNS, generator=generator) * 0.05
    hessian = _dense_hessian(DENSE_COLUMNS, generator, dead_columns)
    layer = torch.nn.Linear(
        DENSE_COLUMNS, DENSE_ROWS, bias=False, dtype=torch.float32, device=device
    )
    with torch.no_grad():
        layer.weight.copy_(weight.to(device))

    qcfg = {
        "bits": 4,
        "group_size": 32,
        "desc_act": False,
        "sym": True,
        "damp_percent": 0.01,
        "device": str(device),
    }
    qcfg.update(qcfg_overrides)
    task = GPTQ(layer, qcfg=QuantizeConfig(**qcfg))
    task.quantizer.configure(perchannel=True)
    task.H = hessian.to(device)
    task._hessian_dirty = False
    task._hessian_total_samples = 4 * DENSE_COLUMNS
    task.nsamples = 4 * DENSE_COLUMNS
    return task


def _fail_nth_block_update(monkeypatch, fail_at):
    """Raise OOM from the ``fail_at``-th CUDA ``Err1.matmul`` block update."""

    real_matmul = torch.Tensor.matmul
    state = {"calls": 0, "failed": False}

    def flaky(self, other):
        if (
            self.device.type == "cuda"
            and self.dim() == 2
            and tuple(self.shape) == (DENSE_ROWS, DENSE_BLOCK)
        ):
            state["calls"] += 1
            if state["calls"] == fail_at:
                state["failed"] = True
                raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 MiB")
        return real_matmul(self, other)

    monkeypatch.setattr(torch.Tensor, "matmul", flaky)
    return state


def _fail_first_dense_w_zeros_like(monkeypatch):
    real_zeros_like = torch.zeros_like
    state = {"failed": False}

    def flaky(tensor, *args, **kwargs):
        if (
            not state["failed"]
            and isinstance(tensor, torch.Tensor)
            and tensor.device.type == "cuda"
            and tuple(tensor.shape) == (DENSE_ROWS, DENSE_COLUMNS)
        ):
            state["failed"] = True
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 MiB")
        return real_zeros_like(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "zeros_like", flaky)
    return state


_BLOCK_RETRY_CONFIGS = {
    "grouped": {},
    "per_channel": {"group_size": -1},
    "desc_act": {"desc_act": True},
    "act_group_aware": {"act_group_aware": True},
    "static_groups": {"static_groups": True},
    "static_groups_desc_act": {"static_groups": True, "desc_act": True},
}


@pytest.mark.parametrize("fail_at", (2, 4))
@pytest.mark.parametrize("config", sorted(_BLOCK_RETRY_CONFIGS))
def test_block_retry_restarts_from_the_pre_pass_weight(monkeypatch, config, fail_at):
    overrides = _BLOCK_RETRY_CONFIGS[config]

    # Reference: the OOM hits the first W-sized allocation, before any block
    # touched ``W``, so the CPU retry is a clean block pass with the same
    # (device-computed) Hessian inverse.
    with monkeypatch.context() as patch:
        clean_state = _fail_first_dense_w_zeros_like(patch)
        reference = _dense_task("cuda", **overrides).quantize(blocksize=DENSE_BLOCK)
    assert clean_state["failed"] is True

    state = _fail_nth_block_update(monkeypatch, fail_at)
    retried = _dense_task("cuda", **overrides).quantize(blocksize=DENSE_BLOCK)

    assert state["failed"] is True
    # Q, scale, zero and g_idx must equal the clean restart; reusing the
    # partially error-propagated ``W`` changes most quantized values.
    for expected, actual in zip(reference[:4], retried[:4]):
        assert torch.equal(expected.cpu(), actual.cpu())


def _fail_first_cuda_cholesky_oom(monkeypatch):
    real_cholesky = torch.linalg.cholesky
    state = {"failed": False}

    def flaky(hessian, *args, **kwargs):
        if hessian.device.type == "cuda" and not state["failed"]:
            state["failed"] = True
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return real_cholesky(hessian, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky", flaky)
    return state


def _fail_first_cuda_hessian_gather(monkeypatch):
    real_getitem = torch.Tensor.__getitem__
    state = {"failed": False}

    def flaky(self, key):
        if (
            not state["failed"]
            and isinstance(key, torch.Tensor)
            and self.device.type == "cuda"
            and tuple(self.shape) == (DENSE_COLUMNS, DENSE_COLUMNS)
        ):
            state["failed"] = True
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return real_getitem(self, key)

    monkeypatch.setattr(torch.Tensor, "__getitem__", flaky)
    return state


@pytest.mark.parametrize(
    ("stage", "overrides", "inject"),
    (
        ("Hessian inverse", {}, _fail_first_cuda_cholesky_oom),
        ("Hessian inverse", {"desc_act": True}, _fail_first_cuda_cholesky_oom),
        ("Hessian permutation", {"desc_act": True}, _fail_first_cuda_hessian_gather),
        (
            "act-group Hessian permutation",
            {"act_group_aware": True},
            _fail_first_cuda_hessian_gather,
        ),
    ),
)
def test_static_groups_cpu_fallback_matches_cpu_run(monkeypatch, stage, overrides, inject):
    overrides = {"static_groups": True, **overrides}
    # No dead channels: their tied Hessian diagonal would make the activation
    # order depend on the device's argsort tie-breaking.
    reference = _dense_task("cpu", dead_columns=(), **overrides).quantize(
        blocksize=DENSE_BLOCK
    )

    messages = []
    monkeypatch.setattr(
        "gptqmodel.quantization.gptq.log.warn",
        lambda message, *args, **kwargs: messages.append(
            message % args if args else message
        ),
    )
    state = inject(monkeypatch)
    retried = _dense_task("cuda", dead_columns=(), **overrides).quantize(
        blocksize=DENSE_BLOCK
    )

    assert state["failed"] is True
    assert f"during {stage} on" in "\n".join(messages)
    assert retried[0].device.type == "cuda"
    for expected, actual in zip(reference[:4], retried[:4]):
        assert torch.equal(expected.cpu(), actual.cpu())


def _dense_qqq_task(device, rows=DENSE_ROWS, columns=DENSE_COLUMNS, static_groups=True):
    generator = torch.Generator().manual_seed(4464)
    weight = torch.randn(rows, columns, generator=generator) * 0.05
    hessian = _dense_hessian(columns, generator)
    layer = torch.nn.Linear(columns, rows, bias=False, dtype=torch.float32, device=device)
    with torch.no_grad():
        layer.weight.copy_(weight.to(device))

    qcfg = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        static_groups=static_groups,
        sym=True,
        damp_percent=0.01,
        device=str(device),
    )
    task = QQQ(layer, qcfg=qcfg)
    task.quantizer.configure(4, perchannel=True, sym=True, groupsize=32)
    task.H = hessian.to(device)
    task._materialized_samples = 4 * columns
    task.nsamples = 4 * columns
    return task


def test_qqq_static_groups_cpu_fallback_matches_cpu_run(monkeypatch):
    reference = _dense_qqq_task("cpu").quantize(blocksize=DENSE_BLOCK)

    state = _fail_first_cuda_cholesky_oom(monkeypatch)
    retried = _dense_qqq_task("cuda").quantize(blocksize=DENSE_BLOCK)

    assert state["failed"] is True
    assert retried[0].device.type == "cuda"
    for expected, actual in zip(reference[:4], retried[:4]):
        assert torch.equal(expected.cpu(), actual.cpu())


def test_qqq_late_inverse_step_failure_finishes_on_cpu(monkeypatch):
    """An OOM after the Cholesky step continues from the factor on the host.

    The input Hessian is released once its factor exists, so the retry must
    not need it.
    """

    real_cholesky_inverse = torch.cholesky_inverse
    state = {"failed": False}

    def flaky(factor, *args, **kwargs):
        if factor.device.type == "cuda" and not state["failed"]:
            state["failed"] = True
            raise RuntimeError("cusolver error: CUSOLVER_STATUS_ALLOC_FAILED")
        return real_cholesky_inverse(factor, *args, **kwargs)

    monkeypatch.setattr(torch, "cholesky_inverse", flaky)
    messages = []
    monkeypatch.setattr(
        "gptqmodel.quantization.qqq.log.warn",
        lambda message, *args, **kwargs: messages.append(
            message % args if args else message
        ),
    )

    retried = _dense_qqq_task("cuda").quantize(blocksize=DENSE_BLOCK)

    assert state["failed"] is True
    assert "falling back to CPU" in "\n".join(messages)
    assert retried[0].device.type == "cuda"
    assert torch.isfinite(retried[0]).all()


def test_qqq_releases_input_hessian_before_inverse_and_block_pass():
    columns, rows = 2048, 64
    hessian_bytes = columns * columns * 4
    task = _dense_qqq_task("cuda", rows=rows, columns=columns, static_groups=False)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    task.quantize()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline

    # The baseline already holds the Hessian. Releasing it as soon as its
    # Cholesky factor exists keeps the inverse at ~3 Hessian-sized buffers on
    # top of it (factor, identity, result); holding the input Hessian through
    # the inverse and the block pass measured ~4.
    assert peak < 3.5 * hessian_bytes, peak / hessian_bytes
