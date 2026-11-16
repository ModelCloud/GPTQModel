# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GPTAQ/FOEM Hessian accumulation across more than one input device.

Issue #4237: ``GPTAQ.process_batch``/``FOEM.process_batch`` allocated
``self.H``/``self.dXXT`` once, on the first batch's device, so a later batch
from another GPU raised at the accumulation. They also paired native inputs by
``native_inps.pop(0)`` arrival order, which the data-parallel forward does not
preserve.

The fix keeps one Hessian/dXXT partial per input device, normalizes each
partial by that device's own sample count, and reduces the partials in device
order at ``quantize()``. Native entries are tagged with their calibration batch
index by ``NativeProcessor`` and consumed by that key.

Oracle: for ``R`` calibration requests and tokens ``x``, GPTAQ accumulates
``H = (2/R) * sum_b X_b X_b^T``; FOEM accumulates ``H = (2/T) * sum_t x_t x_t^T``
with ``T`` total tokens. Both use the same weight for
``dXXT = sum (native - x) x^T``. Splitting either sum by device and reducing
with ``count_device / count_total`` must reproduce it.
"""

import threading

import pytest
import torch
from torch import nn

from gptqmodel.looper.loop_processor import CalibrationConfig
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.native_processor import NATIVE_INPUTS_STATE_KEY, NativeProcessor
from gptqmodel.quantization.config import FOEMConfig, GPTAQConfig, QuantizeConfig
from gptqmodel.quantization.foem import FOEM
from gptqmodel.quantization.gptaq import GPTAQ


KINDS = ("gptaq", "foem")
_CLASSES = {"gptaq": GPTAQ, "foem": FOEM}


def _qcfg(kind, device="auto"):
    if kind == "gptaq":
        return QuantizeConfig(
            bits=4, group_size=8, gptaq=GPTAQConfig(alpha=0.25, device=device)
        )
    return QuantizeConfig(
        bits=4, group_size=8, foem=FOEMConfig(alpha=0.25, beta=0.2, device=device)
    )


def _named(linear):
    return NamedModule(
        linear, name="proj", full_name="model.layers.0.proj", layer_index=0
    )


def _task(kind, named, native_inps):
    # The constructor pops the capture from module state, so install it first.
    named.state[NATIVE_INPUTS_STATE_KEY] = native_inps
    task = _CLASSES[kind](named, _qcfg(kind))
    task.quantizer.configure(perchannel=True)
    return task


def _capture(named, batches):
    """Capture ``(batch_index, x, keep_mask)`` forwards through NativeProcessor."""

    processor = NativeProcessor(
        tokenizer=None,
        qcfg=_qcfg("gptaq", device="cpu"),
        calibration=CalibrationConfig(dataset=None, batch_size=1),
    )
    processor.preprocess(named)
    hook = processor.pre_process_fwd_hook(named.name)
    processor._mask_tls = threading.local()
    for batch_index, x, keep_mask in batches:
        # ModuleLooper sets the active batch index and the [B, S] keep-mask on
        # the processor's thread-local storage before every forward pass.
        processor._set_current_batch_index(batch_index)
        processor._mask_tls.value = keep_mask
        hook(named.module, (x,), None)
        processor._set_current_batch_index(None)
    processor.process(named)
    return named.state[NATIVE_INPUTS_STATE_KEY]


def _oracle(kind, batches):
    """Independent float64 Hessian/dXXT over ``(x, native)`` ``[tokens, cols]``."""

    columns = batches[0][0].shape[1]
    requests = len(batches)
    tokens = sum(x.shape[0] for x, _native in batches)
    weight = 2.0 / (requests if kind == "gptaq" else tokens)
    hessian = torch.zeros((columns, columns), dtype=torch.float64)
    dxxT = torch.zeros_like(hessian)
    for x, native in batches:
        hessian += weight * (x.T @ x)
        dxxT += weight * ((native - x).T @ x)
    return hessian, dxxT


def _relative_error(actual, expected):
    actual = actual.to(device="cpu", dtype=torch.float64)
    expected = expected.to(device="cpu", dtype=torch.float64)
    return float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected))


def _finalize(task, kind):
    getattr(task, f"_finalize_{kind}_hessian")()


def test_native_capture_tags_entries_with_calibration_batch_index():
    """The pairing key must be captured next to the tensor, not inferred later."""

    torch.manual_seed(0)
    named = _named(nn.Linear(8, 4, bias=False))
    x0 = torch.randn(1, 3, 8)
    x1 = torch.randn(1, 2, 8)
    keep_mask = torch.ones(1, 3, dtype=torch.bool)

    # Capture completes batch 1 before batch 0, as data-parallel workers can.
    entries = _capture(named, [(1, x1, keep_mask[:, :2]), (0, x0, keep_mask)])

    assert [entry[0] for entry in entries] == [1, 0]
    assert all(isinstance(entry, tuple) and len(entry) == 2 for entry in entries)
    assert [tuple(entry[1].shape) for entry in entries] == [(2, 8), (3, 8)]


@pytest.mark.parametrize("kind", KINDS)
def test_native_inputs_pair_by_batch_index_not_arrival_order(kind):
    """Replaying batches out of capture order must still pair native inputs."""

    torch.manual_seed(1)
    named = _named(nn.Linear(8, 4, bias=False))
    x0 = torch.randn(1, 3, 8)
    x1 = torch.randn(1, 2, 8)
    native0 = torch.randn(3, 8)
    native1 = torch.randn(2, 8)

    # Capture completed batch 1 before batch 0, and the lengths differ: the old
    # positional pop handed batch 0 batch 1's entry and raised a shape error.
    task = _task(kind, named, [(1, native1), (0, native0)])
    task.add_batch(x0, None, batch_index=0)
    task.add_batch(x1, None, batch_index=1)

    hessian, dxxT = _oracle(kind, [(x0[0], native0), (x1[0], native1)])
    assert _relative_error(task.H, hessian) <= 1e-6
    assert task.dXXT is not None
    assert _relative_error(task.dXXT, dxxT) <= 1e-6
    assert task.nsamples == 5


@pytest.mark.parametrize("kind", KINDS)
def test_missing_batch_index_raises_instead_of_silently_mispairing(kind):
    """A missing native entry is a fail-closed contract, not a silent fallback."""

    torch.manual_seed(2)
    named = _named(nn.Linear(8, 4, bias=False))
    named.state[NATIVE_INPUTS_STATE_KEY] = [(0, torch.randn(3, 8))]
    task = _CLASSES[kind](named, _qcfg(kind))
    task.quantizer.configure(perchannel=True)

    task.add_batch(torch.randn(1, 3, 8), None, batch_index=0)
    with pytest.raises(RuntimeError, match="batch `1`"):
        task.add_batch(torch.randn(1, 3, 8), None, batch_index=1)


def _require_two_cuda_devices():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs 2 visible CUDA devices")


@pytest.mark.parametrize("kind", KINDS)
def test_two_device_partials_reduce_to_float64_oracle(kind):
    """Two devices must reduce to the single-device Hessian over the same rows."""

    _require_two_cuda_devices()
    dev0, dev1 = torch.device("cuda:0"), torch.device("cuda:1")
    torch.manual_seed(3)
    named = _named(nn.Linear(16, 8, bias=False).to(dev0))
    native0 = torch.randn(3, 16)
    native1 = torch.randn(5, 16)
    task = _task(kind, named, [native0, native1])
    x0 = torch.randn(1, 3, 16, device=dev0)
    x1 = torch.randn(1, 5, 16, device=dev1)

    task.add_batch(x0, None)
    task.add_batch(x1, None)

    # One partial per input device before the reduction.
    partials = getattr(task, f"_{kind}_device_hessian")
    assert {torch.device(device) for device in partials} == {dev0, dev1}

    _finalize(task, kind)
    hessian, dxxT = _oracle(kind, [(x0[0].cpu(), native0), (x1[0].cpu(), native1)])
    assert _relative_error(task.H, hessian) <= 1e-6
    assert task.dXXT is not None
    assert _relative_error(task.dXXT, dxxT) <= 1e-6
    assert getattr(task, f"_{kind}_device_hessian") == {}


@pytest.mark.parametrize("kind", KINDS)
def test_two_device_quantize_reduces_partials_inside_quantize(kind):
    """``quantize()`` must reduce the partials itself and emit finite codes."""

    _require_two_cuda_devices()
    dev0, dev1 = torch.device("cuda:0"), torch.device("cuda:1")
    torch.manual_seed(4)
    named = _named(nn.Linear(16, 8, bias=False).to(dev0))
    task = _task(kind, named, [torch.randn(3, 16), torch.randn(5, 16)])

    task.add_batch(torch.randn(1, 3, 16, device=dev0), None)
    task.add_batch(torch.randn(1, 5, 16, device=dev1), None)

    quantized = task.quantize()[0]

    assert torch.isfinite(quantized).all()
    assert tuple(quantized.shape) == (8, 16)
