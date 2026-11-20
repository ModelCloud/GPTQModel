# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""QQQ mock quantization must skip calibrated error propagation (#5839).

``mock_quantization`` promises "skip heavy computations for fast model loading
validation", but the QQQ block path still materialized the Hessian, inverted it,
and ran the eager error-propagation loop. The result was bitwise equal to a
calibrated run and differed from the data-independent RTN grid. Mock mode now
routes to that grid before any Hessian work.
"""

from unittest.mock import patch

import torch
from torch import nn

from gptqmodel.quantization.config import Fallback, QQQConfig
from gptqmodel.quantization.qqq import QQQ


IN_FEATURES = 64
OUT_FEATURES = 32
BITS = 4


def _build(seed: int, mock: bool, fallback=None) -> QQQ:
    torch.manual_seed(seed)
    layer = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False, dtype=torch.float16)
    qcfg = QQQConfig(
        bits=BITS,
        group_size=-1,
        sym=True,
        fallback=fallback,
        mock_quantization=mock,
        device="cpu",
        offload_to_disk=False,
    )
    task = QQQ(layer, qcfg=qcfg)
    task.quantizer.configure(bits=BITS, perchannel=True, groupsize=-1, sym=True)
    return task


def _run(seed: int, calibration_seed: int, *, mock: bool, fallback=None):
    task = _build(seed, mock, fallback)
    generator = torch.Generator().manual_seed(calibration_seed)
    activations = torch.randn(4, 16, IN_FEATURES, generator=generator, dtype=torch.float32)
    task.add_batch(activations, None)
    weight, scale, zero, g_idx, _, _, _, _, samples = task.quantize()
    return task, weight, scale, zero, g_idx, samples


def test_mock_matches_the_data_independent_rtn_grid():
    # Fallback(threshold=...) above the observed sample count selects RTN
    # without consulting the Hessian.
    _, rtn_weight, rtn_scale, _, _, _ = _run(
        11, 21, mock=False, fallback=Fallback(threshold=10_000)
    )
    _, mock_weight, mock_scale, _, _, _ = _run(11, 21, mock=True)

    torch.testing.assert_close(mock_weight, rtn_weight, rtol=0, atol=0)
    torch.testing.assert_close(mock_scale, rtn_scale, rtol=0, atol=0)


def test_mock_ignores_the_calibration_data():
    _, first_weight, first_scale, _, _, _ = _run(11, 21, mock=True)
    _, second_weight, second_scale, _, _, _ = _run(11, 987, mock=True)

    torch.testing.assert_close(second_weight, first_weight, rtol=0, atol=0)
    torch.testing.assert_close(second_scale, first_scale, rtol=0, atol=0)


def test_calibrated_qqq_still_uses_the_hessian():
    # Control: the non-mock path must stay calibration-dependent, otherwise the
    # mock assertions above would pass vacuously.
    _, first_weight, _, _, _, _ = _run(11, 21, mock=False)
    _, second_weight, _, _, _, _ = _run(11, 987, mock=False)

    assert not torch.equal(first_weight, second_weight)


def test_mock_never_inverts_the_hessian():
    task = _build(11, mock=True)
    task.add_batch(torch.randn(4, 16, IN_FEATURES), None)

    with patch.object(torch.linalg, "cholesky", wraps=torch.linalg.cholesky) as cholesky, \
            patch.object(torch.linalg, "cholesky_ex", wraps=torch.linalg.cholesky_ex) as cholesky_ex:
        task.quantize()

    assert cholesky.call_count == 0
    assert cholesky_ex.call_count == 0


def test_mock_keeps_the_configured_layout():
    task, weight, scale, zero, g_idx, samples = _run(11, 21, mock=True)

    assert weight.shape == (OUT_FEATURES, IN_FEATURES)
    assert weight.dtype == torch.float16
    assert scale.shape == (OUT_FEATURES, 1)
    assert zero.shape == (OUT_FEATURES, 1)
    assert g_idx.shape == (IN_FEATURES,)
    assert samples == 64
    # The signed INT4 grid must stay inside its representable range.
    codes = (weight.float() / scale.float()).round()
    assert codes.abs().max().item() <= (2 ** (BITS - 1)) - 1


def test_nan_loss_retry_skips_the_hessian():
    # The NaN-loss recovery sets ``mock_quantization`` and re-enters
    # ``quantize()``. That retry must land on the same data-independent grid
    # instead of inverting the Hessian it just failed on.
    task = _build(11, mock=False, fallback=Fallback(threshold=10_000))
    task.add_batch(torch.randn(4, 16, IN_FEATURES), None)
    task.qcfg.mock_quantization = True

    with patch.object(torch.linalg, "cholesky", wraps=torch.linalg.cholesky) as cholesky:
        weight, scale, _, _, _, _, _, _, _ = task.quantize()

    assert cholesky.call_count == 0
    _, rtn_weight, rtn_scale, _, _, _ = _run(
        11, 21, mock=False, fallback=Fallback(threshold=10_000)
    )
    torch.testing.assert_close(weight, rtn_weight, rtol=0, atol=0)
    torch.testing.assert_close(scale, rtn_scale, rtol=0, atol=0)


def test_mock_skips_hessian_accumulation():
    # The capture must not allocate the ``columns**2`` partial just to throw it
    # away at quantization time; only the sample count is kept for reporting.
    task = _build(11, mock=True)
    task.add_batch(torch.randn(4, 16, IN_FEATURES), None)

    assert task.nsamples == 64
    assert task._device_hessian_partials == {}
    assert task._device_sample_counts == {}

    *_, samples = task.quantize()
    assert samples == 64
