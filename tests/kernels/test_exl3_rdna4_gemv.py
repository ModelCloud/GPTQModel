# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native HIP GEMV for EXL3 decode-sized batches on RDNA4 (issue #5710).

On ROCm ``LinearEXL3.forward`` used to reconstruct the whole packed weight to
fp16 and call hipBLAS for every batch. The native HIP GEMV consumes the packed
trellis directly for batches of at most ``EXL3_GEMV_MAX_M`` (8) rows, so these
tests pin its parity against the reconstruct path, the decline conditions that
keep the fallback in charge, and the ``LinearEXL3`` dispatch boundary.

The reference is the reconstruct path (``{"reconstruct": True}``) because the
NVIDIA-only tensor-core GEMV (``bc.run``) is not available on a HIP build.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.exllamav3.ext import exllamav3_ext, prewarm_exllamav3_extension
from gptqmodel.exllamav3.modules.quant.exl3 import LinearEXL3


def _hip_gemv_skip_reason() -> str | None:
    if torch.version.hip is None:
        return "native EXL3 HIP GEMV requires a ROCm (HIP) build"
    if not torch.cuda.is_available():
        return "native EXL3 HIP GEMV requires a ROCm device"
    name = torch.cuda.get_device_properties(0).gcnArchName
    if not name.startswith("gfx12"):
        return f"native EXL3 HIP GEMV requires RDNA4 (gfx12xx); found {name}"
    return None


_SKIP_REASON = _hip_gemv_skip_reason()

pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")

# (rate, codebook marker). cb 0 only exists at 4 bpw; the half-integer rates
# use mul1, and Q4NX has its own three rates.
_CASES = (
    pytest.param(4.0, {}, id="4bpw-cb0"),
    pytest.param(4.0, {"mcg": True}, id="4bpw-cb1"),
    pytest.param(4.0, {"mul1": True}, id="4bpw-cb2"),
    pytest.param(3.0, {"mcg": True}, id="3bpw-cb1"),
    pytest.param(3.0, {"mul1": True}, id="3bpw-cb2"),
    pytest.param(2.0, {"mcg": True}, id="2bpw-cb1"),
    pytest.param(2.0, {"mul1": True}, id="2bpw-cb2"),
    pytest.param(1.5, {"mul1": True}, id="1.5bpw-mul1"),
    pytest.param(2.5, {"mul1": True}, id="2.5bpw-mul1"),
    pytest.param(3.5, {"mul1": True}, id="3.5bpw-mul1"),
    pytest.param(2.5, {"q4nx": True}, id="q4nx-2.5"),
    pytest.param(3.0, {"q4nx": True}, id="q4nx-3.0"),
    pytest.param(3.5, {"q4nx": True}, id="q4nx-3.5"),
)


@pytest.fixture(scope="module")
def hip_gemv():
    assert prewarm_exllamav3_extension()
    return torch.ops.gptqmodel_exllamav3.bc_linear_exl3_run_hip_gemv


def _packed_projection(k, n, rate, seed=37):
    words = int(16 * rate)
    generator = torch.Generator().manual_seed(seed)
    trellis = torch.randint(-32768, 32768, (k // 16, n // 16, words),
                            generator=generator, dtype=torch.int16).cuda()
    suh = ((2 * torch.randint(0, 2, (k,), generator=generator) - 1) * 0.5).half().cuda()
    svh = (torch.randn(n, generator=generator) * 0.03).half().cuda()
    return trellis, suh, svh


def _marker(device):
    return torch.ones((), dtype=torch.int32, device=device)


def _linear(weights, rate, *, mcg=False, mul1=False, q4nx=False, bias=None):
    trellis, suh, svh = weights
    return LinearEXL3(
        None,
        suh.numel(),
        svh.numel(),
        suh=suh,
        svh=svh,
        trellis=trellis,
        mcg=_marker(trellis.device) if mcg else None,
        mul1=_marker(trellis.device) if mul1 else None,
        bias=bias,
        q4nx=_marker(trellis.device) if q4nx else None,
    )


def _input(rows, k, seed):
    return (torch.randn((rows, k), generator=torch.Generator().manual_seed(seed)) * 0.1).half().cuda()


def _op_args(module, bias):
    return (module.trellis, module.suh, module.svh, float(module.K), bias,
            module.mcg, module.mul1)


@pytest.mark.parametrize("rows", (1, 2, 4, 8))
@pytest.mark.parametrize(("rate", "codebook"), _CASES)
@torch.inference_mode()
def test_hip_gemv_matches_reconstruct(hip_gemv, rate, codebook, rows):
    k, n = 512, 256
    weights = _packed_projection(k, n, rate)
    bias = torch.linspace(-0.02, 0.03, n, device="cuda").half()
    module = _linear(weights, rate, bias=bias, **codebook)
    x = _input(rows, k, seed=rows)
    expected = module.forward(x, {"reconstruct": True})

    result = torch.empty((rows, n), dtype=torch.float16, device=x.device)
    assert hip_gemv(*_op_args(module, bias), x, result, module.q4nx) == 1
    torch.testing.assert_close(result, expected, rtol=2e-3, atol=2e-3)

    # The public forward must take the same kernel and return the same values.
    dispatched = module.forward(x, {"reconstruct": False})
    torch.testing.assert_close(dispatched, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("rows", (1, 4))
@torch.inference_mode()
def test_hip_gemv_wide_output_matches_reconstruct(hip_gemv, rows):
    # n >= 8192 selects the 64-column, 4-way k-split launch geometry.
    k, n = 512, 8192
    weights = _packed_projection(k, n, 4.0)
    module = _linear(weights, 4.0)
    x = _input(rows, k, seed=7)
    expected = module.forward(x, {"reconstruct": True})
    result = torch.empty((rows, n), dtype=torch.float16, device=x.device)
    assert hip_gemv(*_op_args(module, None), x, result, False) == 1
    torch.testing.assert_close(result, expected, rtol=2e-3, atol=2e-3)


@torch.inference_mode()
def test_hip_gemv_applies_bias_and_honours_include_bias(hip_gemv):
    k, n, rate = 512, 256, 3.0
    weights = _packed_projection(k, n, rate)
    bias = torch.linspace(-0.02, 0.03, n, device="cuda")
    module = _linear(weights, rate, mcg=True, bias=bias)
    x = _input(2, k, seed=5)

    with_bias = torch.empty((2, n), dtype=torch.float32, device=x.device)
    without = torch.empty_like(with_bias)
    assert hip_gemv(*_op_args(module, bias), x, with_bias, False) == 1
    assert hip_gemv(*_op_args(module, None), x, without, False) == 1
    torch.testing.assert_close(with_bias - without, bias.expand_as(with_bias), rtol=1e-6, atol=1e-7)

    expected = module.forward(x, {"reconstruct": True}, out_dtype=torch.float32)
    torch.testing.assert_close(with_bias, expected, rtol=2e-3, atol=2e-3)

    expected_no_bias = module.forward(x, {"reconstruct": True, "include_bias": False}, out_dtype=torch.float32)
    actual_no_bias = module.forward(x, {"reconstruct": False, "include_bias": False}, out_dtype=torch.float32)
    torch.testing.assert_close(actual_no_bias, expected_no_bias, rtol=2e-3, atol=2e-3)


@torch.inference_mode()
def test_hip_gemv_declines_batches_over_max_m(hip_gemv):
    k, n, rate = 512, 256, 3.0
    weights = _packed_projection(k, n, rate)
    module = _linear(weights, rate, mcg=True)
    x = _input(9, k, seed=11)

    result = torch.empty((9, n), dtype=torch.float16, device=x.device)
    assert hip_gemv(*_op_args(module, None), x, result, False) == 0

    # Rows over EXL3_GEMV_MAX_M keep the reconstruct path, bit for bit.
    actual = module.forward(x, {"reconstruct": False})
    expected = module.forward(x, {"reconstruct": True})
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.inference_mode()
@pytest.mark.parametrize(("k", "n", "rate"), (
    (576, 256, 4.0),   # k not a multiple of 128
    (512, 144, 4.0),   # n not a multiple of 128
    (512, 256, 5.0),   # no 5 bpw kernel instantiation
))
def test_hip_gemv_declines_ineligible_shapes_and_rates(hip_gemv, k, n, rate):
    weights = _packed_projection(k, n, rate)
    module = _linear(weights, rate)
    x = _input(1, k, seed=3)
    result = torch.empty((1, n), dtype=torch.float16, device=x.device)
    assert hip_gemv(*_op_args(module, None), x, result, False) == 0


@torch.inference_mode()
def test_hip_gemv_declines_q4nx_rate_outside_its_codebook(hip_gemv):
    # 32 words per 256 weights is a 2.0 bpw trellis; Q4NX only packs 2.5, 3.0
    # and 3.5 bpw, so the 8-step-bits cell must not have a kernel.
    k, n = 512, 256
    trellis, suh, svh = _packed_projection(k, n, 2.0)
    x = _input(1, k, seed=3)
    result = torch.empty((1, n), dtype=torch.float16, device=x.device)
    assert hip_gemv(trellis, suh, svh, 2.0, None, False, False, x, result, True) == 0


@torch.inference_mode()
def test_linear_forward_falls_back_for_unsupported_rate():
    # 5 bpw is a valid EXL3 rate but has no HIP GEMV instantiation, so the
    # public forward must keep using reconstruct + hgemm.
    k, n = 512, 256
    weights = _packed_projection(k, n, 5.0)
    module = _linear(weights, 5.0)
    x = _input(2, k, seed=13)
    actual = module.forward(x, {"reconstruct": False})
    expected = module.forward(x, {"reconstruct": True})
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_linear_forward_dispatches_hip_gemv_up_to_eight_rows(monkeypatch):
    k, n, rate = 512, 256, 3.0
    weights = _packed_projection(k, n, rate)
    module = _linear(weights, rate, mcg=True)
    calls = []
    results = []
    original = exllamav3_ext.bc_linear_exl3_run_hip_gemv

    def spy(*args, **kwargs):
        calls.append(args)
        value = original(*args, **kwargs)
        results.append(value)
        return value

    monkeypatch.setattr(exllamav3_ext, "bc_linear_exl3_run_hip_gemv", spy)

    for rows in (1, 8):
        calls.clear()
        results.clear()
        module.forward(_input(rows, k, seed=rows), {"reconstruct": False})
        assert len(calls) == 1
        assert results == [1]

    # The default forward of a larger batch never probes the kernel.
    calls.clear()
    module.forward(_input(9, k, seed=9), {})
    assert calls == []

    # An explicit reconstruct=False still probes it, but the kernel declines
    # over EXL3_GEMV_MAX_M and the layer keeps the reconstruct path.
    for rows in (9, 24):
        calls.clear()
        results.clear()
        actual = module.forward(_input(rows, k, seed=rows), {"reconstruct": False})
        assert all(value == 0 for value in results)
        expected = module.forward(_input(rows, k, seed=rows), {"reconstruct": True})
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
