# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native HIP INT8-activation GEMV for EXL3 decode-sized batches on RDNA4 (issue #5710).

The int8 kernel keeps the packed mul1 trellis and never decodes it: the cb 2
codebook value is affine in the byte sum of ``state * 0x83DCD12D``, so a signed
``v_dot4_i32_i8`` against an int8-quantized activation evaluates one weight per
instruction. On gfx12xx that instruction runs at the same rate as the fp16
``v_dot2_f32_f16`` while replacing the two-instruction codebook decode, which is
what makes the int8 path the faster of the two.

These tests pin the kernel's parity against the reconstruct path, its exactness
against a torch emulation of the same activation quantization, the decline
conditions that keep the fp16 HIP GEMV (and reconstruct) in charge, and the
``LinearEXL3`` dispatch boundary. The kernel is opt-in through
``KernelConfig.exl3_int8_gemv`` (0 off, 1 residual, 2 plain), exactly like the
CUDA path, so every dispatch test states the mode it exercises.
"""

from __future__ import annotations

import pytest
import torch

from gptqmodel.exllamav3.ext import exllamav3_ext, prewarm_exllamav3_extension
from gptqmodel.exllamav3.modules.quant.exl3 import LinearEXL3
from gptqmodel.utils.kernel import KernelConfig


def _hip_gemv_skip_reason() -> str | None:
    if torch.version.hip is None:
        return "native EXL3 HIP GEMV requires a ROCm (HIP) build"
    if not torch.cuda.is_available():
        return "native EXL3 HIP GEMV requires a ROCm device"
    name = torch.cuda.get_device_properties(0).gcnArchName
    if not name.startswith("gfx12"):
        return f"native EXL3 INT8 HIP GEMV requires RDNA4 (gfx12xx); found {name}"
    return None


_SKIP_REASON = _hip_gemv_skip_reason()

pytestmark = pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "")

# The int8 path is mul1 (cb 2) only; the integer rates it instantiates are 2, 3
# and 4 bpw. Residual mode is gated to m == 1, plain int8 to m <= 2.
_MODES = pytest.param(1, 1, id="residual-m1"), pytest.param(2, 1, id="plain-m1"), pytest.param(2, 2, id="plain-m2")


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


def _linear(weights, rate, *, mcg=False, mul1=False, q4nx=False, bias=None, kernel=None):
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
        kernel=kernel,
    )


def _input(rows, k, seed):
    return (torch.randn((rows, k), generator=torch.Generator().manual_seed(seed)) * 0.1).half().cuda()


def _run(op, module, bias, x, result, *, q4nx=False, int8_gemv=0):
    return op(module.trellis, module.suh, module.svh, float(module.K), bias,
              module.mcg, module.mul1, x, result, q4nx, int8_gemv)


def _int8_quantized_reference(module, x, mode):
    """Torch emulation of the kernel's activation quantization.

    Same per-row scale (``max|A_had| / 127``), same round-to-nearest int8 values,
    same residual scale (``q / 254``), and the decoded fp16 weights, accumulated
    in float64 so only the kernel's own fp32 epilogue can differ.
    """
    rows, k = x.shape
    n = module.svh.numel()
    xh = torch.empty_like(x)
    exllamav3_ext.had_r_128(x, xh, module.suh, None, 1.0)
    a = xh.float()
    q = a.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
    v = torch.round(a * (1.0 / q)).clamp_(-127, 127)
    dequant = v * q
    if mode == 1:
        q2 = q / 254.0
        v2 = torch.round((a - dequant) * (254.0 / q)).clamp_(-127, 127)
        dequant = dequant + v2 * q2
    w = module.get_inner_weight_tensor().double()
    y = (dequant.double() @ w).float().view(rows, n)
    exllamav3_ext.had_r_128(y, y, None, module.svh, 1.0)
    return y


@pytest.mark.parametrize(("mode", "rows"), _MODES)
@pytest.mark.parametrize("rate", (2.0, 3.0, 4.0))
@torch.inference_mode()
def test_hip_gemv_int8_matches_reconstruct(hip_gemv, rate, mode, rows):
    k, n = 512, 256
    weights = _packed_projection(k, n, rate)
    bias = torch.linspace(-0.02, 0.03, n, device="cuda").half()
    module = _linear(weights, rate, bias=bias, mul1=True)
    x = _input(rows, k, seed=rows)
    expected = module.forward(x, {"reconstruct": True})

    result = torch.empty((rows, n), dtype=torch.float16, device=x.device)
    assert _run(hip_gemv, module, bias, x, result, int8_gemv=mode) == 2
    # Plain int8 activations are lossy by construction (one byte per element);
    # the residual pass restores ~15-16 effective bits.
    if mode == 1:
        torch.testing.assert_close(result, expected, rtol=3e-3, atol=3e-3)
    else:
        torch.testing.assert_close(result, expected, rtol=2e-2, atol=3e-3)


@pytest.mark.parametrize(("mode", "rows"), _MODES)
@pytest.mark.parametrize("rate", (2.0, 3.0, 4.0))
@torch.inference_mode()
def test_hip_gemv_int8_matches_quantized_reference(hip_gemv, rate, mode, rows):
    # The kernel is exact for the activations it quantizes: it must reproduce a
    # float64 accumulation of the same int8 values to fp32 epilogue rounding.
    k, n = 512, 256
    weights = _packed_projection(k, n, rate, seed=41)
    module = _linear(weights, rate, mul1=True)
    x = _input(rows, k, seed=mode * 10 + rows)
    expected = _int8_quantized_reference(module, x, mode)

    result = torch.empty((rows, n), dtype=torch.float32, device=x.device)
    assert _run(hip_gemv, module, None, x, result, int8_gemv=mode) == 2
    torch.testing.assert_close(result, expected, rtol=2e-3, atol=2e-4)


@pytest.mark.parametrize(("mode", "rows"), _MODES)
@torch.inference_mode()
def test_hip_gemv_int8_wide_output_matches_reconstruct(hip_gemv, mode, rows):
    # n >= 8192 selects the 64-column, 4-way k-split launch geometry.
    k, n = 512, 8192
    weights = _packed_projection(k, n, 4.0, seed=43)
    module = _linear(weights, 4.0, mul1=True)
    x = _input(rows, k, seed=7)
    expected = module.forward(x, {"reconstruct": True})
    result = torch.empty((rows, n), dtype=torch.float16, device=x.device)
    assert _run(hip_gemv, module, None, x, result, int8_gemv=mode) == 2
    torch.testing.assert_close(result, expected, rtol=3e-3 if mode == 1 else 2e-2, atol=3e-3)


@torch.inference_mode()
def test_hip_gemv_int8_declines_rows_over_mode_limit(hip_gemv):
    k, n, rate = 512, 256, 4.0
    weights = _packed_projection(k, n, rate)
    module = _linear(weights, rate, mul1=True)

    # Residual mode is m == 1: two rows must fall through to the fp16 kernel,
    # which returns 1 and matches reconstruct.
    x = _input(2, k, seed=11)
    result = torch.empty((2, n), dtype=torch.float16, device=x.device)
    assert _run(hip_gemv, module, None, x, result, int8_gemv=1) == 1
    torch.testing.assert_close(result, module.forward(x, {"reconstruct": True}), rtol=2e-3, atol=2e-3)

    # Plain int8 is m <= 2; three rows decline to the fp16 kernel.
    x = _input(3, k, seed=12)
    result = torch.empty((3, n), dtype=torch.float16, device=x.device)
    assert _run(hip_gemv, module, None, x, result, int8_gemv=2) == 1

    # Over EXL3_GEMV_MAX_M both kernels decline and reconstruct stays in charge.
    x = _input(9, k, seed=13)
    result = torch.empty((9, n), dtype=torch.float16, device=x.device)
    assert _run(hip_gemv, module, None, x, result, int8_gemv=2) == 0


@torch.inference_mode()
@pytest.mark.parametrize(("mcg", "mul1", "rate", "k", "n", "expected"), (
    (True, False, 4.0, 512, 256, 1),    # cb 1 has no int8 affine form; fp16 takes it
    (False, False, 4.0, 512, 256, 1),   # cb 0 likewise
    (False, True, 1.5, 512, 256, 1),    # half-integer rates have no int8 kernel
    (False, True, 2.5, 512, 256, 1),
    (False, True, 4.0, 576, 256, 0),    # k not a multiple of 128: both decline
    (False, True, 4.0, 512, 144, 0),    # n not a multiple of 128: both decline
    (False, True, 5.0, 512, 256, 0),    # no 5 bpw instantiation
))
def test_hip_gemv_int8_declines_ineligible_rates_and_shapes(hip_gemv, mcg, mul1, rate, k, n, expected):
    # Every cell must stay off the int8 kernel (never 2) and land on the fp16
    # HIP GEMV (1) or reconstruct (0) exactly as the fp16-only dispatch does.
    weights = _packed_projection(k, n, rate)
    module = _linear(weights, rate, mcg=mcg, mul1=mul1)
    x = _input(1, k, seed=3)
    result = torch.empty((1, n), dtype=torch.float16, device=x.device)
    assert _run(hip_gemv, module, None, x, result, int8_gemv=2) == expected


@torch.inference_mode()
def test_hip_gemv_int8_off_keeps_the_fp16_kernel(hip_gemv):
    # int8_gemv == 0 is the default and must not change the fp16 dispatch.
    k, n, rate = 512, 256, 4.0
    weights = _packed_projection(k, n, rate)
    module = _linear(weights, rate, mul1=True)
    x = _input(1, k, seed=17)
    result = torch.empty((1, n), dtype=torch.float16, device=x.device)
    assert _run(hip_gemv, module, None, x, result, int8_gemv=0) == 1


@torch.inference_mode()
def test_linear_forward_dispatches_int8_gemv(monkeypatch):
    k, n, rate = 512, 256, 3.0
    weights = _packed_projection(k, n, rate, seed=19)
    module = _linear(weights, rate, mul1=True, kernel=KernelConfig(exl3_int8_gemv=1))
    calls = []
    results = []
    original = exllamav3_ext.bc_linear_exl3_run_hip_gemv

    def spy(*args, **kwargs):
        calls.append(args)
        value = original(*args, **kwargs)
        results.append(value)
        return value

    monkeypatch.setattr(exllamav3_ext, "bc_linear_exl3_run_hip_gemv", spy)

    x = _input(1, k, seed=23)
    actual = module.forward(x, {"reconstruct": False})
    expected = module.forward(x, {"reconstruct": True})
    assert results == [2]
    assert calls[0][-1] == 1  # the mode reaches the kernel entry point
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)

    # Residual mode is m == 1 only: a two-row batch keeps the fp16 HIP GEMV.
    calls.clear()
    results.clear()
    x = _input(2, k, seed=29)
    module.forward(x, {"reconstruct": False})
    assert results == [1]


@torch.inference_mode()
def test_linear_forward_plain_int8_matches_reconstruct():
    k, n, rate = 512, 256, 4.0
    weights = _packed_projection(k, n, rate, seed=31)
    module = _linear(weights, rate, mul1=True, kernel=KernelConfig(exl3_int8_gemv=2))
    x = _input(2, k, seed=37)
    actual = module.forward(x, {"reconstruct": False})
    expected = module.forward(x, {"reconstruct": True})
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=3e-3)
