# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""#4735: the FP32 Marlin bias stage must fit the budget and stay out of the reduction.

#4657 widened the bias epilogue stage to FP32 in ``marlin_template.h``
(``thread_n_blocks * 16 / 4`` int4 instead of ``/ 8``) but left the host-side
shared-memory model in ``get_kernel_cache_size`` at the old 16-bit size
(``tb_n * 2`` bytes). It also kept the bias region at the *start* of the
B/reduce union, so the wider stage shares memory with the split-K reduction.
A thread config the host accepts can therefore need more dynamic shared memory
than the launch reserves, and the bias copy overlaps reduction scratch.

The reported symptom is nondeterministic, wrong output tiles for the AUTO
kernel: GPTQ ``MarlinLinear`` bf16 on sm_120 and AWQ ``AwqMarlinLinear`` fp16
on sm_89. The bad cells are whole 16x64/64x64 tiles off by 2-5 against an
independent FP64 oracle, while a correct call is within ~0.1 (bf16).

This file pins the two halves of the contract:

* a source check that the host byte count, the kernel int4 count, and the
  dedicated bias offset agree (this is what makes the budget model sound), and
* a real-kernel check: the same input through the same launch repeated 30
  times must be bitwise identical, and within an FP64 oracle bound. The
  determinism half is the issue's own repro and fails on the affected
  architectures even when the source contract is fixed.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import torch

from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
from gptqmodel.nn_modules.qlinear.marlin_awq import AwqMarlinLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.utils.marlin import marlin_import_exception, marlin_runtime_available, marlin_runtime_error


ROOT = Path(__file__).resolve().parents[2]

# One launch per forward; the reporter measured the bad tiles within 30 calls.
REPEATS = 30


def _source(relative: str) -> str:
    return (ROOT / relative).read_text()


def _assigned_expression(source: str, pattern: str) -> str:
    match = re.search(pattern, source, flags=re.DOTALL)
    assert match is not None, f"pattern {pattern!r} not found"
    return " ".join(match.group(1).split())


def test_host_bias_stage_matches_the_fp32_kernel_layout():
    """The host budget must use the FP32 stage size the kernel actually uses."""

    host = _source("gptqmodel_ext/marlin/gptq_marlin.cu")
    kernel = _source("gptqmodel_ext/marlin/marlin_template.h")

    # ``sh_bias_size`` in the kernel is int4; the host model is bytes.
    kernel_bias = _assigned_expression(
        kernel, r"constexpr int sh_bias_size = \(([^;]*)\);"
    )
    assert "thread_n_blocks * 16 / 4" in kernel_bias, kernel_bias

    host_bias = _assigned_expression(host, r"int sh_bias_size = ([^;]*);")
    assert "tb_n * 4" in host_bias, host_bias

    # The bias stage must not overlap the B/reduce union: a split-K reduction
    # writes that union after the bias copy is issued.
    bias_offset = _assigned_expression(
        kernel.replace(" ", ""), r"int4\*sh_bias=sh\+([^;]*);"
    )
    assert "sh_size_b_red_max" in bias_offset, bias_offset

    # ``tmp_size`` must reserve the union *plus* the dedicated bias stage.
    tmp_size = _assigned_expression(
        host, r"int tmp_size = (\([^;]*\))\s*\+\s*sh_bias_size;"
    )
    assert "sh_b_size > sh_red_size" in tmp_size, tmp_size


def test_host_g_idx_stage_is_accounted_in_bytes():
    """``g_idx_stage`` is ``tb_k / 4`` int4, i.e. ``tb_k * 4`` bytes per stage."""

    host = _source("gptqmodel_ext/marlin/gptq_marlin.cu")
    g_idx = _assigned_expression(host, r"int sh_g_idx_size = ([^;]*);")
    assert "stages * tb_k * 4" in g_idx, g_idx


def _skip_reason(dtype: torch.dtype) -> str | None:
    if not torch.cuda.is_available():
        return "CUDA required for the Marlin determinism test"
    if marlin_import_exception is not None:
        return f"Marlin kernel unavailable: {marlin_import_exception}"
    if not marlin_runtime_available(dtype):
        return marlin_runtime_error(dtype)
    return None


def _build_gptq(k: int, n: int, dtype: torch.dtype):
    """Build a v1 GPTQ ``MarlinLinear`` the way the loader does."""

    torch.manual_seed(4735)
    codes = torch.randint(0, 16, (k, n))
    zeros = torch.full((k // 128, n), 8, dtype=torch.int32)
    scales = (torch.randn(k // 128, n) * 0.01 + 0.02).abs().clamp_min(0.01).to(dtype)
    indices = torch.arange(k, dtype=torch.int32) // 128
    weight = (codes.float() - zeros[indices].float()) * scales.float()[indices]

    linear = torch.nn.Linear(k, n, bias=False)
    linear.weight.data.copy_(weight.t().to(dtype))

    kwargs = {
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "desc_act": False,
        "in_features": k,
        "out_features": n,
        "bias": False,
        "register_buffers": True,
        "dtype": dtype,
    }
    source = TorchLinear(**kwargs, format=FORMAT.GPTQ_V2)
    source.pack_block(linear, scales.t(), zeros.t().float(), indices)

    module = MarlinLinear(**kwargs).cuda()
    for name in ("qweight", "qzeros", "scales", "g_idx"):
        src, dst = getattr(source, name, None), getattr(module, name, None)
        if isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor) and src.shape == dst.shape:
            dst.data.copy_(src.cuda())
    module.post_init()
    module.eval()
    return module, weight.cuda()


def _build_awq(k: int, n: int, dtype: torch.dtype):
    """Build a mock AWQ 4-bit g128 ``AwqMarlinLinear`` (no zero-point float)."""

    from gptqmodel.quantization.awq.utils.packing_utils import pack_awq

    torch.manual_seed(4735)
    codes = torch.randint(0, 16, (k, n))
    zeros = torch.full((k // 128, n), 8, dtype=torch.int32)
    scales = (torch.randn(k // 128, n) * 0.01 + 0.02).abs().clamp_min(0.01).to(dtype)
    indices = torch.arange(k, dtype=torch.int32) // 128
    weight = (codes.float() - zeros[indices].float()) * scales.float()[indices]

    qweight, qzeros = pack_awq(codes, zeros, 4)
    kwargs = {
        "bits": 4,
        "group_size": 128,
        "sym": False,
        "desc_act": False,
        "in_features": k,
        "out_features": n,
        "bias": False,
        "register_buffers": True,
        "dtype": dtype,
    }
    module = AwqMarlinLinear(**kwargs, backend="marlin").cuda()
    with torch.no_grad():
        module.qweight.data.copy_(qweight.cuda())
        module.qzeros.data.copy_(qzeros.cuda())
        module.scales.data.copy_(scales.cuda())
    module.post_init()
    module.eval()
    return module, weight.cuda()


def _assert_deterministic_and_correct(module, weight, *, size_m, size_k, dtype, atol):
    x = torch.randn(size_m, size_k, device="cuda", dtype=dtype)
    with torch.inference_mode():
        first = module(x).clone()
        digests = {hashlib.sha1(first.float().cpu().numpy().tobytes()).hexdigest()}
        worst = torch.zeros((), device="cuda")
        for _ in range(REPEATS - 1):
            out = module(x)
            digests.add(hashlib.sha1(out.float().cpu().numpy().tobytes()).hexdigest())
            worst = torch.maximum(worst, (out - first).abs().amax())
    torch.cuda.synchronize()

    assert len(digests) == 1, (
        f"Marlin is not deterministic for M={size_m} K={size_k} {dtype}: "
        f"{len(digests)} distinct outputs in {REPEATS} calls, max|delta|={worst.item():.6f}"
    )

    reference = x.double() @ weight.double()
    error = (first.double() - reference).abs().max().item()
    assert error < atol, (
        f"Marlin disagrees with the FP64 oracle for M={size_m} K={size_k} {dtype}: "
        f"max|err|={error:.4f} (bound {atol})"
    )


# The issue's GPTQ table: K x N at M in {64, 256, 1000}, bf16 on sm_120.
_GPTQ_SHAPES = ((2048, 8192), (4096, 4096))
_GPTQ_M = (64, 256, 1000)

# The issue's AWQ table: fp16 M>=512, N>=2048 on sm_89.
_AWQ_SHAPES = ((2048, 2048), (8192, 2048))
_AWQ_M = (512, 2048)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("size_m", _GPTQ_M)
@pytest.mark.parametrize("size_k,size_n", _GPTQ_SHAPES)
def test_gptq_marlin_bf16_repeated_launches_are_bitwise_identical(size_k, size_n, size_m):
    reason = _skip_reason(torch.bfloat16)
    if reason is not None:
        pytest.skip(reason)

    module, weight = _build_gptq(size_k, size_n, torch.bfloat16)
    _assert_deterministic_and_correct(
        module, weight, size_m=size_m, size_k=size_k, dtype=torch.bfloat16, atol=0.5
    )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("size_m", _AWQ_M)
@pytest.mark.parametrize("size_k,size_n", _AWQ_SHAPES)
def test_awq_marlin_fp16_repeated_launches_are_bitwise_identical(size_k, size_n, size_m):
    reason = _skip_reason(torch.float16)
    if reason is not None:
        pytest.skip(reason)

    module, weight = _build_awq(size_k, size_n, torch.float16)
    _assert_deterministic_and_correct(
        module, weight, size_m=size_m, size_k=size_k, dtype=torch.float16, atol=0.05
    )
