# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""#4740: QQQ must accept a column-strided activation.

``QQQLinear.dynamic_quant`` keeps the GPU int8 quantizer for contiguous input
only and otherwise falls back to element-wise ops, which preserve the input's
strides.  A column-strided view (``x.t()`` of a ``[K, M]`` buffer, the layout
batched decoding feeds) therefore reached the native ``qqq_gemm`` as a
column-major ``quant_A`` that the kernel read row-major: plausible-looking but
uncorrelated output (relative error ~1.4 against the packed-layout call) with
no error raised.

The fix packs the fallback operand and makes the native entry point reject a
non-contiguous ``A``.  These tests pin both halves: the quantized operand is
densely packed, the wrapper matches the packed-layout call and an independent
FP64 dequantized oracle, and the kernel fails loudly instead of returning
garbage.
"""

import pytest
import torch

import gptqmodel.nn_modules.qlinear.qqq as qqq_mod
from gptqmodel.nn_modules.qlinear.qqq import (
    QQQLinear,
    _workspace_numel,
    mul,
)
from gptqmodel.utils.qqq import qqq_runtime_available
from gptqmodel.utils.rocm import IS_ROCM


GROUP_SIZE = 128

NATIVE_QQQ = (
    torch.cuda.is_available()
    and not IS_ROCM
    and qqq_runtime_available()
)

requires_native_qqq = pytest.mark.skipif(
    not NATIVE_QQQ, reason="QQQ's native kernel requires an NVIDIA CUDA device"
)


def _column_strided(rows: int, in_features: int) -> torch.Tensor:
    """Return the ``[rows, K]`` column-strided view batched decoding produces."""

    x = torch.randn(in_features, rows, dtype=torch.float16).t()
    assert x.shape == (rows, in_features)
    assert not x.is_contiguous()
    assert x.stride(-1) != 1
    return x


def _dequantized_weight(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Independent FP64 QQQ dequant: ``(code - 8) * group_scale``.

    The stored per-channel scale cancels against the packed group scale, so the
    observable weight depends only on the 4-bit codes and the group scales.
    Reimplemented here rather than read back from the module so that the
    expected answer never comes from the packer under test.
    """

    group_scale = (
        scales.double()
        .repeat_interleave(GROUP_SIZE, dim=1)
    )
    codes = (weight.double() / group_scale).round().add(8).clamp_(0, 15)
    return (codes - 8) * group_scale


def _build_native_module(in_features: int, out_features: int):
    """Return ``(module, dequantized_weight)`` for a real packed QQQ layer."""

    torch.manual_seed(4740 + in_features + out_features)
    linear = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float16)
    linear.weight.data.normal_(0, 0.02)
    scales = torch.rand(
        (out_features, in_features // GROUP_SIZE), dtype=torch.float16
    ) * 0.02 + 0.01

    module = QQQLinear(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        register_buffers=True,
    )
    module.pack(
        linear,
        scales,
        torch.rand(out_features, dtype=torch.float32) * 0.001 + 0.001,
    )
    module.post_init()

    expected = _dequantized_weight(linear.weight.detach().cpu(), scales).cuda()
    return module.eval().cuda(), expected


def test_dynamic_quant_packs_a_column_strided_activation(monkeypatch):
    """The element-wise fallback must return densely packed rows.

    Hardware independent: the Triton quantizer is disabled so the fallback is
    the path under test on CPU and on CUDA alike.
    """

    monkeypatch.setattr(qqq_mod, "_qqq_gpu_quantizer", lambda: None)

    module = QQQLinear.__new__(QQQLinear)
    torch.nn.Module.__init__(module)

    x_strided = _column_strided(rows=17, in_features=GROUP_SIZE)
    quant_strided, scale_strided = module.dynamic_quant(x_strided)
    quant_contiguous, scale_contiguous = module.dynamic_quant(x_strided.contiguous())

    # The native ``mul`` below dereferences A as packed rows, so the fallback
    # must not hand it the caller's column-major layout.
    assert quant_strided.is_contiguous(), quant_strided.stride()
    assert quant_strided.dtype == torch.int8
    torch.testing.assert_close(scale_strided, scale_contiguous, rtol=0, atol=0)
    assert torch.equal(quant_strided, quant_contiguous)


@requires_native_qqq
@pytest.mark.cuda
@pytest.mark.parametrize("rows", [2, 16, 17, 128])
def test_qqq_linear_matches_the_packed_layout_call_for_a_column_strided_activation(rows):
    """The reported symptom: relative error ~1.4 against the packed-layout call."""

    module, expected_weight = _build_native_module(in_features=896, out_features=4864)

    x_strided = _column_strided(rows=rows, in_features=module.in_features).cuda()
    assert not x_strided.is_contiguous()
    x_contiguous = x_strided.contiguous()

    with torch.inference_mode():
        strided = module(x_strided)
        contiguous = module(x_contiguous)
        repeated = module(x_strided)
    torch.cuda.synchronize()

    assert strided.shape == (rows, module.out_features)
    assert strided.dtype == torch.float16
    assert torch.isfinite(strided.float()).all()
    assert strided.abs().sum() > 0
    torch.testing.assert_close(strided, contiguous, rtol=0, atol=0)
    torch.testing.assert_close(repeated, strided, rtol=0, atol=0)

    relative = (strided.float() - contiguous.float()).norm() / contiguous.float().norm()
    assert relative.item() < 1e-6

    # Independent oracle: the only remaining error is the per-token int8
    # activation quantization, which is ~1.8% on this fixture. Garbage output
    # is uncorrelated with the oracle (relative error ~1.4).
    oracle = x_contiguous.double() @ expected_weight.t()
    oracle_error = (strided.double() - oracle).norm() / oracle.norm()
    assert oracle_error.item() < 5e-2, oracle_error.item()


@requires_native_qqq
@pytest.mark.cuda
def test_qqq_linear_keeps_row_strided_and_batched_inputs_working():
    """Control: layouts the element-wise path already packed contiguously."""

    module, _ = _build_native_module(in_features=896, out_features=4864)
    x = torch.randn(4, module.in_features + 32, device="cuda", dtype=torch.float16)
    row_strided = x[:, 32:]
    assert not row_strided.is_contiguous()

    with torch.inference_mode():
        strided = module(row_strided)
        contiguous = module(row_strided.contiguous())
        batched = module(
            torch.randn(2, 5, module.in_features, device="cuda", dtype=torch.float16)
        )
    torch.cuda.synchronize()

    torch.testing.assert_close(strided, contiguous, rtol=0, atol=0)
    assert batched.shape == (2, 5, module.out_features)


@requires_native_qqq
@pytest.mark.cuda
def test_native_qqq_gemm_rejects_a_non_contiguous_activation():
    """The kernel must not silently read a column-major operand row-major."""

    rows, in_features, out_features = 4, 256, 128
    quant_a = torch.randint(
        -128, 127, (rows, in_features), device="cuda", dtype=torch.int8
    )
    quant_a_bad = quant_a.t().contiguous().t()
    assert not quant_a_bad.is_contiguous()
    assert torch.equal(quant_a_bad, quant_a)

    packed_weight = torch.zeros(
        (in_features // 16, out_features // 8, 4), device="cuda", dtype=torch.int32
    )
    reduce_scratch = torch.zeros(
        (64 * 16, out_features), device="cuda", dtype=torch.int32
    )
    s1 = torch.ones((rows, 1), device="cuda", dtype=torch.float32)
    s2 = torch.ones((out_features,), device="cuda", dtype=torch.float32)
    s3 = torch.ones(
        (in_features // 128, out_features), device="cuda", dtype=torch.float16
    )
    workspace = torch.zeros(
        _workspace_numel(out_features, 16), device="cuda", dtype=torch.int32
    )

    def run(operand):
        out = torch.empty((rows, out_features), device="cuda", dtype=torch.float16)
        mul(operand, packed_weight, reduce_scratch, out, s1, s2, s3, workspace)
        return out

    with pytest.raises(RuntimeError, match="A must be row-major contiguous"):
        run(quant_a_bad)

    # The packed operand still runs, so the guard rejects only the bad layout.
    out = run(quant_a)
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
