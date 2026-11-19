# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""#4740: QQQ must accept a column-strided activation.

``QQQLinear.forward`` reshapes the activation to ``(-1, K)``. For a
column-strided input (``x.t()`` of a ``[K, M]`` buffer, the layout a transposed
prefill produces) that reshape keeps the transposed view. The INT8 fallback in
``dynamic_quant`` is element-wise, so it preserves those strides, and the native
``qqq_gemm`` then reads the packed activation row-major - silently multiplying a
scrambled operand (relative error ~1.4, no error raised).

The tests drive the real AUTO kernel: the strided call must match the
contiguous call, and the native entry must reject a strided operand instead of
returning garbage if a future caller forgets to pack it.
"""

from __future__ import annotations

import pytest
import torch

import gptqmodel.nn_modules.qlinear.qqq as qqq_mod
from gptqmodel.utils.qqq import qqq_gemm, qqq_runtime_available, qqq_runtime_error
from gptqmodel.utils.rocm import IS_ROCM


pytestmark = pytest.mark.cuda

IN_FEATURES = 896
OUT_FEATURES = 4864
GROUP_SIZE = 128
# M = 1 is the case the kernel already got right (the reshape is contiguous for
# a single row); the rest exercise the strided operand.
ROWS = (2, 16, 17, 128)


def _build(cls):
    torch.manual_seed(4740)
    linear = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False, dtype=torch.float16)
    with torch.no_grad():
        linear.weight.normal_(0, 0.02)
    module = cls(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
    )
    groups = IN_FEATURES // GROUP_SIZE
    scales = torch.rand(OUT_FEATURES, groups, dtype=torch.float16) * 0.03 + 0.01
    channels = torch.rand(OUT_FEATURES, dtype=torch.float32) * 0.002 + 0.001
    module.pack(linear, scales, channels)
    module.post_init()
    return module.cuda().eval()


def _column_strided(rows: int) -> torch.Tensor:
    base = torch.randn(IN_FEATURES, rows, device="cuda", dtype=torch.float16)
    x = base.t()
    assert x.shape == (rows, IN_FEATURES)
    assert not x.is_contiguous()
    assert x.stride(-1) > 1
    return x


def _reference(module, x: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return module(x.contiguous())


@pytest.fixture(scope="module")
def qqq_module():
    if not qqq_runtime_available():
        pytest.skip(qqq_runtime_error() or "QQQ CUDA runtime unavailable")
    return _build(qqq_mod.QQQLinear)


@pytest.mark.parametrize("rows", ROWS)
def test_qqq_forward_matches_contiguous_for_column_strided_input(qqq_module, rows):
    x = _column_strided(rows)

    with torch.inference_mode():
        strided = qqq_module(x)
    expected = _reference(qqq_module, x)

    assert strided.shape == expected.shape
    rel = (strided.float() - expected.float()).norm() / expected.float().norm()
    assert rel < 5e-3, f"M={rows}: column-strided output diverged (rel={rel.item():.3e})"


@pytest.mark.parametrize("rows", ROWS)
def test_qqq_forward_matches_torch_reference_for_column_strided_input(rows):
    """Compare against the independent torch kernel, not just the same op twice."""

    if not qqq_runtime_available():
        pytest.skip(qqq_runtime_error() or "QQQ CUDA runtime unavailable")
    native = _build(qqq_mod.QQQLinear)
    oracle = _build(qqq_mod.QQQTorchLinear)
    oracle.load_state_dict(native.state_dict(), strict=False)

    x = _column_strided(rows)
    with torch.inference_mode():
        actual = native(x)
        expected = oracle(x.contiguous())

    rel = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert rel < 1e-2, f"M={rows}: native QQQ diverged from QQQ_TORCH (rel={rel.item():.3e})"


def test_qqq_native_entry_rejects_a_strided_activation(qqq_module):
    """A strided operand must fail loudly at the op boundary, never silently."""

    rows = 8
    x = _column_strided(rows)
    with torch.inference_mode():
        quant_a, scale = qqq_module.dynamic_quant(x)
        assert quant_a.is_contiguous(), "dynamic_quant must return a packed activation"

        strided_a = quant_a.t().t()  # same values, still packed
        assert strided_a.is_contiguous()

        # Build a genuinely strided INT8 operand: transpose a wider buffer.
        wide = torch.zeros(rows, IN_FEATURES * 2, dtype=torch.int8, device="cuda")
        wide[:, ::2] = quant_a
        strided_operand = wide[:, ::2]
        assert not strided_operand.is_contiguous()

        out = torch.empty(rows, OUT_FEATURES, dtype=torch.float16, device="cuda")
        with pytest.raises(RuntimeError, match="contiguous"):
            qqq_gemm(
                strided_operand,
                qqq_module.B,
                qqq_module._reduce_scratch(x.device),
                out,
                scale,
                qqq_module.s_channel,
                qqq_module.s_group,
                qqq_module._split_k_locks(x.device),
                max_par=qqq_module.max_par,
            )


# --- Follow-up coverage (#4740): the fallback operand and the native guard ---


def _column_strided_k(rows: int, in_features: int) -> torch.Tensor:
    """Return the ``[rows, K]`` column-strided view batched decoding produces."""

    x = torch.randn(in_features, rows, dtype=torch.float16).t()
    assert x.shape == (rows, in_features)
    assert not x.is_contiguous()
    assert x.stride(-1) != 1
    return x


def _dequantized_weight(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Independent FP64 QQQ dequant: ``(code - 8) * group_scale``."""

    group_scale = scales.double().repeat_interleave(GROUP_SIZE, dim=1)
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

    module = qqq_mod.QQQLinear(
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
    """The element-wise fallback must return densely packed rows."""

    monkeypatch.setattr(qqq_mod, "_qqq_gpu_quantizer", lambda: None)

    module = qqq_mod.QQQLinear.__new__(qqq_mod.QQQLinear)
    torch.nn.Module.__init__(module)

    x_strided = _column_strided_k(rows=17, in_features=GROUP_SIZE)
    quant_strided, scale_strided = module.dynamic_quant(x_strided)
    quant_contiguous, scale_contiguous = module.dynamic_quant(x_strided.contiguous())

    assert quant_strided.is_contiguous(), quant_strided.stride()
    assert quant_strided.dtype == torch.int8
    torch.testing.assert_close(scale_strided, scale_contiguous, rtol=0, atol=0)
    assert torch.equal(quant_strided, quant_contiguous)


@pytest.mark.cuda
@pytest.mark.parametrize("rows", [2, 16, 17, 128])
def test_qqq_linear_matches_the_packed_layout_call_for_a_column_strided_activation(rows):
    """The reported symptom: relative error ~1.4 against the packed-layout call."""

    if not qqq_runtime_available():
        pytest.skip(qqq_runtime_error() or "QQQ CUDA runtime unavailable")
    if IS_ROCM:
        pytest.skip("QQQ's native kernel requires an NVIDIA CUDA device")

    module, expected_weight = _build_native_module(in_features=896, out_features=4864)

    x_strided = _column_strided_k(rows=rows, in_features=module.in_features).cuda()
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

    oracle = x_contiguous.double() @ expected_weight.t()
    oracle_error = (strided.double() - oracle).norm() / oracle.norm()
    assert oracle_error.item() < 5e-2, oracle_error.item()


@pytest.mark.cuda
def test_qqq_linear_keeps_row_strided_and_batched_inputs_working():
    """Control: layouts the element-wise path already packed contiguously."""

    if not qqq_runtime_available():
        pytest.skip(qqq_runtime_error() or "QQQ CUDA runtime unavailable")
    if IS_ROCM:
        pytest.skip("QQQ's native kernel requires an NVIDIA CUDA device")

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


@pytest.mark.cuda
def test_native_qqq_gemm_rejects_a_non_contiguous_activation():
    """The kernel must not silently read a column-major operand row-major."""

    if not qqq_runtime_available():
        pytest.skip(qqq_runtime_error() or "QQQ CUDA runtime unavailable")
    if IS_ROCM:
        pytest.skip("QQQ's native kernel requires an NVIDIA CUDA device")

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
        qqq_mod._workspace_numel(out_features, 16), device="cuda", dtype=torch.int32
    )

    def run(operand):
        out = torch.empty((rows, out_features), device="cuda", dtype=torch.float16)
        qqq_mod.mul(operand, packed_weight, reduce_scratch, out, s1, s2, s3, workspace)
        return out

    with pytest.raises(RuntimeError, match="contiguous"):
        run(quant_a_bad)

    out = run(quant_a)
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
