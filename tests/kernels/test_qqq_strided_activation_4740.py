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

from gptqmodel.nn_modules.qlinear.qqq import QQQLinear, QQQTorchLinear
from gptqmodel.utils.qqq import qqq_gemm, qqq_runtime_available, qqq_runtime_error


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
    return _build(QQQLinear)


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
    native = _build(QQQLinear)
    oracle = _build(QQQTorchLinear)
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
