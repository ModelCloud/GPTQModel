# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Hadamard transforms must accept a zero-element operand.

Every quant-linear backend applies the online rotation before it checks for an
empty batch. The native extension used to launch a zero-sized grid
(``CUDA error: invalid argument``) and the Torch butterfly used to infer a
``-1`` extent on a zero-element tensor (``RuntimeError``), so a decode/prefill
step with no rows crashed instead of returning an empty tensor.
"""

import pytest
import torch

from gptqmodel.quantization.rotation import hadamard_utils as had


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA or ROCm GPU required"
)


@requires_cuda
@pytest.mark.cuda
@pytest.mark.parametrize(
    "op_name",
    [
        "fast_hadamard_transform",
        "fast_hadamard_transform_12N",
        "fast_hadamard_transform_20N",
        "fast_hadamard_transform_28N",
        "fast_hadamard_transform_40N",
    ],
)
def test_native_hadamard_accepts_an_empty_batch(op_name):
    from gptqmodel.utils.hadamard import hadamard_op

    op = hadamard_op(op_name)
    x = torch.empty((0, 128), device="cuda", dtype=torch.float16)

    y = op(x, 1.0)

    assert y.shape == x.shape
    assert y.dtype == x.dtype


@requires_cuda
@pytest.mark.cuda
def test_native_hadamard_keeps_the_leading_empty_extent():
    from gptqmodel.utils.hadamard import hadamard_transform

    x = torch.empty((3, 0, 256), device="cuda", dtype=torch.float32)

    y = hadamard_transform(x, 1.0)

    assert y.shape == x.shape


@pytest.mark.parametrize("shape", [(0, 256), (2, 0, 256)])
def test_torch_fallback_accepts_a_zero_element_operand(shape):
    x = torch.empty(shape, dtype=torch.float16)

    y = had._TorchHadamardTransform.hadamard_transform(x, 1.0)

    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_apply_online_hadamard_short_circuits_an_empty_operand():
    x = torch.empty((2, 0, 256), dtype=torch.float16)

    y = had.apply_online_hadamard(x, True, False, None, 1)

    assert y.shape == x.shape
    assert y.dtype == x.dtype


@pytest.mark.parametrize(
    "rows, expected_calls",
    [(0, []), (2, ["had_r_128", "hgemm", "had_r_128"])],
)
def test_exl3_forward_guards_empty_batch_before_kernel_launch(monkeypatch, rows, expected_calls):
    """The torch_mode path must not hand a zero-row operand to the kernels."""

    from gptqmodel.exllamav3.modules.quant import exl3 as native_exl3

    calls = []

    class FakeExt:
        @staticmethod
        def had_r_128(x, y, suh, svh, scale):
            calls.append("had_r_128")

        @staticmethod
        def hgemm(x, weight, y):
            calls.append("hgemm")

    class FakeBC:
        def run(self, x, y):
            calls.append("bc")

    monkeypatch.setattr(native_exl3, "ext", FakeExt)
    inner = object.__new__(native_exl3.LinearEXL3)
    inner.in_features = inner.out_features = 128
    inner.out_dtype = torch.float16
    inner.suh = inner.svh = inner.bias = None
    inner.bc = FakeBC()
    inner.get_inner_weight_tensor = lambda: torch.eye(128, dtype=torch.float16)

    x = torch.empty((rows, 128), dtype=torch.float16)
    actual = inner.forward(x, {"reconstruct": True})

    assert actual.shape == (rows, 128)
    assert actual.dtype == torch.float16
    assert calls == expected_calls


@requires_cuda
@pytest.mark.cuda
def test_exllamav3_had_r_128_accepts_an_empty_batch():
    ext = pytest.importorskip("gptqmodel.exllamav3.ext")
    if not ext.exllamav3_runtime_available():
        pytest.skip("ExLlamaV3 extension is not built")

    x = torch.empty((0, 128), device="cuda", dtype=torch.float16)
    y = torch.empty_like(x)

    ext.exllamav3_ext.had_r_128(x, y, None, None, 1.0)
    torch.cuda.synchronize()

    assert y.shape == (0, 128)
    assert y.dtype == torch.float16


@requires_cuda
@pytest.mark.cuda
def test_exllamav3_had_r_128_still_launches_for_nonempty_rows():
    ext = pytest.importorskip("gptqmodel.exllamav3.ext")
    if not ext.exllamav3_runtime_available():
        pytest.skip("ExLlamaV3 extension is not built")

    x = torch.ones((2, 128), device="cuda", dtype=torch.float16)
    y = torch.empty_like(x)

    ext.exllamav3_ext.had_r_128(x, y, None, None, 1.0)
    torch.cuda.synchronize()

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert not torch.equal(y, x)
