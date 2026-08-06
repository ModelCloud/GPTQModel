# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Strict, model-shape accuracy coverage for batched grouped scale search.

This test file is intentionally separate from test_quantizer_scale_search.py so
that CI can run the faster shape-matrix unit tests first and treat the larger
configurations here as an extended regression gate. It still completes in a few
seconds because the Hessians are block-diagonal and the largest shape is the
square 4096x4096 case that all modern LLM MLP/attention projection layers hit.
"""

import os

import pytest
import torch

from gptqmodel.quantization import QuantizeConfig, Quantizer, ScaleSearchConfig


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _method_seed(method: ScaleSearchConfig) -> int:
    return {
        ScaleSearchConfig.ACTIVATION: 100,
        ScaleSearchConfig.HESSIAN: 200,
        ScaleSearchConfig.HYBRID: 300,
        ScaleSearchConfig.MSE: 400,
    }[method]


def _make_group_block_hessian(cols: int, group_size: int, device: str = "cuda", dtype=torch.float32):
    """Build a block-diagonal positive-definite Hessian for grouped scale search."""
    num_groups = cols // group_size
    groups = []
    for _ in range(num_groups):
        a = torch.randn(2 * group_size, group_size, device=device, dtype=dtype)
        groups.append(a.T @ a / a.shape[0])
    return torch.block_diag(*groups)


def _reference_find_params(W: torch.Tensor, hessian: torch.Tensor, qcfg: QuantizeConfig):
    """Reference per-group find_params run on each group of W."""
    rows, cols = W.shape
    group_size = qcfg.group_size
    num_groups = cols // group_size
    scale_ref = []
    zero_ref = []
    for g in range(num_groups):
        q = Quantizer(qcfg)
        q.configure(perchannel=True, grid=100, maxshrink=0.8)
        q.maxq = q.maxq.to(W.device)
        block = W[:, g * group_size : (g + 1) * group_size]
        h_block = hessian[
            g * group_size : (g + 1) * group_size,
            g * group_size : (g + 1) * group_size,
        ]
        q.find_params(block, weight=True, hessian=h_block)
        scale_ref.append(q.scale.reshape(rows))
        zero_ref.append(q.zero.reshape(rows))
    return torch.stack(scale_ref, dim=1), torch.stack(zero_ref, dim=1)


def _batched_find_params(W: torch.Tensor, hessian: torch.Tensor, qcfg: QuantizeConfig, triton_flag: str):
    """Run find_params_batched with a deterministic Triton toggle."""
    rows, cols = W.shape
    group_size = qcfg.group_size
    num_groups = cols // group_size
    W3d = W.reshape(rows, num_groups, group_size)
    method = qcfg.scale_search
    if method == ScaleSearchConfig.ACTIVATION:
        hessian_batched = hessian.diagonal().reshape(num_groups, group_size)
    else:
        hessian_batched = torch.stack(
            [
                hessian[
                    g * group_size : (g + 1) * group_size,
                    g * group_size : (g + 1) * group_size,
                ]
                for g in range(num_groups)
            ],
            dim=0,
        )

    prev = os.environ.get("GPTQMODEL_SCALE_SEARCH_TRITON")
    os.environ["GPTQMODEL_SCALE_SEARCH_TRITON"] = triton_flag
    try:
        q = Quantizer(qcfg)
        q.configure(perchannel=True, grid=100, maxshrink=0.8)
        q.maxq = q.maxq.to(W.device)
        scale_b, zero_b = q.find_params_batched(W3d, weight=True, hessian=hessian_batched)
        torch.cuda.synchronize()
        return scale_b, zero_b
    finally:
        if prev is None:
            os.environ.pop("GPTQMODEL_SCALE_SEARCH_TRITON", None)
        else:
            os.environ["GPTQMODEL_SCALE_SEARCH_TRITON"] = prev


@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("sym", [False, True])
@pytest.mark.parametrize(
    "method",
    [ScaleSearchConfig.ACTIVATION, ScaleSearchConfig.HESSIAN, ScaleSearchConfig.HYBRID],
)
def test_find_params_batched_large_square_matches_per_group_reference(
    group_size, bits, sym, method
):
    """4096x4096 square MLP/attention projection shape, all bits/sym/methods."""
    rows = 4096
    cols = 4096
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(
        7000 + group_size + bits + int(sym) + _method_seed(method)
    )
    W = torch.randn(rows, cols, generator=generator, device=device, dtype=torch.float32)
    hessian = _make_group_block_hessian(cols, group_size, device=device)

    qcfg = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        sym=sym,
        mse=2.0,
        scale_search=method,
    )

    scale_ref, zero_ref = _reference_find_params(W, hessian, qcfg)
    triton_flag = "1" if method == ScaleSearchConfig.ACTIVATION else "0"
    scale_b, zero_b = _batched_find_params(W, hessian, qcfg, triton_flag=triton_flag)

    scale_diff = (scale_b - scale_ref).abs().max().item()
    zero_diff = (zero_b - zero_ref).abs().max().item()
    assert scale_diff < 1e-6, (
        f"scale mismatch {scale_diff} for rows={rows}, cols={cols}, "
        f"{group_size=}, {bits=}, {sym=}, {method=}"
    )
    assert zero_diff < 1e-6, (
        f"zero mismatch {zero_diff} for rows={rows}, cols={cols}, "
        f"{group_size=}, {bits=}, {sym=}, {method=}"
    )


@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("sym", [False, True])
@pytest.mark.parametrize(
    "method",
    [ScaleSearchConfig.ACTIVATION, ScaleSearchConfig.HESSIAN, ScaleSearchConfig.HYBRID],
)
def test_find_params_batched_narrow_rows_matches_per_group_reference(
    group_size, bits, sym, method
):
    """Thin 128x512 matrix with multiple groups, all bits/sym/methods."""
    rows = 128
    cols = 512
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(
        8000 + group_size + bits + int(sym) + _method_seed(method)
    )
    W = torch.randn(rows, cols, generator=generator, device=device, dtype=torch.float32)
    hessian = _make_group_block_hessian(cols, group_size, device=device)

    qcfg = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        sym=sym,
        mse=2.0,
        scale_search=method,
    )

    scale_ref, zero_ref = _reference_find_params(W, hessian, qcfg)
    triton_flag = "1" if method == ScaleSearchConfig.ACTIVATION else "0"
    scale_b, zero_b = _batched_find_params(W, hessian, qcfg, triton_flag=triton_flag)

    scale_diff = (scale_b - scale_ref).abs().max().item()
    zero_diff = (zero_b - zero_ref).abs().max().item()
    assert scale_diff < 1e-6, (
        f"scale mismatch {scale_diff} for rows={rows}, cols={cols}, "
        f"{group_size=}, {bits=}, {sym=}, {method=}"
    )
    assert zero_diff < 1e-6, (
        f"zero mismatch {zero_diff} for rows={rows}, cols={cols}, "
        f"{group_size=}, {bits=}, {sym=}, {method=}"
    )


def test_activation_search_adversarial_reduction_order_matches_exact_path():
    """A Triton reduction-order miss must not bypass the exhaustive scorer."""

    generator = torch.Generator(device="cuda").manual_seed(9001)
    weights = torch.randn((512, 128), generator=generator, device="cuda", dtype=torch.bfloat16) * 1e-3
    importance = torch.exp(torch.randn(128, generator=generator, device="cuda", dtype=torch.float32) * 4)
    hessian = torch.diag(importance)
    qcfg = QuantizeConfig(
        bits=4,
        group_size=128,
        sym=False,
        mse=2.0,
        scale_search=ScaleSearchConfig.ACTIVATION,
    )

    exact_scale, exact_zero = _reference_find_params(weights, hessian, qcfg)
    default_scale, default_zero = _batched_find_params(weights, hessian, qcfg, triton_flag="1")

    assert torch.equal(default_scale, exact_scale)
    assert torch.equal(default_zero, exact_zero)
