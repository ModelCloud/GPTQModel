# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Smoke and math-accuracy tests for the Marlin-kernel scale-search objective."""

import pytest
import torch

from gptqmodel.quantization import GPTQConfig, ScaleSearchConfig, Quantizer


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize(
    "scale_search",
    [
        ScaleSearchConfig.MARLIN,
        ScaleSearchConfig.MARLIN_MSE,
        ScaleSearchConfig.MARLIN_ACTIVATION,
    ],
)
def test_find_params_marlin_produces_finite_scales(bits, group_size, scale_search):
    """Marlin scale search must return finite scales/zeros for supported configs."""

    rows, gs = 64, group_size
    qcfg = GPTQConfig(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
        mse=2.0,
        scale_search=scale_search,
    )
    quantizer = Quantizer(qcfg, shape=(rows, 1), name="test")
    quantizer.configure(perchannel=True, grid=10, maxshrink=0.8)

    W = torch.randn(rows, gs, device="cuda", dtype=torch.float16)
    H = torch.randn(gs, gs, device="cuda", dtype=torch.float32)
    H = H @ H.t() / gs + torch.eye(gs, device="cuda", dtype=torch.float32) * 0.1

    quantizer.find_params(W, weight=True, hessian=H)

    assert torch.isfinite(quantizer.scale).all()
    assert torch.isfinite(quantizer.zero).all()
    assert (quantizer.scale > 0).all()
    assert quantizer.zero.shape == (rows, 1)


@pytest.mark.parametrize("group_size", [32, 64, 128])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize(
    "scale_search",
    [
        ScaleSearchConfig.MARLIN,
        ScaleSearchConfig.MARLIN_MSE,
        ScaleSearchConfig.MARLIN_ACTIVATION,
    ],
)
def test_find_params_batched_marlin_produces_finite_scales(bits, group_size, scale_search):
    """Batched Marlin scale search must return finite per-row/per-group scales."""

    rows, num_groups, gs = 64, 2, group_size
    qcfg = GPTQConfig(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
        mse=2.0,
        scale_search=scale_search,
    )
    quantizer = Quantizer(qcfg, shape=(rows, num_groups), name="test")
    quantizer.configure(perchannel=True, grid=10, maxshrink=0.8)

    W3d = torch.randn(rows, num_groups, gs, device="cuda", dtype=torch.float16)
    H = torch.randn(num_groups, gs, gs, device="cuda", dtype=torch.float32)
    H = torch.einsum("gij,gkj->gik", H, H) / gs + torch.eye(
        gs, device="cuda", dtype=torch.float32
    ) * 0.1

    scale, zero = quantizer.find_params_batched(W3d, weight=True, hessian=H)

    assert torch.isfinite(scale).all()
    assert torch.isfinite(zero).all()
    assert (scale > 0).all()
    assert scale.shape == (rows, num_groups)
    assert zero.shape == (rows, num_groups)


def _base_scale_zero(W: torch.Tensor, qcfg: GPTQConfig):
    """Return the no-scale-search scale/zero for W using the same qcfg bits/group_size."""
    qcfg_no_search = GPTQConfig(
        bits=qcfg.bits,
        group_size=qcfg.group_size,
        desc_act=qcfg.desc_act,
        sym=qcfg.sym,
        mse=0.0,
        scale_search=None,
    )
    quantizer = Quantizer(qcfg_no_search, shape=(W.shape[0], 1), name="base")
    quantizer.configure(perchannel=True, grid=1, maxshrink=1.0)
    quantizer.find_params(W, weight=True)
    return quantizer.scale, quantizer.zero, int(quantizer.maxq.item())


def _make_test_hessian(gs: int, hessian_kind: str):
    """Build a well-conditioned Hessian or activation diagonal for math tests."""
    if hessian_kind == "none":
        return None
    if hessian_kind == "diagonal":
        return torch.rand(gs, device="cuda", dtype=torch.float32) + 0.1
    # Full Hessian: small low-rank component plus a stable identity so the
    # Cholesky factor used by the Marlin path has bounded dynamic range.
    R = torch.randn(gs, gs, device="cuda", dtype=torch.float32) * 0.1
    return R @ R.t() / gs + torch.eye(gs, device="cuda", dtype=torch.float32) * 0.5


@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize(
    ("method", "dense_method", "hessian_kind"),
    [
        (ScaleSearchConfig.MARLIN_MSE, ScaleSearchConfig.MSE, "none"),
        (ScaleSearchConfig.MARLIN_ACTIVATION, ScaleSearchConfig.ACTIVATION, "diagonal"),
        (ScaleSearchConfig.MARLIN, ScaleSearchConfig.HESSIAN, "full"),
    ],
)
def test_marlin_scale_search_loss_matches_dense(
    method, dense_method, hessian_kind, bits, group_size
):
    """Marlin-kernel scale-search loss must match the dense objective for one candidate."""

    rows, gs = 128, group_size
    qcfg = GPTQConfig(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
        mse=2.0,
        scale_search=method,
    )
    quantizer = Quantizer(qcfg, shape=(rows, 1), name="test")
    quantizer.configure(perchannel=True, grid=1, maxshrink=1.0)

    W = torch.randn(rows, gs, device="cuda", dtype=torch.float16)
    H = _make_test_hessian(gs, hessian_kind)

    scale, zero, maxq = _base_scale_zero(W, qcfg)

    # Build the per-output-row quantization error for the base scale.
    W_exp = W.unsqueeze(0)
    scale_exp = scale.unsqueeze(0)
    zero_exp = zero.unsqueeze(0)
    error = quantizer._quantize_scale_search_candidates(
        W_exp,
        scale_exp,
        zero_exp,
        maxq_value=maxq,
    )

    # Dense reference objective.
    if dense_method == ScaleSearchConfig.MSE:
        prepared_dense = None
    else:
        prepared_dense = quantizer._prepare_scale_search_hessian(
            H,
            method=dense_method,
            columns=gs,
            device=W.device,
        )
    dense_loss = quantizer._scale_search_error(
        error,
        method=dense_method,
        mse=2.0,
        hessian=prepared_dense,
    )

    # Marlin-kernel objective (fallbacks to dense for unsupported shapes).
    if method == ScaleSearchConfig.MARLIN_MSE:
        prepared_marlin = None
    else:
        prepared_marlin = quantizer._prepare_scale_search_hessian(
            H,
            method=method,
            columns=gs,
            device=W.device,
        )
    marlin_loss = quantizer._marlin_scale_search_loss(
        W.unsqueeze(1),
        scale_exp,
        zero_exp,
        prepared_marlin,
        method=method,
        bits=bits,
        group_size=group_size,
        sym=True,
        dtype=W.dtype,
        pack_dtype=torch.int32,
        maxq_value=maxq,
        mse=2.0,
    )

    assert getattr(quantizer, "_marlin_scale_search_kernel_ran", False)
    assert getattr(quantizer, "_marlin_scale_search_fallback_count", 0) == 0

    # The kernel path is approximate due to FP16 GEMM; the fallback path is the
    # dense objective.  1% relative + 1e-3 absolute covers both cases.
    assert torch.allclose(
        dense_loss.squeeze(),
        marlin_loss.squeeze(),
        rtol=0.01,
        atol=1e-3,
    ), f"{method.value} loss mismatch for bits={bits}, group_size={group_size}"


def _dense_objective_at_scale(
    W: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    maxq: int,
    prepared_dense: torch.Tensor | None,
    dense_method: ScaleSearchConfig,
    quantizer: Quantizer,
) -> torch.Tensor:
    """Evaluate the dense objective for the given clipping scale."""
    W_exp = W.unsqueeze(0)
    scale_exp = scale.unsqueeze(0)
    zero_exp = zero.unsqueeze(0)
    error = quantizer._quantize_scale_search_candidates(
        W_exp,
        scale_exp,
        zero_exp,
        maxq_value=maxq,
    )
    return quantizer._scale_search_error(
        error,
        method=dense_method,
        mse=2.0,
        hessian=prepared_dense,
    ).sum()


@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize(
    ("method", "dense_method", "hessian_kind"),
    [
        (ScaleSearchConfig.MARLIN_MSE, ScaleSearchConfig.MSE, "none"),
        (ScaleSearchConfig.MARLIN_ACTIVATION, ScaleSearchConfig.ACTIVATION, "diagonal"),
        (ScaleSearchConfig.MARLIN, ScaleSearchConfig.HESSIAN, "full"),
    ],
)
def test_find_params_marlin_objective_close_to_dense(
    method, dense_method, hessian_kind, bits, group_size
):
    """Marlin scale search should choose scales that are near-optimal for the dense objective."""

    rows, gs = 128, group_size
    qcfg_marlin = GPTQConfig(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
        mse=2.0,
        scale_search=method,
    )
    qcfg_dense = GPTQConfig(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
        mse=2.0,
        scale_search=dense_method,
    )
    quantizer_marlin = Quantizer(qcfg_marlin, shape=(rows, 1), name="marlin")
    quantizer_dense = Quantizer(qcfg_dense, shape=(rows, 1), name="dense")
    quantizer_marlin.configure(perchannel=True, grid=20, maxshrink=0.8)
    quantizer_dense.configure(perchannel=True, grid=20, maxshrink=0.8)

    W = torch.randn(rows, gs, device="cuda", dtype=torch.float16)
    H = _make_test_hessian(gs, hessian_kind)

    quantizer_marlin.find_params(W, weight=True, hessian=H)
    quantizer_dense.find_params(W, weight=True, hessian=H)

    assert getattr(quantizer_marlin, "_marlin_scale_search_kernel_ran", False)
    assert getattr(quantizer_marlin, "_marlin_scale_search_fallback_count", 0) == 0

    maxq = int(quantizer_marlin.maxq.item())
    if dense_method == ScaleSearchConfig.MSE:
        prepared_dense = None
    else:
        prepared_dense = quantizer_dense._prepare_scale_search_hessian(
            H,
            method=dense_method,
            columns=gs,
            device=W.device,
        )

    loss_marlin_scale = _dense_objective_at_scale(
        W,
        quantizer_marlin.scale,
        quantizer_marlin.zero,
        maxq,
        prepared_dense,
        dense_method,
        quantizer_dense,
    )
    loss_dense_scale = _dense_objective_at_scale(
        W,
        quantizer_dense.scale,
        quantizer_dense.zero,
        maxq,
        prepared_dense,
        dense_method,
        quantizer_dense,
    )

    # The dense scale is optimal under the dense objective; the Marlin-chosen
    # scale should not materially increase that objective.
    assert torch.allclose(
        loss_marlin_scale,
        loss_dense_scale,
        rtol=0.01,
        atol=1e-2,
    ), f"{method.value} chose a scale far from {dense_method.value} optimum for bits={bits}, group_size={group_size}"
