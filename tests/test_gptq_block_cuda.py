# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Accuracy, edge-case, and free-threaded tests for the fused CUDA GPTQ block kernel."""

import importlib
import os
import subprocess
import sys
from dataclasses import dataclass
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import gptqmodel
from gptqmodel.quantization import QuantizeConfig, ScaleSearchConfig
from gptqmodel.utils.python import has_gil_disabled


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


@dataclass(frozen=True)
class RawCase:
    rows: int
    count: int
    group_size: int
    bits: int
    weight_kind: str
    hessian_kind: str
    groupwise: bool = False


def _serial_block(W1, Q1, Err1, Hinv1, scale, zero, maxq, group_size, groupwise=False):
    """Independent eager reference for one grouped-GPTQ block."""

    for i in range(W1.shape[1]):
        w = W1[:, i]
        q_scale = scale[:, i // group_size]
        q_zero = zero[:, i // group_size]
        if groupwise:
            q = q_scale * torch.clamp(torch.round(w / q_scale), -maxq, maxq)
        else:
            q = q_scale * (
                torch.clamp(torch.round(w / q_scale) + q_zero, 0, maxq) - q_zero
            )
        err = (w - q) / Hinv1[i, i]
        Q1[:, i] = q
        Err1[:, i] = err
        W1[:, i:] = torch.addr(W1[:, i:], err, Hinv1[i, i:], alpha=-1.0)


def _case_inputs(
    case: RawCase,
    seed: int = 0,
    *,
    strided_operands: bool = False,
    device: torch.device | str = "cuda:0",
):
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    groups = case.count // case.group_size

    scale = (
        torch.rand(
            case.rows, groups, generator=generator, device=device, dtype=torch.float32
        )
        + 0.125
    )
    zero = torch.randint(
        0,
        2**case.bits,
        (case.rows, groups),
        generator=generator,
        device=device,
        dtype=torch.int32,
    ).float()

    if case.weight_kind == "gaussian":
        weights = torch.randn(
            case.rows,
            case.count,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
    elif case.weight_kind == "uniform":
        weights = (
            torch.rand(
                case.rows,
                case.count,
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            .mul_(8.0)
            .sub_(4.0)
        )
    elif case.weight_kind == "skewed":
        base = torch.randn(
            case.rows,
            case.count,
            generator=generator,
            device=device,
            dtype=torch.float32,
        ).clamp_(-4.0, 4.0)
        weights = base.exp().sub_(1.0)
    elif case.weight_kind == "heavy_tailed":
        uniform = torch.rand(
            case.rows,
            case.count,
            generator=generator,
            device=device,
            dtype=torch.float32,
        ).clamp_(0.001, 0.999)
        weights = torch.tan((uniform - 0.5) * torch.pi).clamp_(-1e4, 1e4)
    elif case.weight_kind == "zeros":
        weights = torch.zeros(case.rows, case.count, device=device, dtype=torch.float32)
    elif case.weight_kind == "constants":
        constants = torch.linspace(
            -3.5, 3.5, case.rows, device=device, dtype=torch.float32
        )
        weights = constants[:, None].expand(-1, case.count).clone()
    elif case.weight_kind == "rounding_ties":
        # Powers-of-two scales keep exact half-bin ratios representable. A diagonal
        # Hessian prevents previous columns from perturbing those boundaries.
        scale = torch.full(
            (case.rows, groups), 0.25, device=device, dtype=torch.float32
        )
        zero = torch.full(
            (case.rows, groups),
            2 ** (case.bits - 1),
            device=device,
            dtype=torch.float32,
        )
        ratios = torch.tensor(
            [-9.5, -8.5, -7.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 6.5, 7.5, 8.5, 9.5],
            device=device,
            dtype=torch.float32,
        )
        weights = ratios.repeat(
            (case.rows * case.count + ratios.numel() - 1) // ratios.numel()
        )
        weights = (weights[: case.rows * case.count] * 0.25).reshape(
            case.rows, case.count
        )
    elif case.weight_kind == "outliers":
        weights = torch.randn(
            case.rows,
            case.count,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        weights[:, ::11] *= 1_000.0
        weights[:, 1::17] *= 1e-3
    elif case.weight_kind == "single_outlier":
        weights = torch.randn(
            case.rows,
            case.count,
            generator=generator,
            device=device,
            dtype=torch.float32,
        ).mul_(1e-3)
        weights[0, 0] = 1e6
        weights[-1, -1] = -1e6
    elif case.weight_kind == "exact_bins":
        scale = torch.full(
            (case.rows, groups), 0.125, device=device, dtype=torch.float32
        )
        zero = torch.full(
            (case.rows, groups),
            2 ** (case.bits - 1),
            device=device,
            dtype=torch.float32,
        )
        codes = torch.arange(case.rows * case.count, device=device) % (2**case.bits)
        group_index = torch.arange(case.count, device=device) // case.group_size
        weights = scale[:, group_index] * (
            codes.reshape(case.rows, case.count) - zero[:, group_index]
        )
    elif case.weight_kind == "tie_ulp":
        scale = torch.full(
            (case.rows, groups), 0.25, device=device, dtype=torch.float32
        )
        zero = torch.full(
            (case.rows, groups),
            2 ** (case.bits - 1),
            device=device,
            dtype=torch.float32,
        )
        ties = torch.arange(-7.5, 8.0, 1.0, device=device, dtype=torch.float32)
        lower = torch.nextafter(ties, torch.full_like(ties, -torch.inf))
        upper = torch.nextafter(ties, torch.full_like(ties, torch.inf))
        ratios = torch.stack((lower, ties, upper), dim=1).flatten()
        tiled = ratios.repeat(
            (case.rows * case.count + ratios.numel() - 1) // ratios.numel()
        )
        weights = (tiled[: case.rows * case.count] * 0.25).reshape(
            case.rows, case.count
        )
    elif case.weight_kind == "alternating_sparse":
        weights = torch.zeros(case.rows, case.count, device=device, dtype=torch.float32)
        values = torch.logspace(
            -4, 4, (case.count + 2) // 3, device=device, dtype=torch.float32
        )
        weights[:, ::3] = values[: weights[:, ::3].shape[1]]
        weights[:, 1::3] = -values[: weights[:, 1::3].shape[1]]
    elif case.weight_kind == "scale_extremes":
        scale_row = torch.logspace(-6, 6, groups, device=device, dtype=torch.float32)
        scale = scale_row.unsqueeze(0).expand(case.rows, -1).clone()
        zero = torch.arange(groups, device=device, dtype=torch.float32).remainder(
            2**case.bits
        )
        zero = zero.unsqueeze(0).expand(case.rows, -1).clone()
        group_index = torch.arange(case.count, device=device) // case.group_size
        ratios = torch.linspace(
            -20.0, 20.0, case.count, device=device, dtype=torch.float32
        )
        weights = scale[:, group_index] * ratios.unsqueeze(0)
    elif case.weight_kind == "tiny_normals":
        scale = torch.full(
            (case.rows, groups), 2.0**-120, device=device, dtype=torch.float32
        )
        zero = torch.full(
            (case.rows, groups),
            2 ** (case.bits - 1),
            device=device,
            dtype=torch.float32,
        )
        ratios = torch.tensor(
            [-7.5, -1.5, -0.5, -0.0, 0.0, 0.5, 1.5, 7.5],
            device=device,
            dtype=torch.float32,
        )
        tiled = ratios.repeat(
            (case.rows * case.count + ratios.numel() - 1) // ratios.numel()
        )
        weights = (tiled[: case.rows * case.count] * scale[0, 0]).reshape(
            case.rows, case.count
        )
    else:
        raise AssertionError(f"Unknown weight kind: {case.weight_kind}")

    if case.hessian_kind == "identity":
        hessian_inverse = torch.eye(case.count, device=device, dtype=torch.float32)
    elif case.hessian_kind == "diagonal":
        diagonal = torch.logspace(-1, 1, case.count, device=device, dtype=torch.float32)
        hessian_inverse = torch.diag(diagonal)
    elif case.hessian_kind == "correlated":
        factor = torch.randn(
            case.count,
            case.count,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        covariance = factor @ factor.T + 0.5 * torch.eye(case.count, device=device)
        hessian_inverse = torch.linalg.cholesky(
            torch.linalg.inv(covariance), upper=True
        )
    elif case.hessian_kind == "ill_conditioned":
        diagonal = torch.logspace(-4, 4, case.count, device=device, dtype=torch.float32)
        hessian_inverse = torch.diag(diagonal)
        if case.count > 1:
            upper = torch.randn(
                case.count,
                case.count,
                generator=generator,
                device=device,
                dtype=torch.float32,
            ).triu(diagonal=1)
            hessian_inverse.add_(upper * 1e-4)
    else:
        raise AssertionError(f"Unknown Hessian kind: {case.hessian_kind}")

    if strided_operands:
        weight_storage = torch.empty(
            case.rows, case.count * 2, device=device, dtype=torch.float32
        )
        weight_storage[:, ::2] = weights
        weights = weight_storage[:, ::2]
        hessian_storage = torch.empty(
            case.count, case.count * 2, device=device, dtype=torch.float32
        )
        hessian_storage[:, ::2] = hessian_inverse
        hessian_inverse = hessian_storage[:, ::2]
        scale_storage = torch.empty(
            case.rows, groups * 2, device=device, dtype=torch.float32
        )
        scale_storage[:, ::2] = scale
        scale = scale_storage[:, ::2]
        zero_storage = torch.empty(
            case.rows, groups * 2, device=device, dtype=torch.float32
        )
        zero_storage[:, ::2] = zero
        zero = zero_storage[:, ::2]
        assert not weights.is_contiguous()
        assert not hessian_inverse.is_contiguous()
        assert not scale.is_contiguous()
        assert not zero.is_contiguous()

    return weights, hessian_inverse, scale, zero


def _logical_codes(quantized, scale, zero, group_size, *, groupwise):
    group_index = (
        torch.arange(quantized.shape[1], device=quantized.device) // group_size
    )
    expanded_scale = scale[:, group_index]
    if groupwise:
        return torch.round(quantized / expanded_scale).to(torch.int32)
    expanded_zero = zero[:, group_index]
    return torch.round(quantized / expanded_scale + expanded_zero).to(torch.int32)


def _production_logical_codes(quantized, scale, zero, group_index):
    expanded_scale = scale[:, group_index.to(torch.long)]
    expanded_zero = zero[:, group_index.to(torch.long)]
    return torch.round(quantized.float() / expanded_scale + expanded_zero).to(
        torch.int32
    )


def _run_raw_case(
    case: RawCase,
    seed: int = 0,
    *,
    strided_operands: bool = False,
    before_synchronize=None,
    device: torch.device | str = "cuda:0",
):
    import gptqmodel.utils.gptq_block as block_module

    weights, hessian_inverse, scale, zero = _case_inputs(
        case,
        seed,
        strided_operands=strided_operands,
        device=device,
    )
    maxq = 2 ** (case.bits - 1) - 1 if case.groupwise else 2**case.bits - 1

    reference_weights = weights.clone()
    reference_quantized = torch.empty_like(reference_weights)
    reference_errors = torch.empty_like(reference_weights)
    _serial_block(
        reference_weights,
        reference_quantized,
        reference_errors,
        hessian_inverse.contiguous(),
        scale.contiguous(),
        zero.contiguous(),
        maxq,
        case.group_size,
        case.groupwise,
    )

    actual_quantized, actual_errors = block_module.gptq_block_cuda(
        weights,
        hessian_inverse,
        scale,
        zero,
        maxq,
        case.group_size,
        groupwise=case.groupwise,
    )
    if before_synchronize is not None:
        before_synchronize()
    torch.cuda.synchronize(weights.device)

    assert torch.isfinite(actual_quantized).all()
    assert torch.isfinite(actual_errors).all()
    torch.testing.assert_close(actual_quantized, reference_quantized, atol=0, rtol=0)
    # Explicit round-to-nearest multiply/subtract instructions in the CUDA
    # correction step preserve the eager torch.addr operation order exactly.
    torch.testing.assert_close(actual_errors, reference_errors, atol=0, rtol=0)
    torch.testing.assert_close(
        _logical_codes(
            actual_quantized, scale, zero, case.group_size, groupwise=case.groupwise
        ),
        _logical_codes(
            reference_quantized, scale, zero, case.group_size, groupwise=case.groupwise
        ),
        atol=0,
        rtol=0,
    )
    return actual_quantized, actual_errors


def _run_gptq_quantize(
    group_size,
    use_cuda,
    *,
    bits=4,
    sym=False,
    dtype=torch.float16,
    desc_act=False,
    act_group_aware=True,
    static_groups=False,
    cuda_failure=False,
):
    """Run one production GPTQ layer and return its quantization outputs."""
    env_key = "GPTQMODEL_CUDA_BLOCK"
    previous = os.environ.get(env_key)
    os.environ[env_key] = "1" if use_cuda else "0"
    try:
        import gptqmodel.quantization.gptq as gptq_module

        importlib.reload(gptq_module)
        torch.manual_seed(42)
        device = "cuda:0"
        layer = nn.Linear(512, 512, bias=False, dtype=dtype, device=device)
        dense_weight = layer.weight.detach().float().clone()
        qcfg = QuantizeConfig(
            bits=bits,
            group_size=group_size,
            sym=sym,
            desc_act=desc_act,
            act_group_aware=act_group_aware,
            static_groups=static_groups,
            offload_to_disk=False,
            mse=2.0,
            scale_search=ScaleSearchConfig.ACTIVATION,
        )
        quantizer = gptq_module.GPTQ(layer, qcfg=qcfg)
        quantizer.quantizer.configure(perchannel=True)
        calibration = torch.randn(8, 512, dtype=dtype, device=device)
        quantizer.add_batch(calibration, None)
        cuda_launches = 0
        real_cuda_block = gptq_module.gptq_block_cuda

        def counted_cuda_block(*args, **kwargs):
            nonlocal cuda_launches
            cuda_launches += 1
            if cuda_failure:
                raise RuntimeError("deliberate CUDA block failure")
            return real_cuda_block(*args, **kwargs)

        with patch.object(
            gptq_module, "gptq_block_cuda", side_effect=counted_cuda_block
        ):
            output = quantizer.quantize(blocksize=128)
        return dense_weight, output, cuda_launches
    finally:
        if previous is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = previous


def _output_metrics(actual, reference):
    delta = (actual - reference).float()
    reference_fp32 = reference.float()
    return {
        "mae": delta.abs().mean().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "relative_l2": (
            torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference_fp32)
        ).item(),
        "max_abs": delta.abs().max().item(),
    }


def test_gptq_cuda_block_extension_contract_is_registered():
    import gptqmodel.extension as extension_api
    import gptqmodel.utils.gptq_block as block_module

    assert "gptq_block" in extension_api.available_extensions()
    assert (
        block_module._GPTQ_BLOCK_TORCH_OPS_EXTENSION.namespace == "gptqmodel_gptq_block"
    )
    assert block_module._GPTQ_BLOCK_TORCH_OPS_EXTENSION.required_ops == ("quantize",)
    assert block_module._gptq_block_sources() == [
        str(block_module._gptq_block_root() / "gptq_block_cuda.cu")
    ]
    assert "--use_fast_math" not in block_module._gptq_block_cuda_cflags()


def test_gptq_cuda_block_rejects_cpu_before_extension_load():
    import gptqmodel.utils.gptq_block as block_module

    weights = torch.randn(2, 32, dtype=torch.float32)
    hessian_inverse = torch.eye(32, dtype=torch.float32)
    scale = torch.ones(2, 1, dtype=torch.float32)
    zero = torch.zeros(2, 1, dtype=torch.float32)
    with patch.object(block_module, "_extension_api") as extension_api:
        with pytest.raises(ValueError, match="must be CUDA"):
            block_module.gptq_block_cuda(weights, hessian_inverse, scale, zero, 15, 32)
    extension_api.assert_not_called()


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            RawCase(1, 1, 1, 2, "exact_bins", "identity"), id="minimum-single-column"
        ),
        pytest.param(
            RawCase(1, 32, 32, 2, "gaussian", "correlated"),
            id="w2-single-row-correlated",
        ),
        pytest.param(
            RawCase(6, 6, 3, 2, "uniform", "diagonal"), id="short-non-power-group"
        ),
        pytest.param(
            RawCase(3, 31, 1, 3, "alternating_sparse", "ill_conditioned"),
            id="odd-count-sparse",
        ),
        pytest.param(
            RawCase(5, 32, 32, 2, "uniform", "identity"), id="uniform-one-group"
        ),
        pytest.param(
            RawCase(7, 64, 32, 3, "skewed", "correlated"), id="skewed-multiple-groups"
        ),
        pytest.param(RawCase(7, 64, 64, 3, "zeros", "identity"), id="w3-zeros"),
        pytest.param(
            RawCase(5, 64, 32, 5, "uniform", "correlated"), id="w5-correlated"
        ),
        pytest.param(
            RawCase(5, 64, 32, 6, "rounding_ties", "identity"), id="w6-ties"
        ),
        pytest.param(
            RawCase(5, 64, 32, 7, "outliers", "diagonal"), id="w7-outliers"
        ),
        pytest.param(
            RawCase(11, 64, 32, 4, "exact_bins", "identity"), id="w4-exact-bins"
        ),
        pytest.param(
            RawCase(33, 64, 32, 4, "constants", "diagonal"), id="w4-constants"
        ),
        pytest.param(
            RawCase(19, 96, 32, 4, "tie_ulp", "ill_conditioned"), id="w4-odd-ties-tail"
        ),
        pytest.param(
            RawCase(23, 120, 40, 4, "gaussian", "correlated"),
            id="non-power-multi-group",
        ),
        pytest.param(
            RawCase(8, 128, 32, 4, "scale_extremes", "diagonal"), id="w4-scale-extremes"
        ),
        pytest.param(
            RawCase(65, 128, 64, 4, "rounding_ties", "identity"), id="w4-rounding-ties"
        ),
        pytest.param(
            RawCase(96, 128, 128, 8, "outliers", "diagonal"), id="w8-outliers"
        ),
        pytest.param(
            RawCase(13, 128, 64, 4, "heavy_tailed", "correlated"), id="heavy-tailed"
        ),
        pytest.param(
            RawCase(15, 128, 64, 4, "single_outlier", "identity"),
            id="isolated-outliers",
        ),
        pytest.param(
            RawCase(4, 64, 32, 4, "tiny_normals", "identity"),
            id="tiny-normal-signed-zero",
        ),
        pytest.param(
            RawCase(513, 128, 64, 4, "gaussian", "correlated"), id="realistic-odd-rows"
        ),
        pytest.param(
            RawCase(7168, 128, 64, 4, "gaussian", "correlated"),
            id="model-like-wide-projection",
        ),
        pytest.param(
            RawCase(17, 64, 32, 4, "constants", "diagonal", True), id="signed-groupwise"
        ),
        pytest.param(
            RawCase(9, 127, 1, 8, "alternating_sparse", "ill_conditioned", True),
            id="signed-max-inactive-lane-tail",
        ),
    ],
)
@requires_cuda
def test_gptq_cuda_block_matches_eager_reference_across_semantic_cases(case):
    """Bits, shapes, ties, saturation, Hessian spectra, and both formulas match eager math."""
    _run_raw_case(case, seed=17)


@requires_cuda
def test_gptq_cuda_block_randomized_shape_and_value_sweep():
    """Sweep independent seeds across legal tails, group counts, bit widths, and formulas."""
    counts_and_groups = (
        (1, 1),
        (2, 1),
        (7, 1),
        (16, 2),
        (31, 1),
        (32, 4),
        (48, 8),
        (63, 3),
        (64, 16),
        (80, 20),
        (96, 24),
        (112, 16),
        (127, 1),
        (128, 32),
    )
    weight_kinds = ("gaussian", "uniform", "skewed", "heavy_tailed", "outliers")
    hessian_kinds = ("identity", "diagonal", "correlated", "ill_conditioned")
    for seed in range(24):
        count, group_size = counts_and_groups[seed % len(counts_and_groups)]
        _run_raw_case(
            RawCase(
                rows=1 + (seed * 17) % 67,
                count=count,
                group_size=group_size,
                bits=(2, 3, 4, 5, 6, 7, 8)[seed % 7],
                weight_kind=weight_kinds[seed % len(weight_kinds)],
                hessian_kind=hessian_kinds[seed % len(hessian_kinds)],
                groupwise=seed % 3 == 0,
            ),
            seed=10_000 + seed,
        )


@pytest.mark.parametrize(
    "edge",
    [
        "nan-weight",
        "positive-infinite-weight",
        "negative-infinite-weight",
        "zero-scale",
        "negative-scale",
        "nan-scale",
        "infinite-scale",
        "nan-zero",
        "infinite-zero",
        "out-of-range-zero",
        "zero-hessian-diagonal",
        "negative-hessian-diagonal",
        "nan-hessian-diagonal",
        "infinite-hessian-diagonal",
    ],
)
@requires_cuda
def test_gptq_cuda_block_matches_eager_nonfinite_edge_propagation(edge):
    """Accepted FP32 tensors preserve eager NaN/Inf behavior instead of fabricating finite values."""
    import gptqmodel.utils.gptq_block as block_module

    weights = torch.tensor(
        [[0.0, 1.0, -1.0, 0.5]], device="cuda:0", dtype=torch.float32
    )
    hessian_inverse = torch.eye(4, device="cuda:0", dtype=torch.float32)
    scale = torch.ones(1, 1, device="cuda:0", dtype=torch.float32)
    zero = torch.full((1, 1), 8.0, device="cuda:0", dtype=torch.float32)
    if edge == "nan-weight":
        weights[0, 0] = torch.nan
    elif edge == "positive-infinite-weight":
        weights[0, 0] = torch.inf
    elif edge == "negative-infinite-weight":
        weights[0, 0] = -torch.inf
    elif edge == "zero-scale":
        scale.zero_()
    elif edge == "negative-scale":
        scale.fill_(-1.0)
    elif edge == "nan-scale":
        scale.fill_(torch.nan)
    elif edge == "infinite-scale":
        scale.fill_(torch.inf)
    elif edge == "nan-zero":
        zero.fill_(torch.nan)
    elif edge == "infinite-zero":
        zero.fill_(torch.inf)
    elif edge == "out-of-range-zero":
        zero.fill_(1_000.0)
    elif edge == "zero-hessian-diagonal":
        hessian_inverse[0, 0] = 0.0
    elif edge == "negative-hessian-diagonal":
        hessian_inverse[0, 0] = -1.0
    elif edge == "nan-hessian-diagonal":
        hessian_inverse[0, 0] = torch.nan
    elif edge == "infinite-hessian-diagonal":
        hessian_inverse[0, 0] = torch.inf
    else:  # pragma: no cover - protected by the parameter list
        raise AssertionError(edge)

    reference_weights = weights.clone()
    reference_quantized = torch.empty_like(weights)
    reference_errors = torch.empty_like(weights)
    _serial_block(
        reference_weights,
        reference_quantized,
        reference_errors,
        hessian_inverse,
        scale,
        zero,
        15,
        4,
    )
    actual_quantized, actual_errors = block_module.gptq_block_cuda(
        weights,
        hessian_inverse,
        scale,
        zero,
        15,
        4,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual_quantized, reference_quantized, atol=0, rtol=0, equal_nan=True
    )
    torch.testing.assert_close(
        actual_errors, reference_errors, atol=0, rtol=0, equal_nan=True
    )


@pytest.mark.parametrize(
    (
        "bits",
        "group_size",
        "sym",
        "dtype",
        "desc_act",
        "act_group_aware",
        "static_groups",
        "uses_cuda",
    ),
    [
        pytest.param(
            2,
            32,
            False,
            torch.float16,
            False,
            False,
            False,
            True,
            id="w2-g32-no-order-fp16",
        ),
        pytest.param(
            3,
            64,
            False,
            torch.bfloat16,
            True,
            False,
            False,
            True,
            id="w3-g64-desc-act-bf16",
        ),
        pytest.param(
            4, 64, True, torch.float16, False, True, False, True, id="w4-g64-gar-fp16"
        ),
        pytest.param(
            4,
            128,
            False,
            torch.bfloat16,
            False,
            True,
            False,
            True,
            id="w4-g128-maca-bf16",
        ),
        pytest.param(
            5,
            128,
            False,
            torch.bfloat16,
            False,
            False,
            False,
            True,
            id="w5-g128-no-order-bf16",
        ),
        pytest.param(
            6,
            64,
            True,
            torch.float16,
            False,
            True,
            False,
            True,
            id="w6-g64-gar-fp16",
        ),
        pytest.param(
            7,
            128,
            False,
            torch.bfloat16,
            True,
            False,
            False,
            True,
            id="w7-g128-desc-act-bf16",
        ),
        pytest.param(
            8,
            128,
            True,
            torch.bfloat16,
            False,
            False,
            False,
            True,
            id="w8-g128-no-order-bf16",
        ),
        pytest.param(
            4,
            64,
            False,
            torch.float16,
            False,
            False,
            True,
            False,
            id="static-group-eager-fallback",
        ),
        pytest.param(
            4,
            -1,
            False,
            torch.float16,
            False,
            False,
            False,
            False,
            id="no-group-eager-fallback",
        ),
        pytest.param(
            4,
            256,
            False,
            torch.float16,
            False,
            False,
            False,
            False,
            id="group-larger-than-block-eager-fallback",
        ),
    ],
)
@requires_cuda
def test_gptq_cuda_block_matches_production_quantize_and_held_out_outputs(
    bits,
    group_size,
    sym,
    dtype,
    desc_act,
    act_group_aware,
    static_groups,
    uses_cuda,
):
    """The fused production path preserves weights, codes, loss, and independent held-out outputs."""
    options = {
        "bits": bits,
        "sym": sym,
        "dtype": dtype,
        "desc_act": desc_act,
        "act_group_aware": act_group_aware,
        "static_groups": static_groups,
    }
    dense_weight, eager, eager_launches = _run_gptq_quantize(
        group_size, False, **options
    )
    dense_weight_fused, fused, fused_launches = _run_gptq_quantize(
        group_size, True, **options
    )
    eager_weight, eager_scale, eager_zero, eager_g_idx, _, eager_loss, _, _ = eager
    fused_weight, fused_scale, fused_zero, fused_g_idx, _, fused_loss, _, _ = fused

    torch.testing.assert_close(dense_weight, dense_weight_fused, atol=0, rtol=0)
    torch.testing.assert_close(eager_scale, fused_scale, atol=0, rtol=0)
    torch.testing.assert_close(eager_zero, fused_zero, atol=0, rtol=0)
    torch.testing.assert_close(eager_g_idx, fused_g_idx, atol=0, rtol=0)
    torch.testing.assert_close(
        _production_logical_codes(eager_weight, eager_scale, eager_zero, eager_g_idx),
        _production_logical_codes(fused_weight, fused_scale, fused_zero, fused_g_idx),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(eager_weight, fused_weight, atol=0, rtol=0)
    assert eager_launches == 0
    assert (fused_launches > 0) is uses_cuda
    assert isinstance(eager_loss, float) and isinstance(fused_loss, float)
    assert eager_loss == fused_loss

    held_out_generator = torch.Generator(device="cuda:0").manual_seed(1_000_003)
    held_out = torch.randn(
        16, 512, generator=held_out_generator, device="cuda:0", dtype=torch.float32
    )
    dense_output = F.linear(held_out, dense_weight)
    eager_output = F.linear(held_out, eager_weight.float())
    fused_output = F.linear(held_out, fused_weight.float())
    parity = _output_metrics(fused_output, eager_output)
    eager_dense_error = _output_metrics(eager_output, dense_output)
    fused_dense_error = _output_metrics(fused_output, dense_output)

    assert all(
        torch.isfinite(output).all()
        for output in (dense_output, eager_output, fused_output)
    )
    assert parity["mae"] < 2e-4
    assert parity["rmse"] < 5e-4
    # BF16 source weights have a wider representable spacing than FP16. Keep
    # implementation parity far below one source ULP while separately requiring
    # that dense-reference quality does not regress.
    assert parity["relative_l2"] < max(1e-4, torch.finfo(dtype).eps / 32)
    assert parity["max_abs"] < 2e-2
    for metric in ("mae", "rmse", "relative_l2", "max_abs"):
        numerical_slack = max(2e-6, eager_dense_error[metric] * 5e-4)
        assert fused_dense_error[metric] <= eager_dense_error[metric] + numerical_slack

    eager_log_probs = F.log_softmax(eager_output, dim=-1)
    fused_log_probs = F.log_softmax(fused_output, dim=-1)
    kl_divergence = F.kl_div(
        fused_log_probs, eager_log_probs, log_target=True, reduction="batchmean"
    )
    assert abs(kl_divergence.item()) < 1e-6
    assert torch.equal(eager_output.argmax(dim=-1), fused_output.argmax(dim=-1))


@requires_cuda
def test_gptq_cuda_block_launch_failure_falls_back_without_output_drift():
    options = {
        "bits": 4,
        "sym": False,
        "dtype": torch.bfloat16,
        "desc_act": False,
        "act_group_aware": True,
    }
    _, eager, eager_launches = _run_gptq_quantize(64, False, **options)
    _, fallback, fallback_launches = _run_gptq_quantize(
        64, True, cuda_failure=True, **options
    )
    eager_weight, eager_scale, eager_zero, eager_g_idx, _, eager_loss, _, _ = eager
    (
        fallback_weight,
        fallback_scale,
        fallback_zero,
        fallback_g_idx,
        _,
        fallback_loss,
        _,
        _,
    ) = fallback

    assert eager_launches == 0
    assert fallback_launches > 0
    torch.testing.assert_close(eager_weight, fallback_weight, atol=0, rtol=0)
    torch.testing.assert_close(eager_scale, fallback_scale, atol=0, rtol=0)
    torch.testing.assert_close(eager_zero, fallback_zero, atol=0, rtol=0)
    torch.testing.assert_close(eager_g_idx, fallback_g_idx, atol=0, rtol=0)
    assert eager_loss == fallback_loss


@requires_cuda
def test_gptq_cuda_block_repeats_exactly_across_seeds():
    """Ten same-seed launches for each of three seeds must have zero output spread."""
    case = RawCase(32, 128, 64, 4, "gaussian", "correlated")
    for seed in (7, 101, 65_537):
        first_quantized = None
        first_errors = None
        for _ in range(10):
            quantized, errors = _run_raw_case(case, seed)
            if first_quantized is None:
                first_quantized = quantized
                first_errors = errors
            else:
                torch.testing.assert_close(quantized, first_quantized, atol=0, rtol=0)
                torch.testing.assert_close(errors, first_errors, atol=0, rtol=0)


@requires_cuda
def test_gptq_cuda_block_uses_non_default_stream_under_allocator_pressure():
    """Strided wrapper copies and returned outputs remain live on the caller's stream."""
    case = RawCase(96, 128, 64, 4, "gaussian", "correlated")
    pressure = []
    recorded_operands = []
    original_record_stream = torch.Tensor.record_stream

    def capture_record_stream(tensor, stream):
        recorded_operands.append((tensor.is_contiguous(), stream))
        return original_record_stream(tensor, stream)

    def allocate_on_default_stream():
        default_stream = torch.cuda.default_stream("cuda:0")
        with torch.cuda.stream(default_stream):
            pressure.extend(
                torch.empty(128, 128, device="cuda:0", dtype=torch.float32)
                for _ in range(32)
            )

    stream = torch.cuda.Stream(device="cuda:0")
    with (
        torch.cuda.stream(stream),
        patch.object(torch.Tensor, "record_stream", capture_record_stream),
    ):
        _run_raw_case(
            case,
            seed=7,
            strided_operands=True,
            before_synchronize=allocate_on_default_stream,
        )

    stream.synchronize()
    pressure.clear()
    assert len(recorded_operands) == 10
    assert sum(is_contiguous for is_contiguous, _ in recorded_operands) == 6
    assert all(recorded_stream == stream for _, recorded_stream in recorded_operands)


@requires_cuda
def test_gptq_cuda_block_is_deterministic_across_concurrent_streams():
    import gptqmodel.utils.gptq_block as block_module

    case = RawCase(64, 128, 64, 4, "gaussian", "correlated")
    expected = [_run_raw_case(case, seed=seed) for seed in range(4)]
    streams = [torch.cuda.Stream(device="cuda:0") for _ in range(4)]
    actual = []
    for seed, stream in enumerate(streams):
        weights, hessian_inverse, scale, zero = _case_inputs(case, seed=seed)
        with torch.cuda.stream(stream):
            actual.append(
                block_module.gptq_block_cuda(
                    weights, hessian_inverse, scale, zero, 15, 64
                )
            )
    for stream in streams:
        stream.synchronize()

    for (actual_quantized, actual_errors), (expected_quantized, expected_errors) in zip(
        actual, expected
    ):
        torch.testing.assert_close(actual_quantized, expected_quantized, atol=0, rtol=0)
        torch.testing.assert_close(actual_errors, expected_errors, atol=0, rtol=0)


@requires_cuda
def test_gptq_cuda_block_guards_and_restores_the_input_device():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible CUDA devices")

    torch.cuda.set_device(0)
    _run_raw_case(
        RawCase(5, 64, 64, 4, "gaussian", "correlated"),
        seed=808,
        device="cuda:1",
    )
    assert torch.cuda.current_device() == 0


@requires_cuda
def test_gptq_cuda_block_reuses_explicit_output_storage():
    import gptqmodel.utils.gptq_block as block_module

    case = RawCase(16, 64, 64, 4, "gaussian", "correlated")
    weights, hessian_inverse, scale, zero = _case_inputs(case, seed=2026)
    expected_quantized, expected_errors = _run_raw_case(case, seed=2026)
    quantized = torch.full_like(weights, torch.nan)
    errors = torch.full_like(weights, torch.nan)
    recorded_storage = []
    original_record_stream = torch.Tensor.record_stream

    def capture_record_stream(tensor, stream):
        recorded_storage.append(tensor.untyped_storage().data_ptr())
        return original_record_stream(tensor, stream)

    with patch.object(torch.Tensor, "record_stream", capture_record_stream):
        actual_quantized, actual_errors = block_module.gptq_block_cuda(
            weights,
            hessian_inverse,
            scale,
            zero,
            15,
            case.group_size,
            out=(quantized, errors),
        )
    torch.cuda.synchronize(weights.device)

    required_storage = {
        tensor.untyped_storage().data_ptr()
        for tensor in (weights, hessian_inverse, scale, zero, quantized, errors)
    }
    # PyTorch may record internal dispatcher storage while resolving a cold op.
    # Require the six caller-owned allocations exactly once without coupling
    # this lifecycle assertion to framework-internal bookkeeping.
    assert len(required_storage) == 6
    assert all(recorded_storage.count(storage_ptr) == 1 for storage_ptr in required_storage)
    assert actual_quantized.data_ptr() == quantized.data_ptr()
    assert actual_errors.data_ptr() == errors.data_ptr()
    torch.testing.assert_close(actual_quantized, expected_quantized, atol=0, rtol=0)
    torch.testing.assert_close(actual_errors, expected_errors, atol=0, rtol=0)


@pytest.mark.parametrize("strided_operands", [False, True], ids=("contiguous", "strided"))
@requires_cuda
def test_gptq_cuda_block_does_not_mutate_inputs(strided_operands):
    import gptqmodel.utils.gptq_block as block_module

    case = RawCase(9, 128, 64, 4, "outliers", "correlated")
    operands = _case_inputs(case, seed=41, strided_operands=strided_operands)
    snapshots = tuple(tensor.clone() for tensor in operands)

    block_module.gptq_block_cuda(*operands, 15, 64)
    torch.cuda.synchronize(operands[0].device)

    for actual, expected in zip(operands, snapshots):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@requires_cuda
def test_gptq_cuda_block_native_op_rejects_aliases_and_noncontiguous_inputs():
    """The C++ boundary remains fail-closed even when callers bypass the Python wrapper."""
    import gptqmodel.extension as extension_api

    extension_api.load(name="gptq_block")
    op = extension_api.op("gptq_block", "quantize")
    case = RawCase(4, 64, 64, 4, "gaussian", "identity")
    weights, hessian_inverse, scale, zero = _case_inputs(case, seed=99)
    output = torch.empty_like(weights)
    errors = torch.empty_like(weights)

    with pytest.raises(RuntimeError, match="output tensors must not overlap"):
        op(weights, hessian_inverse, scale, zero, 15, 64, False, output, output)
    with pytest.raises(RuntimeError, match="output tensors must not overlap any input"):
        op(weights, hessian_inverse, scale, zero, 15, 64, False, weights, output)

    overlapping_storage = torch.empty(weights.numel() + 1, device="cuda:0")
    overlapping_output = overlapping_storage[:-1].view_as(weights)
    overlapping_errors = overlapping_storage[1:].view_as(weights)
    with pytest.raises(RuntimeError, match="output tensors must not overlap"):
        op(
            weights,
            hessian_inverse,
            scale,
            zero,
            15,
            64,
            False,
            overlapping_output,
            overlapping_errors,
        )

    with pytest.raises(RuntimeError, match="weights must have dtype float32"):
        op(weights.half(), hessian_inverse, scale, zero, 15, 64, False, output, errors)
    with pytest.raises(RuntimeError, match="hessian_inverse must have dtype float32"):
        op(weights, hessian_inverse.half(), scale, zero, 15, 64, False, output, errors)
    with pytest.raises(RuntimeError, match="weights must be rank two"):
        op(weights.unsqueeze(0), hessian_inverse, scale, zero, 15, 64, False, output, errors)
    with pytest.raises(RuntimeError, match="dimensions must be positive"):
        op(
            weights[:0],
            hessian_inverse,
            scale[:0],
            zero[:0],
            15,
            64,
            False,
            output[:0],
            errors[:0],
        )

    oversized_rows = torch.empty(1, 1, device="cuda:0").expand(2**31, 1)
    with pytest.raises(RuntimeError, match="rows must be no greater than"):
        op(
            oversized_rows,
            torch.eye(1, device="cuda:0"),
            torch.ones(1, 1, device="cuda:0"),
            torch.zeros(1, 1, device="cuda:0"),
            15,
            1,
            False,
            torch.empty(1, 1, device="cuda:0"),
            torch.empty(1, 1, device="cuda:0"),
        )

    oversized_weights = torch.empty(4, 129, device="cuda:0")
    oversized_output = torch.empty_like(oversized_weights)
    with pytest.raises(RuntimeError, match="no greater than 128"):
        op(
            oversized_weights,
            torch.eye(129, device="cuda:0"),
            torch.ones(4, 1, device="cuda:0"),
            torch.zeros(4, 1, device="cuda:0"),
            15,
            129,
            False,
            oversized_output,
            torch.empty_like(oversized_output),
        )
    with pytest.raises(RuntimeError, match="group_size must be positive and divide count"):
        op(weights, hessian_inverse, scale, zero, 15, 0, False, output, errors)
    with pytest.raises(RuntimeError, match="group_size must be positive and divide count"):
        op(weights, hessian_inverse, scale, zero, 15, 48, False, output, errors)
    with pytest.raises(RuntimeError, match="maxq must be positive"):
        op(weights, hessian_inverse, scale, zero, 0, 64, False, output, errors)
    with pytest.raises(RuntimeError, match="maxq must be no greater than 255"):
        op(weights, hessian_inverse, scale, zero, 256, 64, False, output, errors)
    with pytest.raises(RuntimeError, match="hessian_inverse must be square"):
        op(weights, hessian_inverse[:-1], scale, zero, 15, 64, False, output, errors)
    with pytest.raises(RuntimeError, match="scale must have shape"):
        op(
            weights,
            hessian_inverse,
            torch.ones(4, 2, device="cuda:0"),
            zero,
            15,
            64,
            False,
            output,
            errors,
        )
    with pytest.raises(RuntimeError, match="zero must match scale shape"):
        op(
            weights,
            hessian_inverse,
            scale,
            torch.zeros(4, 2, device="cuda:0"),
            15,
            64,
            False,
            output,
            errors,
        )
    with pytest.raises(RuntimeError, match="quantized output must match"):
        op(weights, hessian_inverse, scale, zero, 15, 64, False, output[:, :-1], errors)
    with pytest.raises(RuntimeError, match="error output must match"):
        op(weights, hessian_inverse, scale, zero, 15, 64, False, output, errors[:, :-1])
    with pytest.raises(RuntimeError, match="outputs must have dtype float32"):
        op(weights, hessian_inverse, scale, zero, 15, 64, False, output.half(), errors)

    strided_weights = torch.empty(4, 64, device="cuda:0", dtype=torch.float32)[:, ::2]
    contiguous_output = torch.empty(4, 32, device="cuda:0", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="all inputs and outputs must be contiguous"):
        op(
            strided_weights,
            torch.eye(32, device="cuda:0", dtype=torch.float32),
            scale,
            zero,
            15,
            32,
            False,
            contiguous_output,
            torch.empty_like(contiguous_output),
        )


@pytest.mark.skipif(
    not has_gil_disabled(), reason="requires Python free-threading with PYTHON_GIL=0"
)
@requires_cuda
def test_gptq_cuda_block_is_safe_on_process_device_thread_pool():
    """Concurrent production ThreadX workers launch safely across two devices."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible CUDA devices")

    from gptqmodel.utils.gptq_block import prewarm_gptq_block_cuda

    prewarm_gptq_block_cuda()
    case = RawCase(48, 128, 64, 4, "gaussian", "correlated")
    devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    futures = [
        gptqmodel.DEVICE_THREAD_POOL.submit(
            devices[seed % len(devices)],
            _run_raw_case,
            case,
            seed,
            device=devices[seed % len(devices)],
        )
        for seed in range(8)
    ]
    results = [future.result(timeout=120) for future in futures]
    assert len(results) == 8
    assert all(
        torch.isfinite(quantized).all() and torch.isfinite(errors).all()
        for quantized, errors in results
    )


@requires_cuda
def test_gptq_cuda_block_rejects_invalid_inputs_before_launch():
    """Invalid shapes, dtypes, devices, and grouping fail on the host instead of reaching CUDA."""
    import gptqmodel.utils.gptq_block as block_module

    device = torch.device("cuda:0")

    def operands(rows=4, count=64, group_size=64):
        groups = count // group_size
        weights = torch.randn(rows, count, device=device, dtype=torch.float32)
        return (
            weights,
            torch.eye(count, device=device, dtype=torch.float32),
            torch.ones(rows, groups, device=device, dtype=torch.float32),
            torch.zeros(rows, groups, device=device, dtype=torch.float32),
        )

    valid = operands()

    def call(operands_to_use, maxq=15, group_size=64, *, out=None):
        return block_module.gptq_block_cuda(
            operands_to_use[0],
            operands_to_use[1],
            operands_to_use[2],
            operands_to_use[3],
            maxq,
            group_size,
            out=out,
        )

    with patch.object(block_module, "_extension_api") as extension_api:
        zero_rows = (
            torch.empty(0, 64, device=device),
            valid[1],
            torch.empty(0, 1, device=device),
            torch.empty(0, 1, device=device),
        )
        with pytest.raises(ValueError, match="dimensions must be positive"):
            call(zero_rows)
        zero_columns = (
            torch.empty(4, 0, device=device),
            torch.empty(0, 0, device=device),
            torch.empty(4, 0, device=device),
            torch.empty(4, 0, device=device),
        )
        with pytest.raises(ValueError, match="dimensions must be positive"):
            call(zero_columns)
        with pytest.raises(ValueError, match="two-dimensional"):
            call((valid[0][0], *valid[1:]))
        with pytest.raises(ValueError, match="count <= 128"):
            call(operands(count=256))
        oversized_rows = (
            torch.empty(1, 1, device=device).expand(2**31, 1),
            torch.eye(1, device=device),
            torch.ones(1, 1, device=device),
            torch.zeros(1, 1, device=device),
        )
        with pytest.raises(ValueError, match="rows <= 2147483647"):
            call(oversized_rows, group_size=1)
        with pytest.raises(ValueError, match="positive"):
            call(valid, group_size=0)
        with pytest.raises(ValueError, match="must divide"):
            call(valid, group_size=48)

        for operand_index in range(4):
            bad_dtype = list(valid)
            bad_dtype[operand_index] = bad_dtype[operand_index].half()
            with pytest.raises(TypeError, match="float32 weights/Hinv/scale/zero"):
                call(bad_dtype)

        bad_device = list(valid)
        bad_device[3] = bad_device[3].cpu()
        with pytest.raises(ValueError, match="share one device"):
            call(bad_device)

        cpu_operands = tuple(tensor.cpu() for tensor in valid)
        with pytest.raises(ValueError, match="must be CUDA"):
            call(cpu_operands)

        bad_hessian = list(valid)
        bad_hessian[1] = torch.eye(63, device=device)
        with pytest.raises(ValueError, match="hessian_inverse must have shape"):
            call(bad_hessian)

        bad_scale = list(valid)
        bad_scale[2] = torch.ones(4, 2, device=device)
        with pytest.raises(ValueError, match="scale/zero must have shape"):
            call(bad_scale)
        bad_zero = list(valid)
        bad_zero[3] = torch.zeros(4, 2, device=device)
        with pytest.raises(ValueError, match="scale/zero must have shape"):
            call(bad_zero)

        with pytest.raises(ValueError, match="maxq must be positive"):
            call(valid, maxq=0)
        with pytest.raises(ValueError, match="maxq <= 255"):
            call(valid, maxq=256)
        for invalid_maxq in (15.5, True):
            with pytest.raises(TypeError, match="maxq must be an integer"):
                call(valid, maxq=invalid_maxq)
        for invalid_group_size in (64.0, True):
            with pytest.raises(TypeError, match="group_size must be an integer"):
                call(valid, group_size=invalid_group_size)
        with pytest.raises(TypeError, match="groupwise must be a bool"):
            block_module.gptq_block_cuda(*valid, 15, 64, groupwise=1)

        with pytest.raises(ValueError, match="out tensors must match"):
            call(
                valid,
                out=(
                    torch.empty(4, 63, device=device),
                    torch.empty(4, 64, device=device),
                ),
            )
        with pytest.raises(ValueError, match="out tensors must match"):
            call(
                valid,
                out=(
                    torch.empty(4, 64, device=device),
                    torch.empty(4, 63, device=device),
                ),
            )
        with pytest.raises(TypeError, match="out tensors must have dtype float32"):
            call(
                valid,
                out=(
                    torch.empty(4, 64, device=device, dtype=torch.float16),
                    torch.empty(4, 64, device=device),
                ),
            )
        with pytest.raises(TypeError, match="out tensors must have dtype float32"):
            call(
                valid,
                out=(
                    torch.empty(4, 64, device=device),
                    torch.empty(4, 64, device=device, dtype=torch.float16),
                ),
            )
        noncontiguous_out = torch.empty(64, 4, device=device).T
        with pytest.raises(ValueError, match="out tensors must be contiguous"):
            call(valid, out=(noncontiguous_out, torch.empty(4, 64, device=device)))
        with pytest.raises(ValueError, match="out tensors must be contiguous"):
            call(valid, out=(torch.empty(4, 64, device=device), noncontiguous_out))
        with pytest.raises(ValueError, match="must not alias"):
            call(valid, out=(valid[0], torch.empty_like(valid[0])))
        shared_out = torch.empty_like(valid[0])
        with pytest.raises(ValueError, match="must not alias"):
            call(valid, out=(shared_out, shared_out))
        if torch.cuda.device_count() > 1:
            with pytest.raises(ValueError, match="share the weights device"):
                call(
                    valid,
                    out=(
                        torch.empty(4, 64, device="cuda:1"),
                        torch.empty(4, 64, device=device),
                    ),
                )
            with pytest.raises(ValueError, match="share the weights device"):
                call(
                    valid,
                    out=(
                        torch.empty(4, 64, device=device),
                        torch.empty(4, 64, device="cuda:1"),
                    ),
                )

    extension_api.assert_not_called()


def test_gptq_cuda_availability_handles_runtime_failures():
    import gptqmodel.utils.gptq_block as block_module

    with patch.object(torch.cuda, "is_available", return_value=False):
        assert not block_module.gptq_block_cuda_supported()
        assert not block_module.gptq_block_cuda_available()
        assert (
            block_module.gptq_block_cuda_error()
            == "GPTQ CUDA block quantization requires CUDA."
        )

    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.version, "hip", "6.4"),
        patch.object(block_module, "_extension_api") as extension_api,
    ):
        assert not block_module.gptq_block_cuda_supported()
        assert not block_module.gptq_block_cuda_available()
        assert (
            block_module.gptq_block_cuda_error()
            == "GPTQ CUDA block quantization requires NVIDIA CUDA; ROCm is not supported."
        )
    extension_api.assert_not_called()

    unavailable_api = type(
        "UnavailableExtension", (), {"is_available": lambda self, name: False}
    )()
    with patch.object(block_module, "_extension_api", return_value=unavailable_api):
        assert not block_module.gptq_block_cuda_available()

    available_api = type(
        "AvailableExtension",
        (),
        {"is_available": lambda self, name: name == "gptq_block"},
    )()
    with (
        patch.object(block_module, "_extension_api", return_value=available_api),
        patch.object(torch.cuda, "is_available", return_value=True),
    ):
        assert block_module.gptq_block_cuda_available()


def test_gptq_cuda_extension_prewarm_and_error_delegate_to_shared_api():
    import gptqmodel.utils.gptq_block as block_module

    class ExtensionApi:
        def load(self, *, name):
            assert name == "gptq_block"
            return {"gptq_block": True}

        def error(self, name):
            assert name == "gptq_block"
            return "deliberate build error"

    with (
        patch.object(block_module, "_extension_api", return_value=ExtensionApi()),
        patch.object(torch.cuda, "is_available", return_value=True),
    ):
        assert block_module.prewarm_gptq_block_cuda()
        assert block_module.gptq_block_cuda_error() == "deliberate build error"


def test_gptq_cuda_op_handle_is_cached_after_thread_safe_initialization():
    import gptqmodel.utils.gptq_block as block_module

    sentinel = object()

    class ExtensionApi:
        def __init__(self):
            self.calls = 0

        def op(self, name, op_name):
            assert (name, op_name) == ("gptq_block", "quantize")
            self.calls += 1
            return sentinel

    extension_api = ExtensionApi()
    with (
        patch.object(block_module, "_extension_api", return_value=extension_api),
        patch.object(block_module, "_GPTQ_BLOCK_OP", None),
    ):
        assert block_module._gptq_block_op() is sentinel
        assert block_module._gptq_block_op() is sentinel
    assert extension_api.calls == 1


def test_gptq_cuda_op_handle_rechecks_cache_after_waiting_for_initializer():
    import gptqmodel.utils.gptq_block as block_module

    sentinel = object()

    class InitializingLock:
        def __enter__(self):
            block_module._GPTQ_BLOCK_OP = sentinel

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    with (
        patch.object(block_module, "_GPTQ_BLOCK_OP", None),
        patch.object(block_module, "_GPTQ_BLOCK_OP_INIT_LOCK", InitializingLock()),
        patch.object(block_module, "_extension_api") as extension_api,
    ):
        assert block_module._gptq_block_op() is sentinel
    extension_api.assert_not_called()


@requires_cuda
def test_gptq_cuda_block_env_disable_falls_back():
    """With GPTQMODEL_CUDA_BLOCK=0 the production path does not need the CUDA kernel."""
    _, outputs, cuda_launches = _run_gptq_quantize(128, False)
    assert cuda_launches == 0
    assert outputs[0] is not None
    assert outputs[0].numel() > 0


@pytest.mark.skipif(
    os.environ.get("GPTQMODEL_CUDA_FRESH_PROBE") != "1", reason="fresh-process probe"
)
@requires_cuda
def test_gptq_cuda_block_fresh_process_probe():
    seed = int(os.environ.get("GPTQMODEL_CUDA_FRESH_SEED", "23"))
    _run_raw_case(RawCase(8, 64, 64, 4, "rounding_ties", "identity"), seed=seed)


@pytest.mark.skipif(
    os.environ.get("GPTQMODEL_CUDA_COLD_THREAD_PROBE") != "1",
    reason="cold-thread probe",
)
@pytest.mark.skipif(
    not has_gil_disabled(), reason="requires Python free-threading with PYTHON_GIL=0"
)
@requires_cuda
def test_gptq_cuda_block_cold_initialization_probe():
    """Two first-use worker calls may contend for one extension build without racing it."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible CUDA devices")

    case = RawCase(8, 64, 64, 4, "gaussian", "correlated")
    devices = (torch.device("cuda:0"), torch.device("cuda:1"))
    futures = [
        gptqmodel.DEVICE_THREAD_POOL.submit(
            device, _run_raw_case, case, seed, device=device
        )
        for seed, device in enumerate(devices)
    ]
    results = [future.result(timeout=180) for future in futures]
    assert all(
        torch.isfinite(quantized).all() and torch.isfinite(errors).all()
        for quantized, errors in results
    )


@pytest.mark.skipif(
    not has_gil_disabled(), reason="requires Python free-threading with PYTHON_GIL=0"
)
@requires_cuda
def test_gptq_cuda_block_initializes_cleanly_in_fresh_processes():
    """CUDA compile/cache initialization is stable in independent free-threaded processes."""
    test_id = f"{__file__}::test_gptq_cuda_block_fresh_process_probe"
    environment = os.environ.copy()
    environment["GPTQMODEL_CUDA_FRESH_PROBE"] = "1"
    for seed in (23, 101, 65_537):
        environment["GPTQMODEL_CUDA_FRESH_SEED"] = str(seed)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", test_id],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(
    not has_gil_disabled(), reason="requires Python free-threading with PYTHON_GIL=0"
)
@requires_cuda
def test_gptq_cuda_block_cold_initialization_is_thread_safe():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible CUDA devices")

    test_id = f"{__file__}::test_gptq_cuda_block_cold_initialization_probe"
    environment = os.environ.copy()
    environment["GPTQMODEL_CUDA_COLD_THREAD_PROBE"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", test_id],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
