import struct

import pytest
import torch

from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import (
    EXL3_HESSIAN_NUMERICAL_CONTRACT,
    EXL3_HESSIAN_SYMMETRY_CONTRACT,
    regularize_and_transform_hessian_,
    restore_hessian_symmetry_,
)


def _untiled_restore_hessian_symmetry(hessian: torch.Tensor, quant_args: dict) -> None:
    """The pre-tiling reference implementation, kept as the bitwise oracle."""

    symmetric = (hessian + hessian.transpose(0, 1)) * 0.5
    correction_max_abs = float((hessian - symmetric).abs().max().item())
    hessian.copy_(symmetric)
    quant_args["hessian_symmetry_restoration"] = (
        "mean-with-transpose-fp64"
        if hessian.dtype == torch.float64
        else "mean-with-transpose-fp32"
    )
    quant_args["hessian_symmetry_correction_max_abs"] = correction_max_abs


def _bit_view(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(
        torch.int64 if tensor.dtype == torch.float64 else torch.int32
    )


def _float_bits(value: float) -> bytes:
    return struct.pack(">d", value)


@pytest.mark.parametrize(
    "size,tile",
    [
        (1, 8),
        (7, 8),
        (16, 8),
        (63, 64),
        (64, 64),
        (65, 64),
        (129, 64),
        (200, 64),
        (256, 1024),
        (1031, 1024),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tiled_restore_is_bitwise_identical_to_untiled(size, tile, dtype) -> None:
    torch.manual_seed(size * 131 + tile)
    base = torch.randn(size, size, dtype=dtype)
    # Deliberately non-symmetric, with exact cancellation so signed zeros are
    # exercised as well as ordinary rounding.
    if size > 1:
        base[0, 1] = -base[1, 0]
        base[size - 1, 0] = 0.0
        base[0, size - 1] = -0.0

    expected = base.clone()
    expected_args: dict = {}
    _untiled_restore_hessian_symmetry(expected, expected_args)

    actual = base.clone()
    actual_args: dict = {}
    restore_hessian_symmetry_(actual, actual_args, tile=tile)

    assert torch.equal(_bit_view(actual), _bit_view(expected))
    assert actual_args["hessian_symmetry_restoration"] == (
        expected_args["hessian_symmetry_restoration"]
    )
    assert _float_bits(actual_args["hessian_symmetry_correction_max_abs"]) == (
        _float_bits(expected_args["hessian_symmetry_correction_max_abs"])
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tiled_restore_is_bitwise_identical_to_untiled_on_cuda(dtype) -> None:
    size, tile = 1031, 1024
    torch.manual_seed(4818)
    base = torch.randn(size, size, dtype=dtype, device="cuda:0")

    expected = base.clone()
    expected_args: dict = {}
    _untiled_restore_hessian_symmetry(expected, expected_args)

    actual = base.clone()
    actual_args: dict = {}
    restore_hessian_symmetry_(actual, actual_args, tile=tile)

    assert torch.equal(_bit_view(actual), _bit_view(expected))
    assert _float_bits(actual_args["hessian_symmetry_correction_max_abs"]) == (
        _float_bits(expected_args["hessian_symmetry_correction_max_abs"])
    )


def test_restore_hessian_symmetry_averages_without_diagonal_damping() -> None:
    hessian = torch.tensor(
        [[2.0, 0.75, -0.2], [0.25, 3.0, 0.6], [0.4, 0.2, 1.5]],
        dtype=torch.float32,
    )
    expected = (hessian + hessian.T) * 0.5
    expected_diagonal = torch.diagonal(hessian).clone()
    expected_correction = float((hessian - expected).abs().max().item())
    quant_args = {}

    restore_hessian_symmetry_(hessian, quant_args)

    assert torch.equal(hessian, hessian.T)
    assert torch.equal(hessian, expected)
    assert torch.equal(torch.diagonal(hessian), expected_diagonal)
    assert quant_args["hessian_symmetry_restoration"] == "mean-with-transpose-fp32"
    assert quant_args["hessian_symmetry_correction_max_abs"] == pytest.approx(
        expected_correction
    )


def test_restore_hessian_symmetry_rejects_non_square_input() -> None:
    with pytest.raises(ValueError, match="square matrix"):
        restore_hessian_symmetry_(torch.zeros((2, 3)), {})


def test_regularized_hessian_congruence_uses_fp64_without_extra_damping() -> None:
    torch.manual_seed(787)
    size = 256
    vector = torch.randn(size, dtype=torch.float32)
    hessian = torch.outer(vector, vector)
    hessian *= 0.0072 / hessian.diagonal().mean()
    original = hessian.clone()
    signs = torch.where(
        torch.arange(size) % 3 == 0,
        torch.tensor(-1.0),
        torch.tensor(1.0),
    ).unsqueeze(1)
    diag_mean = hessian.diagonal().mean()
    quant_args = {"sigma_reg": 0.025}

    regularize_and_transform_hessian_(
        hessian,
        su=signs,
        diag_mean=diag_mean,
        quant_args=quant_args,
    )

    assert torch.equal(original, original.T)
    assert torch.equal(hessian, hessian.T)
    torch.linalg.cholesky(hessian)
    assert (
        quant_args["hessian_numerical_contract"]
        == EXL3_HESSIAN_NUMERICAL_CONTRACT
    )
    assert (
        quant_args["hessian_symmetry_restoration"]
        == EXL3_HESSIAN_SYMMETRY_CONTRACT
    )
    assert quant_args["hessian_transform_compute_dtype"] == "torch.float64"
    assert quant_args["hessian_storage_dtype"] == "torch.float32"
    assert quant_args["hessian_regularization_placement"] == (
        "before-fp64-congruence"
    )
    assert quant_args["hessian_regularization_diagonal_addend"] == pytest.approx(
        0.025 * diag_mean.item()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp64_hessian_congruence_preserves_margin_at_expert_down_shape() -> None:
    torch.manual_seed(787)
    device = torch.device("cuda:0")
    size = 2048
    vector = torch.randn(size, dtype=torch.float32, device=device)
    hessian = torch.outer(vector, vector)
    hessian *= 0.007227 / hessian.diagonal().mean()
    diag_mean = hessian.diagonal().mean()
    signs = (torch.randn(size, device=device).sign() + 1e-5).sign().unsqueeze(1)
    quant_args = {"sigma_reg": 0.025}

    regularize_and_transform_hessian_(
        hessian,
        su=signs,
        diag_mean=diag_mean,
        quant_args=quant_args,
    )

    _factor, info = torch.linalg.cholesky_ex(hessian, check_errors=False)
    assert info.item() == 0
    assert (
        quant_args["hessian_numerical_contract"]
        == EXL3_HESSIAN_NUMERICAL_CONTRACT
    )
    assert (
        quant_args["hessian_symmetry_restoration"]
        == EXL3_HESSIAN_SYMMETRY_CONTRACT
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("size", [8192])
def test_regularized_congruence_extra_peak_is_one_working_copy(size) -> None:
    """The fp64 symmetrization must not add a second C^2 fp64 working set.

    Before tiling, ``restore_hessian_symmetry_`` materialized
    ``(H + H.T) * 0.5`` plus ``(H - symmetric).abs()``, so the congruence
    peaked at ~4x the fp64 C^2 working copy. The tiled form peaks at ~1x plus
    one tile pair, which is what this bound encodes.
    """

    torch.manual_seed(4818)
    device = torch.device("cuda:0")
    samples = torch.randn(2 * size, size, device=device)
    hessian = samples.T @ samples / (2 * size)
    del samples
    signs = (torch.randn(size, device=device).sign() + 1e-5).sign().unsqueeze(1)
    diag_mean = hessian.diagonal().mean()

    fp64_bytes = size * size * 8
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    regularize_and_transform_hessian_(
        hessian,
        su=signs,
        diag_mean=diag_mean,
        quant_args={"sigma_reg": 0.025},
    )
    torch.cuda.synchronize(device)
    extra_peak = torch.cuda.max_memory_allocated(device) - before

    limit = 1.2 * fp64_bytes
    assert extra_peak <= limit, (
        f"EXL3 fp64 congruence extra peak {extra_peak / 2**20:.1f} MiB exceeds "
        f"{limit / 2**20:.1f} MiB (1.2x the {fp64_bytes / 2**20:.1f} MiB fp64 "
        f"C^2 working copy) for C={size}; the symmetrization is holding "
        "untiled C^2 temporaries again"
    )
