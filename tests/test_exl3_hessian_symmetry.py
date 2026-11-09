import pytest
import torch

from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import (
    EXL3_HESSIAN_NUMERICAL_CONTRACT,
    EXL3_HESSIAN_SYMMETRY_CONTRACT,
    regularize_and_transform_hessian_,
    restore_hessian_symmetry_,
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
