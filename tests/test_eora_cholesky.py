import torch

from gptqmodel.eora.eora import eora_compute_lora


def _make_spd(size: int) -> torch.Tensor:
    torch.manual_seed(0)
    acts = torch.randn(size * 3, size, dtype=torch.float32)
    cov = acts.T @ acts / float(acts.shape[0])
    cov = ((cov + cov.T) * 0.5).to(dtype=torch.float64)
    cov.diagonal().add_(1e-2)
    return cov


def _weighted_error(delta: torch.Tensor, cov: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    residual = delta - (B @ A)
    factor = torch.linalg.cholesky(cov).to(dtype=delta.dtype)
    return torch.linalg.matrix_norm(residual @ factor)


def test_eora_cholesky_fast_path_preserves_weighted_objective():
    torch.manual_seed(1)
    cols = 24
    rows = 32
    rank = 6
    cov = _make_spd(cols)
    delta = torch.randn(rows, cols, dtype=torch.float32)

    A_eigh, B_eigh = eora_compute_lora(
        w_wq_delta=delta,
        name="test",
        eigen_scaling_diag_matrix=cov,
        rank=rank,
        dtype=torch.float32,
        device=torch.device("cpu"),
        use_cholesky=False,
    )
    A_chol, B_chol = eora_compute_lora(
        w_wq_delta=delta,
        name="test",
        eigen_scaling_diag_matrix=cov,
        rank=rank,
        dtype=torch.float32,
        device=torch.device("cpu"),
        use_cholesky=True,
    )

    err_eigh = _weighted_error(delta, cov, A_eigh, B_eigh)
    err_chol = _weighted_error(delta, cov, A_chol, B_chol)
    torch.testing.assert_close(err_chol, err_eigh, rtol=1e-5, atol=1e-5)

    reconstruction_eigh = B_eigh @ A_eigh
    reconstruction_chol = B_chol @ A_chol
    rel_reconstruction_diff = (
        torch.linalg.matrix_norm(reconstruction_chol - reconstruction_eigh)
        / torch.linalg.matrix_norm(reconstruction_eigh).clamp_min(1e-12)
    )
    assert rel_reconstruction_diff < 1e-2


def test_eora_cholesky_fast_path_falls_back_for_non_spd_covariance():
    torch.manual_seed(2)
    cols = 4
    rows = 8
    rank = 2
    cov = torch.diag(torch.tensor([2.0, 1.0, 0.5, -0.01], dtype=torch.float64))
    delta = torch.randn(rows, cols, dtype=torch.float32)

    A_eigh, B_eigh = eora_compute_lora(
        w_wq_delta=delta,
        name="test_fallback",
        eigen_scaling_diag_matrix=cov,
        rank=rank,
        dtype=torch.float32,
        device=torch.device("cpu"),
        use_cholesky=False,
    )
    A_chol, B_chol = eora_compute_lora(
        w_wq_delta=delta,
        name="test_fallback",
        eigen_scaling_diag_matrix=cov,
        rank=rank,
        dtype=torch.float32,
        device=torch.device("cpu"),
        use_cholesky=True,
    )

    torch.testing.assert_close(B_chol @ A_chol, B_eigh @ A_eigh, rtol=1e-5, atol=1e-5)
