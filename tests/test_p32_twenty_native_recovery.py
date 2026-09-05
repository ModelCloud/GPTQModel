"""CPU algebra only: no native deployment, GPU, snapshot or model-quality claim."""
import math

import pytest
import torch

from scripts.p32_twenty.native_recovery import (
    NativeOperator,
    alternating_recovery,
    capture_residual,
    fit_output_residual,
    fit_sparse_residual,
    storage_cost,
)


def data(m=9, k=5, n=4):
    g = torch.Generator(device="cpu").manual_seed(7)
    return torch.randn(m, k, generator=g, dtype=torch.float64), torch.randn(m, n, generator=g, dtype=torch.float64)


@pytest.mark.parametrize("geometry", [(9, 5, 4), (3, 7, 5)])
@pytest.mark.parametrize("rank", [0, 1, 2, 20])
@pytest.mark.parametrize("singular", [False, True])
def test_optimal_output_objective(geometry, rank, singular):
    x, z = data(*geometry)
    if singular:
        x[:, -1] = x[:, 0]
    result = fit_output_residual(x, z, rank)
    # Independent projection oracle: truncate projected outputs, not coefficients.
    projected = x @ torch.linalg.pinv(x) @ z
    u, s, v = torch.linalg.svd(projected, full_matrices=False)
    oracle = (u[:, :rank] * s[:rank]) @ v[:rank]
    torch.testing.assert_close(result(x), oracle, atol=2e-12, rtol=2e-12)
    assert result.a.shape[1] <= rank
    assert result.squared_error == pytest.approx(float((z - oracle).square().sum()), abs=1e-11)
    torch.testing.assert_close(result.dense(), torch.linalg.pinv(x) @ oracle, atol=2e-12, rtol=2e-12)


def test_weight_svd_is_wrong_objective():
    x = torch.diag(torch.tensor([100., 1.], dtype=torch.float64))
    d = torch.diag(torch.tensor([1., 2.], dtype=torch.float64))
    z = x @ d
    fit = fit_output_residual(x, z, 1)
    assert fit.squared_error == pytest.approx(4.)
    u, s, v = torch.linalg.svd(d)
    weight_only = (u[:, :1] * s[:1]) @ v[:1]
    assert float((z - x @ weight_only).square().sum()) == 10000.


def test_zero_and_numerical_rank():
    x, z = data()
    x.zero_()
    fit = fit_output_residual(x, z, 10)
    assert fit.activation_rank == 0
    assert torch.count_nonzero(fit(x)) == 0
    x = torch.diag(torch.tensor([1., 1e-12], dtype=torch.float64))
    assert fit_output_residual(x, x, 2, rcond=1e-10).activation_rank == 1
    assert fit_output_residual(x, x, 2, rcond=0).activation_rank == 2


def test_actual_native_activation_error_is_captured():
    x, _ = data(8, 4, 4)
    x = x.float()
    w = torch.eye(4)
    calls = []

    def native(a):
        assert a.dtype == torch.float32
        calls.append(1)
        return a.round() @ w + 0.125  # activation and epilogue error, exact weights

    z = capture_residual(x, lambda a: a @ w, native)
    expected = x.double() - (x.round() + 0.125).double()
    torch.testing.assert_close(z, expected)
    fit = fit_output_residual(x, z, 4)
    assert calls == [1] and torch.count_nonzero(z) > 0
    assert fit.squared_error < float(z.square().sum())


def test_sparse_output_pursuit_and_refit():
    x = torch.eye(4, dtype=torch.float64)
    z = torch.tensor([[0., 4.], [0., 0.], [-3., 0.], [0., 0.]], dtype=torch.float64)
    s = fit_sparse_residual(x, z, 2)
    torch.testing.assert_close(s(x), z)
    torch.testing.assert_close(s.dense(), z)
    assert s.indices.shape == (2, 2)
    empty = fit_sparse_residual(x * 0, z, 8)
    assert empty.values.numel() == 0
    assert fit_sparse_residual(x, z * 0, 8).values.numel() == 0
    assert fit_sparse_residual(x, z, 0).values.numel() == 0
    # Correlated columns require refitting earlier selected coefficients.
    x = torch.tensor([[1., 1.], [0., 1.], [0., 0.]], dtype=torch.float64)
    z = torch.tensor([[1.], [2.], [0.]], dtype=torch.float64)
    torch.testing.assert_close(fit_sparse_residual(x, z, 2)(x), z)


def test_alternation_storage_and_input_ownership():
    x, _ = data(9, 5, 4)
    w = torch.arange(20, dtype=torch.float64).reshape(5, 4) / 13
    x0, w0 = x.clone(), w.clone()
    targets, native_calls = [], []

    def quantize(target):
        targets.append(target.clone())
        q = target.round()

        def forward(a):
            native_calls.append(1)
            return a.round() @ q

        return NativeOperator(forward, {"codes": 10, "scales": 16, "padding": 6}, "CPU test callable")

    steps = alternating_recovery(x, lambda a: a @ w, w, quantize, 2, iterations=3, sparse_nnz=2)
    assert len(steps) == len(native_calls) == 3
    torch.testing.assert_close(targets[0], w)
    for i, step in enumerate(steps):
        z = x @ w - step.native.forward(x)
        torch.testing.assert_close(step.low_rank.dense(), fit_output_residual(x, z, 2).dense())
        assert step.squared_error == pytest.approx(float((z - step.low_rank(x) - step.sparse(x)).square().sum()))
        if i < 2:
            torch.testing.assert_close(targets[i + 1], w - step.low_rank.dense() - step.sparse.dense())
    torch.testing.assert_close(x, x0)
    torch.testing.assert_close(w, w0)
    cost = storage_cost(steps[0], extra_bytes={"header": 11})
    assert cost["total_bytes"] == 32 + 2 * (5 + 4) * 8 + 2 * (16 + 8) + 11
    assert cost["effective_bpw"] == cost["total_bytes"] * 8 / 20
    assert storage_cost(steps[0])["total_bytes"] == cost["total_bytes"] - 11
    steps[0].native.storage_bytes = {"bad": -1}
    with pytest.raises(ValueError):
        storage_cost(steps[0])
    assert alternating_recovery(x, lambda a: a @ w, w, quantize, 0, iterations=0) == []


@pytest.mark.parametrize("bad", [-1, True, 1.5])
def test_invalid_counts(bad):
    x, z = data()
    with pytest.raises(ValueError):
        fit_output_residual(x, z, bad)
    with pytest.raises(ValueError):
        fit_sparse_residual(x, z, bad)


@pytest.mark.parametrize("bad", [torch.ones(2), torch.ones(0, 2), torch.ones(2, 2, dtype=torch.int64),
                                  torch.full((2, 2), math.nan), torch.full((2, 2), math.inf)])
def test_invalid_matrix(bad):
    with pytest.raises(ValueError):
        fit_output_residual(bad, torch.ones(2, 2), 1)


@pytest.mark.parametrize("rcond", [-1., 1., math.inf, math.nan])
def test_invalid_cutoff(rcond):
    x, z = data()
    with pytest.raises(ValueError):
        fit_output_residual(x, z, 1, rcond=rcond)


def test_invalid_geometry_and_callbacks():
    x, z = data()
    with pytest.raises(ValueError):
        fit_output_residual(x, z[:2], 1)
    with pytest.raises(ValueError):
        fit_sparse_residual(x, z[:2], 1)
    with pytest.raises(ValueError):
        fit_sparse_residual(x, z, 100)
    with pytest.raises(ValueError):
        capture_residual(x, lambda a: z, lambda a: z[:, :1])
    with pytest.raises(ValueError):
        capture_residual(x, lambda a: z[:1], lambda a: z[:1])
    with pytest.raises(ValueError):
        alternating_recovery(x, lambda a: z, x, None, 1)
    with pytest.raises(ValueError):
        alternating_recovery(x, lambda a: z, torch.ones(5, 4),
                             lambda w: NativeOperator(lambda a: z[:1], {}, "invalid"), 1)
