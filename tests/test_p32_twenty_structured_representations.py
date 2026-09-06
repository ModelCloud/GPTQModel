"""CPU algebra fixtures only: no checkpoint/model-quality or performance evidence."""

import pytest
import torch

from scripts.p32_twenty.structured_representations import (
    encode_sparse_walsh,
    fit_additive_codebook,
    fit_tensor_product_codebook,
    fit_tile_basis,
    orthonormal_walsh,
)


def random(shape):
    return torch.randn(
        shape,
        generator=torch.Generator(device="cpu").manual_seed(7),
        dtype=torch.float64,
    )


def test_additive_projection_and_lookup():
    q = random((4, 8, 2))
    saved = q.clone()
    fit = fit_additive_codebook(q)
    residual = q - fit.decode()
    torch.testing.assert_close(
        residual.mean(0), torch.zeros(8, 2, dtype=q.dtype), atol=1e-14, rtol=0
    )
    torch.testing.assert_close(
        residual.mean(1), torch.zeros(4, 2, dtype=q.dtype), atol=1e-14, rtol=0
    )
    exact = random((4, 1, 2)) + random((1, 8, 2))
    torch.testing.assert_close(fit_additive_codebook(exact).decode(), exact)
    i, j = torch.tensor([[0, 3], [2, 1]]), torch.tensor([[7, 0], [3, 4]])
    torch.testing.assert_close(fit.lookup(i, j), fit.decode()[i, j])
    assert torch.equal(saved, q)
    report = fit.storage(64)
    assert report["total_bytes"] == (4 + 8) * 2 * 8 + 48
    assert report["effective_bpw"] == report["total_bytes"] * 8 / 64


@pytest.mark.parametrize("rank", [1, 2, 4])
def test_tensor_product_optimal_residual_and_lookup(rank):
    q = random((5, 6, 2))
    fit = fit_tensor_product_codebook(q, rank)
    s = torch.linalg.svdvals(q.permute(2, 0, 1))
    torch.testing.assert_close(
        (q - fit.decode()).square().sum(), s[:, rank:].square().sum()
    )
    i, j = torch.tensor([[4, 0], [2, 1]]), torch.tensor([[5, 0], [3, 2]])
    torch.testing.assert_close(fit.lookup(i, j), fit.decode()[i, j])
    torch.testing.assert_close(
        fit.lookup(torch.tensor(2), torch.tensor(3)), fit.decode()[2, 3]
    )
    a, b = random((2, 5, rank)), random((2, rank, 6))
    low_rank = (a @ b).permute(1, 2, 0)
    torch.testing.assert_close(
        fit_tensor_product_codebook(low_rank, rank).decode(), low_rank
    )
    assert fit.storage(60)["total_bytes"] == 2 * rank * (5 + 6) * 8 + 48


@pytest.mark.parametrize("ternary", [False, True])
def test_basis_projection_prefixes_padding_and_storage(ternary):
    x = random((3, 11)).T  # noncontiguous and every row has a partial tile
    saved = x.clone()
    last = x.square().sum()
    for rank in (1, 2, 4):
        fit = fit_tile_basis(x, rank=rank, tile_size=8, ternary=ternary)
        assert fit.decode().shape == x.shape
        assert set(fit.codes.unique().tolist()) <= ({-1, 0, 1} if ternary else {-1, 1})
        loss = (x - fit.decode()).square().sum()
        assert loss <= last + 1e-12
        last = loss
        torch.testing.assert_close(
            fit.scales[:, :1], fit_tile_basis(x, 1, 8, ternary).scales
        )
        bits = 2 if ternary else 1
        report = fit.storage()
        assert report["total_bytes"] == 11 * rank * bits + 11 * rank * 8 + 48
        assert (
            report["tensor_payload_bytes"] == fit.codes.numel() + fit.scales.numel() * 8
        )
    assert torch.equal(x, saved)
    zero = fit_tile_basis(torch.zeros(2, 9), rank=4, tile_size=8, ternary=ternary)
    assert not zero.decode().any()
    assert torch.isfinite(zero.scales).all()


def test_signed_exact_and_ternary_optimum_support():
    x = torch.tensor([[3.0, -3.0, 3.0, -3.0, 3.0]], dtype=torch.float64)
    torch.testing.assert_close(fit_tile_basis(x, 1, 8).decode(), x)
    x = torch.tensor([[0.0, -4.0, 0.0, 4.0, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(fit_tile_basis(x, 1, 8, True).decode(), x)
    # Exhaust all 3^4 possible ternary directions, fitting their optimal scalar.
    x = torch.tensor([[0.1, -2.0, 0.9, 4.0]], dtype=torch.float64)
    planes = torch.cartesian_prod(*[torch.tensor([-1.0, 0.0, 1.0], dtype=x.dtype)] * 4)
    scale = (planes * x).sum(1) / planes.square().sum(1).clamp_min(1)
    loss = (x - scale[:, None] * planes).square().sum(1).min()
    torch.testing.assert_close(
        (x - fit_tile_basis(x, 1, 4, True).decode()).square().sum(), loss
    )


@pytest.mark.parametrize("size", [1, 2, 16, 32, 64])
def test_walsh_orthonormal_and_roundtrip(size):
    x = random((2, size))
    saved = x.clone()
    h = torch.ones(1, 1, dtype=x.dtype)
    while h.shape[0] < size:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    h /= size**0.5
    torch.testing.assert_close(orthonormal_walsh(x), x @ h)
    torch.testing.assert_close(orthonormal_walsh(orthonormal_walsh(x)), x)
    torch.testing.assert_close(orthonormal_walsh(x).square().sum(), x.square().sum())
    torch.testing.assert_close(encode_sparse_walsh(x, size, size).decode(), x)
    assert torch.equal(x, saved)


def test_sparse_walsh_parseval_tail_ties_and_storage():
    x = random((3, 16))
    sparse = encode_sparse_walsh(x, 8, 3)
    coeff = orthonormal_walsh(x.reshape(-1, 8))
    discarded = coeff.clone().scatter_(1, sparse.indices, 0)
    torch.testing.assert_close(
        (x - sparse.decode()).square().sum(), discarded.square().sum()
    )
    assert sparse.storage()["total_bytes"] == 6 * 2 + 6 * 3 * 8 + 48
    tail = random((2, 19)).T
    torch.testing.assert_close(encode_sparse_walsh(tail, 8, 8).decode(), tail)
    assert not encode_sparse_walsh(tail, 8, 0).decode().any()
    impulse = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert encode_sparse_walsh(impulse, 4, 2).indices.tolist() == [[0, 1]]
    constant = torch.full((2, 8), 3.0, dtype=torch.float64)
    torch.testing.assert_close(encode_sparse_walsh(constant, 8, 1).decode(), constant)


@pytest.mark.parametrize(
    "fit",
    [
        fit_additive_codebook,
        fit_tensor_product_codebook,
        fit_tile_basis,
        encode_sparse_walsh,
        orthonormal_walsh,
    ],
)
def test_input_validation(fit):
    for bad in (
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
        torch.empty(0),
        torch.tensor([1]),
        torch.tensor(1.0),
    ):
        with pytest.raises(ValueError):
            fit(bad)


def test_invalid_configuration_and_indices():
    q = random((4, 4, 2))
    for rank in (0, 3, 5, True, 1.0):
        with pytest.raises(ValueError):
            fit_tensor_product_codebook(q, rank)
    for fit in (fit_additive_codebook(q), fit_tensor_product_codebook(q)):
        for i, j in (
            (torch.tensor(-1), torch.tensor(0)),
            (torch.tensor(0), torch.tensor(4)),
            (torch.tensor(0.0), torch.tensor(0)),
            (torch.tensor([0]), torch.tensor(0)),
        ):
            with pytest.raises(ValueError):
                fit.lookup(i, j)
        with pytest.raises(ValueError):
            fit.storage(0)
    for block, keep in ((3, 1), (0, 0), (8, 9), (8, -1), (True, 1)):
        with pytest.raises(ValueError):
            encode_sparse_walsh(q, block, keep)
    for rank, tile in ((0, 8), (1, 0), (True, 8), (1, 2.5)):
        with pytest.raises(ValueError):
            fit_tile_basis(q, rank, tile)
    with pytest.raises(ValueError):
        orthonormal_walsh(torch.ones(3))
