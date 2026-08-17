# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.quantization.qvq_spectral import (
    favorable_propagation_spectral_modes,
    propagation_spectral_mode_products,
    realized_propagation_product,
    reconstruct_propagation_spectral_modes,
    select_crossfit_propagation_shaped_svd,
    select_propagation_shaped_svd,
)


def _spectral_case(seed: int = 20260817):
    generator = torch.Generator().manual_seed(seed)
    input_size, output_size, rank = 7, 9, 5
    input_source = torch.randn((input_size, input_size), generator=generator)
    output_source = torch.randn((output_size, output_size), generator=generator)
    input_root = torch.linalg.cholesky(input_source @ input_source.T + torch.eye(input_size))
    output_root = torch.linalg.cholesky(output_source @ output_source.T + torch.eye(output_size))
    left_vectors, _ = torch.linalg.qr(torch.randn((input_size, rank), generator=generator))
    right_vectors, _ = torch.linalg.qr(torch.randn((output_size, rank), generator=generator))
    singular_values = torch.linspace(2.0, 0.25, rank)
    gradient = torch.randn((input_size, output_size), generator=generator)
    return input_root, output_root, left_vectors, singular_values, right_vectors.T, gradient


def test_propagation_spectral_products_match_explicit_unwhitened_atoms():
    input_root, output_root, left_vectors, singular_values, right_vectors_h, gradient = _spectral_case()

    actual = propagation_spectral_mode_products(
        input_root,
        output_root,
        left_vectors,
        singular_values,
        right_vectors_h,
        gradient,
    )
    expected = []
    for mode in range(singular_values.numel()):
        atom = reconstruct_propagation_spectral_modes(
            input_root,
            output_root,
            left_vectors,
            singular_values,
            right_vectors_h,
            torch.tensor([mode]),
        )
        expected.append((gradient * atom).sum())

    torch.testing.assert_close(actual, torch.stack(expected), rtol=2e-5, atol=2e-6)


def test_realized_propagation_product_scores_the_serialized_delta_not_its_teacher():
    gradient = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    baseline = torch.tensor([[1.0, 2.0], [-1.0, 0.0]])
    candidate = torch.tensor([[0.5, 3.0], [-1.0, -0.5]])

    actual = realized_propagation_product(gradient, candidate, baseline)

    expected = (gradient * (candidate - baseline)).sum()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_favorable_modes_preserve_atoms_and_order_by_predicted_decrease():
    input_root, output_root, left_vectors, singular_values, right_vectors_h, _ = _spectral_case()
    products = torch.tensor([0.5, -0.2, -1.5, 0.0, -0.7])

    selected = favorable_propagation_spectral_modes(products, maximum_modes=2)
    correction = reconstruct_propagation_spectral_modes(
        input_root,
        output_root,
        left_vectors,
        singular_values,
        right_vectors_h,
        selected,
    )
    expected = sum(
        (
            reconstruct_propagation_spectral_modes(
                input_root,
                output_root,
                left_vectors,
                singular_values,
                right_vectors_h,
                torch.tensor([mode]),
            )
            for mode in (2, 4)
        ),
        torch.zeros_like(correction),
    )

    assert selected.tolist() == [2, 4]
    torch.testing.assert_close(correction, expected, rtol=2e-5, atol=2e-6)


def test_favorable_modes_can_fail_closed_to_an_exact_zero_correction():
    input_root, output_root, left_vectors, singular_values, right_vectors_h, _ = _spectral_case()
    selected = favorable_propagation_spectral_modes(torch.tensor([0.0, 0.2, 1.0]), maximum_modes=3)
    correction = reconstruct_propagation_spectral_modes(
        input_root,
        output_root,
        left_vectors,
        singular_values,
        right_vectors_h,
        selected,
    )

    assert selected.numel() == 0
    torch.testing.assert_close(correction, torch.zeros_like(correction), rtol=0, atol=0)


def test_select_propagation_shaped_svd_preserves_complete_signed_atoms_in_downstream_order():
    dtype = torch.float64
    input_root = torch.tensor([[1.5, 0.0], [0.2, 1.1]], dtype=dtype)
    output_root = torch.tensor([[1.2, 0.0], [-0.1, 1.4]], dtype=dtype)
    left_vectors = torch.eye(2, dtype=dtype)
    singular_values = torch.tensor([3.0, 2.0], dtype=dtype)
    right_vectors_h = torch.eye(2, dtype=dtype)
    atoms = [
        reconstruct_propagation_spectral_modes(
            input_root,
            output_root,
            left_vectors,
            singular_values,
            right_vectors_h,
            torch.tensor([index]),
        )
        for index in range(2)
    ]
    # Mode 1 is more favorable despite carrying less local spectral energy.
    gradient = -0.25 * atoms[0] / atoms[0].square().sum() - 2.0 * atoms[1] / atoms[1].square().sum()

    selected_left, selected_singular, selected_right_h, products, indices = select_propagation_shaped_svd(
        input_root,
        output_root,
        left_vectors,
        singular_values,
        right_vectors_h,
        gradient,
        maximum_modes=2,
    )

    assert indices.tolist() == [1, 0]
    expected_products = torch.stack([(gradient * atom).sum() for atom in atoms])
    torch.testing.assert_close(products, expected_products)
    assert torch.equal(selected_left, left_vectors[:, indices])
    assert torch.equal(selected_singular, singular_values[indices])
    assert torch.equal(selected_right_h, right_vectors_h[indices])
    selected_sum = reconstruct_propagation_spectral_modes(
        input_root,
        output_root,
        selected_left,
        selected_singular,
        selected_right_h,
        torch.arange(2),
    )
    torch.testing.assert_close(selected_sum, atoms[0] + atoms[1])


def test_crossfit_shaping_requires_every_gradient_fold_to_favor_the_mode():
    dtype = torch.float64
    input_root = torch.eye(2, dtype=dtype)
    output_root = torch.eye(2, dtype=dtype)
    left_vectors = torch.eye(2, dtype=dtype)
    singular_values = torch.tensor([2.0, 1.0], dtype=dtype)
    right_vectors_h = torch.eye(2, dtype=dtype)
    gradients = torch.stack(
        (
            torch.tensor([[-2.0, 0.0], [0.0, -1.0]], dtype=dtype),
            torch.tensor([[0.5, 0.0], [0.0, -3.0]], dtype=dtype),
        )
    )

    selected_left, selected_singular, selected_right_h, fold_products, worst_products, indices = (
        select_crossfit_propagation_shaped_svd(
            input_root,
            output_root,
            left_vectors,
            singular_values,
            right_vectors_h,
            gradients,
            maximum_modes=2,
        )
    )

    assert indices.tolist() == [1]
    torch.testing.assert_close(fold_products, torch.tensor([[-4.0, -1.0], [1.0, -3.0]], dtype=dtype))
    torch.testing.assert_close(worst_products, torch.tensor([1.0, -1.0], dtype=dtype))
    assert torch.equal(selected_left, left_vectors[:, 1:2])
    assert torch.equal(selected_singular, singular_values[1:2])
    assert torch.equal(selected_right_h, right_vectors_h[1:2])


def test_propagation_spectral_helpers_reject_invalid_geometry():
    input_root, output_root, left_vectors, singular_values, right_vectors_h, gradient = _spectral_case()

    with pytest.raises(ValueError, match="gradient must match"):
        propagation_spectral_mode_products(
            input_root,
            output_root,
            left_vectors,
            singular_values,
            right_vectors_h,
            gradient[:-1],
        )
    with pytest.raises(ValueError, match="must be unique"):
        reconstruct_propagation_spectral_modes(
            input_root,
            output_root,
            left_vectors,
            singular_values,
            right_vectors_h,
            torch.tensor([1, 1]),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        favorable_propagation_spectral_modes(
            torch.tensor([-1.0]),
            maximum_modes=1,
            minimum_predicted_decrease=-0.1,
        )
    invalid_root = input_root.clone()
    invalid_root[0, -1] = 1.0
    with pytest.raises(ValueError, match="lower-Cholesky"):
        propagation_spectral_mode_products(
            invalid_root,
            output_root,
            left_vectors,
            singular_values,
            right_vectors_h,
            gradient,
        )
