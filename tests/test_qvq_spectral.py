# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from gptqmodel.quantization.qvq_spectral import (
    favorable_propagation_spectral_modes,
    propagation_spectral_mode_products,
    reconstruct_propagation_spectral_modes,
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
