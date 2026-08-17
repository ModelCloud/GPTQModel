# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


def realized_propagation_product(
    gradient: torch.Tensor,
    candidate: torch.Tensor,
    baseline: torch.Tensor,
) -> torch.Tensor:
    """Return the exact first-order product of a realized serialized delta."""

    tensors = (gradient, candidate, baseline)
    if any(not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point() for tensor in tensors):
        raise TypeError("QVQ realized propagation tensors must be floating-point tensors.")
    if candidate.shape != baseline.shape or gradient.shape != baseline.shape:
        raise ValueError("QVQ realized propagation tensors must have identical shapes.")
    if candidate.device != baseline.device or gradient.device != baseline.device:
        raise ValueError("QVQ realized propagation tensors must share one device.")
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("QVQ realized propagation tensors must contain only finite values.")
    product = (gradient.to(torch.float32) * (candidate.to(torch.float32) - baseline.to(torch.float32))).sum()
    if not torch.isfinite(product):
        raise ValueError("QVQ realized propagation product became non-finite.")
    return product


def _validate_spectral_geometry(
    input_root: torch.Tensor,
    output_root: torch.Tensor,
    left_vectors: torch.Tensor,
    singular_values: torch.Tensor,
    right_vectors_h: torch.Tensor,
) -> tuple[int, int, int]:
    tensors = (input_root, output_root, left_vectors, singular_values, right_vectors_h)
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("QVQ propagation-shaped spectrum inputs must be tensors.")
    if any(not tensor.is_floating_point() for tensor in tensors):
        raise TypeError("QVQ propagation-shaped spectrum inputs must use floating-point dtypes.")
    if any(tensor.device != input_root.device for tensor in tensors[1:]):
        raise ValueError("QVQ propagation-shaped spectrum inputs must share one device.")
    if any(tensor.dtype != input_root.dtype for tensor in tensors[1:]):
        raise ValueError("QVQ propagation-shaped spectrum inputs must share one dtype.")
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("QVQ propagation-shaped spectrum inputs must contain only finite values.")
    if input_root.ndim != 2 or input_root.shape[0] != input_root.shape[1]:
        raise ValueError("QVQ propagation-shaped input root must be square.")
    if output_root.ndim != 2 or output_root.shape[0] != output_root.shape[1]:
        raise ValueError("QVQ propagation-shaped output root must be square.")
    if (
        torch.any(input_root.diagonal() <= 0)
        or torch.any(output_root.diagonal() <= 0)
        or torch.count_nonzero(torch.triu(input_root, diagonal=1))
        or torch.count_nonzero(torch.triu(output_root, diagonal=1))
    ):
        raise ValueError("QVQ propagation-shaped roots must be nonsingular lower-Cholesky factors.")
    input_size = input_root.shape[0]
    output_size = output_root.shape[0]
    if left_vectors.ndim != 2 or left_vectors.shape[0] != input_size:
        raise ValueError("QVQ propagation-shaped left vectors must match the input geometry.")
    rank = left_vectors.shape[1]
    if singular_values.ndim != 1 or singular_values.shape[0] != rank:
        raise ValueError("QVQ propagation-shaped singular values must match the spectral rank.")
    if right_vectors_h.ndim != 2 or tuple(right_vectors_h.shape) != (rank, output_size):
        raise ValueError("QVQ propagation-shaped right vectors must match the output geometry.")
    if torch.any(singular_values < 0):
        raise ValueError("QVQ propagation-shaped singular values must be nonnegative.")
    return input_size, output_size, rank


def propagation_spectral_mode_products(
    input_root: torch.Tensor,
    output_root: torch.Tensor,
    left_vectors: torch.Tensor,
    singular_values: torch.Tensor,
    right_vectors_h: torch.Tensor,
    gradient: torch.Tensor,
) -> torch.Tensor:
    """Return ``<gradient, A_i>`` for every signed residual atom.

    The YAQA-whitened residual has SVD ``U diag(sigma) V.T`` and atom

    ``A_i = C_I^-T (sigma_i u_i v_i.T) C_O^-1``.

    Negative products predict a first-order reduction in the propagated loss
    when that original signed atom is added to the quantized reconstruction.
    """

    input_size, output_size, _ = _validate_spectral_geometry(
        input_root,
        output_root,
        left_vectors,
        singular_values,
        right_vectors_h,
    )
    if not isinstance(gradient, torch.Tensor) or not gradient.is_floating_point():
        raise TypeError("QVQ propagation-shaped gradient must be a floating-point tensor.")
    if tuple(gradient.shape) != (input_size, output_size):
        raise ValueError("QVQ propagation-shaped gradient must match the residual geometry.")
    if gradient.device != input_root.device or gradient.dtype != input_root.dtype:
        raise ValueError("QVQ propagation-shaped gradient must share the spectral device and dtype.")
    if not torch.isfinite(gradient).all():
        raise ValueError("QVQ propagation-shaped gradient must contain only finite values.")

    # Adjoint of A = C_I^-T M C_O^-1:
    # <G, A> = <C_I^-1 G C_O^-T, M>.
    whitened_gradient = torch.linalg.solve_triangular(input_root, gradient, upper=False)
    whitened_gradient = torch.linalg.solve_triangular(
        output_root,
        whitened_gradient.transpose(0, 1),
        upper=False,
    ).transpose(0, 1)
    right_vectors = right_vectors_h.transpose(0, 1)
    products = singular_values * torch.einsum(
        "ir,io,or->r",
        left_vectors,
        whitened_gradient,
        right_vectors,
    )
    if not torch.isfinite(products).all():
        raise ValueError("QVQ propagation-shaped mode products became non-finite.")
    return products


def favorable_propagation_spectral_modes(
    mode_products: torch.Tensor,
    *,
    maximum_modes: int,
    minimum_predicted_decrease: float = 0.0,
) -> torch.Tensor:
    """Rank favorable signed atoms by their predicted first-order decrease."""

    if not isinstance(mode_products, torch.Tensor) or mode_products.ndim != 1:
        raise TypeError("QVQ propagation-shaped mode products must be a rank-1 tensor.")
    if not mode_products.is_floating_point() or not torch.isfinite(mode_products).all():
        raise ValueError("QVQ propagation-shaped mode products must be finite and floating point.")
    if isinstance(maximum_modes, bool) or not isinstance(maximum_modes, int) or maximum_modes < 1:
        raise ValueError("QVQ propagation-shaped maximum mode count must be a positive integer.")
    if isinstance(minimum_predicted_decrease, bool) or not isinstance(
        minimum_predicted_decrease,
        (int, float),
    ):
        raise TypeError("QVQ propagation-shaped minimum decrease must be a real scalar.")
    threshold = torch.as_tensor(
        -float(minimum_predicted_decrease),
        device=mode_products.device,
        dtype=mode_products.dtype,
    )
    if not torch.isfinite(threshold) or threshold > 0:
        raise ValueError("QVQ propagation-shaped minimum decrease must be finite and nonnegative.")
    favorable = torch.nonzero(mode_products < threshold, as_tuple=False).flatten()
    if favorable.numel() == 0:
        return favorable
    ordering = torch.argsort(mode_products[favorable], stable=True)
    return favorable[ordering[:maximum_modes]]


def select_propagation_shaped_svd(
    input_root: torch.Tensor,
    output_root: torch.Tensor,
    left_vectors: torch.Tensor,
    singular_values: torch.Tensor,
    right_vectors_h: torch.Tensor,
    gradient: torch.Tensor,
    *,
    maximum_modes: int,
    minimum_predicted_decrease: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep exact signed SVD atoms whose first-order propagated effect is favorable.

    The returned factors are only filtered and reordered; no singular-vector
    sign, singular value, or atom magnitude is changed. Prefix ranks therefore
    retain the most favorable complete atoms instead of a local-energy prefix.
    """

    mode_products = propagation_spectral_mode_products(
        input_root,
        output_root,
        left_vectors,
        singular_values,
        right_vectors_h,
        gradient,
    )
    mode_indices = favorable_propagation_spectral_modes(
        mode_products,
        maximum_modes=maximum_modes,
        minimum_predicted_decrease=minimum_predicted_decrease,
    )
    return (
        left_vectors[:, mode_indices],
        singular_values[mode_indices],
        right_vectors_h[mode_indices],
        mode_products,
        mode_indices,
    )


def reconstruct_propagation_spectral_modes(
    input_root: torch.Tensor,
    output_root: torch.Tensor,
    left_vectors: torch.Tensor,
    singular_values: torch.Tensor,
    right_vectors_h: torch.Tensor,
    mode_indices: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct an exact sum of selected signed ``U_i sigma_i V_i.T`` atoms."""

    input_size, output_size, rank = _validate_spectral_geometry(
        input_root,
        output_root,
        left_vectors,
        singular_values,
        right_vectors_h,
    )
    if not isinstance(mode_indices, torch.Tensor) or mode_indices.ndim != 1 or mode_indices.dtype != torch.long:
        raise TypeError("QVQ propagation-shaped mode indices must be a rank-1 int64 tensor.")
    if mode_indices.device != input_root.device:
        raise ValueError("QVQ propagation-shaped mode indices must share the spectral device.")
    if mode_indices.numel() == 0:
        return torch.zeros((input_size, output_size), dtype=input_root.dtype, device=input_root.device)
    if torch.any(mode_indices < 0) or torch.any(mode_indices >= rank):
        raise ValueError("QVQ propagation-shaped mode index is outside the available rank.")
    if torch.unique(mode_indices).numel() != mode_indices.numel():
        raise ValueError("QVQ propagation-shaped mode indices must be unique.")

    whitened_correction = (
        left_vectors[:, mode_indices] * singular_values[mode_indices].unsqueeze(0)
    ) @ right_vectors_h[mode_indices]
    correction = torch.linalg.solve_triangular(
        input_root.transpose(0, 1),
        whitened_correction,
        upper=True,
    )
    correction = torch.linalg.solve_triangular(
        output_root.transpose(0, 1),
        correction.transpose(0, 1),
        upper=True,
    ).transpose(0, 1)
    if not torch.isfinite(correction).all():
        raise ValueError("QVQ propagation-shaped correction became non-finite.")
    return correction
