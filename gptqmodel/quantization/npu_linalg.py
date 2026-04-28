# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os

import torch


_DEFAULT_CHOLESKY_BLOCK_SIZE = int(os.getenv("GPTQMODEL_NPU_CHOLESKY_BLOCK_SIZE", "64"))


def _raise_linalg_error(message: str) -> None:
    raise torch._C._LinAlgError(message)


def _validate_square_matrix(matrix: torch.Tensor, op_name: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{op_name} expects a 2D square matrix, got shape {tuple(matrix.shape)}.")
    if not matrix.is_floating_point():
        raise TypeError(f"{op_name} expects a floating point matrix, got dtype {matrix.dtype}.")
    if matrix.device.type != "npu":
        raise ValueError(f"{op_name} is only intended for NPU tensors, got device `{matrix.device}`.")


def _cholesky_panel_lower(panel: torch.Tensor) -> torch.Tensor:
    size = panel.shape[0]
    factor = torch.zeros_like(panel)

    for col in range(size):
        if col == 0:
            diag_sq = panel[col, col]
        else:
            row = factor[col, :col]
            diag_sq = panel[col, col] - torch.sum(row * row)

        if not bool(torch.isfinite(diag_sq) & (diag_sq > 0)):
            _raise_linalg_error("NPU Cholesky failed because the matrix is not positive-definite.")

        diag = torch.sqrt(diag_sq)
        factor[col, col] = diag

        if col + 1 < size:
            if col == 0:
                numerator = panel[col + 1:, col]
            else:
                numerator = panel[col + 1:, col] - factor[col + 1:, :col].matmul(factor[col, :col])
            factor[col + 1:, col] = numerator / diag

    return factor


def npu_cholesky(
    matrix: torch.Tensor,
    *,
    upper: bool = False,
    block_size: int = _DEFAULT_CHOLESKY_BLOCK_SIZE,
) -> torch.Tensor:
    """
    Compute a Cholesky factor on NPU without aten::linalg_cholesky_ex.

    torch-npu 2.9 falls back to CPU for torch.linalg.cholesky on this path.
    This blocked implementation keeps the dense GPTQ Hessian factorization on
    NPU and uses only matmul, elementwise ops, and solve_triangular updates.
    """
    _validate_square_matrix(matrix, "npu_cholesky")

    size = matrix.shape[0]
    if size == 0:
        return matrix.clone()

    block_size = max(1, int(block_size))
    lower = torch.zeros_like(matrix)

    for start in range(0, size, block_size):
        end = min(start + block_size, size)
        panel = matrix[start:end, start:end].clone()

        if start:
            previous = lower[start:end, :start]
            panel = panel - previous.matmul(previous.T)

        panel_factor = _cholesky_panel_lower(panel)
        lower[start:end, start:end] = panel_factor

        if end < size:
            trailing = matrix[end:, start:end]
            if start:
                trailing = trailing - lower[end:, :start].matmul(lower[start:end, :start].T)

            solved = torch.linalg.solve_triangular(
                panel_factor,
                trailing.T.contiguous(),
                upper=False,
            )
            lower[end:, start:end] = solved.T

    return lower.T if upper else lower


def npu_inverse_cholesky_factor(
    matrix: torch.Tensor,
    *,
    block_size: int = _DEFAULT_CHOLESKY_BLOCK_SIZE,
) -> torch.Tensor:
    """
    Return the upper Cholesky factor of inv(matrix) on NPU.

    This matches the GPTQ solver contract used by torch:
        torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(matrix)), upper=True)
    while avoiding the torch-npu CPU fallbacks for cholesky and cholesky_inverse.
    """
    _validate_square_matrix(matrix, "npu_inverse_cholesky_factor")

    lower = npu_cholesky(matrix, upper=False, block_size=block_size)
    identity = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    lower_inv = torch.linalg.solve_triangular(lower, identity, upper=False)
    _, upper = torch.linalg.qr(lower_inv, mode="reduced")
    diag_sign = torch.sign(torch.diagonal(upper))
    diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
    return upper * diag_sign.unsqueeze(1)
