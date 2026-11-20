# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Relative reconstruction-error arithmetic shared by the quantizers."""

from __future__ import annotations

import math


def relative_error_from_energies(error_energy: float, reference_energy: float) -> float:
    """Return ``error_energy / reference_energy`` without a denominator floor.

    Both energies are already FP64 sums of squares, so every positive value is
    a usable denominator. Flooring it to an absolute constant makes a global
    rescaling of the reference and the reconstruction change the reported
    score, and lets a reconstruction that lost the entire signal report a
    near-zero relative error (#5635).

    A non-positive reference energy has no finite ratio. An exactly
    reconstructed zero reference is ``0 / 0`` and reports ``0.0``; any error
    against it is total loss and reports ``+inf``. Callers that require a
    finite score reject that value themselves.
    """

    if reference_energy > 0.0:
        return error_energy / reference_energy
    if error_energy == 0.0:
        return 0.0
    return math.inf
