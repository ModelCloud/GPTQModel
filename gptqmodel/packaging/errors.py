# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Failures raised by :func:`gptqmodel.package`.

Every error here is a refusal to publish something the writer cannot prove
correct. The packaging path never falls back to a lossier container, never
dequantizes a packed weight, and never writes a partially converted file.
"""

from __future__ import annotations


class PackageError(Exception):
    """Base class for every packaging failure."""


class MissingDependencyError(PackageError):
    """An optional dependency the target container needs is not installed."""


class CheckpointFormatError(PackageError):
    """The source directory is not a checkpoint this writer can read."""


class UnsupportedArchitectureError(PackageError):
    """No verified layout is registered for the checkpoint's architecture."""


class UnsupportedQuantTypeError(PackageError):
    """A packed tensor uses a ggml type unavailable for this export."""


class UnsupportedTensorError(PackageError):
    """A checkpoint tensor has no place in the target container."""


class UnsafeTransformError(PackageError):
    """The target layout needs a transform that is not byte-safe on packed rows."""


__all__ = [
    "CheckpointFormatError",
    "MissingDependencyError",
    "PackageError",
    "UnsafeTransformError",
    "UnsupportedArchitectureError",
    "UnsupportedQuantTypeError",
    "UnsupportedTensorError",
]
