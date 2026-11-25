# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Convert a GPT-QModel checkpoint into a single-file runtime container.

``gptqmodel.package(GGUFPackageConfig(...))`` writes one llama.cpp-readable
``.gguf`` file. The packed payload of a GGUF checkpoint is already ggml block
bytes, so packaging is a pure container change: it renames tensors, applies
whole-output-row reorders, and stores dense tensors. It never dequantizes or
re-quantizes a weight, and it refuses a layout it cannot pack byte-exactly.
"""

from __future__ import annotations

from .config import DENSE_DTYPES, GGUFPackageConfig, PackageConfig
from .errors import (
    CheckpointFormatError,
    MissingDependencyError,
    PackageError,
    UnsafeTransformError,
    UnsupportedArchitectureError,
    UnsupportedQuantTypeError,
    UnsupportedTensorError,
)
from .gguf_package import package


__all__ = [
    "DENSE_DTYPES",
    "CheckpointFormatError",
    "GGUFPackageConfig",
    "MissingDependencyError",
    "PackageConfig",
    "PackageError",
    "UnsafeTransformError",
    "UnsupportedArchitectureError",
    "UnsupportedQuantTypeError",
    "UnsupportedTensorError",
    "package",
]
