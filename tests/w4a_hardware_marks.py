# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Hardware gates shared by the W4A test matrix.

Native E4M3 tensor-core products start with Ada Lovelace (SM 8.9), so the FP8
carrier runs on any SM 8.9+ part; Hopper and Blackwell qualify as well. The
block-scaled FP4 carrier is validated on GB10 / SM 12.1 only in this release.

Tests that parametrize several carriers attach the mark to the parameter
(``pytest.param("w4a_nvfp4", marks=NVFP4_HARDWARE)``) instead of decorating the
function, so a single file can exercise the FP8 lane on Ada while its NVFP4
cases stay skipped.
"""

from __future__ import annotations

import pytest
import torch


def _capability() -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability(0)


FP8_HARDWARE = pytest.mark.skipif(
    _capability() is None or _capability() < (8, 9),
    reason="FP8 tensor cores (SM 8.9+ / Ada or newer) required",
)
NVFP4_HARDWARE = pytest.mark.skipif(
    _capability() != (12, 1),
    reason="GB10 / SM 12.1 required for NVFP4",
)
