# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Stable-ABI JIT loader for the RCO budget dynamic program."""

from __future__ import annotations

from pathlib import Path

from .cpp import (
    _TORCH_STABLE_ABI_TARGET,
    TorchOpsJitExtension,
    _jit_cxx_standard,
    default_torch_ops_build_root,
)


def _source() -> str:
    root = Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "rco"
    return str(root / "rco_dp.cpp")


_RCO_DP_EXTENSION = TorchOpsJitExtension(
    name="gptqmodel_rco_dp",
    namespace="gptqmodel_rco",
    required_ops=("rco_dp",),
    sources=lambda: [_source()],
    build_root_env="GPTQMODEL_EXT_BUILD",
    default_build_root=lambda: default_torch_ops_build_root("rco_dp"),
    display_name="rco_dp",
    extra_cflags=["-O3", f"-std={_jit_cxx_standard()}"],
    verbose_env="GPTQMODEL_EXT_VERBOSE",
    requires_cuda=False,
    python_abi_dependent=False,
    torch_stable_abi_target=_TORCH_STABLE_ABI_TARGET,
)
