# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Helpers for turning SLQ bitwidth assignments into GPT-QModel config objects."""

from __future__ import annotations

import re

import numpy as np


def _to_module_key(name: str) -> str:
    """Normalize a group name for use as a dynamic config key.

    Dynamic matching in ``QuantizeConfig`` supports regex and full names. We
    wrap the provided name in ``^...$`` anchors when it does not already contain
    regex metacharacters, otherwise leave it as a raw pattern.
    """

    if re.search(r"[.*+?^${}()|[\]\\]", name):
        return name
    return f"^{re.escape(name)}$"


def build_dynamic_bits(
    group_names: list[str],
    bitwidths: list[int] | np.ndarray,
    assignment: list[int] | np.ndarray,
    *,
    additional_overrides: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """Build a ``QuantizeConfig.dynamic`` dict from an SLQ bitwidth assignment.

    The returned dictionary maps each group name (anchored for regex matching)
    to a dict containing ``{"bits": assigned_bitwidth}``. Additional per-group
    overrides can be merged in.

    Args:
        group_names: Human-readable group/module names.
        bitwidths: Candidate bitwidths (one per column of the cost matrix).
        assignment: Array of selected column indices.
        additional_overrides: Optional extra per-group config entries.

    Returns:
        Dictionary suitable for ``QuantizeConfig(dynamic=...)``.
    """

    bitwidths = np.asarray(bitwidths)
    assignment = np.asarray(assignment)
    dynamic: dict[str, dict] = {}

    overrides = additional_overrides or {}
    for name, idx in zip(group_names, assignment):
        key = _to_module_key(name)
        bits = int(bitwidths[idx])
        entry = {"bits": bits}
        entry.update(overrides.get(name, {}))
        dynamic[key] = entry

    return dynamic
