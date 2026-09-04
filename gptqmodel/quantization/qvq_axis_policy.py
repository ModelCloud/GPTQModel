# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Architecture-declared QVQ transform-axis and shared-input policies."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence

import torch

from ..nn_modules.qlinear.qvq import QVQLinear


QVQTransformAxisPolicy = Mapping[str, tuple[bool, bool]]
QVQGroupedCandidatePolicy = Mapping[str, Sequence[Sequence[str]]]
QVQ_TRANSFORM_AXIS_META_KEY = "qvq_transform_axis_overrides"
QVQ_TRANSFORM_AXIS_SCHEMA = "qvq.transform-axes.v1"


def validate_qvq_transform_axis_overrides(
    overrides: object,
) -> dict[str, tuple[bool, bool]]:
    """Return a strict copy of ``suffix -> (input H, output H)`` overrides."""

    if overrides is None:
        return {}
    if not isinstance(overrides, Mapping):
        raise TypeError("qvq_transform_axis_overrides must be a mapping")
    normalized: dict[str, tuple[bool, bool]] = {}
    for suffix, flags in overrides.items():
        if not isinstance(suffix, str) or not suffix.strip():
            raise TypeError("QVQ transform-axis suffixes must be non-empty strings")
        if (
            not isinstance(flags, (tuple, list))
            or len(flags) != 2
            or any(not isinstance(flag, bool) for flag in flags)
        ):
            raise TypeError(
                "QVQ transform-axis values must be (input_hadamard, output_hadamard) bool pairs"
            )
        normalized[suffix.strip(".")] = (flags[0], flags[1])
    return normalized


def resolve_qvq_transform_axes(
    module_full_name: str,
    overrides: object,
) -> tuple[bool, bool]:
    """Resolve one module without leaking architecture names into QVQ kernels."""

    matches = [
        (suffix, flags)
        for suffix, flags in validate_qvq_transform_axis_overrides(overrides).items()
        if module_full_name == suffix or module_full_name.endswith(f".{suffix}")
    ]
    if not matches:
        return True, True
    matches.sort(key=lambda item: len(item[0]), reverse=True)
    longest = len(matches[0][0])
    selected = {flags for suffix, flags in matches if len(suffix) == longest}
    if len(selected) != 1:
        raise ValueError(
            f"conflicting QVQ transform-axis policies match {module_full_name!r}"
        )
    return selected.pop()


def qvq_shared_input_seed(
    module_full_name: str,
    candidates: object,
) -> int | None:
    """Return a deterministic group seed for a declared sibling, if any.

    Candidate names are relative to their common parent.  The parent path is
    included in the seed, so every decoder layer receives an independent
    shared sign vector while siblings inside that layer receive the same one.
    """

    if candidates is None:
        return None
    if not isinstance(candidates, Mapping):
        raise TypeError("qvq_grouped_p32_candidates must be a mapping")
    parent_name, separator, child_name = module_full_name.rpartition(".")
    if not separator:
        parent_name = ""
        child_name = module_full_name
    matched: list[tuple[str, ...]] = []
    for raw_groups in candidates.values():
        if not isinstance(raw_groups, Sequence) or isinstance(raw_groups, (str, bytes)):
            raise TypeError("QVQ grouped candidate categories must contain sequences of groups")
        for raw_group in raw_groups:
            if not isinstance(raw_group, Sequence) or isinstance(raw_group, (str, bytes)):
                raise TypeError("QVQ grouped candidates must be sequences of relative module names")
            group = tuple(raw_group)
            if len(group) < 2 or any(not isinstance(name, str) or not name for name in group):
                raise ValueError("QVQ grouped candidates require at least two non-empty module names")
            if child_name in group:
                matched.append(group)
    if not matched:
        return None
    if len(matched) != 1:
        raise ValueError(f"QVQ module {module_full_name!r} belongs to multiple shared-input groups")

    # Keep this local import-free and stable across Python processes.
    import zlib

    group = matched[0]
    identity = f"{parent_name}|qvq-shared-input|{','.join(group)}"
    return zlib.crc32(identity.encode("utf-8")) & 0x7FFFFFFF


def apply_qvq_transform_axis_overrides(
    model: torch.nn.Module,
    overrides: object,
) -> int:
    """Apply architecture-owned transform axes to loaded QVQ module shells."""

    normalized = validate_qvq_transform_axis_overrides(overrides)
    if not normalized:
        return 0
    changed = 0
    for module_name, module in model.named_modules():
        if not isinstance(module, QVQLinear):
            continue
        input_hadamard, output_hadamard = resolve_qvq_transform_axes(
            module_name, normalized
        )
        if (
            module.input_hadamard != input_hadamard
            or module.output_hadamard != output_hadamard
        ):
            module.input_hadamard = input_hadamard
            module.output_hadamard = output_hadamard
            changed += 1
    return changed


def set_qvq_transform_axis_metadata(
    quantize_config: object,
    overrides: object,
) -> dict[str, object] | None:
    """Persist folded-axis semantics required to reconstruct a checkpoint."""

    normalized = validate_qvq_transform_axis_overrides(overrides)
    if not normalized:
        return None
    payload: dict[str, object] = {
        "schema": QVQ_TRANSFORM_AXIS_SCHEMA,
        "overrides": {
            suffix: [input_hadamard, output_hadamard]
            for suffix, (input_hadamard, output_hadamard) in normalized.items()
        },
    }
    current_meta = getattr(quantize_config, "meta", None)
    if current_meta is None:
        meta = {}
    elif isinstance(current_meta, dict):
        meta = copy.deepcopy(current_meta)
    else:
        raise TypeError("QVQ quantization metadata must be a dictionary")
    current = meta.get(QVQ_TRANSFORM_AXIS_META_KEY)
    if current is not None and current != payload:
        raise ValueError("refusing to replace conflicting QVQ transform-axis metadata")
    meta[QVQ_TRANSFORM_AXIS_META_KEY] = payload
    setattr(quantize_config, "meta", meta)
    return copy.deepcopy(payload)


def qvq_transform_axis_overrides_from_config(
    quantize_config: object,
) -> dict[str, tuple[bool, bool]]:
    """Parse the versioned folded-axis contract from a quantization config."""

    meta = getattr(quantize_config, "meta", None)
    if meta is None:
        return {}
    if not isinstance(meta, dict):
        raise TypeError("QVQ quantization metadata must be a dictionary")
    payload = meta.get(QVQ_TRANSFORM_AXIS_META_KEY)
    if payload is None:
        return {}
    if not isinstance(payload, dict) or set(payload) != {"schema", "overrides"}:
        raise ValueError("QVQ transform-axis metadata has an invalid structure")
    if payload["schema"] != QVQ_TRANSFORM_AXIS_SCHEMA:
        raise ValueError(
            f"unsupported QVQ transform-axis schema: {payload['schema']!r}"
        )
    return validate_qvq_transform_axis_overrides(payload["overrides"])


__all__ = [
    "apply_qvq_transform_axis_overrides",
    "qvq_shared_input_seed",
    "qvq_transform_axis_overrides_from_config",
    "resolve_qvq_transform_axes",
    "set_qvq_transform_axis_metadata",
    "validate_qvq_transform_axis_overrides",
    "QVQ_TRANSFORM_AXIS_META_KEY",
    "QVQ_TRANSFORM_AXIS_SCHEMA",
]
