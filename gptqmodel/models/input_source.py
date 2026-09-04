# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Generic metadata for grouping modules that consume the same input."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..looper.named_module import NamedModule


@dataclass(frozen=True)
class UniqueInput:
    """Identify an input source that belongs only to one module."""


@dataclass(frozen=True)
class NamedInput:
    """Identify an explicitly named input source within a structural scope."""

    name: str


InputSpec = UniqueInput | NamedInput
INPUT_FLAG = "input"


def parse_input_flag(flags: Sequence[str]) -> tuple[InputSpec | None, list[str]]:
    """Extract one input-source flag from a raw module flag sequence."""

    input_flags = [
        flag
        for flag in flags
        if flag == INPUT_FLAG or flag.startswith(f"{INPUT_FLAG}=")
    ]
    if len(input_flags) > 1:
        raise ValueError("duplicate input flags")

    remaining = [flag for flag in flags if flag not in input_flags]
    if not input_flags:
        return None, remaining

    flag = input_flags[0]
    if flag == INPUT_FLAG:
        return UniqueInput(), remaining

    name = flag[len(INPUT_FLAG) + 1:]
    if not name:
        raise ValueError("input name cannot be empty")
    if ":" in name or any(character.isspace() for character in name):
        raise ValueError("input name cannot contain ':' or whitespace")
    return NamedInput(name), remaining


@dataclass(frozen=True)
class ModuleTreeEntry:
    """Metadata describing one module-tree path."""

    full_path: str
    scope: str
    subset_id: int
    input_spec: InputSpec | None
    not_quantized: bool
    capture_only: bool


@dataclass(frozen=True)
class InputSourceId:
    """Hashable identity for a module input source."""

    scope: str
    subset_id: int | None = None
    name: str | None = None
    module: str | None = None

    def __post_init__(self) -> None:
        values = (self.subset_id, self.name, self.module)
        if sum(value is not None for value in values) != 1:
            raise ValueError("exactly one of subset_id, name, or module must be set")

    @property
    def kind(self) -> str:
        """Return the input-source identity kind."""

        if self.subset_id is not None:
            return "subset"
        if self.name is not None:
            return "named"
        return "unique"


def resolve_input_source(module: NamedModule) -> InputSourceId:
    """Resolve the logical input source for one named module."""

    scope = module.tree_scope_id
    if scope is None or module.subset_id is None:
        parent = module.full_name.rsplit(".", 1)[0] if "." in module.full_name else ""
        return InputSourceId(scope=parent, module=module.full_name)

    spec = module.input_spec
    if isinstance(spec, UniqueInput):
        return InputSourceId(scope=scope, module=module.full_name)
    if isinstance(spec, NamedInput):
        return InputSourceId(scope=scope, name=spec.name)
    return InputSourceId(scope=scope, subset_id=module.subset_id)


def group_input_sources(
    modules: Iterable[NamedModule],
) -> dict[InputSourceId, list[NamedModule]]:
    """Group modules by input-source identity while preserving iteration order."""

    grouped: dict[InputSourceId, list[NamedModule]] = {}
    for module in modules:
        grouped.setdefault(resolve_input_source(module), []).append(module)
    return grouped
