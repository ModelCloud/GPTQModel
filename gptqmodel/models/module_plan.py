# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Immutable module inventory derived from a model definition's ``module_tree``.

Quantization code must never rediscover module roles by inspecting live names.
The definition tree already declares decoder boundaries, aliases, MoE roots,
expert placeholders, projection order, and exclusion/capture flags. This module
turns that declaration into exact, layer-relative paths and role queries.

Only template paths declared by the definition are matched. Expert placeholders
are expanded only at explicit ``#`` nodes, and no query accepts a name merely
because it shares a prefix or suffix with a declared path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ._const import EXPERT_INDEX_PLACEHOLDER as _EXPERT_PLACEHOLDER


_PLACEHOLDER_RE = re.escape(_EXPERT_PLACEHOLDER)
_NOT_QUANTIZE = "!"
_CAPTURE_ONLY = "?"

PROJECTION_ROLE_FLAGS = {
    "q": "q",
    "k": "k",
    "v": "v",
    "o": "o",
    "gate": "gate",
    "up": "up",
    "down": "down",
}

PROJECTION_ROLE_NAMES = {
    "q_proj": ("q",),
    "k_proj": ("k",),
    "v_proj": ("v",),
    "o_proj": ("o",),
    "gate_proj": ("gate",),
    "up_proj": ("up",),
    "down_proj": ("down",),
    "w1": ("gate",),
    "w3": ("up",),
    "w2": ("down",),
    "gate_up_proj": ("gate", "up"),
}


def _split_spec(token: str) -> Tuple[str, List[str]]:
    parts = token.split(":")
    aliases = [alias for alias in parts[0].split("|") if alias]
    name = aliases[0] if aliases else parts[0]
    return name, [part for part in parts[1:] if part]


def _join(parent: str, child: str) -> str:
    if not child:
        return parent
    return f"{parent}.{child}" if parent else child


def _iter_variants(module_tree: Any) -> List[List[Any]]:
    if not isinstance(module_tree, (list, tuple)):
        return []
    if module_tree and all(isinstance(item, (list, tuple)) for item in module_tree):
        return [list(item) for item in module_tree]
    return [list(module_tree)]


@dataclass(frozen=True)
class ModuleSpec:
    """One declared module path plus the flags that define its role."""

    template: str
    flags: Tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.template.rsplit(".", 1)[-1]

    @property
    def parent(self) -> str:
        return self.template.rsplit(".", 1)[0] if "." in self.template else ""

    @property
    def is_moe(self) -> bool:
        return "moe" in self.flags

    @property
    def is_expert_placeholder(self) -> bool:
        return _EXPERT_PLACEHOLDER in self.template

    @property
    def is_root_leaf(self) -> bool:
        return self.template == self.name

    @property
    def quantize(self) -> bool:
        return not any(flag in (_NOT_QUANTIZE, _CAPTURE_ONLY) for flag in self.flags)

    @property
    def capture_only(self) -> bool:
        return _CAPTURE_ONLY in self.flags

    @property
    def roles(self) -> Tuple[str, ...]:
        explicit = tuple(PROJECTION_ROLE_FLAGS[flag] for flag in self.flags if flag in PROJECTION_ROLE_FLAGS)
        if explicit:
            return explicit
        return PROJECTION_ROLE_NAMES.get(self.name, ())

    @property
    def pattern(self) -> re.Pattern[str]:
        escaped = re.escape(self.template)
        return re.compile("^" + escaped.replace(_PLACEHOLDER_RE, r"\d+") + "$")

    def matches(self, path: str) -> bool:
        return bool(self.pattern.match(path))


def collect_module_specs(module_tree: Any) -> Dict[str, ModuleSpec]:
    """Collect every path declared under a tree's decoder ``#`` boundary."""

    specs: Dict[str, ModuleSpec] = {}

    def record(template: str, flags: Sequence[str]) -> None:
        current = specs.get(template)
        merged = tuple(dict.fromkeys((*(current.flags if current else ()), *flags)))
        specs[template] = ModuleSpec(template=template, flags=merged)

    def walk(node: Any, parent: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if not isinstance(key, str):
                    continue
                name, flags = _split_spec(key)
                child = _join(parent, _EXPERT_PLACEHOLDER if name == "#" else name)
                record(child, flags)
                walk(value, child)
            return
        if isinstance(node, (tuple, list)):
            for item in node:
                if isinstance(item, dict):
                    walk(item, parent)
                    continue
                if not isinstance(item, str):
                    continue
                name, flags = _split_spec(item)
                child = parent if name == "#" or parent == name or parent.endswith(f".{name}") else _join(parent, name)
                record(child, flags)

    for variant in _iter_variants(module_tree):
        try:
            boundary = variant.index("#")
        except ValueError:
            continue
        mapping = next((item for item in variant[boundary + 1:] if isinstance(item, dict)), None)
        if mapping is not None:
            walk(mapping, "")
    return specs


class ModulePlan:
    """Layer-relative inventory and role queries for one effective model tree."""

    def __init__(self, specs: Mapping[str, ModuleSpec]):
        self._specs = dict(specs)
        self._by_parent: Dict[str, List[ModuleSpec]] = {}
        for spec in self._specs.values():
            self._by_parent.setdefault(spec.parent, []).append(spec)

    @classmethod
    def from_tree(cls, module_tree: Any) -> "ModulePlan":
        return cls(collect_module_specs(module_tree))

    @property
    def specs(self) -> Tuple[ModuleSpec, ...]:
        return tuple(self._specs.values())

    def find(self, path: str) -> Optional[ModuleSpec]:
        spec = self._specs.get(path)
        if spec is not None:
            return spec
        return next((candidate for candidate in self._specs.values() if candidate.matches(path)), None)

    def is_moe(self, path: str) -> bool:
        spec = self.find(path)
        if spec is None:
            return False
        if spec.is_moe:
            return True
        return any(_path_has_segment(spec.template, moe.template) for moe in self.moe_roots())

    def roles(self, path: str) -> Tuple[str, ...]:
        spec = self.find(path)
        return () if spec is None else spec.roles

    def is_expert(self, path: str) -> bool:
        if not isinstance(path, str) or not path:
            return False
        for container in self.expert_containers():
            prefix = container.split(f".{_EXPERT_PLACEHOLDER}", 1)[0]
            if not path.startswith(f"{prefix}."):
                continue
            remainder = path[len(prefix) + 1:]
            parts = remainder.split(".")
            if parts and parts[0].isdigit():
                return True
        return False

    def moe_roots(self) -> Tuple[ModuleSpec, ...]:
        return tuple(spec for spec in self._specs.values() if spec.is_moe)

    def expert_containers(self) -> Tuple[str, ...]:
        containers = {
            spec.template.rsplit(f".{_EXPERT_PLACEHOLDER}", 1)[0]
            for spec in self._specs.values()
            if spec.is_expert_placeholder
        }
        return tuple(sorted(containers))

    def shared_expert_containers(self) -> Tuple[str, ...]:
        """Return declared non-routed projection containers beneath a MoE root.

        This is inferred from tree structure, not names: any direct projection
        family under the MoE root that is not the routed-expert placeholder is
        a shared/parallel expert family (for example ``shared_experts`` or
        ``shared_mlp``). Router gates have no projection roles and are excluded.
        """
        expert_containers = set(self.expert_containers())
        containers = set()
        for moe in self.moe_roots():
            for spec in self._specs.values():
                if spec.parent == moe.template or not spec.roles:
                    continue
                if not _path_has_segment(spec.template, moe.template):
                    continue
                container = spec.parent
                if container in expert_containers or _EXPERT_PLACEHOLDER in container:
                    continue
                containers.add(container)
        return tuple(sorted(containers))

    def role_paths_in_parent(self, parent: str, role: str) -> Tuple[str, ...]:
        """Return concrete paths for one role under a concrete parent path."""
        matched = []
        for spec in self._by_parent.get(parent, ()):
            if role in spec.roles:
                matched.append(spec.template)
        return tuple(matched)

    def child_roles(self, parent: str, role: str) -> Tuple[str, ...]:
        """Return direct child names that carry ``role`` under a declared parent."""
        return tuple(
            spec.template.rsplit(".", 1)[-1] if "." in spec.template else spec.template
            for spec in self._by_parent.get(parent, ())
            if role in spec.roles
        )

    def direct_child(self, parent: str, name: str) -> bool:
        return any(leaf.template == f"{parent}.{name}" for leaf in self._by_parent.get(parent, ()))

    def concrete_paths(self, template_parent: str, concrete_parent: str) -> Tuple[str, ...]:
        """Swap a declared parent template for its concrete, expanded parent path."""
        paths = []
        for spec in self._by_parent.get(template_parent, ()):
            leaf = spec.template.rsplit(".", 1)[-1] if "." in spec.template else spec.template
            paths.append(_join(concrete_parent, leaf))
        return tuple(paths)

    def shared_container_leaf(self, moe_root: str) -> Optional[str]:
        for container in self.shared_expert_containers():
            if container == moe_root or container.startswith(f"{moe_root}."):
                relative = container[len(moe_root) + 1:] if container != moe_root else ""
                if relative:
                    return relative.split(".", 1)[0]
        return None

    def expert_container_leaf(self, moe_root: str) -> Optional[str]:
        for container in self.expert_containers():
            if container == moe_root or container.startswith(f"{moe_root}."):
                relative = container[len(moe_root) + 1:] if container != moe_root else ""
                if relative:
                    return relative.split(".", 1)[0]
        return None

    def expert_root(self, path: str) -> Optional[str]:
        """Return the expert family path containing a concrete expert leaf."""
        for container in self.expert_containers():
            prefix = container.split(f".{_EXPERT_PLACEHOLDER}", 1)[0]
            if not _path_has_segment(path, prefix):
                continue
            remainder = path[len(prefix) + 1:]
            parts = remainder.split(".")
            if parts and parts[0].isdigit():
                return f"{prefix}.{parts[0]}"
        return None

    def same_moe_family(self, left: str, right: str) -> bool:
        """Return whether two expert leaves belong to the same expanded expert."""
        left_root, right_root = self.expert_root(left), self.expert_root(right)
        return left_root is not None and left_root == right_root


def _path_has_segment(path: str, parent: str) -> bool:
    return path == parent or path.startswith(f"{parent}.")
