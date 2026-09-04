# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Shared-input metadata derived from a model definition's ``module_tree``.

Modules that consume the *same* activation tensor (e.g. ``q_proj``/``k_proj``/``v_proj``
after ``input_layernorm``) produce identical GPTQ Hessians (``H = X^T X`` depends only on
the input). A :class:`SharedInputPlan` groups such modules so calibration logic can collect
the Hessian once per group (the ``leader``) and share it with the ``followers``.

Grouping rules (per decoder layer, relative module paths):

* Default: every quantizable module is a singleton group. Subset digits (``:0``) describe
  quantization/execution order, not tensor identity, so they are never used to infer sharing.
* Explicit opt-in: leaves under the same immediate parent that carry the same ``:in=<tag>``
  flag share an input. Leaves with different tags (or no tag) never share, even when they sit
  in the same subset. Tags may span subsets (e.g. Qwen3.5 ``linear_attn.in_proj_qkv:0:in=x``
  and ``linear_attn.in_proj_z:1:in=x``). Tags are scoped to the parent, so ``in=x`` on
  ``self_attn.*`` and ``mlp.*`` yields two independent groups.
* MoE expert placeholders (``experts.#``) expand per expert; each expert is its own parent,
  so routed experts never share with each other.
* Non-quantized (``:!``) and capture-only (``:?``) leaves never join a group.
* A leaf that appears in several ``module_tree`` variants must carry identical metadata;
  conflicting ``:in=``/``:!``/``:?``/subset flags raise :class:`ValueError`.

Only definitions whose groups were verified against a real forward pass (see
:func:`probe_shared_inputs` and ``tests/module_tree/test_shared_input_cpu_forward.py``)
should carry ``:in=`` tags: a wrong tag silently reuses the wrong Hessian.

All structures are immutable and the derivation is a pure function of the definition plus
``model_config``/``quantize_config``; it is safe to call concurrently (GIL=0).
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

from ._const import EXPERT_INDEX_PLACEHOLDER


SHARED_INPUT_FLAG_PREFIX = "in="

_PLACEHOLDER_RE = re.escape(EXPERT_INDEX_PLACEHOLDER)


def parse_shared_input_tag(flags: Iterable[str]) -> Optional[str]:
    """Return the ``<tag>`` of an ``in=<tag>`` flag or ``None`` when absent."""
    for flag in flags:
        if isinstance(flag, str) and flag.startswith(SHARED_INPUT_FLAG_PREFIX):
            tag = flag[len(SHARED_INPUT_FLAG_PREFIX):]
            if not tag:
                raise ValueError(f"`{SHARED_INPUT_FLAG_PREFIX}` flag requires a non-empty tag")
            return tag
    return None


def _split_spec(token: str) -> Tuple[str, List[str]]:
    """``"gate_proj|w1:0:in=x"`` -> ``("gate_proj", ["0", "in=x"])`` (first alias is the runtime name)."""
    parts = token.split(":")
    aliases = [alias for alias in parts[0].split("|") if alias]
    name = aliases[0] if aliases else parts[0]
    flags = [p for p in parts[1:] if p]
    return name, flags


def _join(parent: str, child: str) -> str:
    if not child:
        return parent
    return f"{parent}.{child}" if parent else child


@dataclass(frozen=True)
class LeafSpec:
    """Per-leaf metadata parsed straight from ``module_tree`` (template paths, pre-expansion)."""

    template: str
    subset_tag: int
    input_tag: Optional[str]
    quantize: bool

    @property
    def pattern(self) -> "re.Pattern[str]":
        return re.compile("^" + re.escape(self.template).replace(_PLACEHOLDER_RE, r"\d+") + "$")


def collect_leaf_specs(module_tree: Any) -> Dict[str, LeafSpec]:
    """Walk every ``module_tree`` variant and return ``{template_path: LeafSpec}``.

    A template repeated across variants must resolve to an identical :class:`LeafSpec`;
    otherwise a :class:`ValueError` is raised so a variant cannot silently re-tag a leaf.
    """
    specs: Dict[str, LeafSpec] = {}

    def record(template: str, flags: List[str]) -> None:
        subset_tag = next((int(f) for f in flags if f.isdigit()), 0)
        quantize = not any(f in ("!", "?") for f in flags)
        new = LeafSpec(
            template=template,
            subset_tag=subset_tag,
            input_tag=parse_shared_input_tag(flags),
            quantize=quantize,
        )
        old = specs.get(template)
        if old is not None and old != new:
            raise ValueError(
                f"conflicting module_tree metadata for `{template}` across variants: {old} vs {new}"
            )
        specs.setdefault(template, new)

    def walk(parent: str, node: Any) -> None:
        if isinstance(node, (tuple, list)):
            for token in node:
                if not isinstance(token, str):
                    continue
                name, flags = _split_spec(token)
                # `"#"` and a leaf repeating its parent's last segment both denote the parent itself.
                if name == "#" or parent == name or parent.endswith(f".{name}"):
                    record(parent, flags)
                else:
                    record(_join(parent, name), flags)
        elif isinstance(node, dict):
            for key, value in node.items():
                if not isinstance(key, str):
                    continue
                kname, _ = _split_spec(key)
                if kname == "#":
                    child = _join(parent, EXPERT_INDEX_PLACEHOLDER)
                else:
                    child = _join(parent, kname)
                walk(child, value)

    for variant in _iter_variants(module_tree):
        mapping = next((item for item in variant if isinstance(item, dict)), None)
        if mapping is None:
            continue
        walk("", mapping)

    return specs


def _iter_variants(module_tree: Any) -> List[List[Any]]:
    if not isinstance(module_tree, list):
        return []
    if module_tree and all(isinstance(item, (list, tuple)) for item in module_tree):
        return [list(item) for item in module_tree]
    return [list(module_tree)]


def resolve_leaf_spec(specs: Mapping[str, LeafSpec], module_name: str) -> Optional[LeafSpec]:
    """Match a concrete (expert-expanded) module path back to its template spec."""
    spec = specs.get(module_name)
    if spec is not None:
        return spec
    for candidate in specs.values():
        if EXPERT_INDEX_PLACEHOLDER in candidate.template and candidate.pattern.match(module_name):
            return candidate
    return None


@dataclass(frozen=True)
class SharedInputGroup:
    """A set of quantizable modules (relative to one decoder layer) that consume the same input."""

    key: str
    parent: str
    modules: Tuple[str, ...]
    subset_indices: Tuple[int, ...]
    explicit: bool = False

    @property
    def leader(self) -> str:
        return self.modules[0]

    @property
    def followers(self) -> Tuple[str, ...]:
        return self.modules[1:]

    @property
    def is_shared(self) -> bool:
        return len(self.modules) > 1

    @property
    def spans_subsets(self) -> bool:
        return len(set(self.subset_indices)) > 1

    def subset_index_of(self, module: str) -> int:
        return self.subset_indices[self.modules.index(module)]


@dataclass(frozen=True)
class SharedInputPlan:
    """Ordered collection of :class:`SharedInputGroup` for one decoder layer template."""

    groups: Tuple[SharedInputGroup, ...]
    _by_module: Dict[str, SharedInputGroup] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        by_module: Dict[str, SharedInputGroup] = {}
        for group in self.groups:
            for module in group.modules:
                if module in by_module:
                    raise ValueError(f"module `{module}` appears in multiple shared-input groups")
                by_module[module] = group
        object.__setattr__(self, "_by_module", by_module)

    @property
    def modules(self) -> Tuple[str, ...]:
        return tuple(self._by_module.keys())

    @property
    def shared_groups(self) -> Tuple[SharedInputGroup, ...]:
        return tuple(g for g in self.groups if g.is_shared)

    @property
    def leaders(self) -> Tuple[str, ...]:
        return tuple(g.leader for g in self.groups)

    @property
    def dedup_count(self) -> int:
        """Number of Hessian collections that can be skipped by sharing."""
        return sum(len(g.followers) for g in self.groups)

    def group_for(self, module: str) -> Optional[SharedInputGroup]:
        return self._by_module.get(module)

    def leader_for(self, module: str) -> str:
        group = self._by_module.get(module)
        return module if group is None else group.leader

    def is_explicit(self, module: str) -> bool:
        group = self._by_module.get(module)
        return group is not None and group.explicit

    def followers_of(self, module: str) -> Tuple[str, ...]:
        group = self._by_module.get(module)
        if group is None or group.leader != module:
            return ()
        return group.followers

    def shares_input(self, a: str, b: str) -> bool:
        ga = self._by_module.get(a)
        return ga is not None and ga is self._by_module.get(b)

    def for_subset(self, subset_index: int) -> "SharedInputPlan":
        """Restrict every group to the modules that live in ``subset_index``."""
        groups: List[SharedInputGroup] = []
        for g in self.groups:
            keep = [(m, s) for m, s in zip(g.modules, g.subset_indices) if s == subset_index]
            if keep:
                groups.append(
                    SharedInputGroup(
                        key=g.key,
                        parent=g.parent,
                        modules=tuple(m for m, _ in keep),
                        subset_indices=tuple(s for _, s in keep),
                        explicit=g.explicit,
                    )
                )
        return SharedInputPlan(groups=tuple(groups))

    def filter_modules(self, module_names: Iterable[str]) -> "SharedInputPlan":
        """Keep only modules present in ``module_names`` (e.g. those that exist in a live layer)."""
        allowed = set(module_names)
        groups: List[SharedInputGroup] = []
        for g in self.groups:
            keep = [(m, s) for m, s in zip(g.modules, g.subset_indices) if m in allowed]
            if keep:
                groups.append(
                    SharedInputGroup(
                        key=g.key,
                        parent=g.parent,
                        modules=tuple(m for m, _ in keep),
                        subset_indices=tuple(s for _, s in keep),
                        explicit=g.explicit,
                    )
                )
        return SharedInputPlan(groups=tuple(groups))

    def with_prefix(self, prefix: str) -> "SharedInputPlan":
        """Return a plan whose module names are prefixed (e.g. ``model.layers.3.``)."""
        if not prefix:
            return self
        if not prefix.endswith("."):
            prefix = prefix + "."
        return SharedInputPlan(
            groups=tuple(
                SharedInputGroup(
                    key=prefix + g.key,
                    parent=prefix + g.parent if g.parent else prefix.rstrip("."),
                    modules=tuple(prefix + m for m in g.modules),
                    subset_indices=g.subset_indices,
                    explicit=g.explicit,
                )
                for g in self.groups
            )
        )


def _strip_flags(name: str) -> str:
    return name.split(":", 1)[0]


def build_shared_input_plan(
    module_tree: Any,
    layer_modules: Sequence[Sequence[str]],
) -> SharedInputPlan:
    """Derive the shared-input plan for expanded ``layer_modules`` (subset blocks).

    ``layer_modules`` must already be expert-expanded and filtered the same way the
    looper sees them (``BaseQModel.simple_layer_modules``). Entries flagged ``:!``/``:?``
    are ignored. Modules without an ``:in=<tag>`` become singleton groups keyed by their
    own path; tagged modules group under ``<parent>:in=<tag>``.
    """
    specs = collect_leaf_specs(module_tree)

    order: List[str] = []
    members: Dict[str, List[Tuple[str, int]]] = {}
    meta: Dict[str, Tuple[str, bool]] = {}
    seen_modules: set[str] = set()

    for subset_index, block in enumerate(layer_modules):
        for raw in block:
            if ":!" in raw or ":?" in raw:
                continue
            name = _strip_flags(raw)
            if name in seen_modules:
                continue
            seen_modules.add(name)

            spec = resolve_leaf_spec(specs, name)
            if spec is not None and not spec.quantize:
                continue

            parent = name.rsplit(".", 1)[0] if "." in name else ""
            if spec is not None and spec.input_tag is not None:
                key = f"{parent}:{SHARED_INPUT_FLAG_PREFIX}{spec.input_tag}"
                explicit = True
            else:
                key = name
                explicit = False

            if key not in members:
                order.append(key)
                members[key] = []
                meta[key] = (parent, explicit)
            members[key].append((name, subset_index))

    groups = tuple(
        SharedInputGroup(
            key=key,
            parent=meta[key][0],
            modules=tuple(m for m, _ in members[key]),
            subset_indices=tuple(s for _, s in members[key]),
            explicit=meta[key][1],
        )
        for key in order
    )
    return SharedInputPlan(groups=groups)


# --------------------------------------------------------------------------- #
# CPU / tiny-model verification probe
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SharedInputMismatch:
    group_key: str
    leader: str
    module: str
    reason: str


@dataclass(frozen=True)
class SharedInputProbeReport:
    """Outcome of :func:`probe_shared_inputs` for one forward pass."""

    verified: Tuple[str, ...]
    mismatches: Tuple[SharedInputMismatch, ...]
    unverified: Tuple[str, ...]
    undeclared: Tuple[Tuple[str, str], ...]
    missing_modules: Tuple[str, ...]
    call_counts: Dict[str, int]

    @property
    def has_errors(self) -> bool:
        """A declared group received different inputs, or undeclared modules received identical ones."""
        return bool(self.mismatches or self.undeclared)

    @property
    def fully_verified(self) -> bool:
        """No errors *and* every planned module exists and every group was exercised."""
        return not self.has_errors and not self.missing_modules and not self.unverified

    @property
    def ok(self) -> bool:
        """Strict gate: alias of :attr:`fully_verified`.

        Use :attr:`has_errors` when unexercised groups are expected (e.g. un-routed MoE experts).
        """
        return self.fully_verified

    def describe(self) -> str:
        lines = [
            f"verified={list(self.verified)}",
            f"unverified={list(self.unverified)}",
            f"missing_modules={list(self.missing_modules)}",
        ]
        for m in self.mismatches:
            lines.append(f"MISMATCH {m.group_key}: {m.module} != {m.leader} ({m.reason})")
        for a, b in self.undeclared:
            lines.append(f"UNDECLARED shared input: {a} ~ {b}")
        return "\n".join(lines)


def _first_tensor(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Optional[torch.Tensor]:
    for value in args:
        if torch.is_tensor(value):
            return value
    for value in kwargs.values():
        if torch.is_tensor(value):
            return value
    return None


def _same_captures(a: List[torch.Tensor], b: List[torch.Tensor]) -> Optional[str]:
    if len(a) != len(b):
        return f"call count {len(b)} != {len(a)}"
    for idx, (x, y) in enumerate(zip(a, b)):
        if x.shape != y.shape:
            return f"call {idx}: shape {tuple(y.shape)} != {tuple(x.shape)}"
        if x.dtype != y.dtype:
            return f"call {idx}: dtype {y.dtype} != {x.dtype}"
        if not torch.equal(x, y):
            return f"call {idx}: values differ (max abs diff {float((x.float() - y.float()).abs().max())})"
    return None


def probe_shared_inputs(
    layer: nn.Module,
    plan: SharedInputPlan,
    forward: Callable[[], Any],
) -> SharedInputProbeReport:
    """Run ``forward`` with pre-hooks on every planned module and check the plan holds.

    * ``mismatches``: modules in a group whose captured inputs differ from the leader.
    * ``unverified``: groups whose modules were never invoked (e.g. un-routed experts).
    * ``undeclared``: same-subset, same-parent module pairs in *different* groups whose
      inputs were byte-identical (a missed ``:in=`` opt-in or a wrong tag).
    * ``missing_modules``: planned modules absent from ``layer``.
    """
    live = dict(layer.named_modules())
    lock = threading.Lock()
    captures: Dict[str, List[torch.Tensor]] = {m: [] for m in plan.modules}
    handles = []
    missing: List[str] = []

    def make_hook(name: str):
        def hook(module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
            tensor = _first_tensor(args, kwargs)
            if tensor is None:
                return None
            snapshot = tensor.detach().to("cpu", copy=True)
            with lock:
                captures[name].append(snapshot)
            return None

        return hook

    for name in plan.modules:
        module = live.get(name)
        if module is None:
            missing.append(name)
            continue
        handles.append(module.register_forward_pre_hook(make_hook(name), with_kwargs=True))

    try:
        with torch.inference_mode():
            forward()
    finally:
        for handle in handles:
            handle.remove()

    verified: List[str] = []
    unverified: List[str] = []
    mismatches: List[SharedInputMismatch] = []

    for group in plan.groups:
        present = [m for m in group.modules if m not in missing]
        if not present:
            continue
        called = [m for m in present if captures[m]]
        if not called:
            unverified.append(group.key)
            continue
        leader = called[0]
        group_ok = True
        for module in present:
            if module == leader:
                continue
            reason = _same_captures(captures[leader], captures[module])
            if reason is not None:
                group_ok = False
                mismatches.append(SharedInputMismatch(group.key, leader, module, reason))
        if group_ok:
            verified.append(group.key)

    undeclared: List[Tuple[str, str]] = []
    by_parent_subset: Dict[Tuple[str, int], List[Tuple[str, SharedInputGroup]]] = {}
    for group in plan.groups:
        for module, subset in zip(group.modules, group.subset_indices):
            if module in missing or not captures[module]:
                continue
            by_parent_subset.setdefault((group.parent, subset), []).append((module, group))
    for bucket in by_parent_subset.values():
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                (ma, ga), (mb, gb) = bucket[i], bucket[j]
                if ga is gb:
                    continue
                if _same_captures(captures[ma], captures[mb]) is None:
                    undeclared.append((ma, mb))

    return SharedInputProbeReport(
        verified=tuple(verified),
        mismatches=tuple(mismatches),
        unverified=tuple(unverified),
        undeclared=tuple(undeclared),
        missing_modules=tuple(missing),
        call_counts={m: len(v) for m, v in captures.items()},
    )
