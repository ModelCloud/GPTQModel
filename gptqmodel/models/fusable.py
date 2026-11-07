# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Definition-owned, immutable contracts for optional projection fusion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import import_module
from types import MappingProxyType


@dataclass(frozen=True)
class FusableProfile:
    """A sibling role group and the native parent types allowed to execute it.

    Dotted type names are resolved lazily, so declaring a profile does not import
    an architecture's Transformers implementation during package import.
    """

    name: str
    roles: tuple[str, ...]
    parent_types: tuple[type | str, ...]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Fusion profile name must be a nonempty string")
        if isinstance(self.roles, str) or isinstance(self.parent_types, str):
            raise TypeError("Fusion roles and parent_types must be sequences, not strings")
        roles, parents = tuple(self.roles), tuple(self.parent_types)
        if len(roles) < 2 or any(not isinstance(role, str) or not role for role in roles) or len(set(roles)) != len(roles):
            raise ValueError("Fusion profile requires at least two distinct nonempty roles")
        if not parents or any(not isinstance(parent, (type, str)) or isinstance(parent, str)
                              and (not parent or "." not in parent) for parent in parents):
            raise ValueError("Fusion parent_types must contain classes or dotted type names")
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "parent_types", parents)

    @lru_cache(maxsize=64)
    def resolve_parent_types(self):
        result = []
        for parent in self.parent_types:
            if isinstance(parent, str):
                module, _, name = parent.rpartition(".")
                parent = getattr(import_module(module), name)
            if not isinstance(parent, type):
                raise TypeError(f"Fusion profile {self.name!r} parent must resolve to a class")
            result.append(parent)
        return tuple(result)


@dataclass(frozen=True)
class FusableConfig:
    """A model definition's opt-in eligibility contract; never enables fusion."""

    model_types: tuple[str, ...] = ()
    profiles: tuple[FusableProfile, ...] = ()
    overrides: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.model_types, str):
            raise TypeError("FusableConfig.model_types must be a sequence, not a string")
        model_types, profiles = tuple(self.model_types), tuple(self.profiles)
        if not model_types or any(not isinstance(name, str) or not name for name in model_types):
            raise ValueError("FusableConfig requires explicit nonempty model_types")
        if len(set(model_types)) != len(model_types):
            raise ValueError("FusableConfig.model_types must be distinct")
        if not profiles or any(not isinstance(profile, FusableProfile) for profile in profiles):
            raise ValueError("FusableConfig requires FusableProfile instances")
        if len({profile.name for profile in profiles}) != len(profiles):
            raise ValueError("FusableConfig profile names must be distinct")
        if not isinstance(self.overrides, Mapping):
            raise TypeError("Fusion overrides must be a mapping of paths or families to bool")
        overrides = dict(self.overrides)
        if any(not isinstance(key, str) or not key or type(value) is not bool for key, value in overrides.items()):
            raise TypeError("Fusion overrides must map nonempty paths or families to bool")
        object.__setattr__(self, "model_types", model_types)
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(self, "overrides", MappingProxyType(overrides))

    def __deepcopy__(self, memo):
        return self  # All instance-owned configuration is immutable.

    def to_dict(self):
        return {"model_types": list(self.model_types), "overrides": dict(self.overrides),
                "profiles": [{"name": profile.name, "roles": list(profile.roles),
                              "parent_types": [parent if isinstance(parent, str) else
                                               f"{parent.__module__}.{parent.__qualname__}"
                                               for parent in profile.parent_types]}
                             for profile in self.profiles]}


@dataclass(frozen=True)
class LlamaFusableConfig(FusableConfig):
    """Validated Llama QKV and gate/up defaults; subclass or replace to extend."""

    model_types: tuple[str, ...] = ("llama",)
    profiles: tuple[FusableProfile, ...] = (
        FusableProfile("qkv", ("q", "k", "v"), ("transformers.models.llama.modeling_llama.LlamaAttention",)),
        FusableProfile("gate_up", ("gate", "up"), ("transformers.models.llama.modeling_llama.LlamaMLP",)),
    )


def require_fusable_config(definition, model_type):
    """Reject unsupported definitions/model types before any fusion mutation."""
    config = getattr(definition, "fusable", None)
    if config is None:
        raise ValueError(f"{definition.__name__} is not fusable: its model definition must declare fusable")
    if not isinstance(config, FusableConfig):
        raise TypeError(f"{definition.__name__}.fusable must be a FusableConfig instance")
    if model_type not in config.model_types:
        raise ValueError(f"{definition.__name__}.fusable does not support model_type={model_type!r}; "
                         f"allowed model_types: {config.model_types}")
    for profile in config.profiles:
        profile.resolve_parent_types()
    return config
