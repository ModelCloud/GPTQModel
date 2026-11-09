# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Definition-owned, immutable contracts for optional projection fusion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType


@dataclass(frozen=True)
class FusableProfile:
    """A sibling role group resolved exclusively through the model's plan."""

    name: str
    roles: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Fusion profile name must be a nonempty string")
        if isinstance(self.roles, str):
            raise TypeError("Fusion roles must be a sequence, not a string")
        roles = tuple(self.roles)
        if len(roles) < 2 or any(not isinstance(role, str) or not role for role in roles) or len(set(roles)) != len(roles):
            raise ValueError("Fusion profile requires at least two distinct nonempty roles")
        object.__setattr__(self, "roles", roles)


def _profiles(value):
    result = tuple(value)
    if any(not isinstance(profile, FusableProfile) for profile in result):
        raise ValueError("FusableConfig requires FusableProfile instances")
    if len({profile.name for profile in result}) != len(result):
        raise ValueError("FusableConfig profile names must be distinct")
    return result


def _origins(*values):
    return tuple(dict.fromkeys(origin for value in values for origin in value))


@dataclass(frozen=True)
class FusableConfig:
    """Immutable target-first composition; an empty builder cannot execute fusion.

    Merge preserves target identity and records unresolved conflicts. Binding a
    definition validates the completed configuration before any model mutation.
    """

    model_types: tuple[str, ...] | str = ()
    profiles: tuple[FusableProfile, ...] = ()
    overrides: Mapping[str, bool] = field(default_factory=dict)
    model_profiles: Mapping[str, tuple[FusableProfile, ...]] = field(default_factory=dict)
    _profile_origins: Mapping = field(default_factory=dict, repr=False)
    _policy_origins: Mapping = field(default_factory=dict, repr=False)
    _layer_overrides: Mapping = field(default_factory=dict, repr=False)
    _local_overrides: frozenset | None = field(default=None, repr=False)
    _replaced_profiles: frozenset = field(default_factory=frozenset, repr=False)
    _profile_conflicts: tuple = field(default=(), repr=False)
    _policy_conflicts: tuple = field(default=(), repr=False)

    def __post_init__(self):
        model_types = (self.model_types,) if isinstance(self.model_types, str) else tuple(self.model_types)
        if not model_types or any(not isinstance(name, str) or not name for name in model_types):
            raise ValueError("FusableConfig requires explicit nonempty model_types")
        if len(set(model_types)) != len(model_types):
            raise ValueError("FusableConfig.model_types must be distinct")
        profiles = _profiles(self.profiles)
        if not isinstance(self.model_profiles, Mapping):
            raise TypeError("Fusion model_profiles must be a mapping")
        variants = {name: _profiles(value) for name, value in self.model_profiles.items()}
        if set(variants) - set(model_types):
            raise ValueError("Fusion model_profiles contains an undeclared model type")
        if not isinstance(self.overrides, Mapping):
            raise TypeError("Fusion overrides must be a mapping of paths or families to bool")
        overrides = dict(self.overrides)
        if any(not isinstance(key, str) or not key or type(value) is not bool for key, value in overrides.items()):
            raise TypeError("Fusion overrides must map nonempty paths or families to bool")
        layers = dict(self._layer_overrides)
        own_origin = (", ".join(model_types),)
        names = {profile.name for group in (profiles, *variants.values()) for profile in group}
        profile_origins = {name: tuple(self._profile_origins.get(name, own_origin)) for name in sorted(names)}
        policy_origins = {key: tuple(self._policy_origins.get(key, own_origin)) for key in (*overrides, *layers)}
        local = frozenset((*overrides, *layers)) if self._local_overrides is None else frozenset(self._local_overrides)
        for name, value in (
            ("model_types", model_types), ("profiles", profiles), ("model_profiles", MappingProxyType(variants)),
            ("overrides", MappingProxyType(overrides)), ("_layer_overrides", MappingProxyType(layers)),
            ("_profile_origins", MappingProxyType(profile_origins)), ("_policy_origins", MappingProxyType(policy_origins)),
            ("_local_overrides", local), ("_replaced_profiles", frozenset(self._replaced_profiles)),
            ("_profile_conflicts", tuple(self._profile_conflicts)), ("_policy_conflicts", tuple(self._policy_conflicts)),
        ):
            object.__setattr__(self, name, value)

    def __deepcopy__(self, memo):
        return self

    def validate(self):
        """Reject incomplete or conflicting composition before installing fusion."""
        if self._profile_conflicts:
            model_type, name, left, right = self._profile_conflicts[0]
            raise ValueError(f"Conflicting fusion profile {name!r} for {model_type!r}: {left.roles} vs {right.roles}; "
                             f"origins: {self.profile_origins(name)}; use replace_profile()")
        if self._policy_conflicts:
            key, left, right = self._policy_conflicts[0]
            raise ValueError(f"Conflicting fusion policy {key!r}: {left} vs {right}; "
                             f"origins: {self._policy_origins[key]}; use override()")
        for model_type in self.model_types:
            if not self.model_profiles.get(model_type, self.profiles):
                raise ValueError(f"FusableConfig for {model_type!r} requires at least one FusableProfile")
        names = {profile.name for group in (self.profiles, *self.model_profiles.values()) for profile in group}
        unknown = {key for key in self.overrides if "." not in key and key not in names}
        unknown.update(family for family, _ in self._layer_overrides if family not in names)
        if unknown:
            raise ValueError(f"Unknown fusion profiles in overrides: {sorted(unknown)}")
        return self

    def profiles_for(self, model_type):
        """Select the explicit runtime alias's fully validated role profiles."""
        if model_type not in self.model_types:
            raise ValueError(f"FusableConfig does not support model_type={model_type!r}")
        self.validate()
        return self.model_profiles.get(model_type, self.profiles)

    def profile_origins(self, name):
        return self._profile_origins.get(name, ())

    def merge(self, *sources, profiles=None, source_model_type=None):
        """Merge validated sources without importing their model identities.

        Named-profile and alias selection require one source. Conflicts remain
        explicit until replace_profile()/override() resolves them or validate()
        rejects the completed composition. There is no last-source-wins rule.
        """
        if not sources or any(not isinstance(source, FusableConfig) for source in sources):
            raise TypeError("merge() requires one or more FusableConfig instances")
        if len(sources) != 1 and (profiles is not None or source_model_type is not None):
            raise ValueError("Profile or alias selection requires exactly one merge source")
        names = None
        if profiles is not None:
            if isinstance(profiles, (str, Mapping)):
                raise TypeError("Fusion profiles must be a sequence of distinct profile names")
            names = tuple(profiles)
            if not names or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
                raise ValueError("Fusion profiles must be nonempty, distinct profile names")
        result = self
        for source in sources:
            source.validate()

            def selected(model_type=None):
                alias = source_model_type if source_model_type is not None else (
                    model_type if model_type in source.model_types else None)
                group = source.profiles_for(alias) if alias is not None else source.profiles
                if names is None:
                    return group
                known = {profile.name: profile for profile in group}
                if set(names) - known.keys():
                    raise ValueError(f"Unknown fusion profiles: {sorted(set(names) - known.keys())}")
                return tuple(known[name] for name in names)

            conflicts = list(result._profile_conflicts)

            def combine(left, right, model_type):
                merged = {profile.name: profile for profile in left}
                for profile in right:
                    previous = merged.get(profile.name)
                    if previous is None:
                        merged[profile.name] = profile
                    elif (previous != profile and profile.name not in result._replaced_profiles
                          and model_type is not None):
                        conflicts.append((model_type, profile.name, previous, profile))
                return tuple(merged.values())

            # Default profiles may be unused when every target has an explicit
            # alias variant. Only actual admitted aliases create obligations.
            combined = combine(result.profiles, selected(), None)
            variants = {}
            for model_type in result.model_types:
                variant = combine(result.model_profiles.get(model_type, result.profiles), selected(model_type), model_type)
                if variant != combined:
                    variants[model_type] = variant
            origins = dict(result._profile_origins)
            contributed = {profile.name for group in (selected(), *(selected(name) for name in result.model_types))
                           for profile in group}
            for name in contributed:
                if name not in result._replaced_profiles:
                    origins[name] = _origins(origins.get(name, ()), source.profile_origins(name))
            policy, layers = dict(result.overrides), dict(result._layer_overrides)
            policy_origins = dict(result._policy_origins)
            policy_conflicts = list(result._policy_conflicts)
            for key, value in (*source.overrides.items(), *source._layer_overrides.items()):
                family = key[0] if isinstance(key, tuple) else key
                if names is not None and family not in names:
                    continue
                if key in result._local_overrides:
                    continue
                destination = layers if isinstance(key, tuple) else policy
                if key in destination and destination[key] != value:
                    policy_conflicts.append((key, destination[key], value))
                else:
                    destination[key] = value
                policy_origins[key] = _origins(policy_origins.get(key, ()), source._policy_origins[key])
            result = replace(result, profiles=combined, model_profiles=variants, overrides=policy,
                             _layer_overrides=layers, _profile_origins=origins, _policy_origins=policy_origins,
                             _profile_conflicts=tuple(dict.fromkeys(conflicts)),
                             _policy_conflicts=tuple(dict.fromkeys(policy_conflicts)))
        return result

    def profile(self, name, *, roles):
        """Add a new local profile; an existing name needs replace_profile()."""
        profile = FusableProfile(name, roles)
        if name in self._profile_origins:
            raise ValueError(f"Fusion profile {name!r} already exists; use replace_profile()")
        return replace(self, profiles=(*self.profiles, profile),
                       model_profiles={alias: (*group, profile) for alias, group in self.model_profiles.items()})

    def replace_profile(self, name, *, roles):
        """Explicitly replace a named profile for every target alias."""
        profile = FusableProfile(name, roles)
        if name not in self._profile_origins:
            raise ValueError(f"Unknown fusion profile {name!r}")

        def substitute(group):
            return tuple(profile if value.name == name else value for value in group)

        return replace(self, profiles=substitute(self.profiles),
                       model_profiles={alias: substitute(group) for alias, group in self.model_profiles.items()},
                       _profile_origins={**self._profile_origins, name: (", ".join(self.model_types),)},
                       _replaced_profiles=self._replaced_profiles | {name},
                       _profile_conflicts=tuple(conflict for conflict in self._profile_conflicts if conflict[1] != name))

    def override(self, family, *, enabled, layers=None):
        """Set explicit local family policy, optionally scoped to layer indices."""
        if not isinstance(family, str) or not family or "." in family or type(enabled) is not bool:
            raise TypeError("override() requires a profile name and a bool enabled value")
        if layers is None:
            keys = (family,)
        else:
            if isinstance(layers, (str, Mapping)):
                raise TypeError("Fusion layers must be a sequence of nonnegative integer indices")
            indices = tuple(layers)
            if not indices or any(type(index) is not int or index < 0 for index in indices):
                raise ValueError("Fusion layers must be nonempty nonnegative integer indices")
            keys = tuple((family, index) for index in dict.fromkeys(indices))
        policy, scoped = dict(self.overrides), dict(self._layer_overrides)
        for key in keys:
            destination = scoped if isinstance(key, tuple) else policy
            if key in self._local_overrides and destination[key] != enabled:
                raise ValueError(f"Conflicting local fusion policy {key!r}")
            destination[key] = enabled
        return replace(self, overrides=policy, _layer_overrides=scoped,
                       _local_overrides=self._local_overrides | set(keys),
                       _policy_origins={**self._policy_origins, **{key: (", ".join(self.model_types),) for key in keys}},
                       _policy_conflicts=tuple(conflict for conflict in self._policy_conflicts if conflict[0] not in keys))

    def policy_for(self, family, layer_index, *, parent, full_parent, overrides=None):
        """Return enabled state and the winning selector for a compiled group."""
        for mapping, scope in ((overrides or {}, "runtime"), (self.overrides, "definition")):
            for key in (full_parent, parent):
                if key in mapping:
                    return mapping[key], f"{scope}: {key}"
            if scope == "definition" and (family, layer_index) in self._layer_overrides:
                key = (family, layer_index)
                return self._layer_overrides[key], f"layer: {family}[{layer_index}] from {self._policy_origins[key]}"
            if family in mapping:
                return mapping[family], f"{scope}: {family}"
        return True, "default: enabled"

    def to_dict(self):
        def serialize(profiles):
            return [{"name": profile.name, "roles": list(profile.roles)} for profile in profiles]

        return {"model_types": list(self.model_types), "overrides": dict(self.overrides),
                "profiles": serialize(self.profiles),
                "model_profiles": {name: serialize(value) for name, value in self.model_profiles.items()},
                "profile_origins": {name: list(origins) for name, origins in self._profile_origins.items()},
                "layer_overrides": [{"profile": family, "layer": index, "enabled": value,
                                     "origins": list(self._policy_origins[(family, index)])}
                                    for (family, index), value in self._layer_overrides.items()],
                "policy_origins": [{"selector": list(key) if isinstance(key, tuple) else key, "origins": list(origins)}
                                   for key, origins in self._policy_origins.items()],
                "conflicts": {
                    "profiles": [{"model_type": alias, "profile": name, "roles": [list(left.roles), list(right.roles)]}
                                 for alias, name, left, right in self._profile_conflicts],
                    "policy": [{"selector": list(key) if isinstance(key, tuple) else key, "values": [left, right]}
                               for key, left, right in self._policy_conflicts],
                }}


@dataclass(frozen=True)
class LlamaFusableConfig(FusableConfig):
    """Validated Llama QKV and gate/up defaults; merge to reuse their policy."""

    model_types: tuple[str, ...] = ("llama",)
    profiles: tuple[FusableProfile, ...] = (
        FusableProfile("qkv", ("q", "k", "v")),
        FusableProfile("gate_up", ("gate", "up")),
    )


def require_fusable_config(definition, model_type):
    """Reject unsupported or unresolved definitions before any fusion mutation."""
    config = getattr(definition, "fusable", None)
    if config is None:
        raise ValueError(f"{definition.__name__} is not fusable: its model definition must declare fusable")
    if not isinstance(config, FusableConfig):
        raise TypeError(f"{definition.__name__}.fusable must be a FusableConfig instance")
    if model_type not in config.model_types:
        raise ValueError(f"{definition.__name__}.fusable does not support model_type={model_type!r}; "
                         f"allowed model_types: {config.model_types}")
    return config.validate()
