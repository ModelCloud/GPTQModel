# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, crash-consistent inline adjacent-tier EXL3 selection."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


INLINE_MIXED_META_KEY = "ds4rt_inline_mixed"
INLINE_MIXED_NAMESPACES_META_KEY = "ds4rt_inline_mixed_by_namespace"
INLINE_MIXED_SCHEMA = "gptqmodel.exl3-inline-mixed"
INLINE_MIXED_SCHEMA_VERSION = 1
INLINE_MIXED_SCORE = (
    "base-tier-hessian-weighted-relative-error-times-natural-gate-squared-mass-v1"
)
INLINE_MIXED_LEGACY_K2_SCORE = (
    "k2-hessian-weighted-relative-error-times-natural-gate-squared-mass-v1"
)
INLINE_MIXED_CHECKPOINT_ROLE = "inline_mixed"
PROJECTION_ORDER = ("w1", "w3", "w2")
PROJECTION_NAMES = {"w1": "gate", "w3": "up", "w2": "down"}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"EXL3 inline-mixed {label} must be a positive integer")
    return value


@dataclass(frozen=True)
class InlineMixedPolicy:
    """Validated immutable policy for one base or MTP quantization namespace."""

    namespace: str
    base_bits: int
    upgrade_bits: int
    extra_bits: Fraction
    projection_ratio: tuple[int, int, int]
    tier_plan_root: Path
    logical_layer_start: int = 0
    logical_layer_count: int | None = None
    score_kind: str = INLINE_MIXED_SCORE

    @property
    def target_bpw(self) -> Fraction:
        return Fraction(self.base_bits, 1) + self.extra_bits

    @property
    def policy_body(self) -> dict[str, Any]:
        body = {
            "schema": INLINE_MIXED_SCHEMA,
            "schema_version": INLINE_MIXED_SCHEMA_VERSION,
            "namespace": self.namespace,
            "base_bits": self.base_bits,
            "upgrade_bits": self.upgrade_bits,
            "extra_bits": {
                "numerator": self.extra_bits.numerator,
                "denominator": self.extra_bits.denominator,
            },
            "target_bpw": (
                f"{self.target_bpw.numerator}/{self.target_bpw.denominator}"
            ),
            "projection_ratio": {
                projection: ratio
                for projection, ratio in zip(PROJECTION_ORDER, self.projection_ratio)
            },
            "score_kind": self.score_kind,
        }
        if self.logical_layer_count is not None:
            body["logical_layer_start"] = self.logical_layer_start
            body["logical_layer_count"] = self.logical_layer_count
        return body

    @property
    def policy_sha256(self) -> str:
        return sha256_json(self.policy_body)

    def namespace_quotas(
        self, *, layer_count: int, experts_per_layer: int
    ) -> dict[str, int]:
        """Return exact largest-remainder projection totals for the namespace."""

        _positive_integer(layer_count, "layer_count")
        _positive_integer(experts_per_layer, "experts_per_layer")
        if self.upgrade_bits - self.base_bits != 1:
            raise ValueError("EXL3 inline-mixed currently requires adjacent K tiers")
        candidate_count = layer_count * experts_per_layer * len(PROJECTION_ORDER)
        ideal_upgrades = Fraction(candidate_count) * self.extra_bits
        # Nearest representable projection count, with deterministic half-up ties.
        total_upgrades = (
            2 * ideal_upgrades.numerator + ideal_upgrades.denominator
        ) // (2 * ideal_upgrades.denominator)
        if not 0 <= total_upgrades <= candidate_count:
            raise ValueError(
                "EXL3 inline-mixed target is outside the adjacent-tier range"
            )

        ratio_sum = sum(self.projection_ratio)
        ideals = {
            projection: Fraction(total_upgrades * ratio, ratio_sum)
            for projection, ratio in zip(PROJECTION_ORDER, self.projection_ratio)
        }
        quotas = {
            projection: ideal.numerator // ideal.denominator
            for projection, ideal in ideals.items()
        }
        remainder = total_upgrades - sum(quotas.values())
        ranked = sorted(
            PROJECTION_ORDER,
            key=lambda projection: (
                -(ideals[projection] - quotas[projection]),
                PROJECTION_ORDER.index(projection),
            ),
        )
        for projection in ranked[:remainder]:
            quotas[projection] += 1
        if any(quota > layer_count * experts_per_layer for quota in quotas.values()):
            raise ValueError(
                "EXL3 inline-mixed projection ratio exceeds one projection class"
            )
        return quotas

    def layer_quotas(
        self,
        *,
        layer_index: int,
        layer_count: int,
        experts_per_layer: int,
    ) -> dict[str, int]:
        """Spread exact namespace quotas causally across decoder layers."""

        quota_layer_count = self.logical_layer_count or layer_count
        quota_layer_index = layer_index - self.logical_layer_start
        if (
            (self.logical_layer_count is None and self.logical_layer_start != 0)
            or self.logical_layer_start + quota_layer_count > layer_count
            or not 0 <= quota_layer_index < quota_layer_count
        ):
            raise ValueError("EXL3 inline-mixed layer index is invalid")
        totals = self.namespace_quotas(
            layer_count=quota_layer_count,
            experts_per_layer=experts_per_layer,
        )
        return {
            projection: (
                ((quota_layer_index + 1) * total) // quota_layer_count
                - (quota_layer_index * total) // quota_layer_count
            )
            for projection, total in totals.items()
        }


def inline_mixed_policy(meta: Any, *, namespace: str | None = None) -> InlineMixedPolicy | None:
    """Parse the private exact-rational policy without accepting float BPW."""

    requested_namespace = namespace
    if isinstance(meta, dict) and INLINE_MIXED_NAMESPACES_META_KEY in meta:
        raw_policies = meta[INLINE_MIXED_NAMESPACES_META_KEY]
        if (
            INLINE_MIXED_META_KEY in meta
            or not isinstance(raw_policies, dict)
            or not raw_policies
            or not set(raw_policies).issubset({"base", "mtp"})
        ):
            raise ValueError("EXL3 inline-mixed namespace policies are invalid")
        policies = {
            name: inline_mixed_policy({INLINE_MIXED_META_KEY: value})
            for name, value in raw_policies.items()
        }
        if any(policy.namespace != name for name, policy in policies.items()):
            raise ValueError("EXL3 inline-mixed namespace policy identity differs")
        if len({(p.base_bits, p.upgrade_bits) for p in policies.values()}) != 1:
            raise ValueError("EXL3 integrated namespaces require the same candidate tiers")
        # The unscoped query establishes shared candidate-journal and Hessian
        # retention policy. Projection work always selects its own namespace.
        if namespace is None:
            return policies.get("base", next(iter(policies.values())))
        return policies.get(namespace)
    if not isinstance(meta, dict) or INLINE_MIXED_META_KEY not in meta:
        return None
    raw = meta[INLINE_MIXED_META_KEY]
    if not isinstance(raw, dict):
        raise ValueError("EXL3 inline-mixed metadata must be an object")
    extra = raw.get("extra_bits")
    ratio = raw.get("projection_ratio")
    root = raw.get("tier_plan_root")
    namespace = raw.get("namespace")
    # Before adjacent tiers were generalized, an omitted score kind meant the
    # K2-labelled scoring contract. Preserve that exact policy hash on resume.
    score_kind = raw.get("score_kind", INLINE_MIXED_LEGACY_K2_SCORE)
    logical_layer_start = raw.get("logical_layer_start", 0)
    logical_layer_count = raw.get("logical_layer_count")
    if (
        raw.get("schema") != INLINE_MIXED_SCHEMA
        or raw.get("schema_version") != INLINE_MIXED_SCHEMA_VERSION
        or namespace not in {"base", "mtp"}
        or not isinstance(extra, dict)
        or not isinstance(ratio, dict)
        or set(ratio) != set(PROJECTION_ORDER)
        or not isinstance(root, str)
        or not root
        or score_kind not in {INLINE_MIXED_SCORE, INLINE_MIXED_LEGACY_K2_SCORE}
        or isinstance(logical_layer_start, bool)
        or not isinstance(logical_layer_start, int)
        or logical_layer_start < 0
        or (logical_layer_start != 0 and logical_layer_count is None)
        or (
            logical_layer_count is not None
            and (
                isinstance(logical_layer_count, bool)
                or not isinstance(logical_layer_count, int)
                or logical_layer_count <= 0
            )
        )
    ):
        raise ValueError("EXL3 inline-mixed metadata has an invalid contract")
    numerator = _positive_integer(extra.get("numerator"), "extra numerator")
    denominator = _positive_integer(extra.get("denominator"), "extra denominator")
    fraction = Fraction(numerator, denominator)
    if not 0 < fraction < 1:
        raise ValueError("EXL3 inline-mixed extra bits must be between zero and one")
    base_bits = _positive_integer(raw.get("base_bits"), "base_bits")
    upgrade_bits = _positive_integer(raw.get("upgrade_bits"), "upgrade_bits")
    policy = InlineMixedPolicy(
        namespace=namespace,
        base_bits=base_bits,
        upgrade_bits=upgrade_bits,
        extra_bits=fraction,
        projection_ratio=tuple(
            _positive_integer(ratio[projection], f"{projection} ratio")
            for projection in PROJECTION_ORDER
        ),
        tier_plan_root=Path(root).expanduser().resolve(),
        logical_layer_start=logical_layer_start,
        logical_layer_count=logical_layer_count,
        score_kind=score_kind,
    )
    if policy.upgrade_bits != policy.base_bits + 1:
        raise ValueError("EXL3 inline-mixed requires adjacent integer tiers")
    if requested_namespace is not None and policy.namespace != requested_namespace:
        return None
    return policy


def projection_score(record: dict[str, Any]) -> float:
    """Compute the accepted base-tier risk proxy from one projection ledger."""

    metrics = record.get("quantizer_metrics")
    routes = record.get("route_evidence")
    error = (
        metrics.get("hessian_weighted_relative_error")
        if isinstance(metrics, dict)
        else None
    )
    gate_mass = (
        routes.get("expert_gate_squared_mass_fraction")
        if isinstance(routes, dict)
        else None
    )
    if (
        not isinstance(error, (int, float))
        or isinstance(error, bool)
        or not math.isfinite(float(error))
        or float(error) < 0
        or not isinstance(gate_mass, (int, float))
        or isinstance(gate_mass, bool)
        or not math.isfinite(float(gate_mass))
        or not 0 <= float(gate_mass) <= 1
    ):
        raise ValueError("EXL3 inline-mixed candidate lacks finite error/route evidence")
    return float(error) * float(gate_mass)


def build_layer_tier_plan(
    *,
    policy: InlineMixedPolicy,
    layer_index: int,
    layer_count: int,
    candidate_records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Select the highest-risk base-tier projections under fixed quotas."""

    records = list(candidate_records)
    identities: dict[tuple[int, str], tuple[dict[str, Any], float]] = {}
    experts: set[int] = set()
    for record in records:
        expert = record.get("expert")
        projection = record.get("projection")
        if (
            record.get("record_kind") != "projection"
            or record.get("block_namespace") != policy.namespace
            or record.get("logical_layer") != layer_index
            or record.get("bits") != policy.base_bits
            or isinstance(expert, bool)
            or not isinstance(expert, int)
            or expert < 0
            or projection not in PROJECTION_ORDER
        ):
            raise ValueError("EXL3 inline-mixed candidate identity is invalid")
        key = (expert, projection)
        if key in identities:
            raise ValueError("EXL3 inline-mixed candidate projection is duplicated")
        identities[key] = (record, projection_score(record))
        experts.add(expert)
    if not experts or experts != set(range(max(experts) + 1)):
        raise ValueError("EXL3 inline-mixed expert IDs are incomplete")
    experts_per_layer = len(experts)
    if len(identities) != experts_per_layer * len(PROJECTION_ORDER):
        raise ValueError("EXL3 inline-mixed layer has incomplete projection coverage")

    quotas = policy.layer_quotas(
        layer_index=layer_index,
        layer_count=layer_count,
        experts_per_layer=experts_per_layer,
    )
    selected: list[dict[str, Any]] = []
    for projection in PROJECTION_ORDER:
        ranked = sorted(
            (
                (score, expert, record)
                for (expert, candidate_projection), (record, score) in identities.items()
                if candidate_projection == projection
            ),
            key=lambda item: (-item[0], item[1]),
        )
        for score, expert, record in ranked[: quotas[projection]]:
            selected.append(
                {
                    "module": record["module"],
                    "expert": expert,
                    "projection": projection,
                    "score": score,
                    "candidate_record_sha256": sha256_json(record),
                }
            )
    selected.sort(key=lambda item: (PROJECTION_ORDER.index(item["projection"]), item["expert"]))
    body = {
        **policy.policy_body,
        "policy_sha256": policy.policy_sha256,
        "layer_index": layer_index,
        "layer_count": layer_count,
        "experts_per_layer": experts_per_layer,
        "quotas": quotas,
        "selected": selected,
    }
    return {**body, "tier_plan_sha256": sha256_json(body)}


class InlineMixedTierPlanStore:
    """Atomically publish immutable per-layer maps before upgraded-tier work."""

    def __init__(self, policy: InlineMixedPolicy) -> None:
        self.policy = policy
        self.root = policy.tier_plan_root

    def path(self, layer_index: int) -> Path:
        if layer_index < 0:
            raise ValueError("EXL3 inline-mixed layer index is invalid")
        return self.root / self.policy.namespace / f"layer-{layer_index:06d}.json"

    def load(
        self,
        *,
        layer_index: int,
        layer_count: int,
        experts_per_layer: int,
    ) -> dict[str, Any] | None:
        """Authenticate one committed tier plan without trusting its filename.

        Projection checkpoints are independently content addressed, but the
        layer plan is the authority that chooses which base-tier candidates
        acquire an upgraded replacement. Recovery validates the complete policy,
        geometry, quota, selection, and digest contract before it uses any
        selected checkpoint.  A missing plan is an incomplete layer; an
        existing malformed plan is corruption and must never become a fresh
        selection on resume.
        """

        path = self.path(layer_index)
        if not path.exists():
            return None
        if not path.is_file() or path.is_symlink():
            raise ValueError("EXL3 inline-mixed tier plan is not a regular file")
        try:
            plan = json.loads(path.read_bytes())
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("EXL3 inline-mixed tier plan cannot be read") from error
        if not isinstance(plan, dict):
            raise ValueError("EXL3 inline-mixed tier plan is not an object")

        expected_keys = {
            *self.policy.policy_body,
            "policy_sha256",
            "layer_index",
            "layer_count",
            "experts_per_layer",
            "quotas",
            "selected",
            "tier_plan_sha256",
        }
        body = {key: value for key, value in plan.items() if key != "tier_plan_sha256"}
        quotas = self.policy.layer_quotas(
            layer_index=layer_index,
            layer_count=layer_count,
            experts_per_layer=experts_per_layer,
        )
        if (
            set(plan) != expected_keys
            or any(
                plan.get(key) != value
                for key, value in self.policy.policy_body.items()
            )
            or plan.get("policy_sha256") != self.policy.policy_sha256
            or plan.get("layer_index") != layer_index
            or plan.get("layer_count") != layer_count
            or plan.get("experts_per_layer") != experts_per_layer
            or plan.get("quotas") != quotas
            or not isinstance(plan.get("tier_plan_sha256"), str)
            or sha256_json(body) != plan["tier_plan_sha256"]
        ):
            raise ValueError("EXL3 inline-mixed tier plan failed validation")

        selected = plan.get("selected")
        if not isinstance(selected, list) or len(selected) != sum(quotas.values()):
            raise ValueError("EXL3 inline-mixed tier plan selection count is invalid")
        observed = {projection: 0 for projection in PROJECTION_ORDER}
        identities: set[tuple[int, str]] = set()
        modules: set[str] = set()
        normalized_order: list[tuple[int, int]] = []
        hex_chars = frozenset("0123456789abcdef")
        for entry in selected:
            if not isinstance(entry, dict) or set(entry) != {
                "module",
                "expert",
                "projection",
                "score",
                "candidate_record_sha256",
            }:
                raise ValueError("EXL3 inline-mixed tier selection is malformed")
            module = entry.get("module")
            expert = entry.get("expert")
            projection = entry.get("projection")
            score = entry.get("score")
            candidate_digest = entry.get("candidate_record_sha256")
            if (
                not isinstance(module, str)
                or not module
                or isinstance(expert, bool)
                or not isinstance(expert, int)
                or not 0 <= expert < experts_per_layer
                or projection not in observed
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or float(score) < 0
                or not isinstance(candidate_digest, str)
                or len(candidate_digest) != 64
                or any(char not in hex_chars for char in candidate_digest)
                or (expert, projection) in identities
                or module in modules
            ):
                raise ValueError("EXL3 inline-mixed tier selection is invalid")
            identities.add((expert, projection))
            modules.add(module)
            observed[projection] += 1
            normalized_order.append((PROJECTION_ORDER.index(projection), expert))
        if observed != quotas or normalized_order != sorted(normalized_order):
            raise ValueError("EXL3 inline-mixed tier selection violates its quotas")
        return plan

    def commit(self, plan: dict[str, Any]) -> dict[str, Any]:
        layer_index = plan.get("layer_index")
        digest = plan.get("tier_plan_sha256")
        body = {key: value for key, value in plan.items() if key != "tier_plan_sha256"}
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or not isinstance(digest, str)
            or sha256_json(body) != digest
            or plan.get("policy_sha256") != self.policy.policy_sha256
            or plan.get("namespace") != self.policy.namespace
        ):
            raise ValueError("EXL3 inline-mixed tier plan is invalid")
        path = self.path(layer_index)
        payload = canonical_json_bytes(plan) + b"\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.parent.is_symlink():
            raise ValueError("EXL3 inline-mixed tier-plan root is unsafe")
        if path.exists():
            if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
                raise ValueError("EXL3 inline-mixed tier plan changed on resume")
            return plan
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        try:
            os.fchmod(descriptor, 0o644)
            with os.fdopen(descriptor, "wb") as target:
                target.write(payload)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary_name, path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return plan


__all__ = [
    "INLINE_MIXED_CHECKPOINT_ROLE",
    "INLINE_MIXED_META_KEY",
    "INLINE_MIXED_LEGACY_K2_SCORE",
    "INLINE_MIXED_SCHEMA",
    "INLINE_MIXED_SCHEMA_VERSION",
    "INLINE_MIXED_SCORE",
    "InlineMixedPolicy",
    "InlineMixedTierPlanStore",
    "PROJECTION_NAMES",
    "PROJECTION_ORDER",
    "build_layer_tier_plan",
    "inline_mixed_policy",
    "projection_score",
    "sha256_json",
]
