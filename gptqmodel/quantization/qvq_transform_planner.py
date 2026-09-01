# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Graph-level transform planning for gated QVQ folding experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import torch


class ProjectionRole(str, Enum):
    ATTENTION_Q = "attention_q"
    ATTENTION_K = "attention_k"
    ATTENTION_V = "attention_v"
    ATTENTION_O = "attention_o"
    MLP_GATE = "mlp_gate"
    MLP_UP = "mlp_up"
    MLP_DOWN = "mlp_down"


class TransformKind(str, Enum):
    HADAMARD = "hadamard"
    STRUCTURED_ORTHOGONAL = "structured_orthogonal"
    ROPE_PAIR = "rope_pair"
    PERMUTATION = "permutation"
    DIAGONAL = "diagonal"
    IDENTITY = "identity"


class TransformPlacement(str, Enum):
    ONLINE = "online"
    SHARED = "shared"
    FUSED = "fused"
    FOLDED = "folded"
    NONE = "none"


@dataclass(frozen=True)
class TransformSpec:
    kind: TransformKind
    placement: TransformPlacement
    basis_id: str | None = None

    @property
    def is_online_full_hadamard(self) -> bool:
        return self.kind == TransformKind.HADAMARD and self.placement in {
            TransformPlacement.ONLINE,
            TransformPlacement.SHARED,
        }

    @property
    def is_online(self) -> bool:
        return self.placement in {
            TransformPlacement.ONLINE,
            TransformPlacement.SHARED,
            TransformPlacement.FUSED,
        }

    @property
    def is_shared(self) -> bool:
        return self.placement == TransformPlacement.SHARED


ONLINE_HADAMARD = TransformSpec(TransformKind.HADAMARD, TransformPlacement.ONLINE)
IDENTITY = TransformSpec(TransformKind.IDENTITY, TransformPlacement.NONE)


@dataclass(frozen=True)
class ProjectionSemantic:
    name: str
    layer_index: int
    role: ProjectionRole
    in_features: int
    out_features: int
    module: torch.nn.Module = field(compare=False, repr=False)


@dataclass(frozen=True)
class ModuleTransformDescriptor:
    module_name: str
    role: ProjectionRole
    input_transform: TransformSpec
    output_transform: TransformSpec
    local_input_scale: bool = True
    local_output_scale: bool = True


@dataclass(frozen=True)
class QVQTransformPlan:
    arm: str
    description: str
    modules: tuple[ModuleTransformDescriptor, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def online_hadamards_per_block(self) -> int:
        return self._online_transform_count(full_hadamard=True) // max(1, self.layer_count)

    @property
    def other_online_transforms_per_block(self) -> int:
        return self._online_transform_count(full_hadamard=False) // max(1, self.layer_count)

    def _online_transform_count(self, *, full_hadamard: bool) -> int:
        unique = set()
        for descriptor in self.modules:
            for side, transform in (
                ("input", descriptor.input_transform),
                ("output", descriptor.output_transform),
            ):
                selected = (
                    transform.is_online_full_hadamard
                    if full_hadamard
                    else transform.is_online and not transform.is_online_full_hadamard
                )
                if not selected:
                    continue
                if transform.is_shared:
                    if not transform.basis_id:
                        raise ValueError("shared QVQ transforms require a basis_id")
                    unique.add((transform.kind, transform.placement, transform.basis_id))
                else:
                    unique.add((descriptor.module_name, side))
        return len(unique)

    @property
    def folded_transforms_per_block(self) -> int:
        return sum(
            descriptor.input_transform.placement == TransformPlacement.FOLDED
            for descriptor in self.modules
        ) // max(1, self.layer_count) + sum(
            descriptor.output_transform.placement == TransformPlacement.FOLDED
            for descriptor in self.modules
        ) // max(1, self.layer_count)

    @property
    def layer_count(self) -> int:
        indices = {descriptor.module_name.rsplit(".", 1)[0] for descriptor in self.modules}
        return max(1, len(self.modules) // 7) if indices else 0


class QVQArchitectureImplementor(Protocol):
    def analyze_model_graph(self, model: torch.nn.Module) -> tuple[ProjectionSemantic, ...]: ...

    def rewrite_dense_weights(
        self,
        model: torch.nn.Module,
        plan: QVQTransformPlan,
        *,
        seed: int,
    ) -> dict[str, Any]: ...


class QVQTransformPlanner:
    """Build role-aware plans while keeping runtime descriptors role-agnostic."""

    def __init__(self, model: torch.nn.Module, implementor: QVQArchitectureImplementor):
        self.model = model
        self.implementor = implementor
        self._semantics: tuple[ProjectionSemantic, ...] | None = None

    def analyze_model_graph(self) -> tuple[ProjectionSemantic, ...]:
        if self._semantics is None:
            semantics = self.implementor.analyze_model_graph(self.model)
            roles_by_layer: dict[int, set[ProjectionRole]] = {}
            for semantic in semantics:
                roles_by_layer.setdefault(semantic.layer_index, set()).add(semantic.role)
            expected = set(ProjectionRole)
            incomplete = {
                layer: sorted(role.value for role in expected - roles)
                for layer, roles in roles_by_layer.items()
                if roles != expected
            }
            if incomplete:
                raise ValueError(f"QVQ transform graph is missing projection roles: {incomplete}")
            self._semantics = semantics
        return self._semantics

    def build_transform_plan(self, arm: str) -> QVQTransformPlan:
        arm = str(arm).strip().upper()
        semantics = self.analyze_model_graph()
        if arm not in {
            "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A8", "A9",
            "A12", "A13", "A14", "A15", "A16", "A17", "A18",
            "A20", "A21", "A22", "A23", "A24", "A25", "A26", "A27", "A28", "A29", "A30",
            "A31",
        }:
            raise ValueError(f"unsupported QVQ transform-search arm: {arm}")

        residual_kind = (
            TransformKind.STRUCTURED_ORTHOGONAL if arm in {"A2", "A9"} else TransformKind.HADAMARD
        )
        learned_vo = arm in {"A2", "A8"}
        descriptions = {
            "A0": "current per-linear two-sided RHT control",
            "A1": "folded global residual basis and head-local V/O basis",
            "A2": "A1 with learned folded orthogonal bases",
            "A3": "A1/A2 plus RoPE-compatible Q/K pair maps",
            "A4": "A3 plus exact SwiGLU permutation/scaling",
            "A5": "A4 with learned structured online down transform",
            "A6": "strongest exact zero-full-Hadamard candidate",
            "A8": "fixed residual Hadamard plus learned V/O",
            "A9": "learned residual basis plus fixed V/O Hadamard",
            "A12": "A4 with online block-H16 down transform",
            "A13": "A4 with online block-H32 down transform",
            "A14": "A4 with online block-H64 down transform",
            "A15": "A4 with one online pairwise down stage",
            "A16": "A4 with two online pairwise down stages",
            "A17": "A4 with online permutation-only down transform",
            "A18": "Smooth-SwiGLU plus permutation and no online down Hadamard",
            "A20": "A1 plus scale-free RoPE-pair Q/K folding",
            "A21": "A1 plus identity RoPE-pair Q/K folding",
            "A22": "A1 plus exact SwiGLU folding while retaining online Q/K and down Hadamards",
            "A23": "A22 without the online down Hadamard",
            "A24": "exact SwiGLU folding only; retain A0 attention and residual axes",
            "A25": "head-local V/O folding only; retain other A0 axes",
            "A26": "head-local V/O plus exact SwiGLU folding; retain other A0 axes",
            "A27": "permutation-only SwiGLU folding; retain A0 attention and residual axes",
            "A28": "head-local V/O plus permutation-only SwiGLU folding",
            "A29": "remove gate/up output Hadamards without SwiGLU reparameterization",
            "A30": "A29 plus head-local V/O folding",
            "A31": "sibling-shared QKV and gate/up input RHTs plus head-local V/O folding",
        }
        descriptors = []
        for semantic in semantics:
            if arm in {"A0", "A24", "A25", "A26", "A27", "A28", "A29", "A30", "A31"}:
                input_transform = output_transform = ONLINE_HADAMARD
            else:
                residual_in = TransformSpec(
                    residual_kind,
                    TransformPlacement.FOLDED,
                    "residual",
                )
                residual_out = residual_in
                vo_kind = TransformKind.STRUCTURED_ORTHOGONAL if learned_vo else TransformKind.HADAMARD
                vo_folded = TransformSpec(vo_kind, TransformPlacement.FOLDED, f"vo.l{semantic.layer_index}")
                input_transform, output_transform = self._a1_specs(
                    semantic.role,
                    residual_in,
                    residual_out,
                    vo_folded,
                )
                if arm in {
                    "A3", "A4", "A5", "A6", "A12", "A13", "A14", "A15", "A16", "A17", "A18",
                    "A20", "A21",
                } and semantic.role in {ProjectionRole.ATTENTION_Q, ProjectionRole.ATTENTION_K}:
                    output_transform = TransformSpec(
                        TransformKind.ROPE_PAIR,
                        TransformPlacement.FOLDED,
                        f"rope_qk.l{semantic.layer_index}",
                    )
            if arm in {"A25", "A26", "A28", "A30", "A31"}:
                vo_folded = TransformSpec(
                    TransformKind.HADAMARD,
                    TransformPlacement.FOLDED,
                    f"vo.l{semantic.layer_index}",
                )
                if semantic.role == ProjectionRole.ATTENTION_V:
                    output_transform = vo_folded
                elif semantic.role == ProjectionRole.ATTENTION_O:
                    input_transform = vo_folded
            if arm == "A31":
                if semantic.role in {
                    ProjectionRole.ATTENTION_Q,
                    ProjectionRole.ATTENTION_K,
                    ProjectionRole.ATTENTION_V,
                }:
                    input_transform = TransformSpec(
                        TransformKind.HADAMARD,
                        TransformPlacement.SHARED,
                        f"sibling.attn.l{semantic.layer_index}",
                    )
                elif semantic.role in {
                    ProjectionRole.MLP_GATE,
                    ProjectionRole.MLP_UP,
                }:
                    input_transform = TransformSpec(
                        TransformKind.HADAMARD,
                        TransformPlacement.SHARED,
                        f"sibling.mlp.l{semantic.layer_index}",
                    )
            if arm in {"A29", "A30"} and semantic.role in {
                ProjectionRole.MLP_GATE,
                ProjectionRole.MLP_UP,
            }:
                output_transform = IDENTITY
            if arm in {
                "A4", "A5", "A6", "A12", "A13", "A14", "A15", "A16", "A17", "A18",
                "A22", "A23", "A24", "A26", "A27", "A28",
            }:
                if semantic.role == ProjectionRole.MLP_GATE:
                    output_transform = TransformSpec(
                        TransformKind.PERMUTATION,
                        TransformPlacement.FOLDED,
                        f"swiglu.l{semantic.layer_index}",
                    )
                elif semantic.role == ProjectionRole.MLP_UP:
                    output_transform = TransformSpec(
                        TransformKind.PERMUTATION if arm in {"A27", "A28"} else TransformKind.DIAGONAL,
                        TransformPlacement.FOLDED,
                        f"swiglu.l{semantic.layer_index}",
                    )
                elif semantic.role == ProjectionRole.MLP_DOWN:
                    input_transform = self._down_transform(arm, semantic.layer_index)
            descriptors.append(
                ModuleTransformDescriptor(
                    module_name=semantic.name,
                    role=semantic.role,
                    input_transform=input_transform,
                    output_transform=output_transform,
                )
            )
        return QVQTransformPlan(
            arm=arm,
            description=descriptions[arm],
            modules=tuple(descriptors),
            metadata={
                "schema": "qvq.transform-plan.v1",
                "gated": True,
                "runtime_implemented": arm
                in {
                    "A0", "A1", "A3", "A4", "A6", "A18", "A20", "A21", "A22", "A23", "A24",
                    "A25", "A26", "A27", "A28", "A29", "A30",
                    "A31",
                },
                "checkpoint_serialization_implemented": arm == "A0",
            },
        )

    @staticmethod
    def _a1_specs(role, residual_in, residual_out, vo_folded):
        if role in {ProjectionRole.ATTENTION_Q, ProjectionRole.ATTENTION_K}:
            return residual_in, ONLINE_HADAMARD
        if role == ProjectionRole.ATTENTION_V:
            return residual_in, vo_folded
        if role == ProjectionRole.ATTENTION_O:
            return vo_folded, residual_out
        if role in {ProjectionRole.MLP_GATE, ProjectionRole.MLP_UP}:
            return residual_in, ONLINE_HADAMARD
        if role == ProjectionRole.MLP_DOWN:
            return ONLINE_HADAMARD, residual_out
        raise ValueError(f"unknown projection role: {role}")

    @staticmethod
    def _down_transform(arm: str, layer_index: int) -> TransformSpec:
        basis_id = f"down.l{layer_index}"
        if arm == "A4":
            return ONLINE_HADAMARD
        if arm == "A5":
            return TransformSpec(TransformKind.STRUCTURED_ORTHOGONAL, TransformPlacement.ONLINE, basis_id)
        if arm in {"A6", "A18", "A23"}:
            return IDENTITY
        if arm in {"A12", "A13", "A14"}:
            return TransformSpec(TransformKind.HADAMARD, TransformPlacement.ONLINE, f"{basis_id}.h{2 ** (int(arm[1:]) - 8)}")
        if arm in {"A15", "A16"}:
            return TransformSpec(TransformKind.STRUCTURED_ORTHOGONAL, TransformPlacement.ONLINE, basis_id)
        if arm == "A17":
            return TransformSpec(TransformKind.PERMUTATION, TransformPlacement.FUSED, basis_id)
        return ONLINE_HADAMARD

    def rewrite_dense_weights(self, plan: QVQTransformPlan, *, seed: int = 0) -> dict[str, Any]:
        if not isinstance(plan, QVQTransformPlan):
            raise TypeError("QVQ dense rewrite requires a QVQTransformPlan")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("QVQ transform seed must be an integer")
        return self.implementor.rewrite_dense_weights(self.model, plan, seed=seed)


__all__ = [
    "ModuleTransformDescriptor",
    "ProjectionRole",
    "ProjectionSemantic",
    "QVQArchitectureImplementor",
    "QVQTransformPlan",
    "QVQTransformPlanner",
    "TransformKind",
    "TransformPlacement",
    "TransformSpec",
]
