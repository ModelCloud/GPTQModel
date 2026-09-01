# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Runtime realization of graph-planned QVQ activation transforms.

This module deliberately knows nothing about projection roles.  Architecture
implementors assign generic shared-basis identifiers in a
``QVQTransformPlan``; the coordinator below validates the packed modules and
reuses their complete input transform across every declared consumer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.quantization.qvq_transform_planner import (
    QVQTransformPlan,
    TransformPlacement,
)


@dataclass(frozen=True)
class QVQSharedInputTransformGroup:
    """A role-agnostic set of modules consuming one transformed activation."""

    basis_id: str
    module_names: tuple[str, ...]


class QVQSharedInputTransformState:
    """Strict single-activation cache shared by sibling QVQ projections."""

    def __init__(
        self, group: QVQSharedInputTransformGroup, modules: dict[str, QVQLinear]
    ):
        if len(group.module_names) < 2:
            raise ValueError(
                f"shared QVQ basis {group.basis_id!r} requires at least two consumers"
            )
        if set(group.module_names) != set(modules):
            raise ValueError(
                f"shared QVQ basis {group.basis_id!r} module mapping does not match its plan"
            )
        reference = modules[group.module_names[0]]
        for module_name in group.module_names:
            module = modules[module_name]
            if module.training:
                raise RuntimeError(
                    f"shared QVQ input transforms are inference-only: {module_name} is training"
                )
            if module.in_features != reference.in_features:
                raise ValueError(
                    f"shared QVQ basis {group.basis_id!r} has incompatible input widths: "
                    f"{reference.in_features} and {module.in_features}"
                )
            if module.input_hadamard != reference.input_hadamard:
                raise ValueError(
                    f"shared QVQ basis {group.basis_id!r} has incompatible input Hadamard flags"
                )
            if (
                module.SU.shape != reference.SU.shape
                or module.SU.dtype != reference.SU.dtype
                or module.SU.device != reference.SU.device
                or not torch.equal(module.SU, reference.SU)
            ):
                raise ValueError(
                    f"shared QVQ basis {group.basis_id!r} requires bit-identical stored SU; "
                    f"{module_name} differs from {group.module_names[0]}"
                )

        self.group = group
        self._source: torch.Tensor | None = None
        self._source_version: int | None = None
        self._transformed: torch.Tensor | None = None
        self._next_consumer_index = 0
        self.transform_invocations = 0
        self.completed_cycles = 0

    @property
    def pending_consumers(self) -> tuple[str, ...]:
        return self.group.module_names[self._next_consumer_index :]

    def reset(self) -> None:
        """Discard an incomplete cycle after an aborted model forward."""

        self._source = None
        self._source_version = None
        self._transformed = None
        self._next_consumer_index = 0

    @staticmethod
    def _tensor_version(x: torch.Tensor) -> int | None:
        if torch.is_inference(x):
            # PyTorch inference tensors intentionally do not track versions.
            # Object identity remains strict enough for sibling calls inside
            # one uninterrupted model forward.
            return None
        return x._version

    def consume(
        self,
        consumer_index: int,
        module_name: str,
        module: QVQLinear,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if consumer_index != self._next_consumer_index:
            reason = (
                "duplicate consumer"
                if consumer_index < self._next_consumer_index
                else "out-of-order consumer"
            )
            raise RuntimeError(
                f"shared QVQ basis {self.group.basis_id!r} received {reason} "
                f"{module_name!r}; pending={self.pending_consumers}"
            )
        expected_name = self.group.module_names[consumer_index]
        if module_name != expected_name:  # pragma: no cover - immutable wrapper invariant
            raise RuntimeError(
                f"shared QVQ basis {self.group.basis_id!r} expected {expected_name!r}, "
                f"got {module_name!r}"
            )

        if consumer_index == 0:
            self._source = x
            self._source_version = self._tensor_version(x)
            self._transformed = module.transform_input(x)
            self.transform_invocations += 1
        elif self._source is not x or (
            self._source_version is not None and self._source_version != x._version
        ):
            raise RuntimeError(
                f"shared QVQ basis {self.group.basis_id!r} received a new or mutated "
                f"activation before all consumers ran; pending={self.pending_consumers}"
            )

        transformed = self._transformed
        if transformed is None:  # pragma: no cover - guarded state invariant
            raise RuntimeError("shared QVQ transform cache is unexpectedly empty")
        self._next_consumer_index += 1
        if self._next_consumer_index == len(self.group.module_names):
            self.completed_cycles += 1
            self.reset()
        return transformed


class QVQSharedInputLinear(torch.nn.Module):
    """A packed QVQ projection consuming a graph-shared input transform."""

    def __init__(
        self,
        linear: QVQLinear,
        state: QVQSharedInputTransformState,
        module_name: str,
        consumer_index: int,
    ):
        super().__init__()
        self.linear = linear
        object.__setattr__(self, "_shared_input_state", state)
        self.module_name = module_name
        self.consumer_index = consumer_index

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    @property
    def out_features(self) -> int:
        return self.linear.out_features

    @property
    def bias(self) -> torch.Tensor | None:
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformed = self._shared_input_state.consume(
            self.consumer_index, self.module_name, self.linear, x
        )
        return self.linear.forward_pretransformed(transformed, output_dtype=x.dtype)


def shared_input_groups(
    plan: QVQTransformPlan,
) -> tuple[QVQSharedInputTransformGroup, ...]:
    """Extract and validate generic shared-input groups from a transform plan."""

    grouped: dict[str, list[str]] = {}
    for descriptor in plan.modules:
        if descriptor.output_transform.placement == TransformPlacement.SHARED:
            raise ValueError(
                f"shared QVQ output transforms are not implemented: {descriptor.module_name}"
            )
        transform = descriptor.input_transform
        if transform.placement != TransformPlacement.SHARED:
            continue
        if not transform.basis_id:
            raise ValueError(
                f"shared QVQ input transform lacks a basis_id: {descriptor.module_name}"
            )
        grouped.setdefault(transform.basis_id, []).append(descriptor.module_name)
    return tuple(
        QVQSharedInputTransformGroup(basis_id, tuple(module_names))
        for basis_id, module_names in grouped.items()
    )


def install_qvq_shared_input_transforms(
    model: torch.nn.Module,
    plan: QVQTransformPlan,
) -> dict[str, QVQSharedInputTransformState]:
    """Install plan-declared shared transforms without changing QVQ payloads."""

    states: dict[str, QVQSharedInputTransformState] = {}
    for group in shared_input_groups(plan):
        modules = {}
        for module_name in group.module_names:
            module = model.get_submodule(module_name)
            if not isinstance(module, QVQLinear):
                raise TypeError(
                    f"shared QVQ consumer {module_name!r} must be a packed QVQLinear, "
                    f"got {type(module).__name__}"
                )
            modules[module_name] = module
        state = QVQSharedInputTransformState(group, modules)
        for consumer_index, module_name in enumerate(group.module_names):
            parent_name, _, child_name = module_name.rpartition(".")
            wrapper = QVQSharedInputLinear(
                modules[module_name], state, module_name, consumer_index
            )
            wrapper.train(modules[module_name].training)
            setattr(model.get_submodule(parent_name), child_name, wrapper)
        states[group.basis_id] = state
    return states


__all__ = [
    "QVQSharedInputLinear",
    "QVQSharedInputTransformGroup",
    "QVQSharedInputTransformState",
    "install_qvq_shared_input_transforms",
    "shared_input_groups",
]
