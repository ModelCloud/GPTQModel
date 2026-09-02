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


class QVQGroupedP32InputTransformState:
    """One shared input transform and one grouped CUDA P32 decode."""

    def __init__(
        self, group: QVQSharedInputTransformGroup, modules: dict[str, QVQLinear]
    ):
        # Reuse the established shared-transform validation before imposing
        # the narrower grouped P32 contract.
        QVQSharedInputTransformState(group, modules)
        if len(group.module_names) not in (2, 3):
            raise ValueError(
                f"grouped QVQ P32 basis {group.basis_id!r} requires two or three consumers"
            )
        reference = modules[group.module_names[0]]
        if reference.in_features % 16:
            raise ValueError(
                f"grouped QVQ P32 basis {group.basis_id!r} requires K divisible by 16"
            )
        if reference.trellis.device.type != "cuda":
            raise ValueError("grouped QVQ P32 execution requires CUDA-resident payloads")

        invariant_names = (
            "bits",
            "codebook_version",
            "vector_size",
            "trellis_window",
            "v2b2_p32",
            "dual_v2",
        )
        k_tiles = reference.in_features // 16
        trellis_parts = []
        selector_parts = []
        alternative_ids = []
        output_tile_ends = []
        output_widths = []
        original_storage_bytes = 0
        for module_name in group.module_names:
            module = modules[module_name]
            differing = [
                name
                for name in invariant_names
                if getattr(module, name) != getattr(reference, name)
            ]
            if differing:
                raise ValueError(
                    f"grouped QVQ P32 basis {group.basis_id!r} has incompatible "
                    f"{module_name}: {differing}"
                )
            if (
                not module.v2b2_p32
                or module.dual_v2
                or module.vector_size != 2
                or module.trellis_window != 16
            ):
                raise ValueError(
                    f"grouped QVQ P32 basis {group.basis_id!r} requires direct V2B2-P32 modules"
                )
            if module.out_features % 16:
                raise ValueError(
                    f"grouped QVQ P32 consumer {module_name!r} requires N divisible by 16"
                )
            if module.trellis.device != reference.trellis.device:
                raise ValueError(
                    f"grouped QVQ P32 basis {group.basis_id!r} spans CUDA devices"
                )
            n_tiles = module.out_features // 16
            tile_count = k_tiles * n_tiles
            if module.trellis.ndim != 2 or module.trellis.shape[0] != tile_count:
                raise ValueError(
                    f"grouped QVQ P32 consumer {module_name!r} has incompatible trellis shape"
                )
            if module.trellis.dtype != torch.int32:
                raise TypeError("grouped QVQ P32 trellises must use int32")
            if (
                module.bank_ids is None
                or module.bank_ids.dtype != torch.uint8
                or module.bank_ids.numel() != tile_count
            ):
                raise ValueError(
                    f"grouped QVQ P32 consumer {module_name!r} needs one packed selector byte per tile"
                )
            if (
                module.bank_alt_id is None
                or module.bank_alt_id.dtype != torch.uint8
                or module.bank_alt_id.numel() != 1
            ):
                raise ValueError(
                    f"grouped QVQ P32 consumer {module_name!r} needs one uint8 alternative-bank ID"
                )
            alternative = int(module.bank_alt_id.item())
            if not 1 <= alternative <= 3:
                raise ValueError(
                    f"grouped QVQ P32 consumer {module_name!r} has invalid alternative-bank ID"
                )
            words_per_tile = module.trellis.shape[1]
            trellis_parts.append(
                module.trellis.contiguous().view(k_tiles, n_tiles, words_per_tile)
            )
            selector_parts.append(
                module.bank_ids.contiguous().view(k_tiles, n_tiles)
            )
            alternative_ids.append(alternative)
            output_tile_ends.append(
                n_tiles + (output_tile_ends[-1] if output_tile_ends else 0)
            )
            output_widths.append(module.out_features)
            original_storage_bytes += (
                module.trellis.numel() * module.trellis.element_size()
                + module.bank_ids.numel() * module.bank_ids.element_size()
                + module.bank_alt_id.numel() * module.bank_alt_id.element_size()
            )

        words_per_tile = reference.trellis.shape[1]
        if any(part.shape[-1] != words_per_tile for part in trellis_parts):
            raise ValueError(
                f"grouped QVQ P32 basis {group.basis_id!r} has incompatible rates"
            )
        self.group = group
        self.modules = tuple(modules[name] for name in group.module_names)
        self.trellis = torch.cat(trellis_parts, dim=1).reshape(-1, words_per_tile)
        self.bank_ids = torch.cat(selector_parts, dim=1).reshape(-1)
        self.bank_alt_ids = torch.tensor(
            alternative_ids,
            dtype=torch.uint8,
            device=reference.trellis.device,
        )
        self.bank_alt_boundaries = tuple(output_tile_ends[:-1])
        self.output_widths = tuple(output_widths)
        self.out_features = sum(output_widths)
        self.original_storage_bytes = original_storage_bytes
        self.grouped_storage_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.trellis, self.bank_ids, self.bank_alt_ids)
        )
        self.metadata_overhead_bytes = self.grouped_storage_bytes - original_storage_bytes
        self._source: torch.Tensor | None = None
        self._source_version: int | None = None
        self._outputs: tuple[torch.Tensor, ...] | None = None
        self._next_consumer_index = 0
        self.transform_invocations = 0
        self.grouped_gemv_invocations = 0
        self.completed_cycles = 0
        self.payloads_released = False

    @property
    def pending_consumers(self) -> tuple[str, ...]:
        return self.group.module_names[self._next_consumer_index :]

    def release_individual_payloads(self) -> None:
        """Drop child payload copies after the interleaved group is complete."""

        if self.payloads_released:
            return
        words_per_tile = self.trellis.shape[1]
        for module in self.modules:
            module.trellis = module.trellis.new_empty((0, words_per_tile))
            module.bank_ids = module.bank_ids.new_empty((0,))
            module.bank_alt_id = module.bank_alt_id.new_empty((0,))
            module._qvq_cuda_bank_cache = None
        self.payloads_released = True

    def reset(self) -> None:
        self._source = None
        self._source_version = None
        self._outputs = None
        self._next_consumer_index = 0

    @staticmethod
    def _tensor_version(x: torch.Tensor) -> int | None:
        return None if torch.is_inference(x) else x._version

    def _grouped_outputs(
        self, transformed: torch.Tensor, output_dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        from gptqmodel.utils.qvq_cuda import qvq_cuda_gemv

        if transformed.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError(
                "grouped QVQ P32 execution requires FP16 or BF16 transformed activations"
            )
        flat = transformed.reshape(-1, transformed.shape[-1]).contiguous()
        reference = self.modules[0]
        grouped_inner = qvq_cuda_gemv(
            flat,
            self.trellis,
            reference.bits,
            out_features=self.out_features,
            codebook_version=reference.codebook_version,
            output_fp32=True,
            vector_size=reference.vector_size,
            bank_ids=self.bank_ids,
            v2b2_p32=True,
            bank_alt_ids=self.bank_alt_ids,
            bank_alt_boundaries=self.bank_alt_boundaries,
            _bank_alt_ids_validated=True,
        )
        self.grouped_gemv_invocations += 1
        leading_shape = transformed.shape[:-1]
        pieces = grouped_inner.split(self.output_widths, dim=-1)
        return tuple(
            module.recover_output(
                piece.contiguous().reshape(*leading_shape, module.out_features),
                output_dtype=output_dtype,
            )
            for module, piece in zip(self.modules, pieces, strict=True)
        )

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
                f"grouped QVQ P32 basis {self.group.basis_id!r} received {reason} "
                f"{module_name!r}; pending={self.pending_consumers}"
            )
        if self.group.module_names[consumer_index] != module_name:
            raise RuntimeError("grouped QVQ P32 wrapper order is inconsistent")

        if consumer_index == 0:
            self._source = x
            self._source_version = self._tensor_version(x)
            transformed = module.transform_input(x)
            self.transform_invocations += 1
            self._outputs = self._grouped_outputs(transformed, x.dtype)
        elif self._source is not x or (
            self._source_version is not None and self._source_version != x._version
        ):
            raise RuntimeError(
                f"grouped QVQ P32 basis {self.group.basis_id!r} received a new or "
                f"mutated activation; pending={self.pending_consumers}"
            )

        outputs = self._outputs
        if outputs is None:
            raise RuntimeError("grouped QVQ P32 output cache is unexpectedly empty")
        output = outputs[consumer_index]
        self._next_consumer_index += 1
        if self._next_consumer_index == len(self.group.module_names):
            self.completed_cycles += 1
            self.reset()
        return output


class QVQGroupedP32Linear(torch.nn.Module):
    """A child projection served by one eager grouped P32 sibling decode."""

    def __init__(
        self,
        linear: QVQLinear,
        state: QVQGroupedP32InputTransformState,
        module_name: str,
        consumer_index: int,
    ):
        super().__init__()
        self.linear = linear
        object.__setattr__(self, "_grouped_p32_state", state)
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
        return self._grouped_p32_state.consume(
            self.consumer_index, self.module_name, self.linear, x
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        del destination, prefix, keep_vars
        raise RuntimeError(
            "grouped QVQ P32 runtime payloads are not checkpoint-serializable"
        )


@dataclass(frozen=True)
class QVQRefactoredP32Runtime:
    """Result of compiling compatible groups with semantic plain fallbacks."""

    grouped_states: dict[str, QVQGroupedP32InputTransformState]
    plain_fallbacks: dict[str, str]


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


def install_qvq_grouped_p32_input_transforms(
    model: torch.nn.Module,
    plan: QVQTransformPlan,
) -> dict[str, QVQGroupedP32InputTransformState]:
    """Install CUDA P32 sibling fusion and release superseded child payloads.

    The grouped buffers contain exactly the same trellis words and packed bank
    selectors as their children, interleaved in the K-major/N-major layout the
    CUDA decoder expects.  Serialization is intentionally unsupported because
    child payload buffers are emptied after installation.
    """

    prepared: list[
        tuple[
            QVQSharedInputTransformGroup,
            dict[str, QVQLinear],
            QVQGroupedP32InputTransformState,
        ]
    ] = []
    for group in shared_input_groups(plan):
        modules = {}
        for module_name in group.module_names:
            module = model.get_submodule(module_name)
            if not isinstance(module, QVQLinear):
                raise TypeError(
                    f"grouped QVQ P32 consumer {module_name!r} must be a packed "
                    f"QVQLinear, got {type(module).__name__}"
                )
            modules[module_name] = module
        prepared.append(
            (group, modules, QVQGroupedP32InputTransformState(group, modules))
        )

    states = {}
    for group, modules, state in prepared:
        state.release_individual_payloads()
        for consumer_index, module_name in enumerate(group.module_names):
            parent_name, _, child_name = module_name.rpartition(".")
            wrapper = QVQGroupedP32Linear(
                modules[module_name], state, module_name, consumer_index
            )
            wrapper.train(modules[module_name].training)
            setattr(model.get_submodule(parent_name), child_name, wrapper)
        states[group.basis_id] = state
    return states


def install_qvq_refactored_p32_runtime(
    model: torch.nn.Module,
    plan: QVQTransformPlan,
) -> QVQRefactoredP32Runtime:
    """Compile compatible sibling groups and retain plain P32 otherwise.

    A grouped input transform is legal only while every consumer has the same
    stored ``SU`` and compatible packed geometry.  Post-quant recovery may
    intentionally make those tensors module-local.  This compiler preserves
    the canonical per-module execution as an explicit fallback instead of
    rejecting the model or sharing an invalid transform.  The strict A41
    installer above remains fail-closed and unchanged.
    """

    prepared: list[
        tuple[
            QVQSharedInputTransformGroup,
            dict[str, QVQLinear],
            QVQGroupedP32InputTransformState,
        ]
    ] = []
    plain_fallbacks: dict[str, str] = {}
    for group in shared_input_groups(plan):
        modules = {}
        for module_name in group.module_names:
            module = model.get_submodule(module_name)
            if not isinstance(module, QVQLinear):
                raise TypeError(
                    f"refactored QVQ P32 consumer {module_name!r} must be a packed "
                    f"QVQLinear, got {type(module).__name__}"
                )
            modules[module_name] = module
        try:
            state = QVQGroupedP32InputTransformState(group, modules)
        except (TypeError, ValueError) as exc:
            plain_fallbacks[group.basis_id] = str(exc)
            continue
        prepared.append((group, modules, state))

    states = {}
    for group, modules, state in prepared:
        state.release_individual_payloads()
        for consumer_index, module_name in enumerate(group.module_names):
            parent_name, _, child_name = module_name.rpartition(".")
            wrapper = QVQGroupedP32Linear(
                modules[module_name], state, module_name, consumer_index
            )
            wrapper.train(modules[module_name].training)
            setattr(model.get_submodule(parent_name), child_name, wrapper)
        states[group.basis_id] = state
    return QVQRefactoredP32Runtime(states, plain_fallbacks)


__all__ = [
    "QVQGroupedP32InputTransformState",
    "QVQGroupedP32Linear",
    "QVQRefactoredP32Runtime",
    "QVQSharedInputLinear",
    "QVQSharedInputTransformGroup",
    "QVQSharedInputTransformState",
    "install_qvq_shared_input_transforms",
    "install_qvq_grouped_p32_input_transforms",
    "install_qvq_refactored_p32_runtime",
    "shared_input_groups",
]
