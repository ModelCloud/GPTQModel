# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Runtime realization of graph-planned QVQ activation transforms.

This module deliberately knows nothing about projection roles.  Architecture
implementors assign generic shared-basis identifiers in a
``QVQTransformPlan``; the coordinator below validates the packed modules and
reuses their complete input transform across every declared consumer.

Checkpointed grouped P32 consumers retain independent optional rank8 factors.
The grouped decoder publishes one transformed activation, then each child adds
its own FP32 inner-domain correction before its output Hadamard/SV/bias path.
All rank8 policy changes are prepared before execution or graph capture.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import torch

from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.quantization.qvq_transform_planner import (
    QVQTransformPlan,
    TransformPlacement,
)

QVQ_GROUPED_P32_RUNTIME_META_KEY = "qvq_grouped_p32_runtime"
QVQ_GROUPED_P32_RUNTIME_SCHEMA = "qvq_grouped_p32_v1"
QVQ_GROUPED_P32_PAYLOAD_LAYOUT = "canonical_per_module_p32"


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


class QVQGroupedP32InputTransformState(torch.nn.Module):
    """One shared input transform, grouped CUDA P32 decode, and child rank8."""

    def __init__(
        self, group: QVQSharedInputTransformGroup, modules: dict[str, QVQLinear]
    ):
        super().__init__()
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
        # Keep child modules as ordinary Python references. Registering them
        # here would create a second module-tree path and duplicate their
        # canonical checkpoint keys.
        object.__setattr__(
            self,
            "linears",
            tuple(modules[name] for name in group.module_names),
        )
        self.register_buffer(
            "trellis",
            torch.cat(trellis_parts, dim=1).reshape(-1, words_per_tile),
            persistent=False,
        )
        self.register_buffer(
            "bank_ids",
            torch.cat(selector_parts, dim=1).reshape(-1),
            persistent=False,
        )
        self.register_buffer(
            "bank_alt_ids",
            torch.tensor(
                alternative_ids,
                dtype=torch.uint8,
                device=reference.trellis.device,
            ),
            persistent=False,
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
        # Rank8 factors remain child-local even though the P32 decode payload
        # is interleaved.  Validate each immutable base hash while canonical
        # child payloads are present; factor bytes are checked only when a
        # quality mode enables that child, so recovery-off never reads them.
        self._rank8_metadata: list[dict | None] = []
        self._rank8_base_hashes: list[str | None] = []
        self._rank8_factor_hashes: list[str | None] = []
        self._rank8_enabled = [False] * len(self.linears)
        self._rank8_configs: list[object | None] = [None] * len(self.linears)
        from .qvq_rank8 import _base, _digest, _metadata

        for consumer_index, module in enumerate(self.linears):
            raw_metadata = module.rank8_metadata
            if raw_metadata is None:
                self._rank8_metadata.append(None)
                self._rank8_base_hashes.append(None)
                self._rank8_factor_hashes.append(None)
                continue
            metadata = _metadata(module)
            tensors, base_metadata = _base(module)
            if metadata.get("base_hash") != _digest(tensors, base_metadata):
                raise ValueError(
                    f"grouped QVQ P32 consumer {self.group.module_names[consumer_index]!r} has a rank8 base hash mismatch"
                )
            self._rank8_metadata.append(metadata)
            self._rank8_base_hashes.append(metadata["base_hash"])
            self._rank8_factor_hashes.append(metadata["factors_hash"])

    @property
    def pending_consumers(self) -> tuple[str, ...]:
        return self.group.module_names[self._next_consumer_index :]

    def release_individual_payloads(self) -> None:
        """Drop child payload copies after the interleaved group is complete."""

        if self.payloads_released:
            return
        words_per_tile = self.trellis.shape[1]
        for module in self.linears:
            module.trellis = module.trellis.new_empty((0, words_per_tile))
            module.bank_ids = module.bank_ids.new_empty((0,))
            module.bank_alt_id = module.bank_alt_id.new_empty((0,))
            module._qvq_cuda_bank_cache = None
        self.payloads_released = True

    def canonical_child_payload(self, consumer_index: int) -> dict[str, torch.Tensor]:
        """Reconstruct one child's canonical tensors from the grouped buffers."""

        if isinstance(consumer_index, bool) or not isinstance(consumer_index, int):
            raise TypeError("grouped QVQ P32 consumer index must be an integer")
        if not 0 <= consumer_index < len(self.output_widths):
            raise IndexError("grouped QVQ P32 consumer index is out of range")
        k_tiles = self.linears[consumer_index].in_features // 16
        words_per_tile = self.trellis.shape[1]
        start = 0 if consumer_index == 0 else self.bank_alt_boundaries[consumer_index - 1]
        end = (
            self.out_features // 16
            if consumer_index == len(self.output_widths) - 1
            else self.bank_alt_boundaries[consumer_index]
        )
        trellis = (
            self.trellis.view(k_tiles, self.out_features // 16, words_per_tile)
            [:, start:end, :]
            .contiguous()
            .reshape(-1, words_per_tile)
        )
        bank_ids = (
            self.bank_ids.view(k_tiles, self.out_features // 16)
            [:, start:end]
            .contiguous()
            .reshape(-1)
        )
        return {
            "trellis": trellis,
            "bank_ids": bank_ids,
            "bank_alt_id": self.bank_alt_ids[consumer_index : consumer_index + 1].contiguous(),
        }

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
        reference = self.linears[0]
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
        flat_input = flat.float() if any(self._rank8_enabled) else None
        outputs = []
        for index, (module, piece) in enumerate(zip(self.linears, pieces, strict=True)):
            # The grouped decoder already returns an FP32 inner accumulator.
            # Add each child correction in that same domain, before the
            # child-local output Hadamard/SV/bias recovery.  Disabled children
            # never touch their factor buffers.
            if self._rank8_enabled[index]:
                self.validate_rank8(index, module)
                hidden = flat_input @ module.rank8_A.float()
                piece = piece.float() + hidden @ module.rank8_B.float()
            outputs.append(
                module.recover_output(
                    piece.contiguous().reshape(*leading_shape, module.out_features),
                    output_dtype=output_dtype,
                )
            )
        return tuple(outputs)

    def prepare_rank8(
        self, consumer_index: int, module: QVQLinear, config: object
    ) -> None:
        """Bind one child quality policy after grouped payload installation.

        Grouped installation releases canonical trellis/selectors, so the
        ordinary per-module preparation path cannot recompute its base hash.
        This method uses the hashes captured during installation and only
        changes static per-child execution flags outside CUDA capture.
        """

        from .qvq_rank8 import CONTRACT, P32WindowConfig, _digest, _versions

        if not isinstance(config, P32WindowConfig):
            raise TypeError("config must be P32WindowConfig")
        if consumer_index < 0 or consumer_index >= len(self.linears):
            raise IndexError("grouped QVQ P32 consumer index is out of range")
        if module is not self.linears[consumer_index]:
            raise RuntimeError("grouped QVQ P32 child binding is inconsistent")
        if self._outputs is not None:
            raise RuntimeError("cannot change rank8 mode during a grouped projection cycle")
        if config.algorithm.startswith("hopper_"):
            raise ValueError(
                "explicit Hopper geometry requires an independent window consumer"
            )
        if config.recovery_projection != "separate_reference":
            raise ValueError(
                "grouped checkpointed P32 currently uses the shared-input reference rank8 projection"
            )
        if config.recovery_kernel != "separate_reference":
            raise ValueError(
                "grouped checkpointed P32 currently uses the separate rank8 epilogue"
            )

        enabled = False
        if config.recovery_mode != "off" and not (
            config.recovery_mode == "auto" and config.quality_mode == "fast"
        ):
            metadata = self._rank8_metadata[consumer_index]
            if metadata is None:
                if config.recovery_mode == "on":
                    raise ValueError("recovery requested without validated tensors")
            else:
                if metadata.get("fit_contract") != CONTRACT or not metadata.get(
                    "validated"
                ):
                    raise ValueError("unvalidated or incompatible recovery contract")
                if module.rank8_A is None or module.rank8_B is None:
                    raise ValueError("recovery requested without validated tensors")
                expected = (
                    (module.in_features, 8),
                    (8, module.out_features),
                )
                if module.rank8_A.shape != expected[0] or module.rank8_B.shape != expected[1]:
                    raise ValueError("invalid rank-8 tensor shapes")
                for tensor in (module.rank8_A, module.rank8_B):
                    if tensor.dtype != torch.float16 or tensor.device != module.trellis.device:
                        raise ValueError("recovery tensors must be FP16 on the window device")
                    if not torch.isfinite(tensor).all():
                        raise ValueError("non-finite recovery tensors")
                factor_hash = _digest({"A": module.rank8_A, "B": module.rank8_B}, {})
                if factor_hash != self._rank8_factor_hashes[consumer_index]:
                    raise ValueError("recovery factors hash mismatch")
                enabled = (
                    config.recovery_mode == "on"
                    or config.quality_mode == "quality"
                    or (
                        config.quality_mode == "balanced"
                        and metadata.get("selected", False)
                    )
                )
        if enabled and module.trellis.device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
            raise ValueError("rank8 FP32 reference requires CUDA matmul TF32 disabled")
        self._rank8_enabled[consumer_index] = enabled
        self._rank8_configs[consumer_index] = config
        module._p32_window_config = config
        module._p32_rank8_enabled = enabled
        module._p32_rank8_versions = _versions(module) if enabled else None

    def validate_rank8(self, consumer_index: int, module: QVQLinear) -> None:
        from .qvq_rank8 import _versions

        if not self._rank8_enabled[consumer_index]:
            return
        if _versions(module) != module._p32_rank8_versions:
            raise RuntimeError(
                "grouped window/recovery state changed; prepare recovery again before execution"
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
    checkpoint_metadata_bytes: int = 0


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


def qvq_grouped_p32_checkpoint_metadata(
    plan: QVQTransformPlan,
) -> dict[str, object]:
    """Build the versioned load-time compiler manifest for a transform plan."""

    groups = shared_input_groups(plan)
    if not groups:
        raise ValueError("grouped QVQ P32 checkpoint metadata requires at least one group")
    return {
        "schema": QVQ_GROUPED_P32_RUNTIME_SCHEMA,
        "payload_layout": QVQ_GROUPED_P32_PAYLOAD_LAYOUT,
        "groups": [
            {
                "basis_id": group.basis_id,
                "module_names": list(group.module_names),
            }
            for group in groups
        ],
    }


def _groups_from_qvq_grouped_p32_checkpoint_metadata(
    metadata: object,
) -> tuple[QVQSharedInputTransformGroup, ...]:
    """Parse a grouped-P32 manifest without accepting ambiguous extensions."""

    if not isinstance(metadata, dict):
        raise TypeError("grouped QVQ P32 checkpoint metadata must be a dictionary")
    expected_keys = {"schema", "payload_layout", "groups"}
    if set(metadata) != expected_keys:
        raise ValueError(
            "grouped QVQ P32 checkpoint metadata must contain exactly "
            f"{sorted(expected_keys)}"
        )
    if metadata["schema"] != QVQ_GROUPED_P32_RUNTIME_SCHEMA:
        raise ValueError(
            f"unsupported grouped QVQ P32 schema: {metadata['schema']!r}"
        )
    if metadata["payload_layout"] != QVQ_GROUPED_P32_PAYLOAD_LAYOUT:
        raise ValueError(
            "unsupported grouped QVQ P32 payload layout: "
            f"{metadata['payload_layout']!r}"
        )
    raw_groups = metadata["groups"]
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("grouped QVQ P32 metadata groups must be a non-empty list")

    groups = []
    basis_ids: set[str] = set()
    module_names_seen: set[str] = set()
    for raw_group in raw_groups:
        if not isinstance(raw_group, dict) or set(raw_group) != {
            "basis_id",
            "module_names",
        }:
            raise ValueError(
                "each grouped QVQ P32 metadata group must contain exactly "
                "['basis_id', 'module_names']"
            )
        basis_id = raw_group["basis_id"]
        raw_module_names = raw_group["module_names"]
        if not isinstance(basis_id, str) or not basis_id:
            raise ValueError("grouped QVQ P32 basis IDs must be non-empty strings")
        if basis_id in basis_ids:
            raise ValueError(f"duplicate grouped QVQ P32 basis ID: {basis_id!r}")
        if not isinstance(raw_module_names, list) or len(raw_module_names) not in (2, 3):
            raise ValueError(
                f"grouped QVQ P32 basis {basis_id!r} requires two or three modules"
            )
        if any(not isinstance(name, str) or not name for name in raw_module_names):
            raise ValueError(
                f"grouped QVQ P32 basis {basis_id!r} has an invalid module name"
            )
        module_names = tuple(raw_module_names)
        duplicate_names = module_names_seen.intersection(module_names)
        if len(set(module_names)) != len(module_names) or duplicate_names:
            raise ValueError(
                "grouped QVQ P32 module names must be unique across the manifest"
            )
        basis_ids.add(basis_id)
        module_names_seen.update(module_names)
        groups.append(QVQSharedInputTransformGroup(basis_id, module_names))
    return tuple(groups)


def qvq_grouped_p32_checkpoint_metadata_bytes(metadata: object) -> int:
    """Return the compact UTF-8 storage cost after strict validation."""

    groups = _groups_from_qvq_grouped_p32_checkpoint_metadata(metadata)
    canonical = {
        "schema": QVQ_GROUPED_P32_RUNTIME_SCHEMA,
        "payload_layout": QVQ_GROUPED_P32_PAYLOAD_LAYOUT,
        "groups": [
            {"basis_id": group.basis_id, "module_names": list(group.module_names)}
            for group in groups
        ],
    }
    return len(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def set_qvq_grouped_p32_checkpoint_metadata(
    quantize_config: object,
    plan: QVQTransformPlan,
) -> dict[str, object]:
    """Attach a canonical grouped-runtime manifest without changing payloads."""

    metadata = qvq_grouped_p32_checkpoint_metadata(plan)
    current_meta = getattr(quantize_config, "meta", None)
    if current_meta is None:
        meta = {}
    elif isinstance(current_meta, dict):
        meta = copy.deepcopy(current_meta)
    else:
        raise TypeError("QVQ quantization metadata must be a dictionary")
    current = meta.get(QVQ_GROUPED_P32_RUNTIME_META_KEY)
    if current is not None and current != metadata:
        raise ValueError("refusing to replace conflicting grouped QVQ P32 metadata")
    meta[QVQ_GROUPED_P32_RUNTIME_META_KEY] = metadata
    setattr(quantize_config, "meta", meta)
    return copy.deepcopy(metadata)


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


def install_qvq_checkpointed_p32_runtime(
    model: torch.nn.Module,
    metadata: object,
) -> QVQRefactoredP32Runtime:
    """Compile canonical per-module tensors into transient grouped buffers.

    The original ``QVQLinear`` modules remain at their canonical model paths.
    Compatible CUDA groups delegate execution to a registered non-persistent
    runtime state; incompatible groups retain ordinary per-module P32
    execution.  State-dict serialization reconstructs each child's exact
    canonical tensor slices and never writes the grouped layout.
    """

    if hasattr(model, "_qvq_grouped_p32_runtime_states"):
        raise RuntimeError("grouped QVQ P32 checkpoint runtime is already installed")
    groups = _groups_from_qvq_grouped_p32_checkpoint_metadata(metadata)
    prepared: list[
        tuple[
            QVQSharedInputTransformGroup,
            dict[str, QVQLinear],
            QVQGroupedP32InputTransformState,
        ]
    ] = []
    plain_fallbacks: dict[str, str] = {}
    for group in groups:
        modules = {}
        for module_name in group.module_names:
            try:
                module = model.get_submodule(module_name)
            except AttributeError as exc:
                raise ValueError(
                    f"grouped QVQ P32 checkpoint references missing module {module_name!r}"
                ) from exc
            if not isinstance(module, QVQLinear):
                raise TypeError(
                    f"grouped QVQ P32 checkpoint consumer {module_name!r} must be a "
                    f"QVQLinear, got {type(module).__name__}"
                )
            modules[module_name] = module
        try:
            state = QVQGroupedP32InputTransformState(group, modules)
        except (TypeError, ValueError) as exc:
            plain_fallbacks[group.basis_id] = str(exc)
            continue
        prepared.append((group, modules, state))

    states = {group.basis_id: state for group, _, state in prepared}
    runtime_states = torch.nn.ModuleList(states.values())
    runtime_states.eval()
    setattr(model, "_qvq_grouped_p32_runtime_states", runtime_states)
    for group, modules, state in prepared:
        state.release_individual_payloads()
        for consumer_index, module_name in enumerate(group.module_names):
            object.__setattr__(
                modules[module_name],
                "_qvq_grouped_p32_delegate",
                (state, consumer_index, module_name),
            )

    compiled = QVQRefactoredP32Runtime(
        grouped_states=states,
        plain_fallbacks=plain_fallbacks,
        checkpoint_metadata_bytes=qvq_grouped_p32_checkpoint_metadata_bytes(metadata),
    )
    object.__setattr__(model, "_qvq_grouped_p32_runtime", compiled)
    return compiled


def install_qvq_grouped_p32_runtime_from_config(
    model: torch.nn.Module,
    quantize_config: object,
) -> QVQRefactoredP32Runtime | None:
    """Install the declared runtime, or leave an unmarked legacy model alone."""

    meta = getattr(quantize_config, "meta", None)
    if meta is None:
        return None
    if not isinstance(meta, dict):
        raise TypeError("QVQ quantization metadata must be a dictionary")
    metadata = meta.get(QVQ_GROUPED_P32_RUNTIME_META_KEY)
    if metadata is None:
        return None
    checkpoint_format = getattr(quantize_config, "format", None)
    checkpoint_format = getattr(checkpoint_format, "value", checkpoint_format)
    if checkpoint_format != "qvq_v2b2_p32":
        raise ValueError(
            "grouped QVQ P32 checkpoint metadata requires format=qvq_v2b2_p32"
        )
    return install_qvq_checkpointed_p32_runtime(model, metadata)


__all__ = [
    "QVQ_GROUPED_P32_PAYLOAD_LAYOUT",
    "QVQ_GROUPED_P32_RUNTIME_META_KEY",
    "QVQ_GROUPED_P32_RUNTIME_SCHEMA",
    "QVQGroupedP32InputTransformState",
    "QVQGroupedP32Linear",
    "QVQRefactoredP32Runtime",
    "QVQSharedInputLinear",
    "QVQSharedInputTransformGroup",
    "QVQSharedInputTransformState",
    "install_qvq_checkpointed_p32_runtime",
    "install_qvq_shared_input_transforms",
    "install_qvq_grouped_p32_input_transforms",
    "install_qvq_grouped_p32_runtime_from_config",
    "install_qvq_refactored_p32_runtime",
    "qvq_grouped_p32_checkpoint_metadata",
    "qvq_grouped_p32_checkpoint_metadata_bytes",
    "set_qvq_grouped_p32_checkpoint_metadata",
    "shared_input_groups",
]
