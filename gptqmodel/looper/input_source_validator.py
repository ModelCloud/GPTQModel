# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Capture and validate inputs shared by module-tree groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import torch

from ..models.input_source import InputSourceId

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .named_module import NamedModule


@dataclass
class InputSourceMismatch:
    """Describe one input-source group whose captured inputs differ."""

    source: InputSourceId
    module_names: list[str]
    shapes: dict[str, list[tuple]]
    dtypes: dict[str, list[torch.dtype]]
    reason: str
    differing: list[tuple[str, str]]


class InputSourceValidationError(RuntimeError):
    """Raised when modules declared to share an input source disagree."""

    def __init__(self, mismatches: list[InputSourceMismatch]) -> None:
        self.mismatches = mismatches
        lines = ["input-source validation failed:"]
        for mismatch in mismatches:
            lines.append(
                "source "
                f"{mismatch.source.kind} scope={mismatch.source.scope!r}, "
                f"modules={mismatch.module_names!r}, "
                f"shapes={mismatch.shapes!r}, reason={mismatch.reason}"
            )
        super().__init__("\n".join(lines))


class InputSourceCapture:
    """Record the first positional or keyword tensor input to each module."""

    def __init__(self, modules: Iterable[NamedModule], *, max_calls: int = 4) -> None:
        self.modules = list(modules)
        self.max_calls = max_calls
        self.captured: dict[str, list[torch.Tensor]] = {
            module.full_name: [] for module in self.modules
        }
        self._handles = []

    def __enter__(self) -> "InputSourceCapture":
        def make_capture(name: str):
            def capture(module: torch.nn.Module, args, kwargs) -> None:
                del module
                tensor = args[0] if args and isinstance(args[0], torch.Tensor) else None
                if tensor is None:
                    tensor = next(
                        (value for value in kwargs.values() if isinstance(value, torch.Tensor)),
                        None,
                    )
                if tensor is None:
                    return
                values = self.captured[name]
                if len(values) < self.max_calls:
                    values.append(tensor)

            return capture

        for named_module in self.modules:
            handle = named_module.module.register_forward_pre_hook(
                make_capture(named_module.full_name),
                with_kwargs=True,
            )
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


@dataclass
class InputSourceValidationReport:
    """Results from comparing captured inputs for declared source groups."""

    checked_sources: int
    checked_modules: int
    mismatches: list[InputSourceMismatch]

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise InputSourceValidationError(self.mismatches)


def _tensor_matches(left: torch.Tensor, right: torch.Tensor) -> tuple[bool, str | None]:
    if left is right:
        return True, None
    if (
        left.data_ptr() == right.data_ptr()
        and left.shape == right.shape
        and left.stride() == right.stride()
        and left.dtype == right.dtype
    ):
        return True, None
    if left.shape != right.shape:
        return False, "shape"
    if left.dtype != right.dtype:
        return False, "dtype"
    if torch.equal(left, right):
        return True, None
    return False, "value"


def validate_input_sources(
    input_sources: dict[InputSourceId, list[NamedModule]],
    captured: dict[str, list[torch.Tensor]],
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> InputSourceValidationReport:
    """Validate that every invoked module in each source group saw equal inputs."""

    checked_sources = 0
    checked_modules = 0
    mismatches = []
    for source, modules in input_sources.items():
        invoked = [module for module in modules if captured.get(module.full_name)]
        if len(invoked) < 2:
            continue
        checked_sources += 1
        checked_modules += len(invoked)
        shapes = {
            module.full_name: [tuple(tensor.shape) for tensor in captured[module.full_name]]
            for module in invoked
        }
        dtypes = {
            module.full_name: [tensor.dtype for tensor in captured[module.full_name]]
            for module in invoked
        }
        reference = invoked[0]
        reference_calls = captured[reference.full_name]
        differing = []
        reason = None
        for other in invoked[1:]:
            other_calls = captured[other.full_name]
            if len(reference_calls) != len(other_calls):
                reason = reason or "call_count"
                differing.append((reference.full_name, other.full_name))
                continue
            for left, right in zip(reference_calls, other_calls):
                matches, difference = _tensor_matches(left, right)
                if not matches:
                    if difference == "value" and (atol != 0.0 or rtol != 0.0):
                        matches = torch.allclose(left, right, atol=atol, rtol=rtol)
                        difference = None if matches else "value"
                    if not matches:
                        reason = reason or difference
                        differing.append((reference.full_name, other.full_name))
                        break
        if reason is not None:
            mismatches.append(
                InputSourceMismatch(
                    source=source,
                    module_names=[module.full_name for module in modules],
                    shapes=shapes,
                    dtypes=dtypes,
                    reason=reason,
                    differing=differing,
                )
            )
    return InputSourceValidationReport(
        checked_sources=checked_sources,
        checked_modules=checked_modules,
        mismatches=mismatches,
    )


def validate_plan_input_sources(plan, captured) -> InputSourceValidationReport:
    """Validate the input-source groups attached to a subset plan."""

    return validate_input_sources(plan.input_sources, captured)
