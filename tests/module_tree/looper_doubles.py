# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Shared ``gptq_model`` doubles for ``ModuleLooper.create_named_modules``.

``ModuleLooper`` asks the owning model which module the calibration forward
should run on::

    packed_module = self.gptq_model._maybe_auto_dtype_cast_forward_module(
        source_module, device=..., named_module=subset[name]
    )

and replaces the subset module only when a different object comes back. The
production method returns the module unchanged when ``auto_dtype_cast`` is not
``AutoDtypeCast.AUTO_NF12``.

The model-tree tests below build dense fixtures that never model NF12 packing,
so they need exactly that disabled branch. These doubles keep the real
signature and the same-object contract, which is what the looper observes; they
must not become the only place the casting contract is exercised. The
cast-and-swap path is covered with a full harness in
``tests/test_auto_dtype_cast.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from torch import nn

    from gptqmodel.looper.named_module import NamedModule


def no_auto_dtype_cast_forward(
    module: "nn.Module",
    *,
    device: "torch.device",
    named_module: "NamedModule",
) -> "nn.Module":
    """Return ``module`` unchanged, as an AUTO_NF12-disabled model would.

    Bind this directly on a ``SimpleNamespace`` fixture; it takes no ``self``.
    """

    return module


class NoAutoDtypeCastForward:
    """Mixin for class-based ``gptq_model`` doubles that never cast a module."""

    def _maybe_auto_dtype_cast_forward_module(
        self,
        module: "nn.Module",
        *,
        device: "torch.device",
        named_module: "NamedModule",
    ) -> "nn.Module":
        return no_auto_dtype_cast_forward(
            module, device=device, named_module=named_module
        )
