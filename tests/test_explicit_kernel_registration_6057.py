# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for explicit kernel selection and registration (#6057).

Downstream libraries, tests, or notebooks defining BaseQuantLinear subclasses
must not inadvertently shadow or conflict with built-in kernels during explicit
backend lookup or AUTO mapping. External kernels must be explicitly registered
via ``register_kernel`` to participate in lookup.
"""

import pytest

from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.importer import (
    get_auto_backend_kernel_mapping,
    get_kernel_for_backend,
    get_kernel_inventory,
    iter_quant_linear_kernels,
    register_kernel,
)


def test_unregistered_external_subclass_does_not_break_explicit_lookup():
    """An unregistered external subclass must not conflict with shipped kernels."""
    key = BACKEND.GPTQ_TORCH, METHOD.GPTQ, FORMAT.GPTQ
    assert get_kernel_for_backend(*key) is TorchLinear

    class Probe(TorchLinear):
        SUPPORTS_FORMATS = TorchLinear.SUPPORTS_FORMATS

    # AUTO mapping remains TorchLinear
    auto_map = get_auto_backend_kernel_mapping()
    assert auto_map[METHOD.GPTQ][FORMAT.GPTQ][BACKEND.GPTQ_TORCH] is TorchLinear

    # Explicit lookup resolves cleanly to TorchLinear without multiple-match ValueError
    assert get_kernel_for_backend(*key) is TorchLinear

    # Introspection via iter_quant_linear_kernels still finds the subclass
    assert Probe in iter_quant_linear_kernels()

    # But inventory excludes it
    assert Probe not in get_kernel_inventory()


def test_register_kernel_distinct_triple():
    """Explicitly registered external kernel with distinct triple resolves."""
    class DistinctCustomLinear(BaseQuantLinear):
        SUPPORTS_BACKENDS = [BACKEND.BITBLAS]
        SUPPORTS_METHODS = [METHOD.QQQ]
        SUPPORTS_FORMATS = {FORMAT.GPTQ: 10}

    register_kernel(DistinctCustomLinear)
    assert DistinctCustomLinear in get_kernel_inventory()

    resolved = get_kernel_for_backend(BACKEND.BITBLAS, METHOD.QQQ, FORMAT.GPTQ)
    assert resolved is DistinctCustomLinear


def test_register_kernel_conflicting_triple_rejected():
    """Conflicting triple registration is rejected with both owner names."""
    class ConflictingLinear(BaseQuantLinear):
        SUPPORTS_BACKENDS = [BACKEND.GPTQ_TORCH]
        SUPPORTS_METHODS = [METHOD.GPTQ]
        SUPPORTS_FORMATS = {FORMAT.GPTQ: 5}

    with pytest.raises(ValueError, match="conflicts with existing owner `TorchLinear`") as exc_info:
        register_kernel(ConflictingLinear)

    assert "ConflictingLinear" in str(exc_info.value)
    assert "TorchLinear" in str(exc_info.value)

    # Prior inventory remains intact and TorchLinear is still usable
    assert get_kernel_for_backend(BACKEND.GPTQ_TORCH, METHOD.GPTQ, FORMAT.GPTQ) is TorchLinear


def test_register_kernel_type_validation_and_idempotency():
    """register_kernel validates type and allows idempotent re-registration."""
    with pytest.raises(TypeError, match="subclass of BaseQuantLinear"):
        register_kernel(object)

    with pytest.raises(TypeError, match="subclass of BaseQuantLinear"):
        register_kernel("NotAClass")

    # Idempotent re-registration of shipped or already-registered class
    register_kernel(TorchLinear)


def test_shipped_priority_zero_reader_reachable():
    """Priority-zero readers remain accessible through explicit lookup."""
    # TorchLinear declares priority for GPTQ format
    torch_kernel = get_kernel_for_backend(BACKEND.GPTQ_TORCH, METHOD.GPTQ, FORMAT.GPTQ)
    assert torch_kernel is TorchLinear
    assert torch_kernel in get_kernel_inventory()
