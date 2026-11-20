# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Test seam for the "is this frame being traced" predicate (#5427).

Kernels branch on ``gptqmodel.utils.torch.dynamo_is_compiling()``, which every
consumer imports by name, so patching ``torch.compiler.is_compiling`` no longer
reaches them. Flipping the predicate's cached ``torch.compiler`` hook instead
does, because all consumers call the same function object, which reads that
module global on every call.
"""

import contextlib

from gptqmodel.utils import torch as gptq_torch


@contextlib.contextmanager
def under_compile():
    """Report ``dynamo_is_compiling()`` as ``True`` for the enclosed calls."""

    previous = gptq_torch._IS_DYNAMO
    gptq_torch._IS_DYNAMO = lambda: True
    try:
        yield
    finally:
        gptq_torch._IS_DYNAMO = previous
