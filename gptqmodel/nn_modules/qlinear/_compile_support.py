# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Per-kernel ``torch.compile`` / CUDA-graph capability protocol.

Every concrete quantized kernel declares two class attributes, so callers can
ask a kernel what it is safe under instead of guessing from its class name:

``SUPPORTS_COMPILE_LEVEL: COMPILE_LEVEL``
    The highest ``torch.compile`` level the kernel's ``forward`` reaches, in
    ascending order. Each level implies every lower one:

    * ``EAGER`` - dynamo cannot trace the forward (graph break, an op without a
      meta implementation, or compile is a measured pessimization).
    * ``DYNAMO`` - ``torch.compile(fullgraph=True, backend="eager")`` traces the
      whole forward with zero graph breaks.
    * ``AOT_EAGER`` - ``backend="aot_eager"`` also works: the forward survives
      fake/meta tensors and ``aot_autograd``.
    * ``INDUCTOR`` - ``backend="inductor"`` lowers it (no unsupported op, no
      dynamic-shape or host-sync hazard the compiler rejects).
    * ``CUDAGRAPH`` - ``mode="reduce-overhead"`` captures it as a CUDA graph,
      i.e. the compiled forward has no host synchronisation, no CPU-side
      allocation decision and no data-dependent Python control flow.

``SUPPORTS_GRAPH_CAPTURE: bool``
    Whether the **eager** forward is safe under an explicit
    ``torch.cuda.CUDAGraph`` capture (``torch.cuda.graph(...)``): no
    synchronisation, no tensor-to-host transfer, no allocation or device
    decision in Python, and no data-dependent branching. This is independent
    of dynamo: a kernel that calls a custom op without a fake implementation
    can be capture-safe and still fail to trace, and a traceable kernel can
    still sync the host inside ``forward``.

``SUPPORTS_COMPILE`` (the pre-existing policy flag) stays separate and means
"compiling this forward is not a pessimization" for the static-cache
``generate()`` path (#4669). A kernel may be traceable and still keep it
``False`` when the eager path is already faster.

The declarations are checked two ways:

* ``tests/kernels/test_kernel_compile_protocol.py`` requires every registered
  kernel to declare both attributes explicitly, and statically rejects a
  ``CUDAGRAPH`` / ``SUPPORTS_GRAPH_CAPTURE`` claim whose ``forward`` contains a
  known host-sync or host-transfer construct.
* ``tests/kernels/test_kernel_compile_levels_*.py`` re-measure a constructed
  instance on real hardware and assert the declared level is reachable.
  ``docs/kernel-compile-matrix.md`` mirrors the declarations for readers and is
  kept in sync by the protocol test.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, Optional


class COMPILE_LEVEL(IntEnum):
    """Ordered ``torch.compile`` capability levels (higher implies lower).

    ``NOT_APPLICABLE`` is for kernels that never run as torch modules (MLX,
    TTNN, XLA): their own compiler owns the graph, so no ``torch.compile``
    level can be claimed for them.
    """

    NOT_APPLICABLE = -1
    EAGER = 0
    DYNAMO = 1
    AOT_EAGER = 2
    INDUCTOR = 3
    CUDAGRAPH = 4


#: How each level is measured, for logs and for the docs table.
LEVEL_PROBES: Dict[COMPILE_LEVEL, str] = {
    COMPILE_LEVEL.NOT_APPLICABLE: "not a torch module (MLX/TTNN/XLA)",
    COMPILE_LEVEL.EAGER: "eager forward only",
    COMPILE_LEVEL.DYNAMO: 'torch.compile(fullgraph=True, backend="eager")',
    COMPILE_LEVEL.AOT_EAGER: 'torch.compile(fullgraph=True, backend="aot_eager")',
    COMPILE_LEVEL.INDUCTOR: 'torch.compile(fullgraph=True, backend="inductor")',
    COMPILE_LEVEL.CUDAGRAPH: 'torch.compile(mode="reduce-overhead") CUDA-graph capture',
}


def compile_level(kernel: Any) -> COMPILE_LEVEL:
    """The declared level of a kernel class or instance.

    Accepts the ``str`` name too, so a serialized capability can round trip.
    An unknown or missing declaration is ``EAGER`` (the safe answer: nothing
    may assume a compiled path works).
    """

    value = getattr(kernel, "SUPPORTS_COMPILE_LEVEL", COMPILE_LEVEL.EAGER)
    if isinstance(value, COMPILE_LEVEL):
        return value
    if isinstance(value, str):
        try:
            return COMPILE_LEVEL[value.upper()]
        except KeyError as error:
            raise ValueError(
                f"{getattr(kernel, '__name__', kernel)}: unknown SUPPORTS_COMPILE_LEVEL "
                f"{value!r}; expected one of {[level.name for level in COMPILE_LEVEL]}."
            ) from error
    if isinstance(value, int):
        return COMPILE_LEVEL(value)
    raise TypeError(
        f"{getattr(kernel, '__name__', kernel)}: SUPPORTS_COMPILE_LEVEL must be a "
        f"COMPILE_LEVEL, got {type(value).__name__}."
    )


def graph_capture_safe(kernel: Any) -> bool:
    """Whether the eager forward may run inside a ``torch.cuda.CUDAGraph``."""

    return bool(getattr(kernel, "SUPPORTS_GRAPH_CAPTURE", False))


def reaches_level(kernel: Any, level: COMPILE_LEVEL) -> bool:
    """Whether the kernel declares at least ``level``."""

    return compile_level(kernel) >= level


def supports_compiled_generation(kernel: Any) -> bool:
    """The #4669 policy answer: is a compiled generation path not a pessimization?"""

    return bool(getattr(kernel, "SUPPORTS_COMPILE", False))


def describe_kernel(kernel: Any) -> Dict[str, Optional[str]]:
    """A JSON-friendly capability row for docs, logs and the registry test."""

    return {
        "kernel": getattr(kernel, "__name__", type(kernel).__name__),
        "module": getattr(kernel, "__module__", type(kernel).__module__),
        "compile_level": compile_level(kernel).name,
        "graph_capture": graph_capture_safe(kernel),
        "compiled_generation": supports_compiled_generation(kernel),
    }


__all__ = [
    "COMPILE_LEVEL",
    "LEVEL_PROBES",
    "compile_level",
    "describe_kernel",
    "graph_capture_safe",
    "reaches_level",
    "supports_compiled_generation",
]
