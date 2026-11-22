# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Per-kernel ``torch.compile`` / CUDA-graph capability protocol.

Every concrete quantized kernel declares two class attributes, so callers can
ask a kernel what it is safe under instead of guessing from its class name:

``SUPPORTS_COMPILE_LEVEL: COMPILE_LEVEL``
    The highest ``torch.compile`` level the kernel's ``forward`` reaches, in
    ascending order. Each level implies every lower one:

    * ``NOT_APPLICABLE`` - the class never runs as a torch module (MLX, TTNN,
      XLA); its own compiler owns the graph.
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

``SUPPORTS_COMPILE_LEVEL_BY_DEVICE: dict[str, COMPILE_LEVEL]``
    Optional per-device override, keyed by ``torch.device.type`` (``"cpu"``,
    ``"cuda"``, ``"xpu"``...). A kernel whose CPU and CUDA forwards have
    different compile ceilings keeps the base level for the device it was
    measured on last and records the other here, so neither device
    over-claims. ``TorchLinear`` is the current case: its CPU forward reaches
    ``INDUCTOR`` after the cache-key and ``ret_idx`` fixes, while its CUDA
    streaming path still allocates a ``torch.cuda.Stream`` and stays ``EAGER``.

    Only the dict in the kernel's **own class body** is consulted. A subclass
    inherits the base level, never another kernel's device answer: the MLX,
    Triton and embedding subclasses of ``TorchLinear`` do not run its CPU
    forward, so they must not inherit ``{"cpu": INDUCTOR}``. A
    ``NOT_APPLICABLE`` kernel (never a torch module) ignores the override
    entirely.

``SUPPORTS_GRAPH_CAPTURE: bool``
    Whether the **eager** forward is safe under an explicit
    ``torch.cuda.CUDAGraph`` capture (``torch.cuda.graph(...)``): no
    synchronisation, no tensor-to-host transfer, no allocation or device
    decision in Python, and no data-dependent branching. This is independent
    of dynamo: a kernel that calls a custom op without a fake implementation
    can be capture-safe and still fail to trace, and a traceable kernel can
    still sync the host inside ``forward``.

    The class attribute answers for the kernel's default runtime state. A
    kernel that serves several stored formats may refine it per instance
    through ``supports_graph_capture()``: ``graph_capture_safe(instance)``
    returns that answer while ``graph_capture_safe(cls)`` keeps returning the
    declaration. ``GGUFTorchLinear`` is the current case -- its device-side
    dequantizers are capturable, the host-dequantized Q2_K/Q3_K formats are
    not (#5014).

The declaration describes the **default runtime state** on the kernel's
primary device: a freshly loaded model, with no opt-in beyond what loading
already does. A kernel that only reaches a higher level after an explicit
opt-in (``TorchLinear.optimize()`` on CPU compiles its decode path, which then
traces to ``CUDAGRAPH``) declares the default level and records the opt-in in
its evidence comment, so a caller that enables a compiled path from the
declaration alone never over-claims.

``SUPPORTS_COMPILE`` (the pre-existing policy flag) stays separate and means
"compiling this forward is not a pessimization" for the static-cache
``generate()`` path (#4669). A kernel may be traceable and still keep it
``False`` when the eager path is already faster.

``SUPPORTS_COMPILE_EAGER_REASON: str``
    The recorded measured loss behind a ``SUPPORTS_COMPILE = False`` on a
    kernel that declares ``SUPPORTS_COMPILE_LEVEL >= INDUCTOR``. Those two
    declarations answer different questions, but a level at or above
    ``INDUCTOR`` is the documented bar for "a compiled decode can pay off", so
    opting out of the policy from there has to say why (a measured loss, not a
    guess). A kernel below ``INDUCTOR``, or one that opts in, leaves it empty.
    This is what stops the two declarations from drifting apart again (#4984).

The declarations are checked two ways:

* ``tests/kernels/test_kernel_compile_protocol.py`` requires every registered
  kernel to declare both attributes explicitly, and statically rejects a
  ``CUDAGRAPH`` / ``SUPPORTS_GRAPH_CAPTURE`` claim whose ``forward`` contains a
  known host-sync or host-transfer construct. It also requires the eager
  reason whenever a compile-capable kernel opts out of the policy.
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

#: At or above this level a compiled static-cache decode can pay off, so a
#: kernel that opts out of the #4669 policy must record the measured loss.
COMPILE_POLICY_FLOOR = COMPILE_LEVEL.INDUCTOR


def _coerce_level(kernel: Any, value: Any) -> COMPILE_LEVEL:
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


def kernel_device_type(kernel: Any) -> Optional[str]:
    """The ``torch.device.type`` an instance is resident on, if it has one."""

    if isinstance(kernel, type):
        return None
    for attribute in ("qweight", "scales", "qzeros", "codes"):
        tensor = getattr(kernel, attribute, None)
        device = getattr(tensor, "device", None)
        if device is not None:
            return device.type
    return None


def compile_level(kernel: Any, device_type: Optional[str] = None) -> COMPILE_LEVEL:
    """The declared level of a kernel class or instance.

    Accepts the ``str`` name too, so a serialized capability can round trip.
    An unknown or missing declaration is ``EAGER`` (the safe answer: nothing
    may assume a compiled path works).

    ``device_type`` selects a ``SUPPORTS_COMPILE_LEVEL_BY_DEVICE`` override.
    When it is omitted and ``kernel`` is an instance, the device of its packed
    buffers is used, so a caller never has to repeat what the module already
    knows. Only an override declared in the kernel's own class body applies;
    see the module docstring for why inheritance would over-claim.
    """

    if device_type is None:
        device_type = kernel_device_type(kernel)
    base = _coerce_level(
        kernel, getattr(kernel, "SUPPORTS_COMPILE_LEVEL", COMPILE_LEVEL.EAGER)
    )
    if base is COMPILE_LEVEL.NOT_APPLICABLE:
        # MLX/TTNN/XLA never run as a torch module, so no device can raise the
        # answer above "not applicable".
        return base
    owner = kernel if isinstance(kernel, type) else type(kernel)
    overrides = owner.__dict__.get("SUPPORTS_COMPILE_LEVEL_BY_DEVICE")
    if device_type and isinstance(overrides, dict) and device_type in overrides:
        return _coerce_level(kernel, overrides[device_type])
    return base


def graph_capture_safe(kernel: Any) -> bool:
    """Whether the eager forward may run inside a ``torch.cuda.CUDAGraph``.

    A class declaration answers for its default runtime state. An *instance*
    may refine it through ``supports_graph_capture()`` when only some of its
    runtime formats are capturable -- GGUF's host-dequantized Q2_K/Q3_K layers
    declare the class-level ``True`` for the device formats (#5014).

    The refinement may only *narrow* the declaration: a subclass that declares
    ``False`` (``GGUFCppKernel``, ``GGUFCudaKernel``) keeps answering ``False``
    even though it inherits the hook from ``GGUFTorchLinear``.
    """

    declared = bool(getattr(kernel, "SUPPORTS_GRAPH_CAPTURE", False))
    if not isinstance(kernel, type) and declared:
        refine = getattr(kernel, "supports_graph_capture", None)
        if callable(refine):
            return bool(refine())
    return declared


def reaches_level(kernel: Any, level: COMPILE_LEVEL) -> bool:
    """Whether the kernel declares at least ``level``."""

    return compile_level(kernel) >= level


def supports_compiled_generation(kernel: Any) -> bool:
    """The #4669 policy answer: is a compiled generation path not a pessimization?"""

    return bool(getattr(kernel, "SUPPORTS_COMPILE", False))


def compile_policy_reason(kernel: Any) -> Optional[str]:
    """The recorded measured loss for a compile-capable kernel that opts out.

    Returns ``None`` when the kernel records no reason. A reason must be a
    non-empty ``str``: the point is to publish *why* a traceable kernel keeps
    the static-cache auto-compile off, not to have a boolean placeholder.

    A ``NOT_APPLICABLE`` kernel (never a torch module) reports no reason: an
    MLX subclass of a torch kernel must not print the torch parent's CPU reason
    in the matrix.
    """

    owner = kernel if isinstance(kernel, type) else type(kernel)
    level = _coerce_level(
        kernel, getattr(owner, "SUPPORTS_COMPILE_LEVEL", COMPILE_LEVEL.EAGER)
    )
    if level is COMPILE_LEVEL.NOT_APPLICABLE:
        return None
    reason = getattr(kernel, "SUPPORTS_COMPILE_EAGER_REASON", "")
    if reason is None or reason == "":
        return None
    name = getattr(kernel, "__name__", type(kernel).__name__)
    if not isinstance(reason, str):
        raise TypeError(
            f"{name}: SUPPORTS_COMPILE_EAGER_REASON must be a str, "
            f"got {type(reason).__name__}."
        )
    reason = reason.strip()
    if not reason:
        # A blank string is "nothing recorded"; the protocol test then rejects a
        # compile-capable opt-out that has no reason.
        return None
    return reason


def compile_policy_is_justified(kernel: Any) -> bool:
    """Whether the policy flag agrees with the declared compile level (#4984).

    A kernel that reaches ``INDUCTOR`` or higher has a traceable, lowerable
    forward, which is the documented bar for "a compiled decode can pay off".
    Keeping the static-cache auto-compile off from there is allowed, but only
    with a recorded measured loss. Opting in from a level below ``INDUCTOR`` is
    the other drift: the compiled path would pay for graph fragments.

    The higher of the default-device level and the CPU level is used: a kernel
    that only reaches ``INDUCTOR`` on CPU still has a traceable CPU decode, so
    opting out of the policy from there needs the same recorded reason.
    """

    level = max(compile_level(kernel), compile_level(kernel, device_type="cpu"))
    reason = compile_policy_reason(kernel)
    if level >= COMPILE_POLICY_FLOOR:
        if supports_compiled_generation(kernel):
            # An opt-in from a compile-capable level: nothing to explain.
            return reason is None
        # An opt-out from a compile-capable level: the loss must be recorded.
        return reason is not None
    # Below the floor this pairing is not the question: the compiled path would
    # pay for graph fragments, and the `>= DYNAMO` opt-in check lives in the
    # protocol test. A reason there is harmless documentation.
    return True


def describe_kernel(kernel: Any) -> Dict[str, Optional[str]]:
    """A JSON-friendly capability row for docs, logs and the registry test."""

    return {
        "kernel": getattr(kernel, "__name__", type(kernel).__name__),
        "module": getattr(kernel, "__module__", type(kernel).__module__),
        "compile_level": compile_level(kernel).name,
        "compile_level_cpu": compile_level(kernel, device_type="cpu").name,
        "graph_capture": graph_capture_safe(kernel),
        "compiled_generation": supports_compiled_generation(kernel),
        "compiled_generation_reason": compile_policy_reason(kernel),
    }


__all__ = [
    "COMPILE_POLICY_FLOOR",
    "COMPILE_LEVEL",
    "LEVEL_PROBES",
    "compile_level",
    "compile_policy_is_justified",
    "compile_policy_reason",
    "describe_kernel",
    "graph_capture_safe",
    "kernel_device_type",
    "reaches_level",
    "supports_compiled_generation",
]
