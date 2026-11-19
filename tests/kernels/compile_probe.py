# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Measure a kernel's ``torch.compile`` / CUDA-graph capability.

Used by the per-family level tests and by maintainers adding a kernel. The
probe is deliberately construction-agnostic: the caller builds a real packed
module (each family already has helpers for that) and hands it here.

Levels, ascending (see ``gptqmodel/nn_modules/qlinear/_compile_support.py``):

* ``eager`` - the plain forward runs;
* ``dynamo`` - ``torch.compile(fullgraph=True, backend="eager")`` traces with
  no graph break;
* ``aot_eager`` - ``backend="aot_eager"`` (meta/fake tensors + aot_autograd);
* ``inductor`` - ``backend="inductor"`` lowers it;
* ``cudagraph`` - ``mode="reduce-overhead"`` captures it as a CUDA graph.

``graph_capture`` is measured separately with an explicit
``torch.cuda.graph(...)`` capture of the eager forward.
"""

from __future__ import annotations

import inspect
import io
import re
import tokenize
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from gptqmodel.nn_modules.qlinear._compile_support import COMPILE_LEVEL


#: Constructs that make an *eager* CUDA-graph capture invalid: each one either
#: synchronises the host, moves data to the CPU, or asks the device for an
#: address at Python level.
GRAPH_UNSAFE_PATTERNS = {
    ".item()": "host synchronisation",
    ".tolist()": "host transfer",
    ".cpu()": "host transfer",
    "data_ptr()": "device address read on the host",
    "torch.cuda.synchronize": "explicit host synchronisation",
    "cudaStreamSynchronize": "explicit host synchronisation",
}


@dataclass
class CompileProbe:
    """Measured capability of one kernel forward."""

    eager: bool = False
    eager_error: Optional[str] = None
    graph_breaks: Optional[int] = None
    break_reasons: List[str] = field(default_factory=list)
    levels: Dict[str, bool] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    graph_capture: Optional[bool] = None
    graph_capture_error: Optional[str] = None
    output_close: Dict[str, bool] = field(default_factory=dict)

    def reached(self, level: COMPILE_LEVEL) -> bool:
        if level is COMPILE_LEVEL.EAGER:
            return self.eager
        return bool(self.levels.get(level.name.lower()))

    @property
    def max_level(self) -> COMPILE_LEVEL:
        best = COMPILE_LEVEL.EAGER
        for level in (
            COMPILE_LEVEL.DYNAMO,
            COMPILE_LEVEL.AOT_EAGER,
            COMPILE_LEVEL.INDUCTOR,
            COMPILE_LEVEL.CUDAGRAPH,
        ):
            if self.levels.get(level.name.lower()):
                best = level
        return best

    def summary(self) -> str:
        parts = [f"eager={self.eager}"]
        if self.graph_breaks is not None:
            parts.append(f"graph_breaks={self.graph_breaks}")
        for name, ok in self.levels.items():
            parts.append(f"{name}={ok}")
        parts.append(f"graph_capture={self.graph_capture}")
        parts.append(f"max={self.max_level.name}")
        return " ".join(parts)


def _reset_dynamo() -> None:
    try:
        torch._dynamo.reset()
    except Exception:  # pragma: no cover - older torch without reset
        pass


def _try_compile(fn: Callable, args: Sequence[Any], *, backend: str, mode: Optional[str] = None):
    """Compile ``fn`` with ``fullgraph=True`` and run one call."""

    _reset_dynamo()
    kwargs: Dict[str, Any] = {"fullgraph": True, "backend": backend}
    if mode is not None:
        kwargs["mode"] = mode
    compiled = torch.compile(fn, **kwargs)
    with torch.inference_mode():
        return compiled(*args)


def probe(
    module: torch.nn.Module,
    args: Sequence[Any],
    *,
    check_cuda_graph: Optional[bool] = None,
    close_tol: float = 2e-2,
) -> CompileProbe:
    """Measure every level for ``module.forward(*args)``.

    ``check_cuda_graph`` defaults to whether CUDA is available. Set it ``False``
    to skip the explicit capture (for example on a CPU-only host).
    """

    fn = module.forward
    result = CompileProbe()

    try:
        with torch.inference_mode():
            reference = fn(*args)
        result.eager = True
    except Exception as error:  # pragma: no cover - reported by the caller
        result.eager_error = f"{type(error).__name__}: {error}"
        return result

    def compare(output: Any, level: str) -> None:
        if not isinstance(output, torch.Tensor) or not isinstance(reference, torch.Tensor):
            return
        try:
            result.output_close[level] = bool(
                torch.allclose(output.float(), reference.float(), atol=close_tol, rtol=1e-2)
            )
        except Exception:
            result.output_close[level] = False

    # Graph breaks: `explain` runs the forward under dynamo and reports every
    # place it gave up. It does not raise, so it is the only probe that works
    # for kernels that are not traceable.
    try:
        _reset_dynamo()
        try:
            # torch >= 2.12: explain(f)(*args)
            explained = torch._dynamo.explain(fn)(*args)
        except TypeError:  # pragma: no cover - older signature
            explained = torch._dynamo.explain(fn, *args)
        result.graph_breaks = int(explained.graph_break_count)
        result.break_reasons = [
            re.sub(r"\s+", " ", str(reason.reason))[:300] for reason in explained.break_reasons
        ]
    except Exception as error:
        result.errors["explain"] = f"{type(error).__name__}: {error}"

    attempts = (
        ("dynamo", {"backend": "eager"}),
        ("aot_eager", {"backend": "aot_eager"}),
        ("inductor", {"backend": "inductor"}),
        ("cudagraph", {"backend": "inductor", "mode": "reduce-overhead"}),
    )
    cpu_inputs = all(isinstance(arg, torch.Tensor) and arg.device.type == "cpu" for arg in args)
    for name, kwargs in attempts:
        if not torch.cuda.is_available() and name == "cudagraph":
            # CUDA-graph capture has no CPU equivalent: `mode="reduce-overhead"`
            # is a CUDA-only graph backend, so this level is CUDA-only by
            # definition. Inductor *does* codegen for CPU, so it stays probed.
            result.levels[name] = False
            result.errors[name] = "skipped: CUDA graph capture is CUDA-only"
            continue
        if name == "inductor" and not torch.cuda.is_available() and not cpu_inputs:
            result.levels[name] = False
            result.errors[name] = "skipped: inductor needs a CPU or CUDA device for these inputs"
            continue
        try:
            output = _try_compile(fn, args, **kwargs)
            result.levels[name] = True
            compare(output, name)
        except Exception as error:
            result.levels[name] = False
            result.errors[name] = f"{type(error).__name__}: {str(error)[:400]}"

    wants_capture = (
        torch.cuda.is_available() if check_cuda_graph is None else check_cuda_graph
    )
    if wants_capture and torch.cuda.is_available():
        result.graph_capture, result.graph_capture_error = _probe_cuda_graph(
            fn, args, reference
        )

    _reset_dynamo()
    return result


def _probe_cuda_graph(fn: Callable, args: Sequence[Any], reference: Any):
    """Capture one eager call in a ``torch.cuda.CUDAGraph``."""

    if not all(isinstance(arg, torch.Tensor) and arg.is_cuda for arg in args):
        return None, "skipped: non-CUDA inputs"
    static_args = [arg.clone() for arg in args]
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn(*static_args)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = fn(*static_args)
        for arg, static in zip(args, static_args):
            static.copy_(arg)
        graph.replay()
        torch.cuda.synchronize()
        if isinstance(output, torch.Tensor) and isinstance(reference, torch.Tensor):
            close = bool(
                torch.allclose(output.float(), reference.float(), atol=2e-2, rtol=1e-2)
            )
            if not close:
                return False, "captured output differs from eager"
        return True, None
    except Exception as error:
        return False, f"{type(error).__name__}: {str(error)[:400]}"


def protocol_kernels():
    """Every quantized kernel that must declare the protocol.

    Registered kernels plus the role-selected modules that skip backend
    discovery (`SUPPORTS_BACKEND_SELECTION = False`), which are still user-facing
    quantized layers.
    """

    from gptqmodel.nn_modules.qlinear.torch import TorchQuantEmbeddings
    from gptqmodel.utils import importer

    return sorted(
        set(importer._registry_kernel_classes()) | {TorchQuantEmbeddings},
        key=lambda cls: cls.__name__,
    )


def skip_without_measurable_device(kernel_cls, *, device_type: Optional[str] = None) -> None:
    """Skip when the declared level cannot be measured on this host.

    Only ``CUDAGRAPH`` and a CUDA-graph *capture* claim are device-only: both
    need a CUDA device at measurement time. ``INDUCTOR`` lowers on CPU too, so a
    CPU-built case that declares it must still run on a CPU-only runner instead
    of turning into a skip (#5110). Pass ``device_type="cpu"`` for a case whose
    module is built on the CPU; the per-device override then decides.
    """

    from gptqmodel.nn_modules.qlinear._compile_support import (
        COMPILE_LEVEL,
        compile_level,
        graph_capture_safe,
    )

    declared = compile_level(kernel_cls, device_type=device_type)
    needs_cuda = declared >= COMPILE_LEVEL.CUDAGRAPH
    if device_type != "cpu" and graph_capture_safe(kernel_cls):
        needs_cuda = True
    if needs_cuda and not torch.cuda.is_available():
        import pytest

        pytest.skip(
            f"{getattr(kernel_cls, '__name__', kernel_cls)} declares "
            f"{declared.name} / capture={graph_capture_safe(kernel_cls)}, "
            "which needs a CUDA device to measure"
        )


def _strip_comments_and_strings(source: str) -> str:
    """Blank out comments and string literals before the static scan.

    A comment that *documents* a host transfer (``# the op reads data_ptr()``)
    is not a host transfer, and an error string is not an executed call. Only
    real code can invalidate a capture, so scan the token stream with
    ``COMMENT``/``STRING`` tokens removed.
    """

    skip = {tokenize.COMMENT, tokenize.STRING}
    pieces: List[str] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            pieces.append(" " if token.type in skip else token.string)
    except (tokenize.TokenError, IndentationError):
        return source
    return "".join(pieces)


def graph_unsafe_constructs(kernel_cls: type) -> List[str]:
    """Static scan of the kernel's own ``forward`` for capture-unsafe calls."""

    forward = getattr(kernel_cls, "forward", None)
    if forward is None:
        return []
    try:
        source = inspect.getsource(forward)
    except (OSError, TypeError):
        return []
    code = _strip_comments_and_strings(source)
    found = []
    for pattern, reason in GRAPH_UNSAFE_PATTERNS.items():
        if pattern in code:
            found.append(f"{pattern} ({reason})")
    return found
