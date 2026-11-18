# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import contextlib
import functools
import importlib
import os
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from enum import Enum
from functools import lru_cache
from typing import Callable, List, Optional, Union

import torch
from packaging import version
from torch.cpu import StreamContext

from ..utils.logger import setup_logger
from ..utils.safe import GC
from . import gte_python_3_13_3, gte_python_3_14, has_gil_disabled, log_gil_requirements_for


# pytorch 2.6.0 fixes many compilation errors
TORCH_HAS_COMPILE = version.parse(torch.__version__).release >= version.Version('2.6').release
TORCH_GTE_28 = version.parse(torch.__version__).release >= version.Version('2.8').release
TORCH_GTE_210 = version.parse(torch.__version__).release >= version.Version('2.10').release

TORCH_HAS_FUSED_OPS = version.parse(torch.__version__).release >= version.Version('2.8').release

HAS_CUDA = False
HAS_XPU = False
HAS_MPS = False
HAS_MLX = False
HAS_NPU = False
HAS_XLA = False

CPU = torch.device("cpu")
META = torch.device("meta")


def as_torch_device(device_like) -> torch.device:
    """Resolve a device-like value without assigning an implicit ordinal.

    GPT-QModel's ``DEVICE`` enum names a kernel support family, and its
    ``DEVICE.ROCM`` member intentionally has the value ``"rocm"`` even though
    PyTorch exposes ROCm through the ``cuda`` namespace.  ``torch.device``
    therefore rejects it. Resolve enums through their runtime ``type`` while
    keeping families indexless, just like ``torch.device("cuda")``. Calling
    ``to_torch_device()`` here would pin family selectors to device 0 instead
    of honoring the caller's current CUDA/XPU/NPU device.
    """
    if isinstance(device_like, torch.device):
        return device_like
    resolver = getattr(device_like, "to_torch_device", None)
    if callable(resolver):
        return torch.device(device_like.type)
    return torch.device(device_like)


class BalanceStrategy(str, Enum):
    MEMORY = "memory", # make vram more spread out
    GPU = "gpu" # vram is less balanced (gpu0) but gpu0 is also used for quantization

DEFAULT_BALANCE_STRATEGY = BalanceStrategy.GPU

# TODO FIX ME...this should be removed
STREAM = None # cache

log = setup_logger()


def timed_gc_collect() -> int:
    """Run ``gc.collect`` and log the elapsed time along with reclaimed object count."""
    start = time.perf_counter()

    # Python 3.14 removed gen1 so there is only gen0 and gen2
    collected = GC.collect()

    duration = time.perf_counter() - start
    log.info(f"gc.collect() reclaimed {collected} objects in {duration:.3f}s")
    return collected

def configure_dynamo_cache(*, ensure_import: bool = False) -> bool:
    """
    Raise dynamo's recompile cache limit to the project minimum (128).

    `torch._dynamo` is imported lazily: pulling it in costs about a second of
    process startup because it drags `torch._inductor` (and its cudagraph
    machinery) along with it, and the cache limit only matters once something
    actually compiles. Callers that are about to compile pass `ensure_import=True`
    so the limit is in place before the first `torch.compile`.
    """
    dynamo = sys.modules.get("torch._dynamo")
    if dynamo is None:
        if not ensure_import:
            return False
        try:
            import torch._dynamo as dynamo
        except Exception:
            # triton built from source maybe incompatible with _dynamo private api
            return False

    try:
        # default of 8 is too low: ci loop model inference may exhaust the cache
        if dynamo.config.cache_size_limit < 128:
            dynamo.config.cache_size_limit = 128
        return True
    except Exception:
        # triton built from source maybe incompatible with _dynamo private api
        return False

try:
    HAS_CUDA = torch.cuda.is_available()
except Exception:
    HAS_CUDA = False

try:
    HAS_XPU = torch.xpu.is_available()
except Exception:
    HAS_XPU = False

try:
    HAS_MPS = torch.mps.is_available()
except Exception:
    HAS_MPS = False


def _ascend_runtime_env_ready() -> bool:
    # torch_npu may report available before the CANN environment is fully
    # sourced. Requiring the standard Ascend paths avoids import-time lazy
    # initialization failures when the Python package is installed but the
    # runtime is not usable in this shell.
    return any(
        os.environ.get(name)
        for name in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH")
    )


try:
    importlib.import_module("torch_npu")
    HAS_NPU = _ascend_runtime_env_ready() and torch.npu.is_available()
except Exception:
    HAS_NPU = False


def _xla_package_installed() -> bool:
    """Return whether the optional Torch/XLA package can be imported.

    This probe must remain lazy: importing ``torch_xla`` initializes its PJRT
    runtime and, with ``pjrt-plugin-tt``, opens every Tenstorrent device.  A
    plain ``import gptqmodel`` on an accelerator host must not claim hardware
    or contend with another process using those devices.
    """
    return importlib.util.find_spec("torch_xla") is not None


def require_xla_runtime():
    """Import and return ``torch_xla`` for an explicit XLA request.

    Callers use this only after selecting or constructing an ``xla`` device.
    The returned module is the authoritative owner of the PJRT runtime.
    """
    try:
        module = importlib.import_module("torch_xla")
    except Exception as exc:
        raise RuntimeError(
            "PyTorch/XLA was requested, but `torch_xla` could not be initialized. "
            "Install the pinned Tenstorrent environment from "
            "`requirements/tenstorrent-py312.txt` and check the active PJRT plugin."
        ) from exc
    return module


def xla_device_count() -> int:
    """Return the addressable XLA device count, recording that PJRT now owns them.

    ``addressable_device_count()`` is the call that actually initializes the
    PJRT plugin, and with ``pjrt-plugin-tt`` that opens every Tenstorrent chip.
    TT-Metal allows a single owner per chip, so the whole process becomes
    unable to use the native TTNN path from this point on. The flag recorded
    here is the only reliable signal of that state -- importing ``torch_xla``
    alone is inert, so a bare ``sys.modules`` check cannot tell the two apart.
    """

    runtime = require_xla_runtime()
    count = int(runtime.runtime.addressable_device_count())
    # Mark the claim only after the runtime answered, so a failed probe does not
    # leave a lingering "claimed" flag that blocks a later native TTNN attempt.
    try:
        runtime._gptqmodel_device_claimed = True
    except Exception:  # pragma: no cover - module objects always accept attributes
        pass
    return count


def is_xla_available() -> bool:
    """Probe XLA availability without initializing hardware at import time."""
    if not _xla_package_installed():
        return False
    try:
        return xla_device_count() > 0
    except Exception:
        return False


def xla_runtime_claimed() -> bool:
    """Whether this process has already initialized XLA and taken its devices.

    Importing ``torch_xla`` is inert, so a ``sys.modules`` check alone cannot
    distinguish "installed" from "in use". Only the flag set after
    ``addressable_device_count()`` answered means the runtime is live, which is
    what a generic barrier may synchronize. Anything else would discover and
    claim another runtime from a job that never selected it.
    """

    module = sys.modules.get("torch_xla")
    return bool(module is not None and getattr(module, "_gptqmodel_device_claimed", False))


# Availability is deliberately not probed here.  Use ``is_xla_available()``
# from an explicit device-resolution path; see ``require_xla_runtime``.
HAS_XLA = _xla_package_installed()


mlx = None


def _detect_mlx() -> bool:
    """Probe for a usable MLX install, binding the package on success.

    MLX is an optional accelerator: absence (or a broken native install) is
    expected on most hosts, so probe failures are reported as "unavailable".
    ``except Exception`` is deliberate -- ``KeyboardInterrupt``/``SystemExit``
    must stay visible rather than being reported as a missing dependency.

    ``mlx`` and ``HAS_MLX`` are updated together so callers guarded by
    ``if HAS_MLX: mlx.core.clear_cache()`` can never see one without the other.
    A partially-installed package binds ``mlx`` while ``mlx.core.metal`` raises,
    so the failure path clears it again.
    """
    global HAS_MLX, mlx

    try:
        import mlx.core  # type: ignore # noqa: F401
        import mlx.core.metal  # noqa: F401
    except Exception:
        HAS_MLX = False
        mlx = None
        return HAS_MLX

    HAS_MLX = True
    return HAS_MLX


# mlx check
HAS_MLX = _detect_mlx()

try:
    torch.backends.fp32_precision
    BACKENDS_HAS_FP32_PRECISION = True
except AttributeError:
    BACKENDS_HAS_FP32_PRECISION = False


def _set_tf32_state(enabled: bool) -> None:
    if BACKENDS_HAS_FP32_PRECISION:
        mode = "tf32" if enabled else "ieee"
        torch.backends.fp32_precision = mode
        torch.backends.cuda.matmul.fp32_precision = mode
        torch.backends.cudnn.fp32_precision = mode
        torch.backends.cudnn.conv.fp32_precision = mode
        torch.backends.cudnn.rnn.fp32_precision = mode
        return

    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def _snapshot_tf32_state():
    if BACKENDS_HAS_FP32_PRECISION:
        return (
            torch.backends.fp32_precision,
            torch.backends.cuda.matmul.fp32_precision,
            torch.backends.cudnn.fp32_precision,
            torch.backends.cudnn.conv.fp32_precision,
            torch.backends.cudnn.rnn.fp32_precision,
        )

    return (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )


def _restore_tf32_state(state) -> None:
    if BACKENDS_HAS_FP32_PRECISION:
        torch.backends.fp32_precision = state[0]
        torch.backends.cuda.matmul.fp32_precision = state[1]
        torch.backends.cudnn.fp32_precision = state[2]
        torch.backends.cudnn.conv.fp32_precision = state[3]
        torch.backends.cudnn.rnn.fp32_precision = state[4]
        return

    torch.backends.cuda.matmul.allow_tf32 = state[0]
    torch.backends.cudnn.allow_tf32 = state[1]


_TF32_GUARD_LOCK = threading.RLock()
_TF32_GUARD_DEPTH = 0
_TF32_GUARD_PREVIOUS = None


def torch_compile_disabled() -> bool:
    """Report whether torch compilation is switched off for this process.

    ``TORCHDYNAMO_DISABLE=1`` is PyTorch's own kill switch. Dynamo already turns
    ``torch.compile`` into a no-op for it; checking here keeps GPT-QModel from
    importing dynamo just to build a wrapper that would never trace.
    """
    return os.environ.get("TORCHDYNAMO_DISABLE", "").strip() == "1"


def _dynamo_exception(*names: str):
    """Resolve a dynamo exception class without importing ``torch._dynamo``."""
    dynamo = sys.modules.get("torch._dynamo")
    exc_module = getattr(dynamo, "exc", None)
    return tuple(
        cls for cls in (getattr(exc_module, name, None) for name in names)
        if isinstance(cls, type)
    )


def _is_compiler_failure(exc: BaseException, *, fullgraph: bool = False) -> bool:
    """Report whether ``exc`` came from the compiler instead of traced user code.

    ``torch.compile`` only builds the wrapper; inductor actually compiles on the
    first invocation, so the construction-time guard cannot observe these. A
    backend failure is always eligible for the eager fallback:
    ``torch._dynamo.exc.BackendCompilerFailed``, which also carries the
    ``RuntimeError: Attempting to use a closed pool`` raised by an inductor
    compile worker. Capture failures such as ``Unsupported`` are only eligible
    without ``fullgraph``: ``fullgraph=True`` is an explicit single-graph
    contract and must still fail closed.

    The remaining fallback is deliberately narrow: a ``RuntimeError`` raised
    inside PyTorch's compiler packages, a ``LookupError`` raised there (dynamo's
    guard bookkeeping raises ``KeyError`` from ``get_guard_fail_reason`` when a
    concurrent compile invalidates its state, issue #5127), or dynamo's
    guard-check ``AssertionError`` (``Guard failed on the same frame it was
    created ... GLOBAL_STATE changed``) which fires when another thread changes
    a process-global setting, such as ``torch.set_default_dtype``, during a
    concurrent first compile (issue #4094). Everything else belongs to the
    traced user code; retrying eagerly would run that code twice and hide the
    original error. The innermost frame is what matters: user code raising the
    same exception type has its own frame at the origin.
    """
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return False
    if isinstance(exc, _dynamo_exception("BackendCompilerFailed", "TorchRuntimeError")):
        return True
    if fullgraph:
        return False
    chain = []
    current = exc
    while current is not None and len(chain) < 8:
        chain.append(current)
        current = current.__cause__ or current.__context__
    if any(isinstance(item, _dynamo_exception("TorchDynamoException")) for item in chain):
        return True
    # Dynamo's own frame sits between the caller and the traced body, so a
    # "contains a compiler frame" test would also match user code raising the
    # same exception. Only the innermost frame identifies where it came from:
    # compile workers and guard checks raise outside the traced frame.
    frames = traceback.extract_tb(exc.__traceback__)
    origin = (frames[-1].filename or "") if frames else ""
    if not any(
        token in origin
        for token in ("/torch/_dynamo/", "/torch/_inductor/", "/torch/_functorch/")
    ):
        return False
    if isinstance(exc, (RuntimeError, LookupError)):
        # Inductor compile workers raise RuntimeError outside the traced frame
        # (for example ``Attempting to use a closed pool``), and dynamo's guard
        # bookkeeping raises KeyError from ``get_guard_fail_reason`` when a
        # concurrent compile changed the process-global state (#5127).
        return True
    return isinstance(exc, AssertionError) and "Guard failed on the same frame it was created" in str(exc)


def _compile_label(target) -> str:
    """Describe the compiled unit in the one warning emitted on fallback."""
    name = getattr(target, "__qualname__", None) or getattr(target, "__name__", None)
    if name:
        return str(name)
    return type(target).__name__


def _describe_compile_failure(exc: BaseException) -> str:
    """Render the outermost and innermost compiler error on one line.

    Dynamo wraps backend errors (``BackendCompilerFailed``) around the real
    cause, for example ``RuntimeError: Attempting to use a closed pool``. The
    inner message is the actionable one, so both are reported.
    """
    def first_line(error):
        lines = str(error).strip().splitlines()
        return lines[0] if lines else ""

    root = exc
    seen = 0
    while (root.__cause__ is not None or root.__context__ is not None) and seen < 8:
        root = root.__cause__ or root.__context__
        seen += 1
    outer = f"{type(exc).__name__}: {first_line(exc)}"
    if root is exc:
        return outer
    return f"{outer} | root cause {type(root).__name__}: {first_line(root)}"


def _warn_compile_fallback(label: str, exc: BaseException) -> None:
    log.warn.once(
        f"`{label}` torch.compile failed at first use; staying on eager execution: "
        f"{_describe_compile_failure(exc)}"
    )


def _dynamo_config():
    """Return ``torch._dynamo.config`` without importing dynamo, or ``None``.

    Importing dynamo only to read one flag would force its side effects on a
    process that never compiled anything. ``torch_compile`` imports it
    (``configure_dynamo_cache(ensure_import=True)``) before a wrapper exists, so
    it is already present whenever this runs in practice.
    """
    dynamo = sys.modules.get("torch._dynamo")
    return getattr(dynamo, "config", None)


def dynamo_is_compiling() -> bool:
    """Return True when dynamo is already tracing this thread.

    ``torch.compile`` is lazy, so the wrapper ``torch_compile`` installs can be
    entered from inside another ``torch.compile`` trace: transformers compiles
    ``generate``'s forward, and ``VPTQLinear.optimize()`` installs one of these
    wrappers on ``dequantize`` at post-init. ``suppress_errors`` is a ContextVar,
    so writing it there is a ``ContextVar mutation not supported`` graph break in
    the outer trace, which turns a cudagraph static-cache decode into
    overwritten-output corruption (#5224). Nested calls therefore stay
    read-only; the eager body is inlined into the outer graph instead.
    """
    compiler = getattr(torch, "compiler", None)
    is_compiling = getattr(compiler, "is_compiling", None)
    if is_compiling is None:
        return False
    try:
        return bool(is_compiling())
    except Exception:
        return False


def _call_with_dynamo_suppression(compiled: Callable, fullgraph: bool, args: tuple, kwargs: dict):
    """Invoke ``compiled`` with dynamo allowed to fall back for the failing frame.

    ``fullgraph=True`` is an explicit single-graph contract and stays
    fail-closed, so suppression is not enabled for it. The previous flag value
    is restored in a ``finally`` block: this is a per-call behaviour switch, not
    a process setting, and a concurrent compile in another thread must not see
    it (a GLOBAL_STATE guard failure from that race is issue #4094).

    A nested compile is already inside an outer trace, where the ContextVar
    write is untraceable; the caller normally short-circuits to the eager body
    before reaching here, and this guard keeps the helper safe on its own.
    """
    if fullgraph or dynamo_is_compiling():
        return compiled(*args, **kwargs)
    config = _dynamo_config()
    if config is None:
        return compiled(*args, **kwargs)
    previous = getattr(config, "suppress_errors", False)
    config.suppress_errors = True
    try:
        return compiled(*args, **kwargs)
    finally:
        config.suppress_errors = previous


class _ModelGlobalStateLock:
    """Process-wide reader/writer lock for torch's global default dtype.

    transformers' ``from_pretrained``/``from_config`` saves, sets and restores
    ``torch.set_default_dtype``; dynamo's ``preserve_global_state`` does the
    same around every trace, and its ``GLOBAL_STATE`` guard re-reads the dtype
    on every compiled call. A compiled call that re-traces inside a load's bf16
    window would restore bf16 after the load restored fp32 (or trip
    ``assert_loaded_model_state``, #5127/#5209), so it may not overlap that
    window.

    Compiled calls only read the process state: they take :meth:`reader` and
    run concurrently with each other. Model construction changes it: it takes
    the writer (also via ``with``), which excludes every in-flight compiled
    call but holds for the construction window only, not for the rest of a
    load. The lock is re-entrant on both sides for the owning thread: a load
    may execute compiled code while it owns the writer, and a compiled call may
    nest into another compiled call.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._readers = 0
        self._writer = None
        self._writer_depth = 0
        self._waiting_writers = 0
        self._local = threading.local()

    def __enter__(self):
        self.acquire_write()
        return self

    def __exit__(self, *_exc):
        self.release_write()
        return False

    def reader(self):
        """Return a context manager holding shared access for one compiled call."""

        return _ModelGlobalStateReader(self)

    def holds_reader(self) -> bool:
        """Whether the calling thread currently holds the shared side."""

        return bool(getattr(self._local, "read_depth", 0))

    def _acquire_reader(self) -> bool:
        """Take the shared side; return False when this thread owns the writer."""

        ident = threading.get_ident()
        if self._writer == ident:
            # A load thread running compiled code is already exclusive; taking
            # a reader here would deadlock against its own writer.
            return False

        depth = getattr(self._local, "read_depth", 0)
        with self._condition:
            if depth == 0:
                # Writer preference: once a load is waiting, new calls queue
                # behind it instead of starving it with a stream of readers.
                while self._writer is not None or self._waiting_writers:
                    self._condition.wait()
            self._readers += 1
        self._local.read_depth = depth + 1
        return True

    def _release_reader(self) -> None:
        self._local.read_depth -= 1
        with self._condition:
            self._readers -= 1
            if self._readers == 0:
                self._condition.notify_all()

    def acquire_write(self):
        ident = threading.get_ident()
        with self._condition:
            if self._writer == ident:
                self._writer_depth += 1
                return
            if getattr(self._local, "read_depth", 0):
                # Upgrading a reader to a writer can deadlock against another
                # reader; the compiled path never needs it, so fail loudly
                # instead of hanging a load.
                raise RuntimeError(
                    "model global-state lock cannot upgrade a reader to a writer"
                )
            self._waiting_writers += 1
            try:
                while self._writer is not None or self._readers:
                    self._condition.wait()
            finally:
                self._waiting_writers -= 1
            self._writer = ident
            self._writer_depth = 1

    def release_write(self):
        with self._condition:
            self._writer_depth -= 1
            if self._writer_depth == 0:
                self._writer = None
                self._condition.notify_all()


class _ModelGlobalStateReader:
    """One compiled call's shared hold on :class:`_ModelGlobalStateLock`.

    A plain object instead of a generator-based context manager: the compiled
    proxies enter this on every forward, so the per-call cost matters.
    """

    __slots__ = ("_lock", "_held")

    def __init__(self, lock: _ModelGlobalStateLock):
        self._lock = lock
        self._held = False

    def __enter__(self):
        self._held = self._lock._acquire_reader()
        return self

    def __exit__(self, *_exc):
        if self._held:
            self._held = False
            self._lock._release_reader()
        return False


_MODEL_GLOBAL_STATE_ACCESSOR = None


def model_global_state_lock():
    """Return the process-wide reader/writer lock for torch's global dtype.

    Loads and the save-side shell rebuild take the writer around the
    transformers construction call that saves, sets and restores
    ``torch.set_default_dtype``. Compiled calls take :meth:`reader`, so they
    overlap each other and the rest of a load, but never the dtype window.
    Look the lock up through the loader accessor so tests and callers share
    one object.
    """
    global _MODEL_GLOBAL_STATE_ACCESSOR

    accessor = _MODEL_GLOBAL_STATE_ACCESSOR
    if accessor is None:
        # Resolve lazily: the loader imports this module at import time. The
        # accessor is cached because a compiled call resolves this on every
        # forward; the accessor itself still returns the loader's current lock
        # object, so tests and callers that replace it keep working.
        from ..models.loader import model_global_state_lock as accessor

        _MODEL_GLOBAL_STATE_ACCESSOR = accessor

    return accessor()


def _install_callable_compile_fallback(original: Callable, compiled: Callable, fullgraph: bool) -> Callable:
    """Return a callable that degrades to eager after a compile failure.

    The failure is never retried by re-running ``original`` from the start. With
    ``fullgraph=False`` dynamo executes the forward as compiled fragments plus
    resume frames, so a compiler failure can arrive after earlier frames already
    mutated state (the KV cache in ``generate()``, Mamba/SSM state, offload
    hooks). Re-running there applies those side effects twice -- the first
    ``generate()`` after ``optimize()`` appended every KV entry a second time and
    died on the mask length (#5119).

    ``torch._dynamo.config.suppress_errors`` makes dynamo skip only the failing
    frame and run that frame eagerly, which executes nothing twice. The flag is
    scoped to this call and restored afterwards; the skip dynamo installs is
    sticky, so later calls stay eager without it. For ``fullgraph=True`` nothing
    has executed yet when the compiler fails, so the eager retry is exact.
    """
    label = _compile_label(original)
    state = {"fallback": False}

    def call(*args, **kwargs):
        if state["fallback"]:
            return original(*args, **kwargs)
        if dynamo_is_compiling():
            # An outer torch.compile is already tracing this frame. Its trace
            # owns error handling; mutate no dynamo config and inline the eager
            # body instead of opening a redundant nested compile (#5224).
            return original(*args, **kwargs)

        def run_compiled():
            try:
                return _call_with_dynamo_suppression(
                    compiled, fullgraph, args, kwargs
                )
            except Exception as exc:
                if not _is_compiler_failure(exc, fullgraph=fullgraph):
                    raise
                state["fallback"] = True
                _warn_compile_fallback(label, exc)
                return original(*args, **kwargs)

        # Dynamo saves/restores the process-global default dtype around every
        # trace and its GLOBAL_STATE guard re-reads it on every call. Any call
        # can re-trace when a guard changes -- not just the first -- so a call
        # may not overlap a load's dtype window; otherwise the window is
        # captured by a re-trace and restored after the load restored fp32
        # (#5127, #5209). The reader side is shared: compiled calls run
        # concurrently with each other and with the rest of a load, and only
        # wait while construction owns the writer. Nested calls from this
        # thread are re-entrant.
        with model_global_state_lock().reader():
            if state["fallback"]:
                return original(*args, **kwargs)
            return run_compiled()

    functools.update_wrapper(call, original)
    # Keep the dynamo wrapper chain truthful: this proxy really does forward to
    # `original`, and `_torchdynamo_wrapper_id` must name this wrapper. Kernels
    # and the fusion lifecycle use the presence of the chain to know that a
    # trace may have run.
    call._torchdynamo_orig_callable = original
    call._torchdynamo_wrapper_id = id(call)
    for attr in ("_is_torch_compile", "_isolate_recompiles_id", "aot_compile",
                 "get_compiler_config"):
        call.__dict__.pop(attr, None)
    return call


def _install_module_compile_fallback(module: torch.nn.Module, compiled, fullgraph: bool):
    """Install the same fallback on a compiled module's ``forward``.

    ``torch.compile(nn.Module)`` returns an ``OptimizedModule`` that callers keep
    because it is still an ``nn.Module`` (attribute delegation, ``modules()`` and
    hooks depend on it). Replacing ``forward`` leaves that identity intact while
    a deferred compile failure runs the original module instead.
    """
    if not isinstance(compiled, torch.nn.Module):
        # Callers assign the result back to an nn.Module attribute
        # (``BaseQModel.optimize`` does ``self.model = torch_compile(...)``), so
        # never return a plain callable for a module input. Keep the eager module
        # and report the unusable compiled object once.
        log.warn.once(
            f"`{_compile_label(module)}` torch.compile returned "
            f"{type(compiled).__name__} instead of nn.Module; using eager execution."
        )
        return module
    label = _compile_label(module)
    compiled_forward = compiled.forward
    state = {"fallback": False}

    def forward(*args, **kwargs):
        if state["fallback"]:
            return module(*args, **kwargs)
        if dynamo_is_compiling():
            # See `_install_callable_compile_fallback.call`: a nested module
            # forward is inlined into the outer graph without a config write.
            return module(*args, **kwargs)

        def run_compiled():
            try:
                return _call_with_dynamo_suppression(
                    compiled_forward, fullgraph, args, kwargs
                )
            except Exception as exc:
                if not _is_compiler_failure(exc, fullgraph=fullgraph):
                    raise
                state["fallback"] = True
                _warn_compile_fallback(label, exc)
                return module(*args, **kwargs)

        # Every forward can re-trace and mutate the process-global default
        # dtype (see the callable proxy above), so it may not overlap a load's
        # dtype window. The reader side is shared, so forwards from different
        # threads run concurrently and only wait for construction.
        with model_global_state_lock().reader():
            if state["fallback"]:
                return module(*args, **kwargs)
            return run_compiled()

    compiled.forward = forward
    return compiled


def dynamo_traced_frames() -> int:
    """Number of frames ``torch.compile`` has traced in this process.

    Dynamo traces a frame lazily on its first call and traces another one for
    every new shape or code path, so a change in this count means the compiled
    call that just returned produced fresh trace state (#5020).
    """

    dynamo = sys.modules.get("torch._dynamo")
    if dynamo is None:
        return 0
    counters = getattr(getattr(dynamo, "utils", None), "counters", None)
    if counters is None:
        return 0
    try:
        return int(counters["frames"]["total"])
    except (KeyError, TypeError, ValueError):
        return 0


def release_compile_trace_garbage() -> int:
    """Break the reference cycles that a compiled call leaves behind.

    Every frame dynamo traces builds ``VariableTracker`` objects. A graph break
    leaves an ``UnspecializedNNModuleVariable`` -> ``SideEffectsProxyDict`` ->
    ``DunderDictVariable`` cycle that keeps the traced module reachable. The
    cycle is unreachable garbage, but reference counting cannot reclaim it, so a
    compiled model - with all of its weights and VRAM - survives the last user
    reference until a full cyclic ``gc.collect()`` (#4937, #5020).

    Only those cycles are collected. Dynamo's compiled-frame cache is left
    intact, so the next call still takes the compiled path instead of
    recompiling. Returns the number of unreachable objects reclaimed.
    """

    return GC.collect()


def torch_compile(module: Union[torch.nn.Module, Callable], backend:str ="inductor", mode: str = None, fullgraph=False):
    # requires torch >2.8 for proper torch.compile + Python 3.13.3t (freethreading)
    if has_gil_disabled() and not gte_python_3_13_3():
        log_gil_requirements_for("Torch Compile")
        return module

    if gte_python_3_14() and not TORCH_GTE_210:
        log_gil_requirements_for("Torch Compile")
        return module

    if not TORCH_HAS_COMPILE:
        return module
    if torch_compile_disabled():
        log.info.once("Optimize: torch.compile disabled; using eager execution.")
        return module
    configure_dynamo_cache(ensure_import=True)
    if HAS_MPS and not TORCH_GTE_28:
        if not torch._dynamo.config.suppress_errors:
            log.warn("To use compile() with MPS, you need to have torch version >= 2.8.0, "
                     "please upgrade it by `pip install -U torch torchaudio torchvision`")
            torch._dynamo.config.suppress_errors = True
        return module
    try:
        compiled = torch.compile(module, backend=backend, mode=mode, fullgraph=fullgraph)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        log.warn.once(f"Failed to compile `{module}`, {e}")
        return module
    if compiled is module:
        return module
    # Inductor compiles on the first call, not here. Keep the eager path
    # reachable so a compile-worker failure degrades instead of aborting.
    if isinstance(module, torch.nn.Module):
        return _install_module_compile_fallback(module, compiled, fullgraph)
    return _install_callable_compile_fallback(module, compiled, fullgraph)


def torch_new_stream():
    global STREAM
    if STREAM is not None:
        return STREAM

    if HAS_CUDA:
        STREAM = torch.cuda.Stream()
        return STREAM
    if HAS_XPU:
        STREAM = torch.xpu.Stream()
        return STREAM
    if HAS_NPU:
        STREAM = torch.npu.Stream()
        return STREAM
    if is_xla_available():
        # Torch/XLA supplies its own lazy execution and synchronization model.
        return None
    return None

def torch_new_stream_ctx():
    if HAS_CUDA:
        return torch.cuda.stream(torch_new_stream())
    if HAS_XPU:
        return torch.xpu.stream(torch_new_stream())
    if HAS_NPU:
        return torch.npu.stream(torch_new_stream())
    if is_xla_available():
        return contextlib.nullcontext()
    return contextlib.nullcontext()

def cuda_primary_context_claimed(index: int) -> Optional[bool]:
    """Report whether this process already created the primary context on ``index``.

    ``torch.cuda.synchronize(index)`` implicitly creates a full CUDA/HIP primary
    context when the process never touched that device, which reserves hundreds
    of MiB on a GPU the run was told to stay off (issue #4653).

    The query reads the driver's primary-context state without creating it, so
    callers can decide whether a synchronize is a no-op. ``None`` means the
    running torch build does not expose the query; callers must then fall back to
    their historical behavior instead of assuming the context is absent. HIP
    exposes the same primary-context model, so the CUDA-namespaced query is also
    valid on ROCm builds.
    """

    query = getattr(torch._C, "_cuda_hasPrimaryContext", None)
    if query is None:
        return None
    try:
        return bool(query(int(index)))
    except Exception:  # pragma: no cover - driver/runtime edge cases
        return None


def accelerator_context_claimed(device_type: str, index: Optional[int] = None) -> bool:
    """Best-effort "this process already initialized that accelerator device".

    CUDA/HIP answers per index through :func:`cuda_primary_context_claimed`.
    XPU/NPU only expose a process-wide ``is_initialized()``; when that hook is
    missing, the historical "assume initialized" answer keeps the old behavior
    instead of silently dropping a needed barrier.
    """

    if device_type == "cuda":
        claimed = cuda_primary_context_claimed(0 if index is None else index)
        return True if claimed is None else claimed

    runtime = getattr(torch, device_type, None)
    is_initialized = getattr(runtime, "is_initialized", None)
    if not callable(is_initialized):
        return True
    try:
        return bool(is_initialized())
    except Exception:  # pragma: no cover - runtime edge cases
        return True


_ACCELERATOR_DEVICE_TYPES = frozenset({"cuda", "xpu", "npu", "mps"})
_DEFUSER_CONVERSION_LOCK = threading.RLock()


def model_accelerator_devices(model) -> List[str]:
    """Concrete accelerator ordinals that already hold model parameters or buffers.

    Placement is read from the tensors rather than inferred from a selector: a
    fused-expert model is converted while it still lives on CPU/meta, and a
    model converted after placement keeps its real ordinals.
    """

    devices: List[str] = []
    seen = set()
    for source in (getattr(model, "parameters", None), getattr(model, "buffers", None)):
        if not callable(source):
            continue
        for tensor in source():
            device = getattr(tensor, "device", None)
            if device is None or device.type not in _ACCELERATOR_DEVICE_TYPES:
                continue
            name = str(device)
            if name in seen:
                continue
            seen.add(name)
            devices.append(name)
    return devices


def _first_accelerator_tensor_device(value) -> Optional[torch.device]:
    """The first accelerator tensor device in ``value`` (one nesting level)."""

    if isinstance(value, torch.Tensor):
        device = value.device
        return device if device.type in _ACCELERATOR_DEVICE_TYPES else None
    if isinstance(value, dict):
        values = value.values()
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        return None
    for item in values:
        device = _first_accelerator_tensor_device(item)
        if device is not None:
            return device
    return None


def _module_accelerator_device(module) -> Optional[torch.device]:
    """The first accelerator device held by ``module``'s parameters or buffers."""

    for source in (getattr(module, "parameters", None), getattr(module, "buffers", None)):
        if not callable(source):
            continue
        for tensor in source():
            device = getattr(tensor, "device", None)
            if device is not None and device.type in _ACCELERATOR_DEVICE_TYPES:
                return device
    return None


def model_execution_device(model, *inputs) -> Optional[torch.device]:
    """The accelerator device a model call executes on, or ``None`` for CPU.

    A multi-device map is not pinned to one GPU: the first accelerator input
    wins, then the input embedding, then the first accelerator parameter or
    buffer. The inputs are the call's positional values (nested tuples, lists
    and dicts are inspected one level deep), so a device map's first stage is
    used for that stage's call.
    """

    for value in inputs:
        device = _first_accelerator_tensor_device(value)
        if device is not None:
            return device

    embedding = None
    try:
        get_input_embeddings = getattr(model, "get_input_embeddings", None)
        if callable(get_input_embeddings):
            embedding = get_input_embeddings()
    except Exception:  # pragma: no cover - exotic module trees
        embedding = None
    for candidate in (embedding, model):
        if candidate is None:
            continue
        device = _module_accelerator_device(candidate)
        if device is not None:
            return device
    return None


@contextmanager
def model_execution_device_scope(model, *inputs):
    """Enter the device a compiled model call executes on, then restore the caller's.

    Inductor initializes its lazy graph patterns from the traced frame's input
    device (``torch/_inductor/fx_passes/joint_graph.py::joint_graph_passes`` ->
    ``lazy_init`` -> ``fuse_attention._sfdp_init``). A quantized-kernel graph
    break can leave that frame without a device-bearing input, and
    ``_get_sfdp_patterns`` then falls back to ``"cuda"`` -- the *current*
    device. A compiled model loaded on ``cuda:1`` therefore claimed the primary
    context of ``cuda:0`` when a later shape recompiled (#5101).

    Entering the model's execution device for the compiled call makes those
    device-less defaults resolve to the GPU that holds the model. Only the
    current device changes; tensor placement is untouched. The previous current
    device is restored on exit, including when the call raises. A CPU model, or
    a model with no accelerator tensor, runs the call unchanged.
    """

    device = model_execution_device(model, *inputs)
    if device is None or device.type not in _ACCELERATOR_DEVICE_TYPES:
        yield
        return

    if device.type == "cuda":
        # Also the ROCm spelling: HIP exposes the CUDA device namespace.
        with torch.cuda.device(device):
            yield
        return

    runtime = getattr(torch, device.type, None)
    device_context = getattr(runtime, "device", None)
    if callable(device_context):
        with device_context(device):
            yield
        return
    yield


@contextmanager
def defuser_conversion_scope(model):
    """Bound Defuser's ``clear_memory()`` to the devices the model occupies.

    Defuser's fused-expert conversion calls ``clear_memory()`` without a device
    list, and that helper synchronizes the *current* CUDA device. Converting a
    CPU/meta model therefore initializes the primary context of cuda:0 --
    hundreds of MiB on a GPU the caller excluded with ``device=("cuda:1",)`` or
    with a CPU-only load (#5105). The pinned Defuser API accepts no device
    argument, so scope the module-level reference it calls for the duration of
    the conversion:

    * a model that already holds accelerator tensors synchronizes exactly those
      ordinals, preserving the barrier its own data needs;
    * a CPU/meta model passes a CPU-only list, so no accelerator is claimed.
      ``torch.cuda.empty_cache()`` still runs and does not create a context.

    Defuser also synchronizes XPU unconditionally; that remains outside this
    scope because the pinned API exposes no per-device hook for it.
    """

    if not torch.cuda.is_available():
        # Defuser's CUDA branch is skipped entirely without a CUDA runtime.
        yield
        return

    device_list = model_accelerator_devices(model) or ["cpu"]

    try:
        from defuser.modeling import moe_experts_interface as defuser_moe
    except Exception:  # pragma: no cover - optional dependency layout
        yield
        return

    original = getattr(defuser_moe, "clear_memory", None)
    if not callable(original):
        yield
        return

    def _scoped_clear_memory(*args, **kwargs):
        if len(args) < 2 and "device_list" not in kwargs:
            kwargs["device_list"] = device_list
        return original(*args, **kwargs)

    with _DEFUSER_CONVERSION_LOCK:
        defuser_moe.clear_memory = _scoped_clear_memory
        try:
            yield
        finally:
            defuser_moe.clear_memory = original


def torch_sync(device: torch.device = None):
    """Synchronize accelerator queues.

    When no device is provided we synchronize every detected accelerator index so
    replication work staged on multiple GPUs/NPUs completes before issuing more
    kernels.

    Only devices this process already initialized are synchronized. A
    ``synchronize`` on an untouched index creates a full primary context there, so
    a run that excluded a GPU (``compute_device_filter`` or
    ``device=("cuda:1",)``) used to claim the excluded GPU merely to wait for work
    it never submitted (issue #4653). Skipping such a device cannot lose a
    barrier: enqueueing work is what creates the context.
    """

    if device is None:
        synchronized_any = False

        if HAS_CUDA:
            dev_count = torch.cuda.device_count()
            if dev_count:
                synchronized_any = True
                for idx in range(dev_count):
                    if not accelerator_context_claimed("cuda", idx):
                        continue
                    torch.cuda.synchronize(idx)

        if HAS_XPU:
            dev_count = torch.xpu.device_count()
            if dev_count and accelerator_context_claimed("xpu"):
                synchronized_any = True
                for idx in range(dev_count):
                    torch.xpu.synchronize(idx)

        if HAS_MPS:
            synchronized_any = True
            torch.mps.synchronize()

        if HAS_NPU:
            dev_count = torch.npu.device_count()
            if dev_count and accelerator_context_claimed("npu"):
                synchronized_any = True
                for idx in range(dev_count):
                    torch.npu.synchronize(idx)

        # Only synchronize XLA if this process already initialized it. Probing
        # availability here would import the runtime and request addressable
        # devices, so an ordinary CUDA/CPU barrier would claim the Tenstorrent
        # chips (or fail on an unrelated XLA error) from a job that never
        # selected that backend.
        if xla_runtime_claimed():
            synchronized_any = True
            require_xla_runtime().sync(wait=True)

        if not synchronized_any:
            torch.cpu.synchronize()
        return

    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device=device)
    elif device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "npu":
        torch.npu.synchronize(device=device)
    elif device.type == "xla":
        require_xla_runtime().sync(wait=True)
    elif device.type == "cpu":
        torch.cpu.synchronize()

def _normalize_device(device: Union[torch.device, str, int, None]) -> Optional[torch.device]:
    if device is None:
        return None
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    if isinstance(device, int):
        if HAS_CUDA and torch.cuda.device_count() > device:
            return torch.device("cuda", device)
        if HAS_XPU and torch.xpu.device_count() > device:
            return torch.device("xpu", device)
        if HAS_NPU and torch.npu.device_count() > device:
            return torch.device("npu", device)
        if is_xla_available() and xla_device_count() > device:
            return torch.device("xla", device)
        raise ValueError(f"Unable to map integer `{device}` to a known accelerator device.")
    raise TypeError(f"Unsupported device specifier type: {type(device)}")


@lru_cache(maxsize=12)
def resolve_empty_cache_callable(device_type: str) -> Optional[Callable[[], None]]:
    try:
        if device_type == "cuda" and HAS_CUDA:
            candidate = torch.cuda.empty_cache
        elif device_type == "xpu" and HAS_XPU:
            candidate = torch.xpu.empty_cache
        elif device_type == "mps" and HAS_MPS:
            candidate = torch.mps.empty_cache
        elif device_type == "npu" and HAS_NPU:
            candidate = torch.npu.empty_cache
        elif device_type == "xla" and is_xla_available():
            candidate = getattr(require_xla_runtime(), "empty_cache", None)
        else:
            candidate = None
    except Exception:
        candidate = None

    if callable(candidate):
        return candidate

    return None


def empty_cache_for_device(device: torch.device) -> bool:
    if device.type == "cuda":
        fn = resolve_empty_cache_callable("cuda")
        if fn is None:
            return False
        target = device if device.index is not None else torch.device("cuda")
        with torch.cuda.device(target):
            fn()
        return True

    if device.type == "xpu":
        fn = resolve_empty_cache_callable("xpu")
        if fn is None:
            return False
        target = device if device.index is not None else torch.device("xpu")
        with torch.xpu.device(target):
            fn()
        return True

    if device.type == "mps":
        fn = resolve_empty_cache_callable("mps")
        if fn is None:
            return False
        fn()
        if HAS_MLX:
            mlx.core.clear_cache()
        return True

    if device.type == "npu":
        fn = resolve_empty_cache_callable("npu")
        if fn is None:
            return False
        target = device if device.index is not None else torch.device("npu")
        with torch.npu.device(target):
            fn()
        return True

    if device.type == "cpu" and HAS_MLX:
        mlx.core.clear_cache()
        return True
    return False


@lru_cache(maxsize=12)
def resolve_memory_stats_callable(device_type: str) -> Optional[Callable[[torch.device], dict]]:
    """Return the allocator-stats callable for `device_type`, or None when absent.

    Mirrors ``resolve_empty_cache_callable``: allocator diagnostics should ask the
    active accelerator, because ``torch.cuda`` only reports CUDA allocations and
    stays empty on NPU/XPU/CPU-only builds.
    """

    try:
        if device_type == "cuda" and HAS_CUDA:
            candidate = torch.cuda.memory_stats
        elif device_type == "xpu" and HAS_XPU:
            candidate = torch.xpu.memory_stats
        elif device_type == "npu" and HAS_NPU:
            candidate = torch.npu.memory_stats
        else:
            candidate = None
    except Exception:
        candidate = None

    if callable(candidate):
        return candidate

    return None


def memory_stats_for_device(device: torch.device) -> dict:
    """Return allocator statistics for `device`, or an empty mapping when unsupported.

    Best-effort telemetry uses this instead of ``torch.cuda.memory_stats``. The
    CUDA call reports nothing for a non-CUDA device, so accelerator memory log
    rows silently read zero on NPU/XPU/CPU-only builds even while the device is
    holding allocations.
    """

    fn = resolve_memory_stats_callable(device.type)
    if fn is None:
        return {}

    try:
        return fn(device)
    except Exception:
        return {}


def torch_empty_cache_any(
    device: Union[torch.device, str, int, None] = None,
    gc: bool = True,
    sync: bool = False,
) -> bool:
    normalized = _normalize_device(device)

    if gc:
        timed_gc_collect()

    if sync:
        torch_sync(device=normalized)

    success = False

    if normalized is None:
        if HAS_CUDA:
            fn = resolve_empty_cache_callable("cuda")
            if fn is not None:
                fn()
                success = True
        if HAS_XPU:
            fn = resolve_empty_cache_callable("xpu")
            if fn is not None:
                fn()
                success = True
        if HAS_MPS:
            fn = resolve_empty_cache_callable("mps")
            if fn is not None:
                fn()
                success = True
        if HAS_NPU:
            fn = resolve_empty_cache_callable("npu")
            if fn is not None:
                fn()
                success = True
        if HAS_MLX:
            mlx.core.clear_cache()
            success = True
        return success

    return empty_cache_for_device(normalized)


def torch_empty_cache(device: torch.device = None, gc: bool = True, sync: bool = False) -> bool:
    return torch_empty_cache_any(device=device, gc=gc, sync=sync)


def release_pinned_host_cache() -> bool:
    """Return PyTorch's cached pinned-host blocks to the operating system.

    ``torch.empty(..., pin_memory=True)`` (used by layer offload) and
    accelerate's checkpoint staging allocate non-swappable host memory. Deleting
    the tensor only returns the block to PyTorch's process-wide pinned cache, so
    ``RssShmem`` stays resident for the life of the process. Call this after the
    staging copies are dead; blocks still owned by live tensors are untouched.

    Returns ``False`` when the Torch build exposes no host cache to empty.
    """

    empty = getattr(torch._C, "_host_emptyCache", None)
    if empty is None:
        return False
    empty()
    return True


def resolve_torch_device(device: Union[torch.device, str, int, None] = None):
    """Resolve ``auto`` or a concrete device without assuming CUDA exists."""
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        return auto_select_torch_device()
    return _normalize_device(device)


def auto_select_torch_device(index: int = 0):
    """Return the fastest available Torch accelerator, with CUDA precedence.

    The ordering preserves existing multi-GPU defaults while allowing the same
    callers to run on Apple MPS and CPU-only machines.
    """
    assert index >= 0, f"device index should be a positive integer: actual = `{index}`"

    if HAS_CUDA:
        # defensive check
        if index > 0 and torch.cuda.device_count() <= index :
            index = 0
        device = torch.device(f"cuda:{index}")
    elif HAS_XPU:
        # defensive check
        if index > 0 and torch.xpu.device_count() <= index:
            index = 0
        device = torch.device(f"xpu:{index}")
    elif HAS_NPU:
        # defensive check
        if index > 0 and torch.npu.device_count() <= index:
            index = 0
        device = torch.device(f"npu:{index}")
    elif is_xla_available():
        count = xla_device_count()
        if index >= count:
            index = 0
        device = torch.device(f"xla:{index}")
    elif HAS_MPS:
        device = torch.device("mps") # mps has no index
    else:
        device = CPU # cpu has no index

    return device


def auto_select_torch_device_unprobed(index: int = 0):
    """Choose an indexable device without initializing an optional runtime.

    ``torch_xla`` is deliberately absent: asking whether it is usable calls
    ``addressable_device_count()``, which initializes the PJRT plugin and claims
    every Tenstorrent chip for the rest of the process. Package presence is not
    evidence that a device exists either -- an installed XLA runtime with zero
    devices would otherwise turn a CPU job into ``xla:0``. Callers that want XLA
    resolve it explicitly through :func:`is_xla_available` and
    :func:`xla_device_count`, which are allowed to pay that cost.
    """
    if HAS_CUDA:
        # Clamp against the visible count: a caller asking for index 1 on a
        # single-GPU host would otherwise get ``cuda:1``, which fails later with
        # "invalid device ordinal" instead of falling back to the device that
        # exists. CUDA is safe to probe here; XLA is not.
        count = torch.cuda.device_count()
        if count > 0:
            return torch.device(f"cuda:{index if index < count else 0}")
    if HAS_XPU:
        count = torch.xpu.device_count()
        if count > 0:
            return torch.device(f"xpu:{index if index < count else 0}")
    if HAS_NPU:
        count = torch.npu.device_count()
        if count > 0:
            return torch.device(f"npu:{index if index < count else 0}")
    if HAS_MPS:
        return torch.device("mps")
    return CPU

# some device types can have multiple gpus cuda/rocm + xpu
def torch_devices() -> List[torch.device]:
    if HAS_CUDA:
        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    elif HAS_XPU:
        return [torch.device(f"xpu:{i}") for i in range(torch.xpu.device_count())]
    elif HAS_NPU:
        return [torch.device(f"npu:{i}") for i in range(torch.npu.device_count())]
    elif is_xla_available():
        count = xla_device_count()
        return [torch.device(f"xla:{i}") for i in range(count)]
    elif HAS_MPS:
        return [torch.device("mps")]
    else:
        return [CPU]


def npu_devices_by_pci_bus_order() -> List[torch.device]:
    """Return visible NPU devices in torch runtime order.

    Ascend exposes process-level NPU visibility through ASCEND_RT_VISIBLE_DEVICES.
    torch-npu remaps that visible set to logical indices, so callers should set
    the env var before process start and then use the resulting torch order.
    """

    if not HAS_NPU:
        return []

    try:
        count = int(torch.npu.device_count())
    except Exception:
        return []
    if count <= 0:
        return []

    devices: List[torch.device] = []
    for logical_index in range(count):
        try:
            devices.append(torch.device("npu", logical_index))
        except (RuntimeError, ValueError):
            return []
    return devices


def last_npu_device_by_pci_bus_order() -> Optional[torch.device]:
    """Return the last visible NPU in torch runtime order, or None when unavailable."""

    devices = npu_devices_by_pci_bus_order()
    if not devices:
        return None
    return devices[-1]

def all_torch_devices() -> List[torch.device]:
    """Discover the active accelerator set only when a caller requests it.

    ``torch_devices()`` is an explicit runtime probe. In particular, instantiating
    ``torch_xla`` opens every Tenstorrent device owned by the PJRT plugin. A
    plain ``import gptqmodel`` therefore must not cache the result in
    ``ALL_DEVICES``.
    """
    return torch_devices()


class _LazyAcceleratorStreams:
    """Create accelerator streams only when a caller actually needs them."""

    def __init__(self, devices: List[torch.device]):
        self._devices = devices
        self._streams: List[Optional[object]] = [None] * len(devices)

    def __len__(self):
        return max(1, len(self._devices))

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int):
        if not self._devices:
            if index == 0:
                return contextlib.nullcontext()
            raise IndexError(index)

        stream = self._streams[index]
        if stream is not None:
            return stream

        device = self._devices[index]
        if device.type == "cuda":
            stream = torch.cuda.Stream(device=device)
        elif device.type == "xpu":
            stream = torch.xpu.Stream(device=device)
        elif device.type == "npu":
            stream = torch.npu.Stream(device=device)
        else:
            stream = contextlib.nullcontext()
        self._streams[index] = stream
        return stream


class _LazyAcceleratorStreamRef:
    """Reference one lazy stream without materializing it at import time."""

    def __init__(self, streams: _LazyAcceleratorStreams, index: int):
        self._streams = streams
        self._index = index

    def get(self):
        return self._streams[self._index]


ALL_STREAMS = _LazyAcceleratorStreams([auto_select_torch_device_unprobed(index=0)])

DEVICE_0 = auto_select_torch_device_unprobed(index=0)
# device_1 may be same as device_0 if there is only 1 visible/active device
DEVICE_1 = auto_select_torch_device_unprobed(index=1)

DEVICE_0_STREAM = _LazyAcceleratorStreamRef(ALL_STREAMS, 0)

NEXT_DEVICE_INDEX = 0

# def device_next_reset():
#     global NEXT_DEVICE_INDEX
#     NEXT_DEVICE_INDEX = 0
#
# def device_next(balance_strategy: BalanceStrategy = DEFAULT_BALANCE_STRATEGY) -> torch.device:
#     global NEXT_DEVICE_INDEX
#
#     devices = all_torch_devices()
#     if len(devices) <= 1:
#         return devices[0]
#
#     device = devices[NEXT_DEVICE_INDEX]
#     if NEXT_DEVICE_INDEX < len(devices) - 1:
#         NEXT_DEVICE_INDEX += 1
#     else:
#         if balance_strategy == BalanceStrategy.MEMORY:
#             NEXT_DEVICE_INDEX = 1
#         else:
#             NEXT_DEVICE_INDEX = 0
#
#     return device

def torch_streamCtx(stream) -> StreamContext:
    if isinstance(stream, _LazyAcceleratorStreamRef):
        stream = stream.get()
    if HAS_CUDA:
        return torch.cuda.stream(stream)
    if HAS_XPU:
        return torch.xpu.stream(stream)
    if HAS_NPU:
        return torch.npu.stream(stream)
    return contextlib.nullcontext()


@contextmanager
def tf32_high_precision_guard():
    """Run with TF32 disabled, reference-counted across concurrent callers.

    ``torch.backends.fp32_precision`` (or ``allow_tf32`` on older builds) is
    process-global. The quantization loop enters this guard per model, so two
    concurrent ``quantize()`` calls used to overwrite each other: the first
    run to finish restored the user's setting while the other run still needed
    IEEE math. The count under a lock keeps the override until the last caller
    exits and restores exactly the setting captured by the first caller
    (issue #4094).
    """

    global _TF32_GUARD_DEPTH, _TF32_GUARD_PREVIOUS

    if not HAS_CUDA:
        yield
        return

    with _TF32_GUARD_LOCK:
        if _TF32_GUARD_DEPTH == 0:
            _TF32_GUARD_PREVIOUS = _snapshot_tf32_state()
            _set_tf32_state(False)
        _TF32_GUARD_DEPTH += 1
    try:
        yield
    finally:
        with _TF32_GUARD_LOCK:
            _TF32_GUARD_DEPTH -= 1
            if _TF32_GUARD_DEPTH == 0:
                previous_state = _TF32_GUARD_PREVIOUS
                _TF32_GUARD_PREVIOUS = None
                if previous_state is not None:
                    _restore_tf32_state(previous_state)


@contextmanager
def tf32_disable_guard():
    if not HAS_CUDA:
        yield
        return

    previous_state = _snapshot_tf32_state()
    _set_tf32_state(False)
    try:
        yield
    finally:
        _restore_tf32_state(previous_state)


@contextmanager
def tf32_enable_guard():
    if not HAS_CUDA:
        yield
        return

    previous_state = _snapshot_tf32_state()
    _set_tf32_state(True)
    try:
        yield
    finally:
        _restore_tf32_state(previous_state)
