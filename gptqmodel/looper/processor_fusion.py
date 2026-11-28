# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Processor fusion: run a downstream processor inside an upstream one.

The downstream processor keeps its own class and lifecycle. Only the *place*
it runs moves: instead of a separate pass that reads staged tensors from
``NamedModule.state``, it executes inside the producer's frame while the
intermediates are still live.

Nothing in this module imports torch. Artifact values are opaque to the scope;
it only tracks ownership and bytes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from threading import get_ident
from typing import Any, Callable, Dict, Optional, Sequence, Tuple


class ProcessorStages(str, Enum):
    """Named points inside a producer's own execution."""

    PREPROCESS = "preprocess"
    POST_QUANTIZE = "post_quantize"  # wq solved, module.weight not installed yet
    POST_INSTALL = "post_install"    # module.weight is final
    PRE_RELEASE = "pre_release"      # last chance before workspaces are freed


@dataclass(frozen=True)
class InlineConfig:
    """How a processor can be fused into an upstream producer."""

    producer: frozenset
    stage: ProcessorStages
    forward_hooks: bool = False
    consumes: Tuple[str, ...] = ()
    produces: Tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.producer, frozenset) or not self.producer:
            raise ValueError("inline producer must be a non-empty frozenset")
        if not isinstance(self.stage, ProcessorStages):
            raise TypeError("inline stage must be a ProcessorStages member")
        for field in (self.consumes, self.produces):
            if type(field) is not tuple or any(type(name) is not str for name in field):
                raise TypeError("inline artifact names must be a tuple of strings")


class ProcessorFusionError(RuntimeError):
    """An inline consumer failed inside its producer's execution frame."""

    def __init__(self, producer, consumer, module, cause: BaseException):
        producer_name = _processor_name(producer)
        consumer_name = _processor_name(consumer)
        module_name = _module_name(module)
        super().__init__(
            f"inline processor `{consumer_name}` failed inside "
            f"`{producer_name}` for module `{module_name}`: {cause}"
        )
        self.producer_name = producer_name
        self.consumer_name = consumer_name
        self.module_name = module_name
        self.cause = cause


def _processor_name(processor) -> str:
    name = getattr(processor, "name", None)
    if callable(name):
        try:
            return str(name())
        except Exception:  # noqa: BLE001 - diagnostics must never mask the failure
            pass
    return type(processor).__name__.lower()


def _module_name(module) -> str:
    return str(
        getattr(module, "full_name", None)
        or getattr(module, "name", None)
        or repr(module)
    )


def _nbytes(value: Any) -> int:
    nbytes = getattr(value, "nbytes", None)
    if callable(nbytes):
        return int(nbytes())
    if isinstance(nbytes, int):
        return nbytes
    numel = getattr(value, "numel", None)
    element_size = getattr(value, "element_size", None)
    if callable(numel) and callable(element_size):
        try:
            return int(numel()) * int(element_size())
        except Exception:  # noqa: BLE001
            return 0
    return 0


class ArtifactScope:
    """Borrowed view of a producer's live intermediates for one module.

    Values are recipes until first access. Everything materialized is owned by
    the scope and dropped on exit. Consumers must not retain borrowed values;
    ``detach`` is the counted escape hatch.
    """

    def __init__(self, producer, module, stage: ProcessorStages, *, consumers=()):
        if not isinstance(stage, ProcessorStages):
            raise TypeError("artifact scope requires a ProcessorStages member")
        self.producer = producer
        self.module = module
        self.stage = stage
        self.consumers = tuple(consumers)
        self._recipes: Dict[str, Callable[[], Any]] = {}
        self._cache: Dict[str, Any] = {}
        self._owner = get_ident()
        self._closed = False
        self.materialized = 0
        self.detached = 0
        self.peak_bytes = 0

    # -- producer side ---------------------------------------------------
    def publish(self, name: str, recipe: Callable[[], Any]) -> None:
        """Register a lazy artifact. ``recipe`` runs at most once per scope."""

        self._check()
        if not callable(recipe):
            raise TypeError("artifact recipe must be callable")
        self._recipes[name] = recipe

    def replace(self, name: str, value: Any) -> None:
        """Hand a transformed value forward to the producer's install step."""

        self._check()
        self._recipes.pop(name, None)
        self._cache[name] = value
        self._sample()

    # -- consumer side ---------------------------------------------------
    def get(self, name: str) -> Any:
        """Materialize an artifact once and return the borrowed value."""

        self._check()
        if name in self._cache:
            return self._cache[name]
        try:
            recipe = self._recipes[name]
        except KeyError:
            raise KeyError(
                f"artifact `{name}` is not published by {_processor_name(self.producer)}"
            ) from None
        value = recipe()
        self._cache[name] = value
        self.materialized += 1
        self._sample()
        return value

    def detach(self, name: str) -> Any:
        """Clone an artifact for retention. Counted so leaks stay visible."""

        value = self.get(name)
        detached = getattr(value, "detach", None)
        if callable(detached):
            value = detached()
        value = value.clone() if hasattr(value, "clone") else copy.deepcopy(value)
        self.detached += 1
        self._emit("fusion_artifact_detached", artifact=name)
        return value

    def try_get(self, name: str) -> Any:
        """Return an optional artifact, or ``None`` when it was not published."""

        self._check()
        if name in self._cache:
            return self._cache[name]
        if name not in self._recipes:
            return None
        return self.get(name)

    # -- lifetime --------------------------------------------------------
    def _sample(self) -> None:
        total = sum(_nbytes(value) for value in self._cache.values())
        self.peak_bytes = max(self.peak_bytes, total)

    def _emit(self, event: str, **fields) -> None:
        try:
            from ..utils.device_telemetry import emit_device_telemetry

            emit_device_telemetry(event, **fields)
        except Exception:  # noqa: BLE001 - telemetry is never load-bearing
            pass

    def _check(self) -> None:
        if self._closed:
            raise RuntimeError("artifact scope is closed; borrowed values are invalid")
        if get_ident() != self._owner:
            raise RuntimeError("artifact scopes belong to the producing thread")

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "producer": _processor_name(self.producer),
            "stage": self.stage.value,
            "module": _module_name(self.module),
            "materialized": self.materialized,
            "detached": self.detached,
            "peak_bytes": self.peak_bytes,
        }

    def __enter__(self) -> "ArtifactScope":
        self._check()
        return self

    def __exit__(self, *exc) -> bool:
        self._closed = True
        self._cache.clear()
        self._recipes.clear()
        return False


@dataclass(frozen=True)
class ExecutionStage:
    """One execution step: a producer plus the consumers fused into it."""

    processor: Any
    inline: Tuple[Any, ...] = ()

    def __post_init__(self):
        if type(self.inline) is not tuple:
            raise TypeError("stage inline consumers must be a tuple")


@dataclass(frozen=True)
class ProcessorFusionPlan:
    """Ordered execution stages; fused consumers do not get their own stage."""

    stages: Tuple[ExecutionStage, ...]
    fused: Tuple[Any, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.fused)

    def stage_for(self, processor) -> Optional[ExecutionStage]:
        for stage in self.stages:
            if stage.processor is processor:
                return stage
        return None

    @property
    def processor_count(self) -> int:
        return len(self.stages) + len(self.fused)


def _can_fuse(producer, consumer) -> bool:
    checker = getattr(consumer, "can_fuse_with", None)
    if callable(checker):
        try:
            return bool(checker(producer))
        except Exception:  # noqa: BLE001 - an incompatible consumer falls back
            return False
    return True


def _with_producer_batch_index(producer, consumer, hook):
    """Run a consumer hook with the producer's calibration batch index."""

    def wrapper(module, inputs, output):
        setter = getattr(consumer, "_set_current_batch_index", None)
        getter = getattr(producer, "current_batch_index", None)
        if callable(setter) and callable(getter):
            setter(getter())
        return hook(module, inputs, output)

    return wrapper


def compose_forward_hooks(producer, producer_hook, module, name, consumers=()):
    """Chain inline consumers' hooks into the producer's forward hook.

    ``HookedLinear.forward()`` raises the early-stop exception from inside
    ``forward()``, before ordinary post-forward hooks run. A consumer
    registered as its own hook would therefore miss the last module of every
    subset and find no statistics when it runs inline. Composing into the
    producer's hook keeps the capture ahead of that stop.
    """

    hooks = []
    if producer_hook is not None:
        hooks.append(producer_hook)
    for consumer in consumers:
        config = getattr(consumer, "inline", None)
        if consumer is producer or config is None or not config.forward_hooks:
            continue
        is_skipped = getattr(consumer, "is_skipped", None)
        if callable(is_skipped) and is_skipped(module):
            continue
        hook = consumer.pre_process_fwd_hook(name)
        if hook is None:
            continue
        hooks.append(_with_producer_batch_index(producer, consumer, hook))
    if not hooks:
        return None
    if len(hooks) == 1:
        return hooks[0]

    def combined(module, inputs, output):
        for hook in hooks:
            hook(module, inputs, output)

    return combined


def _find_producer(processors: Sequence, consumer):
    config = getattr(consumer, "inline", None)
    if config is None:
        return None
    consumed = set(config.consumes)
    for candidate in processors:
        if candidate is consumer:
            break
        if _processor_name(candidate) not in config.producer:
            continue
        emits = set(getattr(candidate, "emits", ()) or ())
        stages = tuple(getattr(candidate, "stages", ()) or ())
        if config.stage not in stages:
            continue
        if not consumed.issubset(emits):
            continue
        if not _can_fuse(candidate, consumer):
            continue
        return candidate
    return None


def plan_processor_fusion(processors: Sequence, *, enabled: bool = True) -> ProcessorFusionPlan:
    """Attach inline consumers to their producers and return the stage order.

    Fusion is acyclic by construction: a consumer only attaches to an earlier
    processor. Consumers that cannot be attached keep their own stage and use
    the materialized ``NamedModule.state`` handoff.
    """

    processors = list(processors)
    for processor in processors:
        processor.inline_processors = ()

    if not enabled:
        return ProcessorFusionPlan(
            stages=tuple(ExecutionStage(processor=processor) for processor in processors)
        )

    attachments = {id(processor): [] for processor in processors}
    fused = []
    for consumer in processors:
        producer = _find_producer(processors, consumer)
        if producer is None:
            continue
        attachments[id(producer)].append(consumer)
        fused.append(consumer)

    fused_ids = {id(consumer) for consumer in fused}
    stages = tuple(
        ExecutionStage(processor=processor, inline=tuple(attachments[id(processor)]))
        for processor in processors
        if id(processor) not in fused_ids
    )
    for processor in processors:
        inline = tuple(attachments[id(processor)])
        if inline:
            processor.inline_processors = inline

    return ProcessorFusionPlan(stages=stages, fused=tuple(fused))
