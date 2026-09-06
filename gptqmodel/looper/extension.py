# SPDX-License-Identifier: Apache-2.0
"""Synchronous execution-boundary extensions, independent of persistence.

Boundary objects are borrowed for the duration of on_boundary only. An
extension must quiesce before treating results as complete. Retaining a
boundary or using it from a worker thread is an error. No tensor serialization
or checkpoint format belongs in this module.
"""

from concurrent.futures import Future, wait
from dataclasses import dataclass
from threading import get_ident
from typing import Protocol


@dataclass(frozen=True)
class LoopStep:
    """An execution identity, separate from transformer-layer indices."""

    kind: str
    index: int
    name: str


class LoopBoundary:
    def __init__(self, step: LoopStep, futures: tuple[Future, ...]):
        self.step = step
        self._futures = futures
        self._owner = get_ident()
        self._active = True

    def quiesce(self) -> tuple[object, ...]:
        """Drain all preceding finalizers, then return results or raise.

        Waiting for every future before raising prevents one failed worker
        from leaving other workers mutating state during recovery.
        """
        if get_ident() != self._owner:
            raise RuntimeError("Loop boundaries belong to the orchestration thread")
        if not self._active:
            raise RuntimeError("Loop boundary is no longer active")
        if self._futures:
            wait(self._futures)
        return tuple(future.result() for future in self._futures)


class LoopExtension(Protocol):
    def on_boundary(self, boundary: LoopBoundary) -> None:
        """Observe a step before the next step can mutate continuation state."""
        ...


class LoopExtensions:
    """Own extension dispatch and outstanding work on the coordinator thread."""

    def __init__(self, extensions=()):
        self._extensions = tuple(extensions)
        self._pending: list[Future] = []

    def __bool__(self):
        return bool(self._extensions)

    def publish(self, step: LoopStep, futures=()) -> None:
        if not self:
            return
        self._pending.extend(futures)
        boundary = LoopBoundary(step, tuple(self._pending))
        try:
            for extension in self._extensions:
                extension.on_boundary(boundary)
        finally:
            boundary._active = False
        # Retain failures/cancellations until quiesce surfaces them. Successful
        # completed futures no longer need to be held across later boundaries.
        self._pending = [
            future for future in self._pending
            if not future.done() or future.cancelled() or future.exception() is not None
        ]
