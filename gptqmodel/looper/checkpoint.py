# SPDX-License-Identifier: Apache-2.0
"""Checkpoint policy composed with looper boundaries and a state adapter."""

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from .checkpoint_store import CheckpointConfig, CheckpointError, CheckpointStore
from .continuation import ContinuationCodec
from .extension import LoopBoundary, LoopPlan


class CheckpointAdapter(Protocol):
    """Adapters own model/processor schema, never scheduling or publication.

    capture() returns full continuation and a complete artifact mapping; blobs
    may be reused by digest across generations. restore() must validate its
    schema before mutating live state. Unsupported processors fail in prepare,
    before the looper runs, not halfway through a checkpoint.
    """

    def capture(self) -> tuple[object, dict[str, bytes | Path]]: ...

    def restore(self, continuation: object, artifacts: dict[str, Path]) -> None: ...


class CheckpointStopped(CheckpointError):
    """A requested stop was committed at a safe boundary; restart can resume."""


class CheckpointExtension:
    """The store lease must enclose prepare, execution, and finalization.

    No signal handlers are installed globally. A caller's SIGINT/SIGTERM handler
    may call request_stop(); it does no I/O or synchronization. Hard kills need
    no handler and recover using the store's published generations.
    """

    def __init__(self, config: CheckpointConfig, adapter: CheckpointAdapter):
        self.store = CheckpointStore(config)
        self.adapter = adapter
        self.plan = None
        self.next_step = 0
        self._committed_cursor = 0
        self._stop_requested = False

    def __enter__(self):
        self.store.__enter__()
        return self

    def __exit__(self, *args):
        return self.store.__exit__(*args)

    def prepare(self, plan: LoopPlan, identity: dict) -> int:
        if self.plan is not None:
            raise CheckpointError("checkpoint extension cannot be prepared twice")
        self.identity = {
            **identity,
            "execution_plan": [asdict(step) for step in plan.steps],
        }
        manifest = self.store.load(self.identity)
        if manifest is not None:
            if manifest["cursor"] > len(plan.steps):
                raise CheckpointError("checkpoint cursor is outside the execution plan")
            continuation = ContinuationCodec.loads(
                self.store.get(manifest["continuation"])
            )
            artifacts = {
                name: self.store.object_path(ref)
                for name, ref in manifest["artifacts"].items()
            }
            self.adapter.restore(continuation, artifacts)
            self.next_step = self._committed_cursor = manifest["cursor"]
        self.plan = plan
        return self.next_step

    def on_start(self, context):
        return self.prepare(context.plan, self.adapter.bind(context))

    def request_stop(self):
        self._stop_requested = True

    def on_boundary(self, boundary: LoopBoundary):
        if self.plan is None:
            raise CheckpointError("prepare the checkpoint extension before execution")
        cursor = self.plan.after(boundary.step)
        if cursor != self.next_step + 1:
            raise CheckpointError("boundary does not follow the execution cursor")
        due = (
            cursor - self._committed_cursor >= self.store.config.every_layers
            or cursor == len(self.plan.steps)
            or self._stop_requested
        )
        if due:
            boundary.quiesce()
            continuation, artifacts = self.adapter.capture()
            continuation_ref = self.store.put(ContinuationCodec.dumps(continuation))
            artifact_refs = {
                name: self.store.put_file(data)
                if isinstance(data, Path)
                else self.store.put(data)
                for name, data in artifacts.items()
            }
            self.store.commit(
                identity=self.identity,
                cursor=cursor,
                continuation=continuation_ref,
                artifacts=artifact_refs,
            )
            self._committed_cursor = cursor
            callback = getattr(self.adapter, "committed", None)
            if callback is not None:
                callback(
                    {
                        name: self.store.root / "objects" / ref
                        for name, ref in artifact_refs.items()
                    }
                )
            try:
                self.store.collect()
            except (OSError, CheckpointError):
                # Publication already succeeded; retry cleanup later.
                logging.getLogger(__name__).warning(
                    "Checkpoint committed but object cleanup failed", exc_info=True
                )
        self.next_step = cursor
        if self._stop_requested:
            raise CheckpointStopped(f"checkpoint committed; resume from step {cursor}")
