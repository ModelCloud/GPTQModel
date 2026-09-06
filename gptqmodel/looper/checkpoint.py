# SPDX-License-Identifier: Apache-2.0
"""Checkpoint policy composed with looper boundaries and a state adapter."""

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from ..utils.device_telemetry import emit_device_telemetry
from .checkpoint_devices import checkpoint_identity_without_physical_gpu_ids
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
        self.store = CheckpointStore(
            config,
            identity_projection=checkpoint_identity_without_physical_gpu_ids
            if config.skip_strict_gpu_check
            else None,
        )
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
        emit_device_telemetry(
            "checkpoint_prepare",
            root=str(self.store.root),
            device_topology=identity.get("device_topology"),
        )
        try:
            manifest = self.store.load(self.identity)
        except CheckpointError as exc:
            emit_device_telemetry(
                "checkpoint_prepare_rejected",
                root=str(self.store.root),
                reason=str(exc),
            )
            raise
        if manifest is not None:
            if manifest["cursor"] > len(plan.steps):
                raise CheckpointError("checkpoint cursor is outside the execution plan")
            physical_identity_matched = manifest["identity"].get(
                "device_topology"
            ) == identity.get("device_topology")
            if self.store.config.skip_strict_gpu_check:
                logging.getLogger(__name__).warning(
                    "Checkpoint strict GPU UUID/serial checks explicitly disabled; "
                    "GPU count, indices, pools, model and capability checks remain enforced. "
                    "Bit-exact quantization across different hardware is not guaranteed."
                )
            emit_device_telemetry(
                "checkpoint_topology_validated",
                cursor=manifest["cursor"],
                expected_topology=manifest["identity"].get("device_topology"),
                actual_topology=identity.get("device_topology"),
                matched=True,
                strict_gpu_check=not self.store.config.skip_strict_gpu_check,
                physical_identity_matched=physical_identity_matched,
            )
            emit_device_telemetry(
                "checkpoint_restore_begin",
                cursor=manifest["cursor"],
                continuation=manifest["continuation"],
            )
            try:
                continuation = ContinuationCodec.loads(
                    self.store.get(manifest["continuation"])
                )
                artifacts = {
                    name: self.store.object_path(ref)
                    for name, ref in manifest["artifacts"].items()
                }
                self.adapter.restore(continuation, artifacts)
            except Exception as exc:
                emit_device_telemetry(
                    "checkpoint_restore_failed",
                    cursor=manifest["cursor"],
                    error_type=type(exc).__name__,
                )
                raise
            self.next_step = self._committed_cursor = manifest["cursor"]
            emit_device_telemetry("checkpoint_restore_complete", cursor=self.next_step)
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
            emit_device_telemetry("checkpoint_capture_begin", cursor=cursor)
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
            emit_device_telemetry(
                "checkpoint_committed",
                cursor=cursor,
                continuation=continuation_ref,
                artifact_count=len(artifact_refs),
            )
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
            emit_device_telemetry("checkpoint_stopped", cursor=cursor)
            raise CheckpointStopped(f"checkpoint committed; resume from step {cursor}")
