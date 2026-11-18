# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Shared run-exit recovery steps for both quantization loopers (#5060).

A ``quantize()`` run can end without quantizing the model: a
``subset_callback`` returns ``StopMainLoop``, a callback raises, Ctrl-C
arrives, or a stage fails. The caller still owns the model, so the run has to
leave it usable. ``docs/cancelled-run-contract.md`` documents that contract.

Both loopers used to carry their own copy of the exit sequence, so every new
obligation had to be remembered in several places, and one of them drifted:
``WeightOnlyLooper`` submitted work to ``DEVICE_THREAD_POOL`` but never fenced
it, while ``ModuleLooper`` had fenced it since #4595.

This module owns the steps that are identical for both loopers. Each looper
keeps the parts that depend on what its own stage acquired (which processors
to release, which base modules to restore) and passes them in, so the shared
part can be extended once instead of twice.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Tuple

from ..utils.logger import setup_logger


log = setup_logger()

#: One ``(label, wait)`` pair per unit of already-submitted work.
FenceWait = Tuple[str, Callable[[], None]]


def fence_submitted_work(waits: Iterable[FenceWait]) -> None:
    """Run every wait, independently, and never raise.

    ``loop()`` drains submitted work only on the success path, so the failure
    path has to fence whatever the run already submitted *before* it touches
    that work's inputs or releases its storage. A finalizer still running after
    teardown re-creates the run directory that ``release_private_offload_root``
    just removed (one leaked bundle set per failed run) and keeps
    packing/replacing modules after ``quantize()`` has raised (#4595).

    Each wait is independent: one that fails must not skip the remaining waits,
    and nothing here may mask the exception that failed the run. The waits are
    taken as ``(label, callable)`` pairs rather than resolved here so each
    looper keeps passing its own module-level pool references, which is also
    what lets tests substitute them.
    """

    for label, wait in waits:
        try:
            wait()
        except Exception:
            log.warning(
                "failed to drain the %s while failing the quantization run",
                label,
                exc_info=True,
            )


def run_exit_waits(
    device_pool,
    *,
    wait_dangling_threads: Optional[Callable[[], None]] = None,
    stream_pool=None,
) -> list[FenceWait]:
    """Build the fence waits a failed run has to drain, in execution order.

    ``device_pool`` and ``stream_pool`` are passed by the caller so the fence
    uses the same module-level objects the run submitted to. The stream pool is
    imported lazily because importing it eagerly would pull the CUDA stream
    machinery into every ``import gptqmodel``.
    """

    if stream_pool is None:
        from ..utils.stream import STREAM_DEVICE_POOL as stream_pool

    waits: list[FenceWait] = [("device thread pool", device_pool.wait)]
    if wait_dangling_threads is not None:
        # Only the calibrated looper keeps finalizer watcher threads; the
        # weight-only stage runs its finalizers inline.
        waits.append(("finalizer watcher threads", wait_dangling_threads))
    waits.append(("stream device pool", stream_pool.wait))
    return waits


def fence_run_work(
    device_pool,
    *,
    wait_dangling_threads: Optional[Callable[[], None]] = None,
    stream_pool=None,
) -> None:
    """Fence one failed run's submitted work. See :func:`fence_submitted_work`."""

    fence_submitted_work(
        run_exit_waits(
            device_pool,
            wait_dangling_threads=wait_dangling_threads,
            stream_pool=stream_pool,
        )
    )


__all__ = ["FenceWait", "fence_run_work", "fence_submitted_work", "run_exit_waits"]
