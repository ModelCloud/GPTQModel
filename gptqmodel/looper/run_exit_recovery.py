# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""One owner for the model-state obligations of a quantization run's exit.

Both loopers must leave the in-session model in the same state whether the run
completed, was stopped by ``StopMainLoop``, raised, or was interrupted. The
obligations are the same (fence submitted work, restore the in-flight layer and
hooks, make completed and unreached layers executable, repair version counters,
retain or release the private offload root), but the two loops used to order
them in separate exception handlers, so every new obligation had to be
remembered twice (#5060).

``RunExitRecovery`` owns that sequence. Each looper supplies the callbacks for
the steps it actually acquired; steps whose callback is ``None`` are skipped.
The success path publishes only after every restore step succeeds, and a
failure in any step enters the same abort path as a stopped or failed body.
Abort steps are independent and best effort: a secondary repair failure is
recorded on ``failures`` and logged, never raised in place of the exception
that ended the run.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from ..utils.logger import setup_logger


log = setup_logger()

_ExitStep = Optional[Callable[[], None]]


class RunExitRecovery:
    """Run-scoped owner of the exit sequence shared by both loopers."""

    def __init__(
        self,
        *,
        fence: _ExitStep = None,
        release_processors: _ExitStep = None,
        restore_inflight: _ExitStep = None,
        restore_offloaded: Optional[Callable[..., None]] = None,
        materialize_unreached: _ExitStep = None,
        rehome_kernels: _ExitStep = None,
        repair_counters: _ExitStep = None,
        release_private_root: _ExitStep = None,
        restore_disk: _ExitStep = None,
        materialize_success: _ExitStep = None,
        release_workspace: _ExitStep = None,
        release_processors_before_restore: bool = True,
    ) -> None:
        self._fence = fence
        self._release_processors = release_processors
        self._restore_inflight = restore_inflight
        self._restore_offloaded = restore_offloaded
        self._materialize_unreached = materialize_unreached
        self._rehome_kernels = rehome_kernels
        self._repair_counters = repair_counters
        self._release_private_root = release_private_root
        self._restore_disk = restore_disk
        self._materialize_success = materialize_success
        self._release_workspace = release_workspace
        # Both loopers release their Device-SMI handles after the fence, but
        # the weight-only contract restores its in-flight layer before that
        # release (tests/test_weight_only_failed_run_device_5011.py). The
        # processor release does not touch model tensors, so only the position
        # differs; the owner still runs it exactly once.
        self._release_processors_before_restore = release_processors_before_restore
        #: ``(step label, exception)`` for repairs that failed while aborting.
        self.failures: List[Tuple[str, BaseException]] = []
        self._aborted = False
        self._completed = False

    @property
    def completed(self) -> bool:
        """True only when the success restore finished and was published."""

        return self._completed

    @property
    def recovered(self) -> bool:
        """True only when a successful run completed without repair failures."""

        return self._completed and not self.failures

    def run(self, body: Callable[[], object]):
        """Run ``body`` and apply the success or abort sequence exactly once."""

        try:
            try:
                result = body()
            except BaseException:
                self.abort()
                raise
            try:
                self.complete()
            except BaseException:
                # The tail failed after the body published its layers. Recover
                # them, then re-raise the tail failure rather than replacing it
                # with a repair error.
                self.abort()
                raise
            return result
        finally:
            # Both loopers release process-wide processor scratch after either
            # exit; a failure here is a secondary repair failure, not the run's.
            self._best_effort("release run workspace", self._release_workspace)

    def complete(self) -> None:
        """Publish a successful run only after every restore step succeeds."""

        self._required(
            "restore offloaded base modules",
            lambda: self._restore_offloaded(strict=True),
        )
        self._required("restore disk-offloaded modules", self._restore_disk)
        self._required("materialize the completed model", self._materialize_success)
        self._required("repair tensor version counters", self._repair_counters)
        self._completed = True

    def abort(self) -> None:
        """Recover a stopped or failed run. Idempotent and never raises."""

        if self._aborted:
            return
        self._aborted = True
        # Fence before touching the state the workers may still mutate, then
        # release the handles that were polling the devices. Each step is
        # independent so one broken repair cannot skip the rest.
        self._best_effort("fence submitted work", self._fence)
        if self._release_processors_before_restore:
            self._best_effort("release processor resources", self._release_processors)
        self._best_effort("restore the in-flight layer", self._restore_inflight)
        self._best_effort(
            "restore offloaded base modules",
            lambda: self._restore_offloaded(strict=False),
        )
        self._best_effort("materialize unreached layers", self._materialize_unreached)
        self._best_effort("rehome failed-run kernels", self._rehome_kernels)
        self._best_effort("repair tensor version counters", self._repair_counters)
        if not self._release_processors_before_restore:
            self._best_effort("release processor resources", self._release_processors)
        self._best_effort(
            "keep or release the offload root", self._release_private_root
        )

    @staticmethod
    def _required(label: str, step: Optional[Callable[[], None]]) -> None:
        if step is not None:
            step()

    def _best_effort(
        self, label: str, step: Optional[Callable[[], None]]
    ) -> None:
        if step is None:
            return
        try:
            step()
        except BaseException as exc:  # noqa: BLE001 - cleanup must not mask the run error
            self.failures.append((label, exc))
            log.warning("run-exit recovery step %r failed", label, exc_info=True)


__all__ = ["RunExitRecovery"]
