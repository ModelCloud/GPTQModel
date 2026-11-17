# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Generic shared-input Hessian capture dedup for quantizer processors.

``gptqmodel.models.shared_input`` already groups sibling projections that
consume the same activation (``:in=<tag>``).  ``GPTQProcessor`` turns that plan
into a per-subset leader/follower election.  This module provides the same
election for the other calibrated quantizers (QQQ, VPTQ, MACaQ, EXL3) so they
do not accumulate an identical ``H = X^T X`` once per sibling.

The processor only decides *who* captures.  Each quantizer owns
``adopt_hessian_from`` because the shape of its statistics differs (QQQ keeps a
token-normalized float32 mean, VPTQ a float64 mean, MACaQ a float64 sum, and
GPTAQ/FOEM also carry ``dXXT``).
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from ..models.shared_input import SharedInputPlan
from ..utils.logger import setup_logger
from .loop_processor import LoopProcessor


log = setup_logger()


def dedup_shared_inputs_enabled(qcfg: Any) -> bool:
    """Return whether the config enables shared-input Hessian dedup.

    Quantizers whose config predates :class:`HessianConfig` (MACaQ, EXL3) have
    no switch; for them dedup is the only path.
    """

    hessian = getattr(qcfg, "hessian", None)
    return bool(getattr(hessian, "dedup_shared_inputs", True))


def elect_shared_input_leaders(
    *,
    plan: Optional[SharedInputPlan],
    subset_names: Iterable[str],
    task_for: Callable[[str], Any],
    signature_for: Callable[[Any], Optional[Tuple[Any, ...]]],
    enabled_for: Callable[[Any], bool],
    log: Any,
) -> Dict[str, str]:
    """Elect one capture leader per explicit group present in ``subset_names``.

    ``signature_for`` returns the numerical settings that must match for two
    modules' Hessians to be interchangeable (input width, accumulation dtype,
    native-delta mode, ...).  A ``None`` signature marks a task that must not
    participate.  Groups split across subsets dedup only among the members
    captured in this pass, matching ``SharedInputGroup.dedup_followers``.
    """

    if plan is None or not plan.shared_groups:
        return {}

    names = list(subset_names)
    leaders: Dict[str, str] = {}
    for group in plan.shared_groups:
        if not group.explicit:
            continue

        members = []
        for name in names:
            if name not in group.modules:
                continue
            task = task_for(name)
            if task is None or not enabled_for(task):
                continue
            signature = signature_for(task)
            if signature is None:
                continue
            members.append((name, signature))

        if len(members) < 2:
            continue

        leader_name, leader_signature = members[0]
        for follower_name, follower_signature in members[1:]:
            if follower_signature != leader_signature:
                log.warn(
                    f"Quantization: shared-input group `{group.key}` mixes Hessian "
                    f"accumulation settings ({leader_name}={leader_signature}, "
                    f"{follower_name}={follower_signature}); not sharing Hessian."
                )
                continue
            leaders[follower_name] = leader_name
    return leaders


class SharedInputCaptureMixin:
    """Leader/follower capture lifecycle for processors that own Hessian tasks.

    Subclasses implement :meth:`_shared_input_task` (and optionally the
    signature/enabled/adopt hooks) and call :meth:`_init_shared_input_capture`
    from their constructor.
    """

    def _init_shared_input_capture(self) -> None:
        # The effective module tree is instance-specific; keep a strong
        # reference and compare by identity so a collected model's id cannot be
        # reused by a different plan owner.
        self._shared_input_plan: Optional[SharedInputPlan] = None
        self._shared_input_plan_owner: Optional[Any] = None
        self._shared_input_plan_lock = threading.Lock()
        self._shared_input_leaders: Dict[str, str] = {}
        self._shared_input_capture_generation = 0
        self.shared_input_dedup_count = 0
        self.shared_input_dedup_telemetry: Dict[str, Any] = {}

    # -- subclass hooks -----------------------------------------------------

    def _shared_input_task(self, name: str) -> Any:
        return self.tasks.get(name)

    def _shared_input_task_enabled(self, task: Any) -> bool:
        return True

    def _shared_input_task_signature(self, task: Any) -> Optional[Tuple[Any, ...]]:
        return (type(task).__name__, int(task.columns))

    def _shared_input_task_adopt(self, follower: Any, leader: Any) -> None:
        follower.adopt_hessian_from(leader)

    def _shared_input_begin_cohort(self, tasks: Dict[str, Any]) -> None:
        """Hook for processors that cache per-cohort Hessian artifacts."""

    # -- lifecycle ----------------------------------------------------------

    def _resolve_shared_input_plan(self, model: Any) -> Optional[SharedInputPlan]:
        with self._shared_input_plan_lock:
            if self._shared_input_plan_owner is model:
                return self._shared_input_plan

            plan: Optional[SharedInputPlan] = None
            shared_input_plan = getattr(model, "shared_input_plan", None)
            if callable(shared_input_plan):
                inner = getattr(model, "model", None)
                plan = shared_input_plan(
                    model_config=getattr(inner, "config", None),
                    quantize_config=getattr(model, "quantize_config", None),
                )

            self._shared_input_plan = plan
            self._shared_input_plan_owner = model
            return plan

    def begin_shared_input_capture(
        self,
        model: Any,
        subset_names: Iterable[str],
        is_lm_head_module: bool = False,
    ) -> Dict[str, str]:
        """Elect one Hessian capture leader per explicit shared-input group."""

        with self.lock:
            self._shared_input_leaders = {}
            tasks = dict(self.tasks)

        self._shared_input_begin_cohort(tasks)

        if is_lm_head_module or not dedup_shared_inputs_enabled(self.qcfg):
            return {}

        plan = self._resolve_shared_input_plan(model)
        leaders = elect_shared_input_leaders(
            plan=plan,
            subset_names=subset_names,
            task_for=self._shared_input_task,
            signature_for=self._shared_input_task_signature,
            enabled_for=self._shared_input_task_enabled,
            log=log,
        )

        with self.lock:
            self._shared_input_capture_generation += 1
            self._shared_input_leaders = leaders
        return dict(leaders)

    def end_shared_input_capture(self, subset_names: Iterable[str]) -> Dict[str, Any]:
        """Copy the leader's finalized statistics into each follower."""

        subset_count = len(subset_names) if isinstance(subset_names, (list, tuple)) else 0
        with self.lock:
            leaders = self._shared_input_leaders
            self._shared_input_leaders = {}

        adopted = 0
        for follower, leader in leaders.items():
            follower_task = self._shared_input_task(follower)
            leader_task = self._shared_input_task(leader)
            if follower_task is None or leader_task is None:
                continue
            self._shared_input_task_adopt(follower_task, leader_task)
            adopted += 1

        with self.lock:
            self.shared_input_dedup_count += adopted
            telemetry = {
                "enabled": dedup_shared_inputs_enabled(self.qcfg),
                "expected_followers": len(leaders),
                "adopted_followers": adopted,
                "cumulative_adopted_followers": self.shared_input_dedup_count,
                "leader_count": len(set(leaders.values())),
                "follower_to_leader": dict(leaders),
                "subset_module_count": subset_count,
                "status": "verified" if adopted == len(leaders) else "mismatch",
            }
            self.shared_input_dedup_telemetry = telemetry
        return dict(telemetry)

    def shared_input_leader(self, name: str) -> Optional[str]:
        """Return the module whose Hessian ``name`` will adopt in this pass."""

        leaders = getattr(self, "_shared_input_leaders", None)
        if not leaders:
            return None

        lock = getattr(self, "lock", None)
        if lock is None:
            return leaders.get(name)
        with lock:
            return leaders.get(name)


class SharedInputCaptureProcessor(SharedInputCaptureMixin, LoopProcessor):
    """Processor base that enables shared-input capture dedup.

    Both bases define ``begin_shared_input_capture`` and
    ``end_shared_input_capture``: the mixin elects capture leaders and adopts
    their statistics, while :class:`LoopProcessor` provides no-op defaults for
    every processor that does not dedup.  Delegating explicitly here resolves
    that multiple-inheritance conflict in one place and keeps the mixin's
    implementations authoritative, without changing either base.
    """

    def begin_shared_input_capture(
        self,
        model: Any,
        subset_names: Iterable[str],
        is_lm_head_module: bool = False,
    ) -> Dict[str, str]:
        return SharedInputCaptureMixin.begin_shared_input_capture(
            self, model, subset_names, is_lm_head_module
        )

    def end_shared_input_capture(self, subset_names: Iterable[str]) -> Dict[str, Any]:
        return SharedInputCaptureMixin.end_shared_input_capture(self, subset_names)
