# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Runtime policy plumbing for exact QVQ Viterbi survivor pruning.

``ViterbiPruningConfig`` is the public serialized schema; this module turns
one of those (or an equivalent mapping) into the small integer policy code the
CUDA V2 segmented grid dispatch consumes. The codes are part of the native op
schema, so they must stay stable:

===  ==========================================================================
0    ``auto`` + ``fallback="baseline"`` -- today's automatic behavior. The
     deprecated ``GPTQMODEL_QVQ_DISABLE_OCTET_GRID`` A/B escape is honored and
     any ineligible call silently keeps the exact baseline recurrence.
1    ``off`` -- norm-band dispatch is deterministically suppressed and the
     legacy environment variable is ignored.
2    ``auto`` + ``fallback="error"`` -- the legacy environment variable is
     honored, but a call that cannot use norm-band pruning raises instead of
     falling back.
3    ``required`` -- the legacy environment variable is ignored and a call that
     cannot use norm-band pruning raises instead of falling back.
===  ==========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


VITERBI_PRUNING_AUTO = 0
VITERBI_PRUNING_OFF = 1
VITERBI_PRUNING_AUTO_ERROR = 2
VITERBI_PRUNING_REQUIRED = 3

_POLICY_CODES = {
    ("auto", "baseline"): VITERBI_PRUNING_AUTO,
    ("auto", "error"): VITERBI_PRUNING_AUTO_ERROR,
    ("off", "baseline"): VITERBI_PRUNING_OFF,
    ("required", "baseline"): VITERBI_PRUNING_REQUIRED,
}


@dataclass(frozen=True)
class ViterbiPruningPolicy:
    """Immutable runtime view of ``ViterbiPruningConfig``."""

    mode: str = "auto"
    strategy: str = "norm_band"
    exact: bool = True
    fallback: str = "baseline"

    @property
    def dispatch_code(self) -> int:
        """Return the native policy code for the V2 segmented grid dispatch."""

        try:
            return _POLICY_CODES[(self.mode, self.fallback)]
        except KeyError:
            raise ValueError(
                f"QVQ Viterbi pruning policy `mode={self.mode!r}` with `fallback={self.fallback!r}` "
                "is not a supported combination."
            ) from None


DEFAULT_VITERBI_PRUNING_POLICY = ViterbiPruningPolicy()


def resolve_viterbi_pruning_policy(value: Any) -> ViterbiPruningPolicy:
    """Normalize ``None``/mapping/config objects into a ``ViterbiPruningPolicy``.

    ``None`` resolves to the ``auto``/``baseline`` default, which reproduces the
    historical automatic behavior exactly and keeps every direct low-level
    caller backward compatible.
    """

    if value is None:
        return DEFAULT_VITERBI_PRUNING_POLICY
    if isinstance(value, ViterbiPruningPolicy):
        return value
    if isinstance(value, Mapping):
        fields = dict(value)
    else:
        try:
            fields = {
                "mode": value.mode,
                "strategy": value.strategy,
                "exact": value.exact,
                "fallback": value.fallback,
            }
        except AttributeError:
            raise TypeError(
                "QVQ `viterbi_pruning` must be a ViterbiPruningConfig, ViterbiPruningPolicy, "
                f"mapping, or None; got {type(value).__name__}."
            ) from None
    unexpected = set(fields) - {"mode", "strategy", "exact", "fallback"}
    if unexpected:
        raise ValueError(f"QVQ `viterbi_pruning` has unexpected keys: {sorted(unexpected)}.")
    policy = ViterbiPruningPolicy(**fields)
    if policy.strategy != "norm_band":
        raise ValueError(
            f"QVQ Viterbi pruning strategy `{policy.strategy!r}` is not implemented; "
            "only the exact `norm_band` strategy exists."
        )
    if policy.exact is not True:
        raise ValueError(
            "QVQ Viterbi pruning `exact=False` is rejected because no approximate strategy exists."
        )
    # Validates the mode/fallback matrix eagerly instead of at dispatch time.
    policy.dispatch_code
    return policy


def viterbi_pruning_dispatch_code(value: Any) -> int:
    """Resolve any accepted policy spelling into its native dispatch code."""

    return resolve_viterbi_pruning_policy(value).dispatch_code


VITERBI_PRUNING_STRICT_CODES = frozenset(
    {VITERBI_PRUNING_AUTO_ERROR, VITERBI_PRUNING_REQUIRED}
)


def viterbi_pruning_is_strict(policy_code: int) -> bool:
    """Return whether ``policy_code`` forbids any silent baseline fallback."""

    return policy_code in VITERBI_PRUNING_STRICT_CODES


def reject_viterbi_pruning_fallback_if_strict(policy_code: int, *, reason: str) -> None:
    """Fail before a baseline fallback when the policy forbids one.

    The native CUDA op enforces the same contract for calls that reach it;
    this Python-side twin covers every outer dispatch guard the native op
    never sees — a CPU/MPS call, a non-FP32 working dtype, non-contiguous
    tensors, or a pre-``sm_80`` device — so ``mode="required"`` and
    ``mode="auto"``+``fallback="error"`` can never be satisfied silently by
    the eager recurrence.
    """

    if viterbi_pruning_is_strict(policy_code):
        raise RuntimeError(
            "QVQ exact norm-band Viterbi pruning was requested with "
            "`viterbi_pruning.mode='required'` or `fallback='error'`, but this call cannot "
            f"use it: {reason}. Set `viterbi_pruning.mode='auto'` with the default "
            "`fallback='baseline'` to keep the exact baseline recurrence instead."
        )
