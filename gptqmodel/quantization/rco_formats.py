# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Per-format adapters that let RCO allocate widths for AWQ, EXL3 and GGUF.

``gptqmodel.quantization.rco`` is a pure optimizer: it takes one weight per
group, one cost per option and one score per (group, option), and returns a
budget-feasible assignment. Everything that differs between backends lives
here:

===============  =========================  =======================  ==============
backend          options                    cost (bits/parameter)    search
===============  =========================  =======================  ==============
AWQ              ``w2`` / ``w3`` / ``w4``   ``bits + (16+bits)/group``  interpolation
EXL3             ``k2`` / ``k3`` / ``k4``   ``bits``                 reconstruction
GGUF             ``iq2_xxs`` ... ``q6_k``   ggml block bpw           reconstruction
===============  =========================  =======================  ==============

Why the fidelity split is forced by the checkpoint layouts
----------------------------------------------------------
The paper-faithful search (GSQ, arXiv 2604.18556 Sec. 4.2 and Table 1, built on
the exact-budget manifold of Helcig & Alistarh, arXiv 2605.00649 Secs. 2-3)
needs the model loss as a differentiable function of the assignment, which
``rco.InterpolatedLlamaLayers`` obtains by interpolating *dense* staged arms.
AWQ stages dense weights and only then packs affine codes, so its arms can be
interpolated. EXL3 stores a trellis and GGUF stores ggml blocks; neither is a
linear function of a dense weight, so those backends score each option with the
Hessian-weighted reconstruction error their quantizer already reports
(``trace(err H err^T) / trace(W H W^T)``). Every plan records which search
produced it in :attr:`RCOPlan.search`, so a proxy allocation is never reported
as the paper search.

The module also owns architecture-generic projection discovery. The staged GSQ
scripts previously hardcoded the seven Llama projections, which silently
omitted Ripple's eighth projection (``self_attn.g_proj``, gated attention).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch

from .config import (
    _GGUF_APPROX_BITS_PER_WEIGHT_BY_ALIAS,
    FORMAT_FIELD_CODE,
    METHOD,
)
from .rco import achieved_bits, budget_constrained_argmax, discover_projections


#: Staged AWQ widths. ``AwqTorchLinear`` packs and runs 2-8 bits; the CUDA
#: GEMM/GEMV kernels are 4-bit only, so a mixed-width AWQ checkpoint needs the
#: torch kernel (see :func:`awq_runtime_requirement`).
AWQ_STAGED_WIDTHS: Tuple[int, ...] = (2, 3, 4)

#: EXL3 packs one integer trellis depth per module, so the option set is integer.
EXL3_STAGED_WIDTHS: Tuple[int, ...] = (2, 3, 4)

#: GGUF quant types that bracket the paper's ~2.4-2.6 bpw budgets. Ordered by
#: increasing cost so a plan reads like a ladder.
GGUF_STAGED_ALIASES: Tuple[str, ...] = (
    "iq2_xxs",
    "iq2_xs",
    "iq2_s",
    "q2_k",
    "iq3_xxs",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
)

class RCOFormatError(ValueError):
    """Raised when an option set or assignment cannot be realised by a backend."""


@dataclass(frozen=True)
class RCOOption:
    """One selectable quantisation width (or GGUF type) for a module."""

    label: str
    cost: float
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RCOPlan:
    """A budget-feasible allocation, ready to feed ``*Config(dynamic=...)``."""

    method: str
    search: str
    target_bits: float
    achieved_bits: float
    names: Tuple[str, ...]
    labels: Tuple[str, ...]
    costs: Tuple[float, ...]
    assignment: Tuple[int, ...]
    dynamic: Dict[str, Dict[str, Any]]
    counts: Dict[str, int]
    resolution: int
    #: Non-empty when the assignment is representable but needs a specific
    #: runtime to serve (e.g. mixed-width AWQ needs the torch kernel).
    runtime_note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "search": self.search,
            "target_bits": self.target_bits,
            "achieved_bits": self.achieved_bits,
            "options": list(self.labels),
            "costs": list(self.costs),
            "module_bits": dict(zip(self.names, (self.labels[i] for i in self.assignment))),
            "dynamic": {name: dict(entry) for name, entry in self.dynamic.items()},
            "width_counts": dict(self.counts),
            "resolution": self.resolution,
            "runtime_note": self.runtime_note,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RCOPlan":
        module_bits: Mapping[str, str] = payload["module_bits"]
        names = tuple(module_bits)
        labels = tuple(payload["options"])
        index = {label: position for position, label in enumerate(labels)}
        return cls(
            method=str(payload["method"]),
            search=str(payload["search"]),
            target_bits=float(payload["target_bits"]),
            achieved_bits=float(payload["achieved_bits"]),
            names=names,
            labels=labels,
            costs=tuple(float(cost) for cost in payload["costs"]),
            assignment=tuple(index[module_bits[name]] for name in names),
            dynamic={name: dict(entry) for name, entry in payload["dynamic"].items()},
            counts={str(k): int(v) for k, v in payload["width_counts"].items()},
            resolution=int(payload.get("resolution", 0)),
            runtime_note=payload.get("runtime_note"),
        )


class RCOFormatAdapter:
    """Maps an RCO assignment onto one backend's ``dynamic`` configuration."""

    method: METHOD
    #: ``"interpolation"`` when staged dense arms can be interpolated, otherwise
    #: ``"reconstruction"`` for the Hessian-weighted error proxy.
    search: str

    def __init__(self, qcfg: Any):
        self.qcfg = qcfg

    def options(self) -> Tuple[RCOOption, ...]:
        raise NotImplementedError

    def costs(self) -> torch.Tensor:
        return torch.tensor([option.cost for option in self.options()], dtype=torch.float32)

    def labels(self) -> Tuple[str, ...]:
        return tuple(option.label for option in self.options())

    def dynamic_entry(self, option: RCOOption) -> Dict[str, Any]:
        return dict(option.payload)

    def validate(self, assignment: Mapping[str, int]) -> None:
        """Reject an assignment the backend cannot represent.

        ``assignment`` maps module name to an index into :meth:`options`.
        """
        del assignment

    def runtime_requirement(self, assignment: Mapping[str, int]) -> Optional[str]:
        """Describe a runtime constraint the assignment imposes, if any."""
        del assignment
        return None

    def build_dynamic(
        self, names: Sequence[str], assignment: Sequence[int]
    ) -> Dict[str, Dict[str, Any]]:
        options = self.options()
        if len(names) != len(assignment):
            raise RCOFormatError("RCO assignment must cover every allocated module")
        dynamic: Dict[str, Dict[str, Any]] = {}
        for name, index in zip(names, assignment):
            position = int(index)
            if not 0 <= position < len(options):
                raise RCOFormatError(
                    f"RCO option index {position} for `{name}` is outside "
                    f"[0, {len(options)})"
                )
            dynamic[str(name)] = self.dynamic_entry(options[position])
        self.validate({str(name): int(index) for name, index in zip(names, assignment)})
        return dynamic


def _scale_overhead_bits(bits: float, group_size: int) -> float:
    """Per-parameter cost of AWQ's fp16 scale plus packed zero point.

    AWQ stores one scale (16 bits) and one zero point (``bits`` wide) per group.
    ``group_size=-1`` means per-output-channel scales, whose overhead depends on
    ``in_features`` and is therefore reported as zero here; callers that need the
    exact figure should pass a positive ``group_size``.
    """
    if group_size <= 0:
        return 0.0
    return (16.0 + bits) / float(group_size)


class AWQFormatAdapter(RCOFormatAdapter):
    """AWQ per-module affine widths (``bits`` in ``dynamic``)."""

    method = METHOD.AWQ
    search = "interpolation"

    def options(self) -> Tuple[RCOOption, ...]:
        group_size = int(getattr(self.qcfg, "group_size", -1) or -1)
        return tuple(
            RCOOption(
                label=f"w{bits}",
                cost=bits + _scale_overhead_bits(bits, group_size),
                payload={"bits": int(bits)},
            )
            for bits in AWQ_STAGED_WIDTHS
        )

    def runtime_requirement(self, assignment: Mapping[str, int]) -> Optional[str]:
        options = self.options()
        chosen = {int(options[int(index)].payload["bits"]) for index in assignment.values()}
        # Staging is correct at any width, but the packed checkpoint has to be
        # served by a kernel that can unpack it.
        return awq_runtime_requirement(sorted(chosen)) or None


def awq_runtime_requirement(widths: Iterable[int]) -> str:
    """Explain how to serve an AWQ checkpoint that is not plain W4."""
    widths = sorted({int(width) for width in widths})
    if widths == [4]:
        return ""
    return (
        f"AWQ RCO selected widths {widths}, but the CUDA GEMM/GEMV kernels only "
        "unpack 4-bit codes. Serve this checkpoint with the portable torch kernel "
        "(`FORMAT.GEMM` + `BACKEND.AWQ_TORCH`, i.e. `AwqTorchLinear`, which packs "
        "and runs 2-8 bits), or restrict the option set to W4."
    )


class EXL3FormatAdapter(RCOFormatAdapter):
    """EXL3 per-module integer trellis widths."""

    method = METHOD.EXL3
    search = "reconstruction"

    def options(self) -> Tuple[RCOOption, ...]:
        return tuple(
            RCOOption(label=f"k{bits}", cost=float(bits), payload={"bits": int(bits)})
            for bits in EXL3_STAGED_WIDTHS
        )

    def validate(self, assignment: Mapping[str, int]) -> None:
        options = self.options()
        known = {int(option.payload["bits"]) for option in options}
        for index in assignment.values():
            bits = int(options[int(index)].payload["bits"])
            if bits not in known or float(bits) != bits:
                raise RCOFormatError(
                    f"EXL3 RCO supports integer trellis widths {sorted(known)}, got {bits}"
                )


class GGUFFormatAdapter(RCOFormatAdapter):
    """GGUF per-module quant types (``bits`` carries the ggml alias)."""

    method = METHOD.GGUF
    search = "reconstruction"

    def __init__(self, qcfg: Any, aliases: Optional[Sequence[str]] = None):
        super().__init__(qcfg)
        self.aliases = tuple(aliases or GGUF_STAGED_ALIASES)

    def options(self) -> Tuple[RCOOption, ...]:
        options = []
        for alias in self.aliases:
            normalized = alias.strip().lower()
            cost = _GGUF_APPROX_BITS_PER_WEIGHT_BY_ALIAS.get(normalized)
            if cost is None:
                raise RCOFormatError(
                    f"GGUF RCO has no bits-per-weight estimate for `{alias}`; "
                    "add it to `_GGUF_APPROX_BITS_PER_WEIGHT_BY_ALIAS`"
                )
            # `clone_weight_only_config_for_module` resolves a string `bits` into
            # both the width and the ggml type, so a bare alias is enough.
            options.append(
                RCOOption(label=normalized, cost=float(cost), payload={"bits": normalized})
            )
        return tuple(options)

    def dynamic_entry(self, option: RCOOption) -> Dict[str, Any]:
        entry = dict(option.payload)
        entry[FORMAT_FIELD_CODE] = option.label
        return entry

    def validate(self, assignment: Mapping[str, int]) -> None:
        options = self.options()
        known = {option.label for option in options}
        for index in assignment.values():
            label = options[int(index)].label
            if label not in known:
                raise RCOFormatError(
                    f"GGUF RCO options must come from {sorted(known)}, got `{label}`"
                )


_ADAPTERS: Dict[METHOD, type] = {
    METHOD.AWQ: AWQFormatAdapter,
    METHOD.EXL3: EXL3FormatAdapter,
    METHOD.GGUF: GGUFFormatAdapter,
}


def adapter_for(qcfg: Any) -> RCOFormatAdapter:
    """Return the RCO adapter matching a quantisation config."""
    method = getattr(qcfg, "method", None)
    if isinstance(method, str):
        method = METHOD(method)
    adapter = _ADAPTERS.get(method)
    if adapter is None:
        supported = ", ".join(sorted(candidate.value for candidate in _ADAPTERS))
        raise RCOFormatError(
            f"RCO has no per-format adapter for method `{method}`; supported: {supported}"
        )
    return adapter(qcfg)


# Projection discovery is shared with the GPTQ RCO scripts. A per-format plan
# and a staged-arm plan must see the same module set, or the two budgets are
# computed over different denominators. `rco.discover_projections` walks the live
# module tree, so a model that adds a projection (Ripple's `self_attn.g_proj`)
# is covered without a per-architecture list.


def module_weights(
    model: torch.nn.Module, names: Sequence[str]
) -> Tuple[torch.Tensor, Tuple[str, ...]]:
    """Parameter counts of the quantised matrices, plus the names that exist.

    Only ``in_features * out_features`` is counted: biases and the surrounding
    norms are not part of the bit budget.
    """
    weights = []
    kept = []
    for name in names:
        try:
            module = model.get_submodule(str(name))
        except AttributeError:
            continue
        in_features = getattr(module, "in_features", None)
        out_features = getattr(module, "out_features", None)
        if in_features is None or out_features is None:
            continue
        weights.append(float(int(in_features) * int(out_features)))
        kept.append(str(name))
    if not kept:
        raise RCOFormatError("RCO found no quantisable projection to allocate")
    return torch.tensor(weights, dtype=torch.float32), tuple(kept)


def relative_reconstruction_error(
    reference: torch.Tensor,
    dequantized: torch.Tensor,
    hessian: Optional[torch.Tensor] = None,
) -> float:
    """Hessian-weighted relative error of one quantised weight matrix.

    ``trace(err H err^T) / trace(W H W^T)`` is the same functional that
    ``quantize_exl3`` reports, so EXL3, GGUF and AWQ scores are comparable.
    Without a Hessian the plain Frobenius ratio is used, which is the
    ``H = I`` special case.
    """
    error = dequantized.detach().to(torch.float32) - reference.detach().to(torch.float32)
    reference = reference.detach().to(torch.float32)
    if hessian is None:
        numerator = error.pow(2).sum()
        denominator = reference.pow(2).sum().clamp_min(1e-30)
    else:
        h = hessian.detach().to(torch.float32)
        if h.shape != (reference.shape[1], reference.shape[1]):
            raise RCOFormatError(
                "Hessian must be the input second moment "
                f"[{reference.shape[1]}, {reference.shape[1]}], got {tuple(h.shape)}"
            )
        err_h = error @ h
        ref_h = reference @ h
        numerator = (err_h * error).sum()
        denominator = (ref_h * reference).sum().clamp_min(1e-30)
    value = float(numerator / denominator)
    if not math.isfinite(value):
        raise RCOFormatError(f"RCO reconstruction error is not finite: {value!r}")
    return value


def reconstruction_scores(
    reference: torch.Tensor,
    quantize: Callable[[RCOOption], torch.Tensor],
    options: Sequence[RCOOption],
    *,
    hessian: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Score every option of one module as ``-relative_error``.

    RCO maximises the score, so a lower reconstruction error is a higher score.
    """
    scores = []
    for option in options:
        dequantized = quantize(option)
        scores.append(-relative_reconstruction_error(reference, dequantized, hessian))
    return torch.tensor(scores, dtype=torch.float32)


def default_resolution(module_count: int) -> int:
    """DP budget buckets for ``budget_constrained_argmax``.

    The solver rounds costs to ``resolution`` buckets, so too few buckets make
    the exact optimum infeasible (four 2-bit and four 3-bit modules on 8 modules
    needs 256, not 500). Sixty-four buckets per module keeps the rounding error
    under ``0.5 / 64`` bits per parameter while staying near 1 GiB of DP state.
    """
    return max(256, 64 * max(1, int(module_count)))


#: Upper bound on the DP's ``choice`` table so a refinement cannot exhaust RAM.
_DP_TABLE_BUDGET_BYTES = 1 << 30


def _dp_table_bytes(module_count: int, resolution: int, target_bits: float) -> float:
    """Approximate size of ``budget_constrained_argmax``'s backtrack table."""
    buckets = max(1, round(target_bits * resolution))
    return module_count * buckets * 8 * 1.1


def repair_overspend(assignment, scores, weights, costs, target_bits):
    """Greedily downgrade modules until the exact budget is met.

    The DP optimizes over a rounded cost grid, so its solution can evaluate a
    few thousandths of a bit above the target even though a feasible assignment
    exists. Each step applies the single-module downgrade with the smallest
    score loss per bit saved, which keeps the repair as close to the DP optimum
    as a greedy pass can be, and stops as soon as the budget holds.
    """
    assignment = assignment.clone()
    achieved = achieved_bits(assignment, weights, costs)
    if achieved <= target_bits:
        return assignment
    costs = costs.to(assignment.device)
    weights = weights.to(assignment.device)
    scores = scores.detach()
    for _ in range(len(assignment)):
        best = None
        for index in range(len(assignment)):
            current = int(assignment[index])
            for option in range(scores.shape[1]):
                if float(costs[option]) >= float(costs[current]):
                    continue
                saved = float(costs[current] - costs[option]) * float(weights[index])
                if saved <= 0:
                    continue
                loss = float(scores[index, current] - scores[index, option])
                key = (loss / saved, -saved)
                if best is None or key < best[0]:
                    best = (key, index, option)
        if best is None:
            break
        assignment[best[1]] = best[2]
        achieved = achieved_bits(assignment, weights, costs)
        if achieved <= target_bits:
            break
    return assignment


def allocate(
    adapter: RCOFormatAdapter,
    scores: torch.Tensor,
    weights: torch.Tensor,
    target_bits: float,
    *,
    names: Sequence[str],
    resolution: Optional[int] = None,
) -> RCOPlan:
    """Solve the rate-constrained assignment and wrap it in an :class:`RCOPlan`."""
    if scores.ndim != 2 or scores.shape != (len(names), len(adapter.options())):
        raise RCOFormatError(
            f"RCO scores must be [{len(names)}, {len(adapter.options())}], "
            f"got {tuple(scores.shape)}"
        )
    if weights.shape != (len(names),):
        raise RCOFormatError("RCO weights must have one entry per allocated module")
    module_count = len(names)
    resolution = default_resolution(module_count) if resolution is None else int(resolution)
    costs = adapter.costs()

    def solve(buckets: int) -> torch.Tensor:
        return budget_constrained_argmax(
            scores, weights, costs, target_bits, resolution=buckets
        )

    assignment = solve(resolution)
    achieved = achieved_bits(assignment, weights, costs)
    # `budget_constrained_argmax` rounds every unit cost onto the budget grid, so
    # a solution can land up to `0.5 * n / resolution` bits above the target even
    # when an exact assignment exists. Refine on a finer grid while the DP table
    # still fits, then judge the result against that rounding bound instead of a
    # fixed epsilon (a fixed 1e-3 rejected every 640-module plan at the default
    # resolution, where the bound is 0.0078).
    tolerance = 0.5 * module_count / resolution
    while (
        achieved > target_bits + 1e-9
        and _dp_table_bytes(module_count, resolution * 2, target_bits) <= _DP_TABLE_BUDGET_BYTES
    ):
        resolution *= 2
        assignment = solve(resolution)
        achieved = achieved_bits(assignment, weights, costs)
        tolerance = 0.5 * module_count / resolution
    if achieved > target_bits + 1e-9:
        assignment = repair_overspend(assignment, scores, weights, costs, target_bits)
        achieved = achieved_bits(assignment, weights, costs)
    if achieved > target_bits + tolerance:
        raise RCOFormatError(
            f"RCO allocation overspent the budget: {achieved:.6f} > {target_bits:.6f} "
            f"(rounding bound {tolerance:.6f} at resolution {resolution})"
        )
    labels = adapter.labels()
    chosen = [labels[int(index)] for index in assignment]
    counts: Dict[str, int] = {}
    for label in chosen:
        counts[label] = counts.get(label, 0) + 1
    index_map = {str(name): int(index) for name, index in zip(names, assignment)}
    dynamic = adapter.build_dynamic(names, assignment)
    runtime_note = adapter.runtime_requirement(index_map)
    return RCOPlan(
        method=str(getattr(adapter.method, "value", adapter.method)),
        search=adapter.search,
        target_bits=float(target_bits),
        achieved_bits=achieved,
        names=tuple(str(name) for name in names),
        labels=labels,
        costs=tuple(float(cost) for cost in costs),
        assignment=tuple(int(index) for index in assignment),
        dynamic=dynamic,
        counts=counts,
        resolution=resolution,
        runtime_note=runtime_note,
    )


__all__ = [
    "AWQFormatAdapter",
    "AWQ_STAGED_WIDTHS",
    "EXL3FormatAdapter",
    "EXL3_STAGED_WIDTHS",
    "GGUFFormatAdapter",
    "GGUF_STAGED_ALIASES",
    "RCOFormatAdapter",
    "RCOFormatError",
    "RCOOption",
    "RCOPlan",
    "adapter_for",
    "allocate",
    "awq_runtime_requirement",
    "default_resolution",
    "discover_projections",
    "module_weights",
    "reconstruction_scores",
    "relative_reconstruction_error",
    "repair_overspend",
]
