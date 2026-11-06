# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Reconstruction-error allocation for AWQ, EXL3 and GGUF.

All adapters here use measured reconstruction errors, not differentiable
Gumbel/KL search. Packed-weight budgets include format metadata and padding
when module shapes are supplied; biases and unquantized tensors are excluded.
Projection identity comes from the shared model-definition tree inventory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch

from .config import (
    FORMAT_FIELD_CODE,
    METHOD,
)
from .rco import budget_constrained_argmax, discover_projections


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
    settings: Dict[str, Any] = field(default_factory=dict)
    weights: Tuple[float, ...] = ()
    module_costs: Tuple[Tuple[float, ...], ...] = ()
    shapes: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    score_metric: str = "relative_frobenius"


    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "settings": dict(self.settings),
            "params": dict(zip(self.names, self.weights)),
            "module_costs": dict(zip(self.names, self.module_costs)),
            "shapes": dict(self.shapes),
            "score_metric": self.score_metric,
            "budget_unit": "packed_weight_bits" if self.shapes else "nominal_weight_bits",
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
        if payload.get("schema_version") != 1:
            raise RCOFormatError("RCO plan requires schema_version=1; regenerate the plan")
        module_bits = payload["module_bits"]
        names = tuple(module_bits)
        labels = tuple(payload["options"])
        settings = dict(payload["settings"])
        adapter = adapter_from_settings(payload["method"], settings, labels)
        if labels != adapter.labels() or payload["search"] != "reconstruction":
            raise RCOFormatError("RCO plan options/search do not match the reconstruction adapter")
        try:
            assignment = tuple(labels.index(module_bits[name]) for name in names)
        except ValueError as exc:
            raise RCOFormatError("RCO plan selects an unknown option") from exc
        dynamic = adapter.build_dynamic(names, assignment)
        if dynamic != payload["dynamic"]:
            raise RCOFormatError("RCO plan dynamic config disagrees with its assignment")
        weights = torch.tensor([payload["params"][name] for name in names], dtype=torch.float64)
        costs = torch.tensor([payload["module_costs"][name] for name in names], dtype=torch.float64)
        shapes = {name: tuple(shape) for name, shape in payload.get("shapes", {}).items()}
        expected_unit = "packed_weight_bits" if shapes else "nominal_weight_bits"
        if payload.get("budget_unit") != expected_unit:
            raise RCOFormatError("RCO plan budget unit disagrees with its module shapes")
        if shapes:
            if set(shapes) != set(names):
                raise RCOFormatError("RCO shapes must cover the complete assignment")
            expected = adapter.cost_matrix([shapes[name] for name in names])
            if not torch.equal(costs, expected) or any(
                weights[i] != math.prod(shapes[name]) for i, name in enumerate(names)
            ):
                raise RCOFormatError("RCO plan costs/counts disagree with module shapes")
        elif not torch.equal(costs, adapter.costs().expand(len(names), -1)):
            raise RCOFormatError("RCO plan nominal costs disagree with its options")
        target = float(payload["target_bits"])
        _validate_inputs(torch.zeros_like(costs), weights, costs, target, names)
        achieved = _achieved(assignment, weights, costs)
        if achieved > target or not math.isclose(achieved, float(payload["achieved_bits"]), rel_tol=1e-12, abs_tol=1e-12):
            raise RCOFormatError("RCO plan has an incorrect or overspent budget")
        counts = {label: sum(value == label for value in module_bits.values()) for label in labels}
        counts = {label: count for label, count in counts.items() if count}
        if counts != payload["width_counts"] or tuple(payload["costs"]) != tuple(adapter.costs().tolist()):
            raise RCOFormatError("RCO plan option costs/counts disagree with its assignment")
        metric = payload.get("score_metric")
        if metric not in {"relative_frobenius", "relative_hessian"}:
            raise RCOFormatError("RCO plan requires an explicit reconstruction score metric")
        return cls(
            method=adapter.method.value, search="reconstruction", target_bits=target,
            achieved_bits=achieved, names=names, labels=labels, costs=tuple(adapter.costs().tolist()),
            assignment=assignment, dynamic=dynamic, counts=counts, resolution=int(payload["resolution"]),
            runtime_note=adapter.runtime_requirement(dict(zip(names, assignment))), settings=settings,
            weights=tuple(weights.tolist()), module_costs=tuple(tuple(row) for row in costs.tolist()),
            shapes=shapes, score_metric=metric,
        )


class RCOFormatAdapter:
    """Maps an RCO assignment onto one backend's ``dynamic`` configuration."""

    method: METHOD
    search = "reconstruction"

    def __init__(self, qcfg: Any):
        self.qcfg = qcfg

    def options(self) -> Tuple[RCOOption, ...]:
        raise NotImplementedError

    def costs(self) -> torch.Tensor:
        return torch.tensor([option.cost for option in self.options()], dtype=torch.float64)

    def labels(self) -> Tuple[str, ...]:
        return tuple(option.label for option in self.options())

    def settings(self):
        return {}

    def cost_matrix(self, shapes):
        """Packed tensor bits / original matrix parameters for each option."""
        rows = []
        for shape in shapes:
            if len(shape) != 2 or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in shape):
                raise RCOFormatError("RCO requires positive integer [out, in] shapes")
            out_features, in_features = shape
            rows.append([self.packed_bits(option, in_features, out_features) / math.prod(shape)
                         for option in self.options()])
        return torch.tensor(rows, dtype=torch.float64)

    def packed_bits(self, option, in_features, out_features):
        raise NotImplementedError

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
        if len(names) != len(assignment) or len(set(names)) != len(names):
            raise RCOFormatError("RCO assignment must cover every allocated module")
        dynamic: Dict[str, Dict[str, Any]] = {}
        for name, index in zip(names, assignment):
            if isinstance(index, bool) or int(index) != index:
                raise RCOFormatError("RCO option indices must be integers")
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

    def __init__(self, qcfg, widths=None):
        super().__init__(qcfg)
        self.widths = _widths(widths, AWQ_STAGED_WIDTHS)
        if qcfg.format != "gemm":
            raise RCOFormatError("AWQ RCO requires FORMAT.GEMM for mixed widths")

    def settings(self):
        return {"group_size": self.qcfg.group_size, "sym": self.qcfg.sym, "format": "gemm"}

    def packed_bits(self, option, in_features, out_features):
        group = in_features if self.qcfg.group_size == -1 else self.qcfg.group_size
        if in_features % group or out_features % 8:
            raise RCOFormatError("AWQ shapes require complete groups and out_features divisible by 8")
        words = math.ceil(out_features * option.payload["bits"] / 32)
        return 32 * words * (in_features + in_features // group) + 16 * out_features * (in_features // group)


    def options(self) -> Tuple[RCOOption, ...]:
        group_size = int(getattr(self.qcfg, "group_size", -1) or -1)
        return tuple(
            RCOOption(
                label=f"w{bits}",
                cost=bits + _scale_overhead_bits(bits, group_size),
                payload={"bits": int(bits)},
            )
            for bits in self.widths
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

    def __init__(self, qcfg, widths=None):
        super().__init__(qcfg)
        self.widths = _widths(widths, EXL3_STAGED_WIDTHS)

    def settings(self):
        return {"codebook": self.qcfg.codebook, "out_scales": self.qcfg.out_scales}

    def packed_bits(self, option, in_features, out_features):
        if in_features % 128 or out_features % 128:
            raise RCOFormatError("EXL3 shapes require in_features and out_features divisible by 128")
        metadata = 32 if self.qcfg.codebook in {"mcg", "mul1"} else 0
        return option.payload["bits"] * in_features * out_features + 16 * (in_features + out_features) + metadata

    def options(self) -> Tuple[RCOOption, ...]:
        return tuple(
            RCOOption(label=f"k{bits}", cost=float(bits), payload={"bits": int(bits)})
            for bits in self.widths
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
        self.aliases = tuple(GGUF_STAGED_ALIASES if aliases is None else aliases)

    def options(self) -> Tuple[RCOOption, ...]:
        options = []
        for alias in self.aliases:
            normalized = alias.strip().lower()
            from ..nn_modules.qlinear.gguf import _GGUF_TYPE_INFO, _normalize_gguf_bits

            try:
                _, qtype = _normalize_gguf_bits(normalized)
                info = _GGUF_TYPE_INFO[qtype]
            except (ValueError, KeyError) as exc:
                raise RCOFormatError(f"Unsupported GGUF option: {alias}") from exc
            cost = info["type_size"] * 8 / info["block_size"]
            # `clone_weight_only_config_for_module` resolves a string `bits` into
            # both the width and the ggml type, so a bare alias is enough.
            options.append(
                RCOOption(label=normalized, cost=float(cost), payload={"bits": normalized})
            )
        if not options or len({option.label for option in options}) != len(options):
            raise RCOFormatError("RCO options must be nonempty and unique")
        return tuple(options)

    def packed_bits(self, option, in_features, out_features):
        from ..nn_modules.qlinear.gguf import _GGUF_TYPE_INFO, _normalize_gguf_bits

        _, qtype = _normalize_gguf_bits(option.label)
        info = _GGUF_TYPE_INFO[qtype]
        return out_features * math.ceil(in_features / info["block_size"]) * info["type_size"] * 8

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


def _widths(values, defaults):
    values = tuple(defaults if values is None else values)
    if (not values or len(set(values)) != len(values)
            or any(isinstance(v, bool) or not isinstance(v, int) or v not in defaults for v in values)):
        raise RCOFormatError(f"RCO widths must be unique integers from {defaults}")
    return values


def adapter_from_settings(method, settings, labels):
    from .config import AWQConfig, EXL3Config, GGUFConfig

    if not labels or any(not isinstance(label, str) for label in labels):
        raise RCOFormatError("RCO option labels must be nonempty strings")
    config_cls = {"awq": AWQConfig, "exl3": EXL3Config, "gguf": GGUFConfig}.get(method)
    if config_cls is None:
        raise RCOFormatError(f"Unsupported RCO method: {method}")
    if method == "gguf":
        return adapter_for(config_cls(bits=labels[0], **settings), options=labels)
    prefix = "w" if method == "awq" else "k"
    try:
        if any(not label.startswith(prefix) for label in labels):
            raise ValueError
        widths = tuple(int(label[1:]) for label in labels)
    except (ValueError, TypeError) as exc:
        raise RCOFormatError("Option labels do not match the RCO method") from exc
    return adapter_for(config_cls(bits=widths[0], **settings), options=widths)


def adapter_for(qcfg: Any, *, options=None) -> RCOFormatAdapter:
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
    return adapter(qcfg, options)


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
        except AttributeError as exc:
            raise RCOFormatError(f"Declared RCO projection is missing: {name}") from exc
        in_features = getattr(module, "in_features", None)
        out_features = getattr(module, "out_features", None)
        if in_features is None or out_features is None:
            raise RCOFormatError(f"RCO projection has no matrix dimensions: {name}")
        weights.append(float(int(in_features) * int(out_features)))
        kept.append(str(name))
    if not kept:
        raise RCOFormatError("RCO found no quantisable projection to allocate")
    return torch.tensor(weights, dtype=torch.float64), tuple(kept)


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
    if reference.ndim != 2 or reference.shape != dequantized.shape:
        raise RCOFormatError("RCO reference and reconstruction must have identical matrix shapes")
    reference = reference.detach().to(torch.float64)
    error = dequantized.detach().to(device=reference.device, dtype=torch.float64) - reference
    if hessian is None:
        numerator = error.pow(2).sum()
        denominator = reference.pow(2).sum().clamp_min(1e-30)
    else:
        h = hessian.detach().to(device=reference.device, dtype=torch.float64)
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
    if not math.isfinite(value) or value < 0:
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

    Sixty-four buckets per module bounds aggregate cost-rounding error by
    ``0.5 / 64`` bits per parameter before the memory cap is applied. The
    discretized optimum can differ from the exact-budget optimum; allocation
    repairs any overspend after solving.
    """
    return max(256, 64 * max(1, int(module_count)))


#: Upper bound on the DP's ``choice`` table.
_DP_TABLE_BUDGET_BYTES = 1 << 30


def _achieved(assignment, weights, costs):
    weights = torch.as_tensor(weights, dtype=torch.float64, device="cpu")
    costs = torch.as_tensor(costs, dtype=torch.float64, device="cpu")
    assignment = torch.as_tensor(assignment, dtype=torch.long, device="cpu")
    selected = costs[assignment] if costs.ndim == 1 else costs[torch.arange(len(assignment)), assignment]
    return float((selected * weights).sum() / weights.sum())


def repair_overspend(assignment, scores, weights, costs, target_bits):
    """Downgrade until the float64 budget holds; each row may need several steps."""
    device = assignment.device
    assignment = assignment.detach().cpu().clone()
    scores, weights, costs = (tensor.detach().cpu().double() for tensor in (scores, weights, costs))
    if costs.ndim == 1:
        costs = costs.expand(len(assignment), -1)
    for _ in range(len(assignment) * (costs.shape[1] - 1)):
        if _achieved(assignment, weights, costs) <= target_bits:
            break
        current_costs = costs.gather(1, assignment[:, None])
        saved = (current_costs - costs) * weights[:, None]
        losses = scores.gather(1, assignment[:, None]) - scores
        ratios = torch.where(saved > 0, losses / saved.clamp_min(1e-300), torch.inf)
        best_ratio = ratios.min()
        if not torch.isfinite(best_ratio):
            break
        # On equal score loss per bit, take the largest saving deterministically.
        choices = torch.where(ratios == best_ratio, saved, -torch.inf)
        flat = int(choices.argmax())
        row, option = divmod(flat, costs.shape[1])
        assignment[row] = option
    return assignment.to(device)


def _validate_inputs(scores, weights, costs, target, names):
    if not names or len(set(names)) != len(names) or any(not isinstance(name, str) or not name for name in names):
        raise RCOFormatError("RCO module names must be nonempty and unique")
    if scores.ndim != 2 or scores.shape[0] != len(names) or costs.shape != scores.shape:
        raise RCOFormatError("RCO scores/costs must have one row per module and one column per option")
    if weights.shape != (len(names),) or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise RCOFormatError("RCO parameter weights must be finite and positive")
    if not torch.isfinite(scores).all() or not torch.isfinite(costs).all() or (costs <= 0).any():
        raise RCOFormatError("RCO scores and positive costs must be finite")
    if not math.isfinite(target) or target <= 0:
        raise RCOFormatError("RCO target must be finite and positive")
    if _achieved(costs.argmin(1), weights, costs) > target:
        raise RCOFormatError("RCO target is below the cheapest feasible assignment")


def allocate(adapter, scores, weights, target_bits, *, names, resolution=None,
             shapes=None, score_metric="relative_frobenius") -> RCOPlan:
    """Solve a discretized reconstruction allocation, then enforce the exact budget."""
    if score_metric not in {"relative_frobenius", "relative_hessian"}:
        raise RCOFormatError("RCO requires an explicit reconstruction score metric")
    options = adapter.options()
    names = tuple(names)
    shapes = dict(shapes or {})
    if shapes and set(shapes) != set(names):
        raise RCOFormatError("RCO shapes must cover every module")
    scores, weights = scores.detach().cpu().double(), weights.detach().cpu().double()
    nominal = adapter.costs()
    costs = adapter.cost_matrix([shapes[name] for name in names]) if shapes else nominal.expand(len(names), -1)
    if scores.shape != (len(names), len(options)):
        raise RCOFormatError("RCO score columns must match the adapter options")
    _validate_inputs(scores, weights, costs, target_bits, names)
    if shapes and any(float(weights[i]) != math.prod(shapes[name]) for i, name in enumerate(names)):
        raise RCOFormatError("RCO weights must equal each matrix's parameter count")
    if resolution is not None and (isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 1):
        raise RCOFormatError("RCO resolution must be a positive integer")
    resolution = default_resolution(len(names)) if resolution is None else resolution
    maximum = int(_DP_TABLE_BUDGET_BYTES / (8.8 * len(names) * target_bits)) - 1
    if maximum < 1:
        raise RCOFormatError("RCO DP exceeds the memory budget")
    resolution = min(resolution, maximum)
    cheapest = costs.argmin(1)
    try:
        assignment = budget_constrained_argmax(scores, weights, costs, target_bits, resolution=resolution)
    except RuntimeError as exc:
        if "no feasible assignment" not in str(exc):
            raise
        # Rounding can make a feasible boundary budget unreachable on this grid.
        assignment = cheapest
    if _achieved(assignment, weights, costs) > target_bits:
        assignment = repair_overspend(assignment, scores, weights, costs, target_bits)
    achieved = _achieved(assignment, weights, costs)
    if achieved > target_bits:
        raise RCOFormatError(f"RCO allocation overspent the budget: {achieved} > {target_bits}")
    labels = adapter.labels()
    counts = {label: sum(labels[int(index)] == label for index in assignment) for label in labels}
    counts = {label: count for label, count in counts.items() if count}
    index_map = dict(zip(names, assignment.tolist()))
    return RCOPlan(
        method=adapter.method.value, search="reconstruction", target_bits=float(target_bits),
        achieved_bits=achieved, names=names, labels=labels, costs=tuple(nominal.tolist()),
        assignment=tuple(assignment.tolist()), dynamic=adapter.build_dynamic(names, assignment), counts=counts,
        resolution=resolution, runtime_note=adapter.runtime_requirement(index_map), settings=adapter.settings(),
        weights=tuple(weights.tolist()), module_costs=tuple(tuple(row) for row in costs.tolist()),
        shapes=shapes, score_metric=score_metric,
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
    "adapter_from_settings",
    "allocate",
    "awq_runtime_requirement",
    "default_resolution",
    "discover_projections",
    "module_weights",
    "reconstruction_scores",
    "relative_reconstruction_error",
    "repair_overspend",
]
