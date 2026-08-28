from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

from ..analysis import AnalysisSelection, QuantizationAnalyzer, _layer_index, _module_role, _weight_matrix
from ..config import quant_bits_width


def _rate_label(rate: Any) -> str:
    return str(float(rate)).rstrip("0").rstrip(".")


class SensitivityProfiler:
    """Dense-model sensitivity scanner with activation-weighted rate curves.

    The profiler deliberately keeps activation statistics diagonal and mergeable:
    ``sum(x**2)`` is enough to estimate candidate output damage without storing
    token activations.  It can therefore scan large dense models in balanced mode
    while retaining the existing weight-only analyzer as the fast fallback.
    """

    def __init__(self, quantize_config, *, candidate_rates: Sequence[Any] = (2, 2.5, 3, 3.5, 4), device="cpu"):
        self.qcfg = quantize_config
        self.candidate_rates = tuple(candidate_rates)
        self.device = torch.device(device)
        self.analyzer = QuantizationAnalyzer(quantize_config, compute_device=self.device)
        self._activation: dict[str, dict[str, Any]] = defaultdict(lambda: {"hessian": None, "max": None, "tokens": 0, "sequences": 0})
        self._hooks: list[Any] = []

    def collect_activations(self, model: nn.Module, batches: Iterable[Any], *, max_batches: int | None = None) -> dict[str, Any]:
        """Collect input-channel diagonal Hessian and activation tails in one dense pass."""
        modules = {name: m for name, m in model.named_modules() if isinstance(m, nn.Linear)}

        seen_this_batch: set[str] = set()
        def hook(name):
            def capture(module, args, kwargs):
                if not args or not torch.is_tensor(args[0]):
                    return
                x = args[0].detach()
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                original_shape = x.shape
                x = x.reshape(-1, x.shape[-1]).to(self.device, dtype=torch.float32)
                mask = kwargs.get("attention_mask") if isinstance(kwargs, Mapping) else None
                if torch.is_tensor(mask) and tuple(mask.shape) == tuple(original_shape[:-1]):
                    mask = mask.reshape(-1).to(self.device, dtype=torch.bool)
                    x = x[mask]
                finite = torch.isfinite(x).all(dim=1)
                x = x[finite]
                if not x.numel():
                    return
                state = self._activation[name]
                h = (x * x).sum(dim=0)
                state["hessian"] = h if state["hessian"] is None else state["hessian"] + h
                amax = x.abs().amax(dim=0)
                state["max"] = amax if state["max"] is None else torch.maximum(state["max"], amax)
                state["tokens"] += int(x.shape[0])
                seen_this_batch.add(name)
            return capture

        self._hooks = [module.register_forward_pre_hook(hook(name), with_kwargs=True) for name, module in modules.items()]
        model_was_training = model.training
        model.eval()
        with torch.inference_mode():
            for index, batch in enumerate(batches):
                if max_batches is not None and index >= max_batches:
                    break
                if isinstance(batch, Mapping):
                    batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                    try:
                        model(**batch)
                    except TypeError:
                        # Small adapter models (and unit-test sequential models)
                        # often expose only positional forward(input).
                        model(batch.get("input_ids", next(iter(batch.values()))))
                else:
                    value = batch.to(self.device) if torch.is_tensor(batch) else batch
                    model(value)
                for name in seen_this_batch:
                    self._activation[name]["sequences"] += 1
                seen_this_batch.clear()
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        if model_was_training:
            model.train()
        return self.activation_summary()

    def activation_summary(self) -> dict[str, Any]:
        out = {}
        for name, state in self._activation.items():
            h = state["hessian"]
            if h is None:
                continue
            total = float(h.sum().item())
            sorted_h = torch.sort(h, descending=True).values
            top = max(1, int(math.ceil(h.numel() * 0.01)))
            out[name] = {
                "tokens": state["tokens"], "sequences": state["sequences"],
                "hessian_l2": total, "hessian_top1pct_mass": float(sorted_h[:top].sum().item() / max(total, 1e-12)),
                "activation_max": float(state["max"].max().item()),
                "activation_p99_proxy": float(torch.quantile(state["max"], 0.99).item()),
            }
        return out

    def scan(self, model: nn.Module, *, batches: Iterable[Any] | None = None, max_batches: int | None = None,
             mode: str = "balanced", selection: AnalysisSelection | None = None) -> dict[str, Any]:
        if mode not in {"fast", "balanced"}:
            raise ValueError("mode must be fast or balanced; deep probes are opt-in and not silently approximated")
        if batches is not None and mode == "balanced":
            self.collect_activations(model, batches, max_batches=max_batches)
        report = self.analyzer.analyze_model(model, selection=selection)
        records = []
        for record in report["records"]:
            name = record["module"]
            module = dict(model.named_modules())[name]
            weight = _weight_matrix(module).to(self.device)
            activation = self._activation.get(name)
            h = activation.get("hessian") if activation else None
            curves = {}
            for rate in self.candidate_rates:
                representation, damage = self._candidate_damage(weight, float(rate), record["group_size"], record["sym"], h)
                curves[_rate_label(rate)] = {"rate": float(rate), "representation_error": representation,
                                             "activation_error": damage, "candidate_kind": "rtn_proxy"}
            confidence = 100.0 if not activation else min(100.0, 100.0 * math.log1p(activation["tokens"]) / math.log1p(262144))
            risk = max(float(v["activation_error"]) for v in curves.values()) if curves else float(record["risk_score"])
            record = dict(record)
            record.update({"depth": (record["layer"] / max(1, self._layer_count(model) - 1)) if record.get("layer") is not None else None,
                           "activation": activation or {"tokens": 0, "sequences": 0},
                           "rates": curves, "sensitivity_score": round(min(100.0, risk * 100.0), 2),
                           "confidence_score": round(confidence, 2)})
            record["priority_score"] = round(min(100.0, 0.5 * record["sensitivity_score"] + 0.5 * record["risk_score"]), 2)
            records.append(record)
        report["schema_version"] = "2.0"
        report["stage"] = "pre_quantization_sensitivity"
        report["mode"] = mode
        report["records"] = records
        report["activation_summary"] = self.activation_summary()
        report["planner"] = self.plan(records)
        report["limitations"] = report.get("limitations", []) + [
            "Balanced mode uses diagonal activation statistics; full Fisher/KFAC and shadow probes are not implied.",
            "MoE routing fields require a model adapter and are reported conservatively until expert coverage is observed.",
        ]
        return report

    @staticmethod
    def _layer_count(model: nn.Module) -> int:
        values = [_layer_index(n) for n, _ in model.named_modules()]
        return max((v for v in values if v is not None), default=0) + 1

    def _candidate_damage(self, weight, rate: float, group_size: int, sym: bool, hessian):
        """Return representation and channel-weighted damage for a rate.

        Fractional QVQ rates are retained as candidates. Until a native QVQ
        reconstruction callback is supplied, half-rates interpolate adjacent
        integer RTN probes and are marked ``rtn_proxy`` in the report.
        """
        lo = max(1, int(math.floor(rate))); hi = max(lo, int(math.ceil(rate)))
        def one(width):
            rows, cols = weight.shape
            qmax = (2 ** (width - 1) - 1) if sym else (2 ** width - 1)
            qmin = -qmax if sym else 0
            eps = torch.finfo(torch.float32).eps
            w = weight.float()
            step = cols if group_size <= 0 else group_size
            q = torch.empty_like(w)
            for start in range(0, cols, step):
                block = w[:, start:min(start + step, cols)]
                if sym:
                    scale = torch.amax(block.abs(), dim=1, keepdim=True).clamp_min(eps) / max(qmax, 1)
                    q[:, start:start + block.shape[1]] = torch.round(block / scale).clamp(qmin, qmax) * scale
                else:
                    mn, mx = block.amin(1, keepdim=True), block.amax(1, keepdim=True)
                    scale = (mx - mn).clamp_min(eps) / max(qmax, 1)
                    zero = torch.round(-mn / scale).clamp(qmin, qmax)
                    q[:, start:start + block.shape[1]] = (torch.round(block / scale + zero).clamp(qmin, qmax) - zero) * scale
            diff2 = (w - q).square(); signal = w.square()
            if hessian is not None and hessian.numel() == cols:
                weights = hessian.to(w.device, dtype=w.dtype).reshape(1, -1)
                num = float((diff2 * weights).sum().item())
                den = float((signal * weights).sum().item())
            else:
                num, den = float(diff2.sum().item()), float(signal.sum().item())
            return math.sqrt(num / max(den, eps)), num / max(den, eps)
        a, da = one(lo)
        if hi == lo:
            return a, da
        b, db = one(hi); alpha = rate - lo
        return a + alpha * (b - a), da + alpha * (db - da)

    def plan(self, records: Sequence[dict[str, Any]], *, target_bpw: float | None = None) -> dict[str, Any]:
        """Greedy marginal-recovery plan; callers can replace it with SLQ ILP."""
        base = float(self.qcfg.bits)
        target = float(target_bpw) if target_bpw is not None else base
        selected = {r["module"]: base for r in records}
        total_numel = max(1, sum(int(r.get("numel", 0)) for r in records))
        budget = max(0.0, target - base)
        if target > base and records:
            candidates = []
            for r in records:
                curves = r.get("rates", {})
                for low, high in zip(self.candidate_rates, self.candidate_rates[1:]):
                    a, b = curves.get(_rate_label(low), {}), curves.get(_rate_label(high), {})
                    if a and b and float(high) > float(low):
                        gain = float(a.get("activation_error", 0)) - float(b.get("activation_error", 0))
                        weight_fraction = int(r.get("numel", 0)) / total_numel
                        candidates.append((gain / max(weight_fraction * float(high - low), 1e-12), r["module"], float(low), float(high), weight_fraction * float(high - low)))
            candidates.sort(reverse=True)
            for _, name, low, rate, weighted_delta in candidates:
                if budget <= 0:
                    break
                # Only permit adjacent transitions; never price W2 -> W4 as one step.
                if abs(selected[name] - low) < 1e-6 and weighted_delta <= budget:
                    selected[name] = rate
                    budget -= weighted_delta
        return {"target_bpw": target, "base_bpw": base, "assignments": selected,
                "budget_units": "parameter-weighted bits/weight", "remaining_budget": max(0.0, budget)}
