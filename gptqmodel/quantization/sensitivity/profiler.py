from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import nn

from ..analysis import AnalysisSelection, QuantizationAnalyzer, _layer_index, _module_role, _weight_matrix


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

        def hook(name):
            def capture(module, args):
                if not args or not torch.is_tensor(args[0]):
                    return
                x = args[0].detach()
                if x.ndim == 1:
                    x = x.reshape(1, -1)
                x = x.reshape(-1, x.shape[-1]).to(self.device, dtype=torch.float32)
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
            return capture

        self._hooks = [module.register_forward_pre_hook(hook(name)) for name, module in modules.items()]
        model_was_training = model.training
        model.eval()
        with torch.inference_mode():
            for index, batch in enumerate(batches):
                if max_batches is not None and index >= max_batches:
                    break
                if isinstance(batch, Mapping):
                    batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                    model(**batch)
                else:
                    value = batch.to(self.device) if torch.is_tensor(batch) else batch
                    model(value)
                for state in self._activation.values():
                    state["sequences"] += 1
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
                bits = int(round(float(rate)))
                stats, _ = self.analyzer._analyze_weight(weight, bit_width=bits, group_size=record["group_size"], sym=record["sym"])
                damage = float(stats["rel_rmse"] ** 2)
                if h is not None and h.numel() == weight.shape[1]:
                    # Diagonal activation-weighted proxy; exact Q(W) error is
                    # represented by the measured RTN relative error curve.
                    channel_mass = float(h.sum().item()) / max(float(h.numel()), 1.0)
                    damage *= channel_mass
                curves[_rate_label(rate)] = {"representation_error": stats["rel_rmse"], "activation_error": damage}
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

    def plan(self, records: Sequence[dict[str, Any]], *, target_bpw: float | None = None) -> dict[str, Any]:
        """Greedy marginal-recovery plan; callers can replace it with SLQ ILP."""
        base = float(self.qcfg.bits)
        target = float(target_bpw) if target_bpw is not None else base
        selected = {r["module"]: base for r in records}
        if target > base and records:
            candidates = []
            for r in records:
                curves = r.get("rates", {})
                for low, high in zip(self.candidate_rates, self.candidate_rates[1:]):
                    a, b = curves.get(_rate_label(low), {}), curves.get(_rate_label(high), {})
                    if a and b and float(high) > float(low):
                        gain = float(a.get("activation_error", 0)) - float(b.get("activation_error", 0))
                        candidates.append((gain / float(high - low), r["module"], float(high)))
            candidates.sort(reverse=True)
            budget = target - base
            for _, name, rate in candidates:
                if budget <= 0:
                    break
                delta = rate - selected[name]
                if delta > 0:
                    selected[name] = rate
                    budget -= delta
        return {"target_bpw": target, "base_bpw": base, "assignments": selected}
