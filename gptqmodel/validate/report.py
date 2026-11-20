# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The result of one validation run, rendered for humans and for CI."""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


__all__ = ["ValidationReport"]


@dataclass(frozen=True)
class ValidationReport:
    """Divergence numbers for one quantized model against its reference.

    ``passed`` is ``None`` when no thresholds were configured, and that is
    deliberate: a report must never read as "validated" when nothing was
    checked against a budget.
    """

    values: Dict[str, Any]
    batches: int = 0
    positions: int = 0
    thresholds: Mapping[str, float] = field(default_factory=dict)

    @property
    def violations(self) -> Dict[str, float]:
        """Metrics that broke their threshold, with the observed value.

        PPL, KLD and RMSE are upper bounds; TOP_N is a floor, because "more of
        the reference's top-k survived" is better.
        """

        broken: Dict[str, float] = {}
        for name, threshold in self.thresholds.items():
            observed = self._observed(name)
            if observed is None:
                continue
            if name == "TOP_N":
                if observed < threshold:
                    broken[name] = observed
            elif observed > threshold:
                broken[name] = observed
        return broken

    def _observed(self, name: str):
        value = self.values.get(name)
        if isinstance(value, dict):
            # A multi-width TOP_N passes only when every requested width clears
            # the floor; report the worst one.
            return min(value.values()) if value else None
        return value

    @property
    def passed(self) -> Optional[bool]:
        if not self.thresholds:
            return None
        return not self.violations

    def to_json(self) -> Dict[str, Any]:
        return {
            "values": self.values,
            "batches": self.batches,
            "positions": self.positions,
            "thresholds": dict(self.thresholds),
            "violations": dict(self.violations),
            "passed": self.passed,
        }

    def markdown(self) -> str:
        """A table a user can paste into a report or a PR."""

        lines = ["| metric | value | threshold | ok |", "| --- | --- | --- | --- |"]
        for name, value in self.values.items():
            threshold = self.thresholds.get(name)
            if isinstance(value, dict):
                for width, width_value in value.items():
                    lines.append(
                        f"| {name}@{width} | {_format(width_value)} | "
                        f"{_format(threshold) if threshold is not None else '-'} | "
                        f"{_ok(name, width_value, threshold)} |"
                    )
                continue
            lines.append(
                f"| {name} | {_format(value)} | "
                f"{_format(threshold) if threshold is not None else '-'} | "
                f"{_ok(name, value, threshold)} |"
            )
        if self.batches or self.positions:
            lines.append("")
            lines.append(f"measured on {self.batches} batch(es), {self.positions} position(s)")
        verdict = {None: "no thresholds configured", True: "within thresholds", False: "outside thresholds"}
        lines.append(f"result: {verdict[self.passed]}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.markdown()


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _ok(name: str, value: Any, threshold: Optional[float]) -> str:
    if threshold is None:
        return "-"
    within = value >= threshold if name == "TOP_N" else value <= threshold
    return "yes" if within else "**no**"
