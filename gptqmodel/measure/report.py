# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The result of one measurement run, rendered for humans and for CI."""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from ..utils.logger import render_table
from .config import MetricGrades


__all__ = ["MeasurementReport"]


@dataclass(frozen=True)
class MeasurementReport:
    """Divergence numbers and their grades for one model against its reference.

    ``overall`` is ``None`` when no metric carried a ``GradedThresholdConfig``:
    a report must never read as "passed" when nothing was checked.
    """

    values: Dict[str, Any]
    grades: Dict[str, MetricGrades] = field(default_factory=dict)
    reference: Mapping[str, Any] = field(default_factory=dict)
    batches: int = 0
    positions: int = 0

    @property
    def overall(self) -> Optional[MetricGrades]:
        if not self.grades:
            return None
        return max(self.grades.values(), key=lambda grade: grade.severity)

    @property
    def passed(self) -> Optional[bool]:
        overall = self.overall
        return None if overall is None else overall is MetricGrades.PASS

    @property
    def violations(self) -> Dict[str, Any]:
        return {name: self.values.get(name) for name, grade in self.grades.items()
                if grade is MetricGrades.FAIL}

    @property
    def warnings(self) -> Dict[str, Any]:
        return {name: self.values.get(name) for name, grade in self.grades.items()
                if grade is MetricGrades.WARN}

    def to_json(self) -> Dict[str, Any]:
        return {
            "values": self.values,
            "grades": {name: grade.value for name, grade in self.grades.items()},
            "reference": dict(self.reference),
            "batches": self.batches,
            "positions": self.positions,
            "overall": None if self.overall is None else self.overall.value,
            "passed": self.passed,
        }

    def markdown(self) -> str:
        """A table a user can paste into a report or a PR."""

        has_reference = bool(self.reference)
        header = "| metric | value | grade |" if not has_reference else "| metric | value | reference | grade |"
        rule = "| --- | --- | --- |" if not has_reference else "| --- | --- | --- | --- |"
        lines = [header, rule]

        for name, value in self.values.items():
            grade = self.grades.get(name)
            if isinstance(value, dict):
                for width, width_value in value.items():
                    columns = f"| {name}@{width} | {_format(width_value)} | "
                    if has_reference:
                        columns += "- | "
                    lines.append(columns + f"{_grade(grade)} |")
                continue
            columns = f"| {name} | {_format(value)} | "
            if has_reference:
                reference = self.reference.get(name)
                columns += f"{_format(reference) if reference is not None else '-'} | "
            lines.append(columns + f"{_grade(grade)} |")

        if self.batches or self.positions:
            lines.append("")
            lines.append(f"measured on {self.batches} batch(es), {self.positions} position(s)")
        lines.append(f"result: {_verdict(self.overall)}")
        return "\n".join(lines)

    def render(self, logger: Any = None) -> str:
        """A fixed-column table for a terminal or a TUI.

        ``markdown()`` is the form to paste into a PR; this is what ``print(report)``
        and ``str(report)`` show. ``render_table`` sizes the columns through
        LogBar, so values line up whatever their width.
        """

        has_reference = bool(self.reference)
        headers = ["metric", "value", "reference", "grade"] if has_reference else ["metric", "value", "grade"]
        rows = []

        for name, value in self.values.items():
            grade = _grade(self.grades.get(name))
            if isinstance(value, dict):
                for width, width_value in value.items():
                    row = [f"{name}@{width}", _format(width_value)]
                    if has_reference:
                        row.append("-")
                    rows.append(row + [grade])
                continue
            row = [name, _format(value)]
            if has_reference:
                reference = self.reference.get(name)
                row.append(_format(reference) if reference is not None else "-")
            rows.append(row + [grade])

        lines = [render_table(rows, headers=headers, tablefmt="simple", logger=logger)]
        if self.batches or self.positions:
            lines.append("")
            lines.append(f"measured on {self.batches} batch(es), {self.positions} position(s)")
        lines.append(f"result: {_verdict(self.overall)}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _grade(grade: Optional[MetricGrades]) -> str:
    return "-" if grade is None else grade.value


def _verdict(grade: Optional[MetricGrades]) -> str:
    return {
        None: "no thresholds configured",
        MetricGrades.PASS: "pass",
        MetricGrades.WARN: "warn",
        MetricGrades.FAIL: "fail",
    }[grade]
