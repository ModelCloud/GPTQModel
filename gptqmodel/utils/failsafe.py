# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Any, Optional, Tuple

from gptqmodel.quantization.config import FailSafe, FailSafeStrategy


def _parse_threshold(setting: Any) -> Tuple[Optional[float], bool]:
    """
    Returns (threshold_value, is_percent).
    Percent values are returned as the fractional percentage (e.g., "5%" -> 5.0).
    """
    if isinstance(setting, str):
        stripped = setting.strip()
        if stripped.endswith("%"):
            try:
                val = float(stripped[:-1])
                return val, True
            except ValueError:
                return None, False
        try:
            return float(stripped), False
        except ValueError:
            return None, False

    if isinstance(setting, (int, float)):
        return float(setting), False

    return None, False


def resolve_failsafe_strategy(strategy: Any) -> FailSafeStrategy:
    """
    Normalize a failsafe strategy; auto currently resolves to midpoint.
    """
    if isinstance(strategy, FailSafe):
        strategy = strategy.strategy
    if isinstance(strategy, dict):
        strategy = strategy.get("strategy", FailSafeStrategy.AUTO)
    if strategy is None:
        resolved = FailSafeStrategy.AUTO
    elif isinstance(strategy, FailSafeStrategy):
        resolved = strategy
    elif isinstance(strategy, str):
        normalized = strategy.strip().lower()
        try:
            resolved = FailSafeStrategy(normalized)
        except ValueError:
            resolved = FailSafeStrategy.AUTO
    else:
        resolved = FailSafeStrategy.AUTO

    if resolved == FailSafeStrategy.AUTO:
        return FailSafeStrategy.RTN

    return resolved


def should_use_failsafe(
    setting: Any,
    observed_samples: float,
    expected_total_samples: Optional[float] = None,
) -> bool:
    if isinstance(setting, FailSafe):
        setting = setting.threshold
    if isinstance(setting, dict):
        setting = setting.get("threshold", None)
    threshold_value, _ = resolve_threshold(setting, expected_total_samples)
    if threshold_value is None:
        return False
    return observed_samples < threshold_value


def resolve_threshold(
    setting: Any,
    expected_total_samples: Optional[float] = None,
) -> Tuple[Optional[float], bool]:
    """
    Resolve a threshold into a raw numeric value and whether it was percent-based.
    """
    if isinstance(setting, FailSafe):
        setting = setting.threshold
    if isinstance(setting, dict):
        setting = setting.get("threshold", None)
    if not setting:
        return None, False

    if setting is True:
        # Tiny positive epsilon so 0 triggers but positive counts do not when using `<`.
        return 1e-9, False

    threshold, is_percent = _parse_threshold(setting)
    if threshold is None:
        return None, False

    if is_percent:
        base = expected_total_samples if expected_total_samples else 1.0
        threshold_value = (threshold / 100.0) * float(base)
    else:
        threshold_value = threshold

    return threshold_value, is_percent
