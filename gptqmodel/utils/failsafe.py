# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Any, Optional, Tuple


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


def should_use_rtn_failsafe(
    setting: Any,
    observed_samples: float,
    expected_total_samples: Optional[float] = None,
) -> bool:
    """
    Determine whether failsafe RTN should activate.

    Rules:
    - False/None: never trigger.
    - True: trigger only when observed <= 0 (preserves legacy behavior).
    - Numeric (int/float): trigger when observed <= numeric threshold.
    - String with '%' suffix: interpret as percentage; trigger when observed <=
      percent/100 * expected_total_samples (if available), otherwise percent/100
      of 1 token for safety.
    """
    if not setting:
        return False

    if setting is True:
        return observed_samples <= 0

    threshold, is_percent = _parse_threshold(setting)
    if threshold is None:
        return False

    if is_percent:
        base = expected_total_samples if expected_total_samples else 1.0
        threshold_value = (threshold / 100.0) * float(base)
    else:
        threshold_value = threshold

    return observed_samples <= threshold_value
