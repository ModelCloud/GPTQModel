# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# """ANSI color utilities for consistent console formatting across the project."""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union

ANSI_RESET = "\033[0m"


class ANSIColor(str, Enum):
    BRIGHT_GREEN = "\033[38;5;76m"  # softer bright green
    GREEN = "\033[32m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    ORANGE = "\033[33m"
    RED = "\033[31m"
    BRIGHT_RED = "\033[91m"


ColorLike = Optional[Union["ANSIColor", str]]


def resolve_color_code(color: ColorLike) -> str:
    if color is None:
        return ""
    if isinstance(color, ANSIColor):
        return color.value
    return str(color)


def color_text(text: str, color: ColorLike) -> str:
    if not text:
        return text
    color_code = resolve_color_code(color)
    if not color_code:
        return text
    return f"{color_code}{text}{ANSI_RESET}"


__all__ = ["ANSI_RESET", "ANSIColor", "ColorLike", "color_text", "resolve_color_code"]
