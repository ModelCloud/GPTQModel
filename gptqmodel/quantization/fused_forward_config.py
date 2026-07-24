# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Configuration group for fusing same-input linear forward passes."""

from dataclasses import asdict, dataclass


@dataclass
class FusedForwardConfig:
    """Control how same-input module groups (q/k/v, gate/up) are fused during calibration.

    Fusion is enabled whenever a ``FusedForwardConfig`` instance is supplied to
    ``QuantizeConfig.fused_forward``; ``None`` disables it.
    """

    splice: str = "view"

    def __post_init__(self):
        if self.splice not in {"view", "contiguous_copy"}:
            raise ValueError(
                f"FusedForwardConfig.splice must be 'view' or 'contiguous_copy', got {self.splice!r}"
            )

    def to_dict(self) -> dict:
        return asdict(self)
