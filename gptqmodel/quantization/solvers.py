# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Immutable solver options. Execution state belongs to each module, not here."""

import math
from dataclasses import asdict, dataclass
from typing import ClassVar


@dataclass(frozen=True)
class GPTQ:
    damp_percent: float = 0.01
    desc_act: bool = False
    block_size: int = 128
    name: ClassVar[str] = "gptq"

    def __post_init__(self):
        if isinstance(self.damp_percent, bool) or not math.isfinite(self.damp_percent) or self.damp_percent <= 0:
            raise ValueError("GPTQ damp_percent must be finite and positive")
        if type(self.desc_act) is not bool:
            raise TypeError("GPTQ desc_act must be bool")
        if type(self.block_size) is not int or self.block_size < 1:
            raise ValueError("GPTQ block_size must be a positive integer")

    def to_dict(self):
        return {"name": self.name, **asdict(self)}


@dataclass(frozen=True)
class AWQ:
    name: ClassVar[str] = "awq"

    def to_dict(self):
        return {"name": self.name}
