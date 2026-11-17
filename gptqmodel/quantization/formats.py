# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Target representations, independent of the algorithm used to solve them."""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class MXFP4:
    name: ClassVar[str] = "mxfp4"
    block_size: ClassVar[int] = 32
    layout: ClassVar[str] = "row_major_low_nibble_first"

    def to_dict(self):
        return {"name": self.name, "block_size": self.block_size, "element": "e2m1",
                "scale": "e8m0", "layout": self.layout}


@dataclass(frozen=True)
class MXFP8:
    element: str = "e4m3"
    name: ClassVar[str] = "mxfp8"

    def __post_init__(self):
        if self.element not in ("e4m3", "e5m2"):
            raise ValueError("MXFP8 element must be e4m3 or e5m2")

    def to_dict(self):
        return {"name": self.name, "element": self.element, "block_size": 32, "scale": "e8m0"}
