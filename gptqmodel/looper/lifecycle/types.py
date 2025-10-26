from __future__ import annotations

from typing import NamedTuple, Optional


class FinalizeProgressInfo(NamedTuple):
    module_label: Optional[str]
    process_name: str
    layer_idx: Optional[int]


__all__ = ["FinalizeProgressInfo"]
