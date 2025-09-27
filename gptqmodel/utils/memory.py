# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations
import threading
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn

Obj = nn.Module | torch.Tensor  # Python 3.10+ union syntax


class MemTracker:
    """
    Tracks memory attributed to Modules or Tensors by **device instance** (e.g., 'cuda:0', 'cuda:1', 'cpu').
    Query with a torch.device to aggregate by **type** (torch.device('cuda')) or a specific index
    (torch.device('cuda:1')).
    """

    def __init__(self) -> None:
        self._allocated_by_dev: Dict[str, int] = {}
        self._freed_by_dev: Dict[str, int] = {}
        self._lock = threading.Lock()

    # ---------- Public API ----------
    def allocate(self, ob: Obj) -> None:
        sizes = self._sizes_by_device_instance(ob)
        with self._lock:
            for dev_key, b in sizes.items():
                self._allocated_by_dev[dev_key] = self._allocated_by_dev.get(dev_key, 0) + b

    def free(self, ob: Obj) -> None:
        sizes = self._sizes_by_device_instance(ob)
        with self._lock:
            for dev_key, b in sizes.items():
                if b <= 0:
                    continue
                self._allocated_by_dev[dev_key] = max(0, self._allocated_by_dev.get(dev_key, 0) - b)
                self._freed_by_dev[dev_key] = self._freed_by_dev.get(dev_key, 0) + b

    def reset(self) -> None:
        with self._lock:
            self._allocated_by_dev.clear()
            self._freed_by_dev.clear()

    def allocated(self, device: torch.device | None = None) -> Tuple[int, str]:
        """Return (raw_bytes, formatted_string) for allocated memory."""
        with self._lock:
            if device is None:
                val = sum(self._allocated_by_dev.values())
            else:
                val = _sum_for_device(self._allocated_by_dev, device)
        return val, format_bytes(val)

    def freed(self, device: torch.device | None = None) -> Tuple[int, str]:
        """Return (raw_bytes, formatted_string) for freed memory."""
        with self._lock:
            if device is None:
                val = sum(self._freed_by_dev.values())
            else:
                val = _sum_for_device(self._freed_by_dev, device)
        return val, format_bytes(val)

    # ---------- Helpers ----------
    def _sizes_by_device_instance(self, ob: Obj) -> Dict[str, int]:
        tensors = list(self._gather_tensors(ob))
        return self._sum_by_devkey_dedup(tensors)

    def _gather_tensors(self, ob: Obj) -> Iterable[torch.Tensor]:
        if isinstance(ob, torch.Tensor):
            yield ob
            return
        for p in ob.parameters(recurse=True):
            yield p.data
        for b in ob.buffers(recurse=True):
            yield b

    def _sum_by_devkey_dedup(self, tensors: Iterable[torch.Tensor]) -> Dict[str, int]:
        seen_keys: set[tuple[int, int]] = set()
        by_dev: Dict[str, int] = {}

        def _accumulate_dense(t: torch.Tensor) -> None:
            if not isinstance(t, torch.Tensor):
                return
            dev = t.device
            if dev.type == "meta":
                return
            dev_key = str(dev)  # e.g., 'cuda:0', 'cpu'
            try:
                st = t.untyped_storage()
                key = (st.data_ptr(), st.nbytes())
                if key in seen_keys:
                    return
                seen_keys.add(key)
                by_dev[dev_key] = by_dev.get(dev_key, 0) + int(st.nbytes())
            except RuntimeError:
                nbytes = int(t.numel() * t.element_size())
                key = (t.data_ptr(), nbytes)
                if key in seen_keys:
                    return
                seen_keys.add(key)
                by_dev[dev_key] = by_dev.get(dev_key, 0) + nbytes

        for t in tensors:
            if t.is_sparse:
                _accumulate_dense(t.indices())
                _accumulate_dense(t.values())
            elif t.layout == torch.sparse_csr:
                _accumulate_dense(t.crow_indices())
                _accumulate_dense(t.col_indices())
                _accumulate_dense(t.values())
            elif getattr(torch, "sparse_csc", None) is not None and t.layout == torch.sparse_csc:
                _accumulate_dense(t.ccol_indices())
                _accumulate_dense(t.row_indices())
                _accumulate_dense(t.values())
            else:
                _accumulate_dense(t)

        return by_dev


def _sum_for_device(table: Dict[str, int], device: torch.device) -> int:
    dev_type = device.type
    idx = device.index
    if idx is None:
        total = 0
        prefix = f"{dev_type}:"
        for k, v in table.items():
            if k == dev_type or k.startswith(prefix):
                total += v
        return total
    else:
        key = f"{dev_type}:{idx}"
        return table.get(key, 0)


# ---------- Optional utility ----------
def format_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0
