# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import threading
from typing import Dict, Generator, Iterable, Tuple

import torch
import torch.nn as nn


# ---------- ANSI COLORS ----------
RESET   = "\033[0m"
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
CYAN    = "\033[96m"
MAGENTA = "\033[95m"

# ---------- DEBUG FLAG ----------
DEBUG_MODE = os.environ.get("DEBUG", "0") == "1"

def _log(msg: str) -> None:
    if DEBUG_MODE:
        print(msg)

# ---------- TYPE ALIASES ----------
Obj = nn.Module | torch.Tensor
ObjOrTuple = Obj | tuple[Obj, ...]


class MemTracker:
    def __init__(self, auto_gc_bytes: int | str | None = "auto") -> None:
        self._allocated_by_dev: Dict[torch.device, int] = {}
        self._freed_by_dev: Dict[torch.device, int] = {}

        # GC accounting
        self._gc_count_by_dev: Dict[torch.device, int] = {}
        self._gc_total_count: int = 0

        self._lock = threading.Lock()

        # Resolve threshold
        self._auto_gc_bytes: int | None = None
        self._resolve_and_set_auto_gc(auto_gc_bytes, context="init")

    # ---------- Public API ----------
    def allocate(self, ob: ObjOrTuple) -> None:
        sizes = self._sizes_for_many(ob)
        with self._lock:
            for dev, b in sizes.items():
                self._allocated_by_dev[dev] = self._allocated_by_dev.get(dev, 0) + b
                _log(f"{RED}[allocate]{RESET} +{format_bytes(b)} on {dev}")

            all_devs = self._all_known_devices_locked()
            type_totals = self._totals_by_type_locked(self._allocated_by_dev)
            type_counts = self._counts_by_type_locked()

        self._print_full_device_summary(
            header=f"{CYAN}[allocate-summary]{RESET}",
            per_device_map=self._allocated_by_dev,
            all_devices=all_devs,
            type_totals=type_totals,
            type_counts=type_counts,
        )

    def free(self, ob: ObjOrTuple) -> None:
        sizes = self._sizes_for_many(ob)
        affected: set[torch.device] = set()

        with self._lock:
            for dev, b in sizes.items():
                if b <= 0:
                    continue
                self._allocated_by_dev[dev] = max(0, self._allocated_by_dev.get(dev, 0) - b)
                self._freed_by_dev[dev] = self._freed_by_dev.get(dev, 0) + b
                affected.add(dev)
                _log(f"{GREEN}[free]{RESET} released {format_bytes(b)} on {dev}")

            all_devs = self._all_known_devices_locked()
            freed_type_totals = self._totals_by_type_locked(self._freed_by_dev)
            type_counts = self._counts_by_type_locked()

        self._print_full_device_summary(
            header=f"{CYAN}[free-summary]{RESET}",
            per_device_map=self._freed_by_dev,
            all_devices=all_devs,
            type_totals=freed_type_totals,
            type_counts=type_counts,
        )

        if self._auto_gc_bytes is not None and self._auto_gc_bytes > 0:
            for dev in affected:
                self._maybe_auto_gc(dev)

    def reset(self) -> None:
        with self._lock:
            self._allocated_by_dev.clear()
            self._freed_by_dev.clear()
            self._gc_count_by_dev.clear()
            self._gc_total_count = 0
        _log(f"{MAGENTA}[reset]{RESET} counters cleared")

    def allocated(self, device: torch.device | None = None) -> Tuple[int, str]:
        with self._lock:
            val = sum(self._allocated_by_dev.values()) if device is None else _sum_for_device(self._allocated_by_dev, device)
        _log(f"{CYAN}[allocated]{RESET} query={device}, result={format_bytes(val)}")
        return val, format_bytes(val)

    def freed(self, device: torch.device | None = None) -> Tuple[int, str]:
        with self._lock:
            val = sum(self._freed_by_dev.values()) if device is None else _sum_for_device(self._freed_by_dev, device)
        _log(f"{CYAN}[freed]{RESET} query={device}, result={format_bytes(val)}")
        return val, format_bytes(val)

    def set_auto_gc(self, bytes_threshold: int | str | None) -> None:
        self._resolve_and_set_auto_gc(bytes_threshold, context="set_auto_gc")

    # ---------- Auto threshold ----------
    def _resolve_and_set_auto_gc(self, val: int | str | None, context: str) -> None:
        auto_requested = (val is None) or (isinstance(val, str) and val.lower() == "auto")

        if not auto_requested:
            if not isinstance(val, int) or val < 0:
                raise ValueError("auto_gc_bytes must be an int >= 0, 'auto', or None")
            with self._lock:
                self._auto_gc_bytes = val
            _log(f"{YELLOW}[{context}]{RESET} auto_gc_bytes set to {format_bytes(val)} (explicit)")
            return

        threshold, debug_msg = self._compute_auto_threshold()
        with self._lock:
            self._auto_gc_bytes = threshold

        if threshold is None or threshold <= 0:
            _log(f"{YELLOW}[{context}]{RESET} auto_gc_bytes: CUDA not available; auto-GC disabled. {debug_msg}")
        else:
            _log(f"{YELLOW}[{context}]{RESET} auto_gc_bytes (auto): {debug_msg} -> {format_bytes(threshold)}")

    def _compute_auto_threshold(self) -> tuple[int | None, str]:
        try:
            if not torch.cuda.is_available():
                return None, "torch.cuda.is_available() == False"
            count = torch.cuda.device_count()
            if count <= 0:
                return None, "torch.cuda.device_count() == 0"
            totals = []
            parts = []
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                total = int(getattr(props, "total_memory", 0))
                totals.append(total)
                parts.append(f"{i}:{format_bytes(total)}")
            if not totals:
                return None, "No visible CUDA totals found"
            min_total = min(totals)
            threshold = min_total // 3
            return threshold, f"visible CUDA -> [{', '.join(parts)}]; min={format_bytes(min_total)}; min/3={format_bytes(threshold)}"
        except Exception as e:
            return None, f"auto detection error: {e}"

    # ---------- Memory accounting helpers ----------
    def _sizes_for_many(self, ob: ObjOrTuple) -> Dict[torch.device, int]:
        agg: Dict[torch.device, int] = {}
        for item in self._iter_objs(ob):
            for dev, b in self._sizes_by_device_instance(item).items():
                agg[dev] = agg.get(dev, 0) + b
        return agg

    def _iter_objs(self, ob: ObjOrTuple) -> Generator[Obj, None, None]:
        if isinstance(ob, tuple):
            for x in ob:
                if isinstance(x, (nn.Module, torch.Tensor)):
                    yield x
                else:
                    raise TypeError(f"Unsupported type in tuple: {type(x)}")
        elif isinstance(ob, (nn.Module, torch.Tensor)):
            yield ob
        else:
            raise TypeError(f"Unsupported type: {type(ob)}")

    def _sizes_by_device_instance(self, ob: Obj) -> Dict[torch.device, int]:
        tensors = list(self._gather_tensors(ob))
        return self._sum_by_dev_dedup(tensors)

    def _gather_tensors(self, ob: Obj) -> Iterable[torch.Tensor]:
        if isinstance(ob, torch.Tensor):
            yield ob
            return
        for p in ob.parameters(recurse=True):
            yield p.data
        for b in ob.buffers(recurse=True):
            yield b

    def _sum_by_dev_dedup(self, tensors: Iterable[torch.Tensor]) -> Dict[torch.device, int]:
        seen_keys: set[tuple[int, int]] = set()
        by_dev: Dict[torch.device, int] = {}

        def _accumulate_dense(t: torch.Tensor) -> None:
            dev = t.device
            if dev.type == "meta":
                return
            try:
                st = t.untyped_storage()
                key = (st.data_ptr(), st.nbytes())
                if key in seen_keys:
                    return
                seen_keys.add(key)
                by_dev[dev] = by_dev.get(dev, 0) + int(st.nbytes())
            except RuntimeError:
                nbytes = int(t.numel() * t.element_size())
                key = (t.data_ptr(), nbytes)
                if key in seen_keys:
                    return
                seen_keys.add(key)
                by_dev[dev] = by_dev.get(dev, 0) + nbytes

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

    # ---------- Summaries ----------
    def _all_known_devices_locked(self) -> list[torch.device]:
        all_set = set(self._allocated_by_dev.keys()) | set(self._freed_by_dev.keys())
        return sorted(all_set, key=lambda d: (d.type, -1 if d.index is None else d.index))

    def _totals_by_type_locked(self, table: Dict[torch.device, int]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for d, v in table.items():
            out[d.type] = out.get(d.type, 0) + v
        return out

    def _counts_by_type_locked(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for d in set(self._allocated_by_dev.keys()) | set(self._freed_by_dev.keys()):
            counts[d.type] = counts.get(d.type, 0) + 1
        return counts

    def _print_full_device_summary(
        self,
        header: str,
        per_device_map: Dict[torch.device, int],
        all_devices: list[torch.device],
        type_totals: Dict[str, int],
        type_counts: Dict[str, int],
    ) -> None:
        if not DEBUG_MODE:
            return
        for dev in all_devices:
            val = per_device_map.get(dev, 0)
            print(f"{header} {dev}: {format_bytes(val)}")
        for dtype in sorted(type_totals.keys()):
            if type_counts.get(dtype, 0) > 1:
                print(f"{header} {dtype}: {format_bytes(type_totals[dtype])}")

    # ---------- Auto-GC ----------
    def _maybe_auto_gc(self, dev: torch.device) -> None:
        threshold = self._auto_gc_bytes
        if threshold is None or threshold <= 0:
            return

        with self._lock:
            current_freed = self._freed_by_dev.get(dev, 0)

        if current_freed < threshold:
            return

        if _run_backend_gc(dev):
            with self._lock:
                self._freed_by_dev[dev] = 0
                self._gc_count_by_dev[dev] = self._gc_count_by_dev.get(dev, 0) + 1
                self._gc_total_count += 1
                per_dev_count = self._gc_count_by_dev[dev]
                total_count = self._gc_total_count

            _log(f"{YELLOW}[auto_gc]{RESET} {dev}: ran GC (count={per_dev_count}), total across devices={total_count}")


def _run_backend_gc(dev: torch.device) -> bool:
    try:
        if dev.type == "cuda":
            if dev.index is not None:
                torch.cuda.set_device(dev.index)
            torch.cuda.empty_cache()
            return True
        if dev.type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.empty_cache()  # type: ignore[attr-defined]
            return True
        if dev.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()  # type: ignore[attr-defined]
            return True
        if dev.type == "npu" and hasattr(torch, "npu"):
            torch.npu.empty_cache()  # type: ignore[attr-defined]
            return True
        return False
    except Exception:
        return False


def _sum_for_device(table: Dict[torch.device, int], query: torch.device) -> int:
    if query.index is None:
        return sum(v for d, v in table.items() if d.type == query.type)
    else:
        return table.get(torch.device(query.type, query.index), 0)


def format_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0

