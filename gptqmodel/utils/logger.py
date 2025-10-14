# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import contextlib
import threading
import time
from collections import OrderedDict
from typing import Dict, Iterator, Optional

from logbar import LogBar


def setup_logger():
    return LogBar.shared()


class QuantizationRegionTimer:
    """Aggregate and display timing statistics for key quantization stages."""

    DEFAULT_REGIONS = [
        ("model_load", "Model load"),
        ("model_reload", "Turtle reload"),
        ("capture_inputs", "Capture inputs"),
        ("forward_hook", "Forward hook"),
        ("pre_quant_forward", "Pre-quant forward"),
        ("process_quant", "Process quant"),
        ("post_quant_forward", "Post-quant replay"),
        ("submodule_finalize", "Submodule finalize"),
        ("submodule_finalize_create", "Finalize create"),
        ("submodule_finalize_pack", "Finalize pack"),
        ("submodule_finalize_offload", "Finalize offload"),
        ("process_finalize", "Process finalize"),
        ("model_save", "Model save"),
    ]

    def __init__(self, logger: Optional[LogBar] = None):
        self.logger = logger or setup_logger()
        self._lock = threading.Lock()
        self._columns = None
        self._header_printed = False
        self._region_labels: "OrderedDict[str, str]" = OrderedDict(self.DEFAULT_REGIONS)
        self._stats: "OrderedDict[str, Dict[str, float | int | str | None]]" = OrderedDict()
        self._pending_refresh = False
        self.reset()

    def reset(self) -> None:
        """Reset accumulated timing data."""
        with self._lock:
            self._stats = OrderedDict(
                (region, self._fresh_stat()) for region in self._region_labels.keys()
            )
            self._header_printed = False
            self._pending_refresh = False

    def _fresh_stat(self) -> Dict[str, float | int | None]:
        return {"total": 0.0, "count": 0, "last": 0.0, "source": None}

    def _ensure_columns_locked(self) -> None:
        if self._columns is not None:
            return

        column_specs = [
            {"label": "region", "width": "fit"},
            {"label": "count", "width": "fit"},
            {"label": "last_s", "width": "fit"},
            {"label": "avg_s", "width": "fit"},
            {"label": "total_s", "width": "fit"},
            {"label": "pct", "width": "fit"},
            {"label": "source", "width": "fit"},
        ]

        self._columns = self.logger.columns(cols=column_specs, padding=1)
        self._columns.info.simulate(
            "model_load", "1", "0.001", "0.001", "0.001", "100.0%", "layer.0"
        )

    def record(self, region: str, duration: float, *, source: Optional[str] = None) -> None:
        """Record a timing sample for a region and emit an updated summary."""

        if duration is None:
            return

        try:
            duration_value = float(duration)
        except (TypeError, ValueError):
            return

        if duration_value < 0:
            duration_value = 0.0

        with self._lock:
            if region not in self._stats:
                if region not in self._region_labels:
                    self._region_labels[region] = region.replace("_", " ").title()
                self._stats[region] = self._fresh_stat()

            stat = self._stats[region]
            stat["total"] = float(stat.get("total", 0.0)) + duration_value
            stat["count"] = int(stat.get("count", 0)) + 1
            stat["last"] = duration_value
            if source is not None:
                stat["source"] = source

            self._pending_refresh = True

    def flush(self) -> None:
        """Emit the current summary if new measurements were recorded."""
        with self._lock:
            if not self._pending_refresh:
                return
            self._print_summary_locked()
            self._pending_refresh = False

    def _print_summary_locked(self) -> None:
        self._ensure_columns_locked()

        # Filter out regions that have not been recorded yet.
        populated = [
            (region, stat)
            for region, stat in self._stats.items()
            if stat.get("count", 0)
        ]

        if not populated:
            return

        overall_total = sum(float(stat.get("total", 0.0)) for _, stat in populated)
        if overall_total <= 0:
            overall_total = 0.0

        # Sort by total descending so hotspots float to the top.
        populated.sort(key=lambda item: float(item[1].get("total", 0.0)), reverse=True)

        if not self._header_printed:
            self._columns.info.header()
            self._header_printed = True

        for region, stat in populated:
            display_name = self._region_labels.get(region, region)
            total = float(stat.get("total", 0.0))
            count = int(stat.get("count", 0))
            last = float(stat.get("last", 0.0))
            avg = total / count if count else 0.0
            pct = (total / overall_total * 100.0) if overall_total > 0 else 0.0
            source = stat.get("source") or ""

            self._columns.info(
                display_name,
                str(count),
                f"{last:.3f}",
                f"{avg:.3f}",
                f"{total:.3f}",
                f"{pct:.1f}%",
                source,
            )

    @contextlib.contextmanager
    def measure(self, region: str, *, source: Optional[str] = None) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.record(region, duration, source=source)

    def snapshot(self) -> Dict[str, Dict[str, float | int | str | None]]:
        with self._lock:
            return {
                region: {
                    "total": float(stat.get("total", 0.0)),
                    "count": int(stat.get("count", 0)),
                    "last": float(stat.get("last", 0.0)),
                    "source": stat.get("source"),
                }
                for region, stat in self._stats.items()
            }


@contextlib.contextmanager
def log_time_block(
    block_name: str,
    *,
    logger: Optional[LogBar] = None,
    module_name: Optional[str] = None,
) -> Iterator[None]:
    """Log the elapsed time of a block to the shared logger."""

    if logger is None:
        logger = setup_logger()

    start = time.perf_counter()
    try:
        yield
    finally:
        time.perf_counter() - start
        #logger.info(f"[time] {label} took {duration:.3f}s")
