# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import contextlib
import numbers
import os
import sys
import threading
import time
from collections import OrderedDict
from functools import wraps
from typing import Any, Dict, Iterator, Optional, Sequence

import pcre
from logbar import LogBar
from logbar.logbar import _clear_progress_stack_locked, _render_progress_stack_locked
from logbar.progress import ProgressBar


_ANSI_ESCAPE_RE = pcre.compile(r"\x1b\[[0-9;]*m")


def _animation_enabled() -> bool:
    """Check whether LogBar should animate/repaint in the current environment."""

    value = os.environ.get("LOGBAR_ANIMATION", "1")
    return str(value).strip().lower() not in {"0", "false", "off", "no"}


def _throttle_interval_seconds() -> float:
    """Return the headless progress-bar redraw interval in seconds."""

    value = os.environ.get("LOGBAR_PROGRESS_OUTPUT_INTERVAL", "1")
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 1.0


_original_progressbar_draw = ProgressBar.draw


@wraps(_original_progressbar_draw)
def _throttled_progressbar_draw(self, force: bool = False):
    """Throttle manual `.draw()` calls in headless mode to a wall-clock interval."""

    if not force and not _animation_enabled():
        interval = _throttle_interval_seconds()
        if interval > 0.0:
            now = time.monotonic()
            last = getattr(self, "_gptqmodel_throttle_last_draw", 0.0)
            if now - last < interval:
                return
            self._gptqmodel_throttle_last_draw = now
    return _original_progressbar_draw(self, force)


ProgressBar.draw = _throttled_progressbar_draw


# Headless-mode suppression of the active progress stack redraw that normally
# happens after every log line. Without this, each INFO log reprints every
# active progress bar, producing one progress line per log message.
_skip_stack_render = threading.local()


def _headless_clear_progress_stack_locked(*, show_cursor: bool = True, for_log_output: bool = False, backend_state=None):
    if getattr(_skip_stack_render, "value", False):
        return
    _clear_progress_stack_locked(show_cursor=show_cursor, for_log_output=for_log_output, backend_state=backend_state)


def _headless_render_progress_stack_locked(precomputed=None, columns_hint=None, backend_state=None):
    if getattr(_skip_stack_render, "value", False):
        return
    _render_progress_stack_locked(precomputed=precomputed, columns_hint=columns_hint, backend_state=backend_state)


_logbar_module = __import__("logbar.logbar", fromlist=["_clear_progress_stack_locked"])
_logbar_module._clear_progress_stack_locked = _headless_clear_progress_stack_locked
_logbar_module._render_progress_stack_locked = _headless_render_progress_stack_locked


_original_emit_log_line_locked = LogBar._emit_log_line_locked


@wraps(_original_emit_log_line_locked)
def _headless_emit_log_line_locked(self, normalized_level, level_label, str_msg, *, allow_defer=True, backend_state=None):
    animation_enabled = _animation_enabled()
    if not animation_enabled:
        _skip_stack_render.value = True
    try:
        return _original_emit_log_line_locked(self, normalized_level, level_label, str_msg, allow_defer=allow_defer, backend_state=backend_state)
    finally:
        if not animation_enabled:
            _skip_stack_render.value = False


LogBar._emit_log_line_locked = _headless_emit_log_line_locked


class _SilentProgress:
    """Minimal no-op progress handle for non-interactive test sessions."""

    def __init__(self, iterable=None):
        self._iterable = iterable if iterable is not None else ()
        self.current_iter_step = 0

    def __iter__(self):
        if isinstance(self._iterable, int):
            return iter(range(self._iterable))
        return iter(self._iterable)

    def __len__(self):
        if isinstance(self._iterable, int):
            return self._iterable
        return len(self._iterable)

    def attach(self, *_args, **_kwargs):
        return self

    def manual(self):
        return self

    def set(self, **_kwargs):
        return self

    def title(self, *_args, **_kwargs):
        return self

    def subtitle(self, *_args, **_kwargs):
        return self

    def draw(self, force: bool = False):
        return self

    def refresh(self):
        return self

    def next(self, step: int = 1):
        self.current_iter_step += int(step)
        return self

    def close(self):
        return None


class _AdaptiveLoggerProxy:
    """Proxy that keeps structured logs while adapting live rendering at call time."""

    def __init__(self, logger: LogBar):
        self._logger = logger

    def pb(self, iterable, *, output_interval: Optional[int] = None):
        if _suppress_live_renderables():
            return _SilentProgress(iterable)
        return self._logger.pb(iterable, output_interval=output_interval)

    def spinner(self, title: str = "", *, interval: float = 0.5, tail_length: int = 4):
        if _suppress_live_renderables():
            return _SilentProgress()
        return self._logger.spinner(title=title, interval=interval, tail_length=tail_length)

    def __getattr__(self, name):
        # __getattr__ only runs when normal attribute lookup misses. If _logger
        # itself is missing from the instance -- e.g. on a copy/pickle
        # reconstructed proxy created via __new__ without __init__ -- then
        # evaluating self._logger below re-enters __getattr__ and recurses
        # forever (RecursionError). This happens in practice when an object that
        # holds this proxy (see line ~223) is deepcopied: copy probes dunders
        # such as __deepcopy__/__setstate__ on the freshly __new__'d instance
        # before its __dict__ is restored. Raise a clean AttributeError for
        # dunders and _logger so those probes fail normally and copy/pickle fall
        # back to their default (which restores _logger).
        if name == "_logger" or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        return getattr(self._logger, name)


def _suppress_live_renderables() -> bool:
    """Disable live progress redraws under non-interactive pytest capture."""

    if "PYTEST_CURRENT_TEST" not in os.environ:
        return False

    try:
        return not sys.stdout.isatty()
    except Exception:
        return True


def live_renderables_suppressed() -> bool:
    """Report whether redraw-based progress should be replaced by durable logs."""

    return _suppress_live_renderables()


def setup_logger():
    return _AdaptiveLoggerProxy(LogBar.shared())


def _table_cell_text(value: Any, floatfmt: Optional[str]) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if floatfmt is not None and isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral):
        return format(float(value), floatfmt)
    return str(value)


def _visible_width(value: str) -> int:
    return len(_ANSI_ESCAPE_RE.sub("", value))


def _pad_table_cell(value: str, width: int) -> str:
    return value + (" " * max(0, width - _visible_width(value)))


def _render_grid_table(headers: Sequence[str], rows: Sequence[Sequence[str]], widths: Sequence[int]) -> str:
    def border() -> str:
        return "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def row_line(values: Sequence[str]) -> str:
        return "| " + " | ".join(_pad_table_cell(value, widths[idx]) for idx, value in enumerate(values)) + " |"

    lines = [border(), row_line(headers), border()]
    lines.extend(row_line(row) for row in rows)
    lines.append(border())
    return "\n".join(lines)


def _render_github_table(headers: Sequence[str], rows: Sequence[Sequence[str]], widths: Sequence[int]) -> str:
    def row_line(values: Sequence[str]) -> str:
        return "| " + " | ".join(_pad_table_cell(value, widths[idx]) for idx, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [row_line(headers), separator]
    lines.extend(row_line(row) for row in rows)
    return "\n".join(lines)


def _render_simple_table(headers: Sequence[str], rows: Sequence[Sequence[str]], widths: Sequence[int]) -> str:
    def row_line(values: Sequence[str]) -> str:
        return "  ".join(_pad_table_cell(value, widths[idx]) for idx, value in enumerate(values))

    separator = "  ".join("-" * width for width in widths)
    lines = [row_line(headers), separator]
    lines.extend(row_line(row) for row in rows)
    return "\n".join(lines)


def render_table(
    rows: Sequence[Sequence[Any]],
    *,
    headers: Sequence[Any],
    tablefmt: str = "grid",
    floatfmt: Optional[str] = None,
    logger: Optional[LogBar] = None,
) -> str:
    """Render a small diagnostic table using LogBar-compatible column sizing."""

    header_text = [str(header) for header in headers]
    row_text: list[list[str]] = []
    for row in rows:
        values = list(row)
        if len(values) != len(header_text):
            raise ValueError(
                f"Row length {len(values)} does not match header length {len(header_text)}"
            )
        row_text.append([_table_cell_text(value, floatfmt) for value in values])

    widths = [_visible_width(header) for header in header_text]
    if header_text:
        columns = (logger or LogBar.shared()).columns(
            cols=[{"label": header, "width": "fit"} for header in header_text],
            padding=1,
        )
        for row in row_text:
            columns.info.simulate(*row)
        widths = [max(widths[idx], width) for idx, width in enumerate(columns.widths)]

    for row in row_text:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], _visible_width(cell))

    tablefmt_normalized = (tablefmt or "grid").lower()
    if tablefmt_normalized == "github":
        return _render_github_table(header_text, row_text, widths)
    if tablefmt_normalized == "simple":
        return _render_simple_table(header_text, row_text, widths)
    return _render_grid_table(header_text, row_text, widths)


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
