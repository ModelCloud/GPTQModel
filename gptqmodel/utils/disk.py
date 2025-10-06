"""Disk IO utilities."""

from __future__ import annotations

import os
import tempfile
import time
from typing import Final

from .logger import setup_logger


log = setup_logger()

DEFAULT_WRITE_SIZE: Final[int] = 100 * 1024 * 1024
BUFFER_SIZE: Final[int] = 32 * 1024 * 1024


def estimate_disk_io_speed(write_size_bytes: int = DEFAULT_WRITE_SIZE) -> float:
    """Return sequential write speed in bytes/s measured via a temporary file.

    The helper writes ``write_size_bytes`` to a temporary file using a 32 MB CPU
    buffer and measures wall-clock time. The temporary file is fsync'd before
    computing the rate. Returns 0.0 on failure.
    """

    if write_size_bytes <= 0:
        raise ValueError("write_size_bytes must be positive")

    chunk_size = min(BUFFER_SIZE, write_size_bytes)
    chunk = b"\0" * chunk_size
    total_written = 0

    try:
        start_time = time.perf_counter()
        with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
            with open(tmp_file.name, "wb", buffering=BUFFER_SIZE) as handle:
                while total_written < write_size_bytes:
                    to_write = min(chunk_size, write_size_bytes - total_written)
                    handle.write(chunk if to_write == chunk_size else chunk[:to_write])
                    total_written += to_write
                handle.flush()
                os.fsync(handle.fileno())
        elapsed = time.perf_counter() - start_time
    except OSError as exc:  # pragma: no cover - depends on runtime environment
        log.warn(f"Disk IO speed estimation failed: {exc}")
        return 0.0

    if elapsed <= 0:
        return float("inf")

    return total_written / elapsed
