# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import contextlib
import time
from typing import Iterator, Optional

from logbar import LogBar


def setup_logger():
    return LogBar.shared()


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

    label = block_name if not module_name else f"{module_name}: {block_name}"
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        #logger.info(f"[time] {label} took {duration:.3f}s")
