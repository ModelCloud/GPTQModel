# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Keep a 2 GiB host reserve while running 1B W4A tests on GB10."""

import os
import sys
from pathlib import Path


GIB = 1024 ** 3
HOST_RESERVE_BYTES = 2 * GIB
# The live host/root headroom calculation below is the actual bound.  Keep this
# ceiling at the GB10's usable unified-memory size so it does not impose an
# unrelated 24/32 GiB limit while the wrapper still reserves exactly 2 GiB.
MAX_TEST_BYTES = 128 * GIB
# This scope intentionally has no swap.  A lower memory.high therefore has no
# useful reclaim target during 512-row calibration and can pin the process in
# mem_cgroup_handle_over_high.  memory.max remains the hard per-run boundary;
# the wrapper separately enforces the 2 GiB host/root reserve.
MAX_HIGH_BYTES = MAX_TEST_BYTES


def _available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Cannot read MemAvailable from /proc/meminfo")


def _cgroup_headroom_bytes() -> int | None:
    root = Path("/sys/fs/cgroup")
    maximum = (root / "memory.max").read_text().strip()
    if maximum == "max":
        return None
    return max(0, int(maximum) - int((root / "memory.current").read_text().strip()))


def _require_limited_scope() -> None:
    group = next(
        (line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
         if line.startswith("0::")),
        None,
    )
    if group is None:
        raise RuntimeError("Cannot identify the current cgroup for the 1B W4A test")
    current = Path("/sys/fs/cgroup") / group.lstrip("/")
    maximum = (current / "memory.max").read_text().strip()
    high = (current / "memory.high").read_text().strip()
    swap = (current / "memory.swap.max").read_text().strip()
    expected = os.environ.get("GPTQMODEL_W4A_MEMORY_MAX_BYTES")
    if (expected is None or maximum == "max" or maximum != expected or
            int(maximum) > MAX_TEST_BYTES or high == "max" or
            int(high) > MAX_HIGH_BYTES or swap != "0"):
        raise RuntimeError(
            "1B W4A tests require the memory-limited, zero-swap scope created "
            "by tests/models/run_w4a_gb10_safe.sh."
        )


def memory_budget_bytes() -> tuple[int, int]:
    """Return a hard and soft scope cap based on current host headroom."""
    available = _available_bytes()
    cgroup = _cgroup_headroom_bytes()
    safe = min(available, cgroup) if cgroup is not None else available
    if safe <= HOST_RESERVE_BYTES:
        raise RuntimeError(
            f"Refusing 1B W4A test: safe memory headroom is {safe / GIB:.1f} GiB, "
            "below the 2 GiB host reserve. "
            "Check free -h, /sys/fs/cgroup/memory.stat, and memory.events before retrying."
        )
    maximum = min(MAX_TEST_BYTES, safe - HOST_RESERVE_BYTES)
    # A lower soft cap can stall CUDA allocation and drive system-wide memory
    # pressure even while the hard cap still has headroom.
    high = min(MAX_HIGH_BYTES, maximum)
    return maximum, high


def require_w4a_test_headroom(*, require_scope: bool = False) -> None:
    """Require only a 2 GiB reserve, plus a bounded test scope."""
    memory_budget_bytes()
    if require_scope:
        _require_limited_scope()


if __name__ == "__main__":
    if sys.argv[1:] == ["--budget"]:
        print(*memory_budget_bytes())
    else:
        require_w4a_test_headroom()
