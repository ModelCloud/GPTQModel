# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Read the CPU allocation and Linux topology without importing compute libraries."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path


def cpu_runtime_inventory() -> dict:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    cores = defaultdict(list)
    nodes = defaultdict(list)
    if affinity is not None:
        for cpu in affinity:
            root = Path(f"/sys/devices/system/cpu/cpu{cpu}")
            try:
                package = int((root / "topology/physical_package_id").read_text())
                core = int((root / "topology/core_id").read_text())
            except (OSError, ValueError):
                continue
            cores[(package, core)].append(cpu)
            for node in root.glob("node[0-9]*"):
                nodes[node.name].append(cpu)
    cgroup = {}
    for name in ("cpu.max", "cpuset.cpus.effective", "cpuset.mems.effective", "memory.max"):
        try:
            cgroup[name] = (Path("/sys/fs/cgroup") / name).read_text().strip()
        except OSError:
            cgroup[name] = None
    return {
        "affinity": affinity,
        "allowed_logical_cpus": None if affinity is None else len(affinity),
        "represented_physical_cores": len(cores) if cores else None,
        "physical_core_groups": [
            {"socket": package, "core": core, "allowed_cpus": cpus}
            for (package, core), cpus in sorted(cores.items())
        ],
        "numa_allowed_cpus": dict(sorted(nodes.items())),
        "cgroup": cgroup,
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES")
        },
    }
