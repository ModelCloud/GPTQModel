import os
from pathlib import Path


def _read_inventory():
    affinity = set(os.sched_getaffinity(0))
    groups = {}
    nodes = {}
    for cpu in sorted(affinity):
        root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = (int((root / "physical_package_id").read_text()), int((root / "core_id").read_text()))
        groups.setdefault(key, []).append(cpu)
    for path in sorted(Path("/sys/devices/system/node").glob("node[0-9]*/cpulist")):
        cpus = set()
        for part in path.read_text().strip().split(","):
            limits = part.split("-")
            cpus.update(range(int(limits[0]), int(limits[-1]) + 1))
        nodes[path.parent.name] = sorted(cpus & affinity)
    return {
        "physical_core_groups": [
            {"socket": p, "core": c, "allowed_cpus": sorted(cpus)} for (p, c), cpus in sorted(groups.items())
        ],
        "numa_allowed_cpus": nodes,
    }


from functools import lru_cache


@lru_cache(maxsize=8)
def _cached_inventory(affinity, online):
    return _read_inventory()


def cpu_runtime_inventory():
    return _cached_inventory(
        tuple(sorted(os.sched_getaffinity(0))), Path("/sys/devices/system/cpu/online").read_text()
    )
