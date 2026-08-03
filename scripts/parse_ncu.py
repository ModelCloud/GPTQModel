import csv
import subprocess
import sys
from pathlib import Path

metrics_of_interest = {
    "Elapsed Cycles",
    "Memory Throughput",
    "DRAM Throughput",
    "SM Throughput",
    "Compute (SM) Throughput",
    "L2 Cache Throughput",
    "L2 Hit Rate",
    "Waves Per SM",
    "Achieved Active Warps Per SM",
    "Theoretical Active Warps per SM",
    "Block Limit SM",
    "Occupancy",
    "Threads",
    "Registers Per Thread",
    "Static Shared Memory Per Block",
    "Shared Memory Configuration Size",
    "Shared Memory Per Block",
    "Block Size",
    "Grid Size",
    "Duration",
    "Average DRAM Bandwidth",
    "DRAM Bytes",
}


def extract(rep_path: Path):
    proc = subprocess.run(
        ["ncu", "-i", str(rep_path), "--csv"],
        capture_output=True, text=True, timeout=120,
    )
    rows = list(csv.DictReader(proc.stdout.splitlines()))
    if not rows:
        print(f"no rows for {rep_path}")
        return
    first = rows[0]
    kernel = first["Kernel Name"]
    block = first["Block Size"]
    grid = first["Grid Size"]
    print(f"\n{rep_path.name}: {kernel}")
    print(f"  grid={grid} block={block}")
    seen = set()
    for row in rows:
        name = row["Metric Name"]
        if name in metrics_of_interest and name not in seen:
            seen.add(name)
            print(f"  {name}: {row['Metric Value']} {row['Metric Unit']}")


for p in sys.argv[1:]:
    extract(Path(p))
