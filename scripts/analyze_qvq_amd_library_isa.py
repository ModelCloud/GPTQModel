#!/usr/bin/env python3
"""Audit exact hipBLASLt symbols selected by a rocprofv3 counter CSV.

Counts the complete global-symbol range (including alternate epilogues), not just
the initial argument-dispatch block returned by --disassemble-symbols. Static
counts and actual issued-counter distributions are deliberately kept separate.
"""

import argparse
import csv
import hashlib
import json
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counters", type=Path, required=True)
    parser.add_argument("--hsaco", type=Path, required=True)
    parser.add_argument("--objdump", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.counters.open() as stream:
        rows = list(csv.DictReader(stream))
    selected = defaultdict(list)
    for row in rows:
        selected[row["Kernel_Name"]].append(row)
    symbols = subprocess.check_output([str(args.objdump), "--syms", str(args.hsaco)], text=True)
    addresses = {}
    for line in symbols.splitlines():
        fields = line.split()
        if len(fields) >= 7 and fields[1:4] == ["g", "F", ".text"]:
            addresses[fields[-1]] = int(fields[0], 16)
    starts = sorted(set(addresses.values()))
    results = []
    for name, dispatches in selected.items():
        if name not in addresses:
            results.append({"kernel": name, "error": "symbol not in supplied code object"})
            continue
        start = addresses[name]
        index = starts.index(start)
        if index + 1 == len(starts):
            raise RuntimeError("Cannot infer terminal symbol extent; provide a known text-section end")
        stop = starts[index + 1]
        output = args.output_dir / f"{start:x}.amdgcn"
        with output.open("w") as stream:
            subprocess.run([str(args.objdump), "-d", "--mcpu=gfx950", f"--start-address={start}",
                            f"--stop-address={stop}", str(args.hsaco)], stdout=stream, check=True)
        opcodes = Counter(re.findall(r"^\s+((?:s_|v_|ds_|buffer_|global_)[a-zA-Z0-9_]+)",
                                     output.read_text(), re.MULTILINE))
        counters = defaultdict(list)
        for row in dispatches:
            counters[row["Counter_Name"]].append(float(row["Counter_Value"]))
        results.append({
            "kernel": name, "start": hex(start), "stop": hex(stop), "asm": str(output),
            "static_total": sum(opcodes.values()), "static_opcodes": dict(sorted(opcodes.items())),
            "issued_counters": {key: {"records": len(values), "min": min(values), "max": max(values),
                                      "mean": statistics.mean(values)} for key, values in counters.items()},
            "resources": {key: sorted({int(row[key]) for row in dispatches}) for key in
                          ("Grid_Size", "Workgroup_Size", "LDS_Block_Size", "Scratch_Size", "VGPR_Count",
                           "Accum_VGPR_Count", "SGPR_Count")},
        })
    with args.hsaco.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"hsaco": str(args.hsaco), "hsaco_sha256": digest,
              "counter_csv": str(args.counters), "symbols": results,
              "warning": "Whole-symbol static counts include untaken branches. Issued counters include all profiled launches."}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"selected_symbols": len(results), "missing": sum("error" in row for row in results)}))


if __name__ == "__main__":
    main()
