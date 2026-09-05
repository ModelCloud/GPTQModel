#!/usr/bin/env python3
"""Extract and audit the actual GPU binary in trusted local FlyDSL compiler dumps."""

import argparse
import csv
import hashlib
import json
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


def decode_binary(mlir):
    matches = re.findall(r'bin = "((?:\\.|[^"\\])*)"', mlir)
    if len(matches) != 1:
        raise ValueError("Expected one embedded binary")
    encoded = matches[0]
    payload = bytearray()
    i = 0
    while i < len(encoded):
        if encoded[i] == "\\":
            if encoded[i + 1:i + 2] in ('"', "\\"):
                payload.append(ord(encoded[i + 1]))
                i += 2
                continue
            pair = encoded[i + 1:i + 3]
            if len(pair) != 2 or not all(c in "0123456789abcdefABCDEF" for c in pair):
                raise ValueError(f"Unexpected MLIR binary escape at {i}")
            payload.append(int(pair, 16))
            i += 3
        else:
            payload.append(ord(encoded[i]))
            i += 1
    if not payload.startswith(b"\x7fELF"):
        raise ValueError("Missing ELF header")
    return bytes(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir-root", type=Path, required=True)
    parser.add_argument("--counters", type=Path, required=True)
    parser.add_argument("--objdump", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.counters.open() as stream:
        rows = list(csv.DictReader(stream))
    results = []
    for source in sorted(args.ir_root.glob("*/19_gpu_module_to_binary.mlir")):
        payload = decode_binary(source.read_text())
        name = source.parent.name
        binary = args.output_dir / f"{name}.hsaco"
        binary.write_bytes(payload)
        symbols = subprocess.check_output([str(args.objdump), "--syms", str(binary)], text=True)
        functions = [line.split() for line in symbols.splitlines()
                     if line.split() and line.split()[-1] == name]
        if len(functions) != 1 or functions[0][1:4] != ["g", "F", ".text"]:
            raise ValueError(f"Missing unique exact function symbol: {name}")
        start, size = int(functions[0][0], 16), int(functions[0][4], 16)
        if size <= 0:
            raise ValueError(f"Missing function extent: {name}")
        asm = subprocess.check_output([
            str(args.objdump), "-d", "--mcpu=gfx950", f"--start-address={start}",
            f"--stop-address={start + size}", str(binary),
        ], text=True)
        (args.output_dir / f"{name}.amdgcn").write_text(asm)
        opcodes = Counter(re.findall(r"^\s+((?:s_|v_|ds_|buffer_|global_)[a-zA-Z0-9_]+)", asm, re.MULTILINE))
        selected = [row for row in rows if row["Kernel_Name"].removesuffix(".kd") == name]
        if not selected:
            raise ValueError(f"Binary has no matched executed counter symbol: {name}")
        counters = defaultdict(list)
        for row in selected:
            counters[row["Counter_Name"]].append(float(row["Counter_Value"]))
        results.append({
            "kernel": name, "source": str(source), "binary": str(binary),
            "binary_sha256": hashlib.sha256(payload).hexdigest(),
            "function_start": start, "function_bytes": size,
            "static_total": sum(opcodes.values()), "opcodes": dict(sorted(opcodes.items())),
            "counters": {k: {"samples": len(v), "min": min(v), "mean": statistics.mean(v), "max": max(v)}
                         for k, v in counters.items()},
            "resources": {k: sorted({int(row[k]) for row in selected}) for k in
                          ("Grid_Size", "Workgroup_Size", "LDS_Block_Size", "Scratch_Size", "VGPR_Count",
                           "Accum_VGPR_Count", "SGPR_Count")},
        })
    if not results:
        raise ValueError("No FlyDSL compiler binaries found")
    report = {"counter_csv": str(args.counters), "kernels": results,
              "scope": "Exact embedded binary disassembly; static counts include all branches, not executed totals"}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
