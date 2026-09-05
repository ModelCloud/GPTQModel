#!/usr/bin/env python3
"""Summarize executed ROCm counters and exact compiler artifacts for YAQA."""

import argparse
import collections
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = collections.defaultdict(list)
    for path in args.profile.rglob("*counter_collection.csv"):
        for row in csv.DictReader(path.open()):
            groups[(row["Kernel_Name"], row["VGPR_Count"], row["LDS_Block_Size"],
                    row["Scratch_Size"], row["Counter_Name"], row["Kernel_Id"])].append(float(row["Counter_Value"]))
    counters = [{"kernel": key[0], "vgpr": int(key[1]), "lds": int(key[2]), "scratch": int(key[3]),
                 "counter": key[4], "kernel_id": key[5], "dispatches": len(values), "mean": statistics.mean(values),
                 "min": min(values), "max": max(values)} for key, values in sorted(groups.items())]
    compiled = []
    for path in sorted(args.cache.rglob("*.amdgcn")):
        text = path.read_text()
        opcodes = collections.Counter(re.findall(r"^\s+((?:s_|v_|ds_|global_|buffer_|scratch_)[a-z0-9_]+)\s",
                                                 text, re.MULTILINE))
        fields = {}
        for name in ("vgpr_count", "sgpr_count", "private_segment_fixed_size", "group_segment_fixed_size"):
            match = re.search(r"\." + name + r":\s*(\d+)", text)
            fields[name] = int(match[1]) if match else None
        compiled.append({"path": str(path), "metadata": json.loads(path.with_suffix(".json").read_text()),
                         "resources": fields, "static_instructions": sum(opcodes.values()),
                         "opcodes": dict(sorted(opcodes.items())),
                         "sha256": {ext: hashlib.sha256(path.with_suffix(ext).read_bytes()).hexdigest()
                                    for ext in (".amdgcn", ".hsaco", ".llir", ".ttgir")
                                    if path.with_suffix(ext).exists()}})
        ir_path = path.with_suffix(".ttir")
        ir = ir_path.read_text() if ir_path.exists() and path.stem == "_survivor_step" else ""
        shapes = re.findall(r"tensor<(\d+)x(\d+)x(\d+)xf32>", ir)
        if shapes:
            shape = max((tuple(map(int, item)) for item in shapes), key=lambda item: item[0] * item[1] * item[2])
            compiled[-1]["specialization"] = {
                "banks": shape[0], "prefixes": shape[1], "q_chunk": shape[2],
                "first": "tt.splat %Previous" not in ir, "merge": "%across" in ir,
                "codebook_fp16": "%C: !tt.ptr<f16>" in ir,
            }
    args.output.write_text(json.dumps({"counter_scope": "issued wave instructions; not FLOPs or bandwidth",
                                      "profile": str(args.profile), "counters": counters,
                                      "compiled": compiled}, indent=2))


if __name__ == "__main__":
    main()
