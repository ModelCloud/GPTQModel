#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time


def run_nvidia_smi(query: str) -> list[list[str]]:
    proc = subprocess.run(
        ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        [part.strip() for part in line.split(",")]
        for line in proc.stdout.splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uuid", required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--max-memory-mib", type=int, default=1024)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("command is required after --")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.uuid:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES mapping mismatch: expected {args.uuid}, "
            f"found {os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )

    records: list[dict[str, object]] = []
    consecutive = 0
    for attempt in range(args.max_attempts):
        gpu_rows = run_nvidia_smi("gpu=index,pci.bus_id,uuid,name,memory.total,memory.used,utilization.gpu")
        app_rows = run_nvidia_smi("compute-apps=gpu_uuid,pid,process_name,used_gpu_memory")
        target_rows = [row for row in gpu_rows if len(row) >= 7 and row[2] == args.uuid]
        target_apps = [row for row in app_rows if len(row) >= 2 and row[0] == args.uuid]
        sample_ok = False
        if len(target_rows) == 1:
            used_mib = int(target_rows[0][5])
            utilization = int(target_rows[0][6])
            sample_ok = used_mib <= args.max_memory_mib and utilization == 0 and not target_apps
        records.append(
            {
                "attempt": attempt + 1,
                "gpu": target_rows,
                "compute_apps": target_apps,
                "sample_ok": sample_ok,
            }
        )
        consecutive = consecutive + 1 if sample_ok else 0
        if consecutive == args.samples:
            break
        if attempt + 1 < args.max_attempts:
            time.sleep(1)

    result = {
        "ok": consecutive == args.samples,
        "expected_uuid": args.uuid,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "required_consecutive_idle_samples": args.samples,
        "max_attempts": args.max_attempts,
        "max_memory_mib": args.max_memory_mib,
        "samples": records,
    }
    print(json.dumps({"launch_idle_gate": result}, sort_keys=True), flush=True)
    if not result["ok"]:
        return 2
    os.execvp(command[0], command)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
