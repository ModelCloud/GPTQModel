#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Direct device-to-device (P2P) vs host-staged copy bandwidth benchmark.

For every ordered device pair this reports, in both directions:

* ``p2p_off`` - ``dst.copy_(src)`` before peer access is enabled (torch host-stages
  when no peer link is active, which is the baseline)
* ``p2p_on``  - the same copy after ``gptqmodel.utils.p2p.enable_peer_access()``
* ``host``    - an explicit ``src -> pinned CPU -> dst`` staging copy
* ``gate``    - the ``PeerStatus`` state (``enabled`` / ``unsupported`` / ``degraded``)

A capability bit is never treated as proof: a pair that reports
``can_device_access_peer == True`` but measures below ``GPTQMODEL_P2P_MIN_GBPS``
is classified ``degraded`` by the module and must fall back to host staging.
``--min-gbps`` is exported to that variable before enablement, so the gate and
this benchmark agree.

``--sku 4090`` only prints consumer-card caveats (Ada consumer parts often report
peer capability while direct throughput is below host staging); it never
hard-codes a whitelist, because the verdict is the measured bandwidth.

Bandwidth is reported in GB/s (``bytes / seconds / 1e9``, the unit used by the
P2P module's own gate) with GiB/s alongside, matching the layout of the existing
``scripts``/``tests`` copy benchmarks.

``--dry-run`` prints the plan and exits 0 without touching a device; fewer than
two visible GPUs also prints the plan and exits 0.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


DEFAULT_MIB = 256
DEFAULT_ITERS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P2P vs host-staged GPU copy bandwidth benchmark.")
    parser.add_argument("--devices", default="0,1", help="Comma-separated device indices, e.g. 0,1.")
    parser.add_argument("--mib", type=int, default=DEFAULT_MIB, help="Payload size per copy in MiB.")
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS, help="Timed iterations per direction.")
    parser.add_argument("--sku", default=None, help="Optional SKU hint, e.g. 4090 (caveats only).")
    parser.add_argument(
        "--min-gbps",
        type=float,
        default=None,
        help="Exported to GPTQMODEL_P2P_MIN_GBPS: below it a peer link is `degraded`.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit 0.")
    return parser.parse_args()


def _parse_devices(value: str) -> list[int]:
    devices = [int(item) for item in value.split(",") if item.strip()]
    if len(devices) < 2:
        raise ValueError("--devices needs at least two indices, e.g. 0,1.")
    if len(set(devices)) != len(devices):
        raise ValueError(f"--devices contains duplicates: {devices}")
    return devices


def _device_report(devices: list[int]) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for index in devices:
        entry: dict[str, Any] = {"index": index, "name": None, "arch": None, "arch_class": None, "error": None}
        try:
            entry["name"] = torch.cuda.get_device_name(index)
        except Exception as exc:  # pragma: no cover - runtime dependent
            entry["error"] = f"get_device_name failed: {type(exc).__name__}: {exc}"
        try:
            from gptqmodel.utils import arch

            entry["arch"] = arch.rocm_arch(index)
            entry["arch_class"] = arch.arch_class(index)
        except Exception as exc:  # pragma: no cover - runtime dependent
            entry["error"] = f"arch probe failed: {type(exc).__name__}: {exc}"
        report.append(entry)
    return report


def _sku_notes(sku: str | None) -> list[str]:
    if sku is None:
        return []
    normalized = sku.strip().lower()
    if normalized in ("4090", "rtx4090", "rtx-4090", "ada-consumer"):
        return [
            "RTX 4090 (Ada consumer, sm_89): the driver may report P2P enabled for a while; "
            "consumer parts commonly show can_device_access_peer == True with direct throughput "
            "below host staging.",
            "The verdict here is the measured bandwidth, not the capability bit: a pair under "
            "--min-gbps / GPTQMODEL_P2P_MIN_GBPS is `degraded` and must host-stage.",
            "When P2P is degraded or unsupported, NCCL is a separate path: verify "
            "torch.distributed.is_nccl_available() and a real all-reduce/ring transfer independently.",
        ]
    return [f"SKU {sku!r}: no specific caveat registered; use the measured bandwidth only."]


def _plan(args: argparse.Namespace, reason: str) -> dict[str, Any]:
    devices = _parse_devices(args.devices)
    return {
        "script": "benchmark_multi_gpu_p2p.py",
        "mode": "dry-run",
        "reason": reason,
        "devices": devices,
        "mib": args.mib,
        "iters": args.iters,
        "sku": args.sku,
        "sku_notes": _sku_notes(args.sku),
        "min_gbps": args.min_gbps if args.min_gbps is not None else os.getenv("GPTQMODEL_P2P_MIN_GBPS", "<default>"),
        "directions": [f"{src}->{dst}" for src in devices for dst in devices if src != dst],
        "measured_per_direction": ["p2p_off GB/s", "p2p_on GB/s", "host-staged GB/s", "peer state", "capability"],
        "sequence": [
            "probe peer matrix (directed, both directions)",
            "measure p2p_off: dst.copy_(src) before enable_peer_access",
            "export GPTQMODEL_P2P_MIN_GBPS from --min-gbps, then enable_peer_access(devices)",
            "measure p2p_on: dst.copy_(src) after enablement",
            "measure host: src -> pinned CPU -> dst",
        ],
        "env": {
            "GPTQMODEL_P2P": os.getenv("GPTQMODEL_P2P", "auto"),
            "GPTQMODEL_P2P_MIN_GBPS": os.getenv("GPTQMODEL_P2P_MIN_GBPS", "<unset>"),
            "GPTQMODEL_P2P_PROBE_MIB": os.getenv("GPTQMODEL_P2P_PROBE_MIB", "<unset>"),
        },
        "device_info": _device_report(devices) if torch.cuda.is_available() else [],
        "commands": {
            "rocm": (
                "/root/miniconda3/envs/gp_314t_rocm/bin/python scripts/benchmark_multi_gpu_p2p.py "
                f"--devices {args.devices} --mib {args.mib} --iters {args.iters}"
            ),
            "cuda_4090": (
                "/root/miniconda3/envs/gp_314t/bin/python scripts/benchmark_multi_gpu_p2p.py "
                f"--sku 4090 --devices {args.devices} --min-gbps 20"
            ),
        },
    }


def _print_plan(plan: dict[str, Any]) -> None:
    print(json.dumps(plan, indent=2, sort_keys=True))
    print(
        "plan: "
        f"devices={plan['devices']} mib={plan['mib']} iters={plan['iters']} "
        f"min_gbps={plan['min_gbps']} directions={plan['directions']}",
        flush=True,
    )
    for note in plan["sku_notes"]:
        print(f"sku-note: {note}")


def _median_gbps(step, src: torch.Tensor, dst: torch.Tensor, iters: int) -> float | None:
    """Time ``step`` per iteration, return median payload GB/s.

    Per-iteration host timing with a device synchronize around every sample is
    deliberate: CUDA-event pairs under-reported a 0.35 ms cross-device copy as
    300+ ms on gfx1201, and a single batched loop was still unstable right after
    allocation. Warm-ups absorb the first-copy context/module load.
    """
    try:
        for _ in range(8):
            step()
            torch.cuda.synchronize(src.device)
            torch.cuda.synchronize(dst.device)
        samples = []
        for _ in range(iters):
            started = time.perf_counter()
            step()
            torch.cuda.synchronize(src.device)
            torch.cuda.synchronize(dst.device)
            samples.append(time.perf_counter() - started)
    except Exception as exc:  # pragma: no cover - runtime dependent
        print(f"copy measurement failed: {type(exc).__name__}: {exc}", flush=True)
        return None
    if not samples:
        return None
    elapsed = statistics.median(samples)
    if elapsed <= 0:
        return None
    return (src.numel() * src.element_size()) / elapsed / 1e9


def _measure(src: torch.Tensor, dst: torch.Tensor, iters: int) -> float | None:
    """Return GB/s for repeated ``dst.copy_(src)``; None when the copy fails."""
    return _median_gbps(lambda: dst.copy_(src, non_blocking=True), src, dst, iters)


def _measure_host_staged(src: torch.Tensor, dst: torch.Tensor, iters: int) -> float | None:
    """Return payload GB/s for an explicit ``src -> pinned CPU -> dst`` copy.

    The rate is reported per payload byte so it is directly comparable with the
    direct ``src -> dst`` measurement: a staged copy moves the payload twice
    (D2H then H2D) and therefore costs roughly 2x the wall time of a direct
    transfer at equal link rate. Dividing by ``2 * bytes`` would inflate this
    number and make every direct path look ~0.5x slower than it is.
    """
    staged = torch.empty(src.shape, dtype=src.dtype, pin_memory=True)

    def step():
        staged.copy_(src, non_blocking=True)
        dst.copy_(staged, non_blocking=True)

    # Payload bytes only: see the docstring for why the 2x transfer volume is
    # deliberately not counted in the rate.
    return _median_gbps(step, src, dst, iters)


def _row(capability: bool, src_index: int, dst_index: int, size_bytes: int, iters: int) -> dict[str, Any]:
    return {
        "src": src_index,
        "dst": dst_index,
        "direction": f"{src_index}->{dst_index}",
        "capability": capability,
        "peer_state": None,
        "peer_bandwidth_gbps": None,
        "size_bytes": size_bytes,
        "iters": iters,
        "p2p_off_gbps": None,
        "p2p_on_gbps": None,
        "host_gbps": None,
    }


def run(args: argparse.Namespace) -> int:
    from gptqmodel.utils import p2p

    devices = _parse_devices(args.devices)
    size_bytes = args.mib * 1024 * 1024
    elements = size_bytes // 2  # fp16 elements
    notes = _sku_notes(args.sku)
    for note in notes:
        print(f"sku-note: {note}", flush=True)

    if args.min_gbps is not None:
        os.environ[p2p.ENV_P2P_MIN_GBPS] = str(args.min_gbps)
        p2p.clear_p2p_cache()
        print(f"exported {p2p.ENV_P2P_MIN_GBPS}={args.min_gbps}", flush=True)

    capability = p2p.peer_matrix(devices)
    print("peer matrix (rows=src, cols=dst):", flush=True)
    for row_index, src in enumerate(devices):
        flags = " ".join("yes" if value else " no" for value in capability[row_index])
        print(f"  {src}: {flags}", flush=True)

    rows: list[dict[str, Any]] = []
    for row_index, src_index in enumerate(devices):
        for col_index, dst_index in enumerate(devices):
            if src_index == dst_index:
                continue
            src = torch.empty(elements, dtype=torch.float16, device=f"cuda:{src_index}")
            dst = torch.empty(elements, dtype=torch.float16, device=f"cuda:{dst_index}")
            src.uniform_()

            row = _row(bool(capability[row_index][col_index]), src_index, dst_index, size_bytes, args.iters)
            # Baseline: peer access has not been requested in this process yet.
            row["p2p_off_gbps"] = _measure(src, dst, args.iters)
            row["host_gbps"] = _measure_host_staged(src, dst, args.iters)
            rows.append(row)
            del src, dst
            torch.cuda.empty_cache()

    # Enablement happens after every baseline measurement, because peer access
    # cannot be revoked inside a process.
    status = p2p.enable_peer_access(devices)
    print(f"peer status: {status.summary()}", flush=True)

    for row in rows:
        src = torch.empty(elements, dtype=torch.float16, device=f"cuda:{row['src']}")
        dst = torch.empty(elements, dtype=torch.float16, device=f"cuda:{row['dst']}")
        src.uniform_()
        row["peer_state"] = status.state(row["src"], row["dst"])
        row["peer_bandwidth_gbps"] = status.bandwidth_gbps.get((row["src"], row["dst"]))
        row["p2p_on_gbps"] = _measure(src, dst, args.iters)
        del src, dst
        torch.cuda.empty_cache()
        row["direct_vs_host"] = (
            None
            if row["p2p_on_gbps"] is None or not row["host_gbps"]
            else round(row["p2p_on_gbps"] / row["host_gbps"], 3)
        )
        row["gibps_on"] = None if row["p2p_on_gbps"] is None else round(row["p2p_on_gbps"] * 1e9 / 2**30, 3)
        print(json.dumps(row, sort_keys=True), flush=True)

    payload = {
        "mode": "measured",
        "devices": devices,
        "capability_matrix": capability,
        "mib": args.mib,
        "iters": args.iters,
        "sku": args.sku,
        "sku_notes": notes,
        "min_gbps": args.min_gbps if args.min_gbps is not None else os.getenv("GPTQMODEL_P2P_MIN_GBPS"),
        "min_ratio": p2p._min_ratio(),
        "peer_status": status.summary(),
        "direct_copy_allowed": status.direct_copy_allowed(),
        "rows": rows,
        "device_info": _device_report(devices),
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def main() -> int:
    args = parse_args()
    if args.dry_run:
        _print_plan(_plan(args, "requested via --dry-run"))
        return 0
    if not torch.cuda.is_available():
        _print_plan(_plan(args, "no CUDA/HIP device visible; nothing measured"))
        return 0
    if torch.cuda.device_count() < 2:
        _print_plan(_plan(args, f"only {torch.cuda.device_count()} device(s) visible; P2P needs two"))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
