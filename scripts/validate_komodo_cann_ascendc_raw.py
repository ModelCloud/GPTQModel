#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_LOCAL_A_CASES = (
    {"rows": 1, "k": 384, "n": 256, "group": 32, "seed": 2200},
    {"rows": 2, "k": 512, "n": 256, "group": 64, "seed": 2201},
    {"rows": 4, "k": 512, "n": 256, "group": 32, "seed": 2202},
    {"rows": 4, "k": 512, "n": 512, "group": 64, "seed": 2203},
    {"rows": 8, "k": 1024, "n": 512, "group": 128, "seed": 2204},
    {"rows": 3, "k": 768, "n": 256, "group": 96, "seed": 2205},
    {"rows": 7, "k": 896, "n": 256, "group": 128, "seed": 2206},
    {"rows": 8, "k": 1024, "n": 512, "group": 32, "seed": 2207},
)


def _parse_devices(raw: str) -> list[int]:
    devices = [int(part) for part in raw.split(",") if part.strip()]
    if not devices:
        raise argparse.ArgumentTypeError("at least one device is required")
    return devices


def _stderr_tail(text: str, lines: int = 40) -> str:
    return "\n".join(text.splitlines()[-lines:])


def _last_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    result: dict[str, Any] | None = None
    offset = 0
    while True:
        start = text.find("{", offset)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            offset = start + 1
            continue
        if isinstance(value, dict):
            result = value
        offset = start + max(end, 1)
    return result


def _apply_custom_opp_env(env: dict[str, str], args: argparse.Namespace) -> None:
    if args.opp_install:
        vendor = Path(args.opp_install).expanduser() / "vendors" / "customize"
        opapi_lib = vendor / "op_api" / "lib" / "libcust_opapi.so"
        env["ASCEND_CUSTOM_OPP_PATH"] = str(vendor) + os.pathsep + env.get("ASCEND_CUSTOM_OPP_PATH", "")
        env["LD_LIBRARY_PATH"] = str(opapi_lib.parent) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        env.setdefault("GPTQMODEL_KOMODO_CANN_ASCENDC_OPAPI_LIB", str(opapi_lib))
    if args.opapi_lib:
        env["GPTQMODEL_KOMODO_CANN_ASCENDC_OPAPI_LIB"] = str(Path(args.opapi_lib).expanduser())


def _case_payload(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    payload = dict(case)
    payload.setdefault("base_m", args.base_m)
    payload.setdefault("base_n", args.base_n)
    payload.setdefault("base_k", args.base_k)
    return payload


def _worker(args: argparse.Namespace) -> int:
    case = json.loads(args.case_json)
    bridge_lib = Path(args.bridge_lib).expanduser()
    if not bridge_lib.exists():
        raise FileNotFoundError(f"bridge library not found: {bridge_lib}")

    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    torch.ops.load_library(str(bridge_lib))
    torch.manual_seed(int(case["seed"]))

    rows = int(case["rows"])
    k = int(case["k"])
    n = int(case["n"])
    group = int(case["group"])
    base_m = int(case["base_m"])
    base_n = int(case["base_n"])
    base_k = int(case["base_k"])

    x = torch.randn((rows, k), device="npu", dtype=torch.float16)
    signed_weight = torch.randint(-8, 8, (k, n), device="npu", dtype=torch.int32).contiguous()
    packed_weight = torch.ops.npu.npu_convert_weight_to_int4pack(signed_weight)
    scales = torch.full((k // group, n), 0.03125, device="npu", dtype=torch.float16)
    offsets = torch.zeros((k // group, n), device="npu", dtype=torch.float16)

    custom = torch.ops.gptqmodel_komodo_cann.komodo_cann_w4_a16_matmul(
        x,
        packed_weight,
        scales,
        offsets,
        None,
        group,
        -1,
        base_m,
        base_n,
        base_k,
    )
    native = torch.ops.npu.npu_weight_quant_batchmatmul(
        x,
        packed_weight,
        scales,
        offsets,
        None,
        None,
        None,
        group,
        1,
    )
    torch.npu.synchronize()
    diff = (custom - native).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    result = {
        "device": int(args.device),
        "rows": rows,
        "k": k,
        "n": n,
        "group": group,
        "base_m": base_m,
        "base_n": base_n,
        "base_k": base_k,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "pass": max_abs <= args.max_abs and mean_abs <= args.mean_abs,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["pass"] else 2


def _launch_worker(device: int, case: dict[str, Any], args: argparse.Namespace) -> subprocess.Popen[str]:
    env = os.environ.copy()
    _apply_custom_opp_env(env, args)
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(device)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--device",
        str(device),
        "--bridge-lib",
        str(args.bridge_lib),
        "--case-json",
        json.dumps(_case_payload(case, args), sort_keys=True),
        "--max-abs",
        str(args.max_abs),
        "--mean-abs",
        str(args.mean_abs),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)


def _run_parent(args: argparse.Namespace) -> int:
    if args.cases_json:
        cases = json.loads(Path(args.cases_json).expanduser().read_text(encoding="utf-8"))
    else:
        cases = list(DEFAULT_LOCAL_A_CASES)
    devices = args.devices
    procs: list[tuple[int, dict[str, Any], subprocess.Popen[str]]] = []
    for index, case in enumerate(cases):
        procs.append((devices[index % len(devices)], case, _launch_worker(devices[index % len(devices)], case, args)))

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for device, case, proc in procs:
        try:
            stdout, stderr = proc.communicate(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            failures.append({"device": device, "case": case, "timeout_s": args.timeout, "stderr_tail": _stderr_tail(stderr)})
            continue

        result = _last_json_object(stdout)
        if proc.returncode != 0 or result is None:
            failures.append(
                {
                    "device": device,
                    "case": case,
                    "returncode": proc.returncode,
                    "stdout_tail": _stderr_tail(stdout),
                    "stderr_tail": _stderr_tail(stderr),
                }
            )
            continue
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    summary = {
        "all_pass": len(results) == len(cases) and all(row["pass"] for row in results) and not failures,
        "count": len(results),
        "expected": len(cases),
        "max_abs_max": max((row["max_abs"] for row in results), default=None),
        "mean_abs_max": max((row["mean_abs"] for row in results), default=None),
        "failures": failures,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    if args.summary_json:
        Path(args.summary_json).expanduser().write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if summary["all_pass"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Komodo-CANN Ascend C raw custom-op package.")
    parser.add_argument("--bridge-lib", required=True, help="Path to gptqmodel_komodo_cann_ascendc_ops.so.")
    parser.add_argument("--opp-install", help="Custom OPP install prefix containing vendors/customize.")
    parser.add_argument("--opapi-lib", help="Explicit libcust_opapi.so path; overrides --opp-install discovery.")
    parser.add_argument("--devices", type=_parse_devices, default=[0], help="Comma-separated physical NPU IDs.")
    parser.add_argument("--cases-json", help="Optional JSON file containing a list of case dictionaries.")
    parser.add_argument("--summary-json", help="Optional path to write a JSON summary.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Seconds to allow each worker process.")
    parser.add_argument("--base-m", type=int, default=16)
    parser.add_argument("--base-n", type=int, default=-256)
    parser.add_argument("--base-k", type=int, default=-128)
    parser.add_argument("--max-abs", type=float, default=0.02)
    parser.add_argument("--mean-abs", type=float, default=0.004)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--device", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--case-json", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        return _worker(args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
