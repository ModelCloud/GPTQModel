#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
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


def _apply_quiet_cann_env(env: dict[str, str]) -> None:
    env.setdefault("ASCEND_GLOBAL_LOG_LEVEL", "3")
    env.setdefault("ASCEND_SLOG_PRINT_TO_STDOUT", "0")


def _enable_quiet_process_output(enabled: bool) -> int | None:
    if not enabled:
        return None
    _apply_quiet_cann_env(os.environ)
    result_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
    finally:
        os.close(devnull_fd)
    return result_fd


def _emit_json_result(result: dict[str, Any], result_fd: int | None) -> None:
    line = json.dumps(result, sort_keys=True) + "\n"
    if result_fd is None:
        print(line, end="", flush=True)
    else:
        os.write(result_fd, line.encode("utf-8"))


def _apply_custom_opp_env(env: dict[str, str], args: argparse.Namespace) -> None:
    if args.opp_install:
        vendor = Path(args.opp_install).expanduser() / "vendors" / "customize"
        opapi_lib = vendor / "op_api" / "lib" / "libcust_opapi.so"
        env["ASCEND_CUSTOM_OPP_PATH"] = str(vendor) + os.pathsep + env.get("ASCEND_CUSTOM_OPP_PATH", "")
        env["LD_LIBRARY_PATH"] = str(opapi_lib.parent) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        env.setdefault("GPTQMODEL_CANNOE_ASCENDC_OPAPI_LIB", str(opapi_lib))
    if args.opapi_lib:
        env["GPTQMODEL_CANNOE_ASCENDC_OPAPI_LIB"] = str(Path(args.opapi_lib).expanduser())


def _case_payload(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    payload = dict(case)
    payload.setdefault("base_m", args.base_m)
    payload.setdefault("base_n", args.base_n)
    payload.setdefault("base_k", args.base_k)
    return payload


def _timing_stats(results: list[dict[str, Any]]) -> dict[str, float | None]:
    timings = [float(row["custom_ms"]) for row in results if row.get("custom_ms") is not None]
    if not timings:
        return {"custom_ms_min": None, "custom_ms_mean": None, "custom_ms_max": None}
    return {
        "custom_ms_min": min(timings),
        "custom_ms_mean": sum(timings) / len(timings),
        "custom_ms_max": max(timings),
    }


def _device_case_batches(devices: list[int], cases: list[dict[str, Any]]) -> list[list[tuple[int, dict[str, Any]]]]:
    return [
        [(devices[offset], case) for offset, case in enumerate(cases[start : start + len(devices)])]
        for start in range(0, len(cases), len(devices))
    ]


def _worker(args: argparse.Namespace) -> int:
    result_fd = _enable_quiet_process_output(not args.no_quiet_cann_logs)
    try:
        return _worker_impl(args, result_fd)
    except Exception as err:
        _emit_json_result(
            {
                "device": int(args.device),
                "error": f"{type(err).__name__}: {err}",
                "pass": False,
            },
            result_fd,
        )
        return 2
    finally:
        if result_fd is not None:
            os.close(result_fd)


def _worker_impl(args: argparse.Namespace, result_fd: int | None) -> int:
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
    warmup = int(args.warmup)
    iters = int(args.iters)

    x = torch.randn((rows, k), device="npu", dtype=torch.float16)
    one_hot_k = int(case.get("one_hot_k", args.one_hot_k))
    if one_hot_k >= 0:
        if one_hot_k >= k:
            raise ValueError(f"one_hot_k={one_hot_k} must be less than k={k}")
        x.zero_()
        x[:, one_hot_k] = 1
    signed_weight = torch.randint(-8, 8, (k, n), device="npu", dtype=torch.int32).contiguous()
    if args.lane_diagnostic:
        pattern = torch.arange(-8, 8, device="npu", dtype=torch.int32).repeat(4)
        signed_weight.zero_()
        signed_weight[0, : pattern.numel()].copy_(pattern)
    packed_weight = torch.ops.npu.npu_convert_weight_to_int4pack(signed_weight)
    scales = torch.full((k // group, n), 0.03125, device="npu", dtype=torch.float16)
    offsets = torch.zeros((k // group, n), device="npu", dtype=torch.float16)

    def custom_call() -> torch.Tensor:
        return torch.ops.gptqmodel_cannoe.cannoe_w4_a16_matmul(
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

    if args.path_diagnostic:
        for _ in range(warmup):
            custom_call()
        torch.npu.synchronize()
        start = time.perf_counter()
        custom = None
        for _ in range(iters):
            custom = custom_call()
        torch.npu.synchronize()
        custom_ms = (time.perf_counter() - start) * 1000.0 / max(1, iters)
        if custom is None:
            custom = custom_call()
            torch.npu.synchronize()
        markers = custom.flatten()[:8].to("cpu", dtype=torch.float32)
        marker_values = [float(value) for value in markers.tolist()]
        marker = marker_values[0] if marker_values else float("nan")
        launch_mode = marker_values[1] if len(marker_values) > 1 else float("nan")
        kernel_mode = marker_values[2] if len(marker_values) > 2 else float("nan")
        block_dim = marker_values[3] if len(marker_values) > 3 else 0.0
        mixed_entry_pass = (
            abs(marker - 911.0) <= 0.5 and int(launch_mode) == 1 and int(kernel_mode) == 0 and block_dim > 0
        )
        aic_tscm_path_pass = (
            abs(marker - 910.0) <= 0.5
            and len(marker_values) >= 8
            and int(marker_values[1]) == 1
            and marker_values[6] > 0
            and marker_values[7] > 0
        )
        result = {
            "device": int(args.device),
            "rows": rows,
            "k": k,
            "n": n,
            "group": group,
            "base_m": base_m,
            "base_n": base_n,
            "base_k": base_k,
            "custom_ms": custom_ms,
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "pass": mixed_entry_pass or aic_tscm_path_pass,
            "markers": marker_values,
        }
        _emit_json_result(result, result_fd)
        return 0 if result["pass"] else 2

    if args.lane_diagnostic:
        custom = custom_call()
        torch.npu.synchronize()
        diagnostic = custom[0, :192].to("cpu", dtype=torch.float32)
        capi_vector = diagnostic[:64]
        scalar = diagnostic[64:128]
        cast_vector = diagnostic[128:192]
        capi_diff = (capi_vector - scalar).abs()
        cast_diff = (cast_vector - scalar).abs()
        capi_max_abs = float(capi_diff.max().item())
        capi_mean_abs = float(capi_diff.mean().item())
        cast_max_abs = float(cast_diff.max().item())
        cast_mean_abs = float(cast_diff.mean().item())
        capi_pass = capi_max_abs <= args.max_abs and capi_mean_abs <= args.mean_abs
        cast_pass = cast_max_abs <= args.max_abs and cast_mean_abs <= args.mean_abs
        result = {
            "device": int(args.device),
            "rows": rows,
            "k": k,
            "n": n,
            "group": group,
            "base_m": base_m,
            "base_n": base_n,
            "base_k": base_k,
            "custom_ms": None,
            "max_abs": min(capi_max_abs, cast_max_abs),
            "mean_abs": min(capi_mean_abs, cast_mean_abs),
            "pass": capi_pass or cast_pass,
            "capi_max_abs": capi_max_abs,
            "capi_mean_abs": capi_mean_abs,
            "cast_max_abs": cast_max_abs,
            "cast_mean_abs": cast_mean_abs,
            "best_path": "cast" if cast_max_abs <= capi_max_abs else "capi",
            "capi_first16": [float(value) for value in capi_vector[:16].tolist()],
            "cast_first16": [float(value) for value in cast_vector[:16].tolist()],
            "scalar_first16": [float(value) for value in scalar[:16].tolist()],
        }
        _emit_json_result(result, result_fd)
        return 0 if result["pass"] else 2

    if args.tile_fill_diagnostic:
        custom = custom_call()
        torch.npu.synchronize()
        diagnostic = custom.flatten()[:256].to("cpu", dtype=torch.float32)
        scalar = diagnostic[:128]
        cast_vector = diagnostic[128:256]
        diff = (cast_vector - scalar).abs()
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
            "custom_ms": None,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "pass": max_abs <= args.max_abs and mean_abs <= args.mean_abs,
            "scalar_first16": [float(value) for value in scalar[:16].tolist()],
            "cast_first16": [float(value) for value in cast_vector[:16].tolist()],
        }
        _emit_json_result(result, result_fd)
        return 0 if result["pass"] else 2

    for _ in range(warmup):
        custom_call()
    torch.npu.synchronize()
    start = time.perf_counter()
    custom = None
    for _ in range(iters):
        custom = custom_call()
    torch.npu.synchronize()
    custom_ms = (time.perf_counter() - start) * 1000.0 / max(1, iters)
    if custom is None:
        custom = custom_call()
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
        "one_hot_k": one_hot_k,
        "custom_ms": custom_ms,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "pass": max_abs <= args.max_abs and mean_abs <= args.mean_abs,
    }
    _emit_json_result(result, result_fd)
    return 0 if result["pass"] else 2


def _launch_worker(device: int, case: dict[str, Any], args: argparse.Namespace) -> subprocess.Popen[str]:
    env = os.environ.copy()
    _apply_custom_opp_env(env, args)
    if not args.no_quiet_cann_logs:
        _apply_quiet_cann_env(env)
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
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--one-hot-k",
        str(args.one_hot_k),
    ]
    if args.no_quiet_cann_logs:
        cmd.append("--no-quiet-cann-logs")
    if args.lane_diagnostic:
        cmd.append("--lane-diagnostic")
    if args.tile_fill_diagnostic:
        cmd.append("--tile-fill-diagnostic")
    if args.path_diagnostic:
        cmd.append("--path-diagnostic")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, start_new_session=True)


def _kill_worker(proc: subprocess.Popen[str]) -> tuple[str, str]:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        proc.kill()
    return proc.communicate()


def _record_worker_result(
    *,
    device: int,
    case: dict[str, Any],
    proc: subprocess.Popen[str],
    stdout: str,
    stderr: str,
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
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
        return
    results.append(result)
    print(json.dumps(result, sort_keys=True), flush=True)


def _run_parent(args: argparse.Namespace) -> int:
    if args.cases_json:
        cases = json.loads(Path(args.cases_json).expanduser().read_text(encoding="utf-8"))
    else:
        cases = list(DEFAULT_LOCAL_A_CASES)
    devices = args.devices
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for batch in _device_case_batches(devices, cases):
        running = [
            {"device": device, "case": case, "proc": _launch_worker(device, case, args), "start": time.monotonic()}
            for device, case in batch
        ]
        while running:
            next_running: list[dict[str, Any]] = []
            for item in running:
                proc = item["proc"]
                device = item["device"]
                case = item["case"]
                if proc.poll() is not None:
                    stdout, stderr = proc.communicate()
                    _record_worker_result(
                        device=device,
                        case=case,
                        proc=proc,
                        stdout=stdout,
                        stderr=stderr,
                        results=results,
                        failures=failures,
                    )
                    continue
                elapsed = time.monotonic() - float(item["start"])
                if elapsed >= args.timeout:
                    stdout, stderr = _kill_worker(proc)
                    failures.append(
                        {"device": device, "case": case, "timeout_s": args.timeout, "stderr_tail": _stderr_tail(stderr)}
                    )
                    continue
                next_running.append(item)
            running = next_running
            if running:
                time.sleep(0.1)

    summary = {
        "all_pass": len(results) == len(cases) and all(row["pass"] for row in results) and not failures,
        "count": len(results),
        "expected": len(cases),
        "max_abs_max": max((row["max_abs"] for row in results), default=None),
        "mean_abs_max": max((row["mean_abs"] for row in results), default=None),
        "warmup": args.warmup,
        "iters": args.iters,
        "failures": failures,
    }
    summary.update(_timing_stats(results))
    print(json.dumps(summary, sort_keys=True), flush=True)
    if args.summary_json:
        Path(args.summary_json).expanduser().write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if summary["all_pass"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Cannoe Ascend C raw custom-op package.")
    parser.add_argument("--bridge-lib", required=True, help="Path to gptqmodel_cannoe_ascendc_ops.so.")
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
    parser.add_argument("--warmup", type=int, default=0, help="Untimed custom-op warmup calls per worker.")
    parser.add_argument("--iters", type=int, default=1, help="Timed custom-op calls per worker.")
    parser.add_argument("--one-hot-k", type=int, default=-1, help="Use one-hot activations at this K index for diagnostics.")
    parser.add_argument(
        "--no-quiet-cann-logs",
        action="store_true",
        help="Do not suppress noisy CANN registration warnings in worker subprocesses.",
    )
    parser.add_argument(
        "--lane-diagnostic",
        action="store_true",
        help=(
            "Run the opt-in int4 lane-layout diagnostic instead of the matmul accuracy benchmark. "
            "Requires a package built with --experimental-int4-lane-diagnostic."
        ),
    )
    parser.add_argument(
        "--tile-fill-diagnostic",
        action="store_true",
        help=(
            "Run the opt-in B-tile fill diagnostic instead of the matmul accuracy benchmark. "
            "Requires a package built with --experimental-vecout-tile-fill-diagnostic."
        ),
    )
    parser.add_argument(
        "--path-diagnostic",
        action="store_true",
        help=(
            "Run the opt-in mixed-entry marker diagnostic instead of the matmul accuracy benchmark. "
            "Requires a package built with --experimental-mixed-entry-diagnostic."
        ),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--device", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--case-json", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.iters <= 0:
        parser.error("--iters must be > 0")
    diagnostic_count = sum(bool(value) for value in (args.lane_diagnostic, args.tile_fill_diagnostic, args.path_diagnostic))
    if diagnostic_count > 1:
        parser.error("--lane-diagnostic, --tile-fill-diagnostic, and --path-diagnostic are mutually exclusive")

    if args.worker:
        return _worker(args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
