# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Command-line client for the GPU allocator daemon."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Optional, Sequence

from .client import GPUAllocatorClient, GPUAllocatorError


def _client_from_args(args: argparse.Namespace) -> GPUAllocatorClient:
    session_id = (
        args.session_id
        or os.environ.get("DEVIN_OUTPOST_SESSION_ID")
        or os.environ.get("DEVIN_SESSION_ID")
        or f"pid-{os.getpid()}"
    )
    return GPUAllocatorClient(base_url=args.base_url, session_id=session_id)


def _timeout_value(raw: Optional[float]) -> Optional[float]:
    """Convert CLI timeout value: negative / None means block indefinitely, 0 means immediate."""
    if raw is None or raw < 0:
        return None
    return raw


def _print_json(obj: object) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _lease_to_export(lease, style: str) -> str:
    """Render shell export statements for a lease."""
    values = lease.as_cuda_visible_devices(style=style, set_env=False)
    lines = [
        f"export GPU_ALLOCATOR_LEASE_ID={lease.lease_id}",
        f"export GPU_ALLOCATOR_GPU_STYLE={style}",
    ]
    if style == "pci_order_index":
        lines.append("export CUDA_DEVICE_ORDER=PCI_BUS_ID")
    lines.append(f"export CUDA_VISIBLE_DEVICES={values}")
    return "\n".join(lines)


def _print_lease(lease, args: argparse.Namespace) -> None:
    fmt = args.format
    style = args.style
    if fmt == "json":
        _print_json(
            {
                "ok": True,
                "lease_id": lease.lease_id,
                "gpus": [gpu.to_dict() for gpu in lease.gpus],
                "cuda_visible_devices": lease.as_cuda_visible_devices(
                    style=style, set_env=False
                ),
            }
        )
    elif fmt == "shell":
        print(_lease_to_export(lease, style))
    elif fmt == "ids":
        print(lease.as_cuda_visible_devices(style=style, set_env=False))
    else:
        raise ValueError(f"Unknown output format: {fmt!r}")


def cmd_acquire(args: argparse.Namespace) -> int:
    client = _client_from_args(args)
    try:
        lease = client.allocate(
            count=args.count,
            timeout=_timeout_value(args.timeout),
            reason=args.reason,
            exclusive=args.exclusive,
        )
    except GPUAllocatorError as exc:
        _print_json({"ok": False, "error": exc.message})
        return 1
    _print_lease(lease, args)
    return 0


def cmd_release(args: argparse.Namespace) -> int:
    client = _client_from_args(args)
    try:
        client.release(args.lease_id)
    except GPUAllocatorError as exc:
        _print_json({"ok": False, "error": exc.message})
        return 1
    _print_json({"ok": True})
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    client = _client_from_args(args)
    try:
        status = client.status()
    except GPUAllocatorError as exc:
        _print_json({"ok": False, "error": exc.message})
        return 1
    _print_json(status)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Acquire GPUs, run a subprocess with CUDA_VISIBLE_DEVICES set, then release."""
    client = _client_from_args(args)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        _print_json({"ok": False, "error": "No command provided"})
        return 1

    try:
        lease = client.allocate(
            count=args.count,
            timeout=_timeout_value(args.timeout),
            reason=args.reason,
            exclusive=args.exclusive,
        )
    except GPUAllocatorError as exc:
        _print_json({"ok": False, "error": exc.message})
        return 1

    style = args.style
    visible_devices = lease.as_cuda_visible_devices(style=style, set_env=False)
    env = os.environ.copy()
    env["GPU_ALLOCATOR_LEASE_ID"] = lease.lease_id
    env["GPU_ALLOCATOR_GPU_STYLE"] = style
    env["CUDA_VISIBLE_DEVICES"] = visible_devices
    if style == "pci_order_index":
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    try:
        result = subprocess.run(command, env=env)
        return result.returncode
    finally:
        try:
            client.release(lease.lease_id)
        except GPUAllocatorError as exc:
            print(
                f"Warning: failed to release lease {lease.lease_id}: {exc.message}",
                file=sys.stderr,
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpu-alloc",
        description="CLI client for the GPU allocator daemon.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("GPU_ALLOCATOR_URL", "http://127.0.0.1:17351"),
        help="Allocator server URL (default: $GPU_ALLOCATOR_URL or http://127.0.0.1:17351)",
    )
    parser.add_argument(
        "--session-id",
        default=os.environ.get("DEVIN_OUTPOST_SESSION_ID")
        or os.environ.get("DEVIN_SESSION_ID")
        or f"pid-{os.getpid()}",
        help="Session identifier (default: $DEVIN_OUTPOST_SESSION_ID, $DEVIN_SESSION_ID, or pid-<pid>)",
    )
    parser.set_defaults(func=lambda _: parser.print_help() or 1)

    subparsers = parser.add_subparsers(dest="command")

    # status
    status_parser = subparsers.add_parser("status", help="Show allocator status")
    status_parser.set_defaults(func=cmd_status)

    # acquire
    acquire_parser = subparsers.add_parser(
        "acquire", help="Acquire a GPU lease and print it"
    )
    acquire_parser.add_argument(
        "-n", "--count", type=int, required=True, help="Number of GPUs"
    )
    acquire_parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=0.0,
        help="Seconds to wait; 0 means immediate, negative means indefinite (default: 0)",
    )
    acquire_parser.add_argument(
        "--reason", default=None, help="Reason for the allocation"
    )
    acquire_parser.add_argument(
        "--exclusive",
        action="store_true",
        default=True,
        help="Request exclusive access (default)",
    )
    acquire_parser.add_argument(
        "--shared",
        action="store_false",
        dest="exclusive",
        help="Request shared (non-exclusive) access",
    )
    acquire_parser.add_argument(
        "--style",
        choices=("uuid", "pci_bus_id", "pci_order_index"),
        default="uuid",
        help="How to identify GPUs in output (default: uuid)",
    )
    acquire_parser.add_argument(
        "--format",
        choices=("json", "shell", "ids"),
        default="json",
        help="Output format (default: json)",
    )
    acquire_parser.set_defaults(func=cmd_acquire)

    # release
    release_parser = subparsers.add_parser("release", help="Release a GPU lease")
    release_parser.add_argument("--lease-id", required=True, help="Lease id to release")
    release_parser.set_defaults(func=cmd_release)

    # run
    run_parser = subparsers.add_parser(
        "run",
        help="Acquire GPUs and run a command with CUDA_VISIBLE_DEVICES set",
    )
    run_parser.add_argument(
        "-n", "--count", type=int, required=True, help="Number of GPUs"
    )
    run_parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=None,
        help="Seconds to wait; omitted or negative means indefinite (default: indefinite)",
    )
    run_parser.add_argument("--reason", default=None, help="Reason for the allocation")
    run_parser.add_argument(
        "--exclusive",
        action="store_true",
        default=True,
        help="Request exclusive access (default)",
    )
    run_parser.add_argument(
        "--shared",
        action="store_false",
        dest="exclusive",
        help="Request shared (non-exclusive) access",
    )
    run_parser.add_argument(
        "--style",
        choices=("uuid", "pci_bus_id", "pci_order_index"),
        default="uuid",
        help="How to identify GPUs for CUDA_VISIBLE_DEVICES (default: uuid)",
    )
    run_parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command and arguments to run",
    )
    run_parser.set_defaults(func=cmd_run)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 1
    return func(args)


if __name__ == "__main__":
    sys.exit(main())
