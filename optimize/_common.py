#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for repository quantization command-line tools."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional, Union


def set_env() -> None:
    """Configure process defaults before importing numerical libraries."""
    if sys.platform == "darwin":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ.setdefault(
        "PYTORCH_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:1024,garbage_collection_threshold:0.5",
    )
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
    os.environ.setdefault("OMP_NESTED", "FALSE")


def set_torch_threads() -> None:
    """Cap PyTorch thread pools used by the command-line tools."""
    import torch

    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)


def parse_gpus(gpu_arg: str) -> list[int]:
    """Parse and validate comma-separated physical GPU identifiers."""
    if not isinstance(gpu_arg, str):
        raise ValueError(f"Invalid physical GPU list: {gpu_arg!r}")

    gpu_tokens = [token.strip() for token in gpu_arg.split(",")]
    if any(not token for token in gpu_tokens):
        raise ValueError(f"Invalid physical GPU list: {gpu_arg!r}")

    try:
        physical_gpus = [int(token) for token in gpu_tokens]
    except ValueError as exc:
        raise ValueError(f"Invalid physical GPU list: {gpu_arg!r}") from exc

    if any(gpu < 0 for gpu in physical_gpus):
        raise ValueError(f"Physical GPU ids must be nonnegative: {physical_gpus}")
    if len(set(physical_gpus)) != len(physical_gpus):
        raise ValueError(f"Physical GPU ids must be unique: {physical_gpus}")
    return physical_gpus


def idle_gate(
    physical_gpus: list[int],
    samples: int = 3,
    interval: float = 2.0,
    memory_slack_mb: int = 256,
    timeout: float = 120.0,
) -> dict[int, dict]:
    """Wait for the selected physical GPUs to remain idle for several samples."""
    accepted: dict[int, dict] = {}
    consecutive = 0
    last_seen: dict[int, dict] = {}
    deadline = time.monotonic() + timeout
    wanted = set(physical_gpus)

    while consecutive < samples:
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"physical GPUs {physical_gpus} did not become idle within {timeout:.0f}s: "
                f"last samples {last_seen} (idle contract: util==0%, mem<= {memory_slack_mb}MiB "
                f"for {samples} consecutive samples)"
            )
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,pci.bus_id,uuid,name,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        seen: set[int] = set()
        all_idle = True
        for line in result.stdout.strip().splitlines():
            idx, pci, uuid, name, mem_used, util = [
                part.strip() for part in line.split(",")
            ]
            idx = int(idx)
            if idx not in wanted:
                continue
            seen.add(idx)
            util_int = int(util)
            mem_used_int = int(mem_used)
            print(
                f"[preflight] GPU {idx} ({name}) util={util_int}% mem_used={mem_used_int}MiB",
                flush=True,
            )
            last_seen[idx] = {
                "pci": pci,
                "uuid": uuid,
                "name": name,
                "util_pct": util_int,
                "memory_used_mb": mem_used_int,
            }
            if util_int == 0 and mem_used_int <= memory_slack_mb:
                accepted[idx] = {
                    "physical_id": idx,
                    "pci": pci,
                    "uuid": uuid,
                    "name": name,
                    "memory_used_mb": mem_used_int,
                }
            else:
                all_idle = False

        missing = wanted - seen
        if missing:
            raise RuntimeError(
                f"physical GPUs {sorted(missing)} not found in nvidia-smi output"
            )
        if all_idle:
            consecutive += 1
        else:
            consecutive = 0
            accepted = {}
        if consecutive < samples:
            time.sleep(interval)

    for idx in sorted(accepted):
        info = accepted[idx]
        print(
            f"[preflight] Accepted GPU {idx} pci={info['pci']} uuid={info['uuid']} "
            f"after {samples} idle samples (util==0%, mem<= {memory_slack_mb}MiB)",
            flush=True,
        )
    return accepted


def verify_visible_uuids(physical_gpus: list[int], gpu_infos: dict[int, dict]) -> None:
    """Confirm CUDA-visible device order matches the requested physical ids."""
    import torch

    torch.cuda.init()
    for local_idx, physical_id in enumerate(physical_gpus):
        visible_uuid = (
            str(torch.cuda.get_device_properties(local_idx).uuid)
            .replace("-", "")
            .lower()
        )
        requested_uuid = (
            gpu_infos[physical_id]["uuid"].removeprefix("GPU-").replace("-", "").lower()
        )
        if visible_uuid != requested_uuid:
            raise RuntimeError(
                f"visible GPU cuda:{local_idx} UUID {visible_uuid} does not match requested "
                f"physical GPU {physical_id} ({gpu_infos[physical_id]['uuid']})"
            )
        print(
            f"[preflight] cuda:{local_idx} == physical GPU {physical_id} "
            f"({gpu_infos[physical_id]['uuid']})",
            flush=True,
        )


def _add_text(example: dict[str, Any]) -> dict[str, Any]:
    """Add a plain-text fallback for chat-style calibration rows."""
    messages = example.get("messages", [])
    if messages and "text" not in example:
        example["text"] = "\n\n".join(
            message.get("content", "") for message in messages
        )
    return example


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(path)
    if "messages" not in frame.columns:
        raise ValueError(
            f"Calibration parquet must contain a 'messages' column: {path}"
        )
    return [
        _add_text({"messages": list(messages)})
        for messages in frame["messages"].tolist()
    ]


def load_calibration_data(
    parquet_path: Optional[Path] = None,
    dataset_path: Optional[Path] = None,
    dataset_name: Optional[str] = None,
    dataset_split: str = "train",
    dataset_size: int = 0,
) -> list[Union[str, dict[str, Any]]]:
    """Load calibration rows from parquet or a Hugging Face dataset."""
    if parquet_path is not None:
        rows = _load_parquet(Path(parquet_path))
    elif dataset_path is not None:
        path = Path(dataset_path)
        if path.is_file() and path.suffix == ".parquet":
            rows = _load_parquet(path)
        elif path.is_dir() and (path / "calibration.parquet").exists():
            rows = _load_parquet(path / "calibration.parquet")
        else:
            from datasets import load_dataset

            dataset = load_dataset(str(path), dataset_name, split=dataset_split)
            rows = [dict(row) for row in dataset.map(_add_text)]
    else:
        raise ValueError("One of --calibration-parquet or --dataset-path is required")

    if dataset_size > 0:
        rows = rows[:dataset_size]
    print(f"[data] Loaded {len(rows)} calibration rows", flush=True)
    return rows


def add_requant_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by embedding requantization commands."""
    parser.add_argument(
        "--model-path", required=True, help="Existing quantized model checkpoint."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output checkpoint directory; a suffix is added to --model-path when omitted.",
    )
    parser.add_argument(
        "--gpus", default="0", help="Comma-separated physical, PCI-bus-ordered GPU ids."
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for idle GPUs.",
    )
    parser.add_argument(
        "--calibration-parquet",
        type=Path,
        default=None,
        help="Parquet with a messages column.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Dataset name or local dataset path.",
    )
    parser.add_argument(
        "--dataset-name", default=None, help="Dataset configuration name."
    )
    parser.add_argument("--dataset-split", default="train", help="Dataset split.")
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=0,
        help="Limit calibration rows; 0 uses all rows.",
    )
    parser.add_argument("--bits", type=int, default=4, help="Embedding weight bits.")
    parser.add_argument(
        "--group-size", type=int, default=64, help="Embedding quantization group size."
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Calibration forward batch size."
    )
    parser.add_argument(
        "--calibration-concat-size",
        type=int,
        default=4096,
        help="Token concatenation size.",
    )
    parser.add_argument(
        "--calibration-concat-separator",
        default="\n",
        help="Separator used when concatenating calibration text.",
    )
    parser.add_argument(
        "--calibration-sort",
        default="desc",
        help="Calibration concatenation sort order.",
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote model code."
    )
    parser.add_argument(
        "--desc-act", action="store_true", help="Use activation ordering."
    )
    parser.add_argument(
        "--no-act-group-aware",
        action="store_true",
        help="Disable activation-group awareness.",
    )
    parser.add_argument(
        "--scale-search", default="activation", help="Scale-search objective."
    )
