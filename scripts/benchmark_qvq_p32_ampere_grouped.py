#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Compare grouped Phase-2 P32 execution with independent child launches."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

RATES = (2.0, 2.5, 3.0, 3.5)
M_VALUES = (1, 2, 4, 8, 16)
GROUPS = {
    "qkv": {
        "k": 2048,
        "child_ns": (2048, 512, 512),
        "bank_alt_ids": (3, 1, 2),
    },
    "gate_up": {
        "k": 2048,
        "child_ns": (8192, 8192),
        "bank_alt_ids": (1, 3),
    },
}
SOURCE_PATHS = (
    Path("gptqmodel/utils/qvq_ampere_cuda.py"),
    Path("gptqmodel_ext/qvq/qvq_ampere_cuda.cu"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("plain", "grouped"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--rates", nargs="+", type=float, default=RATES)
    parser.add_argument("--m-values", nargs="+", type=int, default=M_VALUES)
    parser.add_argument(
        "--groups", nargs="+", choices=tuple(GROUPS), default=tuple(GROUPS)
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--replays-per-sample", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    if any(rate not in RATES for rate in args.rates):
        parser.error("--rates supports only 2, 2.5, 3, and 3.5")
    if any(m not in M_VALUES for m in args.m_values):
        parser.error("--m-values supports only 1, 2, 4, 8, and 16")
    if min(args.warmup, args.samples, args.replays_per_sample) <= 0:
        parser.error("timing counts must be positive")
    return args


def _git_output(repo_root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=repo_root, text=True
    ).strip()


def _source_fingerprint(repo_root: Path) -> str:
    digest = hashlib.sha256()
    digest.update(_git_output(repo_root, "rev-parse", "HEAD").encode())
    for relative_path in SOURCE_PATHS:
        digest.update(str(relative_path).encode())
        digest.update((repo_root / relative_path).read_bytes())
    return digest.hexdigest()


def _h100_uuid() -> str:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name",
            "--format=csv,noheader",
        ],
        text=True,
    )
    h100s = []
    for line in output.splitlines():
        uuid, name = (value.strip() for value in line.split(",", 1))
        if "H100" in name:
            h100s.append((uuid, name))
    if len(h100s) != 1:
        raise RuntimeError(f"expected exactly one physical H100, found {h100s}")
    return h100s[0][0]


def _assert_exclusive_h100(torch: Any, expected_uuid: str) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("benchmark requires exactly one CUDA-visible device")
    properties = torch.cuda.get_device_properties(0)
    actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
    normalized_actual_uuid = actual_uuid.removeprefix("GPU-")
    normalized_expected_uuid = expected_uuid.removeprefix("GPU-")
    if (
        "H100" not in properties.name
        or (properties.major, properties.minor) != (9, 0)
        or normalized_actual_uuid.lower() != normalized_expected_uuid.lower()
    ):
        raise RuntimeError(
            f"expected H100 {expected_uuid}, got {properties.name} "
            f"sm_{properties.major}{properties.minor} {actual_uuid}"
        )
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    foreign = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if (
            len(fields) >= 2
            and fields[0].lower() == expected_uuid.lower()
            and int(fields[1]) != os.getpid()
        ):
            foreign.append(line)
    if foreign:
        raise RuntimeError(f"foreign H100 compute processes detected: {foreign}")
    return {
        "name": properties.name,
        "uuid": expected_uuid,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "sm_count": properties.multi_processor_count,
        "memory_bytes": properties.total_memory,
    }


def _split_counts(group_name: str, size_m: int, child_count: int) -> tuple[int, ...]:
    if size_m <= 2:
        split = 32
    elif size_m == 4:
        split = 40
    elif group_name == "qkv":
        split = 8
    else:
        split = 3
    return (split,) * child_count


def _graph_event_timing(
    torch: Any,
    call: Callable[[], Sequence[Any]],
    *,
    warmup: int,
    samples: int,
    replays_per_sample: int,
) -> tuple[dict[str, float], Any, Sequence[Any]]:
    for _ in range(3):
        eager_outputs = call()
    torch.cuda.synchronize()
    del eager_outputs

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_outputs = call()
    torch.cuda.synchronize()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    stream = torch.cuda.current_stream()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    for start, end in zip(starts, ends, strict=True):
        start.record(stream)
        for _ in range(replays_per_sample):
            graph.replay()
        end.record(stream)
    ends[-1].synchronize()
    values = [
        start.elapsed_time(end) / replays_per_sample
        for start, end in zip(starts, ends, strict=True)
    ]
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    timing = {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "p95_ms": ordered[p95_index],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }
    return timing, graph, captured_outputs


def _plain_group_call(
    qvq_p32_window_ampere: Callable[..., Any],
    input_tensor: Any,
    windows: Sequence[Any],
    levels: Any,
    selectors: Sequence[Any],
    bits: float,
    child_ns: Sequence[int],
    bank_alt_ids: Sequence[int],
    splits: Sequence[int],
) -> tuple[Any, ...]:
    return tuple(
        qvq_p32_window_ampere(
            input_tensor,
            window,
            levels,
            child_selectors,
            bits,
            out_features=child_n,
            bank_alt_id=alt_id,
            split_count=split,
        )
        for window, child_selectors, child_n, alt_id, split in zip(
            windows,
            selectors,
            child_ns,
            bank_alt_ids,
            splits,
            strict=True,
        )
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    expected_uuid = _h100_uuid()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {"1", expected_uuid}:
        raise RuntimeError(
            "set CUDA_VISIBLE_DEVICES to physical H100 index 1 or its UUID before launch"
        )
    os.environ.setdefault("QVQ_AMPERE_ALLOW_SM90_VALIDATION", "1")
    os.environ.setdefault("QVQ_AMPERE_AUTOTUNE", "0")
    sys.path.insert(0, str(args.repo_root))

    import torch

    from gptqmodel.quantization.qvq import (
        pack_qvq_binary_bank_ids,
        repack_p32_planar_to_window,
    )
    from gptqmodel.quantization.qvq_codecs import (
        PGC16_CODEBOOK_VERSION,
        pgc16_levels_for_version,
    )
    from gptqmodel.quantization.qvq_rates import qvq_words_per_tile
    from gptqmodel.utils.qvq_ampere_cuda import qvq_p32_window_ampere

    if args.mode == "grouped":
        from gptqmodel.utils.qvq_ampere_cuda import (
            qvq_p32_window_ampere_group_plan,
            qvq_p32_window_ampere_grouped_packed,
            qvq_pack_p32_window_ampere_group,
        )

    device = _assert_exclusive_h100(torch, expected_uuid)
    torch.cuda.set_device(0)
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous().cuda()
    commit = _git_output(args.repo_root, "rev-parse", "HEAD")
    source_fingerprint = _source_fingerprint(args.repo_root)
    rows = []
    for bits in args.rates:
        for group_name in args.groups:
            group = GROUPS[group_name]
            size_k = int(group["k"])
            child_ns = tuple(int(value) for value in group["child_ns"])
            bank_alt_ids = tuple(int(value) for value in group["bank_alt_ids"])
            generator = torch.Generator(device="cuda").manual_seed(
                20261130 + int(bits * 10) * 100 + list(GROUPS).index(group_name)
            )
            windows = []
            selectors = []
            for child_n in child_ns:
                tile_count = (size_k // 16) * (child_n // 16)
                words_per_tile = qvq_words_per_tile(
                    bits, weight_count=256, vector_size=2
                )
                planar = torch.randint(
                    0,
                    1 << 32,
                    (tile_count, words_per_tile),
                    generator=generator,
                    device="cuda",
                    dtype=torch.int64,
                ).to(torch.int32)
                windows.append(repack_p32_planar_to_window(planar, bits=bits))
                selectors.append(
                    pack_qvq_binary_bank_ids(
                        torch.randint(
                            0,
                            2,
                            (tile_count * 8,),
                            generator=generator,
                            device="cuda",
                            dtype=torch.uint8,
                        )
                    )
                )
                del planar
            windows = tuple(windows)
            selectors = tuple(selectors)

            for size_m in args.m_values:
                splits = _split_counts(group_name, size_m, len(child_ns))
                input_tensor = (
                    torch.randn((size_m, size_k), generator=generator, device="cuda")
                    * 0.1
                ).half()

                plain_call = partial(
                    _plain_group_call,
                    qvq_p32_window_ampere,
                    input_tensor,
                    windows,
                    levels,
                    selectors,
                    bits,
                    child_ns,
                    bank_alt_ids,
                    splits,
                )

                payload = None
                if args.mode == "plain":
                    call = plain_call
                else:
                    plan = qvq_p32_window_ampere_group_plan(
                        input_tensor,
                        windows,
                        levels,
                        selectors,
                        bits,
                        out_features=child_ns,
                        bank_alt_ids=bank_alt_ids,
                        split_counts=splits,
                    )
                    payload = qvq_pack_p32_window_ampere_group(windows, selectors, plan)

                    grouped_call = partial(
                        qvq_p32_window_ampere_grouped_packed,
                        input_tensor,
                        payload,
                        levels,
                    )

                    expected = plain_call()
                    actual = grouped_call()
                    torch.cuda.synchronize()
                    if not all(
                        torch.equal(expected_child, actual_child)
                        for expected_child, actual_child in zip(
                            expected, actual, strict=True
                        )
                    ):
                        raise RuntimeError(
                            f"grouped output differs from plain for {group_name} "
                            f"M{size_m} W{bits:g}"
                        )
                    del expected, actual
                    call = grouped_call

                timing, graph, captured_outputs = _graph_event_timing(
                    torch,
                    call,
                    warmup=args.warmup,
                    samples=args.samples,
                    replays_per_sample=args.replays_per_sample,
                )
                checksum = sum(
                    float(output.float().sum().item()) for output in captured_outputs
                )
                row = {
                    "bits": bits,
                    "group": group_name,
                    "m": size_m,
                    "k": size_k,
                    "child_ns": child_ns,
                    "total_n": sum(child_ns),
                    "split_counts": splits,
                    "timing": timing,
                    "checksum": checksum,
                }
                rows.append(row)
                print(
                    f"{args.label}: W{bits:g} {group_name} "
                    f"M={size_m} K={size_k} N={child_ns} splits={splits} "
                    f"median={timing['median_ms']:.6f} ms",
                    flush=True,
                )
                del graph, captured_outputs, input_tensor, payload
                gc.collect()
                torch.cuda.empty_cache()

            del windows, selectors
            gc.collect()
            torch.cuda.empty_cache()

    if source_fingerprint != _source_fingerprint(args.repo_root):
        raise RuntimeError("kernel source changed during benchmark")
    payload = {
        "label": args.label,
        "mode": args.mode,
        "repo_root": str(args.repo_root),
        "commit": commit,
        "source_fingerprint": source_fingerprint,
        "device": device,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "timing": {
            "method": "CUDA Graph replay bracketed by CUDA events",
            "warmup": args.warmup,
            "samples": args.samples,
            "replays_per_sample": args.replays_per_sample,
        },
        "dtype": "float16 input/levels, float32 accumulation/output",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"result: {args.output}", flush=True)
    return payload


def main() -> None:
    _run(_parse_args())


if __name__ == "__main__":
    main()
