#!/usr/bin/env python3
"""Matched H100 A/B for the F6 M960 large-M2 P32 projection."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def _idle_preflight(uuid: str) -> None:
    for sample in range(3):
        output = subprocess.check_output(
            [
                "nvidia-smi", "-i", uuid,
                "--query-gpu=uuid,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        observed_uuid, utilization, memory = (part.strip() for part in output.split(","))
        if observed_uuid != uuid or int(utilization) != 0 or int(memory) != 0:
            raise RuntimeError(f"GPU is not idle: sample={sample} {output}")
        processes = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
            text=True,
        )
        if uuid in processes:
            raise RuntimeError(f"GPU has a compute process: {processes}")
        time.sleep(0.2)
    print(f"idle_preflight_passed uuid={uuid} samples=3 memory_mib=0 utilization=0", flush=True)


def _load_arm(path: Path):
    library = ctypes.CDLL(str(path.resolve()), mode=ctypes.RTLD_LOCAL)
    call = library.qvq_p32_window_with_row_groups
    call.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 11 + [ctypes.c_void_p]
    call.restype = ctypes.c_int
    last_error = library.qvq_last_error
    last_error.argtypes = []
    last_error.restype = ctypes.c_char_p
    return call, last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--rounds", type=int, default=50)
    args = parser.parse_args()
    if args.rounds < 10:
        parser.error("at least 10 paired rounds are required")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    _idle_preflight(args.gpu_uuid)

    import torch

    baseline = _load_arm(args.baseline)
    candidate = _load_arm(args.candidate)
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    if props.major != 9 or props.minor != 0:
        raise RuntimeError(f"expected sm_90, got sm_{props.major}{props.minor}")
    m, k, n, bits = 960, 2048, 8192, 6
    seed = 20260923
    generator = torch.Generator().manual_seed(seed)
    x = (torch.randn((m, k), generator=generator) * 0.02).half().to(device)
    tiles = (k // 16) * (n // 16)
    trellis = torch.randint(
        -(2**31), 2**31 - 1, (tiles * 4 * bits,), generator=generator, dtype=torch.int32
    )
    trellis = trellis.to(device)
    levels = torch.linspace(-1, 1, 256, dtype=torch.float16, device=device)
    banks = torch.randint(0, 256, (tiles,), generator=generator, dtype=torch.uint8).to(device)
    alt = torch.tensor([3], dtype=torch.uint8, device=device)
    outputs = [torch.empty((m, n), dtype=torch.float32, device=device) for _ in range(2)]
    partials = [torch.empty_like(output) for output in outputs]
    cuda_stream = torch.cuda.Stream(device=device)
    cuda_stream.wait_stream(torch.cuda.current_stream(device))
    stream = cuda_stream.cuda_stream
    calls = [baseline, candidate]

    def invoke(arm: int) -> None:
        status = calls[arm][0](
            x.data_ptr(), trellis.data_ptr(), levels.data_ptr(), banks.data_ptr(), alt.data_ptr(),
            outputs[arm].data_ptr(), partials[arm].data_ptr(),
            m, k, n, bits, 1, 2, 128, 3, 0, 1, 4, stream,
        )
        if status != 0:
            raise RuntimeError(f"arm {arm} returned {status}: {calls[arm][1]().decode()}")

    for _ in range(20):
        invoke(0)
        invoke(1)
    cuda_stream.synchronize()
    if not torch.equal(outputs[0], outputs[1]):
        delta = (outputs[0] - outputs[1]).abs()
        raise RuntimeError(f"kernel parity failed: mae={delta.mean().item()} max={delta.max().item()}")

    timings = [[], []]
    for iteration in range(args.rounds):
        for arm in ((0, 1) if iteration % 2 == 0 else (1, 0)):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record(cuda_stream)
            invoke(arm)
            stop.record(cuda_stream)
            stop.synchronize()
            timings[arm].append(start.elapsed_time(stop) * 1000.0)
    if not torch.equal(outputs[0], outputs[1]):
        raise RuntimeError("post-timing kernel parity failed")
    print(json.dumps({
        "gpu_uuid": args.gpu_uuid,
        "gpu_name": props.name,
        "model_shape": {
            "m": m, "k": k, "n": n, "transition_bits": bits,
            "row_groups": 6, "stage_k_tiles": 3,
        },
        "seed": seed,
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        "bitwise_equal": True,
        "baseline_us": sorted(timings[0])[len(timings[0]) // 2],
        "candidate_us": sorted(timings[1])[len(timings[1]) // 2],
        "baseline_samples_us": timings[0],
        "candidate_samples_us": timings[1],
    }))


if __name__ == "__main__":
    main()
