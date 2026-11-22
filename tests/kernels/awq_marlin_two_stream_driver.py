# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Driver for the two-stream AWQ Marlin determinism regression (#4683).

Run in its own process: on the unfixed code the second launch deadlocks inside
``barrier_acquire`` in ``marlin_template.h``, and a deadlocked CUDA kernel
cannot be torn down from inside the process that owns it. The test that spawns
this driver asserts on its exit code instead of hanging pytest.

Exit codes: 0 = both streams completed with their single-stream result,
1 = a launch returned different bits, 2 = a stream stopped making progress.
"""

from __future__ import annotations

import os
import sys
import threading

import torch

from tests.kernels.test_awq_marlin_determinism_4683 import _marlin_and_torch


# Llama-3.2-1B-Instruct AWQ ``mlp.up_proj``: one of the shapes the issue saw
# diverge, and large enough that the split-K path uses the lock counters.
SIZE_M = 512
SIZE_N = 8192
SIZE_K = 2048
ROUNDS = 25


def main() -> int:
    module, _torch_module = _marlin_and_torch(
        torch.device("cuda:0"),
        in_features=SIZE_K,
        out_features=SIZE_N,
        dtype=torch.float16,
    )
    inputs = {
        name: torch.randn((SIZE_M, SIZE_K), device="cuda", dtype=torch.float16)
        for name in ("a", "b")
    }
    with torch.inference_mode():
        reference = {name: module(x).clone() for name, x in inputs.items()}
    torch.cuda.synchronize()

    progress = {"a": 0, "b": 0}
    mismatched: list[str] = []
    errors: list[str] = []
    barrier = threading.Barrier(2)

    def worker(name: str) -> None:
        try:
            stream = torch.cuda.Stream()
            barrier.wait()
            with torch.cuda.stream(stream), torch.inference_mode():
                for _ in range(ROUNDS):
                    out = module(inputs[name])
                    stream.synchronize()
                    if not torch.equal(out, reference[name]):
                        mismatched.append(name)
                    progress[name] += 1
        except Exception as exc:  # pragma: no cover - surfaced by the caller
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    threads = [
        threading.Thread(target=worker, args=(name,), daemon=True)
        for name in ("a", "b")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(120)

    if errors:
        print(f"ERROR {errors[:2]}", flush=True)
        return 1
    if min(progress.values()) < ROUNDS:
        print(f"HUNG progress={progress} of {ROUNDS}", flush=True)
        return 2
    if mismatched:
        print(f"MISMATCH streams={sorted(set(mismatched))}", flush=True)
        return 1
    print(f"OK rounds={ROUNDS} streams=2", flush=True)
    return 0


if __name__ == "__main__":
    status = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Skip interpreter teardown: a spinning kernel from an unfixed build would
    # otherwise block the exit after the diagnosis above was already printed.
    os._exit(status)
