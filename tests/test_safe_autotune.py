# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import threading
from typing import Callable, List, Tuple

import pytest
import torch

pytest.importorskip("triton")
import triton
import triton.language as tl


THREADS = 6
ITERATIONS = 5
N = 1 << 16


def _build_kernel(autotune_fn: Callable) -> triton.runtime.autotuner.Autotuner:
    configs = [
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
    ]
    decorator = autotune_fn(configs=configs, key=["BLOCK_SIZE"])

    @decorator
    @triton.jit
    def _kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        tl.store(y_ptr + offs, x + 1.0, mask=mask)

    return _kernel


def _run_concurrently(kernel) -> Tuple[int, List[str]]:
    barrier = threading.Barrier(THREADS)
    errors: List[Exception] = []
    samples: List[str] = []
    inputs = [
        (
            torch.rand(N, device="cuda", dtype=torch.float32),
            torch.empty(N, device="cuda", dtype=torch.float32),
        )
        for _ in range(THREADS)
    ]
    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)

    def worker(idx: int):
        torch.cuda.set_device(0)
        x, y = inputs[idx]
        barrier.wait()
        for _ in range(ITERATIONS):
            try:
                kernel[grid](x, y, N)
            except Exception as exc:  # noqa: BLE001 - we want every failure
                errors.append(exc)
                if len(samples) < 3:
                    samples.append(repr(exc))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    torch.cuda.synchronize()
    return len(errors), samples


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton autotune test")
def test_safe_autotune_serializes_calls():
    # Reload Triton's autotuner module to obtain the original, unsafeguarded version.
    triton_autotuner = importlib.import_module("triton.runtime.autotuner")
    triton_autotuner = importlib.reload(triton_autotuner)

    # Build and invoke the kernel using the raw autotune; expect crashes when threads race.
    unsafe_kernel = _build_kernel(triton_autotuner.autotune)
    unsafe_errors, unsafe_samples = _run_concurrently(unsafe_kernel)
    print(f"unsafe_errors={unsafe_errors}")
    if unsafe_samples:
        print("unsafe_exception_samples=", unsafe_samples)
    assert unsafe_errors > 0

    # Reload our safety wrappers so Triton's autotuner is wrapped in locks.
    if "gptqmodel.utils.safe" in sys.modules:
        del sys.modules["gptqmodel.utils.safe"]
    safe_mod = importlib.import_module("gptqmodel.utils.safe")
    safe_kernel = _build_kernel(safe_mod._triton_autotuner.autotune)
    safe_errors, safe_samples = _run_concurrently(safe_kernel)
    print(f"safe_errors={safe_errors}")
    if safe_samples:
        print("safe_exception_samples=", safe_samples)
    assert safe_errors == 0
