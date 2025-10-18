# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import sys
import threading

import pytest
import torch

import gptqmodel  # noqa: F401  # ensures monkey patches run before Triton import


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional dependency
    triton = None
    tl = None


@pytest.mark.skipif(triton is None, reason="Triton is not installed")
def test_triton_autotune_threads_cuda():
    gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
    if gil_enabled:
        pytest.skip("Requires running with PYTHON_GIL=0")
    if not torch.cuda.is_available():
        pytest.skip("CUDA backend required for Triton autotune threading test")

    device = "cuda"
    N = 8192
    configs = [
        triton.Config(kwargs={"BLOCK": 128}, num_warps=2),
        triton.Config(kwargs={"BLOCK": 256}, num_warps=4),
    ]

    @triton.autotune(configs=configs, key=["N"])
    @triton.jit
    def copy_kernel(dst, src, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        values = tl.load(src + offsets, mask=mask)
        tl.store(dst + offsets, values, mask=mask)

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK"]),)
    num_threads = 8
    sync_ready = threading.Barrier(num_threads + 1)
    sync_start = threading.Barrier(num_threads + 1)
    errors = []

    def worker():
        dst = torch.empty(N, device=device, dtype=torch.float32)
        src = torch.randn_like(dst)
        sync_ready.wait()
        sync_start.wait()
        try:
            for _ in range(4):
                dst.zero_()
                copy_kernel[grid](dst, src, N)
        except Exception as exc:  # pragma: no cover - captured for assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for thread in threads:
        thread.start()

    sync_ready.wait()
    sync_start.wait()

    for thread in threads:
        thread.join()

    assert not errors
