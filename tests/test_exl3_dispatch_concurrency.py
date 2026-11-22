# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest
import torch


_CONCURRENT_COLD_START_SCRIPT = """
import sys
import threading

import torch

from gptqmodel.exllamav3.ext import prewarm_exllamav3_extension
from gptqmodel.exllamav3.modules.quant.exl3 import LinearEXL3
from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

# A GIL-enabled interpreter serializes the burst, so it cannot reach the race.
if sys._is_gil_enabled():
    print("GIL_ENABLED")
    sys.exit(4)

if not prewarm_exllamav3_extension():
    print("EXTENSION_UNAVAILABLE")
    sys.exit(3)

device = torch.device("cuda:0")
# Distinct (in_features, out_features) pairs give the tuning cache several
# different keys, so the first-use insertions really overlap. Every size is a
# multiple of the GEMM's 16-element K tile and 128-element N tile.
shapes = [(128, 128), (256, 128), (256, 256), (384, 256)]


def build(k, n, seed):
    torch.manual_seed(seed)
    weight = torch.randn((k, n), dtype=torch.float32, device=device) * 0.02
    activations = torch.randn((256, k), dtype=torch.float32, device=device)
    hessian = (2.0 / activations.shape[0]) * activations.T @ activations
    quant_args = {
        "K": 2,
        "devices": [device],
        "apply_out_scales": None,
        "sigma_reg": 0.025,
        "seed": seed,
        "mcg": True,
    }
    _, _, tensors = quantize_exl3(
        weight,
        {"H": hessian, "count": activations.shape[0], "finalized": False},
        quant_args,
        return_weight_q=False,
    )
    return LinearEXL3(
        config=None,
        in_features=k,
        out_features=n,
        suh=tensors["suh"],
        svh=tensors["svh"],
        trellis=tensors["trellis"],
        mcg=tensors["mcg"],
        out_dtype=torch.float16,
    )


# Building the modules runs quantize_tiles/pack_trellis/reconstruct only; the
# tensor-core GEMM and its caches are still cold when the barrier opens below.
modules = {}
for index, shape in enumerate(shapes):
    modules[shape] = build(shape[0], shape[1], seed=100 + index)

cases = []
for index in range(8):
    shape = shapes[index % len(shapes)]
    m = 1 + index // len(shapes)
    torch.manual_seed(1000 + index)
    x = torch.randn((m, shape[0]), dtype=torch.float16, device=device)
    # The reconstruct + hgemm path is the independent oracle: it never consults
    # the tuning cache or the kernel-attribute set the burst below races on.
    expected = modules[shape].forward(x, {"reconstruct": True}).clone()
    cases.append((shape, m, x, expected))

barrier = threading.Barrier(len(cases))
results = [None] * len(cases)
errors = []


def worker(index):
    try:
        barrier.wait(60)
        shape, m, x, _ = cases[index]
        y = torch.empty((m, shape[1]), dtype=torch.float16, device=device)
        modules[shape].bc.run(x, y)
        torch.cuda.synchronize()
        results[index] = y.clone()
    except BaseException as exc:  # noqa: BLE001 - report, do not mask
        errors.append(f"{type(exc).__name__}: {exc}")


workers = [threading.Thread(target=worker, args=(i,)) for i in range(len(cases))]
for item in workers:
    item.start()
for item in workers:
    item.join(120)

if errors:
    print("ERRORS:", errors)
    sys.exit(5)
if any(result is None for result in results):
    print("INCOMPLETE: a worker produced no result")
    sys.exit(6)

for index, (shape, m, x, expected) in enumerate(cases):
    actual = results[index]
    if not torch.isfinite(actual).all():
        print("NONFINITE at worker", index, shape, m)
        sys.exit(7)
    if not torch.allclose(actual.float(), expected.float(), atol=1e-2, rtol=1e-2):
        print(
            "MISMATCH at worker",
            index,
            shape,
            m,
            float((actual.float() - expected.float()).abs().max()),
        )
        sys.exit(8)

print("OK")
sys.exit(0)
"""


def _free_threaded_build() -> bool:
    """True only for a free-threaded (3.13t/3.14t) interpreter.

    The child process sets ``PYTHON_GIL=0``; a regular build aborts with
    "Disabling the GIL is not supported by this build" before running the
    script, so the test can neither pass nor observe its sentinels (#4768).
    """

    return sysconfig.get_config_var("Py_GIL_DISABLED") == 1


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="EXL3 tensor-core GEMM is NVIDIA-only",
)
@pytest.mark.skipif(
    not _free_threaded_build(),
    reason="the PYTHON_GIL=0 subprocess needs a free-threaded build (#4768)",
)
@pytest.mark.parametrize("attempt", [0, 1], ids=["first", "second"])
def test_concurrent_first_use_of_the_exl3_dispatch_caches(attempt):
    """Concurrent cold-start GEMMs must agree on the dispatch decision.

    ``select_exl3_gemm_kernel_tuned`` fills a process-wide ``std::map`` and
    ``exl3_gemm`` fills a per-device ``std::set`` on first use. Under a
    free-threaded interpreter two Python threads can reach those containers at
    the same time. A fresh process guarantees they are cold when the threads
    arrive, and the reconstruct + hgemm path supplies the expected outputs
    without touching either container. The test pins the observable contract:
    no crash, no exception, and the same result as the independent path.
    """

    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    env["PYTHON_GIL"] = "0"
    result = subprocess.run(
        [sys.executable, "-c", _CONCURRENT_COLD_START_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=3000,
    )

    if "GIL_ENABLED" in result.stdout:
        pytest.skip("this interpreter cannot disable the GIL")
    if "EXTENSION_UNAVAILABLE" in result.stdout:
        pytest.skip("the EXL3 extension could not be built on this host")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


_STREAM_SCRATCH_SCRIPT = """
import os
import sys
import threading
import time

import torch

from gptqmodel.exllamav3.ext import prewarm_exllamav3_extension
from gptqmodel.exllamav3.modules.quant.exl3 import LinearEXL3

if not torch.cuda.is_available() or torch.version.hip is not None:
    print("NO_CUDA")
    sys.exit(4)
if not prewarm_exllamav3_extension():
    print("EXTENSION_UNAVAILABLE")
    sys.exit(5)

device = torch.device("cuda:0")


def make(k, n, seed, K=4):
    torch.manual_seed(seed)
    trellis = torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), dtype=torch.int16, device=device)
    suh = (torch.randint(0, 2, (k,), device=device) * 2 - 1).half()
    svh = (torch.randint(0, 2, (n,), device=device) * 2 - 1).half()
    mcg = torch.tensor(3417055213 - 2**32, dtype=torch.int32, device=device)
    return LinearEXL3(None, k, n, suh=suh, svh=svh, trellis=trellis, mcg=mcg, out_dtype=torch.half)


# M <= 32 takes the tensor-core GEMM: M == 1 exercises the bsz-1 Hadamard
# buffer, M == 4/8 the split-K lock array.  M == 256 takes hgemm and its cuBLAS
# workspace.  Two modules per case mean two threads share a module, so a
# per-module buffer is not enough either.
K, N = 2048, 512
CASES = [(1, 400), (4, 300), (8, 300), (256, 300)]
THREADS = 4
ITERATIONS = {m: iters for m, iters in CASES}

modules = [make(K, N, seed) for seed in (11, 12)]
xs = {}
refs = {}
for m, _ in CASES:
    torch.manual_seed(1000 + m)
    xs[m] = torch.randn((m, K), dtype=torch.float16, device=device)
    refs[m] = [module.forward(xs[m], {}).clone() for module in modules]
torch.cuda.synchronize()

bad = [0] * (THREADS * len(CASES))
done = [0] * (THREADS * len(CASES))
rel = [0.0] * (THREADS * len(CASES))
errors = []


def worker(slot, module_index, m):
    try:
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            for _ in range(ITERATIONS[m]):
                y = modules[module_index].forward(xs[m], {})
                if not torch.equal(y, refs[m][module_index]):
                    bad[slot] += 1
                    rel[slot] = max(
                        rel[slot],
                        ((y - refs[m][module_index]).abs().max() / refs[m][module_index].abs().max()).item(),
                    )
                done[slot] += 1
        stream.synchronize()
    except BaseException as exc:  # noqa: BLE001 - report, do not mask
        errors.append(f"{type(exc).__name__}: {exc}")


workers = []
for case_index, (m, _) in enumerate(CASES):
    for thread_index in range(THREADS):
        slot = case_index * THREADS + thread_index
        workers.append(threading.Thread(target=worker, args=(slot, thread_index % len(modules), m)))

for item in workers:
    item.start()

last = None
progress_at = time.time()
while any(item.is_alive() for item in workers):
    time.sleep(1)
    if done != last:
        last = list(done)
        progress_at = time.time()
    elif time.time() - progress_at > 90:
        print("HANG", list(done), list(bad), flush=True)
        os._exit(3)

if errors:
    print("ERRORS:", errors)
    sys.exit(8)

for case_index, (m, _) in enumerate(CASES):
    for thread_index in range(THREADS):
        slot = case_index * THREADS + thread_index
        if done[slot] != ITERATIONS[m]:
            print("INCOMPLETE", m, slot, done[slot])
            sys.exit(6)
        if bad[slot]:
            print(f"MISMATCH M={m} thread={thread_index} bad={bad[slot]}/{ITERATIONS[m]} rel={rel[slot]:.3f}")
            sys.exit(6)

print("OK")
sys.exit(0)
"""


def test_exl3_linears_do_not_share_a_writable_kernel_buffer(monkeypatch):
    """Two EXL3 layers with the same shape must not alias kernel scratch.

    The bsz == 1 decode path used to take the Hadamard input from
    ``g_tensor_cache``, keyed by ``(device, (1, in_features), dtype)``, so
    every module with the same width shared one writable buffer (issue
    #4619).  A stub kernel object records the tensors the module hands it, and
    the test asserts the two modules have no tensor in common and leave no
    process-wide bsz-1 buffer behind.  The kernel itself is not needed, so this
    runs on a host without the EXL3 extension.
    """

    import gptqmodel.exllamav3.modules.quant.exl3 as exl3_module
    from gptqmodel.exllamav3.util.tensor import g_tensor_cache

    class _RecordingBC:
        def __init__(self, *args):
            self.tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]

    monkeypatch.setattr(exl3_module.ext, "BC_LinearEXL3", _RecordingBC)
    saved_cache = dict(g_tensor_cache.cache)
    g_tensor_cache.cache.clear()
    try:
        def build(seed):
            torch.manual_seed(seed)
            k, n, K = 256, 128, 4
            return exl3_module.LinearEXL3(
                None,
                k,
                n,
                suh=(torch.randint(0, 2, (k,)) * 2 - 1).half(),
                svh=(torch.randint(0, 2, (n,)) * 2 - 1).half(),
                trellis=torch.randint(-32768, 32767, (k // 16, n // 16, 16 * K), dtype=torch.int16),
                mcg=torch.tensor(3417055213 - 2**32, dtype=torch.int32),
                out_dtype=torch.half,
            )

        first, second = build(1), build(2)
        first_ptrs = {tensor.data_ptr() for tensor in first.bc.tensors}
        second_ptrs = {tensor.data_ptr() for tensor in second.bc.tensors}

        assert first_ptrs and second_ptrs
        assert first_ptrs.isdisjoint(second_ptrs), "two EXL3 layers alias a kernel buffer"
        assert g_tensor_cache.cache == {}, "EXL3 kept a process-wide bsz-1 scratch buffer"
    finally:
        g_tensor_cache.cache.clear()
        g_tensor_cache.cache.update(saved_cache)


def _free_threaded_build() -> bool:
    """True only for a free-threaded (3.13t/3.14t) interpreter.

    The child process sets ``PYTHON_GIL=0``; a regular build aborts with
    "Disabling the GIL is not supported by this build" before running the
    script, so the test can neither pass nor observe its sentinels (#4768).
    """

    return sysconfig.get_config_var("Py_GIL_DISABLED") == 1


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="EXL3 tensor-core GEMM is NVIDIA-only",
)
@pytest.mark.skipif(
    not _free_threaded_build(),
    reason="the PYTHON_GIL=0 subprocess needs a free-threaded build (#4768)",
)
def test_concurrent_streams_do_not_share_exl3_kernel_scratch():
    """Concurrent EXL3 forwards on separate streams must be bitwise correct.

    Each M exercises a different shared buffer: M == 1 the bsz-1 Hadamard
    input, M == 4/8 the split-K lock array, M == 256 the cuBLAS hgemm
    workspace.  Four threads over two modules also cover two callers of the
    same module, which a per-module buffer would still corrupt.  The failure
    mode includes a GPU deadlock, so the child process is bounded by its own
    watchdog and by the outer subprocess timeout.
    """

    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    env["PYTHON_GIL"] = "0"
    result = subprocess.run(
        [sys.executable, "-c", _STREAM_SCRATCH_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if "NO_CUDA" in result.stdout:
        pytest.skip("CUDA is not available")
    if "EXTENSION_UNAVAILABLE" in result.stdout:
        pytest.skip("the EXL3 extension could not be built on this host")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
