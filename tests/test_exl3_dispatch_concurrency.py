# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="EXL3 tensor-core GEMM is NVIDIA-only",
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
