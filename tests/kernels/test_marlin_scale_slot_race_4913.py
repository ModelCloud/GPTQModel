# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""#4913: Marlin must not re-read a group's scale/zero-point slot after it is refilled.

For ``group_blocks > thread_k_blocks`` (g128 with ``thread_k = 64``) one
quantization group spans ``g = group_blocks / thread_k_blocks`` pipeline
stages. ``fetch_to_shared`` copies the group's scales and zero points only into
the slot of the group's first stage. The later stages used to re-read that slot
(``sh_s + s_sh_stage * g * (pipe / g)``) *after* the last ``wait_for_stage()``
barrier that protects it. In the same window, faster warps already refill that
slot with stage ``pipe + stages``, which belongs to group ``g + 2``. The
``cp.async`` latency normally hides the race. When the GPU time-slices to
another process, in-flight copies drain before the context is saved, so a
lagging warp reads the next-but-one group's scales. One warp group then gets a
whole 64-row output block wrong, with no error raised.

The fix, which matches vLLM's Marlin since vllm-project/vllm#24722, reads
shared memory only on the group's first stage. Later stages keep the values
already in ``frag_s`` / ``frag_qzp`` / ``frag_zpf``.

* The source test pins that contract on any host.
* The CUDA test is the issue's trigger: another process keeps the GPU busy
  with a trivial ``torch.cuda._sleep`` spin kernel, which forces context time
  slicing, and repeated launches must stay bitwise identical. Before the fix
  this reproduced at about 1 in 200 calls on an RTX 5090 (GPTQ fp16) and about
  1 in 700 on an RTX 4090 (AWQ bf16).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from .test_marlin_tile_determinism_4735 import _build_awq, _build_gptq, _skip_reason


_TEMPLATE = Path(__file__).resolve().parents[2] / "gptqmodel_ext" / "marlin" / "marlin_template.h"


def _lambda_body(source: str, name: str) -> str:
    start = source.index(f"auto {name} = [&]")
    end = source.index("\n  auto ", start + 1)
    return source[start:end]


def test_later_stages_of_a_group_never_read_the_refilled_slot():
    source = _TEMPLATE.read_text()
    scales = _lambda_body(source, "fetch_scales_to_registers")
    zero_points = _lambda_body(source, "fetch_zp_to_registers")

    aliased_slot = re.compile(r"pipe\s*/\s*\(\s*group_blocks\s*/\s*thread_k_blocks\s*\)")
    assert not aliased_slot.search(scales), "scales re-read the first stage's slot"
    assert not aliased_slot.search(zero_points), "zero points re-read the first stage's slot"

    # Only the group's first stage may touch shared memory: once for the scales,
    # and once each for the integer and float zero-point paths.
    first_stage_only = re.compile(r"if \(pipe % g == 0")
    assert len(first_stage_only.findall(scales)) == 1
    assert len(first_stage_only.findall(zero_points)) == 2


_TENANT = """
import sys, time, torch
torch.cuda.set_device(int(sys.argv[1]))
torch.cuda._sleep(1)
torch.cuda.synchronize()
print("ready", flush=True)
deadline = time.time() + float(sys.argv[2])
while time.time() < deadline:
    for _ in range(20):
        torch.cuda._sleep(1_000_000)
    torch.cuda.synchronize()
"""


def _start_gpu_tenant(device_index: int, seconds: float) -> subprocess.Popen:
    """Start a second CUDA context on the same GPU so the driver time-slices."""

    proc = subprocess.Popen(
        [sys.executable, "-c", _TENANT, str(device_index), str(seconds)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=dict(os.environ),
    )
    if proc.stdout.readline().strip() != "ready":
        proc.kill()
        pytest.fail("the GPU tenant process did not start")
    return proc


_CASES = (
    # kind, dtype, M, oracle atol (as in the #4735 determinism tests)
    pytest.param("gptq", torch.float16, 2048, 0.05, id="gptq-fp16"),
    pytest.param("awq", torch.bfloat16, 1024, 0.5, id="awq-zp-bf16"),
)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind,dtype,size_m,atol", _CASES)
def test_group_scales_survive_gpu_time_slicing(kind, dtype, size_m, atol):
    reason = _skip_reason(dtype)
    if reason is not None:
        pytest.skip(reason)

    size_k, size_n, calls, batch = 2048, 8192, 10_000, 100
    module, weight = (_build_gptq if kind == "gptq" else _build_awq)(size_k, size_n, dtype)

    torch.manual_seed(4913)
    x = torch.randn(size_m, size_k, device="cuda", dtype=dtype)
    tenant = _start_gpu_tenant(torch.cuda.current_device(), seconds=300)
    try:
        with torch.inference_mode():
            ref = module(x).clone()
            oracle = x.double() @ weight.double()
            assert (ref.double() - oracle).abs().max().item() < atol
            bad = 0
            for _ in range(calls // batch):
                outs = [module(x) for _ in range(batch)]
                bad += int(torch.stack([(out != ref).any() for out in outs]).sum().item())
    finally:
        tenant.kill()
        tenant.wait()

    assert bad == 0, f"{bad}/{calls} launches returned different output blocks under GPU time slicing"
