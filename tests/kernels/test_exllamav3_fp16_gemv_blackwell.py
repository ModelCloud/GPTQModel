# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Blackwell (sm_120 / sm_121) coverage for the FP16 GEMV dispatch.

Issue #5873: the FP16 GEMV main loop was correct, but its output Hadamard
epilogue read C tiles written by other blocks after a cooperative
``grid.sync()`` and produced whole 128-column tiles that did not match
(0.16-0.82 relative over m 1..8 x k 2048..8192 x n 2048..16384 x 2/3/4 bpw).
The epilogue now runs as the next launch in the same stream, so it observes
the cooperative kernel's stores without a device-scope barrier.

With ``fp16_gemv=True`` the six non-Q4NX cells must dispatch the FP16 GEMV
(90) and match the reconstructed FP32 weight to 2e-3. Q4NX keeps its separate
Ampere/Ada-only gate and must still fall back to the block-pipelined kernel.

The dispatch and parity cases need a Blackwell device because that is where the
regression was measured.
"""

import pytest
import torch

from gptqmodel.exllamav3.ext import prewarm_exllamav3_extension
from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear
from tests.accelerator_gates import requires_nvidia_cuda


GEMV_DISPATCH = 90
REFERENCE_TOLERANCE = {"rtol": 2e-3, "atol": 2e-3}


pytestmark = pytest.mark.skipif(not requires_nvidia_cuda(), reason="native EXL3 decode requires NVIDIA CUDA")


@pytest.fixture(scope="module")
def native():
    assert prewarm_exllamav3_extension()
    return torch.ops.gptqmodel_exllamav3.bc_linear_exl3_run_configured


def _require_blackwell():
    major, minor = torch.cuda.get_device_capability()
    if major != 12:
        pytest.skip(f"the FP16 GEMV Blackwell path requires sm_120/sm_121; this device is sm{major}{minor}")


def _packed_projection(k, n, bits, seed=37):
    generator = torch.Generator().manual_seed(seed)
    trellis = torch.randint(-32768, 32768, (k // 16, n // 16, int(16 * bits)),
                            generator=generator, dtype=torch.int16).cuda()
    su = ((2 * torch.randint(0, 2, (k,), generator=generator) - 1) * 0.5).half().cuda()
    sv = (torch.randn(n, generator=generator) * 0.03).half().cuda()
    return trellis, su, sv


def _input(m, k, seed=19):
    return (torch.randn((m, k), generator=torch.Generator().manual_seed(seed)) * 0.1).half().cuda()


def _reference(weights, x, codebook):
    trellis, su, sv = weights
    model = ExllamaV3TorchLinear.from_tensors(
        in_features=su.numel(),
        out_features=sv.numel(),
        name="torch_oracle",
        tensors={
            "trellis": trellis,
            "suh": su,
            "svh": sv,
            codebook: torch.ones((), dtype=torch.int32, device=x.device),
        },
    ).eval()
    dense = model.get_weight_tensor(torch.float32)
    return (x.double() @ dense.double()).float()


def _run(native, weights, bits, x, gemv, codebook):
    n = weights[2].numel()
    y = torch.empty((x.shape[0], n), device=x.device, dtype=torch.float32)
    path = native(*weights, bits, None, codebook == "mcg", codebook == "mul1",
                  x, y, gemv, 0, False, q4nx=codebook == "q4nx")
    torch.cuda.synchronize()
    return path, y


@torch.inference_mode()
@pytest.mark.parametrize(
    ("codebook", "bits", "rows", "k", "n"),
    (
        ("mcg", 2.0, 1, 2048, 2048),
        ("mcg", 2.0, 1, 4096, 4096),
        ("mcg", 4.0, 1, 4096, 8192),
        ("mul1", 3.0, 1, 2048, 4096),
        ("mcg", 2.0, 8, 2048, 8192),
        ("mul1", 3.0, 8, 2048, 8192),
        ("q4nx", 2.5, 1, 2048, 2048),
    ),
)
def test_fp16_gemv_matches_reference_on_blackwell(native, codebook, bits, rows, k, n):
    _require_blackwell()
    weights = _packed_projection(k, n, bits)
    x = _input(rows, k)
    path, gemv = _run(native, weights, bits, x, True, codebook)
    if codebook == "q4nx":
        # Q4NX has its own Ampere/Ada-only gate; Blackwell keeps the packed path.
        assert path != GEMV_DISPATCH
    else:
        # Dispatch 90 is the FP16 GEMV; the Blackwell path must be selected.
        assert path == GEMV_DISPATCH
    _, packed = _run(native, weights, bits, x, False, codebook)
    torch.testing.assert_close(gemv, packed, **REFERENCE_TOLERANCE)
    expected = _reference(weights, x, codebook)
    torch.testing.assert_close(gemv, expected, **REFERENCE_TOLERANCE)
