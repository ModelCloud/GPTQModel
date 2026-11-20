# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Blackwell (sm_120) coverage for the Q4NX small-batch GEMV dispatch.

The Ada sweep (``tests/benchmark_exl3_q4nx_gemv_ada.py``) admitted the packed
GEMV on sm_89.  ``tests/benchmark_exl3_q4nx_gemv_blackwell.py`` ran the same
matrix on an RTX 5090 (170 SMs, 96 MiB L2).  The sm_120 build of the packed
GEMV is not numerically correct there: all 180 cells disagree with the
reconstructed FP32 weight by 0.16 to 0.82 relative, while the block-pipelined
kernel stays within 3.3e-4.  The same defect shows up for ``mul1`` and ``mcg``,
so it is the GEMV kernel on sm_120, not the Q4NX decoder.  Opening the gate
would therefore trade a 1.06x cold / 1.18x warm geometric mean for wrong
output, and 61 of the 180 cells regress anyway.

These tests pin the decline so it is not "fixed" by adding
``CC_BLACKWELL`` to the arch gate without a working kernel and a new sweep.
The source contract runs on every host; the dispatch and parity cases need an
sm_120 device.
"""

from pathlib import Path

import pytest
import torch

from gptqmodel.exllamav3.ext import prewarm_exllamav3_extension
from tests.accelerator_gates import requires_nvidia_cuda
from tests.kernels.test_exllamav3_native_decode import _input, _packed_projection, _q4nx_reference


ROOT = Path(__file__).resolve().parents[2]
GEMV_SOURCE = ROOT / "gptqmodel_ext/exllamav3/quant/exl3_gemv.cu"
GEMV_DISPATCH = 90
REFERENCE_TOLERANCE = {"rtol": 2e-3, "atol": 2e-3}


def test_q4nx_gemv_arch_gate_declines_blackwell():
    source = GEMV_SOURCE.read_text()
    # Blackwell runs the block-pipelined kernel: the sm_120 GEMV measured
    # 0.16-0.82 relative error against the FP32 reference.
    assert "if (q4nx && cc != CC_AMPERE && cc != CC_ADA) return false;" in source
    assert "cc != CC_BLACKWELL" not in source


pytestmark = pytest.mark.skipif(not requires_nvidia_cuda(), reason="native EXL3 decode requires NVIDIA CUDA")


@pytest.fixture(scope="module")
def native():
    assert prewarm_exllamav3_extension()
    return torch.ops.gptqmodel_exllamav3.bc_linear_exl3_run_configured


def _require_blackwell():
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 0):
        pytest.skip(f"the Q4NX GEMV Blackwell envelope is pinned on sm_120; this device is sm{major}{minor}")


def _run(native, weights, bits, x, gemv):
    n = weights[2].numel()
    y = torch.empty((x.shape[0], n), device=x.device, dtype=torch.float32)
    path = native(*weights, bits, None, False, False, x, y, gemv, 0, False, q4nx=True)
    torch.cuda.synchronize()
    return path, y


@torch.inference_mode()
@pytest.mark.parametrize("bits", (2.5, 3.0, 3.5))
@pytest.mark.parametrize(
    ("rows", "k", "n"),
    ((1, 2048, 2048), (2, 2048, 8192), (4, 4096, 4096), (8, 8192, 2048), (1, 2048, 16384)),
)
def test_q4nx_gemv_declines_on_blackwell_and_packed_matches_reference(native, bits, rows, k, n):
    _require_blackwell()
    weights = _packed_projection(k, n, bits)
    x = _input(rows, k, 19)
    path, packed = _run(native, weights, bits, x, True)
    # 90 is the packed GEMV; anything in 1..7 is the block-pipelined kernel.
    assert 1 <= path <= 7, f"sm_120 must stay on the block-pipelined kernel, got dispatch {path}"
    assert torch.isfinite(packed).all()
    bias = torch.zeros(n, device=x.device)
    expected = _q4nx_reference(weights, x, bias)
    torch.testing.assert_close(packed, expected, **REFERENCE_TOLERANCE)
    # The same request without the GEMV policy selects the same path.
    path_packed, again = _run(native, weights, bits, x, False)
    assert path_packed == path
    torch.testing.assert_close(again, packed, rtol=1e-5, atol=1e-5)
