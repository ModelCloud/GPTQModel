# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Ada (sm_89) coverage for the Q4NX small-batch GEMV dispatch.

The Q4NX GEMV envelope was measured on an RTX 4090 (128 SMs, 72 MiB L2) with
``tests/benchmark_exl3_q4nx_gemv_ada.py``. These tests pin the two decisions
that sweep produced: sm_89 launches the packed GEMV for one to eight rows
where its grid fits one co-resident wave, and the wide config stays inside the
measured 2.5 / 3.5 bpw, k <= 2048, n >= 12288 band instead of becoming an
unconditional Ada fallback.

The source contracts run on every host. The dispatch and parity cases need an
sm_89 device because that is the only Ada part the envelope was measured on.
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
# The GEMV and the block-pipelined kernel sum the same FP32 partials in a
# different order; the sweep's worst measured difference is 7.2e-7 relative.
FALLBACK_TOLERANCE = {"rtol": 1e-4, "atol": 1e-4}


def test_q4nx_gemv_arch_gate_admits_ada():
    source = GEMV_SOURCE.read_text()
    assert "if (q4nx && cc != CC_AMPERE && cc != CC_ADA) return false;" in source


def test_q4nx_gemv_wide_config_stays_in_its_measured_ada_band():
    source = GEMV_SOURCE.read_text()
    # A bare wide fallback (`cfg = 1`) re-enables the 0.73-1.06x shapes the
    # sweep rejected, and the 3.0 bpw large-tile cell where the packed kernel
    # keeps the trellis in L2 and wins. Both are excluded on Ada only.
    assert "const bool ada_wide = cc == CC_ADA && step_bits != 12 && size_k <= 2048 && size_n >= 12288;" in source
    assert "else if (cc == CC_ADA) cfg = ada_wide ? 1 : -1;" in source
    assert "cc == CC_ADA && step_bits == 12 && (int64_t) size_k * size_n >= 64 * 1024 * 1024" in source


pytestmark = pytest.mark.skipif(not requires_nvidia_cuda(), reason="native EXL3 decode requires NVIDIA CUDA")


@pytest.fixture(scope="module")
def native():
    assert prewarm_exllamav3_extension()
    return torch.ops.gptqmodel_exllamav3.bc_linear_exl3_run_configured


def _require_ada():
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (8, 9):
        pytest.skip(f"the Q4NX GEMV Ada envelope is pinned on sm_89; this device is sm{major}{minor}")


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
    ((1, 2048, 2048), (2, 2048, 8192), (4, 4096, 4096), (8, 8192, 2048)),
)
def test_q4nx_gemv_dispatches_on_ada_and_matches_reference(native, bits, rows, k, n):
    _require_ada()
    weights = _packed_projection(k, n, bits)
    x = _input(rows, k, 19)
    path, gemv = _run(native, weights, bits, x, True)
    assert path == GEMV_DISPATCH
    _, fallback = _run(native, weights, bits, x, False)
    torch.testing.assert_close(gemv, fallback, **FALLBACK_TOLERANCE)
    bias = torch.zeros(n, device=x.device)
    expected = _q4nx_reference(weights, x, bias)
    torch.testing.assert_close(gemv, expected, **REFERENCE_TOLERANCE)


@torch.inference_mode()
@pytest.mark.parametrize("bits", (2.5, 3.5))
@pytest.mark.parametrize(("rows", "k", "n"), ((1, 2048, 12288), (8, 2048, 16384)))
def test_q4nx_gemv_wide_band_dispatches_on_ada(native, bits, rows, k, n):
    _require_ada()
    # n > 8192 leaves the narrow grid's co-residency, so dispatch 90 is the wide config.
    weights = _packed_projection(k, n, bits)
    x = _input(rows, k, 19)
    path, gemv = _run(native, weights, bits, x, True)
    assert path == GEMV_DISPATCH
    _, fallback = _run(native, weights, bits, x, False)
    torch.testing.assert_close(gemv, fallback, **FALLBACK_TOLERANCE)


@torch.inference_mode()
@pytest.mark.parametrize(
    ("bits", "k", "n"),
    ((3.0, 8192, 8192), (3.0, 4096, 16384), (2.5, 8192, 16384)),
)
def test_q4nx_gemv_declines_the_measured_regressions_on_ada(native, bits, k, n):
    _require_ada()
    # 3.0 bpw k = n = 8192 is 0.87x warm; the wide config outside its band is
    # 0.73-1.06x. Both must stay on the block-pipelined kernel.
    weights = _packed_projection(k, n, bits)
    x = _input(1, k, 19)
    path, fallback = _run(native, weights, bits, x, True)
    assert 1 <= path <= 7
    assert torch.isfinite(fallback).all()
    _, packed = _run(native, weights, bits, x, False)
    torch.testing.assert_close(fallback, packed, **FALLBACK_TOLERANCE)
