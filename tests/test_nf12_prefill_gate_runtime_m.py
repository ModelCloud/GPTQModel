# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Runtime row count for the NF12 FP32 prefill-gate Tensor Core kernel.

``_linear_fp32_prefill_gate_tc`` used to declare the prefill row count ``m``
as ``tl.constexpr``. A serving process therefore compiled (or loaded from the
disk cache) one kernel per distinct prefill width, on the request path; Lite
production accumulated about 1,800 variants. ``m`` only masks rows, so it is
now an unspecialized runtime argument.

These tests need no GPU:

* the Triton interpreter runs the former kernel (verbatim copy below) and the
  current kernel on the same inputs and requires bit equality, plus a float64
  Torch oracle within the inference tolerance;
* an explicit sm80 compile (stub driver, no CUDA context or launch) binds the
  real launch arguments for many widths and requires one compiled variant,
  with the former kernel as the negative control (one variant per width);
* the PTX floating-point/MMA instruction sequence of the runtime-``m`` kernel
  equals each former per-width variant, so the K loop and per-element
  accumulation order are unchanged.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch


triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")

from gptqmodel_nf12 import _fp32  # noqa: E402


@triton.jit
def _legacy_prefill_gate_tc(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    m: tl.constexpr, n: tl.constexpr, k: tl.constexpr,
    has_bias: tl.constexpr,
    block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
):
    input_rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    output_rows = tl.program_id(1) * block_n + tl.arange(0, block_n)
    depth = tl.arange(0, block_k)
    acc = tl.full((block_m, block_n), 0, tl.float32)
    for block in range(tl.cdiv(k, block_k)):
        col = block * block_k + depth
        activation = tl.load(
            x_ptr + input_rows[:, None] * k + col[None, :],
            (input_rows[:, None] < m) & (col[None, :] < k), other=0,
        )
        weight = tl.load(
            weight_ptr + output_rows[:, None] * k + col[None, :],
            (output_rows[:, None] < n) & (col[None, :] < k), other=0,
        )
        acc = tl.dot(activation, tl.trans(weight).to(tl.float32), acc,
                     input_precision="tf32x3")
    if has_bias:
        acc += tl.load(bias_ptr + output_rows, output_rows < n, other=0)[None, :]
    tl.store(out_ptr + input_rows[:, None] * n + output_rows[None, :], acc,
             (input_rows[:, None] < m) & (output_rows[None, :] < n))


def _launch(kernel, x, weight, bias, out):
    """Launch exactly as ``_linear_fp32_prefill_gate`` does."""
    n, k = weight.shape
    m = x.numel() // k
    kernel[(triton.cdiv(m, 32), triton.cdiv(n, 64))](
        x, weight, bias if bias is not None else x, out,
        m, n, k, bias is not None, 32, 64, 64,
        num_warps=4, num_stages=3,
    )
    return out


INTERPRETER_CASES = (
    # (m, n, k, has_bias); n=384, k=5120 is the production gate geometry.
    (1, 64, 5120, False),
    (7, 64, 5120, True),
    (48, 384, 5120, False),
    (64, 64, 5120, False),
    (255, 64, 5120, False),
    (256, 64, 640, True),
    (385, 64, 640, False),    # first odd width the dispatcher sends here
)


def _interpreter_case(m, n, k, has_bias):
    """Run under ``TRITON_INTERPRET=1``; returns plain values for the parent."""
    generator = torch.Generator().manual_seed(m * 7919 + n + k)
    x = torch.randn(m, k, generator=generator)
    weight = torch.randn(n, k, generator=generator).to(torch.bfloat16)
    bias = torch.randn(n, generator=generator) if has_bias else None
    # Rows past m must stay untouched in both kernels.
    fresh = torch.full((m + 32, n), float("nan"))
    out_new = _launch(_fp32._linear_fp32_prefill_gate_tc, x, weight, bias, fresh.clone())
    out_old = _launch(_legacy_prefill_gate_tc, x, weight, bias, fresh.clone())
    oracle = x.double() @ weight.double().T + (bias.double() if has_bias else 0)
    return {
        "bitwise_equal": bool(torch.equal(out_new.view(torch.int32), out_old.view(torch.int32))),
        "tail_untouched": bool(torch.isnan(out_new[m:]).all()),
        "max_abs_error": float((out_new[:m].double() - oracle).abs().max()),
        "max_rel_error": float(((out_new[:m].double() - oracle).abs()
                                / oracle.abs().clamp_min(1e-6)).max()),
    }


def test_runtime_m_gate_matches_former_kernel_bitwise_interpreted():
    """``TRITON_INTERPRET`` must be set before ``triton.jit`` runs, so use a child."""
    script = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "import test_nf12_prefill_gate_runtime_m as t\n"
        "from triton.runtime.interpreter import InterpretedFunction\n"
        "assert isinstance(t._fp32._linear_fp32_prefill_gate_tc, InterpretedFunction)\n"
        "print(json.dumps([t._interpreter_case(*c) for c in t.INTERPRETER_CASES]))\n"
    )
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="")
    done = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True,
                          text=True, timeout=1800, cwd=Path(__file__).parents[1])
    assert done.returncode == 0, done.stderr[-4000:]
    rows = json.loads(done.stdout.strip().splitlines()[-1])
    for case, row in zip(INTERPRETER_CASES, rows):
        assert row["bitwise_equal"] and row["tail_untouched"], (case, row)
        assert row["max_abs_error"] <= 2e-3 or row["max_rel_error"] <= 2e-3, (case, row)


class _CpuTargetDriver:
    """Just enough of a Triton driver to bind and compile for sm80, never to launch."""

    def __init__(self):
        from triton.backends.compiler import GPUTarget

        self.target = GPUTarget("cuda", 80, 32)

    def get_current_device(self):
        return 0

    def get_current_stream(self, device=None):
        return 0

    def get_current_target(self):
        return self.target

    def get_active_torch_device(self):
        return torch.device("cpu")


@pytest.fixture
def cpu_sm80(tmp_path, monkeypatch):
    pytest.importorskip("triton.backends.nvidia")
    from triton.runtime.driver import driver

    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    previous = driver._active
    driver.set_active(_CpuTargetDriver())
    try:
        yield
    finally:
        driver._active = previous


def _warm(kernel, m, has_bias=False):
    return kernel.warmup(torch.float32, torch.bfloat16, torch.float32, torch.float32,
                         m, 384, 5120, has_bias, 32, 64, 64,
                         num_warps=4, num_stages=3, grid=(1,))


def _ptx_float_ops(compiled):
    ops = []
    for line in compiled.asm["ptx"].splitlines():
        match = re.match(r"^\s*(?:@!?%p\d+\s+)?([a-z][\w.]*)", line)
        if match and any(t in match.group(1) for t in (".f32", ".bf16", ".tf32", "mma")):
            ops.append(match.group(1))
    return ops


def test_prefill_widths_share_one_compiled_variant(cpu_sm80):
    kernel = _fp32._linear_fp32_prefill_gate_tc
    params = {p.name: p for p in kernel.params}
    assert not params["m"].is_constexpr and params["m"].do_not_specialize
    assert params["n"].is_constexpr and params["k"].is_constexpr

    widths = (384, 385, 1000, 3072, 3073, 8192, 49152)
    legacy = triton.jit(_legacy_prefill_gate_tc.fn)
    before = set(kernel.device_caches[0][0]) if 0 in kernel.device_caches else set()
    compiled = [_warm(kernel, m) for m in widths]
    assert len(set(kernel.device_caches[0][0]) - before) == 1
    assert len({id(c) for c in compiled}) == 1

    legacy_compiled = [_warm(legacy, m) for m in widths]
    # Negative control: the former constexpr row count compiled per width.
    assert len(legacy.device_caches[0][0]) == len(widths)
    candidate_ops = _ptx_float_ops(compiled[0])
    assert sum("mma" in op for op in candidate_ops) == 96  # 3 TF32 MMAs x 32 tiles per k step
    for m, old in zip(widths, legacy_compiled):
        assert _ptx_float_ops(old) == candidate_ops, m
