# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Numerics and storage tests for the ROCm bf8/bf4 emulation kernels.

The oracle is ``gptqmodel.utils.tt_blockfloat_quant``: the kernel must consume
the checkpoint's packed codes and produce the same grid values, not merely
values that are close.  Linear outputs are compared against an independent
float32 GEMM over the recomposed weight, which is what a Tenstorrent device
would feed its matrix engine after unpack.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from gptqmodel.nn_modules.qlinear.tt_blockfloat import (
    TTBlockfloatEmbedding,
    TTBlockfloatLinear,
)
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.tt_blockfloat_hip import (
    decompose,
    dequantize,
    embedding,
    linear,
    prewarm_tt_blockfloat_hip_extension,
    quantize,
    tt_blockfloat_hip_supported,
)
from gptqmodel.utils.tt_blockfloat_quant import (
    _unpack_codes,
    quantize_blockfloat,
    recompose_blockfloat,
)


pytestmark = pytest.mark.skipif(not tt_blockfloat_hip_supported(), reason="ROCm GPU required")

ATOL = 2e-3


@pytest.fixture(scope="module", autouse=True)
def _prewarm():
    assert prewarm_tt_blockfloat_hip_extension(), "block-float HIP extension failed to build"


def _levels(bits: int) -> int:
    return 1 << (bits - 2)


def _random_codes(rows: int, cols: int, bits: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    offset = 2 * _levels(bits) - 1
    count = rows * cols // 2 if bits == 4 else rows * cols
    return torch.randint(0, 2 * offset + 1, (count,), dtype=torch.uint8, device="cuda", generator=gen)


def _random_exponents(rows: int, cols: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    return torch.randint(-12, 12, (rows, cols // 16), dtype=torch.int8, device="cuda", generator=gen)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dequant_matches_recompose(bits, dtype):
    rows, cols = 64, 128
    codes = _random_codes(rows, cols, bits, seed=bits)
    exponents = _random_exponents(rows, cols, seed=bits)
    expected = recompose_blockfloat(_unpack_codes(codes, bits, (rows, cols)), exponents, bits=bits)

    got = dequantize(codes, exponents, bits, (rows, cols), dtype)
    assert got.dtype == dtype
    assert torch.equal(got, expected.to(dtype))


@pytest.mark.parametrize("bits", [4, 8])
def test_quantize_matches_torch_oracle(bits):
    gen = torch.Generator(device="cuda").manual_seed(101)
    weight = torch.randn(32, 128, device="cuda", generator=gen)
    weight *= torch.exp2(torch.randint(-12, 12, (32, 1), device="cuda", generator=gen).float())
    # Binade boundaries exercise the exponent search.
    for col in range(128):
        power = torch.exp2(torch.tensor(float((col % 13) - 6), device="cuda"))
        weight[0, col] = torch.nextafter(power, torch.tensor(0.0, device="cuda"))

    codes, exponents = quantize(weight, bits)
    expected, expected_exponents = quantize_blockfloat(weight, bits)
    got = recompose_blockfloat(_unpack_codes(codes, bits, tuple(weight.shape)), exponents, bits=bits)

    assert torch.equal(got, expected)
    assert torch.equal(exponents, expected_exponents.to(torch.int8))


@pytest.mark.parametrize("bits", [4, 8])
def test_decompose_roundtrips_exactly(bits):
    gen = torch.Generator(device="cuda").manual_seed(7)
    weight = quantize_blockfloat(torch.randn(32, 128, device="cuda", generator=gen) * 0.02, bits)[0]
    codes, exponents = decompose(weight, bits)
    got = recompose_blockfloat(_unpack_codes(codes, bits, tuple(weight.shape)), exponents, bits=bits)
    assert torch.equal(got, weight)

    # Off-grid input must be refused, mirroring decompose_blockfloat.
    with pytest.raises(RuntimeError, match="not exactly on the block-float grid"):
        decompose(torch.randn(8, 16, device="cuda") * 0.017, bits)


@pytest.mark.parametrize("bits", [4, 8])
# The prefill launcher switches M block at 16, 32 and 64 rows and the decode
# path is m == 1, so cover both sides of every boundary plus the odd sizes that
# exercise masking.
@pytest.mark.parametrize("m", [1, 15, 16, 17, 31, 32, 33, 64, 65, 100, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_linear_matches_float32_gemm(bits, m, dtype):
    rows, cols = 128, 128
    codes = _random_codes(rows, cols, bits, seed=bits + m)
    exponents = _random_exponents(rows, cols, seed=bits + m)
    weight = recompose_blockfloat(_unpack_codes(codes, bits, (rows, cols)), exponents, bits=bits)

    gen = torch.Generator(device="cuda").manual_seed(31)
    x = torch.randn(m, cols, device="cuda", generator=gen).to(dtype)
    bias = torch.randn(rows, device="cuda", generator=gen).to(dtype)

    got = linear(x, codes, exponents, bits, bias)
    expected = x.float() @ weight.float().T + bias.float()

    assert got.shape == (m, rows)
    assert got.dtype == dtype
    # Compare against the reference rounded to the activation dtype so the
    # mandatory output rounding is not charged to the kernel's arithmetic.
    rounded = expected.to(dtype).float()
    norm_l1 = float((got.float() - rounded).abs().sum() / rounded.abs().sum().clamp_min(1e-6))
    assert norm_l1 < ATOL, f"norm_l1={norm_l1}"


@pytest.mark.parametrize("bits", [4, 8])
def test_embedding_gathers_only_selected_rows(bits):
    rows, cols = 64, 128
    codes = _random_codes(rows, cols, bits, seed=5)
    exponents = _random_exponents(rows, cols, seed=5)
    table = recompose_blockfloat(_unpack_codes(codes, bits, (rows, cols)), exponents, bits=bits)
    ids = torch.tensor([0, 3, 63, 7, 3], dtype=torch.int64, device="cuda")
    got = embedding(ids, codes, exponents, bits, rows, cols, torch.float32)
    assert torch.equal(got, table[ids])


@pytest.mark.parametrize("bits", [4, 8])
def test_module_forward_matches_dense_oracle_and_retains_no_dense_weight(bits):
    rows, cols = 128, 128
    codes = _random_codes(rows, cols, bits, seed=9)
    exponents = _random_exponents(rows, cols, seed=9)
    module = TTBlockfloatLinear(
        bits=bits,
        in_features=cols,
        out_features=rows,
        bias=True,
        backend=BACKEND.TT_BLOCKFLOAT,
        name="proj",
    ).cuda()
    module.set_packed_state(codes, exponents)
    module.bias.copy_(torch.randn(rows, dtype=torch.float16, device="cuda"))

    x = torch.randn(8, cols, device="cuda", generator=torch.Generator(device="cuda").manual_seed(14)).to(torch.float16)
    got = module(x)

    weight = module.dequantize(dtype=torch.float16)
    expected = F.linear(x, weight, module.bias)
    norm_l1 = float((got.float() - expected.float()).abs().sum() / expected.float().abs().sum().clamp_min(1e-6))
    assert norm_l1 < ATOL, f"norm_l1={norm_l1}"

    # The fused path must not leave an expanded weight on the module.
    assert "weight" not in vars(module)
    tensor_attrs = {name for name, value in vars(module).items() if isinstance(value, torch.Tensor)}
    assert not any(name.startswith(("dequant", "weight_", "expanded")) for name in tensor_attrs)


def test_embedding_module_uses_gather_path():
    rows, cols = 64, 128
    bits = 8
    codes = _random_codes(rows, cols, bits, seed=21)
    exponents = _random_exponents(rows, cols, seed=21)
    module = TTBlockfloatEmbedding(rows, cols, bits=bits, name="embed").cuda()
    module.set_packed_state(codes, exponents)

    ids = torch.tensor([[0, 1], [2, 63]], dtype=torch.int64, device="cuda")
    got = module(ids)
    expected = module.dequantize()[ids]
    assert got.shape == (2, 2, cols)
    assert torch.equal(got, expected)


def test_pack_uses_the_device_quantizer():
    bits = 4
    linear_module = torch.nn.Linear(128, 64, bias=False).cuda()
    module = TTBlockfloatLinear(
        bits=bits,
        in_features=128,
        out_features=64,
        bias=False,
        backend=BACKEND.TT_BLOCKFLOAT,
        name="proj",
    ).cuda()
    module.pack(linear_module)

    expected = quantize_blockfloat(linear_module.weight.detach().float(), bits)[0]
    assert torch.equal(module.dequantize(), expected)


_MISMATCH_SCRIPT = """
import sys

import torch

from gptqmodel.utils.tt_blockfloat_hip import (
    embedding,
    linear,
    prewarm_tt_blockfloat_hip_extension,
    quantize,
)

assert prewarm_tt_blockfloat_hip_extension(), "block-float HIP extension failed to build"

rows, cols = 32, 64
weight = torch.randn(rows, cols, dtype=torch.float32, device="cuda")
codes, exponents = quantize(weight, 4)
# Deliberately mismatched: the codes stay on the device that owns the stream.
mismatched = exponents.to({target!r})

try:
    if {kind!r} == "linear":
        linear(torch.randn(4, cols, device="cuda"), codes, mismatched, 4)
    else:
        ids = torch.tensor([0, 3], dtype=torch.int64, device="cuda")
        embedding(ids, codes, mismatched, 4, rows, cols)
except RuntimeError as exc:
    print("REJECTED:", str(exc).splitlines()[0])
    sys.exit(0)

print("ACCEPTED")
sys.exit(3)
"""


def _run_mismatched_exponents(kind: str, target: str) -> subprocess.CompletedProcess:
    """Run one mismatch in its own process.

    A kernel that reads another device's pointer can raise an illegal memory
    access, and on ROCm that poisons the HIP context for every later call in the
    same process. Each case therefore gets a fresh interpreter so the remaining
    tests still report their own result.
    """

    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    return subprocess.run(
        [sys.executable, "-c", _MISMATCH_SCRIPT.format(kind=kind, target=target)],
        env=env,
        capture_output=True,
        text=True,
        timeout=3000,
    )


_CONCURRENT_FIRST_USE_SCRIPT = """
import sys
import threading

import torch

from gptqmodel.utils.tt_blockfloat_hip import (
    dequantize,
    linear,
    prewarm_tt_blockfloat_hip_extension,
    quantize,
)

# The race only exists when the interpreter really runs threads in parallel.
if sys._is_gil_enabled():
    print("GIL_ENABLED")
    sys.exit(4)

assert prewarm_tt_blockfloat_hip_extension(), "block-float HIP extension failed to build"

rows, cols, m = 64, 128, 32
torch.manual_seed(0)
weight = torch.randn(rows, cols, dtype=torch.float32, device="cuda")
codes, exponents = quantize(weight, 4)
# bf16 with m > 1 is the only path that consults the WMMA device cache, and a
# fresh process guarantees that cache is still cold when the threads arrive.
x = torch.randn(m, cols, dtype=torch.bfloat16, device="cuda")

threads = 8
barrier = threading.Barrier(threads)
results = [None] * threads
errors = []


def worker(index):
    try:
        barrier.wait(60)
        results[index] = linear(x, codes, exponents, 4)
    except BaseException as exc:  # noqa: BLE001 - report, do not mask
        errors.append(f"{type(exc).__name__}: {exc}")


workers = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
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

first = results[0]
for index, result in enumerate(results):
    if not torch.equal(result, first):
        print("MISMATCH at worker", index)
        sys.exit(7)

# Independent oracle: the explicit dequant path never consults the WMMA cache,
# so it cannot warm the cache the burst above is racing on.
dense = dequantize(codes, exponents, 4, (rows, cols), torch.float32)
reference = (x.float() @ dense.t()).bfloat16()
if not torch.equal(reference, first):
    print("ORACLE MISMATCH")
    sys.exit(8)

print("OK")
sys.exit(0)
"""


@pytest.mark.parametrize("attempt", [0, 1, 2], ids=["first", "second", "third"])
def test_concurrent_first_use_of_the_wmma_dispatch_cache(attempt):
    """Eight threads must agree on a cold WMMA-cache decision.

    ``ttbf_wmma_device()`` caches a per-device architecture decision in
    process-wide state, and the bf16 ``m > 1`` linear path calls it on every
    launch. A fresh interpreter guarantees a cold cache, so every thread here
    races on the first initialization. The test pins the observable contract: no
    crash, no torn result, and the same output the explicit dequant path
    produces. It runs under ``PYTHON_GIL=0`` because a GIL-enabled interpreter
    would serialize the burst and prove nothing.
    """

    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    env["PYTHON_GIL"] = "0"
    result = subprocess.run(
        [sys.executable, "-c", _CONCURRENT_FIRST_USE_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=3000,
    )

    if "GIL_ENABLED" in result.stdout:
        pytest.skip("this interpreter cannot disable the GIL")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


@pytest.mark.parametrize("kind", ["linear", "embedding"])
def test_host_exponents_are_rejected_before_launch(kind):
    """A CPU exponent buffer has no device pointer the kernel may dereference."""

    result = _run_mismatched_exponents(kind, "cpu")

    assert result.returncode == 0, result.stderr
    assert "REJECTED:" in result.stdout, result.stdout
    assert "must be on a CUDA/ROCm device" in result.stdout
    assert "ACCEPTED" not in result.stdout


@pytest.mark.parametrize("kind", ["linear", "embedding"])
def test_second_device_exponents_are_rejected_before_launch(kind):
    """Exponents on another accelerator would be read as a foreign pointer."""

    if torch.cuda.device_count() < 2:
        pytest.skip("two ROCm devices required")

    result = _run_mismatched_exponents(kind, "cuda:1")

    assert result.returncode == 0, result.stderr
    assert "REJECTED:" in result.stdout, result.stdout
    assert "must share a device" in result.stdout
    assert "ACCEPTED" not in result.stdout


@pytest.mark.parametrize("kind", ["linear", "embedding"])
def test_matching_device_exponents_still_run(kind):
    """The same-device path must keep working after the added validation."""

    result = _run_mismatched_exponents(kind, "cuda:0")

    assert result.returncode == 3, result.stderr
    assert "ACCEPTED" in result.stdout
    assert "REJECTED:" not in result.stdout
