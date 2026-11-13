# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Tenstorrent block-float CUDA emulation kernels.

The point of these kernels is that a ``bf4_b`` / ``bf8_b`` checkpoint
behaves on an NVIDIA part exactly as it does on Tensix, so nearly every
assertion compares against the torch reference in
``gptqmodel.utils.tt_blockfloat_quant`` rather than against a hand-written
expectation.  Quantise and decompose are required to be *bit* identical; the
GEMM is allowed float32 accumulation error.
"""

import pytest
import torch

from gptqmodel.utils import tt_blockfloat_cuda as bf_cuda
from gptqmodel.utils.tt_blockfloat_quant import (
    _unpack_codes,
    decompose_blockfloat,
    quantize_blockfloat,
    recompose_blockfloat,
)


pytestmark = pytest.mark.skipif(
    not bf_cuda.supported(),
    reason="Tenstorrent block-float CUDA kernels need a CUDA device of sm_80 or newer",
)

BITS = (4, 8)
# Small, odd and padded geometries; ``k`` must stay a multiple of the block.
SHAPES = ((64, 128), (1, 256), (17, 48), (128, 512), (3, 16))


def _reference(weight: torch.Tensor, bits: int):
    """The torch quantiser's view: grid weights, 2D codes and exponents."""

    grid, _ = quantize_blockfloat(weight, bits)
    codes, exponents = decompose_blockfloat(grid, bits)
    return grid, codes, exponents


def _cuda_quantize(weight: torch.Tensor, bits: int):
    codes, exponents = bf_cuda.quantize(weight, bits)
    return codes, exponents.reshape(-1)


def test_selftest_runs_on_device():
    assert bf_cuda.selftest() == 0x54544246  # sentinel: "TTBF" = tenstorrent block float


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("rows,k", SHAPES)
def test_quantize_is_bit_exact(bits, rows, k):
    torch.manual_seed(rows * 1000 + k + bits)
    weight = torch.randn(rows, k, device="cuda", dtype=torch.float32) * 0.05

    codes, exponents = _cuda_quantize(weight, bits)
    _, ref_codes, ref_exponents = _reference(weight, bits)

    assert torch.equal(exponents, ref_exponents.reshape(-1).to(torch.int8))
    assert torch.equal(
        _unpack_codes(codes, bits, (rows, k)).to(torch.uint8),
        ref_codes.to(torch.uint8),
    )


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("rows,k", SHAPES)
def test_dequant_matches_reference(bits, rows, k):
    torch.manual_seed(rows * 7 + k + bits)
    weight = torch.randn(rows, k, device="cuda", dtype=torch.float32)

    codes, exponents = _cuda_quantize(weight, bits)
    _, ref_codes, ref_exponents = _reference(weight, bits)
    expected = recompose_blockfloat(ref_codes, ref_exponents, bits)

    got = bf_cuda.dequant(codes, exponents, bits, rows, k, torch.float32)
    assert torch.equal(got, expected.reshape(rows, k))


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("rows,k", SHAPES)
def test_decompose_roundtrips_on_grid_weights(bits, rows, k):
    torch.manual_seed(rows * 13 + k + bits)
    weight = torch.randn(rows, k, device="cuda", dtype=torch.float32)
    grid, _, _ = _reference(weight, bits)

    codes, exponents, failed = bf_cuda.decompose(grid, bits)
    assert int(failed.item()) == 0

    back = bf_cuda.dequant(codes, exponents.reshape(-1), bits, rows, k, torch.float32)
    assert torch.equal(back, grid.reshape(rows, k))


@pytest.mark.parametrize(
    "dtype",
    (torch.bfloat16, torch.float16, torch.float32),
)
@pytest.mark.parametrize(
    "m,n,k",
    (
        (1, 256, 256),
        (15, 512, 512),
        (16, 512, 512),
        (17, 64, 96),
        (64, 256, 128),
        (129, 384, 256),
    ),
)
@pytest.mark.parametrize("bits", BITS)
def test_linear_matches_dense_reference(bits, dtype, m, n, k):
    """The GEMM must agree with a float32 matmul over the same grid weights.

    ``m`` straddles the decode/prefill switch so a change in how the launch is
    chosen cannot silently pick a path that is only correct for one of them.
    """

    torch.manual_seed(m * 31 + n + k + bits)
    weight = torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.05
    grid, _, _ = _reference(weight, bits)
    codes, exponents = _cuda_quantize(weight, bits)

    x = (torch.randn(m, k, device="cuda", dtype=torch.float32) * 0.1).to(dtype)
    bias = (torch.randn(n, device="cuda", dtype=torch.float32) * 0.01).to(dtype)

    got = bf_cuda.linear(x, codes, exponents, bits, bias)

    expected = x.to(torch.float32) @ grid.reshape(n, k).T + bias.to(torch.float32)
    assert got.dtype is dtype
    assert tuple(got.shape) == (m, n)

    relative = ((got.to(torch.float32) - expected).abs().max() / expected.abs().max()).item()
    assert relative < 5e-3


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("bits", (4, 8))
def test_linear_without_bias(bits, dtype):
    torch.manual_seed(bits)
    n, k, m = 96, 128, 33
    weight = torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.05
    grid, _, _ = _reference(weight, bits)
    codes, exponents = _cuda_quantize(weight, bits)

    x = torch.randn(m, k, device="cuda").to(dtype)
    got = bf_cuda.linear(x, codes, exponents, bits)
    expected = x.to(torch.float32) @ grid.reshape(n, k).T

    assert ((got.to(torch.float32) - expected).abs().max() / expected.abs().max()).item() < 5e-3


@pytest.mark.parametrize("bits", (4, 8))
def test_embedding_gathers_and_zero_fills(bits):
    torch.manual_seed(bits + 99)
    rows, k = 40, 64
    weight = torch.randn(rows, k, device="cuda")
    grid, _, _ = _reference(weight, bits)
    codes, exponents = _cuda_quantize(weight, bits)

    ids = torch.tensor([0, 7, rows - 1, rows + 5, 3], device="cuda", dtype=torch.int64)
    got = bf_cuda.embedding(ids, codes, exponents, bits, rows, k, torch.float32)

    expected = grid.reshape(rows, k)[torch.clamp(ids, 0, rows - 1)].clone()
    # Out-of-range ids must read as zero rather than wrapping onto another row.
    expected[3] = 0.0

    assert torch.equal(got, expected)


def test_linear_rejects_inconsistent_exponents():
    """Buffer geometry is checked on the host, not read past on the device."""

    codes = torch.zeros(8, dtype=torch.uint8, device="cuda")
    exponents = torch.zeros(1, dtype=torch.int8, device="cuda")
    x = torch.zeros(1, 256, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros(1, 1, dtype=torch.bfloat16, device="cuda")

    with pytest.raises(RuntimeError):
        torch.ops.gptqmodel_blockfloat_cuda.blockfloat_linear(x, codes, exponents, 4, None, out)


def test_k_must_be_a_multiple_of_the_block():
    codes = torch.zeros(64, dtype=torch.uint8, device="cuda")
    exponents = torch.zeros(2, dtype=torch.int8, device="cuda")
    x = torch.zeros(1, 24, dtype=torch.bfloat16, device="cuda")

    with pytest.raises(ValueError):
        bf_cuda.linear(x, codes, exponents, 4)


def test_unsupported_bit_width_is_rejected():
    """Only the two widths the hardware defines exist; 2 is deliberately absent."""

    x = torch.zeros(1, 16, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError):
        bf_cuda.quantize(x, 2)


def test_linear_module_routes_through_the_kernel():
    """The inference layer must use the kernel, not dequantise-then-matmul."""

    from gptqmodel.nn_modules.qlinear.tt_blockfloat import TTBlockfloatLinear
    from gptqmodel.utils.backend import BACKEND

    torch.manual_seed(5)
    dense = torch.nn.Linear(256, 128, bias=True).cuda()
    module = TTBlockfloatLinear(
        bits=4,
        in_features=256,
        out_features=128,
        bias=True,
        backend=BACKEND.TT_BLOCKFLOAT,
    ).cuda()
    module.pack(dense)

    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    assert module._kernel_linear(x) is not None

    grid = module.dequantize(dtype=torch.float32)
    expected = x.to(torch.float32) @ grid.T + module.bias.to(torch.float32)
    got = module(x)

    assert got.dtype is torch.bfloat16
    assert ((got.to(torch.float32) - expected).abs().max() / expected.abs().max()).item() < 5e-3
    # N-D activations are flattened for the kernel and restored afterwards.
    assert tuple(module(x.reshape(2, 2, 256)).shape) == (2, 2, 128)


def test_embedding_module_routes_through_the_kernel():
    from gptqmodel.nn_modules.qlinear.tt_blockfloat import TTBlockfloatEmbedding

    torch.manual_seed(6)
    module = TTBlockfloatEmbedding(50, 64, bits=8).cuda()
    module.codes.copy_(torch.randint(0, 256, module.codes.shape, device="cuda", dtype=torch.uint8))
    module.exponents.copy_(torch.randint(-8, 4, module.exponents.shape, device="cuda", dtype=torch.int8))

    ids = torch.tensor([[0, 9], [49, 7]], device="cuda")
    assert module._kernel_embedding(ids) is not None
    assert torch.equal(module(ids), torch.nn.functional.embedding(ids, module.dequantize()))

    # Out-of-range ids read as zero instead of indexing another row.
    out_of_range = module(torch.tensor([50, -1], device="cuda"))
    assert torch.count_nonzero(out_of_range) == 0


def _constant_bf8(value: float) -> int:
    """The ``bf8_b`` code for a constant at exponent 0 (levels 64, offset 127)."""

    return int(round(value * 64)) + 127


def test_overlapping_streams_get_independent_scratch():
    """Two streams must not share the decoded-weight buffer.

    Regression for the review finding: the prefill path expanded the weights into
    one cached workspace per device, so two ordinary stream-scoped forwards could
    overwrite each other's decoded weights while both were queued.  The weights
    here are constant, so the exact result is known and any clobber is visible.
    """

    torch.manual_seed(0)
    m, n, k = 1024, 4096, 4096
    values = {"a": 1.0, "b": 1.5}

    cases = {}
    for tag, value in values.items():
        code = _constant_bf8(value)
        cases[tag] = (
            torch.full((n * k,), code, dtype=torch.uint8, device="cuda"),
            torch.zeros((n * (k // 16),), dtype=torch.int8, device="cuda"),
            torch.full((m, n), k * value, dtype=torch.bfloat16, device="cuda"),
        )

    x = torch.ones((m, k), dtype=torch.bfloat16, device="cuda")
    out = {tag: torch.empty((m, n), dtype=torch.bfloat16, device="cuda") for tag in cases}
    streams = {tag: torch.cuda.Stream() for tag in cases}

    for _ in range(3):
        for tag in cases:
            codes, exponents, _ = cases[tag]
            with torch.cuda.stream(streams[tag]):
                torch.ops.gptqmodel_blockfloat_cuda.blockfloat_linear(x, codes, exponents, 8, None, out[tag])
        for stream in streams.values():
            stream.synchronize()
        for tag, (_, _, expected) in cases.items():
            assert torch.equal(out[tag], expected), f"stream {tag} was corrupted"


def test_scratch_is_not_left_live_in_the_async_pool():
    """No expanded weight may stay allocated between calls.

    The reviewer measured 134 MB still live in the default CUDA async pool after
    a forward returned.  Scratch is now owned by the call, so the process pool
    must never grow and torch's own counters must return to baseline.
    """

    import ctypes

    runtime = None
    for soname in ("libcudart.so.13", "libcudart.so.12", "libcudart.so"):
        try:
            runtime = ctypes.CDLL(soname)
            break
        except OSError:
            continue
    if runtime is None:
        pytest.skip("libcudart is not loadable by name in this environment")

    pool = ctypes.c_void_p()
    runtime.cudaDeviceGetDefaultMemPool.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    assert runtime.cudaDeviceGetDefaultMemPool(ctypes.byref(pool), 0) == 0

    def used_bytes() -> int:
        value = ctypes.c_uint64()
        # cudaMemPoolAttrUsedMemCurrent == 0x7
        assert runtime.cudaMemPoolGetAttribute(pool, 7, ctypes.byref(value)) == 0
        return value.value

    torch.manual_seed(1)
    n, k = 4096, 4096
    weight = torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.05
    codes, exponents = _cuda_quantize(weight, 8)

    baseline = used_bytes()
    allocated = torch.cuda.memory_allocated()

    for m in (1024, 512):
        x = torch.ones((m, k), dtype=torch.bfloat16, device="cuda")
        out = bf_cuda.linear(x, codes, exponents, 8)
        del x, out
    torch.cuda.synchronize()

    assert used_bytes() == baseline
    assert torch.cuda.memory_allocated() == allocated


def test_linear_accepts_strided_activations():
    """A sliced activation must not be rejected by the kernel path.

    ``x[:, ::2]`` keeps a stride, so ``reshape(-1, k)`` returns a non-contiguous
    view.  The dense path this replaces accepted it; the kernel materialises it.
    """

    from gptqmodel.nn_modules.qlinear.tt_blockfloat import TTBlockfloatLinear
    from gptqmodel.utils.backend import BACKEND

    torch.manual_seed(12)
    n, k = 32, 32
    module = TTBlockfloatLinear(
        bits=4,
        in_features=k,
        out_features=n,
        bias=False,
        backend=BACKEND.TT_BLOCKFLOAT,
    ).cuda()
    module.pack(torch.nn.Linear(k, n, bias=False).cuda())

    base = torch.randn(3, 2 * k, device="cuda")
    strided = base[:, ::2]
    assert strided.stride() == (2 * k, 2) and not strided.is_contiguous()

    got = module(strided)
    expected = strided.to(torch.float32) @ module.dequantize(dtype=torch.float32).T
    assert ((got.to(torch.float32) - expected).abs().max() / expected.abs().max()).item() < 5e-3

    # The same slice through the raw op, which is the layer that raised.
    codes, exponents = _cuda_quantize(module.dequantize(dtype=torch.float32), 4)
    direct = bf_cuda.linear(strided, codes, exponents, 4)
    assert ((direct.to(torch.float32) - expected).abs().max() / expected.abs().max()).item() < 5e-3


def test_linear_handles_a_full_grid_y():
    """The epilogue must not map every row onto grid.y, which caps at 65535."""

    torch.manual_seed(11)
    m, n, k = 65536, 1, 16
    weight = torch.randn(n, k, device="cuda", dtype=torch.float32)
    grid, _, _ = _reference(weight, 8)
    codes, exponents = _cuda_quantize(weight, 8)

    x = torch.ones((m, k), dtype=torch.bfloat16, device="cuda")
    got = bf_cuda.linear(x, codes, exponents, 8)

    expected = x.to(torch.float32) @ grid.reshape(n, k).T
    assert tuple(got.shape) == (m, n)
    assert ((got.to(torch.float32) - expected).abs().max() / expected.abs().max()).item() < 5e-3
