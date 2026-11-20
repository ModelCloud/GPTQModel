# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Runtime row count for the NF12 FP32 decode and short-prefill kernels.

``_linear_fp32`` and ``_linear_bf16_fp32_tc`` declared the row count ``m`` as
``tl.constexpr``, so every distinct decode or small-prefill width compiled (or
loaded from the disk cache) a new variant on the serving path: 12 widths left
12 cache entries for a 64x2048 router, and 8 widths left 8 for a 2048x256
head. ``m`` only masks activation loads and output stores, so it is now an
unspecialized runtime argument. ``block_m`` (already bucketed by the launcher
to ``min(next_pow2(m), 8)``) and every other tile parameter stay compile-time.

These tests need no GPU for the compile-count checks:

* an explicit sm80 compile (stub driver, no CUDA context or launch) binds the
  real launch arguments for many widths and requires one variant per distinct
  ``block_m`` bucket, with verbatim constexpr-``m`` kernels as negative
  controls that compile one variant per width;
* the PTX floating-point instruction sequence of each runtime-``m`` kernel
  equals the former per-width variants, so the K loop and per-element
  accumulation order are unchanged.

The CUDA test compares the public launchers with the former kernels bitwise
when a native NF12 build is present.
"""

import re

import pytest
import torch


triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")

from gptqmodel_nf12 import _fp32  # noqa: E402


ROUTER_N, ROUTER_K = 64, 2048
HEAD_N, HEAD_K = 2048, 256


@triton.jit
def _legacy_linear_fp32(
    x_ptr, packed_ptr, escapes_ptr, bias_ptr, rows_ptr, out_ptr,
    n: tl.constexpr, output_n: tl.constexpr, k: tl.constexpr, m: tl.constexpr,
    exp_base: tl.constexpr, block_m: tl.constexpr, block_k: tl.constexpr,
    has_bias: tl.constexpr, indexed: tl.constexpr,
):
    output_row = tl.program_id(0)
    # Compare before narrowing: an out-of-range int64 ID can otherwise wrap
    # into a valid int32 row and read the wrong packed weight.
    row_id = tl.load(rows_ptr + output_row) if indexed else output_row
    valid_row = (row_id >= 0) & (row_id < n)
    row_n = row_id.to(tl.int32)
    row_m = tl.program_id(1) * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, block_k)
    accum = tl.full((block_m, block_k), 0, tl.float32)
    packed_u32 = packed_ptr.to(tl.pointer_type(tl.uint32))

    for block in range(tl.cdiv(k, block_k)):
        col = block * block_k + cols
        valid = (col < k) & valid_row
        # One 16x64 super-atom has 1,536 bytes. Its three 512-byte planes
        # hold a meta word and two low-byte words per lane and 16-column atom.
        super_atom = ((row_n // 16) * (k // 64) + col // 64) * 1536
        atom = (col % 64) // 16
        lane = (row_n % 8) * 4 + ((col % 16) % 8) // 2
        slot = ((row_n % 16) // 8) * 2 + ((col % 16) // 8) * 4 + col % 2
        meta_offset = super_atom + 16 * lane + 4 * atom
        low_offset = (super_atom + tl.where(atom < 2, 512, 1024)
                      + 16 * lane + 8 * (atom % 2))
        meta = tl.load(packed_u32 + meta_offset // 4, valid, other=0)
        low0 = tl.load(packed_u32 + low_offset // 4, valid, other=0)
        low1 = tl.load(packed_u32 + low_offset // 4 + 1, valid, other=0)
        low = tl.where(slot < 4, low0 >> (8 * slot), low1 >> (8 * (slot - 4))) & 255
        meta_byte = (meta >> (8 * (slot // 2))) & 255
        high_even = (meta_byte & 0x87) + exp_base
        high_odd = (((meta_byte << 1) & 0x80) | ((meta_byte >> 3) & 0x07)) + exp_base
        high = tl.where(slot % 2 == 0, high_even, high_odd)
        ordinary = (high << 8) | low
        # A zero meta word is an escaped group; low0 is its table index.
        escaped = tl.load(escapes_ptr + low0 * 8 + slot,
                          valid & (meta == 0), other=0).to(tl.uint16, bitcast=True)
        bits = tl.where(meta == 0, escaped.to(tl.uint32), ordinary)
        weight = (bits << 16).to(tl.float32, bitcast=True)
        activation = tl.load(x_ptr + row_m[:, None] * k + col[None, :],
                             (row_m[:, None] < m) & valid[None, :], other=0)
        accum = tl.fma(activation, weight[None, :], accum)

    result = tl.sum(accum, axis=1)
    if has_bias:
        result += tl.load(bias_ptr + row_n, valid_row, other=0)
    tl.store(out_ptr + row_m * output_n + output_row, result, row_m < m)


@triton.jit
def _legacy_linear_bf16_fp32_tc(
    x_ptr, packed_ptr, escapes_ptr, bias_ptr, out_ptr,
    n: tl.constexpr, k: tl.constexpr, m: tl.constexpr,
    exp_base: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
    has_bias: tl.constexpr,
):
    """Decode a short-K output tile once for an FP32-accumulating BF16 MMA."""
    output_rows = tl.program_id(0) * block_n + tl.arange(0, block_n)
    input_rows = tl.program_id(1) * 16 + tl.arange(0, 16)
    row_n = output_rows[:, None]
    col = tl.arange(0, block_k)[None, :]
    valid = (row_n < n) & (col < k)
    packed_u32 = packed_ptr.to(tl.pointer_type(tl.uint32))
    super_atom = ((row_n // 16) * (k // 64) + col // 64) * 1536
    atom = (col % 64) // 16
    lane = (row_n % 8) * 4 + ((col % 16) % 8) // 2
    slot = ((row_n % 16) // 8) * 2 + ((col % 16) // 8) * 4 + col % 2
    meta_offset = super_atom + 16 * lane + 4 * atom
    low_offset = (super_atom + tl.where(atom < 2, 512, 1024)
                  + 16 * lane + 8 * (atom % 2))
    meta = tl.load(packed_u32 + meta_offset // 4, valid, other=0)
    low0 = tl.load(packed_u32 + low_offset // 4, valid, other=0)
    low1 = tl.load(packed_u32 + low_offset // 4 + 1, valid, other=0)
    low = tl.where(slot < 4, low0 >> (8 * slot), low1 >> (8 * (slot - 4))) & 255
    meta_byte = (meta >> (8 * (slot // 2))) & 255
    high_even = (meta_byte & 0x87) + exp_base
    high_odd = (((meta_byte << 1) & 0x80) | ((meta_byte >> 3) & 0x07)) + exp_base
    high = tl.where(slot % 2 == 0, high_even, high_odd)
    ordinary = (high << 8) | low
    escaped = tl.load(escapes_ptr + low0 * 8 + slot,
                      valid & (meta == 0), other=0).to(tl.uint16, bitcast=True)
    bits = tl.where(meta == 0, escaped.to(tl.uint32), ordinary)
    weight = (bits << 16).to(tl.float32, bitcast=True).to(tl.bfloat16)
    activation = tl.load(x_ptr + input_rows[:, None] * k + col,
                         (input_rows[:, None] < m) & (col < k), other=0)
    result = tl.dot(activation, tl.trans(weight))
    if has_bias:
        result += tl.load(bias_ptr + output_rows, output_rows < n, other=0)[None, :]
    tl.store(out_ptr + input_rows[:, None] * n + output_rows[None, :], result,
             (input_rows[:, None] < m) & (output_rows[None, :] < n))


def _block_m(m):
    """The launcher's compile-time row tile: at most four buckets."""
    return min(triton.next_power_of_2(m), 8)


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


def _warm_linear_fp32(kernel, m, n=ROUTER_N, k=ROUTER_K):
    """Bind the launcher's real arguments without a CUDA context."""
    return kernel.warmup(
        torch.float32, torch.int32, torch.int16, torch.float32,
        torch.int32, torch.float32,
        n, n, k, m, 55, _block_m(m), 256, False, False,
        num_warps=4, grid=(1,),
    )


def _warm_bf16_tc(kernel, m, n=HEAD_N, k=HEAD_K):
    return kernel.warmup(
        torch.bfloat16, torch.int32, torch.int16, torch.float32, torch.float32,
        n, k, m, 55, 16, triton.next_power_of_2(k), False,
        num_warps=8, grid=(1,),
    )


def test_linear_fp32_widths_share_compiled_variants(cpu_sm80):
    """Only the launcher's ``block_m`` buckets may compile new variants."""
    kernel = _fp32._linear_fp32
    params = {p.name: p for p in kernel.params}
    assert not params["m"].is_constexpr and params["m"].do_not_specialize
    assert params["block_m"].is_constexpr and params["k"].is_constexpr

    widths = (1, 2, 3, 4, 5, 7, 9, 16, 40, 100, 200, 255)
    buckets = {_block_m(m) for m in widths}
    assert buckets == {1, 2, 4, 8}

    before = set(kernel.device_caches[0][0]) if 0 in kernel.device_caches else set()
    compiled = [_warm_linear_fp32(kernel, m) for m in widths]
    assert len(set(kernel.device_caches[0][0]) - before) == len(buckets)
    assert len({id(c) for c in compiled}) == len(buckets)

    # Negative control: the former constexpr row count compiled per width.
    legacy = triton.jit(_legacy_linear_fp32.fn)
    legacy_compiled = [_warm_linear_fp32(legacy, m) for m in widths]
    assert len(legacy.device_caches[0][0]) == len(widths)
    assert len({id(c) for c in legacy_compiled}) == len(widths)


def test_bf16_fp32_tc_widths_share_one_compiled_variant(cpu_sm80):
    kernel = _fp32._linear_bf16_fp32_tc
    params = {p.name: p for p in kernel.params}
    assert not params["m"].is_constexpr and params["m"].do_not_specialize
    assert params["n"].is_constexpr and params["k"].is_constexpr

    widths = (2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16)
    before = set(kernel.device_caches[0][0]) if 0 in kernel.device_caches else set()
    compiled = [_warm_bf16_tc(kernel, m) for m in widths]
    assert len(set(kernel.device_caches[0][0]) - before) == 1
    assert len({id(c) for c in compiled}) == 1

    # Negative control: the former constexpr row count compiled per width.
    legacy = triton.jit(_legacy_linear_bf16_fp32_tc.fn)
    legacy_compiled = [_warm_bf16_tc(legacy, m) for m in widths]
    assert len(legacy.device_caches[0][0]) == len(widths)
    assert len({id(c) for c in legacy_compiled}) == len(widths)


def _ptx_float_ops(compiled):
    ops = []
    for line in compiled.asm["ptx"].splitlines():
        match = re.match(r"^\s*(?:@!?%p\d+\s+)?([a-z][\w.]*)", line)
        if match and any(t in match.group(1) for t in (".f32", ".bf16", ".tf32", "mma")):
            ops.append(match.group(1))
    return ops


def test_linear_fp32_ptx_arithmetic_matches_former_variants(cpu_sm80):
    kernel = _fp32._linear_fp32
    legacy = triton.jit(_legacy_linear_fp32.fn)
    for m in (2, 5, 9, 40, 200):
        assert _ptx_float_ops(_warm_linear_fp32(kernel, m)) == _ptx_float_ops(
            _warm_linear_fp32(legacy, m)), m


def test_bf16_fp32_tc_ptx_arithmetic_matches_former_variants(cpu_sm80):
    kernel = _fp32._linear_bf16_fp32_tc
    legacy = triton.jit(_legacy_linear_bf16_fp32_tc.fn)
    for m in (2, 5, 9, 16):
        assert _ptx_float_ops(_warm_bf16_tc(kernel, m)) == _ptx_float_ops(
            _warm_bf16_tc(legacy, m)), m


def _nf12_native_ready() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    try:
        import gptqmodel_nf12

        return bool(gptqmodel_nf12.available())
    except Exception:  # noqa: BLE001 - an unloadable optional accelerator is a skip
        return False


@pytest.mark.skipif(not _nf12_native_ready(), reason="needs the native NVIDIA-CUDA NF12 kernels")
def test_cuda_launchers_match_former_kernels_bitwise_and_share_variants():
    import gptqmodel_nf12 as nf12

    generator = torch.Generator(device="cuda").manual_seed(5393)
    router = nf12.encode(
        (torch.randn((ROUTER_N, ROUTER_K), generator=generator, device="cuda") * 0.02).to(torch.bfloat16)
    )
    head = nf12.encode(
        (torch.randn((HEAD_N, HEAD_K), generator=generator, device="cuda") * 0.02).to(torch.bfloat16)
    )
    bias = torch.randn(ROUTER_N, generator=generator, device="cuda")

    widths = (2, 3, 4, 5, 7, 9, 16, 40, 100, 200)
    kernel = _fp32._linear_fp32
    before = set(kernel.device_caches[0][0]) if 0 in kernel.device_caches else set()
    for m in widths:
        x = torch.randn((m, ROUTER_K), generator=generator, device="cuda")
        got = nf12.linear_fp32(x, router, bias)
        expected = torch.empty((m, ROUTER_N), dtype=torch.float32, device="cuda")
        _legacy_linear_fp32[(ROUTER_N, triton.cdiv(m, _block_m(m)))](
            x, router.packed, router.escapes, bias, x, expected,
            ROUTER_N, ROUTER_N, ROUTER_K, m, router.exp_base, _block_m(m), 256,
            True, False, num_warps=4,
        )
        assert torch.equal(got.view(torch.int32), expected.view(torch.int32)), m
    assert len(set(kernel.device_caches[0][0]) - before) == len({_block_m(m) for m in widths})

    x = torch.randn((5, ROUTER_K), generator=generator, device="cuda")
    got = nf12.linear_fp32(x, router, bias).double()
    oracle = x.double() @ router.decode().double().T + bias.double()
    assert (got - oracle).abs().max() <= 1e-5 * oracle.abs().max()

    rows = torch.tensor([0, 3, 7, 11, 63], dtype=torch.int64, device="cuda")
    x = torch.randn((7, ROUTER_K), generator=generator, device="cuda")
    got = nf12.linear_fp32(x, router, bias, rows)
    expected = torch.empty((7, rows.numel()), dtype=torch.float32, device="cuda")
    _legacy_linear_fp32[(rows.numel(), triton.cdiv(7, _block_m(7)))](
        x, router.packed, router.escapes, bias, rows, expected,
        ROUTER_N, rows.numel(), ROUTER_K, 7, router.exp_base, _block_m(7), 256,
        True, True, num_warps=4,
    )
    assert torch.equal(got.view(torch.int32), expected.view(torch.int32))

    tc_kernel = _fp32._linear_bf16_fp32_tc
    before = set(tc_kernel.device_caches[0][0]) if 0 in tc_kernel.device_caches else set()
    # ``linear_bf16_fp32`` sends only 2..16 rows to the Tensor Core kernel.
    tc_widths = (2, 3, 4, 5, 7, 9, 12, 15, 16)
    for m in tc_widths:
        x = torch.randn((m, HEAD_K), generator=generator, device="cuda", dtype=torch.bfloat16)
        got = nf12.linear_bf16_fp32(x, head)
        expected = torch.empty((m, HEAD_N), dtype=torch.float32, device="cuda")
        _legacy_linear_bf16_fp32_tc[(triton.cdiv(HEAD_N, 16), triton.cdiv(m, 16))](
            x, head.packed, head.escapes, x, expected,
            HEAD_N, HEAD_K, m, head.exp_base, 16, 256, False,
            num_warps=8,
        )
        assert torch.equal(got.view(torch.int32), expected.view(torch.int32)), m
    assert len(set(tc_kernel.device_caches[0][0]) - before) == 1
