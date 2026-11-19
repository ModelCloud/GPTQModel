# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Multi-row NF12 FP32 gate kernel: same bits as the B1 plane kernel, fewer reloads.

``_linear_fp32_gate_planes`` (one gate row per CTA) re-reads every activation
row in each of its ``n`` CTAs. ``_linear_fp32_gate_planes_rows`` decodes
``rows_n`` gate rows per CTA for ``rows_m`` register-resident activation rows.
Its contract is per-output bit identity with the plane kernel for every row
count and geometry.

CPU-only evidence (no GPU, no driver):

* sm80 compile of the Gluon kernel: the accumulator keeps the plane kernel's
  per-row lane layout (sizePerThread 8 x 32 threads x 4 warps along K) for
  every geometry, with no local-memory spills;
* a PTX emulator (``nf12_ptx_emulator``) executes the compiled reference and
  candidate kernels on synthetic BF16 gates with escaped groups and requires
  bit equality per output; a reference compiled with a different K tile
  (512 lanes x 10 steps) is the negative control that proves the emulator
  sees reduction-order changes.

The CUDA test compares the real launches when a GPU is present.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


triton = pytest.importorskip("triton")
pytest.importorskip("triton.backends.nvidia")

from gptqmodel_nf12 import _fp32  # noqa: E402


if _fp32.gate_rows_error() is not None:  # pragma: no cover - Triton without Gluon
    pytest.skip(_fp32.gate_rows_error(), allow_module_level=True)

sys.path.insert(0, str(Path(__file__).parent))
from nf12_ptx_emulator import Emulator, Memory  # noqa: E402


N, K = 384, 5120
PLANE_SIG = {
    "x_ptr": "*fp32", "meta_ptr": "*i32", "low_ptr": "*u8", "escapes_ptr": "*i16",
    "bias_ptr": "*fp32", "out_ptr": "*fp32", "n": "constexpr", "exp_base": "constexpr",
    "has_bias": "constexpr", "m": "constexpr", "block_k": "constexpr",
}
ROWS_SIG = {
    "x_ptr": "*fp32", "meta_ptr": "*i32", "low_ptr": "*u8", "escapes_ptr": "*i16",
    "bias_ptr": "*fp32", "out_ptr": "*fp32", "m": "i32", "n": "constexpr",
    "exp_base": "constexpr", "has_bias": "constexpr", "rows_n": "constexpr",
    "rows_m": "constexpr", "unroll": "constexpr", "full_tiles": "constexpr",
}
PINNED_LAYOUT = ("#ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], "
                 "warpsPerCTA = [1, 4], order = [1, 0]}>")


def _compile(fn, signature, constexprs, gluon=False):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    source_type = ASTSource
    if gluon:
        from triton.experimental.gluon._runtime import GluonASTSource as source_type
    attrs = {(i,): [["tt.divisibility", 16]]
             for i, kind in enumerate(signature.values()) if kind.startswith("*")}
    source = source_type(fn=fn, signature=signature, constexprs=constexprs, attrs=attrs)
    return triton.compile(source, target=GPUTarget("cuda", 80, 32), options={"num_warps": 4})


def _plane_ptx(m, exp_base, block_k=1024):
    return _compile(_fp32._linear_fp32_gate_planes, PLANE_SIG, {
        "n": N, "exp_base": exp_base, "has_bias": False, "m": m, "block_k": block_k,
    }).asm["ptx"]


def _rows_kernel(m, exp_base, rows_n, rows_m, unroll):
    return _compile(_fp32._linear_fp32_gate_planes_rows, ROWS_SIG, {
        "n": N, "exp_base": exp_base, "has_bias": False,
        "rows_n": rows_n, "rows_m": rows_m, "unroll": unroll,
        "full_tiles": m % rows_m == 0,
    }, gluon=True)


def _synthetic_gate(seed=898):
    """BF16 [384, 5120] gate-like weights plus outliers that force escaped groups."""
    g = torch.Generator().manual_seed(seed)
    dense = torch.randn((N, K), generator=g) * 0.02
    rows = torch.randint(0, N, (64,), generator=g)
    cols = torch.randint(0, K, (64,), generator=g)
    dense[rows, cols] = torch.randn(64, generator=g) * 40.0   # exponent above the window
    dense[rows.flip(0), cols] = 1e-9                          # exponent below the window
    dense[5, 8:16] = 0.0                                      # all-zero group: meta word 0
    return dense.bfloat16()


def _exp_base(dense):
    top = ((dense.view(torch.int16).to(torch.int32) & 65535) >> 8) & 127
    hist = torch.bincount(top.flatten(), minlength=128)
    return max(range(121), key=lambda base: int(hist[base:base + 8].sum()))


def _gate_view(dense, exp_base):
    """``prepare_fp32_gate`` on CPU tensors (same ops; the GPU decode is the input)."""
    n = dense.shape[0]
    bits = (dense.view(torch.int16).to(torch.int32) & 65535).view(n, 640, 8)
    exponent = (bits >> 7) & 255
    sign = bits >> 15
    offset = (exponent - 2 * exp_base) >> 1
    meta_bytes = ((sign[:, :, 0::2] << 7) | offset[:, :, 0::2] |
                  (sign[:, :, 1::2] << 6) | (offset[:, :, 1::2] << 3))
    meta = (meta_bytes[:, :, 0] | (meta_bytes[:, :, 1] << 8) |
            (meta_bytes[:, :, 2] << 16) | (meta_bytes[:, :, 3] << 24)).contiguous()
    low = (bits & 255).to(torch.uint8).view(n, K).contiguous()
    valid = (((exponent >= 2 * exp_base) & (exponent <= 2 * exp_base + 15)).all(-1) & (meta != 0))
    escaped_groups = ~valid
    escapes = bits[escaped_groups].to(torch.int16).contiguous()
    count = escapes.shape[0]
    low_words = low.view(torch.int32).view(n, 640, 2)
    low_words[:, :, 0][escaped_groups] = torch.arange(count, dtype=torch.int32)
    low_words[:, :, 1][escaped_groups] = 0
    meta[escaped_groups] = 0
    return meta, low, escapes, count


def _emulate(ptx, grid, mem, pointers, m):
    out = np.full((m, N), np.nan, np.float32)
    out_address = mem.alloc(out)
    emulator = Emulator(ptx, mem)
    args = list(pointers) + [out_address, m]
    args += [0] * (len(emulator.params) - len(args))  # Triton scratch pointers
    emulator.run(grid, args)
    return mem.read(out_address, out.nbytes).view(np.float32).reshape(m, N)


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


def _warm_rows(kernel, m, rows_n=2, rows_m=6, unroll=False, has_bias=False):
    args = [torch.float32, torch.int32, torch.uint8, torch.int16,
            torch.float32, torch.float32, m, N, 55, has_bias, rows_n, rows_m, unroll]
    if "full_tiles" in {p.name for p in kernel.params}:
        args.append(m % rows_m == 0)
    return kernel.warmup(*args, num_warps=4, grid=(1,))


def _warm_plane(kernel, m):
    return kernel.warmup(torch.float32, torch.int32, torch.uint8, torch.int16,
                         torch.float32, torch.float32,
                         N, 55, False, m, 1024, num_warps=4, grid=(1,))


def test_rows_widths_share_compiled_variants(cpu_sm80):
    """The row count is a runtime argument; only the mask-skip flag specializes."""
    kernel = _fp32._linear_fp32_gate_planes_rows
    params = {p.name: p for p in kernel.params}
    assert not params["m"].is_constexpr and params["m"].do_not_specialize
    assert params["n"].is_constexpr and params["rows_m"].is_constexpr
    assert params["full_tiles"].is_constexpr

    full_widths = (6, 12, 18, 24)
    before = set(kernel.device_caches[0][0]) if 0 in kernel.device_caches else set()
    compiled = [_warm_rows(kernel, m) for m in full_widths]
    assert len(set(kernel.device_caches[0][0]) - before) == 1
    assert len({id(c) for c in compiled}) == 1

    partial_widths = (7, 13, 19)
    compiled += [_warm_rows(kernel, m) for m in partial_widths]
    assert len(set(kernel.device_caches[0][0]) - before) == 2
    assert len({id(c) for c in compiled}) == 2

    # Negative control: a sibling kernel whose row count is still constexpr
    # compiles one variant per width, so the cache count above detects a
    # regression back to per-width specialization.
    plane = _fp32._linear_fp32_gate_planes
    plane_before = set(plane.device_caches[0][0]) if 0 in plane.device_caches else set()
    plane_compiled = [_warm_plane(plane, m) for m in full_widths]
    assert len(set(plane.device_caches[0][0]) - plane_before) == len(full_widths)
    assert len({id(c) for c in plane_compiled}) == len(full_widths)


@pytest.fixture(scope="module")
def gate():
    dense = _synthetic_gate()
    exp_base = _exp_base(dense)
    meta, low, escapes, count = _gate_view(dense, exp_base)
    assert count > 0
    return dense, exp_base, meta, low, escapes


@pytest.mark.parametrize("rows_n,rows_m,unroll", [(1, 6, False), (2, 6, False), (2, 3, True),
                                                  (4, 3, False), (4, 2, True), (8, 1, False)])
def test_accumulator_layout_is_pinned_and_spill_free(rows_n, rows_m, unroll):
    compiled = _rows_kernel(24, 55, rows_n, rows_m, unroll)
    ttgir = compiled.asm["ttgir"]
    assert PINNED_LAYOUT in ttgir
    assert "tt.reduce" in ttgir and "axis = 1" in ttgir
    assert "ld.local" not in compiled.asm["ptx"] and "st.local" not in compiled.asm["ptx"]


def test_b1_plane_kernel_lane_layout_is_the_pinned_one():
    """The reference reduces a 1-D [1024] tile with the same per-thread lane mapping."""
    ttgir = _compile(_fp32._linear_fp32_gate_planes, PLANE_SIG, {
        "n": N, "exp_base": 55, "has_bias": False, "m": 6, "block_k": 1024}).asm["ttgir"]
    assert ("#ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], "
            "warpsPerCTA = [4], order = [0]}>") in ttgir


@pytest.mark.parametrize("m,geometries", [
    (6, [(2, 6, False), (4, 3, True), (8, 1, False)]),
    (12, [(2, 6, True), (4, 2, False)]),
])
def test_rows_kernel_matches_b1_plane_bits_in_ptx_emulation(gate, m, geometries):
    dense, exp_base, meta, low, escapes = gate
    g = torch.Generator().manual_seed(m)
    x = torch.randn((m, K), generator=g) * torch.exp(torch.randn((m, K), generator=g))
    mem = Memory()
    pointers = [mem.alloc(x.numpy()), mem.alloc(meta.numpy()), mem.alloc(low.numpy()),
                mem.alloc(escapes.numpy()), mem.alloc(np.zeros(N, np.float32))]
    reference = np.zeros((m, N), np.float32)
    plane = _plane_ptx(6, exp_base)
    for s in range(m // 6):
        rows = mem.alloc(x[s * 6:(s + 1) * 6].numpy())
        reference[s * 6:(s + 1) * 6] = _emulate(plane, (N, 1), mem, [rows, *pointers[1:]], 6)
    oracle = x.double().numpy() @ dense.double().numpy().T
    scale = np.abs(oracle).max(1, keepdims=True)
    assert (np.abs(reference - oracle) / scale).max() < 3e-7

    for rows_n, rows_m, unroll in geometries:
        ptx = _rows_kernel(m, exp_base, rows_n, rows_m, unroll).asm["ptx"]
        got = _emulate(ptx, (N // rows_n, -(-m // rows_m)), mem, pointers, m)
        assert np.array_equal(got.view(np.uint32), reference.view(np.uint32)), (rows_n, rows_m, unroll)

    # Negative control: a 512-lane x 10-step tree must be visible as different bits.
    other = np.zeros((6, N), np.float32)
    other[:] = _emulate(_plane_ptx(6, exp_base, block_k=512), (N, 1), mem,
                        [mem.alloc(x[:6].numpy()), *pointers[1:]], 6)
    assert (other.view(np.uint32) != reference[:6].view(np.uint32)).mean() > 0.2


def test_launcher_validates_geometry_and_requires_prepared_view():
    weight = type("W", (), {"shape": (N, K), "device": torch.device("cpu"), "exp_base": 55})()
    with pytest.raises(ValueError, match="prepare_fp32_gate"):
        _fp32.linear_fp32_gate_rows(torch.zeros(6, K), weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("m", [1, 6, 7, 12, 13, 18, 24])
def test_cuda_rows_kernel_matches_b1_launches_bitwise(m):
    import gptqmodel_nf12 as nf12

    dense = _synthetic_gate().cuda()
    weight = nf12.encode(dense)
    nf12.prepare_fp32_gate(weight)
    x = torch.randn((m, K), device="cuda")
    if m == 1:
        reference = nf12.linear_fp32(x, weight)
    else:
        reference = torch.cat([nf12.linear_fp32_staged_b1(x[s:s + 6], weight)
                               for s in range(0, m, 6)])
    for rows_n, rows_m, unroll in [(1, 6, False), (2, 6, False), (2, 3, True), (4, 3, False),
                                   (4, 2, True), (8, 1, False)]:
        got = nf12.linear_fp32_gate_rows(x, weight, rows_n=rows_n, rows_m=rows_m, unroll=unroll)
        assert torch.equal(got.view(torch.int32), reference.view(torch.int32)), (rows_n, rows_m, unroll)
