# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Custom TT-Metalium kernel for GPTQ int4 dequantization.

The TTNN dequant path builds a dense BF16 tile with a chain of ops (nibble
planes, stack, permute, two embedding gathers, subtract, multiply) and measures
~1.27 ms per layer against ~0.06 ms for the GEMM it feeds. Almost all of that is
op dispatch, not arithmetic.

This module dispatches a **custom kernel** instead, so the nibble extraction and
the group scale/zero application happen in one pass on the core, with no
intermediate tensors. It is the first step of the fused int4 design recorded in
``hw/tenstorrent.md`` section 9: prove the numerics in a kernel, then widen to a
GEMM that consumes the dequantized tile inside the compute path rather than
writing it back out.

Validated on a p300c: 8192 codes dequantized **bit-exactly** against an
independent torch reference (BF16 output, truncation matched).

Dispatch note: the wheel's JIT include paths are baked to the vendor's build
container, so ``generic_op`` with ``SourceType.SOURCE_CODE`` only compiles if the
kernel includes ``api/dataflow/dataflow_api.h`` (the flat ``dataflow_api.h`` does
not resolve) and addresses DRAM through ``InterleavedAddrGen<true>``.
"""

from __future__ import annotations

import struct
from collections import OrderedDict
from typing import Tuple

import torch


#: One core, one page in, one page out. ``nwords`` is a runtime argument.
INT4_DEQUANT_KERNEL_SRC = r"""
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t in_addr    = get_arg_val<uint32_t>(0);
    uint32_t out_addr   = get_arg_val<uint32_t>(1);
    uint32_t nwords     = get_arg_val<uint32_t>(2);
    uint32_t scale_bits = get_arg_val<uint32_t>(3);
    uint32_t zero       = get_arg_val<uint32_t>(4);

    float scale;
    __builtin_memcpy(&scale, &scale_bits, 4);
    float fzero = (float)zero;

    constexpr uint32_t IN_CB = 0;
    constexpr uint32_t OUT_CB = 16;
    uint32_t in_bytes  = nwords * 4;
    uint32_t out_bytes = nwords * 8 * 2;

    cb_reserve_back(IN_CB, 1);
    uint32_t in_l1 = get_write_ptr(IN_CB);
    InterleavedAddrGen<true> in_gen = {.bank_base_address = in_addr, .page_size = in_bytes};
    noc_async_read(in_gen.get_noc_addr(0), in_l1, in_bytes);
    noc_async_read_barrier();
    cb_push_back(IN_CB, 1);

    cb_reserve_back(OUT_CB, 1);
    uint32_t out_l1 = get_write_ptr(OUT_CB);

    volatile tt_l1_ptr uint32_t* words = (volatile tt_l1_ptr uint32_t*)in_l1;
    volatile tt_l1_ptr uint16_t* out   = (volatile tt_l1_ptr uint16_t*)out_l1;

    for (uint32_t i = 0; i < nwords; ++i) {
        uint32_t word = words[i];
        for (uint32_t lane = 0; lane < 8; ++lane) {
            uint32_t code = (word >> (4u * lane)) & 0xFu;
            float v = scale * ((float)code - fzero);
            uint32_t bits;
            __builtin_memcpy(&bits, &v, 4);
            out[i * 8 + lane] = (uint16_t)(bits >> 16);
        }
    }
    cb_push_back(OUT_CB, 1);

    InterleavedAddrGen<true> out_gen = {.bank_base_address = out_addr, .page_size = out_bytes};
    noc_async_write(out_l1, out_gen.get_noc_addr(0), out_bytes);
    noc_async_write_barrier();
}
"""

#: Tiled variant. Each core is told its page range through **per-core runtime
#: args**, so the kernel never derives an index from logical coordinates -- that
#: mapping is not guaranteed to start at 0, and an out-of-range page index was
#: the likely cause of an earlier deadlock.
INT4_DEQUANT_TILED_SRC = r"""
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t in_addr    = get_arg_val<uint32_t>(0);
    uint32_t out_addr   = get_arg_val<uint32_t>(1);
    uint32_t first_page = get_arg_val<uint32_t>(2);
    uint32_t n_pages    = get_arg_val<uint32_t>(3);
    uint32_t page_words = get_arg_val<uint32_t>(4);
    uint32_t scale_bits = get_arg_val<uint32_t>(5);
    uint32_t zero       = get_arg_val<uint32_t>(6);

    float scale;
    __builtin_memcpy(&scale, &scale_bits, 4);
    float fzero = (float)zero;

    constexpr uint32_t IN_CB = 0;
    constexpr uint32_t OUT_CB = 16;

    InterleavedAddrGen<true> in_gen  = {.bank_base_address = in_addr,  .page_size = page_words * 4u};
    InterleavedAddrGen<true> out_gen = {.bank_base_address = out_addr, .page_size = page_words * 16u};

    for (uint32_t p = 0; p < n_pages; ++p) {
        uint32_t page_id = first_page + p;

        cb_reserve_back(IN_CB, 1);
        uint32_t in_l1 = get_write_ptr(IN_CB);
        noc_async_read(in_gen.get_noc_addr(page_id), in_l1, page_words * 4u);
        noc_async_read_barrier();
        cb_push_back(IN_CB, 1);
        cb_wait_front(IN_CB, 1);

        cb_reserve_back(OUT_CB, 1);
        uint32_t out_l1 = get_write_ptr(OUT_CB);

        volatile tt_l1_ptr uint32_t* words = (volatile tt_l1_ptr uint32_t*)in_l1;
        volatile tt_l1_ptr uint16_t* out   = (volatile tt_l1_ptr uint16_t*)out_l1;

        for (uint32_t i = 0; i < page_words; ++i) {
            uint32_t word = words[i];
            for (uint32_t lane = 0; lane < 8; ++lane) {
                uint32_t code = (word >> (4u * lane)) & 0xFu;
                float v = scale * ((float)code - fzero);
                uint32_t bits;
                __builtin_memcpy(&bits, &v, 4);
                out[i * 8u + lane] = (uint16_t)(bits >> 16);
            }
        }
        cb_push_back(OUT_CB, 1);
        cb_wait_front(OUT_CB, 1);

        noc_async_write(out_l1, out_gen.get_noc_addr(page_id), page_words * 16u);
        noc_async_write_barrier();

        cb_pop_front(IN_CB, 1);
        cb_pop_front(OUT_CB, 1);
    }
}
"""


def pack_codes_to_words(codes: torch.Tensor) -> torch.Tensor:
    """Pack logical 4-bit codes (low nibble first) into uint32 words."""
    flat = codes.reshape(-1).to(torch.int64)
    if flat.numel() % 8:
        raise ValueError(f"need a multiple of 8 codes, got {flat.numel()}")
    words = torch.zeros(flat.numel() // 8, dtype=torch.int64)
    for lane in range(8):
        words |= (flat[lane::8] & 0xF) << (4 * lane)
    return words.to(torch.int32)


def bf16_truncate(values: torch.Tensor) -> torch.Tensor:
    """Match the kernel's float->BF16 truncation (it drops the low 16 bits)."""

    bits = values.to(torch.float32).view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return ((bits >> 16) << 16).to(torch.int32).view(torch.float32)


def reference_dequant(words: torch.Tensor, scale: float, zero: int) -> torch.Tensor:
    """Independent torch reference for what the kernel must produce."""

    flat = words.reshape(-1).to(torch.int64) & 0xFFFFFFFF
    lanes = [((flat >> (4 * lane)) & 0xF).to(torch.float32) for lane in range(8)]
    codes = torch.stack(lanes, dim=1).reshape(-1)
    return bf16_truncate(scale * (codes - float(zero)))


def dequantize_on_device(
    words: torch.Tensor, scale: float, zero: int, *, device
) -> torch.Tensor:
    """Run the custom kernel and return the dequantized BF16 values on the host."""

    from .ttnn_runtime import import_ttnn

    ttnn = import_ttnn()
    flat = words.reshape(-1).to(torch.int32)
    nwords = int(flat.numel())
    ncodes = nwords * 8

    inp = ttnn.to_device(
        ttnn.from_torch(flat.reshape(1, nwords), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
        device,
    )
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, ncodes]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    core = ttnn.CoreCoord(0, 0)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    in_fmt = ttnn.CBFormatDescriptor(
        buffer_index=0, data_format=ttnn.uint32, page_size=nwords * 4
    )
    out_fmt = ttnn.CBFormatDescriptor(
        buffer_index=16, data_format=ttnn.bfloat16, page_size=ncodes * 2
    )
    cbs = [
        ttnn.CBDescriptor(total_size=nwords * 4, core_ranges=grid, format_descriptors=[in_fmt]),
        ttnn.CBDescriptor(total_size=ncodes * 2, core_ranges=grid, format_descriptors=[out_fmt]),
    ]

    scale_bits = struct.unpack("<I", struct.pack("<f", float(scale)))[0]
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [inp.buffer_address(), out.buffer_address(), nwords, scale_bits, int(zero)]

    kd = ttnn.KernelDescriptor(
        kernel_source=INT4_DEQUANT_KERNEL_SRC,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=grid,
        compile_time_args=[],
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    program = ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=cbs)
    ttnn.generic_op([inp, out], program)
    ttnn.synchronize_device(device)
    return ttnn.to_torch(out).reshape(-1).to(torch.float32)


def dequantize_tiled_on_device(
    words: torch.Tensor,
    scale: float,
    zero: int,
    *,
    device,
    grid: Tuple[int, int] = (8, 8),
    page_words: int = 1152,
) -> torch.Tensor:
    """Dequantize a whole layer over a 2D grid, one page per chunk.

    Page-based addressing is required: a single huge page does not rotate DRAM
    banks correctly, so the tensor is stored one page per chunk. Each core's page
    range is passed as per-core runtime args rather than derived from logical
    coordinates.
    """

    from .ttnn_runtime import import_ttnn

    ttnn = import_ttnn()
    flat = words.reshape(-1).to(torch.int32)
    nwords = int(flat.numel())
    grid_w, grid_h = int(grid[0]), int(grid[1])
    cores = grid_w * grid_h
    if nwords % page_words:
        raise ValueError(f"nwords={nwords} must be divisible by page_words={page_words}")
    pages = nwords // page_words
    if pages % cores:
        raise ValueError(f"pages={pages} must be divisible by cores={cores} ({grid_w}x{grid_h})")
    per_core = pages // cores

    inp = ttnn.to_device(
        ttnn.from_torch(
            flat.reshape(pages, page_words), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        ),
        device,
    )
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([pages, page_words * 8]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    lo = ttnn.CoreCoord(0, 0)
    hi = ttnn.CoreCoord(grid_w - 1, grid_h - 1)
    core_range = ttnn.CoreRangeSet([ttnn.CoreRange(lo, hi)])

    in_fmt = ttnn.CBFormatDescriptor(
        buffer_index=0, data_format=ttnn.uint32, page_size=page_words * 4
    )
    out_fmt = ttnn.CBFormatDescriptor(
        buffer_index=16, data_format=ttnn.bfloat16, page_size=page_words * 16
    )
    cbs = [
        ttnn.CBDescriptor(total_size=page_words * 4, core_ranges=core_range, format_descriptors=[in_fmt]),
        ttnn.CBDescriptor(total_size=page_words * 16, core_ranges=core_range, format_descriptors=[out_fmt]),
    ]

    scale_bits = struct.unpack("<I", struct.pack("<f", float(scale)))[0]
    rt = ttnn.RuntimeArgs()
    for core in range(cores):
        rt[core][0] = [
            inp.buffer_address(), out.buffer_address(),
            core * per_core, per_core, page_words, scale_bits, int(zero),
        ]

    kd = ttnn.KernelDescriptor(
        kernel_source=INT4_DEQUANT_TILED_SRC,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_range,
        compile_time_args=[],
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    ttnn.generic_op([inp, out], ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=cbs))
    ttnn.synchronize_device(device)
    return ttnn.to_torch(out).reshape(-1).to(torch.float32)


def verify_on_device(nwords: int, scale: float, zero: int, *, device, seed: int = 5) -> Tuple[bool, float]:
    """Dispatch the kernel and compare against the torch reference."""

    torch.manual_seed(seed)
    codes = torch.randint(0, 16, (nwords * 8,), dtype=torch.int64)
    words = pack_codes_to_words(codes)
    got = dequantize_on_device(words, scale, zero, device=device)
    expected = reference_dequant(words, scale, zero)
    return bool(torch.equal(got, expected)), float((got - expected).abs().max())

# --- multi-core grid fused GEMM --------------------------------------------
#
# One N tile per core, whole K reduction, dequant to L1, exact GPTQ. Validated at
# 64 cores (8x8): norm_l1 0.00146, bad_frac 0.000.
#
#   nk/nm/grid     M      K      N    cores   time
#   4/2  2x2      64    128    128       4   0.067 ms
#   8/8  4x4     256    256    512      16   0.117 ms
#   24/8 8x8     256    768   2048      64   0.346 ms
#
# ~3.6x slower than ttnn.linear on the same shape (0.0949 ms). Ablation puts the
# gap in data movement, not arithmetic: staging 45%, BRISC LUT dequant 53%,
# matmul 2%.
#
# Three layout facts are load-bearing, all found the hard way:
#   * per-core runtime args are indexed rt[x][y] by logical coordinate, not a
#     linear index (that fails with "Number of runtime args cannot be modified");
#   * the output tensor must be TILE_LAYOUT: the writer pushes raw tile bytes, so
#     a row-major destination corrupts every face but the first (bad_frac 0.986,
#     row 0 correct to column 15 -- looks like a compute bug);
#   * io_tensors must include every weight buffer, or the kernel reads addresses
#     the program does not track.

#: Device-resident packed weights for :func:`run_fused_gemm_grid`, keyed by the
#: source tensor identity. Bounded so a benchmark loop cannot grow it without end;
#: the oldest entry is dropped first.
_GRID_WEIGHT_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_GRID_WEIGHT_CACHE_MAX = 8


def _GRID_WEIGHT_CACHE_KEY(codes, nk, gw, gh):
    """Cache key for the packed/uploaded grid weights.

    Reuses :func:`tensor_cache_key`, which handles inference tensors (no version
    counter) by falling back to storage identity -- model weights are normally
    packed inside ``InferenceMode``, so a naive ``_version`` read would never hit.
    """

    from ..nn_modules.qlinear._runtime_cache import tensor_cache_key

    return (tensor_cache_key(codes), nk, gw, gh)


def clear_grid_weight_cache() -> None:
    """Drop cached packed grid weights (call on device close or weight mutation)."""

    _GRID_WEIGHT_CACHE.clear()


FUSED_GRID_READER = r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    uint32_t w_addr   = get_arg_val<uint32_t>(0);
    uint32_t act_addr = get_arg_val<uint32_t>(1);
    uint32_t lut_addr = get_arg_val<uint32_t>(2);
    uint32_t out_addr = get_arg_val<uint32_t>(3);
    uint32_t nk       = get_arg_val<uint32_t>(4);
    uint32_t nact     = get_arg_val<uint32_t>(5);
    uint32_t out_page = get_arg_val<uint32_t>(6);
    uint32_t nm       = get_arg_val<uint32_t>(7);
    (void)out_addr; (void)out_page; (void)nm;
    constexpr uint32_t ACT_CB=0, W_CB=1, RAW_CB=2, LUT_CB=3;
    constexpr uint32_t WPT = 128, WB = WPT*4u;

    cb_reserve_back(LUT_CB, 1);
    uint32_t lut_l1 = get_write_ptr(LUT_CB);
    // 256-entry pair table: one u32 store decodes two codes, halving the stores.
    InterleavedAddrGen<true> lg = {.bank_base_address = lut_addr, .page_size = 1024};
    noc_async_read(lg.get_noc_addr(0), lut_l1, 1024);

    cb_reserve_back(ACT_CB, nact);
    uint32_t a_base = get_write_ptr(ACT_CB);
    InterleavedAddrGen<true> ag = {.bank_base_address = act_addr, .page_size = 2048};
    cb_reserve_back(RAW_CB, nk);
    uint32_t r_base = get_write_ptr(RAW_CB);
    InterleavedAddrGen<true> wg = {.bank_base_address = w_addr, .page_size = WB};

    for (uint32_t t = 0; t < nact; ++t) noc_async_read(ag.get_noc_addr(t), a_base + t*2048u, 2048);
    for (uint32_t k = 0; k < nk; ++k) noc_async_read(wg.get_noc_addr(k), r_base + k*WB, WB);
    noc_async_read_barrier();

    cb_reserve_back(W_CB, nk);
    uint32_t w_base = get_write_ptr(W_CB);
    volatile tt_l1_ptr uint32_t* lut = (volatile tt_l1_ptr uint32_t*)lut_l1;
    for (uint32_t k = 0; k < nk; ++k) {
        volatile tt_l1_ptr uint32_t* w = (volatile tt_l1_ptr uint32_t*)(r_base + k*WB);
        volatile tt_l1_ptr uint32_t* o = (volatile tt_l1_ptr uint32_t*)(w_base + k*2048u);
        for (uint32_t i = 0; i < WPT; ++i) {
            uint32_t x = w[i];
            o[i*4u+0] = lut[(x      ) & 0xFFu];
            o[i*4u+1] = lut[(x >>  8) & 0xFFu];
            o[i*4u+2] = lut[(x >> 16) & 0xFFu];
            o[i*4u+3] = lut[(x >> 24) & 0xFFu];
        }
    }
    cb_push_back(ACT_CB, nact);
    cb_push_back(W_CB, nk);

}
"""

FUSED_GRID_COMPUTE = r"""
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_hw_startup.h"
void kernel_main() {
    uint32_t nk = get_arg_val<uint32_t>(0);
    uint32_t nm = get_arg_val<uint32_t>(1);
    constexpr uint32_t ACT_CB=0, W_CB=1, OUT_CB=16;
    compute_kernel_hw_startup<SrcOrder::Reverse>(ACT_CB, W_CB, OUT_CB);
    matmul_init(ACT_CB, W_CB);
    cb_wait_front(ACT_CB, nm*nk);
    cb_wait_front(W_CB, nk);
    for (uint32_t mi = 0; mi < nm; ++mi) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < nk; ++k) matmul_tiles(ACT_CB, W_CB, mi*nk + k, k, 0);
        tile_regs_commit(); tile_regs_wait();
        cb_reserve_back(OUT_CB, 1);
        pack_tile(0, OUT_CB);
        cb_push_back(OUT_CB, 1);
        tile_regs_release();
    }
}
"""

FUSED_GRID_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t out_page = get_arg_val<uint32_t>(1);
    uint32_t nm       = get_arg_val<uint32_t>(2);
    constexpr uint32_t CB = 16;
    InterleavedAddrGen<true> og = {.bank_base_address = out_addr, .page_size = 2048};
    for (uint32_t i = 0; i < nm; ++i) {
        cb_wait_front(CB, 1);
        noc_async_write(get_read_ptr(CB), og.get_noc_addr(out_page + i), 2048);
        noc_async_write_barrier();
        cb_pop_front(CB, 1);
    }
}
"""


def run_fused_gemm_grid(act, codes, scale, zero, *, device, grid=(8, 8)):
    """Fused int4 GEMM across a 2D core grid: one N tile per core, full K.

    `act` is [M, K] with M a multiple of 32; `codes` is [cores, nk, 1024] (or a
    flattened [cores, nk*1024]) -- one 32x32 weight tile set per core, becoming a
    column block of the output. Returns [M, cores*32].

    Validated on a p300c at 64 cores: norm_l1 0.00146, bad_frac 0.000.
    """
    from .ttnn_runtime import import_ttnn

    ttnn = import_ttnn()
    gw, gh = int(grid[0]), int(grid[1])
    cores = gw * gh
    if codes.dim() == 2:
        codes = codes.reshape(cores, -1, 1024)
    nk = codes.numel() // (cores * 1024)
    if codes.numel() != cores * nk * 1024:
        raise ValueError(f"codes must be [cores={cores}, nk, 1024]; got {codes.numel()}")
    if act.dim() != 2 or act.shape[1] != nk * 32:
        raise ValueError(f"act must be [M, {nk * 32}], got {tuple(act.shape)}")
    m = int(act.shape[0])
    if m % 32:
        raise ValueError(f"M={m} must be a multiple of 32")
    nm, nt = m // 32, cores

    # Packing and uploading the weights is ~57 ms at 64 cores, far more than the
    # ~0.24 ms kernel, so it is cached on the weight tensor's identity. Without
    # this the public function is unusable in a model loop.
    cache_key = (_GRID_WEIGHT_CACHE_KEY(codes, nk, gw, gh))
    cached = _GRID_WEIGHT_CACHE.get(cache_key)
    if cached is not None:
        wd = list(cached)
    else:
        order_t = tile_order_indices()
        wd = []
        for n in range(nt):
            rows = torch.stack([
                pack_codes_to_words(codes[n, k][order_t]) for k in range(nk)
            ]).to(torch.int32)
            wd.append(ttnn.to_device(ttnn.from_torch(rows, dtype=ttnn.uint32,
                                                    layout=ttnn.ROW_MAJOR_LAYOUT), device))
        _GRID_WEIGHT_CACHE[cache_key] = tuple(wd)
        _GRID_WEIGHT_CACHE.move_to_end(cache_key)
        while len(_GRID_WEIGHT_CACHE) > _GRID_WEIGHT_CACHE_MAX:
            _GRID_WEIGHT_CACHE.popitem(last=False)
    actd = ttnn.to_device(ttnn.from_torch(act.to(torch.bfloat16).reshape(1, m, nk * 32),
                                          dtype=ttnn.bfloat16,
                                          layout=ttnn.TILE_LAYOUT), device)
    # The LUT is 1 KiB and depends only on (scale, zero); uploading it per call
    # was measurable. Cache it alongside the weights.
    lut_key = ("lut", float(scale), int(zero))
    lutd = _GRID_WEIGHT_CACHE.get(lut_key)
    if lutd is None:
        lutd = ttnn.to_device(ttnn.from_torch(build_dequant_pair_lut(scale, zero),
                                              dtype=ttnn.uint32,
                                              layout=ttnn.ROW_MAJOR_LAYOUT), device)
        _GRID_WEIGHT_CACHE[lut_key] = lutd
        _GRID_WEIGHT_CACHE.move_to_end(lut_key)
        while len(_GRID_WEIGHT_CACHE) > _GRID_WEIGHT_CACHE_MAX:
            _GRID_WEIGHT_CACHE.popitem(last=False)
    # TILE_LAYOUT is required: the writer pushes raw tile bytes.
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([nt, nm, 32, 32]), ttnn.bfloat16,
                                         ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)

    core_range = ttnn.CoreRangeSet([
        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gw - 1, gh - 1))
    ])

    def _cb(i, fmt, page, pages):
        return ttnn.CBDescriptor(
            total_size=page * pages, core_ranges=core_range,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=page)
            ],
        )

    cbs = [
        _cb(0, ttnn.bfloat16, 2048, nm * nk),
        _cb(1, ttnn.bfloat16, 2048, nk),
        _cb(2, ttnn.uint32, 512, nk),
        _cb(3, ttnn.uint32, 1024, 1),
        _cb(16, ttnn.bfloat16, 2048, 1),
    ]

    rt_r, rt_c, rt_w = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for n in range(nt):
        # Logical-coordinate indexing, not a linear index.
        x, y = n % gw, n // gw
        rt_r[x][y] = [wd[n].buffer_address(), actd.buffer_address(),
                      lutd.buffer_address(), out.buffer_address(),
                      nk, nm * nk, n * nm, nm]
        rt_c[x][y] = [nk, nm]
        rt_w[x][y] = [out.buffer_address(), n * nm, nm]

    ks = [
        ttnn.KernelDescriptor(kernel_source=FUSED_GRID_READER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=core_range,
            compile_time_args=[], runtime_args=rt_r, config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=FUSED_GRID_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=core_range,
            compile_time_args=[], runtime_args=rt_c,
            config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True)),
        ttnn.KernelDescriptor(kernel_source=FUSED_GRID_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=core_range,
            compile_time_args=[], runtime_args=rt_w, config=ttnn.WriterConfigDescriptor()),
    ]
    prog = ttnn.ProgramDescriptor(kernels=ks, semaphores=[], cbs=cbs)
    io = [actd, lutd] + wd + [out]
    ttnn.generic_op(io, prog)
    ttnn.synchronize_device(device)

    pages = ttnn.to_torch(out).reshape(nt, nm, 32, 32)
    return pages.permute(1, 2, 0, 3).reshape(nm * 32, nt * 32).to(torch.float32)


__all__ = [
    "FUSED_COMPUTE_COPY",
    "FUSED_GEMM_COMPUTE",
    "FUSED_GEMM_LUT_READER",
    "FUSED_GEMM_M_COMPUTE",
    "FUSED_GEMM_M_READER",
    "FUSED_GEMM_M_WRITER",
    "FUSED_GRID_COMPUTE",
    "FUSED_GRID_READER",
    "FUSED_GRID_WRITER",
    "FUSED_GEMM_READER",
    "FUSED_MATMUL_COMPUTE",
    "FUSED_MATMUL_READER",
    "FUSED_DEQUANT_READER",
    "FUSED_TILE_WRITER",
    "INT4_DEQUANT_KERNEL_SRC",
    "INT4_DEQUANT_TILED_SRC",
    "bf16_truncate",
    "build_dequant_lut",
    "build_dequant_pair_lut",
    "clear_grid_weight_cache",
    "dequantize_on_device",
    "dequantize_tiled_on_device",
    "pack_codes_to_words",
    "permute_to_tile_order",
    "run_fused_gemm",
    "run_fused_gemm_lut",
    "run_fused_gemm_grid",
    "run_fused_gemm_m",
    "run_fused_matmul",
    "run_fused_tile",
    "tile_order_indices",
    "reference_dequant",
    "verify_on_device",
]


# --- fused data path -------------------------------------------------------
#
# The measured problem with the standalone dequant kernel is that it still writes
# a dense BF16 tile to DRAM: 80.98 ms for one 768x768 layer on a single core,
# versus 1.27 ms for the TTNN op chain. Removing the DRAM round trip is the point
# of the fused design, so the dequant writes into an L1 circular buffer that the
# compute stage consumes directly.
#
# Layout lesson: a 32x32 tile is four 16x16 faces (face0..face3, each row-major),
# so codes must be permuted into that memory order before they are packed into
# words. Writing them linearly leaves every value in the wrong logical position:
# measured as `exact=False, maxerr=0.2588` before the permutation, bit-exact
# after.

FUSED_DEQUANT_READER = r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    uint32_t in_addr    = get_arg_val<uint32_t>(0);
    uint32_t nwords     = get_arg_val<uint32_t>(1);
    uint32_t scale_bits = get_arg_val<uint32_t>(2);
    uint32_t zero       = get_arg_val<uint32_t>(3);

    float scale;
    __builtin_memcpy(&scale, &scale_bits, 4);
    float fzero = (float)zero;

    constexpr uint32_t WORDS_CB = 1;
    constexpr uint32_t TILE_CB  = 0;

    cb_reserve_back(WORDS_CB, 1);
    uint32_t wl1 = get_write_ptr(WORDS_CB);
    InterleavedAddrGen<true> gen = {.bank_base_address = in_addr, .page_size = nwords * 4u};
    noc_async_read(gen.get_noc_addr(0), wl1, nwords * 4u);
    noc_async_read_barrier();
    cb_push_back(WORDS_CB, 1);
    cb_wait_front(WORDS_CB, 1);

    cb_reserve_back(TILE_CB, 1);
    uint32_t tl1 = get_write_ptr(TILE_CB);
    volatile tt_l1_ptr uint32_t* words = (volatile tt_l1_ptr uint32_t*)wl1;
    volatile tt_l1_ptr uint16_t* out   = (volatile tt_l1_ptr uint16_t*)tl1;
    for (uint32_t i = 0; i < nwords; ++i) {
        uint32_t word = words[i];
        for (uint32_t lane = 0; lane < 8; ++lane) {
            uint32_t code = (word >> (4u * lane)) & 0xFu;
            float v = scale * ((float)code - fzero);
            uint32_t bits;
            __builtin_memcpy(&bits, &v, 4);
            out[i * 8u + lane] = (uint16_t)(bits >> 16);
        }
    }
    cb_push_back(TILE_CB, 1);
}
"""

FUSED_COMPUTE_COPY = r"""
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"

void kernel_main() {
    constexpr uint32_t in_cb = 0;
    constexpr uint32_t out_cb = 16;
    uint32_t n = get_arg_val<uint32_t>(0);
    init_sfpu(in_cb, out_cb);
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(in_cb, 1);
        cb_reserve_back(out_cb, 1);
        tile_regs_acquire();
        copy_tile_to_dst_init_short(in_cb);
        copy_tile(in_cb, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_cb);
        tile_regs_release();
        cb_pop_front(in_cb, 1);
        cb_push_back(out_cb, 1);
    }
}
"""

FUSED_TILE_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t bytes    = get_arg_val<uint32_t>(1);
    constexpr uint32_t CB = 16;
    cb_wait_front(CB, 1);
    uint32_t l1 = get_read_ptr(CB);
    InterleavedAddrGen<true> gen = {.bank_base_address = out_addr, .page_size = bytes};
    noc_async_write(l1, gen.get_noc_addr(0), bytes);
    noc_async_write_barrier();
    cb_pop_front(CB, 1);
}
"""


def tile_order_indices(tile: int = 32, face: int = 16) -> torch.Tensor:
    """Return the memory-order permutation of a square tile.

    ``order[k]`` is the logical row-major index stored at memory position ``k``,
    given that the hardware lays a tile out as consecutive row-major faces.
    """

    if tile % face:
        raise ValueError(f"tile={tile} must be a multiple of face={face}")
    n = tile // face
    order = []
    for f in range(n * n):
        fr, fc = (f // n) * face, (f % n) * face
        for i in range(face):
            for j in range(face):
                order.append((fr + i) * tile + (fc + j))
    return torch.tensor(order, dtype=torch.long)


def permute_to_tile_order(codes: torch.Tensor, tile: int = 32) -> torch.Tensor:
    """Reorder logical row-major values into the tile's memory order."""

    return codes.reshape(-1)[tile_order_indices(tile)]


def run_fused_tile(
    codes: torch.Tensor, scale: float, zero: int, *, device
) -> torch.Tensor:
    """Dequantize one 32x32 tile through the fused path and return it logically.

    The packed words are produced from tile-ordered codes, the BRISC expands them
    into an L1 circular buffer, the compute stage consumes that buffer, and only
    the finished tile is written to DRAM.
    """

    from .ttnn_runtime import import_ttnn

    ttnn = import_ttnn()
    if codes.numel() != 1024:
        raise ValueError(f"expects exactly one 32x32 tile (1024 codes), got {codes.numel()}")
    tile_words = 128

    words = pack_codes_to_words(permute_to_tile_order(codes))
    inp = ttnn.to_device(
        ttnn.from_torch(words.reshape(1, tile_words), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
        device,
    )
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    f_words = ttnn.CBFormatDescriptor(buffer_index=1, data_format=ttnn.uint32, page_size=tile_words * 4)
    f_tile = ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048)
    f_out = ttnn.CBFormatDescriptor(buffer_index=16, data_format=ttnn.bfloat16, page_size=2048)
    cbs = [
        ttnn.CBDescriptor(total_size=2048, core_ranges=grid, format_descriptors=[f_tile]),
        ttnn.CBDescriptor(total_size=tile_words * 4, core_ranges=grid, format_descriptors=[f_words]),
        ttnn.CBDescriptor(total_size=2048, core_ranges=grid, format_descriptors=[f_out]),
    ]

    scale_bits = struct.unpack("<I", struct.pack("<f", float(scale)))[0]
    rt_r = ttnn.RuntimeArgs()
    rt_r[0][0] = [inp.buffer_address(), tile_words, scale_bits, int(zero)]
    rt_c = ttnn.RuntimeArgs()
    rt_c[0][0] = [1]
    rt_w = ttnn.RuntimeArgs()
    rt_w[0][0] = [out.buffer_address(), 2048]

    ks = [
        ttnn.KernelDescriptor(kernel_source=FUSED_DEQUANT_READER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=grid,
            compile_time_args=[], runtime_args=rt_r, config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=FUSED_COMPUTE_COPY,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=grid,
            compile_time_args=[], runtime_args=rt_c, config=ttnn.ComputeConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=FUSED_TILE_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=grid,
            compile_time_args=[], runtime_args=rt_w, config=ttnn.WriterConfigDescriptor()),
    ]
    ttnn.generic_op([inp, out], ttnn.ProgramDescriptor(kernels=ks, semaphores=[], cbs=cbs))
    ttnn.synchronize_device(device)
    return ttnn.to_torch(out).reshape(-1).to(torch.float32)


FUSED_MATMUL_COMPUTE = r"""
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_hw_startup.h"

void kernel_main() {
    constexpr uint32_t ACT_CB = 0;
    constexpr uint32_t W_CB   = 1;
    constexpr uint32_t OUT_CB = 16;

    // Matmul maps in0 -> SrcB and in1 -> SrcA, so passing (W, ACT) yields W x ACT.
    compute_kernel_hw_startup(W_CB, ACT_CB, OUT_CB);

    cb_wait_front(ACT_CB, 1);
    cb_wait_front(W_CB, 1);
    cb_reserve_back(OUT_CB, 1);

    tile_regs_acquire();
    matmul_init(W_CB, ACT_CB, 0);
    matmul_tiles(W_CB, ACT_CB, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, OUT_CB);
    tile_regs_release();

    cb_pop_front(ACT_CB, 1);
    cb_pop_front(W_CB, 1);
    cb_push_back(OUT_CB, 1);
}
"""

#: Reader used by the matmul variant: activation tile plus dequantized weight tile.
FUSED_MATMUL_READER = r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    uint32_t in_addr    = get_arg_val<uint32_t>(0);
    uint32_t act_addr   = get_arg_val<uint32_t>(1);
    uint32_t nwords     = get_arg_val<uint32_t>(2);
    uint32_t scale_bits = get_arg_val<uint32_t>(3);
    uint32_t zero       = get_arg_val<uint32_t>(4);

    float scale;
    __builtin_memcpy(&scale, &scale_bits, 4);
    float fzero = (float)zero;

    constexpr uint32_t ACT_CB = 0;
    constexpr uint32_t W_CB   = 1;
    constexpr uint32_t RAW_CB = 2;

    cb_reserve_back(ACT_CB, 1);
    uint32_t a_l1 = get_write_ptr(ACT_CB);
    InterleavedAddrGen<true> ag = {.bank_base_address = act_addr, .page_size = 2048};
    noc_async_read(ag.get_noc_addr(0), a_l1, 2048);
    noc_async_read_barrier();
    cb_push_back(ACT_CB, 1);

    cb_reserve_back(RAW_CB, 1);
    uint32_t wl1 = get_write_ptr(RAW_CB);
    InterleavedAddrGen<true> wg = {.bank_base_address = in_addr, .page_size = nwords * 4u};
    noc_async_read(wg.get_noc_addr(0), wl1, nwords * 4u);
    noc_async_read_barrier();
    cb_push_back(RAW_CB, 1);
    cb_wait_front(RAW_CB, 1);

    cb_reserve_back(W_CB, 1);
    uint32_t tl1 = get_write_ptr(W_CB);
    volatile tt_l1_ptr uint32_t* words = (volatile tt_l1_ptr uint32_t*)wl1;
    volatile tt_l1_ptr uint16_t* out   = (volatile tt_l1_ptr uint16_t*)tl1;
    for (uint32_t i = 0; i < nwords; ++i) {
        uint32_t word = words[i];
        for (uint32_t lane = 0; lane < 8; ++lane) {
            uint32_t code = (word >> (4u * lane)) & 0xFu;
            float v = scale * ((float)code - fzero);
            uint32_t bits;
            __builtin_memcpy(&bits, &v, 4);
            out[i * 8u + lane] = (uint16_t)(bits >> 16);
        }
    }
    cb_push_back(W_CB, 1);
}
"""


def run_fused_matmul(
    act: torch.Tensor, codes: torch.Tensor, scale: float, zero: int, *, device
) -> torch.Tensor:
    """One fused int4 GEMM tile: dequant in L1, then the FPU multiplies.

    Returns ``W x ACT`` (matmul maps in0 to SrcB and in1 to SrcA, so the weight is
    passed first). The packed checkpoint never expands into DRAM: the dequantized
    tile lives only in an L1 circular buffer that the matmul consumes.

    Measured against a torch reference on device: ``norm_l1`` 0.0052 and
    ``maxerr`` 0.0185 for BF16 accumulation; the transposed orientation is off by
    ``norm_l1`` 1.43, so this is the correct operand order.
    """

    from .ttnn_runtime import import_ttnn

    ttnn = import_ttnn()
    if act.numel() != 1024 or codes.numel() != 1024:
        raise ValueError("expects one 32x32 activation tile and one 32x32 code tile")
    tile_words = 128

    words = pack_codes_to_words(permute_to_tile_order(codes))
    inp = ttnn.to_device(
        ttnn.from_torch(words.reshape(1, tile_words), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
        device,
    )
    actd = ttnn.to_device(
        ttnn.from_torch(act.to(torch.bfloat16).reshape(1, 1, 32, 32), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT),
        device,
    )
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    def _cb(index, fmt, size):
        return ttnn.CBDescriptor(
            total_size=size, core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=size)],
        )

    cbs = [
        _cb(0, ttnn.bfloat16, 2048),
        _cb(1, ttnn.bfloat16, 2048),
        _cb(2, ttnn.uint32, tile_words * 4),
        _cb(16, ttnn.bfloat16, 2048),
    ]

    scale_bits = struct.unpack("<I", struct.pack("<f", float(scale)))[0]
    rt_r = ttnn.RuntimeArgs()
    rt_r[0][0] = [inp.buffer_address(), actd.buffer_address(), tile_words, scale_bits, int(zero)]
    rt_c = ttnn.RuntimeArgs()
    rt_c[0][0] = [0]
    rt_w = ttnn.RuntimeArgs()
    rt_w[0][0] = [out.buffer_address(), 2048]

    ks = [
        ttnn.KernelDescriptor(kernel_source=FUSED_MATMUL_READER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=grid,
            compile_time_args=[], runtime_args=rt_r, config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=FUSED_MATMUL_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=grid,
            compile_time_args=[], runtime_args=rt_c, config=ttnn.ComputeConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=FUSED_TILE_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=grid,
            compile_time_args=[], runtime_args=rt_w, config=ttnn.WriterConfigDescriptor()),
    ]
    ttnn.generic_op([inp, actd, out], ttnn.ProgramDescriptor(kernels=ks, semaphores=[], cbs=cbs))
    ttnn.synchronize_device(device)
    return ttnn.to_torch(out).reshape(32, 32).to(torch.float32)


FUSED_GEMM_READER = r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    uint32_t w_addr     = get_arg_val<uint32_t>(0);
    uint32_t act_addr   = get_arg_val<uint32_t>(1);
    uint32_t nk         = get_arg_val<uint32_t>(2);
    uint32_t scale_bits = get_arg_val<uint32_t>(3);
    uint32_t zero       = get_arg_val<uint32_t>(4);

    float scale;
    __builtin_memcpy(&scale, &scale_bits, 4);
    float fzero = (float)zero;

    constexpr uint32_t ACT_CB = 0;
    constexpr uint32_t W_CB   = 1;
    constexpr uint32_t RAW_CB = 2;
    constexpr uint32_t WORDS_PER_TILE = 128;
    constexpr uint32_t WORD_BYTES = WORDS_PER_TILE * 4u;

    // Staged like the LUT reader: all tiles in flight, one barrier. This variant
    // keeps the software-float expansion so the two can be compared honestly.
    cb_reserve_back(ACT_CB, nk);
    uint32_t a_base = get_write_ptr(ACT_CB);
    InterleavedAddrGen<true> ag = {.bank_base_address = act_addr, .page_size = 2048};
    cb_reserve_back(RAW_CB, nk);
    uint32_t r_base = get_write_ptr(RAW_CB);
    InterleavedAddrGen<true> wg = {.bank_base_address = w_addr, .page_size = WORD_BYTES};

    for (uint32_t k = 0; k < nk; ++k) {
        noc_async_read(ag.get_noc_addr(k), a_base + k * 2048u, 2048);
        noc_async_read(wg.get_noc_addr(k), r_base + k * WORD_BYTES, WORD_BYTES);
    }
    noc_async_read_barrier();

    cb_reserve_back(W_CB, nk);
    uint32_t w_base = get_write_ptr(W_CB);
    for (uint32_t k = 0; k < nk; ++k) {
        volatile tt_l1_ptr uint32_t* words =
            (volatile tt_l1_ptr uint32_t*)(r_base + k * WORD_BYTES);
        volatile tt_l1_ptr uint16_t* out =
            (volatile tt_l1_ptr uint16_t*)(w_base + k * 2048u);
        for (uint32_t i = 0; i < WORDS_PER_TILE; ++i) {
            uint32_t word = words[i];
            for (uint32_t lane = 0; lane < 8; ++lane) {
                uint32_t code = (word >> (4u * lane)) & 0xFu;
                float v = scale * ((float)code - fzero);
                uint32_t bits;
                __builtin_memcpy(&bits, &v, 4);
                out[i * 8u + lane] = (uint16_t)(bits >> 16);
            }
        }
    }
    cb_push_back(ACT_CB, nk);
    cb_push_back(W_CB, nk);
}
"""

FUSED_GEMM_COMPUTE = r"""
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_hw_startup.h"

void kernel_main() {
    uint32_t nk = get_arg_val<uint32_t>(0);
    constexpr uint32_t ACT_CB = 0;
    constexpr uint32_t W_CB   = 1;
    constexpr uint32_t OUT_CB = 16;

    compute_kernel_hw_startup<SrcOrder::Reverse>(ACT_CB, W_CB, OUT_CB);
    matmul_init(ACT_CB, W_CB);

    // tile_regs_acquire() also zeroes the destination, so the K loop accumulates
    // from zero without a separate clear.
    tile_regs_acquire();
    for (uint32_t k = 0; k < nk; ++k) {
        cb_wait_front(ACT_CB, nk);
        cb_wait_front(W_CB, nk);
        matmul_tiles(ACT_CB, W_CB, k, k, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(OUT_CB, 1);
    pack_tile(0, OUT_CB);
    cb_push_back(OUT_CB, 1);
    tile_regs_release();
}
"""


def run_fused_gemm(
    act: torch.Tensor,
    codes: torch.Tensor,
    scale: float,
    zero: int,
    *,
    device,
) -> torch.Tensor:
    """Fused int4 GEMM for one output tile with the full K reduction on chip.

    ``act`` is ``[32, K]`` and ``codes`` is ``K*32`` 4-bit codes laid out as
    ``K/32`` weight tiles of ``32x32``. Returns ``[32, 32]`` equal to
    ``act @ dequant(codes)``.

    Verified on device: ``norm_l1`` 0.0047 at K=768, 0.0046 at K=256.

    Performance, measured not extrapolated: **~3.4 ms per output tile at K=768**.
    A dispatch of a trivial kernel costs 0.019 ms, so this is kernel time, and it
    is dominated by dequantization running on BRISC, which is an RV32 core with no
    FPU -- every `code -> float` conversion is software floating point. TTNN does
    a whole 768x768 layer in ~1.33 ms, so this is far slower until the expansion
    moves to the SFPU in the compute path, which is what the design intends.
    """

    from .ttnn_runtime import import_ttnn

    ttnn = import_ttnn()
    nk = codes.numel() // 1024
    if codes.numel() != nk * 1024:
        raise ValueError("codes must be a whole number of 32x32 tiles")
    if act.shape != (32, nk * 32):
        raise ValueError(f"act must be [32, {nk * 32}], got {tuple(act.shape)}")

    words = torch.zeros(nk * 128, dtype=torch.int64)
    tile_order = tile_order_indices()
    for k in range(nk):
        mem = codes[k * 1024:(k + 1) * 1024][tile_order]
        for lane in range(8):
            words[k * 128:(k + 1) * 128] |= (mem[lane::8] & 0xF) << (4 * lane)

    inp = ttnn.to_device(
        ttnn.from_torch(words.to(torch.int32).reshape(nk, 128), dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT),
        device,
    )
    actd = ttnn.to_device(
        ttnn.from_torch(act.to(torch.bfloat16).reshape(1, 32, nk * 32), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT),
        device,
    )
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    def _cb(index, fmt, page_size, pages=1):
        return ttnn.CBDescriptor(
            total_size=page_size * pages, core_ranges=grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page_size)
            ],
        )

    cbs = [
        _cb(0, ttnn.bfloat16, 2048, nk),
        _cb(1, ttnn.bfloat16, 2048, nk),
        _cb(2, ttnn.uint32, 512, nk),
        _cb(16, ttnn.bfloat16, 2048),
    ]

    scale_bits = struct.unpack("<I", struct.pack("<f", float(scale)))[0]
    rt_r = ttnn.RuntimeArgs()
    rt_r[0][0] = [inp.buffer_address(), actd.buffer_address(), nk, scale_bits, int(zero)]
    rt_c = ttnn.RuntimeArgs()
    rt_c[0][0] = [nk]
    rt_w = ttnn.RuntimeArgs()
    rt_w[0][0] = [out.buffer_address(), 2048]

    ks = [
        ttnn.KernelDescriptor(kernel_source=FUSED_GEMM_READER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=grid,
            compile_time_args=[], runtime_args=rt_r, config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=FUSED_GEMM_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=grid,
            compile_time_args=[], runtime_args=rt_c,
            config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True)),
        ttnn.KernelDescriptor(kernel_source=FUSED_TILE_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=grid,
            compile_time_args=[], runtime_args=rt_w, config=ttnn.WriterConfigDescriptor()),
    ]
    ttnn.generic_op([inp, actd, out], ttnn.ProgramDescriptor(kernels=ks, semaphores=[], cbs=cbs))
    ttnn.synchronize_device(device)
    return ttnn.to_torch(out).reshape(32, 32).to(torch.float32)


FUSED_GEMM_LUT_READER = r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    uint32_t w_addr   = get_arg_val<uint32_t>(0);
    uint32_t act_addr = get_arg_val<uint32_t>(1);
    uint32_t lut_addr = get_arg_val<uint32_t>(2);
    uint32_t nk       = get_arg_val<uint32_t>(3);

    constexpr uint32_t ACT_CB = 0;
    constexpr uint32_t W_CB   = 1;
    constexpr uint32_t RAW_CB = 2;
    constexpr uint32_t LUT_CB = 3;
    constexpr uint32_t WORDS_PER_TILE = 128;
    constexpr uint32_t WORD_BYTES = WORDS_PER_TILE * 4u;

    cb_reserve_back(LUT_CB, 1);
    uint32_t lut_l1 = get_write_ptr(LUT_CB);
    InterleavedAddrGen<true> lg = {.bank_base_address = lut_addr, .page_size = 32};
    noc_async_read(lg.get_noc_addr(0), lut_l1, 32);

    // Every activation tile and every packed word, issued back to back. Reads are
    // async, so one barrier covers the whole batch instead of one per K tile.
    // (A single noc_async_read spanning many pages is invalid; issue per page.)
    cb_reserve_back(ACT_CB, nk);
    uint32_t a_base = get_write_ptr(ACT_CB);
    InterleavedAddrGen<true> ag = {.bank_base_address = act_addr, .page_size = 2048};
    cb_reserve_back(RAW_CB, nk);
    uint32_t r_base = get_write_ptr(RAW_CB);
    InterleavedAddrGen<true> wg = {.bank_base_address = w_addr, .page_size = WORD_BYTES};

    for (uint32_t k = 0; k < nk; ++k) {
        noc_async_read(ag.get_noc_addr(k), a_base + k * 2048u, 2048);
        noc_async_read(wg.get_noc_addr(k), r_base + k * WORD_BYTES, WORD_BYTES);
    }
    noc_async_read_barrier();

    // Expand every weight tile into its own page via the 16-entry table.
    cb_reserve_back(W_CB, nk);
    uint32_t w_base = get_write_ptr(W_CB);
    volatile tt_l1_ptr uint16_t* lut = (volatile tt_l1_ptr uint16_t*)lut_l1;
    for (uint32_t k = 0; k < nk; ++k) {
        volatile tt_l1_ptr uint32_t* words =
            (volatile tt_l1_ptr uint32_t*)(r_base + k * WORD_BYTES);
        volatile tt_l1_ptr uint16_t* out =
            (volatile tt_l1_ptr uint16_t*)(w_base + k * 2048u);
        for (uint32_t i = 0; i < WORDS_PER_TILE; ++i) {
            uint32_t word = words[i];
            out[i * 8u + 0] = lut[(word      ) & 0xFu];
            out[i * 8u + 1] = lut[(word >>  4) & 0xFu];
            out[i * 8u + 2] = lut[(word >>  8) & 0xFu];
            out[i * 8u + 3] = lut[(word >> 12) & 0xFu];
            out[i * 8u + 4] = lut[(word >> 16) & 0xFu];
            out[i * 8u + 5] = lut[(word >> 20) & 0xFu];
            out[i * 8u + 6] = lut[(word >> 24) & 0xFu];
            out[i * 8u + 7] = lut[(word >> 28) & 0xFu];
        }
    }
    cb_push_back(ACT_CB, nk);
    cb_push_back(W_CB, nk);
}
"""


def build_dequant_pair_lut(scale: float, zero: int) -> torch.Tensor:
    """Pack all 256 two-code results into 256 uint32 words (1024 bytes).

    A packed GPTQ word holds eight codes in eight nibbles, i.e. four *bytes* of
    two codes each. Indexing a 256-entry table by a byte decodes both codes at
    once and emits a single 32-bit store, halving the store count versus the
    16-entry table. Measured on a 64-core layer: 0.3463 ms -> 0.2379 ms (1.46x).

    Values come from the correctly rounded BF16 of ``scale * (code - zero)``, so a
    lookup is also more accurate than truncating soft-float.
    """

    values = (float(scale) * (torch.arange(16, dtype=torch.float32) - float(zero))).to(torch.bfloat16)
    bits = values.view(torch.int16).to(torch.int64) & 0xFFFF
    pair = torch.zeros(256, dtype=torch.int64)
    for low in range(16):
        for high in range(16):
            # Low nibble first, matching GPTQ's checkpoint convention.
            pair[low | (high << 4)] = bits[low] | (bits[high] << 16)
    return pair.to(torch.int32).reshape(1, 256)


def build_dequant_lut(scale: float, zero: int) -> torch.Tensor:
    """Pack the 16 possible BF16 results into 8 uint32 words, two per word.

    ``scale * (code - zero)`` takes only 16 distinct values, so the kernel can
    index a table instead of doing software floating point on BRISC (which has no
    FPU). The table is built from the exact BF16 rounding, so the lookup is also
    more accurate than the previous truncating soft-float path.
    """

    values = (float(scale) * (torch.arange(16, dtype=torch.float32) - float(zero))).to(torch.bfloat16)
    bits = values.view(torch.int16).to(torch.int64) & 0xFFFF
    return (bits[0::2] | (bits[1::2] << 16)).to(torch.int32).reshape(1, 8)


def run_fused_gemm_lut(
    act: torch.Tensor, codes: torch.Tensor, scale: float, zero: int, *, device
) -> torch.Tensor:
    """Fused int4 GEMM with a lookup-table dequant on the data-movement core.

    Same fused structure as :func:`run_fused_gemm` -- packed words expand into an
    L1 circular buffer and never reach DRAM -- but the expansion is a table
    lookup. Measured at K=768: **0.259 ms per output tile** versus 3.395 ms for
    the soft-float version, a 13x improvement, with ``norm_l1`` 0.00145 against
    0.00478 (the table holds correctly rounded BF16 rather than truncated values).

    Not yet a win over TTNN: a 768x768 layer is 576 output tiles, so even perfect
    spread over the 11x10 grid lands near TTNN's ~1.33 ms for the whole layer.
    Beating it needs the per-K DRAM barrier pipelined out.
    """

    from .ttnn_runtime import import_ttnn

    ttnn = import_ttnn()
    nk = codes.numel() // 1024
    if codes.numel() != nk * 1024:
        raise ValueError("codes must be a whole number of 32x32 tiles")
    if act.shape != (32, nk * 32):
        raise ValueError(f"act must be [32, {nk * 32}], got {tuple(act.shape)}")

    words = torch.zeros(nk * 128, dtype=torch.int64)
    tile_order = tile_order_indices()
    for k in range(nk):
        mem = codes[k * 1024:(k + 1) * 1024][tile_order]
        for lane in range(8):
            words[k * 128:(k + 1) * 128] |= (mem[lane::8] & 0xF) << (4 * lane)

    inp = ttnn.to_device(
        ttnn.from_torch(words.to(torch.int32).reshape(nk, 128), dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT),
        device,
    )
    actd = ttnn.to_device(
        ttnn.from_torch(act.to(torch.bfloat16).reshape(1, 32, nk * 32), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT),
        device,
    )
    lutd = ttnn.to_device(
        ttnn.from_torch(build_dequant_lut(scale, zero), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
        device,
    )
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    def _cb(index, fmt, page_size, pages=1):
        return ttnn.CBDescriptor(
            total_size=page_size * pages, core_ranges=grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page_size)
            ],
        )

    # All K activation and weight tiles are staged at once, so both buffers hold
    # nk pages and the kernel needs a single barrier rather than one per tile.
    cbs = [
        _cb(0, ttnn.bfloat16, 2048, nk),
        _cb(1, ttnn.bfloat16, 2048, nk),
        _cb(2, ttnn.uint32, 512, nk),
        _cb(3, ttnn.uint32, 32),
        _cb(16, ttnn.bfloat16, 2048),
    ]

    rt_r = ttnn.RuntimeArgs()
    rt_r[0][0] = [inp.buffer_address(), actd.buffer_address(), lutd.buffer_address(), nk]
    rt_c = ttnn.RuntimeArgs()
    rt_c[0][0] = [nk]
    rt_w = ttnn.RuntimeArgs()
    rt_w[0][0] = [out.buffer_address(), 2048]

    ks = [
        ttnn.KernelDescriptor(kernel_source=FUSED_GEMM_LUT_READER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=grid,
            compile_time_args=[], runtime_args=rt_r, config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=FUSED_GEMM_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=grid,
            compile_time_args=[], runtime_args=rt_c,
            config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True)),
        ttnn.KernelDescriptor(kernel_source=FUSED_TILE_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=grid,
            compile_time_args=[], runtime_args=rt_w, config=ttnn.WriterConfigDescriptor()),
    ]
    ttnn.generic_op([inp, actd, lutd, out], ttnn.ProgramDescriptor(kernels=ks, semaphores=[], cbs=cbs))
    ttnn.synchronize_device(device)
    return ttnn.to_torch(out).reshape(32, 32).to(torch.float32)


# --- multi-M-tile fused GEMM ----------------------------------------------
#
# The dequant cost is per *weight tile*, not per output row, so it amortises over
# M. Measured at K=768 with one N tile (latency is flat while M grows, because the
# same 24 dequantized weight tiles serve every M tile):
#
#   M=32    0.2405 ms   (0.2405 ms per 32 rows)
#   M=64    0.2417 ms   (0.1208 ms per 32 rows)
#   M=128   0.2443 ms   (0.0611 ms per 32 rows)
#   M=256   0.2535 ms   (0.0317 ms per 32 rows)  -> 7.6x
#
# This is why the fused path is worth building for prefill/batched shapes and is
# only at parity for decode.

FUSED_GEMM_M_READER = r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    uint32_t w_addr   = get_arg_val<uint32_t>(0);
    uint32_t act_addr = get_arg_val<uint32_t>(1);
    uint32_t lut_addr = get_arg_val<uint32_t>(2);
    uint32_t nk       = get_arg_val<uint32_t>(3);
    uint32_t nact     = get_arg_val<uint32_t>(4);

    constexpr uint32_t ACT_CB = 0;
    constexpr uint32_t W_CB   = 1;
    constexpr uint32_t RAW_CB = 2;
    constexpr uint32_t LUT_CB = 3;
    constexpr uint32_t WORDS_PER_TILE = 128;
    constexpr uint32_t WORD_BYTES = WORDS_PER_TILE * 4u;

    cb_reserve_back(LUT_CB, 1);
    uint32_t lut_l1 = get_write_ptr(LUT_CB);
    InterleavedAddrGen<true> lg = {.bank_base_address = lut_addr, .page_size = 32};
    noc_async_read(lg.get_noc_addr(0), lut_l1, 32);

    cb_reserve_back(ACT_CB, nact);
    uint32_t a_base = get_write_ptr(ACT_CB);
    InterleavedAddrGen<true> ag = {.bank_base_address = act_addr, .page_size = 2048};
    cb_reserve_back(RAW_CB, nk);
    uint32_t r_base = get_write_ptr(RAW_CB);
    InterleavedAddrGen<true> wg = {.bank_base_address = w_addr, .page_size = WORD_BYTES};

    for (uint32_t t = 0; t < nact; ++t) {
        noc_async_read(ag.get_noc_addr(t), a_base + t * 2048u, 2048);
    }
    for (uint32_t k = 0; k < nk; ++k) {
        noc_async_read(wg.get_noc_addr(k), r_base + k * WORD_BYTES, WORD_BYTES);
    }
    noc_async_read_barrier();

    // Each weight tile is dequantized exactly once, however many M tiles read it.
    cb_reserve_back(W_CB, nk);
    uint32_t w_base = get_write_ptr(W_CB);
    volatile tt_l1_ptr uint16_t* lut = (volatile tt_l1_ptr uint16_t*)lut_l1;
    for (uint32_t k = 0; k < nk; ++k) {
        volatile tt_l1_ptr uint32_t* words =
            (volatile tt_l1_ptr uint32_t*)(r_base + k * WORD_BYTES);
        volatile tt_l1_ptr uint16_t* out =
            (volatile tt_l1_ptr uint16_t*)(w_base + k * 2048u);
        for (uint32_t i = 0; i < WORDS_PER_TILE; ++i) {
            uint32_t word = words[i];
            out[i * 8u + 0] = lut[(word      ) & 0xFu];
            out[i * 8u + 1] = lut[(word >>  4) & 0xFu];
            out[i * 8u + 2] = lut[(word >>  8) & 0xFu];
            out[i * 8u + 3] = lut[(word >> 12) & 0xFu];
            out[i * 8u + 4] = lut[(word >> 16) & 0xFu];
            out[i * 8u + 5] = lut[(word >> 20) & 0xFu];
            out[i * 8u + 6] = lut[(word >> 24) & 0xFu];
            out[i * 8u + 7] = lut[(word >> 28) & 0xFu];
        }
    }
    cb_push_back(ACT_CB, nact);
    cb_push_back(W_CB, nk);
}
"""

FUSED_GEMM_M_COMPUTE = r"""
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/compute_kernel_hw_startup.h"

void kernel_main() {
    uint32_t nk = get_arg_val<uint32_t>(0);
    uint32_t nm = get_arg_val<uint32_t>(1);
    constexpr uint32_t ACT_CB = 0;
    constexpr uint32_t W_CB   = 1;
    constexpr uint32_t OUT_CB = 16;

    compute_kernel_hw_startup<SrcOrder::Reverse>(ACT_CB, W_CB, OUT_CB);
    matmul_init(ACT_CB, W_CB);

    cb_wait_front(ACT_CB, nm * nk);
    cb_wait_front(W_CB, nk);

    for (uint32_t mi = 0; mi < nm; ++mi) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < nk; ++k) {
            matmul_tiles(ACT_CB, W_CB, mi * nk + k, k, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(OUT_CB, 1);
        pack_tile(0, OUT_CB);
        cb_push_back(OUT_CB, 1);
        tile_regs_release();
    }
}
"""

FUSED_GEMM_M_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t nm       = get_arg_val<uint32_t>(1);
    constexpr uint32_t CB = 16;
    InterleavedAddrGen<true> g = {.bank_base_address = out_addr, .page_size = 2048};
    for (uint32_t i = 0; i < nm; ++i) {
        cb_wait_front(CB, 1);
        uint32_t l1 = get_read_ptr(CB);
        noc_async_write(l1, g.get_noc_addr(i), 2048);
        noc_async_write_barrier();
        cb_pop_front(CB, 1);
    }
}
"""


def run_fused_gemm_m(
    act: torch.Tensor, codes: torch.Tensor, scale: float, zero: int, *, device
) -> torch.Tensor:
    """Fused int4 GEMM over ``M = nm*32`` rows, one N tile, whole K reduction.

    ``act`` is ``[nm*32, K]`` and ``codes`` is ``K/32`` weight tiles of ``32x32``
    (one N tile). Returns ``[nm*32, 32]``.

    The weight tiles are dequantized once and reused by every M tile, which is the
    entire point: measured latency is flat in M (0.2405 ms at M=32 through
    0.2535 ms at M=256), so per-row cost falls 7.6x by M=256. Verified
    ``norm_l1`` 0.0014 at M=64/128/256.
    """

    from .ttnn_runtime import import_ttnn

    ttnn = import_ttnn()
    nk = codes.numel() // 1024
    if codes.numel() != nk * 1024:
        raise ValueError("codes must be a whole number of 32x32 tiles")
    if act.dim() != 2 or act.shape[1] != nk * 32:
        raise ValueError(f"act must be [M, {nk * 32}], got {tuple(act.shape)}")
    m = int(act.shape[0])
    if m % 32:
        raise ValueError(f"M={m} must be a multiple of the 32-row tile")
    nm = m // 32

    words = torch.zeros(nk * 128, dtype=torch.int64)
    tile_order = tile_order_indices()
    for k in range(nk):
        mem = codes[k * 1024:(k + 1) * 1024][tile_order]
        for lane in range(8):
            words[k * 128:(k + 1) * 128] |= (mem[lane::8] & 0xF) << (4 * lane)

    inp = ttnn.to_device(
        ttnn.from_torch(words.to(torch.int32).reshape(nk, 128), dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT),
        device,
    )
    actd = ttnn.to_device(
        ttnn.from_torch(act.to(torch.bfloat16).reshape(1, m, nk * 32), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT),
        device,
    )
    lutd = ttnn.to_device(
        ttnn.from_torch(build_dequant_lut(scale, zero), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
        device,
    )
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, m, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG,
    )

    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    def _cb(index, fmt, page_size, pages=1):
        return ttnn.CBDescriptor(
            total_size=page_size * pages, core_ranges=grid,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page_size)
            ],
        )

    cbs = [
        _cb(0, ttnn.bfloat16, 2048, nm * nk),
        _cb(1, ttnn.bfloat16, 2048, nk),
        _cb(2, ttnn.uint32, 512, nk),
        _cb(3, ttnn.uint32, 32),
        _cb(16, ttnn.bfloat16, 2048),
    ]

    rt_r = ttnn.RuntimeArgs()
    rt_r[0][0] = [inp.buffer_address(), actd.buffer_address(), lutd.buffer_address(), nk, nm * nk]
    rt_c = ttnn.RuntimeArgs()
    rt_c[0][0] = [nk, nm]
    rt_w = ttnn.RuntimeArgs()
    rt_w[0][0] = [out.buffer_address(), nm]

    ks = [
        ttnn.KernelDescriptor(kernel_source=FUSED_GEMM_M_READER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=grid,
            compile_time_args=[], runtime_args=rt_r, config=ttnn.ReaderConfigDescriptor()),
        ttnn.KernelDescriptor(kernel_source=FUSED_GEMM_M_COMPUTE,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=grid,
            compile_time_args=[], runtime_args=rt_c,
            config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True)),
        ttnn.KernelDescriptor(kernel_source=FUSED_GEMM_M_WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE, core_ranges=grid,
            compile_time_args=[], runtime_args=rt_w, config=ttnn.WriterConfigDescriptor()),
    ]
    ttnn.generic_op([inp, actd, lutd, out], ttnn.ProgramDescriptor(kernels=ks, semaphores=[], cbs=cbs))
    ttnn.synchronize_device(device)
    return ttnn.to_torch(out).reshape(m, 32).to(torch.float32)
