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


__all__ = [
    "FUSED_COMPUTE_COPY",
    "FUSED_MATMUL_COMPUTE",
    "FUSED_MATMUL_READER",
    "FUSED_DEQUANT_READER",
    "FUSED_TILE_WRITER",
    "INT4_DEQUANT_KERNEL_SRC",
    "INT4_DEQUANT_TILED_SRC",
    "bf16_truncate",
    "dequantize_on_device",
    "dequantize_tiled_on_device",
    "pack_codes_to_words",
    "permute_to_tile_order",
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
