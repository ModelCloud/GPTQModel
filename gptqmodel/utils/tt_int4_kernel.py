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


def verify_on_device(nwords: int, scale: float, zero: int, *, device, seed: int = 5) -> Tuple[bool, float]:
    """Dispatch the kernel and compare against the torch reference."""

    torch.manual_seed(seed)
    codes = torch.randint(0, 16, (nwords * 8,), dtype=torch.int64)
    words = pack_codes_to_words(codes)
    got = dequantize_on_device(words, scale, zero, device=device)
    expected = reference_dequant(words, scale, zero)
    return bool(torch.equal(got, expected)), float((got - expected).abs().max())


__all__ = [
    "INT4_DEQUANT_KERNEL_SRC",
    "bf16_truncate",
    "dequantize_on_device",
    "pack_codes_to_words",
    "reference_dequant",
    "verify_on_device",
]
