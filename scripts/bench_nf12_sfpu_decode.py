#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Measure 32-lane NF12 reconstruction and sparse escape application on Blackhole."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from bench_nf12_tenstorrent import _errors, _measure, _same
from safetensors import safe_open

from gptqmodel.utils.nf12_tenstorrent import prepare_nf12
from gptqmodel.utils.nf12_ttnn import TTNNNF12, _cb, _cores, _kernel
from gptqmodel.utils.nf12_ttnn_sfpu import _linear_operation, _pack_four, _to_tiles
from gptqmodel.utils.ttnn_runtime import import_ttnn


def _decode(ttnn, device, words, blocks, base, *, dense, metadata):
    shape = (blocks * 4, 32, 32)
    output = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.uint16, ttnn.TILE_LAYOUT,
                                           device, ttnn.DRAM_MEMORY_CONFIG)
    cores, assignments = _cores(ttnn, device, blocks)
    reader = _kernel(ttnn, "sfpu_decode_reader", ttnn.ReaderConfigDescriptor(), cores, assignments,
                     (int(dense),), [words], lambda start, end: [words.buffer_address(), start, end])
    compute = _kernel(ttnn, "sfpu_decode_compute", ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False),
                      cores, assignments, (int(dense), base), (), lambda start, end: [end - start])
    metadata_addresses = [tensor.buffer_address() for tensor in metadata]
    writer = _kernel(ttnn, "sfpu_decode_writer", ttnn.WriterConfigDescriptor(), cores, assignments,
                     (int(bool(metadata)),), [output, *metadata],
                     lambda start, end: [output.buffer_address(), start, end, *metadata_addresses])
    buffers = [_cb(ttnn, cores, 0, ttnn.uint16, 2048, 2 * (4 if dense else 3)),
               _cb(ttnn, cores, 16, ttnn.uint16, 2048, 8)]
    if metadata:
        buffers.append(_cb(ttnn, cores, 24, ttnn.uint32, 128))
    program = ttnn.ProgramDescriptor(kernels=[reader, compute, writer], semaphores=[], cbs=buffers)
    return ttnn.generic_op([words, *metadata, output], program)


def _weight(size, checkpoint, special):
    if checkpoint is not None:
        index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
        name = "model.llm.unembed.weight"
        with safe_open(checkpoint / index[name], framework="pt") as shard:
            weight = shard.get_slice(name)[:size]
        return 54, weight

    base = 56
    codes = torch.randint(4096, (size, size), dtype=torch.int32, generator=torch.Generator().manual_seed(12))
    codes.flatten()[:4096] = torch.arange(4096)
    bits = (codes & 0xff) | ((((codes >> 8) & 7) + base) << 8) | ((codes & 0x800) << 4)
    if special:
        # Place out-of-window zeros, subnormals, infinities and NaNs around tile boundaries.
        values = torch.tensor([0, 0x8000, 1, 0x8001, 0x7f80, 0xff80, 0x7f81, 0xffc1])
        positions = torch.arange(8) + 31 * size + 28
        bits.flatten()[positions] = values.to(torch.int32)
    return base, bits.to(torch.int16).view(torch.bfloat16)


def _exercise(ttnn, device, copy, sync, size, repeats, checkpoint, special):
    base, weight = _weight(size, checkpoint, special)
    bf16_bits = weight.view(torch.int16).int() & 0xffff
    tiles = _to_tiles(bf16_bits)
    blocks = tiles.shape[0] // 4
    expected = tiles.to(torch.int16)

    packed, patches = _pack_four(tiles, base)
    output_bytes = weight.nbytes
    for dense, data in ((True, tiles), (False, packed)):
        words = ttnn.from_torch(data, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT,
                                device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        metadata = tuple(ttnn.from_torch(patch, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                         device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                         for patch in patches if not dense)
        output, result = _measure(
            lambda words=words, dense=dense, metadata=metadata: _decode(
                ttnn, device, words, blocks, base, dense=dense, metadata=metadata,
            ), sync, repeats, output_bytes,
        )
        torch.testing.assert_close(copy(output).to(torch.int16), expected, atol=0, rtol=0)
        print(json.dumps({"kind": "bf16_tile_copy" if dense else "sfpu_decode", "shape": list(weight.shape),
                          "input_bytes": data.numel() * 2 + sum(m.volume() * 4 for m in metadata),
                          "output_bytes": output_bytes, **result}), flush=True)
        del output, words, metadata

    layer = TTNNNF12(device, *prepare_nf12(weight, "sfpu.measurement", exp_base=base))
    ids = layer.upload_ids(torch.arange(len(weight)))
    output, result = _measure(lambda: layer.gather_device(ids, len(weight)), sync, repeats, output_bytes)
    _same(copy(output), weight)
    print(json.dumps({"kind": "canonical_decode", "shape": list(weight.shape), "input_bytes": layer.resident_bytes,
                      "output_bytes": output_bytes, **result}), flush=True)


def _measure_linear_pair(calls, sync, repeats):
    for call in calls:
        call()
        sync()
    elapsed = [[], []]
    for repeat in range(repeats):
        for index in (repeat % 2, 1 - repeat % 2):
            start = time.perf_counter_ns()
            calls[index]()
            sync()
            elapsed[index].append((time.perf_counter_ns() - start) / 1000)
    return tuple((statistics.median(times), sorted(times)[math.ceil(0.95 * repeats) - 1]) for times in elapsed)


def _exercise_linear(ttnn, device, sync, size, repeats, checkpoint, prefetch_groups, m_sizes):
    base, weight = _weight(size, checkpoint, False)
    n, k = weight.shape
    if (k // 128) % prefetch_groups:
        raise ValueError("The number of prefetch groups must divide K/128")
    bits = weight.view(torch.int16).int() & 0xffff
    tiles = _to_tiles(bits).transpose(-2, -1).contiguous()
    packed, patches = _pack_four(tiles, base)
    residents = []
    for dense, data, dtype in ((True, tiles.to(torch.int16).view(torch.bfloat16), ttnn.bfloat16),
                               (False, packed, ttnn.uint16)):
        words = ttnn.from_torch(data, dtype=dtype, layout=ttnn.TILE_LAYOUT,
                                device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        metadata = tuple(ttnn.from_torch(patch, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                         device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                         for patch in patches if not dense)
        residents.append((dense, words, metadata, data.numel() * 2 + sum(m.volume() * 4 for m in metadata)))

    for m in m_sizes:
        x = torch.randn((m, k), generator=torch.Generator().manual_seed(54)).bfloat16()
        activations = [
            ttnn.from_torch(torch.nn.functional.pad(rows, (0, 0, 0, 32 - len(rows))), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for rows in x.split(32)
        ]
        calls = []
        for dense, words, metadata, _ in residents:
            slabs = [_linear_operation(ttnn, device, words, activation, n, k, base, dense=dense, metadata=metadata,
                                       prefetch_groups=1 if dense else prefetch_groups) for activation in activations]
            calls.append(lambda slabs=slabs: tuple(slab() for slab in slabs))

        timings = _measure_linear_pair(calls, sync, repeats)
        results = [torch.cat([ttnn.to_torch(output) for output in call()]) for call in calls]
        _same(*results)
        golden = torch.nn.functional.linear(x.float(), weight.float()).bfloat16()
        oracle = _errors(results[1][:m], golden)
        assert oracle["scaled_rmse"] <= 0.002, oracle
        for (dense, _, _, weight_bytes), (median, p95) in zip(residents, timings):
            print(json.dumps({"kind": "matched_bf16_linear" if dense else "sfpu_linear", "shape": [n, k], "m": m,
                              "weight_bytes": weight_bytes, "weight_passes": len(activations), "median_us": median,
                              "p95_us": p95, "effective_GBps": weight_bytes * len(activations) / median / 1000}),
                  flush=True)
        print(json.dumps({"matched_linear_exact": True, "shape": [n, k], "m": m,
                          "packed_prefetch_groups": prefetch_groups, "torch_fp32_oracle": oracle}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chip", required=True, type=int)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path, help="Use the first --size rows of the public Inkling 2 Mini head")
    parser.add_argument("--special", action="store_true", help="Add special BF16 values to synthetic input")
    parser.add_argument("--linear", action="store_true", help="Run the bias-free SFPU and matmul experiment")
    parser.add_argument("--prefetch-groups", type=int, default=1, help="Read this many compressed groups per NoC batch")
    parser.add_argument("--m", nargs="+", type=int, default=[1], help="Logical M sizes to validate in linear mode")
    args = parser.parse_args()
    if args.size < 64 or args.size % 64 or args.repeats < 1:
        parser.error("Use a size that is a positive multiple of 64 and at least one repeat")
    if args.linear and (args.special or (args.checkpoint is None and args.size % 128)):
        parser.error("For linear without a checkpoint, use a synthetic size divisible by 128 and finite weights")
    if args.prefetch_groups < 1:
        parser.error("The number of prefetch groups must be positive")
    if min(args.m) < 1:
        parser.error("M sizes must be positive")
    torch.set_num_threads(3)
    ttnn = import_ttnn()
    device = ttnn.open_device(device_id=args.chip)
    try:
        with torch.inference_mode():
            def sync(_output=None):
                ttnn.synchronize_device(device)

            if args.linear:
                _exercise_linear(ttnn, device, sync, args.size, args.repeats, args.checkpoint,
                                 args.prefetch_groups, args.m)
            else:
                _exercise(ttnn, device, ttnn.to_torch, sync, args.size, args.repeats, args.checkpoint, args.special)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
