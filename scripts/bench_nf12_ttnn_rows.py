#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Validate and benchmark native row NF12 against resident TTNN BF16 embedding."""

import argparse
import json
from pathlib import Path

import torch
from bench_nf12_tenstorrent import _gather_oracle, _measure, _same
from safetensors import safe_open

from gptqmodel.utils.nf12_tenstorrent import prepare_nf12
from gptqmodel.utils.nf12_ttnn_rows import TTNNNF12RowEmbedding
from gptqmodel.utils.ttnn_runtime import import_ttnn


def _check_raw(device):
    base = 57
    raw = torch.randint(0, 4096, (545, 129), dtype=torch.int16)
    raw = ((raw & 2047) | ((raw & 2048) << 4)) + (base << 8)
    raw[:2, :128] = 0
    raw[3, :128] = base << 8
    raw[513, 121:129] = torch.tensor([0, -32768, 1, -32767, 32640, -128, 32641, -63], dtype=torch.int16)
    weight = raw.view(torch.bfloat16)
    layer = TTNNNF12RowEmbedding(device, *prepare_nf12(weight, "synthetic.rows", exp_base=base))
    ids = torch.cat((torch.arange(len(weight)), torch.tensor([15, 16, 31, 32, -1, len(weight), 2**32 + 16, 513])))
    ids = ids.reshape(7, 79)
    _same(layer.embedding(ids), _gather_oracle(weight, ids))
    merged = layer.prepare_gather(layer.upload_ids(ids), ids.numel(), stripes_per_task=2, ids_per_task=8)()
    _same(import_ttnn().to_torch(merged)[:, :weight.shape[1]].reshape(7, 79, -1), _gather_oracle(weight, ids))
    _same(layer.embedding(ids, padding_idx=-32), _gather_oracle(weight, ids.masked_fill(ids == 513, -1)))
    scalar = torch.tensor(16, dtype=torch.int32)
    _same(layer.embedding(scalar), weight[16])
    assert layer.embedding(torch.empty((2, 0), dtype=torch.long)).shape == (2, 0, weight.shape[1])
    print(json.dumps({"raw_bf16_verified": len(weight), "escaped_groups": layer.escape_groups,
                      "invalid_duplicate_padding_rank": True}), flush=True)


def _cases(rows):
    generator = torch.Generator().manual_seed(9)
    scattered = torch.cat((torch.tensor([15, 16, 31, 32, rows - 1, 0, 16]),
                           torch.randperm(rows, generator=generator)[:25]))
    return {
        "one": torch.tensor([rows - 1]),
        "repeat": torch.full((1, 32), 16, dtype=torch.long),
        "contiguous": torch.arange(32).reshape(2, 16),
        "scattered": scattered.reshape(4, 8),
    }


def _bf16_bench(ttnn, device, weight, cases, sync, repeats, dense_upload):
    print(json.dumps({"phase": "dense_upload", "mode": dense_upload}), flush=True)
    layout = ttnn.TILE_LAYOUT if dense_upload == "tile-convert" else ttnn.ROW_MAJOR_LAYOUT
    dense = ttnn.from_torch(weight, dtype=ttnn.bfloat16, layout=layout, device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if dense_upload == "tile-convert":
        dense = ttnn.to_layout(dense, ttnn.ROW_MAJOR_LAYOUT)
    sync()
    print(json.dumps({"phase": "dense_resident", "bytes": dense.volume() * 2}), flush=True)
    metrics = {}
    for case, ids in cases.items():
        resident = ttnn.from_torch(ids.flatten().to(torch.int32), dtype=ttnn.uint32,
                                   layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        output = ttnn.allocate_tensor_on_device(ttnn.Shape((ids.numel(), weight.shape[1])), ttnn.bfloat16,
                                                ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
        result, latency = _measure(lambda: ttnn.embedding(resident, dense, output_tensor=output),
                                   sync, repeats, ids.numel() * weight.shape[1] * 2)
        _same(ttnn.to_torch(result).reshape(*ids.shape, weight.shape[1]), _gather_oracle(weight, ids))
        metrics[case] = latency
        print(json.dumps({"layout": "bf16", "case": case, "ids": ids.numel(),
                          "unique_ids": ids.unique().numel(), **latency}), flush=True)
    return metrics


def _packed_traffic(ids, stripes, stripe_chunk, row_chunk, workers):
    rows = ids.flatten().tolist()
    padded_rows = (len(rows) + row_chunk - 1) // row_chunk * row_chunk
    tasks = ((stripes + stripe_chunk - 1) // stripe_chunk) * (padded_rows // row_chunk)
    workers = min(workers, tasks)
    total = 0
    for worker in range(workers):
        last = None
        for task in range(tasks * worker // workers * row_chunk, tasks * (worker + 1) // workers * row_chunk):
            chunk, output_row = divmod(task, padded_rows)
            if output_row >= len(rows):
                continue
            cache_key = (chunk, rows[output_row])
            if cache_key != last:
                total += min(stripe_chunk, stripes - chunk * stripe_chunk) * 192
                last = cache_key
    return total


def _benchmark(ttnn, device, weight, repeats, dense_upload, exhaustive, stripe_chunks, row_chunks):
    n, k = weight.shape
    cases = _cases(n)

    def sync(_output=None):
        ttnn.synchronize_device(device)

    bf16 = _bf16_bench(ttnn, device, weight, cases, sync, repeats, dense_upload)
    print(json.dumps({"phase": "canonical_pack"}), flush=True)
    descriptor, canonical = prepare_nf12(weight, "model.llm.embed.weight")
    print(json.dumps({"phase": "row_transcode"}), flush=True)
    layer = TTNNNF12RowEmbedding(device, descriptor, canonical)
    row_bytes = layer.storage_descriptor.padded_columns * 3 // 2
    stripes = row_bytes // 192
    compute_grid = device.compute_with_storage_grid_size()
    print(json.dumps({"phase": "packed_resident", "resident_bytes": layer.resident_bytes,
                      "escape_groups": layer.escape_groups, "stripe_row_bytes": row_bytes}), flush=True)
    for case, ids in cases.items():
        count = ids.numel()
        _same(layer.embedding(ids), _gather_oracle(weight, ids))
        resident = layer.upload_ids(ids)
        for chunk in stripe_chunks:
            for row_chunk in row_chunks:
                if row_chunk > count:
                    continue
                call = layer.prepare_gather(resident, count, stripes_per_task=chunk, ids_per_task=row_chunk)
                result, latency = _measure(call, sync, repeats, count * k * 2)
                _same(ttnn.to_torch(result)[:, :k].reshape(*ids.shape, k), _gather_oracle(weight, ids))
                print(json.dumps({"layout": layer.storage_descriptor.layout, "case": case, "ids": count,
                                  "unique_ids": ids.unique().numel(), "uncached_packed_row_bytes": row_bytes,
                                  "stripes_per_task": chunk, "ids_per_task": row_chunk,
                                  "packed_reads_excluding_escapes": _packed_traffic(
                                      ids, stripes, chunk, row_chunk, compute_grid.x * compute_grid.y,
                                  ),
                                  "bf16_over_nf12_speedup": bf16[case]["median_us"] / latency["median_us"],
                                  **latency}), flush=True)
    if exhaustive:
        for start in range(0, n, 2048):
            stop = min(start + 2048, n)
            _same(layer.embedding(torch.arange(start, stop)), weight[start:stop])
        print(json.dumps({"checkpoint_rows_bit_exact": n}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chip", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dense-upload", choices=("tile-convert", "row-major"), default="tile-convert")
    parser.add_argument("--exhaustive", action="store_true")
    parser.add_argument("--stripe-chunks", type=int, nargs="+", default=[1])
    parser.add_argument("--id-chunks", type=int, nargs="+", default=[1])
    args = parser.parse_args()
    torch.manual_seed(12)
    torch.set_num_threads(4)
    ttnn = import_ttnn()
    device = ttnn.open_device(device_id=args.chip)
    try:
        _check_raw(device)
        if args.checkpoint:
            name = "model.llm.embed.weight"
            weights = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
            with safe_open(args.checkpoint / weights[name], framework="pt", device="cpu") as shard:
                _benchmark(ttnn, device, shard.get_tensor(name), args.repeats, args.dense_upload, args.exhaustive,
                           args.stripe_chunks, args.id_chunks)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
