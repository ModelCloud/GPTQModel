# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# EXL3 Q4NX codebook by AlexBrandes (Alex Brandes).

"""Benchmark the EXL3 Q4NX encoder (quantize_tiles + pack_trellis) on Ada.

The encoder is the slow half of EXL3 Q4NX: ``quantize_tiles_q4nx`` runs the
Viterbi search and ``pack_trellis_q4nx`` writes the packed trellis. This script
times both halves separately on one representative weight, so a change to
either kernel or to the tile batching constants can be measured end to end.

The 4.0 bpw row is not a Q4NX rate (Q4NX covers 2.5/3.0/3.5). It runs the
integer K=4 EXL3 path through the same public helpers as a reference point.

Usage::

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=<4090> \
        python tests/benchmark_exl3_q4nx_encode_ada.py

Add ``--save-trellis /tmp/trellis.pt`` on one build and ``--expect-trellis
/tmp/trellis.pt`` on another to prove the packed output is bit-identical.
"""

import argparse
import statistics
from pathlib import Path

import torch

from gptqmodel.exllamav3 import ext
from gptqmodel.exllamav3.modules.quant.exl3_lib import quantize as exl3_quantize
from gptqmodel.exllamav3.modules.quant.exl3_lib.quantize import (
    quantize_tiles,
    tile_core_permute,
)


Q4NX_RATES = (2.5, 3.0, 3.5)
REFERENCE_RATE = 4.0


def _tiles_from_weight(weight: torch.Tensor, g_scale: float = 1.0) -> torch.Tensor:
    """Turn a (k, n) fp16 weight into the (tiles, 256) fp32 tile set EXL3 quantizes.

    Mirrors the layout the LDLQ loop hands to ``quantize_tiles``: the
    regularized weight is scaled by the global scale, then each 16-row block is
    permuted into tensor-core fragment order.
    """

    rows, columns = weight.shape
    weight_r = weight.float()
    weight_r /= weight_r.square().mean().sqrt()
    weight_r *= g_scale
    blocks = [
        tile_core_permute(weight_r[start:start + 16], columns // 16)
        for start in range(0, rows, 16)
    ]
    return torch.cat(blocks, dim=0).contiguous()


def _median_ms(function, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _quant_args(rate: float, device: torch.device) -> dict:
    args = {"K": rate, "devices": [device]}
    if rate in Q4NX_RATES:
        args["q4nx"] = True
    return args


def _quantize(tiles, quant_args, batch: int | None):
    """Quantize every tile; returns the encoded suffixes shaped (tiles_k, tiles_n, 256)."""

    if batch is None:
        _, encoded = quantize_tiles(tiles, quant_args)
    else:
        wide = exl3_quantize._Q4NX_WIDE_TILE_BATCH
        narrow = exl3_quantize._Q4NX_NARROW_TILE_BATCH
        exl3_quantize._Q4NX_WIDE_TILE_BATCH = batch
        exl3_quantize._Q4NX_NARROW_TILE_BATCH = batch
        try:
            _, encoded = quantize_tiles(tiles, quant_args)
        finally:
            exl3_quantize._Q4NX_WIDE_TILE_BATCH = wide
            exl3_quantize._Q4NX_NARROW_TILE_BATCH = narrow
    return encoded


def _pack(encoded_3d, rate: float, device: torch.device) -> torch.Tensor:
    packed_shape = (encoded_3d.shape[0], encoded_3d.shape[1], int(16 * rate))
    packed = torch.empty(packed_shape, dtype=torch.int16, device=device)
    if rate in Q4NX_RATES:
        ext.exllamav3_ext.pack_trellis_q4nx(packed, encoded_3d, int(4 * rate))
    else:
        ext.exllamav3_ext.pack_trellis(packed, encoded_3d, int(rate))
    return packed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--columns", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--rates",
        type=float,
        nargs="+",
        default=[*Q4NX_RATES, REFERENCE_RATE],
        help="EXL3 bitrates to encode; Q4NX covers 2.5/3.0/3.5, 4.0 is the integer reference path",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="override the Q4NX tile batching constant (tiles per launch)",
    )
    parser.add_argument("--save-trellis", type=Path, default=None)
    parser.add_argument("--expect-trellis", type=Path, default=None)
    args = parser.parse_args()
    if args.rows % 16 or args.columns % 16:
        parser.error("--rows/--columns must be multiples of 16")
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be >= 0 and --repeats >= 1")
    if args.batch is not None and args.batch < 1:
        parser.error("--batch must be positive")

    if not torch.cuda.is_available():
        raise SystemExit("this benchmark requires a CUDA device")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    weight = torch.randn(
        (args.rows, args.columns), generator=generator, device=device, dtype=torch.float16
    )
    weight_bytes = weight.numel() * weight.element_size()
    tiles = _tiles_from_weight(weight)
    tiles_2d_shape = (args.rows // 16, args.columns // 16, 256)

    print(
        f"device: {torch.cuda.get_device_name(device)}  "
        f"weight: {args.rows}x{args.columns} fp16 ({weight_bytes / 2**20:.1f} MiB)  "
        f"tiles: {tiles.shape[0]}  warmup: {args.warmup}  repeats: {args.repeats}"
        + (f"  batch: {args.batch}" if args.batch is not None else "")
    )
    print("bpw | quantize ms | pack ms | total ms | weight GB/s | packed bytes")

    saved = {}
    for rate in args.rates:
        quant_args = _quant_args(rate, device)
        expected = None
        if args.expect_trellis is not None:
            payload = torch.load(args.expect_trellis, map_location="cpu")
            if rate not in payload:
                raise SystemExit(f"{args.expect_trellis} has no {rate} bpw entry")
            expected = payload[rate]

        encoded = _quantize(tiles, quant_args, args.batch).reshape(tiles_2d_shape)
        packed = _pack(encoded, rate, device)
        if expected is not None:
            actual = packed.to("cpu")
            if actual.shape != expected.shape or not torch.equal(actual, expected):
                raise SystemExit(f"{rate} bpw packed trellis differs from {args.expect_trellis}")
            print(f"{rate:>3} | bit-identical to {args.expect_trellis}", flush=True)
        if args.save_trellis is not None:
            saved[rate] = packed.to("cpu")

        quantize_ms = _median_ms(lambda: _quantize(tiles, quant_args, args.batch), args.warmup, args.repeats)
        pack_ms = _median_ms(lambda: _pack(encoded, rate, device), args.warmup, args.repeats)
        total_ms = quantize_ms + pack_ms
        print(
            f"{rate:>3} | {quantize_ms:11.3f} | {pack_ms:7.3f} | {total_ms:8.3f} | "
            f"{weight_bytes / total_ms / 1e6:10.2f} | {packed.numel() * packed.element_size()}",
            flush=True,
        )

    if args.save_trellis is not None:
        torch.save(saved, args.save_trellis)
        print(f"saved packed trellis for {sorted(saved)} to {args.save_trellis}")


if __name__ == "__main__":
    main()
