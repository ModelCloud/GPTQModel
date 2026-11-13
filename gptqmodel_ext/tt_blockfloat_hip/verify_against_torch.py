#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Compare ``ttbf_selftest --dump DIR`` outputs with the Torch oracle.

The standalone hipcc harness validates the kernels against its own CPU
implementation.  This script closes the loop against the *checkpoint* oracle in
``gptqmodel.utils.tt_blockfloat_quant`` (CPU torch is enough), so a ROCm host
without a ROCm-built PyTorch can still prove bit-exactness:

    hipcc --offload-arch=gfx1201 -O3 -std=c++17 \
        tt_blockfloat_hip.hip tt_blockfloat_hip_selftest.cpp -o ttbf_selftest
    ./ttbf_selftest --dump /tmp/ttbf_dump
    python verify_against_torch.py /tmp/ttbf_dump
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gptqmodel.utils.tt_blockfloat_quant import (  # noqa: E402
    _unpack_codes,
    quantize_blockfloat,
    recompose_blockfloat,
)


ROWS, COLS = 128, 128
LINEAR_M = 32


def _read(path: Path, dtype) -> torch.Tensor:
    return torch.from_numpy(np.fromfile(path, dtype=dtype))


def _codes_and_exponents(dump: Path, suffix: str, bits: int):
    codes = _read(dump / f"q{suffix}_codes.bin", np.uint8)
    exponents = _read(dump / f"q{suffix}_exp.bin", np.int8).reshape(ROWS, COLS // 16)
    unpacked = _unpack_codes(codes, bits, (ROWS, COLS))
    return unpacked, exponents


def check_quantize(dump: Path, bits: int) -> list[str]:
    suffix = "4" if bits == 4 else "8"
    source = _read(dump / f"q{suffix}_in.bin", np.float32).reshape(ROWS, COLS)
    unpacked, exponents = _codes_and_exponents(dump, suffix, bits)
    got = recompose_blockfloat(unpacked, exponents, bits=bits)
    expected, _ = quantize_blockfloat(source, bits)
    return [
        f"bf{bits} quantize weights bit-exact",
        "PASS" if torch.equal(got, expected) else "FAIL",
    ]


def check_dequant(dump: Path, bits: int) -> list[list[str]]:
    suffix = "4" if bits == 4 else "8"
    unpacked, exponents = _codes_and_exponents(dump, suffix, bits)
    expected = recompose_blockfloat(unpacked, exponents, bits=bits)
    got_f32 = _read(dump / f"d{suffix}_out_f32.bin", np.float32).reshape(ROWS, COLS)
    got_bf16 = _read(dump / f"d{suffix}_out_bf16_as_f32.bin", np.float32).reshape(ROWS, COLS)
    return [
        [
            f"bf{bits} dequant f32 bit-exact",
            "PASS" if torch.equal(got_f32, expected) else "FAIL",
        ],
        [
            f"bf{bits} dequant bf16 rounding",
            "PASS" if torch.equal(got_bf16, expected.to(torch.bfloat16).float()) else "FAIL",
        ],
    ]


def check_linear(dump: Path) -> list[str]:
    codes = _read(dump / "lin_codes.bin", np.uint8)
    exponents = _read(dump / "lin_exp.bin", np.int8).reshape(ROWS, COLS // 16)
    weight = recompose_blockfloat(_unpack_codes(codes, 4, (ROWS, COLS)), exponents, bits=4)
    x = _read(dump / "lin_x.bin", np.float32).reshape(LINEAR_M, COLS)
    bias = _read(dump / "lin_bias.bin", np.float32)
    got = _read(dump / "lin_out.bin", np.float32).reshape(LINEAR_M, ROWS)
    expected = x @ weight.T + bias
    norm_l1 = float((got - expected).abs().sum() / expected.abs().sum().clamp_min(1e-6))
    return [
        "bf4 linear m=32 norm_l1",
        f"{norm_l1:.3g}",
        "PASS" if norm_l1 < 2e-3 else "FAIL",
    ]


def check_embedding(dump: Path) -> list[str]:
    unpacked, exponents = _codes_and_exponents(dump, "4", 4)
    table = recompose_blockfloat(unpacked, exponents, bits=4)
    ids = _read(dump / "emb_ids.bin", np.int64)
    got = _read(dump / "emb_out.bin", np.float32).reshape(ids.numel(), COLS)
    expected = table[ids]
    return [
        "bf4 embedding gather bit-exact",
        "PASS" if torch.equal(got, expected) else "FAIL",
    ]


def check_linear_bf16(dump: Path) -> list[str]:
    codes = _read(dump / "lin_codes.bin", np.uint8)
    exponents = _read(dump / "lin_exp.bin", np.int8).reshape(ROWS, COLS // 16)
    weight = recompose_blockfloat(_unpack_codes(codes, 4, (ROWS, COLS)), exponents, bits=4)

    def _bf16(name: str, shape) -> torch.Tensor:
        raw = _read(dump / name, np.uint16)
        return raw.view(torch.bfloat16).reshape(shape)

    x = _bf16("lin_bf16_x.bin", (LINEAR_M, COLS))
    bias = _bf16("lin_bf16_bias.bin", (ROWS,))
    got = _bf16("lin_bf16_out.bin", (LINEAR_M, ROWS))
    expected = (x.float() @ weight.T + bias.float()).to(torch.bfloat16)
    norm_l1 = float((got.float() - expected.float()).abs().sum() / expected.float().abs().sum().clamp_min(1e-6))
    return [
        "bf4 linear bf16 (WMMA) norm_l1",
        f"{norm_l1:.3g}",
        "PASS" if norm_l1 < 2e-3 else "FAIL",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path, help="directory written by --dump")
    args = parser.parse_args()

    rows: list[list[str]] = []
    rows.append(check_quantize(args.dump, 4))
    rows.append(check_quantize(args.dump, 8))
    rows.extend(check_dequant(args.dump, 4))
    rows.extend(check_dequant(args.dump, 8))
    rows.append(check_linear(args.dump))
    rows.append(check_linear_bf16(args.dump))
    rows.append(check_embedding(args.dump))

    failures = 0
    for row in rows:
        if row[-1] == "FAIL":
            failures += 1
        print(f"{row[0]:<44} {'  '.join(row[1:])}")
    print("OK" if failures == 0 else f"FAILED ({failures})")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
