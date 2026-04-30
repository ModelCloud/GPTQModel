#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json

import torch

import torch_npu  # noqa: F401

from gptqmodel.utils.komodo_cann import komodo_cann_v3_runtime_error, load_komodo_cann_v3


def _load_probe():
    if not load_komodo_cann_v3():
        raise RuntimeError(komodo_cann_v3_runtime_error() or "Failed to load Komodo-CANN V3 probe.")


def _case(*, rows: int, in_features: int, out_features: int, group_size: int, bias: bool):
    x = torch.randn((rows, in_features), device="npu", dtype=torch.float16)
    signed_weight = torch.randint(
        low=-8,
        high=8,
        size=(in_features, out_features),
        device="npu",
        dtype=torch.int32,
    ).contiguous()
    packed_weight = torch.ops.npu.npu_convert_weight_to_int4pack(signed_weight)
    groups = 1 if group_size == 0 else (in_features + group_size - 1) // group_size
    scales = torch.full((groups, out_features), 0.03125, device="npu", dtype=torch.float16)
    offsets = torch.zeros((groups, out_features), device="npu", dtype=torch.float16)
    bias_tensor = torch.randn((out_features,), device="npu", dtype=torch.float16) if bias else None

    native = torch.ops.npu.npu_weight_quant_batchmatmul(
        x,
        packed_weight,
        scales,
        offsets,
        None,
        None,
        bias_tensor,
        group_size,
    )
    v3 = torch.ops.gptqmodel_komodo_cann.w4a16_matmul(
        x,
        packed_weight,
        scales,
        offsets,
        bias_tensor,
        group_size,
        1,
        16,
        256,
        64,
    )
    torch.npu.synchronize()
    diff = (native - v3).abs()
    return {
        "rows": rows,
        "in_features": in_features,
        "out_features": out_features,
        "group_size": group_size,
        "bias": bias,
        "packed_weight_shape": list(packed_weight.shape),
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "native_sum": float(native.float().sum().item()),
        "v3_sum": float(v3.float().sum().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe aclnnWeightQuantBatchMatmulV3 through a Komodo-CANN torch op.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--in-features", type=int, default=256)
    parser.add_argument("--out-features", type=int, default=256)
    parser.add_argument("--group-size", type=int, action="append", default=None)
    parser.add_argument("--bias", action="store_true")
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    _load_probe()
    group_sizes = args.group_size or [0, 32, 64, 128]
    results = [
        _case(
            rows=args.rows,
            in_features=args.in_features,
            out_features=args.out_features,
            group_size=group_size,
            bias=args.bias,
        )
        for group_size in group_sizes
    ]
    print(json.dumps({"device": args.device, "results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
