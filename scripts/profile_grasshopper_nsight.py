# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from gptqmodel import extension  # noqa: E402
from gptqmodel.utils import grasshopper  # noqa: E402


def _dtype(name: str) -> torch.dtype:
    normalized = name.strip().lower().replace("torch.", "")
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _accumulation(value: str, dtype: torch.dtype) -> torch.dtype | str:
    normalized = value.strip().lower().replace("torch.", "")
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    if normalized in {"input", "native"}:
        return dtype
    raise ValueError(f"Unsupported accumulation dtype: {value}")


def _zero_int_tensor(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.empty(shape, device=device, dtype=torch.int32).random_(
        -(2**31), 2**31 - 1
    )


def _make_case(args: argparse.Namespace, device: torch.device):
    dtype = _dtype(args.dtype)
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU does not report BF16 support.")

    if args.bits not in (3, 4, 8):
        raise ValueError("--bits must be 3, 4, or 8.")
    if args.group_size not in (32, 64, 128):
        raise ValueError("--group-size must be 32, 64, or 128.")
    if args.in_features % 256 != 0:
        raise ValueError("--in-features must be divisible by 256.")
    if args.in_features % args.group_size != 0:
        raise ValueError("--in-features must be divisible by --group-size.")
    if args.out_features % 32 != 0:
        raise ValueError("--out-features must be divisible by 32.")

    qweight_rows = (args.in_features // 32) * args.bits
    groups = args.in_features // args.group_size
    qzero_cols = (args.out_features // 32) * args.bits

    if args.mode == "gemv":
        x = torch.randn(args.in_features, device=device, dtype=dtype).contiguous()
    else:
        x = torch.randn(
            args.batch, args.in_features, device=device, dtype=dtype
        ).contiguous()

    qweight = _zero_int_tensor((qweight_rows, args.out_features), device)
    qzeros = _zero_int_tensor((groups, qzero_cols), device)
    scales = (
        torch.rand(groups, args.out_features, device=device, dtype=dtype) * 0.02
        + 0.001
    ).contiguous()

    down = None
    up = None
    up_qweight = None
    up_scales = None
    up_shape = None
    if args.lora == "dense":
        if args.mode == "gemv":
            down = torch.randn(args.rank, device=device, dtype=dtype).contiguous()
        else:
            down = torch.randn(args.batch, args.rank, device=device, dtype=dtype).contiguous()
        up = torch.randn(args.rank, args.out_features, device=device, dtype=dtype).contiguous()
    elif args.lora == "int8":
        if args.mode == "gemv":
            down = torch.randn(args.rank, device=device, dtype=dtype).contiguous()
        else:
            down = torch.randn(args.batch, args.rank, device=device, dtype=dtype).contiguous()
        up_values = args.rank * args.out_features
        up_qweight = torch.empty(up_values, device=device, dtype=torch.int8).random_(-127, 127)
        up_scales = (
            torch.rand(
                (up_values + args.lora_group_size - 1) // args.lora_group_size,
                device=device,
                dtype=dtype,
            )
            * 0.02
            + 0.001
        ).contiguous()
        up_shape = (args.rank, args.out_features)

    return dtype, x, qweight, scales, qzeros, down, up, up_qweight, up_scales, up_shape


def _call(args: argparse.Namespace, tensors, accumulation):
    _dtype_value, x, qweight, scales, qzeros, down, up, up_qweight, up_scales, up_shape = tensors
    if args.mode == "gemv":
        if args.lora == "none":
            return grasshopper.gemv(
                x, qweight, scales, qzeros, args.group_size, accumulation, args.bits
            )
        if args.lora == "dense":
            return grasshopper.gemv_lora(
                x, qweight, scales, qzeros, down, up, args.group_size, accumulation, args.bits
            )
        return grasshopper.gemv_lora_int8(
            x,
            qweight,
            scales,
            qzeros,
            down,
            up_qweight,
            up_scales,
            up_shape,
            args.group_size,
            args.lora_group_size,
            accumulation,
            args.bits,
        )

    if args.lora == "none":
        return grasshopper.gemm(
            x, qweight, scales, qzeros, args.group_size, accumulation, args.bits
        )
    if args.lora == "dense":
        return grasshopper.gemm_lora(
            x, qweight, scales, qzeros, down, up, args.group_size, accumulation, args.bits
        )
    return grasshopper.gemm_lora_int8(
        x,
        qweight,
        scales,
        qzeros,
        down,
        up_qweight,
        up_scales,
        up_shape,
        args.group_size,
        args.lora_group_size,
        accumulation,
        args.bits,
    )


def _kernel_shape(args: argparse.Namespace) -> tuple[int, int, int, int]:
    single_row_gemv = args.mode == "gemv" or args.batch == 1
    if args.bits == 3:
        ktile_half2 = 128
    elif (
        args.mode == "gemm"
        and args.bits in (4, 8)
        and args.group_size == 64
        and args.batch >= 8
        and args.out_features >= 2048
        and (args.out_features >= 8192 or args.in_features >= 8192)
    ):
        ktile_half2 = 32
    elif args.out_features >= 2048:
        ktile_half2 = 64
    elif single_row_gemv:
        ktile_half2 = 32
    else:
        ktile_half2 = 128
    batch_tile_rows = 1
    if args.mode == "gemm" and args.batch >= 2:
        if args.out_features >= 2048:
            if (
                args.bits != 3
                and args.batch >= 8
                and (args.out_features >= 8192 or args.in_features >= 8192)
            ):
                batch_tile_rows = 8
            else:
                batch_tile_rows = 4
        elif args.batch >= 4:
            batch_tile_rows = 4
    q_rows_per_tile = (ktile_half2 * args.bits) // 16
    qweight_rows = (args.in_features // 32) * args.bits
    blocks_x = (qweight_rows + q_rows_per_tile - 1) // q_rows_per_tile
    blocks_y = (args.out_features + 255) // 256
    blocks_z = (
        args.batch + batch_tile_rows - 1
    ) // batch_tile_rows if args.mode == "gemm" else 1
    return blocks_x, blocks_y, blocks_z, batch_tile_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GrassHopper/VecQuant3 Nsight profiling driver."
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--mode", choices=("gemv", "gemm"), default="gemv")
    parser.add_argument("--lora", choices=("none", "dense", "int8"), default="none")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--accumulation", default="fp32")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--in-features", type=int, default=8192)
    parser.add_argument("--out-features", type=int, default=8192)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--lora-group-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--capture-every", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    props = torch.cuda.get_device_properties(device)

    print(
        "device="
        f"{args.device}:{props.name} sm_{props.major}{props.minor} "
        f"mode={args.mode} lora={args.lora} bits={args.bits} group={args.group_size} "
        f"dtype={args.dtype} accumulation={args.accumulation} "
        f"batch={args.batch} in={args.in_features} out={args.out_features}"
    )
    blocks_x, blocks_y, blocks_z, batch_tile_rows = _kernel_shape(args)
    print(
        f"expected_kernel_grid=({blocks_x}, {blocks_y}, {blocks_z}) "
        f"ctas={blocks_x * blocks_y * blocks_z} threads_per_cta=256 "
        f"batch_tile_rows={batch_tile_rows}"
    )

    extension.load("vecquant3")
    dtype = _dtype(args.dtype)
    accumulation = _accumulation(args.accumulation, dtype)
    tensors = _make_case(args, device)

    for _ in range(args.warmup):
        _call(args, tensors, accumulation)
    torch.cuda.synchronize(device)

    timings_ms: list[float] = []
    output = None
    torch.cuda.nvtx.range_push(
        f"grasshopper_{args.mode}_{args.lora}_bits{args.bits}_g{args.group_size}"
    )
    for iteration in range(args.iters):
        if args.capture_every > 1 and iteration % args.capture_every:
            output = _call(args, tensors, accumulation)
            continue
        start = time.perf_counter()
        output = _call(args, tensors, accumulation)
        torch.cuda.synchronize(device)
        timings_ms.append((time.perf_counter() - start) * 1000.0)
    torch.cuda.nvtx.range_pop()

    if output is None:
        raise RuntimeError("No profiling iterations executed.")
    checksum = float(output.float().mean().detach().cpu())
    print(
        f"timing_ms mean={statistics.fmean(timings_ms):.4f} "
        f"median={statistics.median(timings_ms):.4f} "
        f"min={min(timings_ms):.4f} max={max(timings_ms):.4f} "
        f"checksum={checksum:.6e}"
    )


if __name__ == "__main__":
    main()
