# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Benchmark Pangolin (gptq_p) planar GPU kernels vs the Torch planar path and 4-bit Marlin.

Compares, per (bits, shape, batch):
- torch-eager : eager planar unpack/dequant + matmul (the pre-kernel fallback)
- tri-dequant : Triton planar_dequant + cuBLAS matmul
- tri-fused   : Triton fused planar dequant+matmul (planar_matmul)
- tri-gemv    : Triton fused decode-regime GEMV (planar_gemv, small M only)
- marlin-4bit : 4-bit Marlin GEMM as the speed-of-light ceiling

Usage (with GPU allocator leasing):
    python -m gpu_allocator.cli --base-url http://127.0.0.1:17351 run -n 1 --style uuid -- \
        python scripts/benchmark_planar_kernels.py
"""

import argparse
import math
import statistics
import subprocess
import sys


def preflight_idle_check(util_threshold: int = 5) -> None:
    """Stdlib-only idle check on the visible device before importing torch."""
    import os

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        print("preflight: nvidia-smi unavailable; skipping idle check")
        return
    for line in out.strip().splitlines():
        uuid, util, mem = (part.strip() for part in line.split(","))
        if visible and uuid not in visible:
            continue
        if int(util) > util_threshold:
            print(f"preflight: device {uuid} busy (util={util}%); aborting")
            sys.exit(2)
        print(f"preflight: device {uuid} idle (util={util}%, mem={mem} MiB)")


def bench_gpu(fn, warmup: int = 10, iters: int = 50):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times), min(times)


def build_planar_module(bits: int, in_features: int, out_features: int, group_size: int = 128):
    import torch
    import torch.nn as nn

    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.quantization import FORMAT

    torch.manual_seed(bits)
    maxq = (1 << bits) - 1
    groups = math.ceil(in_features / group_size)
    linear = nn.Linear(in_features, out_features, bias=False)
    scales = torch.rand(out_features, groups) * 0.01 + 0.005
    zeros = torch.randint(0, maxq + 1, (out_features, groups)).float()
    g_idx = torch.arange(in_features, dtype=torch.int32) // group_size

    module = TorchLinear(
        bits=bits,
        group_size=group_size,
        sym=False,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        format=FORMAT.GPTQ_P,
        register_buffers=False,
    )
    module.pack_block(linear, scales, zeros, g_idx)
    module.qweight = module.qweight.cuda()
    module.qzeros = module.qzeros.cuda()
    module.scales = module.scales.cuda()
    module.g_idx = module.g_idx.cuda()
    module.eval()
    return module


def build_marlin_module(in_features: int, out_features: int, group_size: int = 128):
    import torch
    import torch.nn as nn

    from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.quantization import FORMAT

    ok, err = MarlinLinear.validate_once()
    if not ok:
        raise RuntimeError(f"marlin unavailable: {err}")

    torch.manual_seed(4)
    groups = math.ceil(in_features / group_size)
    linear = nn.Linear(in_features, out_features, bias=False)
    scales = torch.rand(out_features, groups) * 0.01 + 0.005
    zeros = torch.full((out_features, groups), 8).float()
    g_idx = torch.arange(in_features, dtype=torch.int32) // group_size

    packer = TorchLinear(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        format=FORMAT.GPTQ_V2,
        register_buffers=False,
    )
    packer.pack_block(linear, scales, zeros, g_idx)

    module = MarlinLinear(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        format=FORMAT.GPTQ_V2,
    )
    module.qweight.data = packer.qweight
    module.qzeros.data = packer.qzeros
    module.scales.data = packer.scales
    module.g_idx.data = packer.g_idx
    module = module.cuda()
    module.post_init()
    module.eval()
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--shapes", default="default",
                        choices=["default", "laguna", "glm45", "glm45-base", "all"],
                        help="laguna = Laguna S 2.1, glm45 = GLM-4.5-Air, glm45-base = GLM-4.5, all = union")
    parser.add_argument("--batches", type=int, nargs="*", default=None)
    parser.add_argument("--bits", type=int, nargs="*", default=None,
                        help="bits to benchmark (default: 3 5 6 7)")
    parser.add_argument("--skip-fused", action="store_true",
                        help="skip tri-fused (planar_matmul); avoids its long per-shape Triton compiles")
    args = parser.parse_args()

    preflight_idle_check()

    import torch

    from gptqmodel.nn_modules.triton_utils.planar import planar_dequant, planar_gemv, planar_matmul
    from gptqmodel.utils.pangolin import PANGOLIN_SUPPORTED_M, ensure_pangolin_runtime_available, pangolin_gemv

    pangolin_ok = ensure_pangolin_runtime_available()
    if not pangolin_ok:
        print("pangolin native GEMV unavailable; its column will be n/a")

    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(device)
    print(f"device      : {torch.cuda.get_device_name(device)} (cc {props.major}.{props.minor}, "
          f"{props.multi_processor_count} SMs, {props.total_memory / (1 << 30):.1f} GiB)")
    print(f"torch       : {torch.__version__} cuda {torch.version.cuda}")
    import triton
    print(f"triton      : {triton.__version__}")
    print(f"dtype       : {args.dtype}, iters={args.iters} (median ms)")

    # Real model shapes (K x N), derived from public HF configs:
    #   Laguna S 2.1: hidden=2048, dense intermediate=8192, moe/shared=512,
    #                 heads=48, kv_heads=8, head_dim=128, n_experts=256.
    #   GLM-4.5-Air:  hidden=4096, dense intermediate=10944, moe=1408,
    #                 heads=96, kv_heads=8, head_dim=128, n_routed=128.
    #   GLM-4.5:      hidden=5120, dense intermediate=12288, moe=1536,
    #                 heads=96, kv_heads=8, head_dim=128, n_routed=160.
    laguna_shapes = [
        (2048, 6144),   # q_proj
        (2048, 1024),   # k_proj / v_proj
        (6144, 2048),   # o_proj
        (2048, 8192),   # dense gate / up
        (8192, 2048),   # dense down
        (2048, 512),    # moe/shared gate / up
        (512, 2048),    # moe/shared down
        (2048, 256),    # router gate
    ]
    glm45_air_shapes = [
        (4096, 12288),  # q_proj
        (4096, 1024),   # k_proj / v_proj
        (12288, 4096),  # o_proj
        (4096, 10944),  # dense gate / up
        (10944, 4096),  # dense down
        (4096, 1408),   # moe/shared gate / up
        (1408, 4096),   # moe/shared down
        (4096, 128),    # router gate
    ]
    glm45_base_shapes = [
        (5120, 12288),  # q_proj
        (5120, 1024),   # k_proj / v_proj
        (12288, 5120),  # o_proj
        (5120, 12288),  # dense gate / up
        (12288, 5120),  # dense down
        (5120, 1536),   # moe/shared gate / up
        (1536, 5120),   # moe/shared down
        (5120, 160),    # router gate
    ]

    if args.shapes == "laguna":
        shapes = laguna_shapes
    elif args.shapes == "glm45":
        shapes = glm45_air_shapes
    elif args.shapes == "glm45-base":
        shapes = glm45_base_shapes
    elif args.shapes == "all":
        shapes = laguna_shapes + glm45_air_shapes + glm45_base_shapes
    else:
        shapes = [(4096, 4096), (4096, 11008), (11008, 4096)]

    if args.batches is not None:
        batches = args.batches
    elif args.shapes in ("laguna", "glm45", "glm45-base", "all"):
        batches = [1, 2, 4, 8, 16, 32]
    else:
        batches = [1, 16, 512, 2048]
    bits_list = args.bits or [3, 5, 6, 7]

    marlin_cache = {}
    for shape in shapes:
        try:
            marlin_cache[shape] = build_marlin_module(*shape)
        except Exception as exc:  # noqa: BLE001
            print(f"marlin ceiling unavailable for {shape}: {exc}")
    marlin_ok = bool(marlin_cache)

    header = (f"| {'bits':>4} | {'K x N':>13} | {'M':>5} | {'torch-eager':>12} | {'tri-dequant':>12} | "
              f"{'tri-fused':>10} | {'tri-gemv':>10} | {'pangolin':>10} | {'marlin-4bit':>11} | "
              f"{'best-vs-eager':>13} |")
    sep = "|" + "|".join("-" * (len(col) + 2) for col in
                         ["bits", "K x N".rjust(13), "M".rjust(5), "torch-eager".rjust(12),
                          "tri-dequant".rjust(12), "tri-fused".rjust(10), "tri-gemv".rjust(10),
                          "pangolin".rjust(10), "marlin-4bit".rjust(11), "best-vs-eager".rjust(13)]) + "|"
    print()
    print(header)
    print(sep)

    for bits in bits_list:
        for shape in shapes:
            k, n = shape
            module = build_planar_module(bits, k, n)
            module._triton_dequant_enabled = False  # force eager path for the baseline

            for m in batches:
                torch.manual_seed(m)
                x = (torch.randn(m, k, dtype=dtype, device=device) * 0.5)

                def eager():
                    w = module.dequantize_weight().to(dtype)
                    return torch.matmul(x, w)

                def tri_dequant():
                    w = planar_dequant(dtype, module.qweight, module.scales, module.qzeros,
                                       module.g_idx, bits)
                    return torch.matmul(x, w)

                def tri_fused():
                    return planar_matmul(x, module.qweight, module.scales, module.qzeros,
                                         module.g_idx, bits)

                def tri_gemv():
                    return planar_gemv(x, module.qweight, module.scales, module.qzeros,
                                       module.g_idx, bits)

                def native_gemv():
                    return pangolin_gemv(x, module.qweight, module.scales, module.qzeros,
                                          module.g_idx, bits)

                eager_ms, _ = bench_gpu(eager, iters=args.iters)
                dq_ms, _ = bench_gpu(tri_dequant, iters=args.iters)
                if args.skip_fused:
                    fused_ms = None
                    fused_str = "       n/a"
                else:
                    fused_ms, _ = bench_gpu(tri_fused, iters=args.iters)
                    fused_str = f"{fused_ms:10.3f}"
                if m <= 16:
                    gemv_ms, _ = bench_gpu(tri_gemv, iters=args.iters)
                    gemv_str = f"{gemv_ms:10.3f}"
                else:
                    gemv_ms = None
                    gemv_str = "       n/a"

                if pangolin_ok and m in PANGOLIN_SUPPORTED_M:
                    pangolin_ms, _ = bench_gpu(native_gemv, iters=args.iters)
                    pangolin_str = f"{pangolin_ms:10.3f}"
                else:
                    pangolin_ms = None
                    pangolin_str = "       n/a"

                if marlin_ok and shape in marlin_cache:
                    marlin = marlin_cache[shape]
                    with torch.inference_mode():
                        marlin_ms, _ = bench_gpu(lambda: marlin(x), iters=args.iters)
                    marlin_str = f"{marlin_ms:11.3f}"
                else:
                    marlin_str = "        n/a"

                best = min(t for t in (dq_ms, fused_ms, gemv_ms, pangolin_ms) if t is not None)
                print(f"| {bits:>4} | {k:>5} x {n:>5} | {m:>5} | {eager_ms:12.3f} | {dq_ms:12.3f} | "
                      f"{fused_str} | {gemv_str} | {pangolin_str} | {marlin_str} | "
                      f"{eager_ms / best:12.2f}x |", flush=True)

            module = None
            torch.cuda.empty_cache()

    print()
    print("torch-eager/tri-dequant/tri-fused are per-call transient dequant (no dense weight caching).")


if __name__ == "__main__":
    main()
