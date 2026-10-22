#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CPU runtime audit for the non-GPTQ/AWQ quantized-linear implementations.

Each optimized path is compared with the portable path it replaces. Accuracy
is normalized against an independent FP32 dequantize-and-matmul oracle. The
benchmark is intentionally method-local: it measures runtime dispatch and
packing, not end-to-end model quality.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch
import torch.nn as nn

from gptqmodel.nn_modules.exllamav3_torch import (
    ExllamaV3TorchLinear,
    _cpu_bf16x3_support,
)
from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.nn_modules.qlinear.paroquant import ParoLinear
from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear
from gptqmodel.nn_modules.qlinear.vptq_torch import VPTQLinear
from gptqmodel.quantization.awq.utils.packing_utils import (
    pack_awq,
    reverse_awq_order,
    unpack_awq,
)
from gptqmodel.quantization.config import VPTQConfig
from gptqmodel.quantization.vptq import VPTQ
from gptqmodel.utils.logger import render_table
from gptqmodel.utils.paroquant import (
    apply_paroquant_rotation_reference,
    build_identity_rotation_buffers,
)


def _parse_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in raw.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise ValueError("token counts must be positive integers")
    return values


def _bench(fn, *, warmup: int, iters: int) -> tuple[float, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        samples = []
        for _ in range(iters):
            start = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples), statistics.mean(samples)


def _relative_error(output: torch.Tensor, reference: torch.Tensor) -> float:
    difference = output.float() - reference.float()
    return (difference.norm() / reference.float().norm().clamp_min(1e-12)).item()


def _visible_relative_error(output: torch.Tensor, reference: torch.Tensor) -> float:
    visible_reference = reference.to(dtype=output.dtype)
    difference = output.float() - visible_reference.float()
    return (difference.norm() / visible_reference.float().norm().clamp_min(1e-12)).item()


def _gops(tokens: int, in_features: int, out_features: int, median_ms: float) -> float:
    return (2.0 * tokens * in_features * out_features) / (median_ms * 1e6)


def _pack_time(fn, *, repeats: int = 5) -> float:
    fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def _fp32_awq_oracle(
    rotated: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    weight, zeros = unpack_awq(qweight, qzeros, bits)
    weight, zeros = reverse_awq_order(weight, zeros, bits)
    weight = torch.bitwise_and(weight, (1 << bits) - 1).float()
    zeros = torch.bitwise_and(zeros, (1 << bits) - 1).float()
    group_index = torch.arange(weight.shape[0]) // group_size
    dense = (weight - zeros[group_index]) * scales.float()[group_index]
    output = rotated.float() @ dense
    if bias is not None:
        output = output + bias.float()
    return output


def _build_qqq(
    in_features: int,
    out_features: int,
    group_size: int,
) -> tuple[QQQTorchLinear, dict[str, object]]:
    torch.manual_seed(100 + in_features)
    linear = nn.Linear(in_features, out_features, bias=False, dtype=torch.float16)
    linear.weight.data.normal_(0.0, 0.04)
    scales = (
        torch.rand(
            (out_features, in_features // group_size),
            dtype=torch.float16,
        )
        * 0.02
        + 0.005
    )
    s_channel = torch.rand(out_features, dtype=torch.float32) * 0.5 + 0.75

    def make():
        module = QQQTorchLinear(
            bits=4,
            group_size=group_size,
            sym=True,
            desc_act=False,
            in_features=in_features,
            out_features=out_features,
            bias=False,
            register_buffers=True,
        )
        module.pack(linear, scales, s_channel)
        module.post_init()
        return module.eval()

    module = make()
    weight, channel = module._cached_weight_for_torch()
    dense = weight.float() * channel
    return module, {"make": make, "dense": dense}


def _build_paro(
    in_features: int,
    out_features: int,
    group_size: int,
) -> tuple[ParoLinear, dict[str, object]]:
    torch.manual_seed(200 + in_features)
    groups = in_features // group_size
    codes = torch.randint(0, 16, (in_features, out_features), dtype=torch.int32)
    zeros = torch.randint(0, 16, (groups, out_features), dtype=torch.int32)
    scales = torch.rand((groups, out_features), dtype=torch.float16) * 0.02 + 0.005
    bias = torch.randn(out_features, dtype=torch.float16) * 0.01
    qweight, qzeros = pack_awq(codes, zeros, 4)
    pairs, theta, channel_scales = build_identity_rotation_buffers(
        in_features=in_features,
        group_size=group_size,
        krot=4,
        dtype=torch.bfloat16,
    )
    theta.normal_()
    channel_scales.uniform_(0.75, 1.25)

    def make():
        module = ParoLinear(
            bits=4,
            group_size=group_size,
            sym=True,
            desc_act=False,
            in_features=in_features,
            out_features=out_features,
            bias=True,
            register_buffers=True,
            krot=4,
        )
        module.qweight.copy_(qweight)
        module.qzeros.copy_(qzeros)
        module.scales.copy_(scales)
        module.bias.copy_(bias)
        module.pairs.copy_(pairs)
        module.theta.copy_(theta)
        module.channel_scales.copy_(channel_scales)
        module.post_init()
        return module.eval()

    module = make()
    return module, {
        "make": make,
        "qweight": qweight,
        "qzeros": qzeros,
        "scales": scales,
        "bias": bias,
        "group_size": group_size,
    }


def _build_fp8(
    in_features: int,
    out_features: int,
) -> tuple[TorchFP8Linear, dict[str, object]]:
    torch.manual_seed(300 + in_features)
    linear = nn.Linear(in_features, out_features, bias=True, dtype=torch.bfloat16)
    linear.weight.data.normal_(0.0, 0.04)
    linear.bias.data.normal_(0.0, 0.01)

    def make():
        module = TorchFP8Linear(
            bits=8,
            group_size=-1,
            desc_act=False,
            sym=True,
            in_features=in_features,
            out_features=out_features,
            bias=True,
            pack_dtype=torch.int32,
            format="float8_e4m3fn",
            weight_scale_method="row",
            weight_block_size=None,
            register_buffers=False,
        )
        module.pack_original(linear=linear, scales=None, zeros=None)
        module._scaled_mm_hard_disabled = True
        return module.eval()

    return make(), {"make": make, "linear": linear}


def _vptq_problem(
    in_features: int,
    out_features: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(400 + in_features)
    weight = torch.randn(out_features, in_features, generator=generator) * 0.05
    features = (
        torch.randn(256, in_features, generator=generator)
        @ torch.randn(
            in_features,
            in_features,
            generator=generator,
        )
        * 0.2
    )
    hessian = (features.double().t() @ features.double()) / features.shape[0]
    return weight, hessian


def _build_vptq(
    in_features: int,
    out_features: int,
    group_size: int,
) -> tuple[VPTQLinear, dict[str, object]]:
    weight, hessian = _vptq_problem(in_features, out_features, group_size)
    qcfg = VPTQConfig(
        num_centroids=[-1, 64],
        num_res_centroids=[-1, -1],
        vector_lens=[-1, 4],
        group_size=group_size,
        kmeans_iters=4,
        block_size=64,
    )
    linear = nn.Linear(in_features, out_features, bias=False, dtype=torch.bfloat16)
    linear.weight.data.copy_(weight.to(torch.bfloat16))
    quantizer = VPTQ(module=linear, qcfg=qcfg)
    quantizer.fallback = None
    result, _loss = quantizer.quantize_weight(
        weight,
        hessian,
        target_dtype=torch.bfloat16,
    )

    def make():
        module = VPTQLinear(
            bits=qcfg.bits,
            in_features=in_features,
            out_features=out_features,
            bias=False,
            vector_lens=qcfg.vector_lens,
            num_centroids=qcfg.num_centroids,
            num_res_centroids=qcfg.num_res_centroids,
            npercent=qcfg.npercent,
            group_size=qcfg.group_size,
            group_num=qcfg.group_num,
            enable_norm=qcfg.enable_norm,
            norm_dim=qcfg.norm_dim,
            enable_perm=qcfg.enable_perm,
            dtype=torch.bfloat16,
            register_buffers=False,
        )
        module.pack(linear=linear, vptq_result=result)
        module.post_init()
        return module.eval()

    return make(), {"make": make, "result": result, "linear": linear}


def _build_gguf(
    bits: str,
    in_features: int,
    out_features: int,
) -> tuple[GGUFTorchLinear, dict[str, object]]:
    torch.manual_seed(500 + in_features + len(bits))
    linear = nn.Linear(in_features, out_features, bias=True, dtype=torch.float32)
    linear.weight.data.normal_(0.0, 0.03)
    linear.bias.data.normal_(0.0, 0.01)

    def make():
        module = GGUFTorchLinear(
            bits=bits,
            group_size=-1,
            sym=True,
            desc_act=False,
            in_features=in_features,
            out_features=out_features,
            bias=True,
            register_buffers=True,
        )
        module.pack_original(linear=linear, scales=None, zeros=None)
        module.gguf_fused_cpu_max_rows = 1024
        module.gguf_fused_cpu_min_matrix_elements = 0
        module.autotune_enabled = False
        module.clear_autotune()
        return module.eval()

    return make(), {"make": make, "linear": linear}


def _build_exl3(
    in_features: int,
    out_features: int,
) -> tuple[ExllamaV3TorchLinear, dict[str, object]]:
    generator = torch.Generator().manual_seed(700 + in_features)
    bits = 4
    tensors = {
        "trellis": torch.randint(
            -32768,
            32767,
            (in_features // 16, out_features // 16, bits * 16),
            dtype=torch.int16,
            generator=generator,
        ),
        "suh": (torch.rand(in_features, generator=generator) * 0.02 + 0.001).to(torch.float16),
        "svh": (torch.rand(out_features, generator=generator) * 0.02 + 0.001).to(torch.float16),
        "bias": torch.randn(out_features, generator=generator, dtype=torch.float16) * 0.01,
    }

    def make() -> ExllamaV3TorchLinear:
        return ExllamaV3TorchLinear.from_tensors(
            in_features=in_features,
            out_features=out_features,
            name="cpu_audit_exl3",
            tensors=tensors,
        ).eval()

    module = make()
    return module, {
        "make": make,
        "weight": module.get_weight_tensor(torch.float32),
        "bias": module.bias.float(),
    }


def _exl3_ieee_forward(
    module: ExllamaV3TorchLinear,
    x: torch.Tensor,
) -> torch.Tensor:
    previous = torch.backends.mkldnn.matmul.fp32_precision
    torch.backends.mkldnn.matmul.fp32_precision = "ieee"
    try:
        output = x.float() @ module.get_weight_tensor(torch.float32)
    finally:
        torch.backends.mkldnn.matmul.fp32_precision = previous
    output = output + module.bias.float()
    return output.view(*x.shape[:-1], module.out_features).to(x.dtype)


def _add_row(
    rows: list[list[str]],
    *,
    method: str,
    variant: str,
    shape: tuple[int, int],
    tokens: int,
    timing: tuple[float, float],
    baseline_ms: float,
    relative_error: float,
    visible_error: float,
) -> None:
    median_ms, mean_ms = timing
    out_features, in_features = shape
    rows.append(
        [
            method,
            variant,
            f"{out_features}x{in_features}",
            str(tokens),
            f"{median_ms:.3f}",
            f"{mean_ms:.3f}",
            f"{_gops(tokens, in_features, out_features, median_ms):.2f}",
            f"{baseline_ms / median_ms:.2f}x",
            f"{relative_error:.4e}",
            f"{visible_error:.4e}",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="512x512")
    parser.add_argument("--tokens", default="1,7,25,64,128")
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    out_features, in_features = (int(part) for part in args.shape.lower().split("x", 1))
    tokens_list = _parse_ints(args.tokens)
    group_size = 128
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # The inter-op pool may already be initialized by an imported runtime.
        pass

    rows: list[list[str]] = []
    pack_rows: list[list[str]] = []

    qqq, qqq_meta = _build_qqq(in_features, out_features, group_size)
    paro, paro_meta = _build_paro(in_features, out_features, group_size)
    fp8, fp8_meta = _build_fp8(in_features, out_features)
    vptq, vptq_meta = _build_vptq(in_features, out_features, group_size)
    exl3, exl3_meta = _build_exl3(in_features, out_features)
    gguf_variants = [
        (label, *_build_gguf(qtype, in_features, out_features))
        for label, qtype in (
            ("GGUF Q4_0", "q4_0"),
            ("GGUF Q8_0", "q8_0"),
            ("GGUF Q4_K", "q4_k"),
            ("GGUF Q5_K", "q5_k"),
            ("GGUF Q6_K", "q6_k"),
        )
    ]

    pack_rows.extend(
        [
            ["QQQ", f"{out_features}x{in_features}", f"{_pack_time(qqq_meta['make']):.3f}"],
            ["ParoQuant", f"{out_features}x{in_features}", f"{_pack_time(paro_meta['make']):.3f}"],
            ["FP8", f"{out_features}x{in_features}", f"{_pack_time(fp8_meta['make']):.3f}"],
            ["VPTQ", f"{out_features}x{in_features}", f"{_pack_time(vptq_meta['make']):.3f}"],
            [
                "EXL3",
                f"{out_features}x{in_features}",
                f"{_pack_time(lambda: exl3_meta['make']().get_weight_tensor(torch.float32)):.3f}",
            ],
        ]
    )
    pack_rows.extend(
        [
            [label, f"{out_features}x{in_features}", f"{_pack_time(meta['make']):.3f}"]
            for label, _, meta in gguf_variants
        ]
    )

    for tokens in tokens_list:
        torch.manual_seed(600 + tokens)
        x = torch.randn(tokens, in_features, dtype=torch.bfloat16)

        qqq_dense = qqq_meta["dense"].to(torch.float32)
        qqq_oracle = x.float() @ qqq_dense

        def qqq_fallback(x_input: torch.Tensor) -> torch.Tensor:
            quant_a, scale_a = qqq.dynamic_quant(x_input)
            weight, scale_channel = qqq._cached_weight_for_torch()
            accum = torch.matmul(quant_a.float(), weight.float())
            return (accum * scale_a * scale_channel).to(x_input.dtype)

        qqq_opt_time = _bench(lambda: qqq(x), warmup=args.warmup, iters=args.iters)
        qqq_base_time = _bench(lambda: qqq_fallback(x), warmup=args.warmup, iters=args.iters)
        with torch.inference_mode():
            qqq_output = qqq(x)
        _add_row(
            rows,
            method="QQQ",
            variant="int8 mm + cached weight",
            shape=(out_features, in_features),
            tokens=tokens,
            timing=qqq_opt_time,
            baseline_ms=qqq_base_time[0],
            relative_error=_relative_error(qqq_output, qqq_oracle),
            visible_error=_visible_relative_error(qqq_output, qqq_oracle),
        )
        _add_row(
            rows,
            method="QQQ",
            variant="fp32 matmul",
            shape=(out_features, in_features),
            tokens=tokens,
            timing=qqq_base_time,
            baseline_ms=qqq_base_time[0],
            relative_error=_relative_error(qqq_fallback(x), qqq_oracle),
            visible_error=_visible_relative_error(qqq_fallback(x), qqq_oracle),
        )

        rotated = apply_paroquant_rotation_reference(
            x,
            paro.pairs,
            paro.theta,
            scales=paro.channel_scales,
            group_size=group_size,
        )
        paro_oracle = _fp32_awq_oracle(
            rotated,
            paro_meta["qweight"],
            paro_meta["qzeros"],
            paro_meta["scales"],
            paro_meta["bias"],
            bits=4,
            group_size=group_size,
        )

        def paro_dense(x_input: torch.Tensor) -> torch.Tensor:
            rotated_input = paro._rotate_inputs(x_input)
            return paro._forward_dense(rotated_input)

        paro_opt_time = _bench(lambda: paro(x), warmup=args.warmup, iters=args.iters)
        paro_base_time = _bench(lambda: paro_dense(x), warmup=args.warmup, iters=args.iters)
        with torch.inference_mode():
            paro_output = paro(x)
        _add_row(
            rows,
            method="ParoQuant",
            variant="vectorized rotation + AVX512",
            shape=(out_features, in_features),
            tokens=tokens,
            timing=paro_opt_time,
            baseline_ms=paro_base_time[0],
            relative_error=_relative_error(paro_output, paro_oracle),
            visible_error=_visible_relative_error(paro_output, paro_oracle),
        )
        _add_row(
            rows,
            method="ParoQuant",
            variant="dense dequant + matmul",
            shape=(out_features, in_features),
            tokens=tokens,
            timing=paro_base_time,
            baseline_ms=paro_base_time[0],
            relative_error=_relative_error(paro_dense(x), paro_oracle),
            visible_error=_visible_relative_error(paro_dense(x), paro_oracle),
        )

        fp8_oracle = (
            torch.matmul(
                x.float(),
                fp8.dequantize_weight(dtype=torch.float32),
            )
            + fp8.bias.float()
        )

        fp8_time = _bench(lambda: fp8(x), warmup=args.warmup, iters=args.iters)
        with torch.inference_mode():
            fp8_output = fp8(x)
        _add_row(
            rows,
            method="FP8",
            variant="dequant + matmul (no dense cache)",
            shape=(out_features, in_features),
            tokens=tokens,
            timing=fp8_time,
            baseline_ms=fp8_time[0],
            relative_error=_relative_error(fp8_output, fp8_oracle),
            visible_error=_visible_relative_error(fp8_output, fp8_oracle),
        )

        vptq_oracle = torch.nn.functional.linear(
            x.float(),
            vptq.dequantize().float(),
        )

        vptq_time = _bench(lambda: vptq(x), warmup=args.warmup, iters=args.iters)
        with torch.inference_mode():
            vptq_output = vptq(x)
        _add_row(
            rows,
            method="VPTQ",
            variant="dequant + matmul (no dense cache)",
            shape=(out_features, in_features),
            tokens=tokens,
            timing=vptq_time,
            baseline_ms=vptq_time[0],
            relative_error=_relative_error(vptq_output, vptq_oracle),
            visible_error=_visible_relative_error(vptq_output, vptq_oracle),
        )

        exl3_oracle = (
            x.double() @ exl3_meta["weight"].double()
            + exl3_meta["bias"].double()
        )
        exl3_opt_time = _bench(
            lambda: exl3(x),
            warmup=args.warmup,
            iters=args.iters,
        )
        exl3_base_time = _bench(
            lambda: _exl3_ieee_forward(exl3, x),
            warmup=args.warmup,
            iters=args.iters,
        )
        with torch.inference_mode():
            exl3_output = exl3(x)
            exl3_base_output = _exl3_ieee_forward(exl3, x)
        _add_row(
            rows,
            method="EXL3",
            variant="auto (oneDNN BF16x3)",
            shape=(out_features, in_features),
            tokens=tokens,
            timing=exl3_opt_time,
            baseline_ms=exl3_base_time[0],
            relative_error=_relative_error(exl3_output, exl3_oracle),
            visible_error=_visible_relative_error(exl3_output, exl3_oracle),
        )
        _add_row(
            rows,
            method="EXL3",
            variant="IEEE FP32 matmul",
            shape=(out_features, in_features),
            tokens=tokens,
            timing=exl3_base_time,
            baseline_ms=exl3_base_time[0],
            relative_error=_relative_error(exl3_base_output, exl3_oracle),
            visible_error=_visible_relative_error(exl3_base_output, exl3_oracle),
        )

        for label, module, _ in gguf_variants:
            oracle = x.float() @ module.dequantize_weight(dtype=torch.float32)
            fused_time = _bench(
                lambda: module._forward_fused_k(x),
                warmup=args.warmup,
                iters=args.iters,
            )
            dense_time = _bench(
                lambda: module._forward_dequant_matmul(x),
                warmup=args.warmup,
                iters=args.iters,
            )
            _add_row(
                rows,
                method=label,
                variant="fused block matmul",
                shape=(out_features, in_features),
                tokens=tokens,
                timing=fused_time,
                baseline_ms=dense_time[0],
                relative_error=_relative_error(module._forward_fused_k(x), oracle),
                visible_error=_visible_relative_error(module._forward_fused_k(x), oracle),
            )
            _add_row(
                rows,
                method=label,
                variant="dense dequant + matmul",
                shape=(out_features, in_features),
                tokens=tokens,
                timing=dense_time,
                baseline_ms=dense_time[0],
                relative_error=_relative_error(module._forward_dequant_matmul(x), oracle),
                visible_error=_visible_relative_error(module._forward_dequant_matmul(x), oracle),
            )

    print("Environment")
    print(
        render_table(
            [
                ["torch", torch.__version__],
                ["torch_threads", str(torch.get_num_threads())],
                ["interop_threads", str(torch.get_num_interop_threads())],
                ["exl3_bf16x3", str(_cpu_bf16x3_support())],
                ["shape", f"{out_features}x{in_features}"],
                ["tokens", args.tokens],
                ["warmup", str(args.warmup)],
                ["iters", str(args.iters)],
                ["timer", "time.perf_counter; median per-call wall time"],
            ],
            headers=["field", "value"],
            tablefmt="grid",
        )
    )
    print("\nPacking / loading")
    print(
        render_table(
            pack_rows,
            headers=["method", "shape", "median ms"],
            tablefmt="grid",
        )
    )
    print("\nInference")
    print(
        render_table(
            rows,
            headers=[
                "method",
                "variant",
                "shape",
                "M",
                "median ms",
                "mean ms",
                "effective GOPS",
                "vs baseline",
                "raw error vs FP32",
                "visible error",
            ],
            tablefmt="grid",
        )
    )
    print(
        "\nAccuracy uses normalized L2 error against an FP32 oracle built from "
        "the method's own dequantized weight. `raw` compares the kernel output "
        "to the unrounded oracle; `visible` rounds the oracle to the output "
        "dtype first. VPTQ packing time loads a precomputed quantization "
        "result and excludes k-means training."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
