"""QQQ CPU accuracy and warm-forward latency, including historical INT8 storage."""

import argparse
import json

import torch
from benchmark_cpu_method_kernels import _bench, _build_qqq, _relative_error

from gptqmodel.nn_modules.qlinear.qqq import _qqq_cpu_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default="512x512,4096x4096,4096x14336")
    parser.add_argument("--tokens", default="1,7,25,128")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--uncached-iters", type=int, default=3)
    parser.add_argument("--aten", action="store_true", help="Also time exact FP32 ATen int4pack for per-channel QQQ")
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    kernel = _qqq_cpu_kernel()
    if kernel is None:
        raise RuntimeError("Packed QQQ CPU kernel unavailable")
    rows = []
    with torch.inference_mode():
        for shape in args.shapes.split(","):
            in_features, out_features = map(int, shape.split("x"))
            for group_size in (-1, 128):
                module, reference = _build_qqq(in_features, out_features, group_size)
                weight = reference["weight"]
                channel = reference["channel"]
                old_weight = weight.to(torch.int8).contiguous()
                if args.aten and group_size == -1:
                    codes = (module._unpack_weight_codes().t().int() ^ 8).contiguous()
                    aten_packed = torch.ops.aten._convert_weight_to_int4pack_for_cpu(codes, 1)
                    aten_scales = torch.zeros(in_features // 128, out_features, 2)
                    aten_scales[:, :, 0] = 16
                for dtype in (torch.float16, torch.bfloat16):
                    for tokens in map(int, args.tokens.split(",")):
                        torch.manual_seed(800 + tokens)
                        activation = torch.randn(tokens, in_features, dtype=dtype)
                        scale = activation.abs().amax(-1, keepdim=True).div(127).float()
                        quantized = (activation / scale).round().clamp(-128, 127).to(torch.int8)
                        expected_accum = quantized.float() @ weight
                        expected = expected_accum * scale * channel
                        packed, group_scales, _ = module._packed_weight_for_cpu()
                        actual_accum = kernel(quantized, packed, group_scales, module.group_size)
                        actual = module(activation)
                        torch.testing.assert_close(actual_accum, expected_accum, rtol=0, atol=0)
                        torch.testing.assert_close(actual, expected.to(dtype), rtol=0, atol=0)

                        def legacy_cached():
                            module._torch_weight_state_key()
                            quant_a, scale_a = module.dynamic_quant(activation)
                            accum = module._int8_matmul(quant_a, old_weight)
                            if accum is None:
                                accum = quant_a.float() @ old_weight.float()
                            return (accum * scale_a * channel).to(dtype)

                        def uncached():
                            quant_a, scale_a = module.dynamic_quant(activation)
                            unpacked, channels = module._dequantize_weight_for_torch()
                            unpacked = unpacked.to(torch.int8).contiguous()
                            accum = module._int8_matmul(quant_a, unpacked)
                            if accum is None:
                                accum = quant_a.float() @ unpacked.float()
                            return (accum * scale_a * channels).to(dtype)

                        torch.testing.assert_close(legacy_cached(), actual, rtol=0, atol=0)
                        torch.testing.assert_close(uncached(), actual, rtol=0, atol=0)
                        packed_ms, _ = _bench(lambda: module(activation), warmup=args.warmup, iters=args.iters)
                        cached_ms, _ = _bench(legacy_cached, warmup=args.warmup, iters=args.iters)
                        uncached_ms, _ = _bench(uncached, warmup=min(2, args.warmup), iters=args.uncached_iters)
                        aten_ms = None
                        if args.aten and group_size == -1:
                            def aten_forward():
                                quant_a, scale_a = module.dynamic_quant(activation)
                                accum = torch.ops.aten._weight_int4pack_mm_for_cpu(
                                    quant_a.float(), aten_packed, 128, aten_scales,
                                )
                                return (accum * scale_a * channel).to(dtype)

                            torch.testing.assert_close(aten_forward(), actual, rtol=0, atol=0)
                            aten_ms, _ = _bench(aten_forward, warmup=min(2, args.warmup), iters=args.uncached_iters)
                        row = {
                            "shape": shape,
                            "group": group_size,
                            "dtype": str(dtype),
                            "tokens": tokens,
                            "packed_ms": packed_ms,
                            "legacy_cached_ms": cached_ms,
                            "uncached_ms": uncached_ms,
                            "aten_fp32_ms": aten_ms,
                            "speedup_vs_uncached": uncached_ms / packed_ms,
                            "speedup_vs_legacy": cached_ms / packed_ms,
                            "accum_max_abs": (actual_accum - expected_accum).abs().max().item(),
                            "visible_max_abs": (actual.float() - expected.to(dtype).float()).abs().max().item(),
                            "raw_max_abs": (actual.float() - expected).abs().max().item(),
                            "raw_relative_error": _relative_error(actual, expected),
                            "packed_cache_bytes": sum(tensor.numel() * tensor.element_size() for tensor in module._torch_packed_cache),
                            "legacy_cache_bytes": old_weight.numel() + channel.numel() * channel.element_size(),
                        }
                        rows.append(row)
                        print(json.dumps(row), flush=True)
    if args.output:
        with open(args.output, "w") as output_file:
            json.dump({
                "torch": torch.__version__, "threads": args.threads, "warmup": args.warmup,
                "iters": args.iters, "uncached_iters": args.uncached_iters, "rows": rows,
            }, output_file, indent=2)


if __name__ == "__main__":
    main()
