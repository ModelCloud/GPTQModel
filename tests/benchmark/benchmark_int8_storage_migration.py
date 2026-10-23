"""Packed INT4 alternatives versus historical, prohibited INT4-to-INT8 expansion."""

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from benchmark_cpu_method_kernels import _bench
from benchmark_int4_qzero_group_scaled_cpu import _pack_gptq_buffers

from gptqmodel.nn_modules.qlinear.torch_aten_kernel import TorchAtenLinear
from gptqmodel.nn_modules.qlinear.torch_aten_kernel_awq import TorchAtenAwqLinear
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512 import TorchCpuAvx512Linear
from gptqmodel.nn_modules.qlinear.torch_cpu_avx512_awq import TorchCpuAvx512AwqLinear
from gptqmodel.nn_modules.qlinear.torch_int8 import Int8PackedModule
from gptqmodel.quantization.awq.utils.packing_utils import pack_awq


def make_layer(cls, method, codes, zeros, scales, group_size):
    in_features, out_features = codes.shape
    layer = cls(bits=4, group_size=group_size, sym=False, desc_act=False,
                in_features=in_features, out_features=out_features, bias=False)
    if method == "GPTQ":
        qweight, qzeros = _pack_gptq_buffers(codes, zeros)
        layer.qzero_format(format=2)
    else:
        qweight, qzeros = pack_awq(codes, zeros, 4)
    layer.qweight.copy_(qweight)
    layer.qzeros.copy_(qzeros)
    layer.scales.copy_(scales)
    layer.post_init()
    layer.eval()
    return layer


def metrics(actual, raw):
    difference = actual.float() - raw
    visible = raw.to(actual.dtype)
    return {
        "raw_max_abs": difference.abs().max().item(),
        "raw_relative": (difference.norm() / raw.norm().clamp_min(1e-12)).item(),
        "visible_max_abs": (actual.float() - visible.float()).abs().max().item(),
        "visible_gate": torch.allclose(actual, visible, rtol=2e-3, atol=2e-3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", default="512x512,4096x4096")
    parser.add_argument("--groups", default="32,128")
    parser.add_argument("--tokens", default="1,7,25,128")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    rows = []
    with torch.inference_mode():
        for shape in args.shapes.split(","):
            in_features, out_features = map(int, shape.split("x"))
            for group_size in map(int, args.groups.split(",")):
                torch.manual_seed(109 + group_size)
                codes = torch.randint(0, 16, (in_features, out_features), dtype=torch.uint8)
                zeros = torch.randint(0, 16, (in_features // group_size, out_features), dtype=torch.uint8)
                scales = torch.randint(1, 4, zeros.shape).half() / 256
                weight = ((codes.float().reshape(-1, group_size, out_features) - zeros[:, None].float())
                          * scales[:, None].float()).reshape(in_features, out_features)
                channel_scale = (weight.abs().amax(0) / 127).clamp_min(1e-10)
                historical_codes = (weight / channel_scale).round().clamp(-128, 127).to(torch.int8)
                for dtype in (torch.float16, torch.bfloat16):
                    historical = Int8PackedModule(historical_codes.t().contiguous(), channel_scale.to(dtype)).eval()
                    for method in ("GPTQ", "AWQ"):
                        aten_cls = TorchAtenLinear if method == "GPTQ" else TorchAtenAwqLinear
                        avx_cls = TorchCpuAvx512Linear if method == "GPTQ" else TorchCpuAvx512AwqLinear
                        aten = make_layer(aten_cls, method, codes, zeros, scales, group_size)
                        packed = make_layer(avx_cls, method, codes, zeros, scales, group_size) if dtype == torch.bfloat16 else None
                        for tokens in map(int, args.tokens.split(",")):
                            activation = torch.arange(tokens * in_features).reshape(tokens, -1).remainder(9).sub(4).to(dtype) / 16
                            raw = activation.float() @ weight
                            row = {"shape": shape, "group": group_size, "dtype": str(dtype), "method": method, "tokens": tokens}
                            for name, layer in (("aten", aten), ("packed_avx512", packed), ("historical_int8", historical)):
                                if layer is None:
                                    continue
                                output = layer(activation)
                                error = metrics(output, raw)
                                if name != "historical_int8":
                                    torch.testing.assert_close(output, raw.to(dtype), rtol=0, atol=0)
                                    assert layer.qweight.untyped_storage().nbytes() == in_features * out_features // 2
                                latency, _ = _bench(partial(layer, activation), warmup=args.warmup, iters=args.iters)
                                row[name] = {"ms": latency, **error}
                            rows.append(row)
                            print(json.dumps(row), flush=True)
    Path(args.output).write_text(json.dumps({"torch": torch.__version__, "threads": args.threads, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
