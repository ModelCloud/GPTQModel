# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""CUDA graph replay latency for GPTQ W4A GB10 paths and dense BF16."""

import torch
from triton.testing import do_bench_cudagraph

from gptqmodel.nn_modules.qlinear.w4a_floatx import W4AFP8Linear
from gptqmodel.nn_modules.qlinear.w4a_nvfp4 import W4ANVFP4Linear


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("This benchmark requires GB10 / SM121.")
    print(torch.cuda.get_device_name(), flush=True)
    k = n = 2048
    for rows in (1, 128):
        x = torch.randn((rows, k), device="cuda", dtype=torch.bfloat16)
        dense = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        for cls in (W4AFP8Linear, W4ANVFP4Linear):
            module = cls(
                bits=4, group_size=128, sym=True, desc_act=False,
                in_features=k, out_features=n, bias=False,
            ).cuda()
            module.qweight.random_(0, 0x7FFFFFFF)
            module.qzeros.fill_(0x77777777)
            module.scales.fill_(0.125)
            if isinstance(module, W4ANVFP4Linear):
                module.activation_global_scale.fill_(0.03125)
            module.post_init()
            module(x)
            latency_ms = do_bench_cudagraph(lambda m=module: m(x), rep=20)
            print(f"rows={rows} {cls.__name__}: {latency_ms * 1000:.2f} us", flush=True)
            del module
        latency_ms = do_bench_cudagraph(lambda: torch.nn.functional.linear(x, dense), rep=20)
        print(f"rows={rows} DenseBF16: {latency_ms * 1000:.2f} us", flush=True)


if __name__ == "__main__":
    main()
