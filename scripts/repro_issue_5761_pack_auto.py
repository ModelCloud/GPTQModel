#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Measure AUTO packer selection for an explicit CPU target (issue #5761).

Packs a 4096x4096 BF16 INT4 projection with ``pack_impl="auto"`` and
``QuantizeConfig.device="cpu"`` through both entry points, prints the packer
sequence, the reported label and the median pack time, and checks the packed
buffers against explicit ``block`` and ``original`` packing.

On a host with a visible CUDA/ROCm GPU, the pre-fix code prints
``calls=['pack_gpu', 'pack_original']`` (the GPU packer rejects the CPU target)
and ``pack_module`` reports ``label=module.pack_gpu`` for the slower original
packer. After the fix both entry points report ``calls=['pack_block']``.
"""

from __future__ import annotations

import argparse
import statistics
import time
from unittest.mock import patch

import torch
from torch import nn

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization.config import FORMAT, GPTQConfig
from gptqmodel.utils.model import pack_module
from gptqmodel.utils.packing import hf_pack_layer


PACKERS = ("pack_gpu", "pack_block", "pack_original")


def _config(pack_impl: str) -> GPTQConfig:
    return GPTQConfig(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        device="cpu",
        pack_impl=pack_impl,
        format=FORMAT.GPTQ_V2,
        offload_to_disk=False,
    )


def _module(linear: nn.Linear) -> TorchLinear:
    return TorchLinear(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=False,
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=False,
        register_buffers=False,
        format=FORMAT.GPTQ_V2,
        compile=False,
    )


def _pack(linear, scales, zeros, g_idx, *, entry: str, pack_impl: str):
    module = _module(linear)
    calls = []
    originals = {name: getattr(TorchLinear, name) for name in PACKERS}

    def traced(name):
        def wrapper(*args, **kwargs):
            calls.append(name)
            return originals[name](*args, **kwargs)

        return wrapper

    with (
        patch.object(TorchLinear, "pack_gpu", traced("pack_gpu")),
        patch.object(TorchLinear, "pack_block", traced("pack_block")),
        patch.object(TorchLinear, "pack_original", traced("pack_original")),
    ):
        start = time.perf_counter()
        if entry == "model":
            label = pack_module(
                "proj",
                {"proj": module},
                scales,
                zeros,
                g_idx,
                {"proj": linear},
                TorchLinear,
                lock=None,
                quantize_config=_config(pack_impl),
                device="cpu",
            )
        else:
            hf_pack_layer(
                module,
                linear,
                scales,
                zeros,
                g_idx,
                pack_impl=pack_impl,
                device="cpu",
                checkpoint_format=FORMAT.GPTQ_V2,
            )
            label = None
        elapsed = time.perf_counter() - start
    return module, calls, label, elapsed


def _assert_same(actual, expected, *, entry: str, pack_impl: str) -> None:
    for name in ("qweight", "qzeros", "scales", "g_idx"):
        left = getattr(actual, name)
        right = getattr(expected, name)
        if left.dtype != right.dtype or not torch.equal(left, right):
            raise SystemExit(f"{entry}: `{name}` differs from explicit `{pack_impl}` packing")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=3, help="timed samples after one warmup")
    args = parser.parse_args()

    torch.set_num_threads(1)
    size = args.size
    layer = nn.Linear(size, size, bias=False, dtype=torch.bfloat16)
    torch.manual_seed(321)
    with torch.no_grad():
        layer.weight.copy_((torch.randint(0, 16, (size, size), dtype=torch.int32) - 8) * 0.5)
    scales = torch.full((size, size // 128), 0.5)
    zeros = torch.full_like(scales, 8.0)
    g_idx = torch.arange(size, dtype=torch.int32) // 128

    print(f"cuda_available={torch.cuda.is_available()}")
    for entry in ("model", "hf"):
        expected_block = _pack(layer, scales, zeros, g_idx, entry=entry, pack_impl="block")[0]
        expected_original = _pack(layer, scales, zeros, g_idx, entry=entry, pack_impl="original")[0]
        samples = []
        calls = []
        label = None
        for index in range(args.samples + 1):
            module, calls, label, elapsed = _pack(
                layer, scales, zeros, g_idx, entry=entry, pack_impl="auto"
            )
            if index:
                samples.append(elapsed)
        _assert_same(module, expected_block, entry=entry, pack_impl="block")
        _assert_same(module, expected_original, entry=entry, pack_impl="original")
        print(
            f"{entry}: calls={calls} label={label} median_ms={statistics.median(samples) * 1000:.1f} "
            f"samples_ms={[round(sample * 1000, 1) for sample in samples]}"
        )


if __name__ == "__main__":
    main()
