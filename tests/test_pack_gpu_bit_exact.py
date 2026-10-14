# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""`pack_gpu` must be bit-exact with the CPU block packer.

Any numeric drift between the accelerator and CPU packers silently changes the
quantized checkpoint, so both are pinned to identical output here.
"""

import pytest
import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear.torch import TorchLinear


def _make_module(out_features: int, in_features: int, bits: int, group_size: int) -> TorchLinear:
    module = TorchLinear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        register_buffers=True,
    )
    module.qzero_format(format=2)
    return module


def _inputs(out_features: int, in_features: int, bits: int, group_size: int):
    groups = in_features // group_size
    generator = torch.Generator().manual_seed(0)
    weight = (
        torch.randint(-8, 8, (out_features, in_features), generator=generator).to(torch.float16) * 0.5
    )
    linear = nn.Linear(in_features, out_features, bias=False).to(torch.float16)
    with torch.no_grad():
        linear.weight.copy_(weight)
    scales = (torch.rand(out_features, groups, generator=generator) * 0.02 + 0.005).to(torch.float16)
    zeros = torch.full((out_features, groups), 1 << (bits - 1), dtype=torch.int32)
    g_idx = torch.arange(in_features, dtype=torch.int32) // group_size
    return linear, scales, zeros, g_idx


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("shape", [(256, 256), (512, 128), (128, 512)])
def test_pack_gpu_matches_cpu_block_packer(bits: int, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device required for GPU packing test")

    torch.cuda.set_device(0)
    out_features, in_features = shape
    group_size = 128
    linear, scales, zeros, g_idx = _inputs(out_features, in_features, bits, group_size)

    cpu_module = _make_module(out_features, in_features, bits, group_size)
    cpu_module.pack_block(
        linear=linear,
        scales=scales.clone(),
        zeros=zeros.clone(),
        g_idx=g_idx.clone(),
        block_in=128,
        workers=1,
    )

    gpu_module = _make_module(out_features, in_features, bits, group_size)
    gpu_module.pack_gpu(
        linear=linear.to("cuda"),
        scales=scales.to("cuda").clone(),
        zeros=zeros.to("cuda").clone(),
        g_idx=g_idx.to("cuda").clone(),
        device="cuda",
    )

    assert gpu_module.qweight.shape == cpu_module.qweight.shape
    assert gpu_module.qzeros.shape == cpu_module.qzeros.shape
    assert torch.equal(gpu_module.qweight, cpu_module.qweight)
    assert torch.equal(gpu_module.qzeros, cpu_module.qzeros)
    assert torch.equal(gpu_module.scales, cpu_module.scales)
    assert torch.equal(gpu_module.g_idx, cpu_module.g_idx)


def test_pack_gpu_rejects_embeddings():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device required for GPU packing test")

    module = _make_module(256, 256, 4, 128)
    embedding = nn.Embedding(256, 256)
    with pytest.raises(ValueError, match="embedding"):
        module.pack_gpu(
            linear=embedding.to("cuda"),
            scales=torch.ones(256, 2, dtype=torch.float16, device="cuda"),
            zeros=torch.zeros(256, 2, dtype=torch.int32, device="cuda"),
            g_idx=torch.zeros(256, dtype=torch.int32, device="cuda"),
            device="cuda",
        )
