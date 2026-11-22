# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""`pack_gpu` must be bit-exact with the CPU block packer.

Any numeric drift between the accelerator and CPU packers silently changes the
quantized checkpoint, so both are pinned to identical output here.
"""

import pytest
import torch
import torch.nn as nn

from gptqmodel.nn_modules import qlinear as qlinear_module
from gptqmodel.nn_modules.qlinear import _pack_gpu_block_in
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.utils.packing import _normalize_pack_impl


def _make_module(
    out_features: int,
    in_features: int,
    bits: int,
    group_size: int,
    fmt: FORMAT | None = None,
) -> TorchLinear:
    module = TorchLinear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        register_buffers=True,
        format=fmt,
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


# 3-bit continuous and every planar width used to pack through a per-32-group
# Python loop in ``pack_gpu``; these layouts must stay bit-exact against the CPU
# block packer after the vectorized rewrite (#5733).
LAYOUTS = [
    (2, None),
    (4, None),
    (8, None),
    (3, FORMAT.GPTQ_V2),
    (3, FORMAT.GPTQ_P),
    (5, FORMAT.GPTQ_P),
    (6, FORMAT.GPTQ_P),
    (7, FORMAT.GPTQ_P),
]


@pytest.mark.parametrize(("bits", "fmt"), LAYOUTS)
@pytest.mark.parametrize("shape", [(256, 256), (512, 128), (128, 512)])
def test_pack_gpu_matches_cpu_block_packer(bits: int, fmt: FORMAT | None, shape):
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device required for GPU packing test")

    torch.cuda.set_device(0)
    out_features, in_features = shape
    group_size = 128
    linear, scales, zeros, g_idx = _inputs(out_features, in_features, bits, group_size)

    cpu_module = _make_module(out_features, in_features, bits, group_size, fmt=fmt)
    cpu_module.pack_block(
        linear=linear,
        scales=scales.clone(),
        zeros=zeros.clone(),
        g_idx=g_idx.clone(),
        block_in=128,
        workers=1,
    )

    gpu_module = _make_module(out_features, in_features, bits, group_size, fmt=fmt)
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


@pytest.mark.parametrize(("bits", "fmt"), LAYOUTS)
def test_pack_gpu_batched_multi_block_matches_cpu_block_packer(bits: int, fmt: FORMAT | None):
    """The vectorized ``pack_gpu`` qweight path must stay bit-exact when the
    input features span several ``block_in`` chunks.

    Forcing a small ``block_in`` makes ``pack_gpu`` iterate over multiple
    blocks so the per-block destination-row offset (``base_group``) is
    exercised; the batched reshape/shift/sum must reproduce the CPU layout.
    """

    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device required for GPU packing test")

    torch.cuda.set_device(0)
    out_features, in_features = 512, 512
    group_size = 128
    linear, scales, zeros, g_idx = _inputs(out_features, in_features, bits, group_size)

    cpu_module = _make_module(out_features, in_features, bits, group_size, fmt=fmt)
    cpu_module.pack_block(
        linear=linear,
        scales=scales.clone(),
        zeros=zeros.clone(),
        g_idx=g_idx.clone(),
        block_in=128,
        workers=1,
    )

    gpu_module = _make_module(out_features, in_features, bits, group_size, fmt=fmt)
    gpu_module.pack_gpu(
        linear=linear.to("cuda"),
        scales=scales.to("cuda").clone(),
        zeros=zeros.to("cuda").clone(),
        g_idx=g_idx.to("cuda").clone(),
        block_in=128,
        device="cuda",
    )

    assert torch.equal(gpu_module.qweight, cpu_module.qweight)
    assert torch.equal(gpu_module.qzeros, cpu_module.qzeros)


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


@pytest.mark.parametrize(
    ("out_features", "requested", "expected"),
    [
        # 4 Mi elements per block, rounded down to a 32-channel word boundary.
        (14336, 8192, 288),
        (8192, 8192, 512),
        (4096, 8192, 1024),
        # Narrow projections keep the caller's block: the budget is an upper
        # bound, not a target.
        (256, 8192, 8192),
        (512, 128, 128),
        # A projection wider than the whole budget still gets one word.
        (1 << 25, 8192, 32),
    ],
)
def test_pack_gpu_block_budget_derivation(out_features, requested, expected):
    assert _pack_gpu_block_in(out_features, requested, 32) == expected


def test_pack_gpu_derived_block_is_bit_exact(monkeypatch):
    """A budget-derived block split must not change a single packed bit.

    The element budget is shrunk so a small matrix spans many blocks; the
    result must still equal the old whole-matrix CPU packer
    (``block_in=8192``), which is the reference both GPU paths are pinned to.
    """

    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device required for GPU packing test")

    torch.cuda.set_device(0)
    out_features, in_features = 512, 4096
    group_size = 128
    linear, scales, zeros, g_idx = _inputs(out_features, in_features, 4, group_size)

    monkeypatch.setattr(qlinear_module, "PACK_GPU_MAX_BLOCK_ELEMS", 64 << 10)

    # Old whole-matrix packing, computed by the CPU reference packer.
    whole = _make_module(out_features, in_features, 4, group_size)
    whole.pack_block(
        linear=linear,
        scales=scales.clone(),
        zeros=zeros.clone(),
        g_idx=g_idx.clone(),
        block_in=8192,
        workers=1,
    )

    derived = _make_module(out_features, in_features, 4, group_size)
    derived.pack_gpu(
        linear=linear.to("cuda"),
        scales=scales.to("cuda").clone(),
        zeros=zeros.to("cuda").clone(),
        g_idx=g_idx.to("cuda").clone(),
        device="cuda",
    )

    assert torch.equal(derived.qweight, whole.qweight)
    assert torch.equal(derived.qzeros, whole.qzeros)
    assert torch.equal(derived.scales, whole.scales)


def test_auto_pack_impl_prefers_accelerator(monkeypatch):
    module = _make_module(256, 256, 4, 128)
    linear = nn.Linear(256, 256, bias=False)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _normalize_pack_impl("auto", module, linear) == "gpu"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _normalize_pack_impl("auto", module, linear) == "block"

    # Embeddings keep the original packer even with an accelerator present: the
    # block and GPU packers lay the packed words out along the input axis, which
    # is not the embedding layout (#5381).
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert _normalize_pack_impl("auto", module, nn.Embedding(256, 256)) == "original"
