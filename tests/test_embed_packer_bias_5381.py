# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Requantizing input embeddings must not crash in the default packer (#5381).

``pack_impl`` defaults to ``"auto"``.  For an ``nn.Embedding`` source that used
to resolve to ``pack_block``, which reads ``linear.bias`` unconditionally and
raised ``AttributeError`` before packing anything.  The block/GPU packers lay
the packed words out along the input axis, which is not the embedding layout, so
embeddings must resolve to ``pack_original``; an explicit ``"block"`` request
has to fall back through the documented ``ValueError`` path instead of raising
``AttributeError``.
"""

import pytest
import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.utils.packing import hf_pack_layer


def _make_module(
    out_features: int, in_features: int, bits: int, group_size: int, bias: bool = False
) -> TorchLinear:
    module = TorchLinear(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        register_buffers=True,
    )
    module.qzero_format(format=2)
    return module


def _embedding_inputs(num_embeddings: int, embedding_dim: int, bits: int, group_size: int):
    """A tiny embedding plus the per-group affine parameters a quantizer emits."""
    generator = torch.Generator().manual_seed(5381)
    weight = (
        torch.randint(-8, 8, (num_embeddings, embedding_dim), generator=generator)
        .to(torch.float16)
        * 0.5
    )
    embedding = nn.Embedding(num_embeddings, embedding_dim).to(torch.float16)
    with torch.no_grad():
        embedding.weight.copy_(weight)

    groups = num_embeddings // group_size
    scales = (torch.rand(embedding_dim, groups, generator=generator) * 0.02 + 0.005).to(torch.float16)
    zeros = torch.full((embedding_dim, groups), 1 << (bits - 1), dtype=torch.int32)
    g_idx = torch.arange(num_embeddings, dtype=torch.int32) // group_size
    return embedding, scales, zeros, g_idx


def _reference_pack(embedding, scales, zeros, g_idx, bits, group_size):
    """The verified workaround from the report: the original packer.

    Routed through ``hf_pack_layer`` so both sides receive the same checkpoint
    qzero conversion.
    """
    num_embeddings, embedding_dim = embedding.weight.shape
    module = _make_module(embedding_dim, num_embeddings, bits, group_size)
    hf_pack_layer(
        module=module,
        linear=embedding,
        scales=scales.clone(),
        zeros=zeros.clone(),
        g_idx=g_idx.clone(),
        pack_impl="original",
    )
    return module


@pytest.mark.parametrize("pack_impl", ["auto", "block"])
@pytest.mark.parametrize("num_embeddings,embedding_dim", [(128, 128), (256, 128)])
def test_embedding_pack_does_not_raise_attribute_error(pack_impl, num_embeddings, embedding_dim):
    # A square embedding (num_embeddings == embedding_dim) is the shape that
    # used to reach ``linear.bias`` instead of failing the g_idx length check.
    bits, group_size = 4, 128
    embedding, scales, zeros, g_idx = _embedding_inputs(
        num_embeddings, embedding_dim, bits, group_size
    )
    expected = _reference_pack(embedding, scales, zeros, g_idx, bits, group_size)

    module = _make_module(embedding_dim, num_embeddings, bits, group_size)
    hf_pack_layer(
        module=module,
        linear=embedding,
        scales=scales.clone(),
        zeros=zeros.clone(),
        g_idx=g_idx.clone(),
        pack_impl=pack_impl,
    )

    assert module.bias is None
    assert torch.equal(module.qweight, expected.qweight)
    assert torch.equal(module.qzeros, expected.qzeros)
    assert torch.equal(module.scales, expected.scales)
    assert torch.equal(module.g_idx, expected.g_idx)


def test_pack_block_reports_unsupported_embedding_layout_as_value_error():
    """Callers fall back on ``ValueError``; the failure mode must stay that."""
    # Square on purpose: the g_idx length check cannot mask the bias read here,
    # so this is the shape that raised AttributeError before the fix.
    num_embeddings, embedding_dim, bits, group_size = 128, 128, 4, 128
    embedding, scales, zeros, g_idx = _embedding_inputs(
        num_embeddings, embedding_dim, bits, group_size
    )
    module = _make_module(embedding_dim, num_embeddings, bits, group_size)

    with pytest.raises(ValueError):
        module.pack_block(
            linear=embedding,
            scales=scales,
            zeros=zeros,
            g_idx=g_idx,
            block_in=128,
            workers=1,
        )


def test_pack_block_still_copies_a_linear_bias():
    """Control: the bias fix must not drop a real bias."""
    out_features, in_features, bits, group_size = 256, 128, 4, 128
    generator = torch.Generator().manual_seed(5382)
    weight = (
        torch.randint(-8, 8, (out_features, in_features), generator=generator)
        .to(torch.float16)
        * 0.5
    )
    linear = nn.Linear(in_features, out_features, bias=True).to(torch.float16)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(torch.arange(out_features, dtype=torch.float16) / 100)

    groups = in_features // group_size
    scales = (torch.rand(out_features, groups, generator=generator) * 0.02 + 0.005).to(torch.float16)
    zeros = torch.full((out_features, groups), 1 << (bits - 1), dtype=torch.int32)
    g_idx = torch.arange(in_features, dtype=torch.int32) // group_size

    module = _make_module(out_features, in_features, bits, group_size, bias=True)
    module.pack_block(
        linear=linear,
        scales=scales,
        zeros=zeros,
        g_idx=g_idx,
        block_in=128,
        workers=1,
    )

    assert module.bias is not None
    assert torch.equal(module.bias, linear.bias.detach().to("cpu"))


def test_pack_gpu_keeps_rejecting_embeddings():
    """``pack_gpu`` never supported the embedding layout; the guard stays."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the pack_gpu guard test")

    embedding = nn.Embedding(256, 128).to(torch.float16)
    module = _make_module(128, 256, 4, 128)
    scales = torch.full((128, 2), 0.01, dtype=torch.float16)
    zeros = torch.full((128, 2), 8, dtype=torch.int32)
    g_idx = torch.arange(256, dtype=torch.int32) // 128

    with pytest.raises(ValueError, match="does not support embedding"):
        module.pack_gpu(
            linear=embedding,
            scales=scales,
            zeros=zeros,
            g_idx=g_idx,
        )
