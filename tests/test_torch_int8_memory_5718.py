# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""``TorchInt8Linear`` must not keep a second full-size copy of the codes.

``post_init`` repacks the checkpoint into the group-preserving INT8 runtime
layout. Before #5718 the checkpoint ``qweight`` stayed resident as well, so
every layer used about 2x its 8-bit weight bytes. The repack is an exact
inverse, so the checkpoint words are now rebuilt on ``state_dict()``/save (and
before a re-transform) instead of being kept.
"""

import torch

from gptqmodel.nn_modules.qlinear.torch_int8 import TorchInt8Linear
from gptqmodel.utils.model import get_state_dict_for_save
from tests.test_torch_weight_storage import build_case


def _resident_bytes(module):
    seen = set()
    total = 0
    for tensor in (*module.buffers(), *module.parameters()):
        if tensor is None or id(tensor) in seen:
            continue
        seen.add(id(tensor))
        total += tensor.numel() * tensor.element_size()
    return total


def _build(in_features=2048, out_features=512, group_size=128):
    source, weight = build_case(
        bits=8,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
    )
    checkpoint = {name: tensor.clone() for name, tensor in source.state_dict().items()}
    module = TorchInt8Linear(
        bits=8,
        group_size=group_size,
        sym=False,
        desc_act=True,
        in_features=in_features,
        out_features=out_features,
    )
    module.load_state_dict(checkpoint, strict=True)
    module.post_init()
    module.eval()
    return module, checkpoint, weight


def test_post_init_releases_the_checkpoint_qweight_copy():
    module, _, weight = _build()
    code_bytes = module.in_features * module.out_features

    assert module._checkpoint_qweight() is None
    assert module._buffers.get("qweight") is None
    # One copy of the codes plus the small affine metadata. Before the fix the
    # checkpoint qweight doubled this (about 2.08x).
    assert _resident_bytes(module) < 1.15 * code_bytes

    activation = (
        torch.arange(7 * module.in_features).reshape(7, -1).remainder(11).sub(5).to(torch.float32) / 16
    )
    with torch.inference_mode():
        first = module(activation)
        repeated = module(activation)
    torch.testing.assert_close(repeated, first, rtol=0, atol=0)
    torch.testing.assert_close(first.double(), activation.double() @ weight.double(), rtol=2e-3, atol=2e-3)
    assert module._checkpoint_qweight() is None, "repeated forwards must not materialize qweight"


def test_state_dict_and_writer_rebuild_the_checkpoint_words():
    module, checkpoint, _ = _build()

    saved = module.state_dict()
    assert set(saved) == set(checkpoint)
    for name, tensor in checkpoint.items():
        torch.testing.assert_close(saved[name], tensor, rtol=0, atol=0)

    # The writer walks buffers, so the released slot must be published first
    # or a save would silently omit the packed weights.
    owner = torch.nn.Module()
    owner.add_module("projection", module)
    sources = get_state_dict_for_save(owner)
    assert "projection.qweight" in sources
    torch.testing.assert_close(
        sources["projection.qweight"].source, checkpoint["qweight"], rtol=0, atol=0
    )


def test_reload_after_release_keeps_the_runtime_layout_current():
    module, _, weight = _build()
    state = {name: tensor.clone() for name, tensor in module.state_dict().items()}
    module.load_state_dict(state, strict=True)
    # Loading materializes the checkpoint words; the next forward re-derives
    # the INT8 layout and releases them again.
    activation = torch.ones(3, module.in_features, dtype=torch.float32) / 8
    with torch.inference_mode():
        output = module(activation)
    assert module._checkpoint_qweight() is None
    torch.testing.assert_close(output.double(), activation.double() @ weight.double(), rtol=2e-3, atol=2e-3)


def test_unreconstructable_padding_keeps_the_checkpoint_source():
    """Fail closed: a padded tail the layout cannot reproduce stays resident."""

    in_features, out_features = 17, 8
    generator = torch.Generator().manual_seed(5718)
    padded = -(-in_features // 4) * 4
    # Padding rows carry real packed bytes, so the exact checkpoint tail has no
    # inverse. The source must stay authoritative rather than export zeros.
    codes = torch.randint(0, 256, (padded, out_features), generator=generator)
    zeros = torch.randint(0, 256, (1, out_features), generator=generator)
    scales = (torch.rand(1, out_features, generator=generator) * 0.01 + 1e-3).half()
    groups = torch.zeros(in_features, dtype=torch.int32)

    def pack(words):
        return (
            (words[..., 0] << 0)
            | (words[..., 1] << 8)
            | (words[..., 2] << 16)
            | (words[..., 3] << 24)
        ).to(torch.int32).contiguous()

    checkpoint = {
        "qweight": pack(codes.reshape(padded // 4, 4, out_features).permute(0, 2, 1)),
        "qzeros": pack(zeros.reshape(1, out_features // 4, 4)),
        "scales": scales,
        "g_idx": groups,
    }
    module = TorchInt8Linear(
        bits=8,
        group_size=-1,
        sym=False,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=False,
    )
    module.load_state_dict(checkpoint, strict=True)
    module.post_init()
    module.eval()

    assert module._checkpoint_qweight() is not None
    torch.testing.assert_close(module.state_dict()["qweight"], checkpoint["qweight"], rtol=0, atol=0)
    expected = (codes[:in_features] - zeros[0]).double() * scales.double()[0]
    torch.testing.assert_close(module.dequantize_weight().double(), expected, rtol=0, atol=0)
