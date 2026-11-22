# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CPU packing must leave checkpoint buffers updatable outside inference mode.

Both packers created persistent ``qweight``/``qzeros``/``scales`` buffers inside
``torch.inference_mode()`` (or inside inference-mode helpers), so an ordinary
``load_state_dict(assign=False)`` raised
``Inplace update to inference tensor outside InferenceMode`` (#5814).
"""

import pytest
import torch
from torch import nn

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT
from gptqmodel.utils.packing import hf_pack_layer


GROUP_SIZE = 32
SIZE = 32
BUFFERS = ("qweight", "qzeros", "scales")


def _quant(bits=4):
    return TorchLinear(
        bits=bits,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=SIZE,
        out_features=SIZE,
        bias=False,
        format=FORMAT.GPTQ_V2,
        compile=False,
    ).eval()


def _pack(quant, value, pack_impl, post_init=True, inference=False):
    layer = nn.Linear(SIZE, SIZE, bias=False, dtype=torch.float16)
    with torch.no_grad():
        layer.weight.fill_(value)
    args = (
        quant,
        layer,
        torch.ones(SIZE, 1),
        torch.full((SIZE, 1), 8),
        torch.zeros(SIZE, dtype=torch.int32),
    )
    kwargs = {
        "checkpoint_format": FORMAT.GPTQ_V2,
        "pack_impl": pack_impl,
        "device": "cpu",
        "post_init": post_init,
    }
    if inference:
        with torch.inference_mode():
            hf_pack_layer(*args, **kwargs)
    else:
        hf_pack_layer(*args, **kwargs)
    return quant


def _packed(value, pack_impl, bits=4, post_init=True):
    return _pack(_quant(bits), value, pack_impl, post_init=post_init)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("pack_impl", ["block", "original"])
def test_packing_leaves_mutable_checkpoint_buffers(pack_impl, bits):
    module = _packed(1.0, pack_impl, bits=bits)
    for name in BUFFERS:
        assert not getattr(module, name).is_inference(), name


@pytest.mark.parametrize("pack_impl", ["block", "original"])
def test_packing_inside_ambient_inference_mode_stays_mutable(pack_impl):
    # Loading pipelines may pack inside ``torch.inference_mode()``; the long-lived
    # checkpoint buffers must still come out mutable. The module itself is built
    # outside that scope, as in the reported loading path.
    module = _pack(_quant(), 1.0, pack_impl, inference=True)
    for name in BUFFERS:
        assert not getattr(module, name).is_inference(), name


@pytest.mark.parametrize("assign", [False, True])
@pytest.mark.parametrize("pack_impl", ["block", "original"])
def test_ambient_inference_packing_accepts_a_state_dict_copy(pack_impl, assign):
    # #5919 removed the inference-mode override from the original packer, so a
    # module packed inside ``torch.inference_mode()`` kept inference tensors and
    # an ordinary copy failed before the module ever ran a forward.
    target = _pack(_quant(), 1.0, pack_impl, inference=True)
    source = _pack(_quant(), 2.0, pack_impl)

    target.load_state_dict(source.state_dict(), assign=assign)

    for name in BUFFERS:
        torch.testing.assert_close(
            getattr(target, name), getattr(source, name), rtol=0, atol=0
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pack_gpu needs a CUDA device")
def test_pack_gpu_leaves_mutable_checkpoint_buffers():
    module = TorchLinear(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=SIZE,
        out_features=SIZE,
        bias=False,
        pack_dtype=torch.int32,
        format=FORMAT.GPTQ_V2,
        compile=False,
    ).eval()
    layer = nn.Linear(SIZE, SIZE, bias=False, dtype=torch.float16).cuda()
    with torch.no_grad():
        layer.weight.fill_(1.0)
    module.pack_gpu(
        linear=layer,
        scales=torch.ones(SIZE, 1),
        zeros=torch.full((SIZE, 1), 8),
        g_idx=torch.zeros(SIZE, dtype=torch.int32),
        device="cuda",
    )
    for name in BUFFERS:
        assert not getattr(module, name).is_inference(), name


@pytest.mark.parametrize("assign", [False, True])
@pytest.mark.parametrize("pack_impl", ["block", "original"])
def test_state_dict_update_after_warmed_forward(pack_impl, assign):
    torch.set_num_threads(1)
    x = torch.eye(SIZE, dtype=torch.float16)[:1]
    source = _packed(1.0, pack_impl)
    target = _packed(2.0, pack_impl)
    expected = torch.ones_like(x)
    torch.testing.assert_close(source(x), expected, rtol=0, atol=0)
    # Warm the destination before the update: a second pack or a stale runtime
    # layout must not block the copy either.
    torch.testing.assert_close(target(x), 2 * expected, rtol=0, atol=0)

    state = {name: value.clone() for name, value in source.state_dict().items()}
    target.load_state_dict(state, assign=assign)

    for name in BUFFERS:
        torch.testing.assert_close(
            getattr(target, name), getattr(source, name), rtol=0, atol=0
        )
    torch.testing.assert_close(target(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("pack_impl", ["block", "original"])
def test_repack_over_existing_module_stays_mutable(pack_impl):
    module = _packed(1.0, pack_impl)
    layer = nn.Linear(SIZE, SIZE, bias=False, dtype=torch.float16)
    with torch.no_grad():
        layer.weight.fill_(2.0)
    hf_pack_layer(
        module,
        layer,
        torch.ones(SIZE, 1),
        torch.full((SIZE, 1), 8),
        torch.zeros(SIZE, dtype=torch.int32),
        checkpoint_format=FORMAT.GPTQ_V2,
        pack_impl=pack_impl,
        device="cpu",
    )
    for name in BUFFERS:
        assert not getattr(module, name).is_inference(), name
    torch.testing.assert_close(
        module(torch.eye(SIZE, dtype=torch.float16)[:1]),
        torch.full((1, SIZE), 2.0, dtype=torch.float16),
        rtol=0,
        atol=0,
    )
