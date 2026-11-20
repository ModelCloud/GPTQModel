# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AUTO packing must keep narrow GPTQ storage words on the original packer.

``pack_impl="auto"`` selected CPU block packing from method presence alone, and
the block packer asserts 32-bit words.  Continuous int8/int16 layouts therefore
crashed from both packing entry points on CPU-only runtimes (#5806).
"""

import pytest
import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization.config import FORMAT, GPTQConfig
from gptqmodel.utils.model import pack_module
from gptqmodel.utils.packing import _normalize_pack_impl, hf_pack_layer


BITS = 4
GROUP_SIZE = 16
IN_FEATURES = 32
OUT_FEATURES = 32


def _linear():
    torch.manual_seed(80)
    linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.randint(-8, 8, linear.weight.shape).float() * 0.25)
    return linear


def _module(pack_dtype):
    return TorchLinear(
        bits=BITS,
        group_size=GROUP_SIZE,
        desc_act=False,
        sym=True,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        pack_dtype=pack_dtype,
        format=FORMAT.GPTQ_V2,
        compile=False,
    )


def _inputs():
    scales = torch.full((OUT_FEATURES, IN_FEATURES // GROUP_SIZE), 0.25)
    zeros = torch.full((OUT_FEATURES, IN_FEATURES // GROUP_SIZE), 8, dtype=torch.int32)
    g_idx = torch.arange(IN_FEATURES, dtype=torch.int32) // GROUP_SIZE
    return scales, zeros, g_idx


def _quantize_config(pack_dtype, pack_impl):
    return GPTQConfig(
        bits=BITS,
        group_size=GROUP_SIZE,
        desc_act=False,
        sym=True,
        device="cpu",
        pack_dtype=pack_dtype,
        pack_impl=pack_impl,
        format=FORMAT.GPTQ_V2,
        offload_to_disk=False,
    )


def _pack(entry, pack_dtype, pack_impl):
    linear = _linear()
    module = _module(pack_dtype)
    scales, zeros, g_idx = _inputs()
    if entry == "hf":
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
    else:
        pack_module(
            "proj",
            {"proj": module},
            scales,
            zeros,
            g_idx,
            {"proj": linear},
            TorchLinear,
            lock=None,
            quantize_config=_quantize_config(pack_dtype, pack_impl),
            device="cpu",
        )
    module.post_init()
    return module


def _assert_same_payload(actual, expected):
    assert actual.qweight.dtype == expected.qweight.dtype
    torch.testing.assert_close(actual.qweight, expected.qweight, rtol=0, atol=0)
    torch.testing.assert_close(actual.qzeros, expected.qzeros, rtol=0, atol=0)
    torch.testing.assert_close(actual.scales, expected.scales, rtol=0, atol=0)


@pytest.mark.parametrize("accelerator", [False, True])
@pytest.mark.parametrize("pack_dtype", [torch.int8, torch.int16])
def test_auto_keeps_narrow_words_on_original(monkeypatch, accelerator, pack_dtype):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: accelerator)
    module = _module(pack_dtype)
    assert module.pack_dtype_bits != 32
    assert _normalize_pack_impl("auto", module, _linear()) == "original"


@pytest.mark.parametrize("accelerator,expected", [(False, "block"), (True, "gpu")])
def test_auto_still_prefers_accelerated_32bit_words(monkeypatch, accelerator, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: accelerator)
    assert _normalize_pack_impl("auto", _module(torch.int32), _linear()) == expected


@pytest.mark.parametrize("pack_dtype", [torch.int8, torch.int16])
def test_pack_block_rejects_narrow_words_before_mutating(pack_dtype):
    module = _module(pack_dtype)
    linear = _linear()
    scales, zeros, g_idx = _inputs()
    before = set(module.state_dict())
    with pytest.raises(ValueError, match="32-bit packing words"):
        module.pack_block(linear=linear, scales=scales, zeros=zeros, g_idx=g_idx)
    assert set(module.state_dict()) == before


@pytest.mark.parametrize("entry", ["model", "hf"])
@pytest.mark.parametrize("pack_dtype", [torch.int8, torch.int16, torch.int32])
def test_entry_points_pack_narrow_words_exactly(monkeypatch, entry, pack_dtype):
    # The crash only appears on a CPU-only runtime: with an accelerator visible,
    # AUTO already fell back from ``pack_gpu`` to the original packer.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    expected = _pack(entry, pack_dtype, "original")
    packed = _pack(entry, pack_dtype, "auto")
    _assert_same_payload(packed, expected)
    torch.testing.assert_close(
        packed.dequantize_weight().float(), _linear().weight.T, rtol=0, atol=0
    )


@pytest.mark.parametrize("entry", ["model", "hf"])
@pytest.mark.parametrize("pack_dtype", [torch.int8, torch.int16])
def test_explicit_block_falls_back_for_narrow_words(entry, pack_dtype):
    expected = _pack(entry, pack_dtype, "original")
    packed = _pack(entry, pack_dtype, "block")
    _assert_same_payload(packed, expected)
