# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""3-bit zero-point words pack 32 columns at a time.

``out_features % 32 != 0`` cannot be represented by the 10-1-10-1-10 layout.
``pack_gpu`` used to drop the tail columns silently (writing ``(4, 3)`` where
the registered buffer is ``(4, 5)``), and ``pack_original`` raised IndexError
from the same tail. Every packer must fail closed with ValueError instead so
``hf_pack_layer(pack_impl="auto")`` cannot emit a truncated checkpoint.
"""

import pytest
import torch
import torch.nn as nn

from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.utils.packing import hf_pack_layer


IN_FEATURES = 128
OUT_FEATURES = 48
GROUP_SIZE = 32


def _module(register_buffers: bool = True) -> TorchLinear:
    return TorchLinear(
        bits=3,
        group_size=GROUP_SIZE,
        desc_act=False,
        sym=True,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=register_buffers,
    )


def _fixture(device="cpu"):
    linear = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False).to(device)
    groups = IN_FEATURES // GROUP_SIZE
    scales = torch.full((OUT_FEATURES, groups), 0.02, device=device)
    zeros = torch.full((OUT_FEATURES, groups), 8, dtype=torch.int32, device=device)
    g_idx = torch.arange(IN_FEATURES, dtype=torch.int32, device=device) // GROUP_SIZE
    return linear, scales, zeros, g_idx


def test_registered_3bit_zero_words_keep_the_tail_width():
    module = _module()
    # ceil(48 * 3 / 32) == 5 words; the GPU packer used to emit only 3.
    assert module.qzeros.shape == (IN_FEATURES // GROUP_SIZE, 5)


def test_pack_original_rejects_3bit_tail():
    linear, scales, zeros, g_idx = _fixture()
    module = _module()
    with pytest.raises(ValueError, match="divisible by 32"):
        module.pack_original(linear=linear, scales=scales, zeros=zeros, g_idx=g_idx)


def test_pack_block_rejects_3bit_tail():
    linear, scales, zeros, g_idx = _fixture()
    module = _module()
    with pytest.raises(ValueError, match="divisible by 32"):
        module.pack_block(linear=linear, scales=scales, zeros=zeros, g_idx=g_idx, workers=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pack_gpu requires CUDA")
def test_pack_gpu_rejects_3bit_tail():
    linear, scales, zeros, g_idx = _fixture("cuda")
    module = _module().to("cuda")
    with pytest.raises(ValueError, match="divisible by 32"):
        module.pack_gpu(linear=linear, scales=scales, zeros=zeros, g_idx=g_idx, device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="auto selects pack_gpu on CUDA")
def test_hf_pack_layer_auto_fails_closed_for_3bit_tail():
    linear, scales, zeros, g_idx = _fixture("cuda")
    module = _module().to("cuda")
    with pytest.raises(ValueError, match="divisible by 32"):
        hf_pack_layer(
            module=module,
            linear=linear,
            scales=scales,
            zeros=zeros,
            g_idx=g_idx,
            pack_impl="auto",
        )
