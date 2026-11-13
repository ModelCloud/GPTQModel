# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Per-device dispatch checks for the Tenstorrent block-float CUDA extension."""

import pytest
import torch
import torch.nn.functional as F

from gptqmodel.nn_modules.qlinear.tt_blockfloat import TTBlockfloatEmbedding, TTBlockfloatLinear
from gptqmodel.utils import tt_blockfloat_cuda as bf_cuda
from gptqmodel.utils.backend import BACKEND


def test_ready_checks_each_target_before_using_cached_build(monkeypatch):
    """An sm_70 current device must not disable sm_80 or dispatch back to sm_70."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device=None: (7, 0) if device is None or torch.device(device).index == 0 else (8, 0),
    )
    monkeypatch.setattr(bf_cuda, "_READY", None)
    builds = []
    monkeypatch.setattr(bf_cuda, "prewarm", lambda: builds.append(True) or True)

    assert bf_cuda.ready() is False
    assert bf_cuda.ready("cuda:0") is False
    assert builds == []
    assert bf_cuda.ready("cuda:1") is True
    assert builds == [True]
    assert bf_cuda.ready("cuda:0") is False
    assert bf_cuda.ready("cuda:1") is True
    assert builds == [True]


@pytest.mark.skipif(
    torch.version.hip is not None or torch.cuda.device_count() < 2,
    reason="requires two NVIDIA CUDA devices",
)
def test_blockfloat_modules_check_their_own_gpu(monkeypatch):
    """A module on cuda:1 must pass that ordinal to the CUDA capability gate."""

    device = torch.device("cuda:1")
    selected = []
    monkeypatch.setattr(bf_cuda, "ready", lambda target: selected.append(target) or False)

    linear = TTBlockfloatLinear(
        bits=4, in_features=16, out_features=3, bias=False, backend=BACKEND.TT_BLOCKFLOAT
    ).to(device)
    x = torch.ones((1, 16), dtype=torch.bfloat16, device=device)
    got = linear(x)
    expected = F.linear(x, linear.dequantize(dtype=x.dtype))
    assert torch.equal(got, expected)

    embedding = TTBlockfloatEmbedding(3, 16, bits=4).to(device)
    ids = torch.tensor([0, 2], dtype=torch.int64, device=device)
    gathered = embedding(ids)
    reference = F.embedding(ids, embedding.dequantize(dtype=embedding.output_dtype))
    assert torch.equal(gathered, reference)
    assert selected == [device, device]
