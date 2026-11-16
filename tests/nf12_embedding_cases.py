# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""BF16 embedding vectors shared by CUDA and ROCm NF12 runtime tests."""

from __future__ import annotations

import torch
import torch.nn.functional as F


OBSERVED_LITE_PREFILL_CASES = [
    (1, torch.int64),
    (256, torch.int64),
    (312, torch.int32),
    (313, torch.int64),
    (512, torch.int32),
    (513, torch.int64),
    (1536, torch.int64),
    (2048, torch.int32),
    (2049, torch.int64),
    (3072, torch.int32),
    (3072, torch.int64),
    (4095, torch.int64),
]


def small_embedding_case(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(121)
    weight = torch.randn((128, 256), generator=generator, device=device).bfloat16()
    weight[3, 0] = -0.0
    ids = torch.tensor([[3, 127], [0, 9]], device=device, dtype=torch.int64)
    return weight, ids


def observed_lite_prefill_case(
    device: torch.device,
    m: int,
    id_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(10101 + m)
    weight = (torch.randn((512, 5120), generator=generator, device=device) * 0.02).bfloat16()
    weight[0, 0] = -0.0
    weight[-1, -1] = 2.0**-20
    ids = torch.randint(weight.shape[0], (m,), generator=generator, device=device, dtype=id_dtype)
    ids[0], ids[-1] = 0, weight.shape[0] - 1
    return weight, ids


def assert_embedding_lookup_exact(
    lookup,
    weight: torch.Tensor,
    ids: torch.Tensor,
    *,
    check_invalid: bool = False,
) -> torch.Tensor:
    """Independent dense Torch gather, including invalid vocabulary-shard IDs."""
    oracle = F.embedding(ids, weight)
    for _ in range(2):
        actual = lookup(ids)
        assert actual.dtype is torch.bfloat16
        assert tuple(actual.shape) == (*ids.shape, weight.shape[-1])
        assert torch.equal(actual.view(torch.int16), oracle.view(torch.int16))
    if check_invalid:
        invalid = ids.clone()
        invalid[1], invalid[2] = -1, weight.shape[0]
        actual = lookup(invalid)
        assert torch.equal(actual[1:3].view(torch.int16), torch.zeros_like(actual[1:3]).view(torch.int16))
        assert torch.equal(actual[3:].view(torch.int16), oracle[3:].view(torch.int16))
    return actual
