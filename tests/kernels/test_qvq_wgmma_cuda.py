# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from gptqmodel.quantization.qvq import (
    QVQ_V2B2_P32_LR_RING_STEPS,
    QVQ_V2B2_P32_LR_RINGS_PER_TILE,
    local_ring_states_from_edges,
    pack_local_ring_states,
    pack_qvq_binary_bank_ids,
    reconstruct_local_ring_inner_weight,
)
from gptqmodel.quantization.qvq_codecs import PGC16_CODEBOOK_VERSION, pgc16_levels_for_version
from gptqmodel.utils.qvq_wgmma_cuda import qvq_wgmma_w3_m16, qvq_wgmma_w3_m16_tma


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def _w3_case(*, k: int, n: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    tiles = (k // 32) * (n // 8)
    edges = torch.randint(
        0,
        1 << 6,
        (tiles, QVQ_V2B2_P32_LR_RINGS_PER_TILE, QVQ_V2B2_P32_LR_RING_STEPS),
        generator=generator,
        dtype=torch.int64,
    )
    states = local_ring_states_from_edges(edges, bits=3.0)
    trellis = pack_local_ring_states(states, bits=3.0)
    selectors = torch.randint(
        0,
        2,
        (tiles * QVQ_V2B2_P32_LR_RINGS_PER_TILE,),
        generator=generator,
        dtype=torch.uint8,
    )
    bank_ids = pack_qvq_binary_bank_ids(selectors)
    input = torch.randn((16, k), generator=generator, dtype=torch.float16)
    inner = reconstruct_local_ring_inner_weight(
        trellis,
        bits=3.0,
        in_features=k,
        out_features=n,
        bank_ids=bank_ids,
        bank_alt_id=torch.tensor([3], dtype=torch.uint8),
    )
    reference = input.float() @ inner.float()
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).contiguous()
    return input.cuda(), trellis.cuda(), levels.cuda(), bank_ids.cuda(), reference.cuda()


@pytest.mark.parametrize("kernel", (qvq_wgmma_w3_m16, qvq_wgmma_w3_m16_tma))
@pytest.mark.parametrize("split_count", (1, 2))
def test_qvq_wgmma_w3_m16_matches_dense_reference(kernel, split_count):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 H100/H200 required")
    # K=1024 gives split1 four K256 stages, so the test exercises two-stage
    # pipeline reuse instead of merely filling each stage once.
    input, trellis, levels, bank_ids, reference = _w3_case(k=1024, n=512, seed=20260830 + split_count)
    actual = kernel(
        input,
        trellis,
        levels,
        bank_ids,
        out_features=512,
        bank_alt_id=3,
        split_count=split_count,
    )
    torch.cuda.synchronize()
    error = (actual.float() - reference).abs()
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert error.max().item() <= 4e-2
