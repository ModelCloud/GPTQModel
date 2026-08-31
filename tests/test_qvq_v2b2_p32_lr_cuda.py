# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CUDA correctness and dispatch coverage for the QVQ LR32 K32 x N8 kernel."""

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
from gptqmodel.quantization.qvq_rates import qvq_transition_bits
from gptqmodel.utils.qvq_cuda import qvq_cuda_gemv, qvq_cuda_supported

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not qvq_cuda_supported(), reason="requires NVIDIA CUDA compute capability >= 8.0"),
]


def _lr_case(bits: float, *, m: int, k: int, n: int, seed: int, dtype=torch.float16):
    transition_bits = qvq_transition_bits(bits, vector_size=2)
    generator = torch.Generator().manual_seed(seed)
    tiles = (k // 32) * (n // 8)
    edges = torch.randint(
        0,
        1 << transition_bits,
        (tiles, QVQ_V2B2_P32_LR_RINGS_PER_TILE, QVQ_V2B2_P32_LR_RING_STEPS),
        generator=generator,
        dtype=torch.int64,
    )
    states = local_ring_states_from_edges(edges, bits=bits)
    trellis = pack_local_ring_states(states, bits=bits)
    selectors = torch.randint(0, 2, (tiles * QVQ_V2B2_P32_LR_RINGS_PER_TILE,), generator=generator, dtype=torch.uint8)
    packed_selectors = pack_qvq_binary_bank_ids(selectors)
    x = torch.randn((m, k), generator=generator, dtype=dtype)
    inner = reconstruct_local_ring_inner_weight(
        trellis,
        bits=bits,
        in_features=k,
        out_features=n,
        bank_ids=packed_selectors,
        bank_alt_id=torch.tensor([3], dtype=torch.uint8),
    )
    return x.cuda(), trellis.cuda(), packed_selectors.cuda(), x.float() @ inner.float()


@pytest.mark.parametrize("bits", (1.0, 1.5, 2.0, 2.5, 3.0, 3.5))
@pytest.mark.parametrize("m,k,n", ((1, 64, 40), (4, 128, 32), (17, 2048, 256)))
def test_lr32_cuda_matches_local_ring_dense_reference(bits, m, k, n):
    x, trellis, bank_ids, reference = _lr_case(bits, m=m, k=k, n=n, seed=20260829 + int(bits * 10) + m)
    actual = qvq_cuda_gemv(
        x,
        trellis,
        bits,
        out_features=n,
        output_fp32=True,
        bank_ids=bank_ids,
        v2b2_p32_lr=True,
        bank_alt_id=3,
    )
    error = (actual - reference.cuda()).abs()
    assert torch.isfinite(actual).all()
    assert error.max().item() <= 2e-3


def test_lr32_cuda_repeated_launches_are_deterministic():
    x, trellis, bank_ids, reference = _lr_case(2.0, m=4, k=512, n=64, seed=20260830)
    outputs = [
        qvq_cuda_gemv(
            x,
            trellis,
            2.0,
            out_features=64,
            output_fp32=True,
            bank_ids=bank_ids,
            v2b2_p32_lr=True,
            bank_alt_id=3,
        )
        for _ in range(5)
    ]
    torch.cuda.synchronize()
    assert all(torch.equal(outputs[0], output) for output in outputs[1:])
    assert (outputs[0] - reference.cuda()).abs().max().item() <= 2e-3


def test_lr32_cuda_honors_explicit_split_count():
    x, trellis, bank_ids, reference = _lr_case(2.0, m=1, k=128, n=32, seed=20260835)
    actual = qvq_cuda_gemv(
        x,
        trellis,
        2.0,
        out_features=32,
        output_fp32=True,
        bank_ids=bank_ids,
        v2b2_p32_lr=True,
        bank_alt_id=3,
        lr_split_count=4,
    )
    assert (actual - reference.cuda()).abs().max().item() <= 2e-3


def test_lr32_cuda_uses_current_non_default_stream():
    x, trellis, bank_ids, reference = _lr_case(2.0, m=1, k=2048, n=256, seed=20260831)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = qvq_cuda_gemv(
            x,
            trellis,
            2.0,
            out_features=256,
            output_fp32=True,
            bank_ids=bank_ids,
            v2b2_p32_lr=True,
            bank_alt_id=3,
        )
        completion = torch.cuda.Event()
        completion.record(stream)
    completion.synchronize()
    assert (actual - reference.cuda()).abs().max().item() <= 2e-3


def test_lr32_cuda_supports_bfloat16_typed_output():
    x, trellis, bank_ids, reference = _lr_case(
        2.0, m=4, k=128, n=32, seed=20260832, dtype=torch.bfloat16
    )
    actual = qvq_cuda_gemv(
        x,
        trellis,
        2.0,
        out_features=32,
        output_fp32=False,
        bank_ids=bank_ids,
        v2b2_p32_lr=True,
        bank_alt_id=3,
    )
    assert actual.dtype == torch.bfloat16
    typed_reference = reference.cuda().to(actual.dtype).float()
    assert (actual.float() - typed_reference).abs().max().item() <= 2e-2


def test_lr32_cuda_rejects_split_count_before_integer_narrowing():
    x, trellis, bank_ids, _ = _lr_case(1.0, m=1, k=64, n=8, seed=20260833)
    with pytest.raises(ValueError, match=r"in \[1, 64\]"):
        qvq_cuda_gemv(
            x,
            trellis,
            1.0,
            out_features=8,
            output_fp32=True,
            bank_ids=bank_ids,
            v2b2_p32_lr=True,
            bank_alt_id=3,
            lr_split_count=2**32 + 1,
        )


def test_lr32_cuda_validates_split_count_for_empty_batches():
    x, trellis, bank_ids, _ = _lr_case(1.0, m=0, k=32, n=8, seed=20260834)
    with pytest.raises(ValueError, match=r"in \[1, min\(K/32, 64\)\]"):
        qvq_cuda_gemv(
            x,
            trellis,
            1.0,
            out_features=8,
            output_fp32=True,
            bank_ids=bank_ids,
            v2b2_p32_lr=True,
            bank_alt_id=3,
            lr_split_count=2,
        )

    with pytest.raises(ValueError, match="valid only for V2B2-P32-LR"):
        qvq_cuda_gemv(
            torch.zeros((0, 16), device="cuda", dtype=torch.float16),
            torch.zeros((1, 8), device="cuda", dtype=torch.int32),
            1.0,
            out_features=16,
            output_fp32=True,
            v2b2_p32=True,
            bank_ids=torch.zeros((1,), device="cuda", dtype=torch.uint8),
            bank_alt_id=3,
            lr_split_count=1,
        )


def test_lr32_cuda_rejects_row_grid_overflow_before_launch():
    # NVIDIA devices expose 65,535 blocks in grid Y; ROWS=32 above M=16.
    m = 65_535 * 32 + 1
    x = torch.empty((m, 32), device="cuda", dtype=torch.float16)
    trellis = torch.zeros((1, 8), device="cuda", dtype=torch.int32)
    bank_ids = torch.zeros((1,), device="cuda", dtype=torch.uint8)

    with pytest.raises(RuntimeError, match="CUDA grid-Y blocks"):
        qvq_cuda_gemv(
            x,
            trellis,
            1.0,
            out_features=8,
            output_fp32=True,
            bank_ids=bank_ids,
            v2b2_p32_lr=True,
            bank_alt_id=1,
        )


def test_lr32_cuda_rejects_tile_index_overflow_before_shape_check():
    x = torch.empty((0, 8192), device="cuda", dtype=torch.float16)
    trellis = torch.empty((0, 8), device="cuda", dtype=torch.int32)
    bank_ids = torch.empty((0,), device="cuda", dtype=torch.uint8)

    with pytest.raises(ValueError, match="tile count exceeds the int32 kernel index limit"):
        qvq_cuda_gemv(
            x,
            trellis,
            1.0,
            out_features=67_108_872,
            output_fp32=True,
            bank_ids=bank_ids,
            v2b2_p32_lr=True,
            bank_alt_id=1,
        )


def test_lr32_cuda_rejects_split_partition_overflow_before_shape_check():
    x = torch.empty((0, 2_147_483_616), device="cuda", dtype=torch.float16)
    trellis = torch.empty((0, 8), device="cuda", dtype=torch.int32)
    bank_ids = torch.empty((0,), device="cuda", dtype=torch.uint8)

    with pytest.raises(ValueError, match="split_count overflows the int32 kernel partition limit"):
        qvq_cuda_gemv(
            x,
            trellis,
            1.0,
            out_features=8,
            output_fp32=True,
            bank_ids=bank_ids,
            v2b2_p32_lr=True,
            bank_alt_id=1,
            lr_split_count=64,
        )


def test_lr32_cuda_rejects_non_lr_layouts():
    x = torch.zeros((1, 32), device="cuda", dtype=torch.float16)
    trellis = torch.zeros((1, 8), device="cuda", dtype=torch.int32)
    selectors = torch.zeros((1,), device="cuda", dtype=torch.uint8)

    with pytest.raises(ValueError, match="exactly one packed selector"):
        qvq_cuda_gemv(x, trellis, 1.0, out_features=8, v2b2_p32_lr=True)
    with pytest.raises(ValueError, match="W1 through W3.5"):
        qvq_cuda_gemv(
            x,
            trellis,
            4.0,
            out_features=8,
            bank_ids=selectors,
            v2b2_p32_lr=True,
            bank_alt_id=1,
        )
    with pytest.raises(ValueError, match="K32/N8"):
        qvq_cuda_gemv(
            torch.zeros((1, 48), device="cuda", dtype=torch.float16),
            trellis,
            1.0,
            out_features=8,
            bank_ids=selectors,
            v2b2_p32_lr=True,
            bank_alt_id=1,
        )
