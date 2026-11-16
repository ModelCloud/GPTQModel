# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""R9700 NF12 embedding: exact row bits, placement, and packed storage."""

from __future__ import annotations

import gc

import pytest
import torch

from gptqmodel.utils import auto_dtype_cast, nf12_codec, nf12_hip
from tests.nf12_embedding_cases import (
    OBSERVED_LITE_PREFILL_CASES,
    assert_embedding_lookup_exact,
    observed_lite_prefill_case,
    small_embedding_case,
)


def _hip_device_or_skip(index: int = 0) -> torch.device:
    if torch.version.hip is None or torch.cuda.device_count() <= index:
        pytest.skip("requires a ROCm GPU")
    device = torch.device("cuda", index)
    if not nf12_hip.nf12_hip_supported(device):
        pytest.skip("requires an RDNA4 GPU")
    with torch.cuda.device(device):
        if not nf12_hip.prewarm_nf12_hip_extension():
            pytest.fail(f"NF12 HIP extension unavailable: {nf12_hip.last_error()}")
    return device


def _oracle(weight: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Ordinary dense row indexing; invalid vocabulary IDs become +0.0."""
    ids = ids.cpu()
    expected = torch.zeros((*ids.shape, weight.shape[1]), dtype=torch.bfloat16)
    valid = (ids >= 0) & (ids < weight.shape[0])
    expected[valid] = weight[ids[valid].long()]
    return expected


def test_hip_embedding_shares_cuda_small_case():
    device = _hip_device_or_skip()
    weight, ids = small_embedding_case(device)
    packed, escapes, base = nf12_hip.encode(weight)
    assert_embedding_lookup_exact(
        lambda values: nf12_hip.embedding(values, packed, escapes, base),
        weight,
        ids,
    )


@pytest.mark.parametrize("m,id_dtype", OBSERVED_LITE_PREFILL_CASES)
def test_hip_embedding_shares_cuda_observed_lite_prefill_cases(m, id_dtype):
    device = _hip_device_or_skip()
    weight, ids = observed_lite_prefill_case(device, m, id_dtype)
    packed, escapes, base = nf12_hip.encode(weight)
    assert_embedding_lookup_exact(
        lambda values: nf12_hip.embedding(values, packed, escapes, base),
        weight,
        ids,
        check_invalid=m == 512,
    )
    assert packed.nbytes + escapes.nbytes + 1 < weight.nbytes


@pytest.mark.parametrize("shape", [(16, 64), (32, 128), (48, 2048)])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_hip_embedding_matches_dense_bits(shape, id_dtype):
    device = _hip_device_or_skip()
    n, k = shape
    generator = torch.Generator().manual_seed(721 + n + k)
    weight = (torch.randn(shape, generator=generator) * 0.02).bfloat16()
    # Signed zeros, subnormals, infinities and NaN payloads must survive a
    # gather exactly; float-value comparison would hide some of these cases.
    edges = (
        torch.tensor(
            [0x0000, 0x8000, 0x0001, 0x8001, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1],
            dtype=torch.int32,
        )
        .to(torch.int16)
        .view(torch.bfloat16)
    )
    weight[7, : edges.numel()] = edges
    packed, escapes, base = nf12_codec.encode(weight)
    packed, escapes = packed.to(device), escapes.to(device)
    ids = torch.tensor(
        [[0, 7, 8, 15, n - 1, n], [-1, 16, 7, 0, n + 17, 1]],
        dtype=id_dtype,
        device=device,
    )
    if id_dtype is torch.int64:
        ids[0, 0] = 2**32 + 1
        ids[1, 0] = -(2**32)
    actual = nf12_hip.embedding(ids, packed, escapes, base)
    assert actual.dtype is torch.bfloat16
    assert tuple(actual.shape) == (*ids.shape, k)
    assert torch.equal(actual.cpu().view(torch.int16), _oracle(weight, ids).view(torch.int16))


def test_hip_rowmajor_embedding_matches_dense_bits_with_collisions():
    device = _hip_device_or_skip()
    bits = torch.full((16, 128), 0x3F00, dtype=torch.int16)
    bits[8:, :64] = -16640
    bits[:8, 64:] = -16640
    bits[0, 64] = 0x7FC0  # NaN payload
    bits[8, 64] = -32768  # -0.0
    bits[1, 65] = 1  # subnormal
    bits[9, 65] = -1
    weight = bits.view(torch.bfloat16)
    packed, escapes, base = nf12_codec.encode(weight, exp_base=63)
    row_packed, row_escapes, original, collisions = nf12_hip.rowmajor_pack(packed.to(device), escapes.to(device), base)

    assert original == 2
    assert collisions == 254
    assert row_packed.shape == packed.shape
    assert row_escapes.shape == (2 * original + collisions, 4)
    ids = torch.tensor([[-1, 0, 1], [7, 8, 15]], dtype=torch.int64, device=device)
    actual = nf12_hip.rowmajor_embedding(ids, row_packed, row_escapes, base)
    expected = torch.zeros_like(actual)
    valid = (ids >= 0) & (ids < weight.shape[0])
    expected[valid] = weight.to(device)[ids[valid].long()]
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))

    shaped = (
        torch.arange(24, device=device, dtype=torch.int32).remainder(weight.shape[0]).reshape(2, 3, 4).transpose(0, 2)
    )
    assert not shaped.is_contiguous()
    actual = nf12_hip.rowmajor_embedding(shaped, row_packed, row_escapes, base)
    assert tuple(actual.shape) == (*shaped.shape, weight.shape[1])
    assert torch.equal(
        actual.view(torch.int16),
        torch.nn.functional.embedding(shaped, weight.to(device)).view(torch.int16),
    )

    empty = nf12_hip.rowmajor_embedding(shaped[:0], row_packed, row_escapes, base)
    assert tuple(empty.shape) == (0, *shaped.shape[1:], 128)


def test_hip_rowmajor_one_id_covers_full_llama_embedding_row():
    device = _hip_device_or_skip()
    weight, ids = observed_lite_prefill_case(device, 1, torch.int64)
    packed, escapes, base = nf12_codec.encode(weight)
    row_packed, row_escapes, _, _ = nf12_hip.rowmajor_pack(packed.to(device), escapes.to(device), base)

    actual = nf12_hip.rowmajor_embedding(ids, row_packed, row_escapes, base)
    expected = torch.nn.functional.embedding(ids, weight)

    assert weight.shape == (512, 5120)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_hip_auto_embedding_retains_only_row_major_storage(monkeypatch):
    device = _hip_device_or_skip()
    monkeypatch.setattr(auto_dtype_cast, "nf12_supported_device", lambda _device: True)
    source = torch.nn.Embedding(32, 128, dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        source.weight.copy_(torch.randn_like(source.weight) * 0.02)
        source.weight[7, 0] = -0.0

    spec = auto_dtype_cast.eligible_source_module(source, device)
    packed = auto_dtype_cast.cast_source_module(source, spec)
    ids = torch.tensor([[-1, 0, 7], [15, 31, 32]], device=device)
    actual = packed(ids)
    expected = torch.zeros_like(actual)
    valid = (ids >= 0) & (ids < source.num_embeddings)
    expected[valid] = source(ids[valid])

    assert packed.row_major
    assert {name for name, _ in packed.named_buffers()} == {
        "nf12_row_packed",
        "nf12_row_escapes",
        "nf12_exp_base",
    }
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    restored = auto_dtype_cast.restore_source_module(packed, spec, device=device)
    assert torch.equal(
        restored.weight.view(torch.int16),
        source.weight.view(torch.int16),
    )


def test_hip_embedding_noncontiguous_ids_and_empty_input():
    device = _hip_device_or_skip()
    weight = torch.arange(32 * 128, dtype=torch.float32).reshape(32, 128).sin().bfloat16()
    packed, escapes, base = nf12_codec.encode(weight)
    packed, escapes = packed.to(device), escapes.to(device)
    ids = torch.tensor([[1, 10, 31], [8, 7, -2]], device=device, dtype=torch.int64).T
    assert not ids.is_contiguous()
    actual = nf12_hip.embedding(ids, packed, escapes, base)
    assert torch.equal(actual.cpu().view(torch.int16), _oracle(weight, ids).view(torch.int16))
    ids_3d = (
        torch.arange(24, device=device, dtype=torch.int64).remainder(weight.shape[0]).reshape(2, 3, 4).transpose(0, 2)
    )
    assert not ids_3d.is_contiguous()
    actual_3d = nf12_hip.embedding(ids_3d, packed, escapes, base)
    assert tuple(actual_3d.shape) == (*ids_3d.shape, weight.shape[1])
    assert torch.equal(actual_3d.cpu().view(torch.int16), _oracle(weight, ids_3d).view(torch.int16))
    scalar = nf12_hip.embedding(ids[0, 0], packed, escapes, base)
    assert tuple(scalar.shape) == (128,)
    assert torch.equal(scalar.cpu().view(torch.int16), weight[int(ids[0, 0])].view(torch.int16))
    empty = nf12_hip.embedding(ids[:0], packed, escapes, base)
    assert tuple(empty.shape) == (0, 2, 128)


def test_hip_embedding_uses_packed_device_not_current_device():
    target = _hip_device_or_skip(1)
    other = torch.device("cuda", 0)
    weight = torch.arange(32 * 128, dtype=torch.float32).reshape(32, 128).cos().bfloat16()
    packed, escapes, base = nf12_codec.encode(weight)
    packed, escapes = packed.to(target), escapes.to(target)
    ids = torch.tensor([1, 17, 31], device=target, dtype=torch.int32)
    stream = torch.cuda.Stream(device=target)
    with torch.cuda.device(target), torch.cuda.stream(stream):
        with torch.cuda.device(other):
            actual = nf12_hip.embedding(ids, packed, escapes, base)
            assert torch.cuda.current_device() == other.index
    stream.synchronize()
    assert actual.device == target
    assert torch.equal(actual.cpu().view(torch.int16), _oracle(weight, ids).view(torch.int16))


def test_hip_rowmajor_embedding_uses_packed_device_and_current_stream():
    target = _hip_device_or_skip(1)
    other = torch.device("cuda", 0)
    weight = torch.arange(32 * 128, dtype=torch.float32).reshape(32, 128).cos().bfloat16()
    packed, escapes, base = nf12_codec.encode(weight)
    row_packed, row_escapes, _, _ = nf12_hip.rowmajor_pack(packed.to(target), escapes.to(target), base)
    ids = torch.tensor([1, 17, 31], device=target, dtype=torch.int32)
    stream = torch.cuda.Stream(device=target)
    with torch.cuda.device(target), torch.cuda.stream(stream):
        with torch.cuda.device(other):
            actual = nf12_hip.rowmajor_embedding(ids, row_packed, row_escapes, base)
            assert torch.cuda.current_device() == other.index
    stream.synchronize()
    assert actual.device == target
    assert torch.equal(
        actual.cpu().view(torch.int16),
        _oracle(weight, ids).view(torch.int16),
    )


def test_hip_embedding_retains_no_dense_weight_between_calls():
    device = _hip_device_or_skip()
    weight = (torch.randn(64, 512, generator=torch.Generator().manual_seed(8)) * 0.02).bfloat16()
    packed, escapes, base = nf12_codec.encode(weight)
    packed, escapes = packed.to(device), escapes.to(device)
    ids = torch.tensor([7, 17, 63], device=device, dtype=torch.int64)
    expected = _oracle(weight, ids).view(torch.int16)
    packed_before = packed.clone()
    escapes_before = escapes.clone()

    warm = nf12_hip.embedding(ids, packed, escapes, base)
    del warm
    torch.cuda.synchronize(device)
    gc.collect()
    baseline = torch.cuda.memory_allocated(device)
    for _ in range(4):
        actual = nf12_hip.embedding(ids, packed, escapes, base)
        assert torch.equal(actual.cpu().view(torch.int16), expected)
        del actual
        torch.cuda.synchronize(device)
        gc.collect()
        # Allocator-reserved bytes are irrelevant; live allocation must not
        # climb by a reconstructed [N, K] BF16 matrix after each call.
        assert torch.cuda.memory_allocated(device) == baseline
    assert torch.equal(packed, packed_before)
    assert torch.equal(escapes, escapes_before)


def test_hip_embedding_reads_mutated_packed_weight_each_call():
    device = _hip_device_or_skip()
    original = torch.full((16, 64), -0.015625, dtype=torch.bfloat16)
    replacement = torch.full((16, 64), -0.03125, dtype=torch.bfloat16)
    first, first_escapes, base = nf12_codec.encode(original, exp_base=60)
    second, second_escapes, second_base = nf12_codec.encode(replacement, exp_base=60)
    assert base == second_base and first_escapes.shape == second_escapes.shape
    packed, escapes = first.to(device), first_escapes.to(device)
    ids = torch.tensor([0, 15], device=device, dtype=torch.int64)
    assert torch.equal(nf12_hip.embedding(ids, packed, escapes, base), original[ids.cpu()].to(device))
    packed.copy_(second.to(device))
    escapes.copy_(second_escapes.to(device))
    actual = nf12_hip.embedding(ids, packed, escapes, base)
    assert torch.equal(actual.view(torch.int16), replacement[ids.cpu()].to(device).view(torch.int16))


def test_hip_embedding_rejects_invalid_inputs():
    device = _hip_device_or_skip()
    weight = torch.zeros((16, 64), dtype=torch.bfloat16)
    packed, escapes, base = nf12_codec.encode(weight)
    packed, escapes = packed.to(device), escapes.to(device)
    ids = torch.tensor([0, 1], device=device)
    with pytest.raises(RuntimeError, match="int32 or int64"):
        nf12_hip.embedding(ids.float(), packed, escapes, base)
    with pytest.raises(RuntimeError, match="share a CUDA/HIP device"):
        nf12_hip.embedding(ids.cpu(), packed, escapes, base)
    with pytest.raises(RuntimeError, match="exp_base"):
        nf12_hip.embedding(ids, packed, escapes, 121)
