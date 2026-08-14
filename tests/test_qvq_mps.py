# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0


from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from gptqmodel.nn_modules.qlinear.qvq import QVQLinear, _qvq_hadamard_fused
from gptqmodel.quantization.qvq import pack_qvq_bank_ids, reconstruct_qvq_inner_weight
from gptqmodel.quantization.qvq_codecs import pgc16_codebook
from gptqmodel.quantization.qvq_rates import qvq_transition_bits
from gptqmodel.utils.planar_packing import planar_pack_rows
from gptqmodel.utils.qvq_mps import (
    QVQ_MPS_BITS,
    _multirow_vector_width,
    _v4_row_tile,
    _v4_use_k64,
    _v4_use_k128,
    qvq_mps_gemv,
    qvq_mps_overlap_scores,
    qvq_mps_supported,
    qvq_mps_viterbi,
)

pytestmark = [
    pytest.mark.mps,
    pytest.mark.skipif(
        not qvq_mps_supported(), reason="requires runtime Metal shaders"
    ),
]


def _case(bits: float, m: int, *, k: int = 32, n: int = 32, vector_size: int = 2):
    transition_bits = qvq_transition_bits(bits, vector_size=vector_size)
    generator = torch.Generator().manual_seed(8100 + transition_bits * 10 + m + vector_size)
    tiles = (k // 16) * (n // 16)
    edges = torch.randint(
        0,
        1 << transition_bits,
        (256 // vector_size, tiles),
        generator=generator,
        dtype=torch.int32,
    )
    trellis = planar_pack_rows(edges, transition_bits).T.contiguous()
    x = torch.randn((m, k), generator=generator).half()
    inner = reconstruct_qvq_inner_weight(
        trellis,
        bits=bits,
        vector_size=vector_size,
        in_features=k,
        out_features=n,
    )
    reference = (x.float() @ inner.float()).half()
    return (x.to("mps"), trellis.to("mps")), reference


def _banked_case(bits: float, m: int, *, k: int = 32, n: int = 32):
    transition_bits = qvq_transition_bits(bits, vector_size=4)
    generator = torch.Generator().manual_seed(18100 + transition_bits * 10 + m + k + n)
    tile_count = (k // 16) * (n // 16)
    edges = torch.randint(0, 1 << transition_bits, (64, tile_count), generator=generator, dtype=torch.int32)
    trellis = planar_pack_rows(edges, transition_bits).T.contiguous()
    dense_bank_ids = torch.arange(tile_count, dtype=torch.uint8).remainder_(4)
    bank_ids = pack_qvq_bank_ids(dense_bank_ids)
    x = torch.randn((m, k), generator=generator).half()
    inner = reconstruct_qvq_inner_weight(
        trellis,
        bits=bits,
        vector_size=4,
        in_features=k,
        out_features=n,
        bank_ids=bank_ids,
    )
    reference = (x.float() @ inner.float()).half()
    return (x.to("mps"), trellis.to("mps"), bank_ids.to("mps")), reference


def _assert_dense_accuracy_metrics(actual: torch.Tensor, reference: torch.Tensor) -> None:
    actual_f32 = actual.float()
    reference_f32 = reference.float()
    error = actual_f32 - reference_f32
    relative_l2 = error.norm() / reference_f32.norm().clamp_min(torch.finfo(torch.float32).tiny)
    cosine = F.cosine_similarity(actual_f32, reference_f32, dim=-1)
    forward_kld = F.kl_div(
        actual_f32.log_softmax(dim=-1),
        reference_f32.softmax(dim=-1),
        reduction="batchmean",
    )

    assert error.square().mean().item() < 1e-3
    assert relative_l2.item() < 2e-3
    assert torch.all(cosine > 0.99999)
    assert forward_kld.item() < 2e-5
    assert torch.equal(actual_f32.argmax(dim=-1), reference_f32.argmax(dim=-1))
    assert torch.equal(actual_f32.topk(5, dim=-1).indices, reference_f32.topk(5, dim=-1).indices)


def test_qvq_mps_viterbi_rejects_invalid_numeric_inputs():
    sequences = torch.zeros((1, 1, 2), dtype=torch.float32, device="mps")
    codebook = pgc16_codebook(dtype=torch.float32).to("mps")

    bad_sequences = sequences.clone()
    bad_sequences[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="sequences and codebook must contain only finite values"):
        qvq_mps_viterbi(bad_sequences, codebook, 1)

    bad_codebook = codebook.clone()
    bad_codebook[0, 0] = torch.inf
    with pytest.raises(ValueError, match="sequences and codebook must contain only finite values"):
        qvq_mps_viterbi(sequences, bad_codebook, 1)

    for value in (-1.0, float("nan")):
        step_weights = torch.full((1, 1), value, dtype=torch.float32, device="mps")
        with pytest.raises(ValueError, match="step weights must be finite and nonnegative"):
            qvq_mps_viterbi(sequences, codebook, 1, step_weights=step_weights)

    overlap_limit = 1 << (16 - qvq_transition_bits(1))
    for value in (-1, overlap_limit):
        overlap = torch.tensor([value], dtype=torch.int64, device="mps")
        with pytest.raises(ValueError, match=rf"overlap must be in \[0, {overlap_limit - 1}\]"):
            qvq_mps_viterbi(sequences, codebook, 1, overlap=overlap)


def test_qvq_mps_overlap_scores_rejects_invalid_numeric_inputs():
    sequences = torch.zeros((1, 2, 2), dtype=torch.float32, device="mps")
    codebook = pgc16_codebook(dtype=torch.float32).to("mps")

    bad_sequences = sequences.clone()
    bad_sequences[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="sequences and codebook must contain only finite values"):
        qvq_mps_overlap_scores(bad_sequences, codebook, 1, boundary=0)

    bad_codebook = codebook.clone()
    bad_codebook[0, 0] = torch.inf
    with pytest.raises(ValueError, match="sequences and codebook must contain only finite values"):
        qvq_mps_overlap_scores(sequences, bad_codebook, 1, boundary=0)

    for value in (-1.0, float("nan")):
        step_weights = torch.full((1, 2), value, dtype=torch.float32, device="mps")
        with pytest.raises(ValueError, match="step weights must be finite and nonnegative"):
            qvq_mps_overlap_scores(sequences, codebook, 1, boundary=0, step_weights=step_weights)

    empty_sequences = torch.empty((0, 2, 2), dtype=torch.float32, device="mps")
    with pytest.raises(ValueError, match="requires a nonempty batch and sequence"):
        qvq_mps_overlap_scores(empty_sequences, codebook, 1, boundary=0)


@pytest.mark.parametrize("bits", QVQ_MPS_BITS)
@pytest.mark.parametrize("m", (1, 4))
def test_qvq_mps_all_planar_bits_match_dense_reference(bits, m):
    operands, reference = _case(bits, m)

    actual = qvq_mps_gemv(*operands, bits, out_features=reference.shape[1]).cpu()

    torch.testing.assert_close(actual, reference, rtol=1e-3, atol=1e-3)
    _assert_dense_accuracy_metrics(actual, reference)


@pytest.mark.parametrize("bits", QVQ_MPS_BITS)
def test_qvq_mps_fp32_output_all_planar_bits_preserves_fp16_result(bits):
    operands, reference = _case(bits, 1)

    actual = qvq_mps_gemv(
        *operands, bits, out_features=reference.shape[1], output_fp32=True
    ).cpu()

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.half(), reference, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("bits", (1, 1.5, 2, 2.5, 3, 3.5, 4))
@pytest.mark.parametrize("m", (1, 4, 17))
def test_qvq_v4_mps_matches_dense_reference(bits, m):
    operands, reference = _case(bits, m, vector_size=4)

    actual = qvq_mps_gemv(
        *operands,
        bits,
        out_features=reference.shape[1],
        vector_size=4,
    ).cpu()

    torch.testing.assert_close(actual, reference, rtol=1e-3, atol=1e-3)
    _assert_dense_accuracy_metrics(actual, reference)


@pytest.mark.parametrize("bits", (1, 1.5, 2, 2.5, 3, 3.5, 4))
@pytest.mark.parametrize("m", (1, 4, 17))
def test_qvq_v4_banked_mps_matches_dense_reference(bits, m):
    (x, trellis, bank_ids), reference = _banked_case(bits, m)

    actual = qvq_mps_gemv(
        x,
        trellis,
        bits,
        out_features=reference.shape[1],
        vector_size=4,
        bank_ids=bank_ids,
    ).cpu()

    torch.testing.assert_close(actual, reference, rtol=1e-3, atol=1e-3)
    _assert_dense_accuracy_metrics(actual, reference)


@pytest.mark.parametrize("bits", (1, 1.5, 2, 2.5, 3, 3.5, 4))
def test_qvq_v4_banked_mps_fp32_output_preserves_fp16_result(bits):
    (x, trellis, bank_ids), reference = _banked_case(bits, 1)

    actual = qvq_mps_gemv(
        x,
        trellis,
        bits,
        out_features=reference.shape[1],
        vector_size=4,
        bank_ids=bank_ids,
        output_fp32=True,
    ).cpu()

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.half(), reference, rtol=1e-3, atol=1e-3)


def test_qvq_v4_linear_mps_lifecycle_preserves_vector_size():
    (x, trellis), reference = _case(2.5, 4, vector_size=4)
    layer = QVQLinear(
        bits=2.5,
        in_features=x.shape[1],
        out_features=reference.shape[1],
        name="proj",
        vector_size=4,
        tensors={
            "trellis": trellis,
            "SU": torch.ones(x.shape[1], dtype=torch.float32, device="mps"),
            "SV": torch.ones(reference.shape[1], dtype=torch.float32, device="mps"),
        },
    ).eval()
    layer.post_init()

    with patch("gptqmodel.utils.qvq_mps.qvq_mps_gemv", wraps=qvq_mps_gemv) as native_gemv:
        actual = layer._inner_forward(x).cpu()

    native_gemv.assert_called_once()
    assert native_gemv.call_args.kwargs["vector_size"] == 4
    assert native_gemv.call_args.kwargs["output_fp32"] is True

    torch.testing.assert_close(actual, reference.float(), rtol=1e-3, atol=1e-3)
    _assert_dense_accuracy_metrics(actual, reference)


def test_qvq_v4_banked_linear_mps_lifecycle_uses_native_kernel():
    (x, trellis, bank_ids), reference = _banked_case(2.5, 4)
    layer = QVQLinear(
        bits=2.5,
        in_features=x.shape[1],
        out_features=reference.shape[1],
        name="banked_proj",
        vector_size=4,
        bank_count=4,
        tensors={
            "trellis": trellis,
            "SU": torch.ones(x.shape[1], dtype=torch.float32, device="mps"),
            "SV": torch.ones(reference.shape[1], dtype=torch.float32, device="mps"),
            "bank_ids": bank_ids,
        },
    ).eval()
    layer.post_init()

    with patch("gptqmodel.utils.qvq_mps.qvq_mps_gemv", wraps=qvq_mps_gemv) as native_gemv:
        actual = layer._inner_forward(x).cpu()

    native_gemv.assert_called_once()
    assert torch.equal(native_gemv.call_args.kwargs["bank_ids"], bank_ids)
    assert native_gemv.call_args.kwargs["output_fp32"] is True
    torch.testing.assert_close(actual, reference.float(), rtol=1e-3, atol=1e-3)
    _assert_dense_accuracy_metrics(actual, reference)


def test_qvq_v4_banked_linear_mps_rebuilds_selectors_after_mutation_or_replacement():
    (x, trellis, bank_ids), reference = _banked_case(2, 4)
    layer = QVQLinear(
        bits=2,
        in_features=x.shape[1],
        out_features=reference.shape[1],
        name="mutable_banked_proj",
        vector_size=4,
        bank_count=4,
        tensors={
            "trellis": trellis,
            "SU": torch.ones(x.shape[1], dtype=torch.float32, device="mps"),
            "SV": torch.ones(reference.shape[1], dtype=torch.float32, device="mps"),
            "bank_ids": bank_ids,
        },
    ).eval()
    layer.post_init()

    initial = layer._inner_forward(x)
    layer.bank_ids.fill_(0x55)  # Four packed bank-1 selectors.
    mutated_reference = layer._reference_inner_forward(x)
    mutated = layer._inner_forward(x)
    layer.bank_ids = torch.full_like(layer.bank_ids, 0xAA)  # Four packed bank-2 selectors.
    replaced_reference = layer._reference_inner_forward(x)
    replaced = layer._inner_forward(x)

    assert not torch.equal(initial.cpu(), mutated.cpu())
    assert not torch.equal(mutated.cpu(), replaced.cpu())
    torch.testing.assert_close(mutated, mutated_reference.float(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(replaced, replaced_reference.float(), rtol=1e-3, atol=1e-3)


def test_qvq_mps_fp16_hadamard_normalizes_before_narrow_butterfly_overflow():
    x = torch.zeros((1, 32), device="mps", dtype=torch.float16)
    x[0, 0] = 60000

    actual = _qvq_hadamard_fused(x)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.cpu(), torch.full((1, 32), 10608.0, dtype=torch.float16), rtol=0, atol=8)


def test_qvq_v4_banked_mps_full_layer_preserves_fp32_inner_range():
    width = 2048
    tile_count = (width // 16) ** 2
    trellis = torch.zeros((tile_count, 16), dtype=torch.int32, device="mps")
    bank_ids = torch.zeros((tile_count + 3) // 4, dtype=torch.uint8, device="mps")
    sv = torch.full((width,), 1 / 4096, dtype=torch.float32)
    layer = QVQLinear(
        bits=2,
        in_features=width,
        out_features=width,
        name="range_safe_banked_proj",
        vector_size=4,
        bank_count=4,
        tensors={
            "trellis": trellis,
            "SU": torch.ones(width, dtype=torch.float32, device="mps"),
            "SV": sv.to("mps"),
            "bank_ids": bank_ids,
        },
    ).eval()
    layer.post_init()
    x = torch.zeros((1, width), dtype=torch.float16)
    x[0, 0] = 60000

    transformed = _qvq_hadamard_fused(x.to("mps"))
    inner = layer._inner_forward(transformed)
    actual = layer(x.to("mps")).cpu().float()

    assert inner.dtype == torch.float32
    assert inner.abs().max() > torch.finfo(torch.float16).max
    assert torch.isfinite(actual).all()
    inner_weight = layer.get_inner_weight_tensor(dtype=torch.float32).cpu()
    expected = _qvq_hadamard_fused(transformed.cpu().float() @ inner_weight) * sv
    relative_l2 = (actual - expected).norm() / expected.norm()
    assert relative_l2 < 1e-3


def test_qvq_mps_full_layer_rescales_coherent_input_transform_range():
    width = 2048
    trellis = torch.zeros(((width // 16) ** 2, 16), dtype=torch.int32)
    sv = torch.full((width,), 1e-4, dtype=torch.float32)
    layer = QVQLinear(
        bits=2,
        in_features=width,
        out_features=width,
        name="coherent_range_proj",
        tensors={
            "trellis": trellis.to("mps"),
            "SU": torch.ones(width, dtype=torch.float32, device="mps"),
            "SV": sv.to("mps"),
        },
    ).eval()
    layer.post_init()
    x = torch.full((1, width), 60000, dtype=torch.float16)

    transformed = _qvq_hadamard_fused(x.float())
    inner_weight = layer.get_inner_weight_tensor(dtype=torch.float32).cpu()
    expected = _qvq_hadamard_fused(transformed @ inner_weight) * sv
    actual = layer(x.to("mps")).cpu().float()

    assert transformed.abs().max() > torch.finfo(torch.float16).max
    assert torch.isfinite(expected).all()
    assert torch.isfinite(actual).all()
    relative_l2 = (actual - expected).norm() / expected.norm()
    assert relative_l2 < 5e-4


@pytest.mark.parametrize("bits", (1, 2.5, 4))
def test_qvq_v4_banked_mps_k128_matches_dense_reference(bits):
    (x, trellis, bank_ids), reference = _banked_case(bits, 17, k=144, n=8192)
    actual = qvq_mps_gemv(
        x,
        trellis,
        bits,
        out_features=reference.shape[1],
        vector_size=4,
        bank_ids=bank_ids,
    ).cpu()
    torch.testing.assert_close(actual, reference, rtol=1e-3, atol=1e-3)
    _assert_dense_accuracy_metrics(actual, reference)


def test_qvq_v4_banked_mps_k64_matches_dense_reference():
    (x, trellis, bank_ids), reference = _banked_case(4, 24, k=64, n=8192)
    actual = qvq_mps_gemv(
        x,
        trellis,
        4,
        out_features=reference.shape[1],
        vector_size=4,
        bank_ids=bank_ids,
    ).cpu()
    torch.testing.assert_close(actual, reference, rtol=1e-3, atol=1e-3)
    _assert_dense_accuracy_metrics(actual, reference)


@pytest.mark.parametrize("bits", (1, 2.5, 4))
def test_qvq_v4_mps_repeated_launches_are_deterministic(bits):
    operands, reference = _case(bits, 3, vector_size=4)
    outputs = [
        qvq_mps_gemv(
            *operands,
            bits,
            out_features=reference.shape[1],
            vector_size=4,
        ).cpu()
        for _ in range(10)
    ]

    assert all(torch.equal(outputs[0], output) for output in outputs[1:])


@pytest.mark.parametrize(
    ("transition_bits", "m", "k", "n", "expected"),
    (
        (8, 4, 2048, 2048, 8), (8, 4, 2048, 8192, 4), (8, 8, 2048, 2048, 4),
        (8, 16, 2048, 2048, 16), (8, 16, 2048, 8192, 8), (8, 17, 2048, 2048, 4),
        (8, 24, 2048, 2048, 8), (8, 32, 2048, 2048, 16), (8, 32, 2048, 8192, 8),
        (4, 16, 8192, 2048, 8), (16, 24, 8192, 2048, 4),
        (16, 16, 2048, 8192, 16), (16, 32, 2048, 8192, 4), (4, 32, 8192, 8192, 4),
    ),
)
def test_qvq_v4_mps_row_tile_dispatch(transition_bits, m, k, n, expected):
    assert _v4_row_tile(transition_bits, m, k, n) == expected


@pytest.mark.parametrize(
    ("m", "k", "n", "expected"),
    ((4, 8192, 2048, False), (16, 4096, 4096, False),
     (16, 8192, 2048, False), (17, 8192, 2048, False),
     (17, 2048, 8192, True), (24, 8192, 2048, True), (32, 8192, 8192, True)),
)
def test_qvq_v4_mps_k64_dispatch(m, k, n, expected):
    assert _v4_use_k64(m, k, n) is expected


@pytest.mark.parametrize(
    ("transition_bits", "m", "k", "n", "expected"),
    ((4, 16, 2048, 8192, True), (16, 16, 2048, 8192, False),
     (16, 17, 2048, 8192, True), (16, 24, 8192, 2048, True),
     (8, 16, 8192, 8192, False), (8, 32, 8192, 8192, False)),
)
def test_qvq_v4_mps_k128_dispatch(transition_bits, m, k, n, expected):
    assert _v4_use_k128(transition_bits, m, k, n) is expected


@pytest.mark.parametrize("bits", (1, 1.5, 2, 2.5, 3, 3.5, 4))
@pytest.mark.parametrize("k", (64, 80))
def test_qvq_v4_mps_k64_matches_dense_reference(bits, k):
    operands, reference = _case(bits, 17, k=k, n=8192, vector_size=4)

    actual = qvq_mps_gemv(
        *operands, bits, out_features=reference.shape[1], vector_size=4,
    ).cpu()

    torch.testing.assert_close(actual, reference, rtol=1e-3, atol=1e-3)
    _assert_dense_accuracy_metrics(actual, reference)


@pytest.mark.parametrize("bits", (1, 1.5, 2, 2.5, 3, 3.5, 4))
def test_qvq_v4_mps_k128_tail_matches_dense_reference(bits):
    operands, reference = _case(bits, 17, k=144, n=8192, vector_size=4)

    actual = qvq_mps_gemv(
        *operands, bits, out_features=reference.shape[1], vector_size=4,
    ).cpu()

    torch.testing.assert_close(actual, reference, rtol=1e-3, atol=1e-3)
    _assert_dense_accuracy_metrics(actual, reference)


@pytest.mark.parametrize("bits", QVQ_MPS_BITS)
def test_qvq_mps_repeated_launches_are_deterministic(bits):
    operands, reference = _case(bits, 3)
    outputs = [
        qvq_mps_gemv(*operands, bits, out_features=reference.shape[1]).cpu()
        for _ in range(3)
    ]

    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(outputs[1], outputs[2])






@pytest.mark.parametrize(
    ("transition_bits", "m", "k", "n", "expected"),
    (
        (16, 15, 2048, 8192, 8),
        (16, 16, 8192, 2048, 8),
        (16, 14, 8192, 2048, 4),
        (16, 16, 4096, 4096, 4),
        (14, 16, 8192, 2048, 4),
        (16, 17, 8192, 2048, 4),
    ),
)
def test_qvq_mps_multirow_vector_width_dispatch(transition_bits, m, k, n, expected):
    assert _multirow_vector_width(transition_bits, m, k, n) == expected


def test_qvq_mps_w8_n8_dispatch_matches_dense_reference():
    operands, reference = _case(8, 15, k=8192, n=16)

    actual = qvq_mps_gemv(*operands, 8, out_features=reference.shape[1]).cpu()

    torch.testing.assert_close(actual, reference, rtol=1e-3, atol=1e-3)




def test_qvq_mps_empty_batch_and_contract_guards():
    x = torch.zeros((0, 16), device="mps", dtype=torch.float16)
    trellis = torch.zeros((1, 16), device="mps", dtype=torch.int32)
    assert qvq_mps_gemv(x, trellis, 2, out_features=16).shape == (0, 16)
    assert qvq_mps_gemv(x, trellis, 2, out_features=16, output_fp32=True).dtype == torch.float32
    with pytest.raises(TypeError, match="out_features must be an integer"):
        qvq_mps_gemv(x, trellis, 2, out_features=16.5)
    with pytest.raises(TypeError, match="output_fp32 must be a bool"):
        qvq_mps_gemv(x, trellis, 2, out_features=16, output_fp32=1)
    with pytest.raises(ValueError, match="pgc16-v1"):
        qvq_mps_gemv(
            x,
            trellis,
            2,
            out_features=16,
            codebook_version="unsupported",
        )
    with pytest.raises(ValueError, match="W1 through W4"):
        qvq_mps_gemv(x, trellis, 4.5, out_features=16, vector_size=4)
    with pytest.raises(ValueError, match="vector_size=4"):
        qvq_mps_gemv(
            x,
            trellis,
            2,
            out_features=16,
            bank_ids=torch.zeros(1, device="mps", dtype=torch.uint8),
        )
    with pytest.raises(TypeError, match="packed uint8"):
        qvq_mps_gemv(
            x,
            trellis,
            2,
            out_features=16,
            vector_size=4,
            bank_ids=torch.zeros(1, device="mps", dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="packed shape"):
        qvq_mps_gemv(
            x,
            trellis,
            2,
            out_features=16,
            vector_size=4,
            bank_ids=torch.zeros(2, device="mps", dtype=torch.uint8),
        )

    cases = [
        ((x, trellis, 9), {"out_features": 16}, ValueError, "rate"),
        ((x[:, :, None], trellis, 2), {"out_features": 16}, ValueError, "2D"),
        ((x.cpu(), trellis, 2), {"out_features": 16}, ValueError, "MPS device"),
        (
            (
                torch.zeros((16, 2), device="mps", dtype=torch.float16).T,
                trellis,
                2,
            ),
            {"out_features": 16},
            ValueError,
            "contiguous",
        ),
        (
            (torch.zeros((1, 15), device="mps", dtype=torch.float16), trellis, 2),
            {"out_features": 16},
            ValueError,
            "divisible",
        ),
        (
            (
                torch.zeros((1, 16), device="mps", dtype=torch.float16),
                trellis[:, :8],
                2,
            ),
            {"out_features": 16},
            ValueError,
            "trellis",
        ),
    ]
    for args, kwargs, error, message in cases:
        with pytest.raises(error, match=message):
            qvq_mps_gemv(*args, **kwargs)
