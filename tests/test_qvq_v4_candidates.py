import pytest
import torch

from gptqmodel.quantization.qvq import QVQQuantizationTelemetry, quantize_qvq_linear
from gptqmodel.quantization.qvq_v4_candidates import (
    V4Candidate,
    decode_v4_candidate_states,
    score_v4_candidate,
    select_v4_candidate,
    v4_candidate_codebook,
)


def test_candidate_decoder_is_explicitly_not_the_production_mapping():
    states = torch.tensor([0, 1, 0xA5A5, 0xFFFF], dtype=torch.int64)
    canonical = decode_v4_candidate_states(states, xor_mask=0xA5A5)
    alternate = decode_v4_candidate_states(states, xor_mask=0x5A5A)
    assert canonical.shape == alternate.shape == (4, 4)
    assert not torch.equal(canonical, alternate)


@pytest.mark.parametrize("bad", [-1, 1 << 16, True, "0"])
def test_candidate_rejects_invalid_masks(bad):
    with pytest.raises((TypeError, ValueError)):
        V4Candidate("bad", bad)


def test_candidate_score_and_selection_are_finite_and_baseline_safe():
    baseline = V4Candidate("canonical", 0xA5A5)
    alternate = V4Candidate("alternate", 0x5A5A, 0.95)
    codebook = v4_candidate_codebook(baseline, device="cpu")
    targets = codebook[::2048][:16] + 0.001
    scores = [
        score_v4_candidate(targets, baseline, codebook=codebook),
        score_v4_candidate(targets, alternate),
    ]
    assert all(torch.isfinite(torch.tensor([s.mse, s.p95, s.represented_orthants])).all() for s in scores)
    selected = select_v4_candidate(scores, baseline=baseline)
    assert selected in {baseline, alternate}


def test_candidate_selection_rejects_nonfinite_and_missing_baseline():
    baseline = V4Candidate("canonical", 0xA5A5)
    with pytest.raises(ValueError, match="score list"):
        select_v4_candidate([], baseline=baseline)
    with pytest.raises(ValueError, match="exactly one baseline"):
        select_v4_candidate(
            [
                type("Score", (), {"candidate": V4Candidate("alternate", 0x5A5A), "mse": 1.0, "p95": 1.0, "represented_orthants": 1.0})()
            ],
            baseline=baseline,
        )
    with pytest.raises(ValueError, match="finite"):
        select_v4_candidate(
            [
                type(
                    "Score",
                    (),
                    {"candidate": baseline, "mse": float("nan"), "p95": 1.0, "represented_orthants": 1.0},
                )()
            ],
            baseline=baseline,
        )


def test_experimental_canonical_codebook_matches_production_v4_quantization():
    generator = torch.Generator().manual_seed(71)
    weight = torch.randn(16, 16, generator=generator)
    hessian = torch.eye(16)
    expected = quantize_qvq_linear(weight, hessian, bits=2, vector_size=4, trellis_batch_size=1)
    actual = quantize_qvq_linear(
        weight,
        hessian,
        bits=2,
        vector_size=4,
        trellis_batch_size=1,
        experimental_codebook=v4_candidate_codebook(V4Candidate("canonical", 0xA5A5)),
    )
    assert torch.equal(actual.trellis, expected.trellis)
    assert torch.equal(actual.inner_weight, expected.inner_weight)
    assert torch.equal(actual.weight, expected.weight)
    with pytest.raises(RuntimeError, match="evaluation-only"):
        actual.serialized_tensors()


def test_production_qvq_result_exposes_serialization_tensors():
    generator = torch.Generator().manual_seed(73)
    weight = torch.randn(16, 16, generator=generator)
    hessian = torch.eye(16)
    result = quantize_qvq_linear(weight, hessian, bits=2, vector_size=4, trellis_batch_size=1)
    payload = result.serialized_tensors()
    assert set(payload) == {"trellis", "SU", "SV"}
    assert all(tensor is getattr(result, name) for name, tensor in payload.items())


def test_qvq_quantization_telemetry_is_opt_in_and_preserves_math():
    generator = torch.Generator().manual_seed(72)
    weight = torch.randn(16, 16, generator=generator)
    hessian = torch.eye(16)
    control = quantize_qvq_linear(weight, hessian, bits=2, vector_size=4, trellis_batch_size=1)
    measured = quantize_qvq_linear(
        weight,
        hessian,
        bits=2,
        vector_size=4,
        trellis_batch_size=1,
        telemetry=QVQQuantizationTelemetry(),
    )
    assert control.telemetry is None
    assert torch.equal(measured.trellis, control.trellis)
    assert torch.equal(measured.weight, control.weight)
    assert measured.telemetry is not None
    assert measured.telemetry["counters"]["weight_elements"] == 256
    assert measured.telemetry["counters"]["tail_biting_chunks"] == 1
    assert measured.telemetry["counters"]["viterbi_recurrence_passes"] == 2
    assert "native_viterbi_launches" not in measured.telemetry["counters"]
    assert {
        "rht_weight",
        "rht_hessian",
        "codebook",
        "baseline_encode",
        "block_ldl_factor",
        "block_ldl_feedback",
        "block_ldl_viterbi",
        "block_ldl_reconstruct",
        "candidate_reconstruct_proxy",
        "pack_trellis",
    } <= measured.telemetry["phases"].keys()
    assert all(phase["host_dispatch_ms"] >= 0 for phase in measured.telemetry["phases"].values())
    assert all(phase["gpu_ms"] is None for phase in measured.telemetry["phases"].values())


def test_qvq_telemetry_rejects_recording_or_refinalizing_after_finalize():
    telemetry = QVQQuantizationTelemetry()
    assert telemetry.finalize() == {"phases": {}, "counters": {}}
    with pytest.raises(RuntimeError, match="only once"):
        telemetry.finalize()
    with pytest.raises(RuntimeError, match="after finalization"):
        telemetry.count("late")


@pytest.mark.parametrize(
    "codebook,exception,message",
    [
        (torch.empty(65536, 2), ValueError, "shape"),
        (torch.zeros(65536, 4, dtype=torch.int32), TypeError, "floating-point"),
        (torch.full((65536, 4), float("nan")), ValueError, "finite"),
    ],
)
def test_experimental_codebook_fails_closed(codebook, exception, message):
    with pytest.raises(exception, match=message):
        quantize_qvq_linear(
            torch.eye(16),
            torch.eye(16),
            bits=2,
            vector_size=4,
            trellis_batch_size=1,
            experimental_codebook=codebook,
        )
